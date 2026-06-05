#!/usr/bin/env python3
"""APEX matrix runner for CI orchestration.

This script orchestrates parallel APEX runs across:
- Workflow versions (V1, V2)
- Zones (deployment topology boundaries)
- Configurations (device, backend, scene types)

Design:
- Runnable in CI (GitHub Actions) or locally
- Collects all PerformanceCapsules into run-specific directory
- Supports dry-run mode for testing
- Fails fast on errors unless --continue-on-error

Usage:
    # Run V1 and V2 across all zones
    python scripts/apex_matrix_runner.py \\
        --run-id "$(git rev-parse HEAD)" \\
        --commit-sha "$(git rev-parse HEAD)" \\
        --zones local \\
        --output-dir ./apex_results

    # Dual-run (V1 + V2) with multiple zones
    python scripts/apex_matrix_runner.py \\
        --run-id abc123 \\
        --commit-sha abc123 \\
        --zones us-west-2a us-west-2b us-east-1a \\
        --workflow-versions v1 v2 \\
        --output-dir ./apex_results

Exit codes:
    0: Success
    1: Configuration error
    2: Execution error (at least one run failed)
    3: Aggregation error

Version: 1.0.0
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from transformation_portal.metrics.aggregator import (
    compute_global_stats,
    compute_per_zone_stats,
    compute_worst_zone_p95,
    log_aggregated_stats_to_ledger,
)
from transformation_portal.metrics.contracts import Observation, RunSpec
from transformation_portal.metrics.performance_capsule import PerformanceCapsule

__version__ = "1.0.0"

APEX_DA3_MODEL_KEY = "da3-metric"


class ApexConfigError(ValueError):
    """Configuration error (invalid flags, unknown backend_id, etc.).

    This exception indicates user configuration problems, not execution failures.
    Should result in exit code 1 (configuration error).
    """

    pass


def _get_pipeline_version() -> str:
    """Get pipeline version from package metadata or fallback to git SHA.

    Returns:
        Version string (e.g., "0.1.0" or "git-abc123")
    """
    try:
        from importlib.metadata import version

        return version("transformation-portal")
    except Exception:
        # Fallback to git SHA if package not installed
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            return f"git-{result.stdout.strip()}"
        except Exception:
            return "unknown"


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def auto_detect_device() -> str:
    """Auto-detect best available device for inference.

    Returns:
        "mps" if Apple Silicon available, "cuda" if NVIDIA GPU available, else "cpu"

    Note:
        Catches all exceptions (not just ImportError) to handle broken torch installs
        (e.g., CUDA driver mismatches, corrupted shared libs). Degrades to CPU.
    """
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception as e:
        # Torch not available or broken, default to CPU
        logger.debug(f"Torch import/check failed ({e}), falling back to CPU")
        return "cpu"


def check_ml_dependencies(backend_id: str) -> tuple[bool, list[str]]:
    """Check availability of ML dependencies for real pipeline execution.

    Validates dependencies for the selected backend. torch is ALWAYS required
    for real execution; additional deps (transformers, etc.) are backend-specific.

    Args:
        backend_id: Backend identifier (e.g., "da3", "depth_pro", "mock").

    Returns:
        (all_available, missing_packages)

    Raises:
        ApexConfigError: When backend_id is unknown (configuration error).
        RuntimeError: When backend fails to declare dependencies (backend bug).

    Note:
        Catches all exceptions (not just ImportError) to handle broken installs
        (e.g., missing CUDA libraries, corrupted shared libraries). Treats broken
        dependencies as missing to provide clear error messages.
    """
    from transformation_portal.depth.backends import get_registry

    # Get backend from registry using public API
    registry = get_registry()

    backend_cls = registry.get_backend_class(backend_id)
    if backend_cls is None:
        # Unknown backend: fail fast with clear guidance
        available = registry.available_backend_ids()
        available_str = ", ".join(available) if available else "(none)"
        raise ApexConfigError(
            f"Unknown backend_id '{backend_id}'.\n"
            f"Available backends: {available_str}\n\n"
            f"Fix: choose a valid backend_id or register the backend.\n"
            f"See: docs/apex/phase3/README.md for backend registration."
        )

    # torch always required + backend-specific packages
    backend_packages: list[str] = []
    if hasattr(backend_cls, "required_packages") and callable(backend_cls.required_packages):
        try:
            backend_packages = list(backend_cls.required_packages())
        except Exception as e:
            logger.error(
                "Failed to get requirements for backend '%s': %s. " "This is a backend implementation error.",
                backend_id,
                e,
            )
            raise RuntimeError(
                f"Backend '{backend_id}' failed to declare dependencies: {e}\n"
                f"This is a backend implementation bug; please report it."
            ) from e

    if hasattr(backend_cls, "runtime_required_packages"):
        try:
            backend_instance = backend_cls()
            backend_packages = list(backend_instance.runtime_required_packages())
        except FileNotFoundError as e:
            raise ApexConfigError(
                f"Backend '{backend_id}' has an invalid isolated runtime configuration: {e}\n"
                "Fix the configured subprocess Python path before running the APEX matrix."
            ) from e
        except Exception as e:
            logger.error(
                "Failed to resolve runtime-specific requirements for backend '%s': %s. "
                "This is a backend implementation error.",
                backend_id,
                e,
            )
            raise RuntimeError(
                f"Backend '{backend_id}' failed to resolve runtime-specific dependencies: {e}\n"
                f"This is a backend implementation bug; please report it."
            ) from e

    required = ["torch"] + backend_packages
    # Dedupe while preserving order
    required = list(dict.fromkeys(required))

    logger.debug(f"Backend '{backend_id}' requires: {required}")

    # Check all required packages
    missing = []
    for pkg in required:
        try:
            # Use importlib to check if package can be imported
            # This properly detects both missing and broken installs
            spec = importlib.util.find_spec(pkg)
            if spec is None:
                logger.debug(f"{pkg} not found (spec is None)")
                missing.append(pkg)
            else:
                # Try actual import to catch broken installs (missing .so files, etc.)
                # Use importlib.import_module instead of __import__ for better testability
                importlib.import_module(pkg)
        except Exception as e:
            logger.debug(f"{pkg} import failed ({type(e).__name__}: {e}), treating as missing")
            missing.append(pkg)

    all_available = len(missing) == 0
    return all_available, missing


def _model_key_for_backend(backend_id: str) -> Optional[str]:
    """Return governed model selectors that the matrix runner must pin."""
    if backend_id == "da3":
        return APEX_DA3_MODEL_KEY
    return None


def run_apex_for_config(
    run_spec: RunSpec,
    zone: str,
    output_dir: Path,
    dry_run: bool = False,
    synthetic: bool = False,
    input_dir: Optional[Path] = None,
    sample_size: Optional[int] = None,
) -> Observation:
    """Execute APEX run for a specific configuration.

    Args:
        run_spec: Run specification
        zone: Zone identifier
        output_dir: Output directory for capsules
        dry_run: If True, skip actual execution
        synthetic: If True, mark observations as synthetic
        input_dir: Directory containing test images
        sample_size: Number of images to process (None = all)

    Returns:
        Observation with captured capsules
    """
    logger.info(f"Running APEX: workflow={run_spec.workflow_version}, zone={zone}")

    if dry_run:
        logger.info("DRY RUN: Skipping actual execution")
        # Create mock capsule for testing
        mock_capsule = PerformanceCapsule(
            image_id=f"test_image_{zone}",
            image_path="/path/to/test.jpg",
            input_hash="abcdef123456",
            original_shape=(4000, 6000),
            enforced_shape=(4000, 6000),
            pixel_count=24_000_000,
            dimension_adjustment="exact",
            backend_id=run_spec.backend_id,
            device=run_spec.device,
            timings={"total": 10.0, "inference": 8.0},
            workflow_version=run_spec.workflow_version,
            zone=zone,
            scene_type=run_spec.scene_type,
            is_synthetic=True,  # Mark as synthetic
        )

        return Observation(
            run_spec=run_spec,
            zone=zone,
            capsules=[mock_capsule],
        )

    # Real pipeline execution
    if not input_dir or not input_dir.exists():
        raise ValueError(f"Input directory required for real execution: {input_dir}")

    # Check ML dependencies early (fail fast with clear message)
    ml_available, missing = check_ml_dependencies(run_spec.backend_id)
    if not ml_available:
        error_msg = (
            f"Backend '{run_spec.backend_id}' requires ML dependencies: {', '.join(missing)}\n\n"
            "Install with:\n"
            '  pip install -e ".[ml]"\n\n'
            "Or use --dry-run for synthetic testing without ML deps."
        )
        raise RuntimeError(error_msg)

    import hashlib
    import signal

    from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
    from transformation_portal.lux_depth_v3.input_discovery import DiscoveryConfig, discover_images
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
    from transformation_portal.lux_depth_v3.raw_loader import RAW_EXTENSIONS
    from transformation_portal.metrics.timing import timing_context

    # Timeout handler for long-running operations
    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("Image processing timeout")

    # Discover input images (standard + RAW formats)
    discovery_config = DiscoveryConfig(strict_mode=False)
    # Include standard image formats + RAW camera formats
    standard_exts = [".jpg", ".jpeg", ".png"]
    raw_exts = sorted(RAW_EXTENSIONS)
    all_extensions = standard_exts + raw_exts
    images = discover_images(input_dir, discovery_config, all_extensions)

    if not images:
        raise ValueError(f"No images found in {input_dir}")

    # Apply sample size limit
    if sample_size is not None:
        images = sorted(images)[:sample_size]

    logger.info(f"Processing {len(images)} images with {run_spec.workflow_version} workflow")

    # Create workflow-specific output directory
    workflow_output = output_dir / f"{run_spec.workflow_version}_{zone}"
    workflow_output.mkdir(parents=True, exist_ok=True)

    # Configure pipeline based on workflow version
    # V1 = depth only, V2 = depth + enhancement
    enable_v2 = run_spec.workflow_version == "v2"

    config = EnhanceConfig(
        model_variant=ModelVariant.METRIC_LARGE,
        model_key=_model_key_for_backend(run_spec.backend_id),
        depth_device=run_spec.device,
        v2_device=run_spec.device,
        depth_backend=run_spec.backend_id,
        generate_pbr=False,  # Disable PBR for performance testing
        save_float_depth=False,
        strict_inputs=False,
        # V2 enhancement only for v2 workflow
        v2_preset="default" if enable_v2 else None,
    )

    # Initialize orchestrator
    orchestrator = EnhanceOrchestrator(
        config=config,
        output_root=workflow_output,
        verify_outputs=False,  # Speed up for benchmarking
    )

    capsules = []
    timeout_seconds = 300  # 5 minutes per image max

    # Cache pipeline version once (avoid repeated git/metadata lookups per image)
    pipeline_version = _get_pipeline_version()

    # Process each image with timing instrumentation
    for image_path in images:
        try:
            from transformation_portal.lux_depth_v3.input_manager import ImageInput
            from transformation_portal.metrics.performance_capsule import compute_dimension_adjustment

            image_input = ImageInput(image_path)

            # Set timeout alarm (Unix only)
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

            try:
                # Wrap processing in timing context
                timings = {}

                with timing_context("total", timings, device=run_spec.device):
                    # Load and preprocess image
                    with timing_context("load_decode", timings, device=run_spec.device):
                        from PIL import Image

                        with Image.open(image_path) as img:
                            original_shape = (img.height, img.width)

                    # Run pipeline (V1 = depth only, V2 = depth + enhancement)
                    result = orchestrator.enhance_image(image_input, input_root=input_dir)

                # Cancel timeout
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

                # Extract metadata from result
                if result and result.get("status") == "ok":
                    # Compute input hash for reproducibility (chunked to handle large files)
                    h = hashlib.sha256()
                    with image_path.open("rb") as f:
                        while chunk := f.read(8192):
                            h.update(chunk)
                    input_hash = h.hexdigest()[:16]

                    # Get enforced shape from result or use original
                    enforced_shape = result.get("enforced_shape", original_shape)
                    pixel_count = enforced_shape[0] * enforced_shape[1]

                    # Create performance capsule
                    capsule = PerformanceCapsule(
                        image_id=image_path.stem,
                        image_path=str(image_path),
                        input_hash=input_hash,
                        original_shape=original_shape,
                        enforced_shape=enforced_shape,
                        pixel_count=pixel_count,
                        dimension_adjustment=compute_dimension_adjustment(original_shape, enforced_shape),
                        backend_id=run_spec.backend_id,
                        model_variant=config.model_variant.value.name,
                        device=run_spec.device,
                        dtype="float32",
                        timings=timings,
                        workflow_version=run_spec.workflow_version,
                        zone=zone,
                        scene_type=run_spec.scene_type,
                        pipeline_version=pipeline_version,  # Cached from metadata/git
                        is_synthetic=synthetic,  # Respects --synthetic flag
                    )

                    capsules.append(capsule)
                    logger.info(f"✅ {image_path.name}: {timings['total']:.2f}s")
                else:
                    logger.warning(f"⚠️ {image_path.name}: processing returned non-ok status")

            except TimeoutError:
                logger.error(f"❌ {image_path.name}: timeout after {timeout_seconds}s")
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                continue

        except Exception as e:
            logger.error(f"❌ {image_path.name}: {e}")
            # Continue processing other images
            continue

    if not capsules:
        raise RuntimeError(f"No images successfully processed for {run_spec.workflow_version}/{zone}")

    return Observation(
        run_spec=run_spec,
        zone=zone,
        capsules=capsules,
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="APEX matrix runner for CI orchestration")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Run identification
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--commit-sha", required=True, help="Git commit SHA")

    # Configuration matrix
    parser.add_argument(
        "--workflow-versions",
        nargs="+",
        default=["v1", "v2"],
        choices=["v1", "v2"],
        help="Workflow versions to run (default: v1 v2)",
    )
    parser.add_argument("--zones", nargs="+", default=["local"], help="Zones to run across (default: local)")
    parser.add_argument(
        "--device",
        help="Device for inference (auto-detects mps/cuda/cpu if not specified)",
    )
    parser.add_argument("--backend-id", default="da3", help="Backend (default: da3)")
    parser.add_argument("--scene-type", help="Optional scene type filter")

    # Output
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument(
        "--ledger-db",
        type=Path,
        default=Path("./apex_performance.db"),
        help="Ledger database path (default: ./apex_performance.db)",
    )

    # Input
    parser.add_argument("--input-dir", type=Path, help="Input directory with test images (required for real execution)")
    parser.add_argument("--sample-size", type=int, help="Number of images to process per workflow (default: all)")

    # Execution control
    parser.add_argument("--dry-run", action="store_true", help="Dry run (skip actual execution, use mock data)")
    parser.add_argument(
        "--synthetic", action="store_true", help="Mark observations as synthetic (auto-enabled with --dry-run)"
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running even if some configurations fail")

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = auto_detect_device()
        logger.info(f"Auto-detected device: {args.device}")

    # Auto-enable synthetic flag when dry-run is used
    if args.dry_run and not args.synthetic:
        args.synthetic = True
        logger.info("Auto-enabled --synthetic (paired with --dry-run)")

    # Validate input requirements for real execution
    if not args.dry_run:
        if not args.input_dir:
            logger.error("❌ --input-dir required for real execution (or use --dry-run)")
            return 1
        if not args.input_dir.exists():
            logger.error(f"❌ Input directory does not exist: {args.input_dir}")
            return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- CRITICAL FIX: Initialize ledger before any aggregation ---
    from transformation_portal.metrics.ledger import PerformanceLedger

    try:
        ledger = PerformanceLedger(args.ledger_db)
        if not ledger.ensure_ready():
            logger.error("Ledger initialization failed")
            return 1
        logger.info(f"✅ Ledger initialized: {args.ledger_db}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ledger: {e}")
        return 1
    # --- END CRITICAL FIX ---

    # Collect all observations
    all_observations: List[Observation] = []
    errors = []

    timestamp = datetime.now(timezone.utc).isoformat()

    # Run matrix
    for workflow_version in args.workflow_versions:
        for zone in args.zones:
            try:
                # Create RunSpec
                run_spec = RunSpec(
                    run_id=args.run_id,
                    commit_sha=args.commit_sha,
                    workflow_version=workflow_version,
                    zones=args.zones,
                    device=args.device,
                    backend_id=args.backend_id,
                    scene_type=args.scene_type,
                    timestamp=timestamp,
                )

                # Execute run
                observation = run_apex_for_config(
                    run_spec=run_spec,
                    zone=zone,
                    output_dir=args.output_dir,
                    dry_run=args.dry_run,
                    synthetic=args.synthetic,
                    input_dir=args.input_dir,
                    sample_size=args.sample_size,
                )

                all_observations.append(observation)

                # Write observation to disk
                obs_file = args.output_dir / f"observation_{workflow_version}_{zone}.json"
                obs_file.write_text(json.dumps(observation.to_dict(), indent=2))
                logger.info(f"Wrote observation to {obs_file}")

            except ApexConfigError as e:
                # Configuration errors should fail fast with clear message
                logger.error(f"❌ Configuration error: {e}")
                return 1
            except Exception as e:
                error_msg = f"Failed to run {workflow_version} in zone {zone}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

                if not args.continue_on_error:
                    logger.error("Aborting due to error (use --continue-on-error to continue)")
                    return 2

    # Aggregate results
    logger.info("Aggregating results...")

    try:
        # Group observations by workflow version
        v1_capsules = []
        v2_capsules = []

        for obs in all_observations:
            if obs.run_spec.workflow_version == "v1":
                v1_capsules.extend(obs.capsules)
            else:
                v2_capsules.extend(obs.capsules)

        # Initialize worst-zone tracking
        v1_worst_p95 = None
        v2_worst_p95 = None

        # Compute per-zone and global stats for V1
        if v1_capsules:
            v1_per_zone = compute_per_zone_stats(v1_capsules)
            v1_global = compute_global_stats(v1_capsules)
            v1_worst_zone, v1_worst_p95 = compute_worst_zone_p95(v1_per_zone)

            logger.info(f"V1 worst-zone p95: {v1_worst_p95:.2f}s (zone: {v1_worst_zone})")

            # Write per-zone stats to ledger
            log_aggregated_stats_to_ledger(
                run_id=args.run_id,
                commit_sha=args.commit_sha,
                workflow_version="v1",
                timestamp=timestamp,
                per_zone_stats=v1_per_zone,
                ledger_db_path=str(args.ledger_db),
            )

            # Write global stats to ledger (zone=None)
            log_aggregated_stats_to_ledger(
                run_id=args.run_id,
                commit_sha=args.commit_sha,
                workflow_version="v1",
                timestamp=timestamp,
                per_zone_stats={None: v1_global},
                ledger_db_path=str(args.ledger_db),
            )

        # Compute per-zone and global stats for V2
        if v2_capsules:
            v2_per_zone = compute_per_zone_stats(v2_capsules)
            v2_global = compute_global_stats(v2_capsules)
            v2_worst_zone, v2_worst_p95 = compute_worst_zone_p95(v2_per_zone)

            logger.info(f"V2 worst-zone p95: {v2_worst_p95:.2f}s (zone: {v2_worst_zone})")

            # Write per-zone stats to ledger
            log_aggregated_stats_to_ledger(
                run_id=args.run_id,
                commit_sha=args.commit_sha,
                workflow_version="v2",
                timestamp=timestamp,
                per_zone_stats=v2_per_zone,
                ledger_db_path=str(args.ledger_db),
            )

            # Write global stats to ledger (zone=None)
            log_aggregated_stats_to_ledger(
                run_id=args.run_id,
                commit_sha=args.commit_sha,
                workflow_version="v2",
                timestamp=timestamp,
                per_zone_stats={None: v2_global},
                ledger_db_path=str(args.ledger_db),
            )

        # Write summary
        summary = {
            "run_id": args.run_id,
            "commit_sha": args.commit_sha,
            "timestamp": timestamp,
            "workflow_versions": args.workflow_versions,
            "zones": args.zones,
            "v1_worst_zone_p95": v1_worst_p95,
            "v2_worst_zone_p95": v2_worst_p95,
            "errors": errors,
        }

        summary_file = args.output_dir / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        logger.info(f"Wrote summary to {summary_file}")

    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        return 3

    if errors:
        logger.warning(f"Completed with {len(errors)} errors")
        return 2

    logger.info("APEX matrix run completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
