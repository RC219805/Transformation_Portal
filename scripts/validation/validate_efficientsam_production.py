#!/usr/bin/env python3
"""
Production Validation: EfficientSAM Segmentation Backend

Tests the EfficientSAM backend integration on 800 Picacho luxury real estate images.

Expected outcomes:
- Material masks generated for each image
- Pixel operations applied based on detected materials
- Enhanced images with surface-aware finishing
- Manifest contains segmentation telemetry
- Performance metrics within expected bounds
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zipfile import BadZipFile

import numpy as np

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure logging for direct command-line execution."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _validate_efficientsam_evidence(
    result: Dict[str, Any],
    *,
    output_root: Path,
    plan: Any,
    verified_evidence_cache: Dict[Path, Dict[str, Any]] | None = None,
) -> Tuple[Path, Dict[str, Any], Path]:
    """Validate the per-image artifact evidence required by this entrypoint."""

    resolved_output_root = output_root.resolve(strict=True)
    from transformation_portal.lux_depth_v3.execution_evidence import (
        read_confined_artifact_snapshot,
        require_required_artifacts,
        verify_execution_evidence_file,
    )

    def confined_file(path_value: str, *, label: str, root_relative: bool = False) -> Path:
        try:
            candidate = Path(path_value)
            if root_relative:
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise ValueError("root-relative path is not canonical")
                candidate = resolved_output_root / candidate
            resolved_path = candidate.resolve(strict=True)
            resolved_path.relative_to(resolved_output_root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{label} is missing or outside the current output root: {path_value}") from exc
        if not resolved_path.is_file():
            raise ValueError(f"{label} is not a regular file: {resolved_path}")
        return resolved_path

    manifest_value = result.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("successful result did not identify a combined manifest")
    manifest_path = confined_file(manifest_value, label="combined manifest")
    try:
        manifest_snapshot = read_confined_artifact_snapshot(
            resolved_output_root,
            manifest_path,
            context="EfficientSAM combined manifest",
            max_bytes=64 * 1024 * 1024,
        )
    except RuntimeError as exc:
        raise ValueError(f"combined manifest cannot be read safely: {manifest_path}") from exc

    try:
        manifest_data = json.loads(manifest_snapshot.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"combined manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest_data, dict):
        raise ValueError(f"combined manifest root is not an object: {manifest_path}")

    materials_v3 = manifest_data.get("materials_v3")
    if not isinstance(materials_v3, dict) or materials_v3.get("enabled") is not True:
        raise ValueError("combined manifest does not confirm Materials V3 execution")
    segmentation = materials_v3.get("segmentation_metadata")
    if not isinstance(segmentation, dict):
        raise ValueError("combined manifest lacks segmentation metadata")
    if segmentation.get("backend") != "efficientsam":
        raise ValueError(
            "combined manifest does not confirm the EfficientSAM backend " f"(recorded={segmentation.get('backend')!r})"
        )
    mask_count = segmentation.get("mask_count")
    if isinstance(mask_count, bool) or not isinstance(mask_count, int) or mask_count <= 0:
        raise ValueError(f"combined manifest records an invalid mask count: {mask_count!r}")

    manifest_mask_value = segmentation.get("mask_artifact_path")
    result_mask_value = result.get("segmentation_mask_path")
    if not isinstance(manifest_mask_value, str) or not manifest_mask_value:
        raise ValueError("combined manifest does not identify a segmentation mask artifact")
    if not isinstance(result_mask_value, str) or not result_mask_value:
        raise ValueError("successful result does not identify a segmentation mask artifact")

    environment = manifest_data.get("environment")
    execution_contract = environment.get("execution_contract") if isinstance(environment, dict) else None
    evidence_value = execution_contract.get("execution_evidence_path") if isinstance(execution_contract, dict) else None
    if not isinstance(evidence_value, str) or not evidence_value:
        raise ValueError("combined manifest does not identify canonical execution evidence")
    evidence_path = confined_file(
        evidence_value,
        label="execution evidence",
        root_relative=True,
    )
    evidence_cache = verified_evidence_cache if verified_evidence_cache is not None else {}
    verified_evidence = evidence_cache.get(evidence_path)
    if verified_evidence is None:
        try:
            verified_evidence = verify_execution_evidence_file(
                evidence_path,
                output_root=resolved_output_root,
                plan=plan,
            )
            require_required_artifacts(verified_evidence)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"canonical execution evidence verification failed: {evidence_path}") from exc
        evidence_cache[evidence_path] = verified_evidence

    def verified_record(relative_path: str) -> Dict[str, Any]:
        produced = verified_evidence.get("produced_artifacts")
        if not isinstance(produced, list):
            raise ValueError("canonical execution evidence has no produced-artifact records")
        matches = [
            record
            for outcome in produced
            if isinstance(outcome, dict) and isinstance(outcome.get("artifacts"), list)
            for record in outcome["artifacts"]
            if isinstance(record, dict) and record.get("path") == relative_path
        ]
        if len(matches) != 1:
            raise ValueError(f"canonical execution evidence does not uniquely bind {relative_path}")
        return matches[0]

    if not manifest_snapshot.matches(verified_record(manifest_snapshot.relative_path)):
        raise ValueError("combined manifest bytes do not match canonical execution evidence")

    manifest_mask_path = confined_file(manifest_mask_value, label="segmentation mask artifact")
    result_mask_path = confined_file(result_mask_value, label="result segmentation mask artifact")
    if manifest_mask_path != result_mask_path:
        raise ValueError("result segmentation mask does not match combined-manifest evidence")
    try:
        mask_snapshot = read_confined_artifact_snapshot(
            resolved_output_root,
            manifest_mask_path,
            context="EfficientSAM mask artifact",
            max_bytes=256 * 1024 * 1024,
        )
    except RuntimeError as exc:
        raise ValueError(f"segmentation mask artifact cannot be read safely: {manifest_mask_path}") from exc
    if not mask_snapshot.matches(verified_record(mask_snapshot.relative_path)):
        raise ValueError("segmentation mask bytes do not match canonical execution evidence")

    if segmentation.get("mask_artifact_format") != "npz":
        raise ValueError("combined manifest does not identify the mask artifact as NPZ")
    try:
        with np.load(io.BytesIO(mask_snapshot.data), allow_pickle=False) as archive:
            mask_names = tuple(sorted(archive.files))
            if len(set(mask_names)) != len(mask_names):
                raise ValueError("segmentation mask archive contains duplicate member names")
            masks = tuple(np.asarray(archive[name]) for name in mask_names)
    except (BadZipFile, OSError, ValueError, KeyError) as exc:
        raise ValueError(f"segmentation mask artifact is not a readable safe NPZ: {manifest_mask_path}") from exc
    if len(masks) != mask_count:
        raise ValueError(
            "segmentation mask artifact count does not match combined-manifest evidence "
            f"(artifact={len(masks)}, recorded={mask_count})"
        )
    if not masks:
        raise ValueError("segmentation mask artifact contains no masks")
    expected_shape: Tuple[int, int] | None = None
    for name, mask in zip(mask_names, masks):
        if mask.ndim != 2 or mask.size == 0 or not np.issubdtype(mask.dtype, np.floating):
            raise ValueError(f"segmentation mask {name!r} is not a non-empty 2D floating array")
        if not np.isfinite(mask).all() or np.any(mask < 0.0) or np.any(mask > 1.0):
            raise ValueError(f"segmentation mask {name!r} contains invalid probability values")
        mask_shape = (int(mask.shape[0]), int(mask.shape[1]))
        if expected_shape is None:
            expected_shape = mask_shape
        elif mask_shape != expected_shape:
            raise ValueError("segmentation mask artifact contains inconsistent shapes")
    recorded_shape = segmentation.get("mask_artifact_shape")
    if list(expected_shape or ()) != recorded_shape:
        raise ValueError(
            "segmentation mask dimensions do not match combined-manifest evidence "
            f"(artifact={list(expected_shape or ())}, recorded={recorded_shape!r})"
        )

    return manifest_path, manifest_data, manifest_mask_path


def run_validation() -> int:
    """Execute production validation with EfficientSAM backend."""

    print("=" * 80)
    print("EfficientSAM Production Validation")
    print("=" * 80)
    print()

    # Configuration
    input_dir = Path("input_images/800 Picacho")
    output_dir = Path(f"output_800_picacho_efficientsam_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Count input images
    input_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG"))
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Images found: {len(input_images)}")
    print()
    if not input_images:
        logger.error("No input images found; validation cannot produce evidence")
        return 1

    # Import pipeline components
    logger.info("Importing pipeline components...")
    try:
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
    except ImportError as e:
        logger.error(f"Failed to import pipeline components: {e}")
        return 1

    # Create configuration with EfficientSAM enabled
    logger.info("Creating configuration...")
    config = EnhanceConfig(
        # Quality tier
        quality_tier="apex",
        # Depth backend (commercial-safe)
        depth_backend="da3",
        depth_device="auto",  # Auto-detect MPS/CUDA/CPU
        # Materials V3 with EfficientSAM
        enable_materials_v3=True,
        apply_pixel_ops=True,
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,  # Fail if EfficientSAM unavailable
        # Materials V3 parameters
        min_coverage_px=500,
        min_mean_conf=0.2,
        glass_response_enabled=True,
        # V2 enhancement
        enable_v2=True,
        v2_preset="default",
        # PBR
        generate_pbr=True,
        # Caching
        enable_depth_cache=True,
        # Performance
        enable_parallel_processing=True,
        max_parallel_workers=1,  # Serial for controlled validation
        # Emit flags
        output_bit_depth=16,
        emit_run_card=True,
    )

    print()
    print("Configuration:")
    print(f"  Quality tier: {config.quality_tier}")
    print(f"  Depth backend: {config.depth_backend}")
    print(f"  Depth device: {config.depth_device}")
    print(f"  Materials V3: {config.enable_materials_v3}")
    print(f"  Segmentation backend: {config.material_segmentation_backend}")
    print(f"  Strict backend: {config.strict_backend}")
    print(f"  V2 enhancement: {config.enable_v2}")
    print(f"  PBR generation: {config.generate_pbr}")
    print()

    # Create orchestrator
    logger.info("Initializing orchestrator...")
    try:
        prepared = prepare_lux_execution(
            config,
            input_dir,
            [image_path.absolute() for image_path in sorted(input_images)],
        )
        input_images = list(prepared.input_files)
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_dir)
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}")
        return 1

    # Process images
    logger.info("Starting pipeline...")
    print()
    print("-" * 80)
    print()

    start_time = time.time()
    results: List[Dict] = []
    errors: List[Dict] = []
    verified_evidence_cache: Dict[Path, Dict[str, Any]] = {}

    try:
        batch_results = orchestrator.enhance_batch(
            prepared.input_root,
            input_files=list(prepared.input_files),
        )
        if len(batch_results) != len(input_images):
            raise RuntimeError("Pipeline returned an incomplete batch result set")
    except Exception as exc:
        logger.error(f"Authoritative batch execution failed: {exc}")
        batch_results = []
        errors.extend(
            {
                "image": image_path.name,
                "error": str(exc),
                "elapsed_sec": 0.0,
            }
            for image_path in input_images
        )

    for idx, (image_path, result) in enumerate(zip(input_images, batch_results), 1):
        logger.info(f"[{idx}/{len(input_images)}] Processed: {image_path.name}")
        if not isinstance(result, dict):
            image_elapsed = 0.0
            error = "pipeline returned a non-object result"
        else:
            image_elapsed = float(result.get("runtime_s", 0.0) or 0.0)
            error = str(result.get("error") or result.get("reason") or "pipeline returned non-ok status")

        if isinstance(result, dict) and result.get("status") == "ok":
            try:
                manifest_path, manifest_data, mask_path = _validate_efficientsam_evidence(
                    result,
                    output_root=output_dir,
                    plan=prepared.plan,
                    verified_evidence_cache=verified_evidence_cache,
                )
            except (OSError, ValueError) as exc:
                error = f"invalid EfficientSAM evidence: {exc}"
            else:
                results.append(
                    {
                        "image": image_path.name,
                        "status": "success",
                        "elapsed_sec": round(image_elapsed, 2),
                        "result": result,
                        "manifest_path": manifest_path,
                        "manifest_data": manifest_data,
                        "mask_path": mask_path,
                    }
                )
                logger.info(f"  ✓ Complete in {image_elapsed:.2f}s with validated EfficientSAM evidence")
                print()
                continue

        if error:
            errors.append(
                {
                    "image": image_path.name,
                    "error": error,
                    "elapsed_sec": round(image_elapsed, 2),
                }
            )
            logger.error(f"  ✗ Failed after {image_elapsed:.2f}s: {error}")
        print()

    total_elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 80)
    print("Validation Complete")
    print("=" * 80)
    print()
    print(f"Total images: {len(input_images)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(errors)}")
    print(f"Total time: {total_elapsed:.2f}s")

    if results:
        avg_time = sum(r["elapsed_sec"] for r in results) / len(results)
        print(f"Average time per image: {avg_time:.2f}s")

    print()

    # Analyze outputs
    if output_dir.exists():
        logger.info("Analyzing outputs...")

        # Count files
        all_files = list(output_dir.rglob("*"))
        total_files = len([f for f in all_files if f.is_file()])
        total_size_mb = sum(f.stat().st_size for f in all_files if f.is_file()) / (1024 * 1024)

        print(f"Output directory: {output_dir}")
        print(f"  Total files: {total_files}")
        print(f"  Total size: {total_size_mb:.1f} MB")
        print()

        manifests = [entry["manifest_path"] for entry in results]
        masks = [entry["mask_path"] for entry in results]
        print(f"Validated combined manifests: {len(manifests)}")
        print(f"Validated segmentation mask artifacts: {len(masks)}")

        if results:
            first_manifest = results[0]["manifest_path"]
            manifest_data = results[0]["manifest_data"]
            logger.info(f"Analyzing manifest: {first_manifest.name}")
            mat_v3 = manifest_data["materials_v3"]
            print()
            print("Sample manifest (materials_v3 section):")
            print(json.dumps(mat_v3, indent=2)[:500])
            print()
            backend = mat_v3["segmentation_metadata"]["backend"]
            print(f"✓ Segmentation backend recorded: {backend}")
            logger.info("✓ EfficientSAM backend confirmed in manifest")

        print()

    # Report errors
    if errors:
        print("Errors:")
        for err in errors:
            print(f"  {err['image']}: {err['error']}")
        print()

    # Exit code
    if len(results) == len(input_images):
        logger.info("✓ All images processed successfully")
        return 0
    elif results:
        logger.warning(f"⚠ Partial success: {len(results)}/{len(input_images)} images")
        return 2
    else:
        logger.error("✗ All images failed")
        return 1


if __name__ == "__main__":
    _configure_logging()
    sys.exit(run_validation())
