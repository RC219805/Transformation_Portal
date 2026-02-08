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
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from transformation_portal.metrics.aggregator import (
    compute_global_stats,
    compute_per_zone_stats,
    compute_worst_zone_p95,
    log_aggregated_stats_to_ledger,
)
from transformation_portal.metrics.contracts import Observation, RunSpec
from transformation_portal.metrics.performance_capsule import PerformanceCapsule

__version__ = "1.0.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_apex_for_config(
    run_spec: RunSpec,
    zone: str,
    output_dir: Path,
    dry_run: bool = False,
) -> Observation:
    """Execute APEX run for a specific configuration.

    Args:
        run_spec: Run specification
        zone: Zone identifier
        output_dir: Output directory for capsules
        dry_run: If True, skip actual execution

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
        )

        return Observation(
            run_spec=run_spec,
            zone=zone,
            capsules=[mock_capsule],
        )

    # TODO: Integrate with actual pipeline runner
    # For now, this is a placeholder that would call:
    # - Lux Depth V3 pipeline (V1)
    # - New V2 pipeline (V2)
    # Both should emit PerformanceCapsules

    raise NotImplementedError("Actual pipeline integration not yet implemented. Use --dry-run for testing.")


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
    parser.add_argument("--device", default="mps", help="Device (default: mps)")
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

    # Execution control
    parser.add_argument("--dry-run", action="store_true", help="Dry run (skip actual execution, use mock data)")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running even if some configurations fail")

    args = parser.parse_args()

    # BLOCKER FIX #1: Enforce dry-run until real pipeline is wired
    if not args.dry_run:
        logger.error("❌ Real pipeline integration not yet implemented")
        logger.error("   Use --dry-run to test APEX scaffolding")
        logger.error("   Track progress: docs/APEX_REAL_PIPELINE_INTEGRATION.md")
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
                )

                all_observations.append(observation)

                # Write observation to disk
                obs_file = args.output_dir / f"observation_{workflow_version}_{zone}.json"
                obs_file.write_text(json.dumps(observation.to_dict(), indent=2))
                logger.info(f"Wrote observation to {obs_file}")

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
