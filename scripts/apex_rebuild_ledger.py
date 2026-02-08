#!/usr/bin/env python3
"""
APEX Ledger Rebuilder - CI Correctness Hardened Edition.

Ingests raw Observation JSON files from CI artifacts into SQLite ledger.
Fails loud on errors to prevent silent data loss.

Exit codes:
    0: Success
    2: No observation files found
    3: Zero capsules ingested
    4: Some files failed to ingest
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from transformation_portal.metrics.ledger import PerformanceLedger
from transformation_portal.metrics.performance_capsule import PerformanceCapsule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("apex_rebuild")


def backfill_capsule_metadata(
    capsule_dict: Dict[str, Any],
    obs_zone: str | None,
    obs_workflow_version: str | None,
) -> Dict[str, Any]:
    """Backfill zone and workflow_version from parent Observation if missing."""
    if "zone" not in capsule_dict and obs_zone is not None:
        capsule_dict["zone"] = obs_zone
        logger.debug(f"Backfilled zone={obs_zone} for capsule {capsule_dict.get('image_id')}")

    if "workflow_version" not in capsule_dict and obs_workflow_version is not None:
        capsule_dict["workflow_version"] = obs_workflow_version
        logger.debug(f"Backfilled workflow_version={obs_workflow_version}")

    return capsule_dict


def rebuild_ledger(input_dir: Path, db_path: Path) -> int:
    """Rebuild ledger from observation JSON files.

    Returns:
        Exit code (0=success, 2=no files, 3=no capsules, 4=failures)
    """
    logger.info(f"Rebuilding ledger: {db_path}")
    logger.info(f"Input directory: {input_dir}")

    # Initialize ledger (creates tables if needed)
    ledger = PerformanceLedger(db_path)
    if not ledger.ensure_ready():
        logger.error("Ledger initialization failed")
        return 1

    # Find all observation files
    files = sorted(input_dir.glob("**/observation_*.json"))
    if not files:
        logger.error(f"❌ No observation_*.json files found under {input_dir}")
        return 2

    logger.info(f"Found {len(files)} observation files")

    ingested = 0
    failed_files = 0

    for filepath in files:
        try:
            # Read with explicit UTF-8 encoding
            data = json.loads(filepath.read_text(encoding="utf-8"))

            # Extract parent observation metadata
            obs_zone = data.get("zone")
            run_spec = data.get("run_spec") or {}
            obs_workflow_version = run_spec.get("workflow_version")

            capsules_data = data.get("capsules", [])
            if not isinstance(capsules_data, list):
                raise ValueError(f"capsules is not a list in {filepath}")

            # Ingest each capsule
            for cap_dict in capsules_data:
                if not isinstance(cap_dict, dict):
                    logger.warning(f"Skipping non-dict capsule in {filepath}")
                    continue

                # Backfill metadata from parent observation
                cap_dict = backfill_capsule_metadata(cap_dict, obs_zone, obs_workflow_version)

                # Convert to PerformanceCapsule and log
                capsule = PerformanceCapsule.from_dict(cap_dict)
                ledger.log_capsule(capsule)
                ingested += 1

            logger.info(f"✓ Ingested {len(capsules_data)} capsules from {filepath.name}")

        except Exception as e:
            failed_files += 1
            logger.exception(f"❌ Failed to ingest {filepath}: {e}")

    # Summary
    logger.info("=" * 70)
    logger.info(f"Observation files: {len(files)} total | {failed_files} failed")
    logger.info(f"Capsules ingested: {ingested}")
    logger.info("=" * 70)

    # Fail loud on errors
    if ingested == 0:
        logger.error("❌ CRITICAL: Zero capsules ingested")
        return 3

    if failed_files > 0:
        logger.error(f"❌ {failed_files} files failed to ingest")
        return 4

    logger.info("✅ Ledger rebuild successful")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Rebuild APEX ledger from observation JSON files")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing observation_*.json files")
    parser.add_argument(
        "--ledger-db",
        type=Path,
        default=Path("apex_performance.db"),
        help="Path to ledger database (default: apex_performance.db)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    sys.exit(rebuild_ledger(args.input_dir, args.ledger_db))


if __name__ == "__main__":
    main()
