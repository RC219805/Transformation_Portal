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
from typing import Any, Dict

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


def rebuild_ledger(input_dir: Path, db_path: Path, clean: bool = False) -> int:
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

    # Clean rebuild: truncate both capsules AND derived aggregates
    if clean:
        import sqlite3

        logger.warning("🧹 CLEAN MODE: Truncating performance_capsules and apex_runs tables")

        tables_to_truncate = ["performance_capsules", "apex_runs"]
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Check which tables exist.
            # SAFETY: `placeholders` is `?,?,...` only; every value flows through
            # `tables_to_truncate`, which is bound as the parameter sequence below.
            placeholders = ",".join("?" for _ in tables_to_truncate)
            existing_tables = {
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN (" + placeholders + ")",
                    tables_to_truncate,
                ).fetchall()
            }
            # Truncate each existing table.
            # SAFETY: `table_name` only iterates over the hardcoded `tables_to_truncate` literal
            # above; sqlite does not parameterize identifiers, so interpolation is unavoidable.
            # Do NOT change `tables_to_truncate` to accept caller-supplied names without adding
            # a strict whitelist check here.
            for table_name in tables_to_truncate:
                if table_name in existing_tables:
                    logger.info(f"  Truncating {table_name}...")
                    cursor.execute(f"DELETE FROM {table_name}")  # noqa: S608  # see SAFETY note above
            conn.commit()
        logger.info("✅ Clean complete")

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
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Truncate capsules AND aggregates before rebuild (prevents duplicates)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    sys.exit(rebuild_ledger(args.input_dir, args.ledger_db, clean=args.clean))


if __name__ == "__main__":
    main()
