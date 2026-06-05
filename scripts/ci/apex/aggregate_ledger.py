#!/usr/bin/env python3
"""
APEX Ledger Aggregator.

Computes per-zone statistics and logs them to apex_runs table.
Replaces inline Python in GitHub Actions workflow.
"""
import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformation_portal.metrics.aggregator import compute_per_zone_stats, log_aggregated_stats_to_ledger
from transformation_portal.metrics.performance_capsule import PerformanceCapsule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apex_aggregate")


def aggregate_ledger(
    db_path: Path,
    run_id: str,
    commit_sha: str,
) -> int:
    """Compute and log aggregated statistics.

    In schema v3, run_id/commit_sha are NOT columns on performance_capsules;
    they live only in apex_runs. This aggregator assumes the DB contains
    capsules from a single run (CI creates fresh DB per workflow execution).

    Returns:
        Exit code (0=success, non-zero=failure)
    """
    logger.info(f"Aggregating ledger: {db_path}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Commit SHA: {commit_sha}")

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Check if this run is already aggregated (idempotency)
            existing_count = conn.execute("SELECT COUNT(*) as count FROM apex_runs WHERE run_id = ?", (run_id,)).fetchone()[
                "count"
            ]

            if existing_count > 0:
                logger.warning(
                    f"Run {run_id} already has {existing_count} aggregated rows. " "Will re-aggregate (ON CONFLICT REPLACE)."
                )

            # Load all capsules
            # ASSUMPTION: This DB contains capsules from only this run
            # (CI workflow creates fresh DB per matrix execution)
            query = "SELECT capsule_json FROM performance_capsules"
            logger.info("Loading all capsules (assuming single-run DB)")
            rows = conn.execute(query).fetchall()

            if not rows:
                logger.warning("No capsules found in ledger")
                return 1

            # Parse all capsules
            capsules = []
            for row in rows:
                cap_dict = json.loads(row["capsule_json"])
                capsule = PerformanceCapsule.from_dict(cap_dict)
                capsules.append(capsule)

            logger.info(f"Loaded {len(capsules)} capsules (assuming all from run {run_id})")

            # Aggregate per workflow version
            for workflow_version in ("v1", "v2"):
                wf_capsules = [c for c in capsules if getattr(c, "workflow_version", None) == workflow_version]

                if not wf_capsules:
                    logger.info(f"No capsules for {workflow_version}, skipping")
                    continue

                logger.info(f"Computing stats for {workflow_version} ({len(wf_capsules)} capsules)")

                per_zone_stats = compute_per_zone_stats(wf_capsules)
                timestamp = datetime.now(timezone.utc).isoformat()

                log_aggregated_stats_to_ledger(
                    run_id=run_id,
                    commit_sha=commit_sha,
                    workflow_version=workflow_version,
                    timestamp=timestamp,
                    per_zone_stats=per_zone_stats,
                    ledger_db_path=str(db_path),
                )

                logger.info(f"✓ Logged stats for {len(per_zone_stats)} zones ({workflow_version})")

        logger.info("✅ Aggregation complete")
        return 0

    except Exception as e:
        logger.exception(f"❌ Aggregation failed: {e}")
        return 2


def main():
    parser = argparse.ArgumentParser(description="Aggregate APEX ledger statistics")
    parser.add_argument("--ledger-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit-sha", required=True)

    args = parser.parse_args()

    if not args.ledger_db.exists():
        logger.error(f"Ledger database not found: {args.ledger_db}")
        sys.exit(1)

    sys.exit(aggregate_ledger(args.ledger_db, args.run_id, args.commit_sha))


if __name__ == "__main__":
    main()
