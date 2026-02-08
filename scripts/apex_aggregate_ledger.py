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

    Returns:
        Exit code (0=success, non-zero=failure)
    """
    logger.info(f"Aggregating ledger: {db_path}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Commit SHA: {commit_sha}")

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # BLOCKER FIX #3: Scope capsules by run_id and commit_sha
            # Check schema capabilities
            schema_cursor = conn.execute("PRAGMA table_info(performance_capsules)")
            columns = {row[1] for row in schema_cursor.fetchall()}

            where_clauses = []
            params = []

            # Build scoped query based on available columns
            if "run_id" in columns:
                where_clauses.append("run_id = ?")
                params.append(run_id)

            if "commit_sha" in columns:
                where_clauses.append("commit_sha = ?")
                params.append(commit_sha)

            if where_clauses:
                where_sql = " WHERE " + " AND ".join(where_clauses)
                query = f"SELECT capsule_json FROM performance_capsules{where_sql}"
                logger.info(f"Scoping capsules: {where_sql}")
                cursor = conn.execute(query, params)
            else:
                # BLOCKER FIX #3: Refuse unsafe aggregation per contract
                logger.error(
                    "❌ REFUSING TO AGGREGATE: Schema lacks run_id/commit_sha columns. "
                    "This would mix data from multiple runs and produce incorrect verdicts. "
                    "Update ledger schema to v3 or migrate data."
                )
                return 2  # Hard fail per quality firewall

            rows = cursor.fetchall()

            if not rows:
                logger.warning("No capsules found in ledger")
                return 1

            capsules = []
            for row in rows:
                cap_dict = json.loads(row["capsule_json"])
                capsule = PerformanceCapsule.from_dict(cap_dict)
                capsules.append(capsule)

            logger.info(f"Loaded {len(capsules)} capsules")

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
