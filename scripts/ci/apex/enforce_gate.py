#!/usr/bin/env python3
"""
APEX Gate Enforcement Script.

Queries ledger for V2 failures and enforces (or reports) based on mode.

Modes:
  - enforce: Exit non-zero if any V2 buckets fail (blocks CI)
  - shadow:  Report failures but always exit zero (informational)
  - disabled: Skip entirely
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

from transformation_portal.metrics.measurement import decode_aggregate_row

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("apex_gate")


def enforce_gate(db_path: Path, run_id: str, commit_sha: str, mode: str = "enforce") -> int:
    """
    Check for V2 failures and enforce based on mode.

    Returns:
        0 if pass (or shadow mode), 1 if fail and enforcing
    """
    if mode == "disabled":
        logger.info("Gate enforcement disabled, skipping")
        return 0
    if mode not in ("enforce", "shadow"):
        logger.error(f"Unknown mode: {mode}")
        return 2

    if not db_path.exists():
        logger.error(f"Ledger database not found: {db_path}")
        return 2

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT bucket_name, zone, workflow_version, p50, p95, p99,
                       count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json
                FROM apex_runs
                WHERE run_id = ?
                  AND workflow_version = 'v2'
                  AND commit_sha = ?
            """,
                (run_id, commit_sha),
            )

            rows = [decode_aggregate_row(dict(row)) for row in cursor.fetchall()]
            eligible = [row for row in rows if not row["is_insufficient_data"]]
            insufficient_count = len(rows) - len(eligible)
            if insufficient_count:
                logger.info(
                    "Insufficient measurement evidence in %d V2 bucket(s); excluded from enforcement", insufficient_count
                )
            if not eligible:
                logger.info("APEX gate has insufficient measurement evidence; no V2 bucket assessed (non-blocking)")
                return 0
            failures = [row for row in eligible if row["pass_fail"] == "fail"]

            if not failures:
                if insufficient_count:
                    logger.info("No assessed V2 bucket failed; overall evidence remains insufficient (non-blocking)")
                else:
                    logger.info("✅ APEX gate passed (no assessed V2 FAIL buckets)")
                return 0

            # Report failures
            logger.warning(f"⚠️  Found {len(failures)} failing V2 bucket(s):")
            for row in failures:
                zone_str = row["zone"] or "Global"
                logger.warning(
                    "  • %s [%s]: p95=%.2fs (limit=%.2fs); %d independent measurements (%s)",
                    row["bucket_name"],
                    zone_str,
                    row["p95"],
                    row["threshold_p95"],
                    row["count"],
                    row["measurement_unit"],
                )

            if mode == "shadow":
                logger.info("🔍 Shadow mode: reporting only, not blocking (gate would have failed)")
                return 0
            else:
                logger.error(f"❌ APEX gate failed: {len(failures)} bucket(s) exceeded thresholds")
                return 1

    except Exception as e:
        logger.exception(f"Gate enforcement error: {e}")
        return 2


def main():
    parser = argparse.ArgumentParser(description="APEX gate enforcement")
    parser.add_argument("--ledger-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--mode",
        choices=["enforce", "shadow", "disabled"],
        default="enforce",
        help="Gate mode: enforce (block), shadow (report only), or disabled",
    )

    args = parser.parse_args()
    # Import-time warnings may already have initialized root logging. Set only
    # the CLI's logger so insufficient-evidence explanations remain observable.
    logger.setLevel(logging.INFO)
    sys.exit(enforce_gate(args.ledger_db, args.run_id, args.commit_sha, args.mode))


if __name__ == "__main__":
    main()
