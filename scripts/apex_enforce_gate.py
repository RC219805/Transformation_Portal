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

    if not db_path.exists():
        logger.error(f"Ledger database not found: {db_path}")
        return 2

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT bucket_name, zone, p95, threshold_p95, pass_fail
                FROM apex_runs
                WHERE run_id = ?
                  AND workflow_version = 'v2'
                  AND commit_sha = ?
                  AND pass_fail = 'fail'
            """,
                (run_id, commit_sha),
            )

            failures = cursor.fetchall()

            if not failures:
                logger.info("✅ APEX gate passed (no V2 FAIL buckets)")
                return 0

            # Report failures
            logger.warning(f"⚠️  Found {len(failures)} failing V2 bucket(s):")
            for bucket, zone, p95, threshold, status in failures:
                zone_str = zone or "Global"
                logger.warning(f"  • {bucket} [{zone_str}]: p95={p95:.2f}s (limit={threshold:.2f}s)")

            if mode == "shadow":
                logger.info("🔍 Shadow mode: reporting only, not blocking (gate would have failed)")
                return 0
            elif mode == "enforce":
                logger.error(f"❌ APEX gate failed: {len(failures)} bucket(s) exceeded thresholds")
                return 1
            else:
                logger.error(f"Unknown mode: {mode}")
                return 2

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
    sys.exit(enforce_gate(args.ledger_db, args.run_id, args.commit_sha, args.mode))


if __name__ == "__main__":
    main()
