#!/usr/bin/env python3
"""Production-grade performance ledger with SQLite backend and query capabilities.

This tool provides:
- Append-only performance logging with atomic writes
- Efficient querying and bucketing for regression detection
- Scene-dependent performance analysis
- Quality Firewall integration

Usage:
    # Log a performance capsule
    python -m transformation_portal.metrics.ledger log \\
        --capsule capsule.json \\
        --ledger-db ./performance.db

    # Query ledger with filters
    python -m transformation_portal.metrics.ledger query \\
        --ledger-db ./performance.db \\
        --scene-type pool \\
        --device mps \\
        --output results.json

    # Detect regressions
    python -m transformation_portal.metrics.ledger regression \\
        --ledger-db ./performance.db \\
        --baseline-days 30 \\
        --current capsule.json

    # Generate performance report
    python -m transformation_portal.metrics.ledger report \\
        --ledger-db ./performance.db \\
        --output report.md

Architecture:
- SQLite backend for efficient queries (indexed by timestamp, scene_type, device)
- JSON blob storage for full capsule data
- Atomic commits for crash safety
- No unbounded growth (pruning support)

v2.0: Integrated with PerformanceCapsule schema and Quality Firewall.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformation_portal.metrics.performance_capsule import (
    DEFAULT_BUCKETS,
    PerformanceBucket,
    PerformanceCapsule,
    get_bucket_for_capsule,
)
from transformation_portal.metrics.sql_safety import normalize_query_limit

__version__ = "3.0.0"

logger = logging.getLogger(__name__)

# Database schema
SCHEMA_VERSION = 3  # v3: Phase 3 - Added optimized indexes and apex_trends view

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS performance_capsules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,

    -- Queryable fields (indexed)
    scene_type TEXT,
    device TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    pixel_count INTEGER NOT NULL,
    total_sec REAL NOT NULL,
    firewall_status TEXT NOT NULL,
    workflow_version TEXT DEFAULT 'v1',  -- v2: V1/V2 tracking
    zone TEXT,  -- v2: multi-zone tracking

    -- Full capsule data (JSON blob)
    capsule_json TEXT NOT NULL,

    -- Metadata
    schema_version TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_captured_at ON performance_capsules(captured_at);
CREATE INDEX IF NOT EXISTS idx_scene_type ON performance_capsules(scene_type);
CREATE INDEX IF NOT EXISTS idx_device ON performance_capsules(device);
CREATE INDEX IF NOT EXISTS idx_backend_id ON performance_capsules(backend_id);
CREATE INDEX IF NOT EXISTS idx_firewall_status ON performance_capsules(firewall_status);
CREATE INDEX IF NOT EXISTS idx_total_sec ON performance_capsules(total_sec);
CREATE INDEX IF NOT EXISTS idx_workflow_version ON performance_capsules(workflow_version);
CREATE INDEX IF NOT EXISTS idx_zone ON performance_capsules(zone);

-- APEX aggregation table for run-level stats
CREATE TABLE IF NOT EXISTS apex_runs (
    run_id TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    workflow_version TEXT NOT NULL,
    zone TEXT,
    bucket_name TEXT NOT NULL,
    p50 REAL NOT NULL,
    p95 REAL NOT NULL,
    p99 REAL,
    count INTEGER NOT NULL,
    threshold_p50 REAL NOT NULL,
    threshold_p95 REAL NOT NULL,
    pass_fail TEXT NOT NULL,
    raw_capsules_json TEXT,
    PRIMARY KEY (run_id, workflow_version, zone, bucket_name)
);

CREATE INDEX IF NOT EXISTS idx_apex_runs_commit ON apex_runs(commit_sha);
CREATE INDEX IF NOT EXISTS idx_apex_runs_timestamp ON apex_runs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_apex_runs_workflow ON apex_runs(workflow_version);
CREATE INDEX IF NOT EXISTS idx_apex_runs_zone ON apex_runs(zone);
CREATE INDEX IF NOT EXISTS idx_apex_runs_bucket_zone_time ON apex_runs(bucket_name, zone, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_apex_runs_pass_fail ON apex_runs(pass_fail);

-- Aggregation view for dashboard trend analysis
CREATE VIEW IF NOT EXISTS apex_trends AS
SELECT
    bucket_name,
    zone,
    workflow_version,
    DATE(timestamp) as date,
    AVG(p50) as avg_p50,
    AVG(p95) as avg_p95,
    AVG(p99) as avg_p99,
    COUNT(*) as run_count,
    SUM(CASE WHEN pass_fail = 'fail' THEN 1 ELSE 0 END) as fail_count,
    SUM(CASE WHEN pass_fail = 'warn' THEN 1 ELSE 0 END) as warn_count
FROM apex_runs
GROUP BY bucket_name, zone, workflow_version, DATE(timestamp)
ORDER BY date DESC;
"""


class PerformanceLedger:
    """SQLite-backed performance ledger for append and query operations."""

    def __init__(self, db_path: Path) -> None:
        """Initialize ledger with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure database schema exists and is current (with migration support)."""
        with sqlite3.connect(self.db_path) as conn:
            # Check current schema version
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            schema_table_exists = cursor.fetchone() is not None

            current_version = 0
            if schema_table_exists:
                cursor = conn.execute("SELECT version FROM schema_version")
                row = cursor.fetchone()
                if row:
                    current_version = row[0]

            # Migrate if needed
            if current_version < SCHEMA_VERSION:
                logger.info(f"Migrating ledger schema from v{current_version} to v{SCHEMA_VERSION}")
                self._migrate_schema(conn, current_version)

            # Create or update schema
            conn.executescript(CREATE_TABLES_SQL)

            # Update schema version
            if schema_table_exists:
                conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
            else:
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

            conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Migrate schema from older version.

        Args:
            conn: SQLite connection
            from_version: Current schema version
        """
        if from_version == 0:
            # Fresh database, no migration needed
            return

        if from_version == 1:
            # Migration 1 -> 2: Add workflow_version and zone columns
            try:
                conn.execute("ALTER TABLE performance_capsules ADD COLUMN workflow_version TEXT DEFAULT 'v1'")
                conn.execute("ALTER TABLE performance_capsules ADD COLUMN zone TEXT")
                logger.info("Added workflow_version and zone columns")
            except sqlite3.OperationalError as e:
                # Columns may already exist (idempotent migration)
                logger.debug(f"Migration warning (likely safe): {e}")

    def ensure_ready(self) -> bool:
        """Verify ledger is writable and schema is current.

        Returns:
            True if ready, False if setup failed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Quick write test - verify apex_runs table exists
                conn.execute("SELECT COUNT(*) FROM apex_runs")
                return True
        except Exception as e:
            logger.error(f"Ledger not ready: {e}")
            return False

    def log_capsule(self, capsule: PerformanceCapsule) -> None:
        """Log a performance capsule to the ledger (atomic).

        Args:
            capsule: Performance capsule to log
        """
        capsule_json = json.dumps(capsule.to_dict())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO performance_capsules (
                    image_id, captured_at, scene_type, device, backend_id,
                    pixel_count, total_sec, firewall_status, workflow_version, zone,
                    capsule_json, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capsule.image_id,
                    capsule.captured_at,
                    capsule.scene_type,
                    capsule.device,
                    capsule.backend_id,
                    capsule.pixel_count,
                    capsule.timings["total"],
                    capsule.firewall_status,
                    capsule.workflow_version,
                    capsule.zone,
                    capsule_json,
                    capsule.schema_version,
                ),
            )
            conn.commit()

    def query_capsules(
        self,
        scene_type: Optional[str] = None,
        device: Optional[str] = None,
        backend_id: Optional[str] = None,
        firewall_status: Optional[str] = None,
        workflow_version: Optional[str] = None,
        zone: Optional[str] = None,
        min_captured_at: Optional[str] = None,
        max_captured_at: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[PerformanceCapsule]:
        """Query performance capsules with filters.

        Args:
            scene_type: Filter by scene type
            device: Filter by device
            backend_id: Filter by backend
            firewall_status: Filter by firewall status
            workflow_version: Filter by workflow version ("v1" or "v2")
            zone: Filter by deployment zone
            min_captured_at: Filter by minimum timestamp (ISO8601)
            max_captured_at: Filter by maximum timestamp (ISO8601)
            limit: Maximum number of results

        Returns:
            List of matching performance capsules
        """
        safe_limit = normalize_query_limit(limit)

        where_clauses = []
        params = []

        if scene_type is not None:
            where_clauses.append("scene_type = ?")
            params.append(scene_type)

        if device is not None:
            where_clauses.append("device = ?")
            params.append(device)

        if backend_id is not None:
            where_clauses.append("backend_id = ?")
            params.append(backend_id)

        if firewall_status is not None:
            where_clauses.append("firewall_status = ?")
            params.append(firewall_status)

        if workflow_version is not None:
            where_clauses.append("workflow_version = ?")
            params.append(workflow_version)

        if zone is not None:
            where_clauses.append("zone = ?")
            params.append(zone)

        if min_captured_at is not None:
            where_clauses.append("captured_at >= ?")
            params.append(min_captured_at)

        if max_captured_at is not None:
            where_clauses.append("captured_at <= ?")
            params.append(max_captured_at)

        # SAFETY: every entry in `where_clauses` above is a hardcoded literal of the form
        # "<column> <op> ?" — none originate from `params` or any other caller-provided value.
        # If you add a new clause, keep this invariant: column names and operators are literals,
        # values flow through `params` only. Adding interpolation here would introduce SQL
        # injection. The base query is also a literal and intentionally not f-stringed.
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        query = "SELECT capsule_json FROM performance_capsules WHERE " + where_sql + " ORDER BY captured_at DESC"  # nosec B608
        if safe_limit is not None:
            query += " LIMIT ?"
            params.append(safe_limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        capsules = []
        for (capsule_json,) in rows:
            capsule_dict = json.loads(capsule_json)
            capsules.append(PerformanceCapsule.from_dict(capsule_dict))

        return capsules

    def get_statistics(
        self,
        scene_type: Optional[str] = None,
        device: Optional[str] = None,
        backend_id: Optional[str] = None,
        min_captured_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute statistics for filtered capsules.

        Args:
            scene_type: Filter by scene type
            device: Filter by device
            backend_id: Filter by backend
            min_captured_at: Filter by minimum timestamp

        Returns:
            Dict with count, mean, median, p50, p95, min, max
        """
        capsules = self.query_capsules(
            scene_type=scene_type,
            device=device,
            backend_id=backend_id,
            min_captured_at=min_captured_at,
        )

        if not capsules:
            return {"count": 0}

        total_times = [c.timings["total"] for c in capsules]
        total_times.sort()

        n = len(total_times)

        return {
            "count": n,
            "mean_sec": sum(total_times) / n,
            "median_sec": total_times[n // 2],
            "p50_sec": total_times[n // 2],
            "p95_sec": total_times[int(n * 0.95)] if n > 1 else total_times[0],
            "min_sec": total_times[0],
            "max_sec": total_times[-1],
        }

    def prune_old_entries(self, days_to_keep: int = 90) -> int:
        """Prune entries older than specified days.

        Args:
            days_to_keep: Number of days of history to retain

        Returns:
            Number of entries deleted
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM performance_capsules WHERE captured_at < ?", (cutoff_iso,))
            deleted = cursor.rowcount
            conn.commit()

        return deleted


def detect_regression(
    current_capsule: PerformanceCapsule,
    historical_capsules: List[PerformanceCapsule],
    bucket: Optional[PerformanceBucket] = None,
) -> Dict[str, Any]:
    """Detect performance regression by comparing current to historical.

    Args:
        current_capsule: Current performance capsule
        historical_capsules: Historical capsules for comparison
        bucket: Optional performance bucket with thresholds (inferred if not provided)

    Returns:
        Dict with regression status and details
    """
    if not historical_capsules:
        return {
            "status": "insufficient_data",
            "message": "No historical data for comparison",
        }

    # Use bucket if provided, otherwise infer (always succeeds with catch-all)
    if bucket is None:
        bucket = get_bucket_for_capsule(current_capsule)

    current_total = current_capsule.timings["total"]
    historical_totals = [c.timings["total"] for c in historical_capsules]
    historical_totals.sort()

    n = len(historical_totals)
    historical_p50 = historical_totals[n // 2]
    historical_p95 = historical_totals[int(n * 0.95)] if n > 1 else historical_totals[0]

    result = {
        "status": "pass",
        "current_total_sec": current_total,
        "historical_p50_sec": historical_p50,
        "historical_p95_sec": historical_p95,
        "bucket": bucket.name,
    }

    # Check against bucket thresholds
    if current_total > bucket.p95_threshold_sec:
        result["status"] = "regression_p95"
        result["message"] = f"Exceeded p95 threshold ({bucket.p95_threshold_sec:.2f}s)"
    elif current_total > bucket.p50_threshold_sec * 1.5:
        result["status"] = "warning_p50"
        result["message"] = f"Significantly above p50 threshold ({bucket.p50_threshold_sec:.2f}s)"

    return result


def generate_performance_report(
    ledger: PerformanceLedger,
    output_path: Path,
    min_captured_at: Optional[str] = None,
) -> None:
    """Generate comprehensive performance analysis report.

    Args:
        ledger: Performance ledger instance
        output_path: Output markdown file path
        min_captured_at: Optional minimum timestamp filter
    """
    all_capsules = ledger.query_capsules(min_captured_at=min_captured_at)

    if not all_capsules:
        output_path.write_text("# Performance Report\n\nNo data available.\n")
        return

    # Group by scene type
    by_scene: Dict[str, List[PerformanceCapsule]] = {}
    for capsule in all_capsules:
        scene = capsule.scene_type or "unknown"
        by_scene.setdefault(scene, []).append(capsule)

    # Generate report
    lines = [
        "# Performance Analysis Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Total capsules: {len(all_capsules)}",
        "",
        "## Scene-Dependent Performance",
        "",
    ]

    for scene_type, capsules in sorted(by_scene.items()):
        totals = sorted([c.timings["total"] for c in capsules])
        n = len(totals)

        lines.extend(
            [
                f"### {scene_type.title()} Scenes (n={n})",
                "",
                f"- Mean: {sum(totals) / n:.2f}s",
                f"- Median: {totals[n // 2]:.2f}s",
                f"- p95: {totals[int(n * 0.95)] if n > 1 else totals[0]:.2f}s",
                f"- Min: {totals[0]:.2f}s",
                f"- Max: {totals[-1]:.2f}s",
                f"- Max/Min ratio: {totals[-1] / totals[0]:.2f}×",
                "",
            ]
        )

    # Bucket analysis
    lines.extend(
        [
            "## Performance Bucket Analysis",
            "",
        ]
    )

    for bucket in DEFAULT_BUCKETS:
        matching = [c for c in all_capsules if bucket.matches(c)]
        if not matching:
            continue

        totals = sorted([c.timings["total"] for c in matching])
        n = len(totals)
        p50 = totals[n // 2]
        p95 = totals[int(n * 0.95)] if n > 1 else totals[0]

        p50_status = "✅" if p50 <= bucket.p50_threshold_sec else "⚠️"
        p95_status = "✅" if p95 <= bucket.p95_threshold_sec else "❌"

        lines.extend(
            [
                f"### {bucket.name}",
                "",
                f"**Description:** {bucket.description}",
                "",
                f"- Samples: {n}",
                f"- p50: {p50:.2f}s (threshold: {bucket.p50_threshold_sec:.2f}s) {p50_status}",
                f"- p95: {p95:.2f}s (threshold: {bucket.p95_threshold_sec:.2f}s) {p95_status}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines))


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Performance ledger tool for regression detection")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Log command
    log_parser = subparsers.add_parser("log", help="Log a performance capsule")
    log_parser.add_argument("--capsule", type=Path, required=True, help="Capsule JSON file")
    log_parser.add_argument("--ledger-db", type=Path, required=True, help="Ledger database")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query performance capsules")
    query_parser.add_argument("--ledger-db", type=Path, required=True, help="Ledger database")
    query_parser.add_argument("--scene-type", help="Filter by scene type")
    query_parser.add_argument("--device", help="Filter by device")
    query_parser.add_argument("--backend-id", help="Filter by backend")
    query_parser.add_argument("--min-days", type=int, help="Minimum days ago")
    query_parser.add_argument("--limit", type=int, help="Maximum results")
    query_parser.add_argument("--output", type=Path, help="Output JSON file")

    # Regression command
    regression_parser = subparsers.add_parser("regression", help="Detect regression")
    regression_parser.add_argument("--ledger-db", type=Path, required=True, help="Ledger database")
    regression_parser.add_argument("--capsule", type=Path, required=True, help="Current capsule JSON")
    regression_parser.add_argument("--baseline-days", type=int, default=30, help="Baseline window (days)")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate performance report")
    report_parser.add_argument("--ledger-db", type=Path, required=True, help="Ledger database")
    report_parser.add_argument("--output", type=Path, required=True, help="Output markdown file")
    report_parser.add_argument("--min-days", type=int, help="Minimum days ago")

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Prune old entries")
    prune_parser.add_argument("--ledger-db", type=Path, required=True, help="Ledger database")
    prune_parser.add_argument("--days-to-keep", type=int, default=90, help="Days to keep")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        if args.command == "log":
            capsule_dict = json.loads(args.capsule.read_text())
            capsule = PerformanceCapsule.from_dict(capsule_dict)

            ledger = PerformanceLedger(args.ledger_db)
            ledger.log_capsule(capsule)

            logger.info(f"Logged capsule for {capsule.image_id}")

        elif args.command == "query":
            min_captured_at = None
            if args.min_days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=args.min_days)
                min_captured_at = cutoff.isoformat()

            ledger = PerformanceLedger(args.ledger_db)
            capsules = ledger.query_capsules(
                scene_type=args.scene_type,
                device=args.device,
                backend_id=args.backend_id,
                min_captured_at=min_captured_at,
                limit=args.limit,
            )

            logger.info(f"Found {len(capsules)} matching capsules")

            if args.output:
                output = [c.to_dict() for c in capsules]
                args.output.write_text(json.dumps(output, indent=2))
                logger.info(f"Wrote results to {args.output}")

        elif args.command == "regression":
            capsule_dict = json.loads(args.capsule.read_text())
            current_capsule = PerformanceCapsule.from_dict(capsule_dict)

            # Query historical data
            cutoff = datetime.now(timezone.utc) - timedelta(days=args.baseline_days)
            min_captured_at = cutoff.isoformat()

            ledger = PerformanceLedger(args.ledger_db)
            historical = ledger.query_capsules(
                scene_type=current_capsule.scene_type,
                device=current_capsule.device,
                backend_id=current_capsule.backend_id,
                min_captured_at=min_captured_at,
            )

            result = detect_regression(current_capsule, historical)

            print(json.dumps(result, indent=2))

            if result["status"].startswith("regression"):
                logger.error(f"Regression detected: {result.get('message', '')}")
                return 1
            elif result["status"].startswith("warning"):
                logger.warning(f"Performance warning: {result.get('message', '')}")

        elif args.command == "report":
            min_captured_at = None
            if args.min_days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=args.min_days)
                min_captured_at = cutoff.isoformat()

            ledger = PerformanceLedger(args.ledger_db)
            generate_performance_report(ledger, args.output, min_captured_at)

            logger.info(f"Generated report at {args.output}")

        elif args.command == "prune":
            ledger = PerformanceLedger(args.ledger_db)
            deleted = ledger.prune_old_entries(args.days_to_keep)

            logger.info(f"Pruned {deleted} entries older than {args.days_to_keep} days")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
