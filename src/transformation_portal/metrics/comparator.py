"""APEX workflow comparator for regression detection.

This module compares current performance to baseline and detects regressions.

Comparison strategies:
- Commit-to-commit: Compare current commit to previous commit
- Branch-to-main: Compare feature branch to main branch
- V1-to-V2: Compare V2 workflow to V1 baseline

Design principles:
- Support dual-run comparison (V1 vs V2 on same commit)
- Configurable regression thresholds (default: 15% for fail, 10% for warn)
- Bucket-level comparison (not just global)
- Explain why regression was flagged

Usage:
    from transformation_portal.metrics.comparator import (
        compare_to_baseline,
        detect_v1_v2_regression,
    )

    # Compare current to historical baseline
    report = compare_to_baseline(
        current_stats=current_stats,
        baseline_stats=baseline_stats,
        current_run_id="abc123",
        baseline_run_id="def456",
        current_commit_sha="abc123",
        baseline_commit_sha="def456",
    )

    # Compare V2 to V1 (dual-run)
    report = detect_v1_v2_regression(
        v2_stats=v2_stats,
        v1_stats=v1_stats,
        run_id="abc123",
        commit_sha="abc123",
    )

Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional

from transformation_portal.metrics.contracts import BucketStats, RegressionReport
from transformation_portal.metrics.sql_safety import normalize_query_limit

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# Regression thresholds
DEFAULT_WARN_THRESHOLD = 0.10  # 10% regression triggers warning
DEFAULT_FAIL_THRESHOLD = 0.15  # 15% regression triggers failure


def compute_regression_delta(
    current: float,
    baseline: float,
) -> float:
    """Compute regression delta as fractional increase.

    Args:
        current: Current metric value
        baseline: Baseline metric value

    Returns:
        Fractional increase (e.g., 0.15 = 15% regression)
    """
    if baseline <= 0:
        # Avoid division by zero - treat as no regression
        return 0.0

    return (current - baseline) / baseline


def compare_to_baseline(
    current_stats: Dict[str, BucketStats],
    baseline_stats: Dict[str, BucketStats],
    current_run_id: str,
    baseline_run_id: str,
    current_commit_sha: str,
    baseline_commit_sha: str,
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
) -> RegressionReport:
    """Compare current performance to baseline and detect regressions.

    Args:
        current_stats: Current bucket statistics
        baseline_stats: Baseline bucket statistics
        current_run_id: Current run identifier
        baseline_run_id: Baseline run identifier
        current_commit_sha: Current commit SHA
        baseline_commit_sha: Baseline commit SHA
        warn_threshold: Regression threshold for warning (default: 0.10 = 10%)
        fail_threshold: Regression threshold for failure (default: 0.15 = 15%)

    Returns:
        RegressionReport with verdict and details
    """
    bucket_regressions: Dict[str, float] = {}
    max_regression = 0.0
    max_regression_bucket = ""

    # Compare p95 for each bucket
    for bucket_name, current_bucket in current_stats.items():
        if bucket_name not in baseline_stats:
            # New bucket, skip comparison
            continue

        baseline_bucket = baseline_stats[bucket_name]

        # Compare p95 (primary metric for Quality Firewall)
        regression = compute_regression_delta(
            current_bucket.p95,
            baseline_bucket.p95,
        )

        bucket_regressions[bucket_name] = regression

        if regression > max_regression:
            max_regression = regression
            max_regression_bucket = bucket_name

    # Determine overall status
    status: Literal["pass", "warn", "fail"]
    if max_regression >= fail_threshold:
        status = "fail"
        explanation = (
            f"Regression detected: {max_regression_bucket} p95 increased by "
            f"{max_regression * 100:.1f}% (threshold: {fail_threshold * 100:.1f}%)"
        )
    elif max_regression >= warn_threshold:
        status = "warn"
        explanation = (
            f"Performance degradation: {max_regression_bucket} p95 increased by "
            f"{max_regression * 100:.1f}% (threshold: {warn_threshold * 100:.1f}%)"
        )
    else:
        status = "pass"
        explanation = f"No significant regression detected (max: {max_regression * 100:.1f}%)"

    return RegressionReport(
        baseline_run_id=baseline_run_id,
        baseline_commit_sha=baseline_commit_sha,
        current_run_id=current_run_id,
        current_commit_sha=current_commit_sha,
        bucket_regressions=bucket_regressions,
        max_regression=max_regression,
        max_regression_bucket=max_regression_bucket,
        status=status,
        explanation=explanation,
    )


def detect_v1_v2_regression(
    v2_stats: Dict[str, BucketStats],
    v1_stats: Dict[str, BucketStats],
    run_id: str,
    commit_sha: str,
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
) -> RegressionReport:
    """Compare V2 workflow to V1 baseline (dual-run on same commit).

    This is a special case of compare_to_baseline where baseline is V1
    and current is V2, both from the same commit.

    Args:
        v2_stats: V2 workflow bucket statistics
        v1_stats: V1 workflow bucket statistics (baseline)
        run_id: Run identifier (same for both V1 and V2)
        commit_sha: Commit SHA (same for both)
        warn_threshold: Regression threshold for warning
        fail_threshold: Regression threshold for failure

    Returns:
        RegressionReport comparing V2 to V1
    """
    return compare_to_baseline(
        current_stats=v2_stats,
        baseline_stats=v1_stats,
        current_run_id=f"{run_id}_v2",
        baseline_run_id=f"{run_id}_v1",
        current_commit_sha=commit_sha,
        baseline_commit_sha=commit_sha,
        warn_threshold=warn_threshold,
        fail_threshold=fail_threshold,
    )


def query_baseline_stats(
    ledger_db_path: str,
    workflow_version: str,
    zone: Optional[str] = None,
    commit_sha: Optional[str] = None,
    limit: int = 1,
) -> Dict[str, BucketStats]:
    """Query baseline statistics from ledger.

    Uses two-step query to avoid silent data loss from LIMIT:
    1. Find latest run_id matching filters
    2. Get ALL buckets for that run

    Args:
        ledger_db_path: Path to ledger database
        workflow_version: Workflow version ("v1" or "v2")
        zone: Optional zone filter (None = global)
        commit_sha: Optional commit SHA (None = latest)
        limit: Number of runs to consider (default: 1 = most recent)

    Returns:
        Dict mapping bucket_name -> BucketStats
    """
    import sqlite3

    safe_limit = normalize_query_limit(limit)
    if safe_limit is None:
        raise ValueError("limit must be an integer in range 1..1000")

    where_clauses = ["workflow_version = ?"]
    params = [workflow_version]

    if zone is not None:
        where_clauses.append("zone = ?")
        params.append(zone)

    if commit_sha is not None:
        where_clauses.append("commit_sha = ?")
        params.append(commit_sha)

    where_sql = " AND ".join(where_clauses)

    baseline_stats: Dict[str, BucketStats] = {}

    with sqlite3.connect(ledger_db_path) as conn:
        # Step 1: Find latest run_id
        # SAFETY: `where_sql` joins only the hardcoded "<column> = ?" literals built
        # above; all caller values bind through `run_params`, and LIMIT is validated
        # by normalize_query_limit() and bound via the `?` placeholder.
        run_query = f"""
            SELECT run_id, timestamp
            FROM apex_runs
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT ?
        """  # nosec B608
        run_params = [*params, safe_limit]
        cursor = conn.execute(run_query, run_params)
        latest_run = cursor.fetchone()

        if not latest_run:
            return baseline_stats

        latest_run_id = latest_run[0]

        # Step 2: Get ALL buckets for that run (no LIMIT)
        bucket_query = """
            SELECT bucket_name, p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail
            FROM apex_runs
            WHERE run_id = ? AND workflow_version = ?
        """
        bucket_params = [latest_run_id, workflow_version]

        if zone is not None:
            bucket_query += " AND zone = ?"
            bucket_params.append(zone)

        cursor = conn.execute(bucket_query, bucket_params)
        for row in cursor.fetchall():
            bucket_name, p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail = row
            baseline_stats[bucket_name] = BucketStats(
                bucket_name=bucket_name,
                count=count,
                p50=p50,
                p95=p95,
                p99=p99,
                mean=p50,  # Approximation (not stored in apex_runs)
                min=p50 * 0.8,  # Approximation
                max=p95 * 1.2,  # Approximation
                threshold_p50=threshold_p50,
                threshold_p95=threshold_p95,
                pass_fail=pass_fail,
            )

    return baseline_stats
