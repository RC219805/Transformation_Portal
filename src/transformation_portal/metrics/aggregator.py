"""APEX workflow aggregator for per-zone and global performance statistics.

This module computes aggregated statistics from raw PerformanceCapsules:
- Per-zone stats (zone -> bucket -> p50/p95/p99)
- Global stats (across all zones)
- Worst-zone detection (max p95 across zones)

Design principles:
- Pure functions (no side effects, testable)
- Support both V1 and V2 workflows
- Bucket-aware aggregation (match capsules to buckets first)
- Handle sparse data gracefully (missing zones, empty buckets)

Usage:
    from transformation_portal.metrics.aggregator import (
        compute_per_zone_stats,
        compute_global_stats,
        compute_worst_zone_p95,
    )

    # Compute per-zone stats
    per_zone = compute_per_zone_stats(capsules, buckets)
    # {"zone-1": {"bucket-1": BucketStats(...), ...}, ...}

    # Compute global stats (all zones)
    global_stats = compute_global_stats(capsules, buckets)
    # {"bucket-1": BucketStats(...), ...}

    # Find worst zone
    worst_zone, worst_p95 = compute_worst_zone_p95(per_zone)

Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from transformation_portal.metrics.contracts import BucketStats
from transformation_portal.metrics.performance_capsule import DEFAULT_BUCKETS, PerformanceBucket, PerformanceCapsule

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


def compute_bucket_stats(
    capsules: List[PerformanceCapsule],
    bucket: PerformanceBucket,
    min_samples: int = 20,
) -> Optional[BucketStats]:
    """Compute statistics for capsules matching a specific bucket.

    Args:
        capsules: List of performance capsules
        bucket: Bucket definition with filters and thresholds
        min_samples: Minimum samples required for valid statistics (contract: 20)

    Returns:
        BucketStats if any capsules match, None otherwise
    """
    # Filter capsules matching this bucket
    matching = [c for c in capsules if bucket.matches(c)]

    if not matching:
        return None

    # Extract total times and sort
    total_times = sorted([c.timings["total"] for c in matching])
    n = len(total_times)

    # Compute percentiles (use proper median for p50)
    if n % 2 == 0:
        p50 = (total_times[n // 2 - 1] + total_times[n // 2]) / 2.0
    else:
        p50 = total_times[n // 2]
    p95 = total_times[int(n * 0.95)] if n > 1 else total_times[0]
    p99 = total_times[int(n * 0.99)] if n > 1 else total_times[0]
    mean = sum(total_times) / n

    # BLOCKER FIX #2: Enforce minimum sample size per APEX Contract v1.0.0
    # Compute stats but mark as insufficient_data (never blocks)
    if n < min_samples:
        return BucketStats(
            bucket_name=bucket.name,
            count=n,
            p50=p50,
            p95=p95,
            p99=p99,
            mean=mean,
            min=total_times[0],
            max=total_times[-1],
            threshold_p50=bucket.p50_threshold_sec,
            threshold_p95=bucket.p95_threshold_sec,
            pass_fail="pass",  # Nominal verdict; flag indicates insufficient data
            is_insufficient_data=True,  # Never blocks per contract
        )

    # Determine pass/fail status (only for n >= min_samples)
    if p95 > bucket.p95_threshold_sec:
        pass_fail: Literal["pass", "warn", "fail"] = "fail"
    elif p50 > bucket.p50_threshold_sec * 1.2:
        pass_fail = "warn"
    else:
        pass_fail = "pass"

    return BucketStats(
        bucket_name=bucket.name,
        count=n,
        p50=p50,
        p95=p95,
        p99=p99,
        mean=mean,
        min=total_times[0],
        max=total_times[-1],
        threshold_p50=bucket.p50_threshold_sec,
        threshold_p95=bucket.p95_threshold_sec,
        pass_fail=pass_fail,
        is_insufficient_data=False,  # Sufficient samples
    )


def validate_workflow_version_consistency(
    capsules: List[PerformanceCapsule],
    *,
    strict: bool = True,
) -> None:
    """Validate workflow version consistency within zones (proxy for run contamination).

    Checks that each zone contains capsules from only one workflow version (v1 OR v2).
    Mixed workflow versions within a zone may indicate multi-run ledger contamination.

    Args:
        capsules: List of performance capsules to validate
        strict: If True, raises ValueError on mixed versions (default).
                If False, logs warning only (for forensic analysis).

    Raises:
        ValueError: If strict=True and mixed workflow versions detected in any zone

    Design note:
        This function validates workflow_version + zone consistency as a HEURISTIC
        for detecting multi-run contamination. Stronger invariants (commit_sha, run_id)
        require schema evolution to add those fields to performance_capsules table.

        Current approach is sufficient for common CI contamination scenarios where
        mixed workflow versions indicate accidental data mixing from separate runs.

        In CI: DB is ephemeral per job (unique temp/workspace path), so contamination
        should not occur unless DB path or artifacts are reused across jobs.
        Locally: pass strict=False and filter by commit/run explicitly in queries.
    """
    if not capsules:
        return

    # Extract unique run contexts (image_id prefixes, timestamps, commit metadata)
    # For now, just validate workflow_version consistency within each zone
    # (a proxy for detecting mixed-run contamination)

    zones_workflows: Dict[str, set[str]] = {}
    for capsule in capsules:
        zone = capsule.zone or "unknown"
        wf_ver = getattr(capsule, "workflow_version", "v1")
        zones_workflows.setdefault(zone, set()).add(wf_ver)

    # Each zone should have capsules from only v1 OR v2, not both
    # (unless explicitly doing dual-workflow comparison)
    mixed_zones = {z: wfs for z, wfs in zones_workflows.items() if len(wfs) > 1}

    if mixed_zones:
        msg = (
            f"Detected mixed workflow versions in zones: {mixed_zones}. "
            f"This may indicate multi-run ledger contamination or intentional V1/V2 comparison."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)


def compute_per_zone_stats(
    capsules: List[PerformanceCapsule],
    buckets: Optional[List[PerformanceBucket]] = None,
) -> Dict[str, Dict[str, BucketStats]]:
    """Compute per-zone bucket statistics.

    Args:
        capsules: List of performance capsules
        buckets: List of bucket definitions (defaults to DEFAULT_BUCKETS)

    Returns:
        Dict mapping zone -> bucket_name -> BucketStats
        Example: {"us-west-2a": {"pool_medium_mps": BucketStats(...), ...}, ...}
    """
    if buckets is None:
        buckets = DEFAULT_BUCKETS

    # Validate workflow version consistency (intentionally strict: raises on contamination)
    validate_workflow_version_consistency(capsules)

    # Group capsules by zone
    capsules_by_zone: Dict[str, List[PerformanceCapsule]] = {}
    for capsule in capsules:
        zone = capsule.zone or "unknown"
        capsules_by_zone.setdefault(zone, []).append(capsule)

    # Compute stats for each zone
    per_zone_stats: Dict[str, Dict[str, BucketStats]] = {}

    for zone, zone_capsules in capsules_by_zone.items():
        zone_stats: Dict[str, BucketStats] = {}

        for bucket in buckets:
            bucket_stats = compute_bucket_stats(zone_capsules, bucket)
            if bucket_stats:
                zone_stats[bucket.name] = bucket_stats

        if zone_stats:
            per_zone_stats[zone] = zone_stats

    return per_zone_stats


def compute_global_stats(
    capsules: List[PerformanceCapsule],
    buckets: Optional[List[PerformanceBucket]] = None,
) -> Dict[str, BucketStats]:
    """Compute global bucket statistics (across all zones).

    Args:
        capsules: List of performance capsules
        buckets: List of bucket definitions (defaults to DEFAULT_BUCKETS)

    Returns:
        Dict mapping bucket_name -> BucketStats
    """
    if buckets is None:
        buckets = DEFAULT_BUCKETS

    global_stats: Dict[str, BucketStats] = {}

    for bucket in buckets:
        bucket_stats = compute_bucket_stats(capsules, bucket)
        if bucket_stats:
            global_stats[bucket.name] = bucket_stats

    return global_stats


def compute_worst_zone_p95(per_zone_stats: Dict[str, Dict[str, BucketStats]]) -> Tuple[Optional[str], Optional[float]]:
    """Find the zone with the worst (highest) p95 across all buckets.

    This is critical for gating - we gate on worst-case user experience.

    Args:
        per_zone_stats: Per-zone bucket statistics

    Returns:
        Tuple of (worst_zone_name, worst_p95) or (None, None) if no stats
    """
    worst_zone: Optional[str] = None
    worst_p95: Optional[float] = None

    for zone, bucket_stats in per_zone_stats.items():
        for bucket_name, stats in bucket_stats.items():
            if worst_p95 is None or stats.p95 > worst_p95:
                worst_p95 = stats.p95
                worst_zone = zone

    return worst_zone, worst_p95


def log_aggregated_stats_to_ledger(
    run_id: str,
    commit_sha: str,
    workflow_version: str,
    timestamp: str,
    per_zone_stats: Mapping[Any, Mapping[str, BucketStats]],
    ledger_db_path: str,
) -> None:
    """Write aggregated stats to apex_runs table in ledger.

    Args:
        run_id: Run identifier
        commit_sha: Git commit SHA
        workflow_version: Workflow version ("v1" or "v2")
        timestamp: ISO8601 timestamp
        per_zone_stats: Per-zone bucket statistics
        ledger_db_path: Path to ledger database
    """
    import sqlite3

    with sqlite3.connect(ledger_db_path) as conn:
        for zone, bucket_stats in per_zone_stats.items():
            for bucket_name, stats in bucket_stats.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO apex_runs (
                        run_id, commit_sha, timestamp, workflow_version, zone,
                        bucket_name, p50, p95, p99, count,
                        threshold_p50, threshold_p95, pass_fail, raw_capsules_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        commit_sha,
                        timestamp,
                        workflow_version,
                        zone,
                        bucket_name,
                        stats.p50,
                        stats.p95,
                        stats.p99,
                        stats.count,
                        stats.threshold_p50,
                        stats.threshold_p95,
                        stats.pass_fail,
                        None,  # raw_capsules_json (optional)
                    ),
                )
        conn.commit()

    logger.info(f"Logged aggregated stats for run {run_id} ({workflow_version})")
