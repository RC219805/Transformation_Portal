"""Versioned, unit-aware APEX measurement evidence; legacy data is descriptive."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.metrics.performance_capsule import (
    DEFAULT_BUCKETS,
    MEASUREMENT_SCHEMA_VERSION,
    PerformanceBucket,
    PerformanceCapsule,
)

SUMMARY_SCHEMA = "tp.apex.measurement.summary.v1"
MIN_SAMPLES = 20


def _context(capsule: PerformanceCapsule) -> tuple[Any, ...]:
    return (
        capsule.measurement_unit,
        capsule.workload_id,
        capsule.workflow_version,
        capsule.zone,
        capsule.backend_id,
        capsule.model_variant,
        capsule.device,
        capsule.dtype,
        capsule.config_hash,
        capsule.pipeline_version,
        capsule.is_synthetic,
    )


def validate_measurement_identities(capsules: Sequence[PerformanceCapsule]) -> None:
    """Reject conflicting identities before workflow, zone or bucket filtering."""
    seen: dict[str, tuple[Any, ...]] = {}
    images: dict[str, tuple[Any, ...]] = {}
    for capsule in capsules:
        # Mutable capsules must still satisfy their declared contract at use time.
        capsule.__post_init__()
        if capsule.measurement_id is None:
            continue
        identity = capsule.measurement_id
        signature = (*_context(capsule), capsule.timings["total"])
        if identity in seen and seen[identity] != signature:
            raise ValueError(f"Conflicting measurement identity: {identity}")
        seen[identity] = signature
        if capsule.measurement_unit == "image_wall_seconds":
            image = (
                capsule.input_hash,
                capsule.original_shape,
                capsule.enforced_shape,
                capsule.pixel_count,
                capsule.scene_type,
                capsule.texture_complexity,
                canonicalize_json(capsule.tile_config),
                capsule.tile_count,
                capsule.cache_hit,
                capsule.cache_key,
                capsule.dimension_adjustment,
            )
            if identity in images and images[identity] != image:
                raise ValueError(f"Image measurement identity reused for different inputs: {identity}")
            images[identity] = image


def assess_measurements(
    capsules: Sequence[PerformanceCapsule],
    bucket: PerformanceBucket,
    *,
    min_samples: int = MIN_SAMPLES,
) -> tuple[dict[str, Any], list[float]]:
    """Count explicit invocations once, preserving total at its measured boundary."""
    if type(min_samples) is not int or min_samples < MIN_SAMPLES:
        raise ValueError("min_samples cannot weaken the minimum of 20")
    bucket.__post_init__()
    validate_measurement_identities(capsules)
    matching = [c for c in capsules if bucket.matches(c)]
    explicit = [c for c in matching if c.schema_version == MEASUREMENT_SCHEMA_VERSION]
    unique = {c.measurement_id: c for c in explicit}
    contexts = {_context(c) for c in explicit}
    units = {c.measurement_unit for c in explicit}
    workloads = {c.workload_id for c in explicit}
    if len(units) > 1:
        raise ValueError("Cannot aggregate mixed measurement units")
    unit = next(iter(units)) if units else None
    workload = next(iter(workloads)) if len(workloads) == 1 else None
    context_id = hashlib.sha256(canonicalize_json(next(iter(contexts)))).hexdigest() if len(contexts) == 1 else None
    count = len(unique) if len(contexts) <= 1 else 0
    reason = ""
    if len(explicit) != len(matching):
        reason = "unattributed_legacy"
    elif len(contexts) != 1:
        reason = "incompatible_workload_or_context"
    elif any(c.is_synthetic for c in matching):
        reason = "synthetic_execution"
    elif any(
        not isinstance(value, str) or not value or value.strip() != value or value.lower() in ("unknown", "unbound")
        for c in matching
        for value in (c.config_hash, c.pipeline_version, c.model_variant, c.backend_id, c.device, c.dtype)
    ):
        reason = "unbound_execution_context"
    elif any(
        c.workflow_version not in ("v1", "v2")
        or (c.zone is not None and (not isinstance(c.zone, str) or not c.zone or c.zone.strip() != c.zone))
        for c in matching
    ):
        reason = "unbound_execution_context"
    elif bucket.measurement_unit != unit or not bucket.workload_id:
        reason = "unbound_or_incompatible_budget"
    elif bucket.workload_id not in ("*", workload):
        reason = "incompatible_budget_workload"
    elif count < min_samples:
        reason = "insufficient_independent_measurements"
    # Legacy vectors are retained only as descriptive statistics. A complete
    # explicit group uses each invocation once, including identical durations.
    totals = (
        [float(c.timings["total"]) for c in unique.values()]
        if len(explicit) == len(matching)
        else [float(c.timings["total"]) for c in matching]
    )
    assessment = {
        "independent_count": count,
        "capsule_count": len(matching),
        "measurement_unit": unit,
        "workload_id": workload,
        "context_id": context_id,
        "is_insufficient_data": bool(reason),
        "reason": reason,
    }
    return assessment, sorted(totals)


def encode_measurement_summary(
    capsules: Sequence[PerformanceCapsule],
    bucket: PerformanceBucket,
    *,
    min_samples: int = MIN_SAMPLES,
) -> dict[str, Any]:
    """Bind the recomputable observation vector and budget in existing JSON storage."""
    assessment, _ = assess_measurements(capsules, bucket, min_samples=min_samples)
    return {
        "schema": SUMMARY_SCHEMA,
        "min_samples": min_samples,
        "bucket": asdict(bucket),
        "capsules": [c.to_dict() for c in capsules],
        "assessment": assessment,
    }


def decode_aggregate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute persisted verdicts; absent historical evidence cannot authorize one."""
    result = dict(row)
    raw = result.get("raw_capsules_json")
    if raw is None:
        evidence: Any = None
    elif isinstance(raw, str):
        evidence = json.loads(raw)
    else:
        evidence = raw
    if evidence is None or isinstance(evidence, list):
        result.update(
            is_insufficient_data=True,
            independent_count=0,
            capsule_count=result.get("count", 0),
            measurement_unit=None,
            workload_id=None,
            context_id=None,
            measurement_zone=None,
            insufficiency_reason="unattributed_legacy",
        )
        return result
    if not isinstance(evidence, dict) or evidence.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("Unknown measurement summary schema")
    if set(evidence) != {"schema", "min_samples", "bucket", "capsules", "assessment"}:
        raise ValueError("Invalid measurement summary fields")
    if not isinstance(evidence["capsules"], list) or not isinstance(evidence["bucket"], dict):
        raise ValueError("Invalid measurement summary payload")
    bucket = PerformanceBucket(**evidence["bucket"])
    for known in DEFAULT_BUCKETS:
        if bucket.name == known.name and asdict(bucket) != asdict(known):
            raise ValueError("Persisted budget differs from current named budget")
    capsules = [PerformanceCapsule.from_dict(c) for c in evidence["capsules"]]
    assessment, _ = assess_measurements(capsules, bucket, min_samples=evidence["min_samples"])
    if canonicalize_json(assessment) != canonicalize_json(evidence["assessment"]):
        raise ValueError("Measurement assessment differs from its observations")
    if result.get("bucket_name") != bucket.name:
        raise ValueError("Measurement bucket does not match aggregate row")
    if any(c.workflow_version != result.get("workflow_version") for c in capsules):
        raise ValueError("Measurement workflow does not match aggregate row")
    zone = result.get("zone")
    if zone is not None and any((c.zone or "unknown") != zone for c in capsules):
        raise ValueError("Measurement zone does not match aggregate row")
    # Lazy import keeps the pure assessment helper independent of aggregate types.
    from transformation_portal.metrics.aggregator import compute_bucket_stats

    stats = compute_bucket_stats(capsules, bucket, min_samples=evidence["min_samples"])
    if stats is None:
        raise ValueError("Measurement summary has no matching observations")
    for name in ("count", "p50", "p95", "p99", "threshold_p50", "threshold_p95", "pass_fail"):
        stored = result.get(name)
        actual = getattr(stats, name)
        if isinstance(actual, (int, float)):
            if (
                isinstance(stored, bool)
                or not isinstance(stored, (int, float))
                or not math.isfinite(stored)
                or stored != actual
            ):
                raise ValueError(f"Aggregate {name} differs from measurement evidence")
        elif stored != actual:
            raise ValueError(f"Aggregate {name} differs from measurement evidence")
    result.update(stats.to_dict())
    result.update(
        independent_count=assessment["independent_count"],
        capsule_count=assessment["capsule_count"],
        measurement_unit=assessment["measurement_unit"],
        workload_id=assessment["workload_id"],
        context_id=assessment["context_id"],
        measurement_zone=next(c.zone for c in capsules if bucket.matches(c)),
        insufficiency_reason=assessment["reason"],
    )
    return result


def recompute_bucket_stats(stats: Any) -> dict[str, Any]:
    """Apply the same evidence check to direct and rehydrated BucketStats callers."""
    row = stats.to_dict()
    evidence = row.pop("measurement_evidence", None)
    row["raw_capsules_json"] = evidence
    if evidence is not None:
        capsules = evidence.get("capsules", [])
        row["workflow_version"] = capsules[0].get("workflow_version") if capsules else None
        row["zone"] = None
    return decode_aggregate_row(row)


def stats_are_comparable(current: Any, baseline: Any) -> bool:
    """Require recomputed evidence and the same boundary except workflow label."""
    left = recompute_bucket_stats(current)
    right = recompute_bucket_stats(baseline)
    if left["is_insufficient_data"] or right["is_insufficient_data"]:
        return False
    left_capsules = [PerformanceCapsule.from_dict(payload) for payload in left["measurement_evidence"]["capsules"]]
    right_capsules = [PerformanceCapsule.from_dict(payload) for payload in right["measurement_evidence"]["capsules"]]
    validate_measurement_identities([*left_capsules, *right_capsules])
    if {c.measurement_id for c in left_capsules} & {c.measurement_id for c in right_capsules}:
        return False
    if any(left[key] != right[key] for key in ("measurement_unit", "workload_id")):
        return False

    def comparison_context(decoded: Mapping[str, Any]) -> tuple[Any, ...]:
        evidence = decoded["measurement_evidence"]
        bucket = PerformanceBucket(**evidence["bucket"])
        capsule = next(
            c for c in (PerformanceCapsule.from_dict(payload) for payload in evidence["capsules"]) if bucket.matches(c)
        )
        context = _context(capsule)
        return (*context[:2], *context[3:])  # A V1/V2 label is the intended comparison axis.

    return comparison_context(left) == comparison_context(right)


def recompute_regression_report(report: Any, current_bucket_stats: Mapping[str, Any] | None = None) -> Any:
    """Legacy regression floats do not authorize a rule without both vectors."""
    evidence = report.measurement_evidence
    if evidence is None:
        return None
    if not isinstance(evidence, dict) or evidence.get("schema") != "tp.apex.comparison.summary.v1":
        raise ValueError("Unknown comparison measurement schema")
    if set(evidence) != {"schema", "current", "baseline", "warn_threshold", "fail_threshold"}:
        raise ValueError("Invalid comparison measurement fields")
    from transformation_portal.metrics.comparator import compare_to_baseline
    from transformation_portal.metrics.contracts import BucketStats

    recomputed = compare_to_baseline(
        {name: BucketStats.from_dict(stats) for name, stats in evidence["current"].items()},
        {name: BucketStats.from_dict(stats) for name, stats in evidence["baseline"].items()},
        report.current_run_id,
        report.baseline_run_id,
        report.current_commit_sha,
        report.baseline_commit_sha,
        evidence["warn_threshold"],
        evidence["fail_threshold"],
    )
    for field in ("bucket_regressions", "max_regression", "max_regression_bucket", "status"):
        if getattr(report, field) != getattr(recomputed, field):
            raise ValueError("Comparison verdict differs from measurement evidence")
    if current_bucket_stats is not None:
        for name in recomputed.bucket_regressions:
            if name not in current_bucket_stats or canonicalize_json(
                evidence["current"][name].get("measurement_evidence")
            ) != canonicalize_json(current_bucket_stats[name].measurement_evidence):
                raise ValueError("Comparison is not bound to current qualifying bucket evidence")
    return recomputed if recomputed.bucket_regressions else None
