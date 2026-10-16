"""Measurement independence, unit/budget compatibility and durable evidence contracts."""

import json
import sqlite3
from dataclasses import replace

import pytest

from transformation_portal.metrics.aggregator import (
    compute_bucket_stats,
    compute_per_zone_stats,
    log_aggregated_stats_to_ledger,
)
from transformation_portal.metrics.comparator import compare_to_baseline, query_baseline_stats
from transformation_portal.metrics.contracts import BucketStats, Judgement
from transformation_portal.metrics.ledger import PerformanceLedger
from transformation_portal.metrics.measurement import decode_aggregate_row, recompute_bucket_stats, stats_are_comparable
from transformation_portal.metrics.performance_capsule import PerformanceBucket, PerformanceCapsule

pytestmark = pytest.mark.unit


def capsule(identity="measurement-1", *, unit="image_wall_seconds", total=10.0, **overrides):
    payload = dict(
        image_id=identity,
        image_path=f"/{identity}.jpg",
        input_hash=identity,
        original_shape=(100, 100),
        enforced_shape=(100, 100),
        pixel_count=10000,
        dimension_adjustment="exact",
        timings={"total": total},
        schema_version="2.1.0",
        measurement_id=identity,
        measurement_unit=unit,
        workload_id="source-and-recipe-digest",
        config_hash="frozen-config",
        backend_id="da3",
        model_variant="metric",
        pipeline_version="source-1",
        workflow_version="v1",
        zone="local",
        device="cpu",
    )
    payload.update(overrides)
    return PerformanceCapsule(**payload)


def budget(**overrides):
    payload = dict(
        name="explicit-image-budget",
        filters={},
        p50_threshold_sec=9.0,
        p95_threshold_sec=12.0,
        measurement_unit="image_wall_seconds",
        workload_id="*",
    )
    payload.update(overrides)
    return PerformanceBucket(**payload)


def independent_images(*, prefix="measurement", **overrides):
    return [capsule(f"{prefix}-{i}", **overrides) for i in range(20)]


def test_equal_durations_with_twenty_distinct_identities_are_independent():
    stats = compute_bucket_stats(independent_images(), budget())
    assert stats.count == 20
    assert not stats.is_insufficient_data
    assert stats.p50 == stats.p95 == 10.0


def test_duplicate_identity_is_idempotent_but_conflicts_fail_before_filtering():
    first = capsule()
    stats = compute_bucket_stats([first, replace(first)], budget())
    assert stats.count == 1
    for changed in (
        replace(first, timings={"total": 20.0}),
        replace(first, zone="other"),
        replace(first, device="mps"),
        replace(first, workload_id="other"),
        replace(first, input_hash="other"),
        replace(first, scene_type="pool"),
        replace(first, texture_complexity="smooth"),
        replace(first, cache_hit=True),
        replace(first, tile_config={"tile_size": 16}),
    ):
        with pytest.raises(ValueError, match="identity"):
            compute_bucket_stats([first, changed], budget(filters={"zone": "local"}))


def test_batch_invocations_count_once_each_with_original_lifecycle_totals():
    records = [capsule("call-1", unit="batch_wall_seconds", image_id=f"photo-{i}", total=30.0) for i in range(20)]
    records += [capsule("call-2", unit="batch_wall_seconds", image_id=f"photo-{i}", total=40.0) for i in range(20)]
    stats = compute_bucket_stats(records, budget())
    assert stats.count == 2
    assert stats.p50 == 35.0 and stats.p95 == 40.0
    assert stats.is_insufficient_data
    assert stats.measurement_evidence["assessment"]["capsule_count"] == 40


def test_per_zone_and_global_identity_conflicts_cannot_be_hidden():
    with pytest.raises(ValueError, match="identity"):
        compute_per_zone_stats([capsule(), capsule(zone="other")], [budget()])


@pytest.mark.parametrize(
    "changed",
    [
        {"workload_id": "other"},
        {"device": "mps"},
        {"config_hash": "other"},
        {"model_variant": "other"},
        {"pipeline_version": "other"},
        {"zone": "other"},
    ],
)
def test_wildcard_budget_never_pools_incompatible_execution_context(changed):
    records = independent_images()
    records[-1] = replace(records[-1], **changed)
    stats = compute_bucket_stats(records, budget())
    assert stats.is_insufficient_data and stats.count == 0
    assert stats.measurement_evidence["assessment"]["reason"] == "incompatible_workload_or_context"


def test_mixed_units_are_rejected_and_unbound_budgets_abstain():
    with pytest.raises(ValueError, match="mixed measurement units"):
        compute_bucket_stats([capsule(), capsule("batch", unit="batch_wall_seconds")], budget())
    for binding in ({"measurement_unit": None, "workload_id": None}, {"workload_id": "different-workload"}):
        stats = compute_bucket_stats(independent_images(), budget(**binding))
        assert stats.is_insufficient_data
    stats = compute_bucket_stats(independent_images(unit="batch_wall_seconds"), budget())
    assert stats.count == 20 and stats.is_insufficient_data


@pytest.mark.parametrize(
    "field,value",
    [
        ("p50_threshold_sec", float("nan")),
        ("p95_threshold_sec", float("inf")),
        ("p95_threshold_sec", -1),
        ("p50_threshold_sec", True),
        ("measurement_unit", "unrecognized"),
        ("workload_id", " "),
    ],
)
def test_declared_budget_binding_rejects_malformed_thresholds_and_metadata(field, value):
    with pytest.raises(ValueError):
        budget(**{field: value})


def test_legacy_mixed_or_synthetic_records_do_not_authorize_budget():
    records = independent_images()
    records[-1] = replace(records[-1], schema_version="2.0.0", measurement_id=None, measurement_unit=None, workload_id=None)
    assert compute_bucket_stats(records, budget()).is_insufficient_data
    assert compute_bucket_stats(independent_images(is_synthetic=True), budget()).is_insufficient_data


@pytest.mark.parametrize("field", ["config_hash", "pipeline_version", "model_variant", "backend_id", "device", "dtype"])
@pytest.mark.parametrize("value", ["", " ", "unknown", "unbound"])
def test_measurement_execution_context_cannot_be_blank_or_declared_unbound(field, value):
    stats = compute_bucket_stats(independent_images(**{field: value}), budget())
    assert stats.is_insufficient_data
    assert stats.measurement_evidence["assessment"]["reason"] == "unbound_execution_context"


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_measurement_schema_requires_boolean_execution_mode(value):
    with pytest.raises(ValueError, match="boolean is_synthetic"):
        capsule(is_synthetic=value)


@pytest.mark.parametrize("field,value", [("workflow_version", "invalid"), ("zone", ""), ("zone", " ")])
def test_measurement_topology_cannot_use_invalid_workflow_or_blank_declared_zone(field, value):
    stats = compute_bucket_stats(independent_images(**{field: value}), budget())
    assert stats.is_insufficient_data
    assert stats.measurement_evidence["assessment"]["reason"] == "unbound_execution_context"


def test_persisted_evidence_roundtrip_and_tamper_detection(tmp_path):
    db = tmp_path / "ledger.db"
    PerformanceLedger(db)
    stats = compute_bucket_stats(independent_images(), budget())
    log_aggregated_stats_to_ledger(
        "run-1", "source-1", "v1", "2026-10-05T00:00:00Z", {"local": {stats.bucket_name: stats}}, str(db)
    )
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM apex_runs").fetchone())
    decoded = decode_aggregate_row(row)
    assert decoded["independent_count"] == 20 and not decoded["is_insufficient_data"]
    assert (
        query_baseline_stats(str(db), "v1", zone="local")[stats.bucket_name].measurement_evidence == stats.measurement_evidence
    )
    for field, value in (("count", 21), ("p95", 100.0), ("pass_fail", "fail"), ("workflow_version", "v2"), ("zone", "other")):
        with pytest.raises(ValueError):
            decode_aggregate_row(dict(row, **{field: value}))
    evidence = json.loads(row["raw_capsules_json"])
    evidence["assessment"]["independent_count"] = 999
    with pytest.raises(ValueError, match="assessment"):
        decode_aggregate_row(dict(row, raw_capsules_json=json.dumps(evidence)))


def test_baseline_global_selection_is_explicit_and_duplicate_guard_remains(tmp_path):
    db = tmp_path / "ledger.db"
    PerformanceLedger(db)
    stats = compute_bucket_stats(independent_images(), budget())
    log_aggregated_stats_to_ledger(
        "run",
        "source",
        "v1",
        "2026-10-05T00:00:00Z",
        {"local": {stats.bucket_name: stats}, None: {stats.bucket_name: stats}},
        str(db),
    )
    selected = query_baseline_stats(str(db), "v1")
    assert selected[stats.bucket_name].measurement_evidence == stats.measurement_evidence
    # SQLite nullable PKs permit duplicate global rows: readers reject ambiguity.
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO apex_runs SELECT * FROM apex_runs WHERE zone IS NULL")
    with pytest.raises(ValueError, match="explicit zone"):
        query_baseline_stats(str(db), "v1")


def test_legacy_bucket_stats_and_nested_judgement_keep_exact_serialized_keys():
    stats = BucketStats("legacy", 20, 1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 1.0, 2.0, "fail")
    payload = stats.to_dict()
    assert "measurement_evidence" not in payload
    assert BucketStats.from_dict(payload).to_dict() == payload
    judgement = Judgement("run", "v1", "local", {"legacy": stats}, None, "fail", "old row")
    serialized = judgement.to_dict()
    assert serialized["bucket_stats"]["legacy"] == payload
    assert Judgement.from_dict(serialized).to_dict() == serialized
    assert recompute_bucket_stats(stats)["is_insufficient_data"]


def test_comparison_requires_independent_compatible_evidence():
    first = compute_bucket_stats(independent_images(), budget())
    second = compute_bucket_stats(independent_images(prefix="current", total=20.0), budget())
    report = compare_to_baseline(
        {first.bucket_name: second}, {first.bucket_name: first}, "new", "old", "new-source", "old-source"
    )
    assert report.status == "fail" and report.max_regression == 1.0
    for different in (
        compute_bucket_stats(independent_images(prefix="current", workload_id="other"), budget()),
        compute_bucket_stats(independent_images(prefix="current", device="mps"), budget()),
        replace(second, measurement_evidence=None),
    ):
        assert not stats_are_comparable(first, different)
        assert not compare_to_baseline(
            {first.bucket_name: different}, {first.bucket_name: first}, "new", "old", "new-source", "old-source"
        ).bucket_regressions


def test_v1_v2_label_is_comparison_axis_but_recipe_substitution_is_not():
    first = compute_bucket_stats(independent_images(), budget())
    second = compute_bucket_stats(independent_images(prefix="v2", workflow_version="v2"), budget())
    assert stats_are_comparable(first, second)
    changed = compute_bucket_stats(
        independent_images(prefix="v2", workflow_version="v2", config_hash="different-recipe"), budget()
    )
    assert not stats_are_comparable(first, changed)


def test_comparison_uses_matching_context_after_nonmatching_first_capsule():
    scoped_budget = budget(filters={"scene_type": "pool"})
    unrelated = capsule("unrelated", scene_type="aerial", model_variant="unrelated-model")
    first = compute_bucket_stats([unrelated, *independent_images(scene_type="pool")], scoped_budget)
    second = compute_bucket_stats(
        [
            replace(unrelated, measurement_id="other-unrelated"),
            *independent_images(prefix="other", scene_type="pool", model_variant="different-matching-model"),
        ],
        scoped_budget,
    )
    assert not stats_are_comparable(first, second)
    compatible = compute_bucket_stats(
        [
            replace(unrelated, measurement_id="other-unrelated", model_variant="other-unrelated"),
            *independent_images(prefix="other", scene_type="pool"),
        ],
        scoped_budget,
    )
    assert stats_are_comparable(first, compatible)


def test_comparison_rejects_shared_or_conflicting_measurement_identities():
    first = compute_bucket_stats(independent_images(), budget())
    assert not stats_are_comparable(first, first)
    conflicting = compute_bucket_stats(independent_images(total=20.0), budget())
    with pytest.raises(ValueError, match="identity"):
        stats_are_comparable(first, conflicting)
    distinct_equal_values = compute_bucket_stats(independent_images(prefix="fresh"), budget())
    assert stats_are_comparable(first, distinct_equal_values)


@pytest.mark.parametrize("warn,fail", [(float("nan"), 0.15), (0.1, float("inf")), (-0.1, 0.15), (0.2, 0.1), (True, 0.15)])
def test_comparison_threshold_metadata_cannot_disable_classification(warn, fail):
    with pytest.raises(ValueError, match="thresholds"):
        compare_to_baseline({}, {}, "current", "baseline", "source", "source", warn, fail)
