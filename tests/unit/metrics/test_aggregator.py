"""Tests for metrics.aggregator — pure aggregation functions."""

from __future__ import annotations

import pytest

from transformation_portal.metrics.aggregator import (
    compute_bucket_stats,
    compute_global_stats,
    compute_per_zone_stats,
    compute_worst_zone_p95,
    validate_workflow_version_consistency,
)
from transformation_portal.metrics.contracts import BucketStats
from transformation_portal.metrics.performance_capsule import PerformanceBucket, PerformanceCapsule

pytestmark = pytest.mark.unit


def _capsule(total_time: float, zone: str = "us-west-2a", scene_type: str = "pool", device: str = "mps") -> PerformanceCapsule:
    return PerformanceCapsule(
        image_id="img001",
        image_path="/tmp/img001.png",
        input_hash="abc123",
        original_shape=(1920, 1080),
        enforced_shape=(1920, 1080),
        pixel_count=1920 * 1080,
        dimension_adjustment="none",
        timings={"total": total_time, "inference": total_time * 0.8},
        zone=zone,
        scene_type=scene_type,
        device=device,
        workflow_version="v1",
    )


def _bucket(name: str = "pool_mps", p95_threshold: float = 15.0) -> PerformanceBucket:
    return PerformanceBucket(
        name=name,
        filters={"scene_type": "pool", "device": "mps"},
        p50_threshold_sec=10.0,
        p95_threshold_sec=p95_threshold,
    )


def _make_capsules(times: list[float], **kwargs) -> list[PerformanceCapsule]:
    return [_capsule(t, **kwargs) for t in times]


class TestComputeBucketStats:
    def test_returns_none_when_no_matching_capsules(self):
        """Empty capsule list → None."""
        assert compute_bucket_stats([], _bucket()) is None

    def test_returns_none_when_no_capsules_match_filter(self):
        """Capsules that don't match bucket filter → None."""
        capsules = [_capsule(5.0, scene_type="aerial")]
        assert compute_bucket_stats(capsules, _bucket()) is None

    def test_returns_bucket_stats_instance(self):
        """Returns a BucketStats when capsules match."""
        capsules = _make_capsules([5.0, 6.0, 7.0])
        result = compute_bucket_stats(capsules, _bucket())
        assert isinstance(result, BucketStats)

    def test_p50_computed_correctly(self):
        """Median of sorted times is returned as p50."""
        capsules = _make_capsules([1.0, 2.0, 3.0])
        stats = compute_bucket_stats(capsules, _bucket())
        assert stats.p50 == pytest.approx(2.0)

    def test_pass_when_p95_below_threshold(self):
        """pass_fail='pass' when p95 < threshold and sufficient samples."""
        times = [5.0] * 25  # 25 samples, all 5s well below threshold=15
        capsules = _make_capsules(times)
        stats = compute_bucket_stats(capsules, _bucket(p95_threshold=15.0))
        assert stats.pass_fail == "pass"

    def test_fail_when_p95_above_threshold(self):
        """pass_fail='fail' when p95 > threshold and sufficient samples."""
        # 25 samples, most are 20s which exceeds threshold=15
        times = [20.0] * 25
        capsules = _make_capsules(times)
        stats = compute_bucket_stats(capsules, _bucket(p95_threshold=15.0))
        assert stats.pass_fail == "fail"

    def test_insufficient_data_when_below_min_samples(self):
        """n < min_samples (default=20) → is_insufficient_data=True."""
        capsules = _make_capsules([5.0] * 5)
        stats = compute_bucket_stats(capsules, _bucket(), min_samples=20)
        assert stats.is_insufficient_data is True

    def test_insufficient_data_never_fails(self):
        """Insufficient data always yields pass_fail='pass'."""
        capsules = _make_capsules([999.0] * 5)  # terrible latency but few samples
        stats = compute_bucket_stats(capsules, _bucket(), min_samples=20)
        assert stats.pass_fail == "pass"

    def test_sufficient_data_can_fail(self):
        """With enough samples, high p95 yields 'fail'."""
        capsules = _make_capsules([100.0] * 25)  # all very slow
        stats = compute_bucket_stats(capsules, _bucket(p95_threshold=15.0), min_samples=20)
        assert stats.is_insufficient_data is False
        assert stats.pass_fail == "fail"

    def test_bucket_name_propagated(self):
        """BucketStats.bucket_name matches the bucket's name."""
        capsules = _make_capsules([5.0])
        stats = compute_bucket_stats(capsules, _bucket(name="my_bucket"))
        assert stats.bucket_name == "my_bucket"

    def test_count_matches_matching_capsule_count(self):
        """BucketStats.count equals number of matching capsules."""
        capsules = _make_capsules([5.0, 6.0, 7.0])
        stats = compute_bucket_stats(capsules, _bucket())
        assert stats.count == 3


class TestValidateWorkflowVersionConsistency:
    def test_single_version_passes(self):
        """All capsules with same workflow version in same zone → no error."""
        capsules = _make_capsules([5.0] * 3)
        validate_workflow_version_consistency(capsules)

    def test_mixed_versions_strict_raises(self):
        """Mixed v1+v2 in the same zone raises ValueError in strict mode."""
        c1 = _capsule(5.0)
        c2 = _capsule(6.0)
        # Override workflow_version to v2 via object replacement
        import dataclasses

        c2 = dataclasses.replace(c2, workflow_version="v2")
        with pytest.raises(ValueError, match="mixed workflow"):
            validate_workflow_version_consistency([c1, c2], strict=True)

    def test_mixed_versions_non_strict_no_exception(self):
        """Mixed versions with strict=False logs warning but doesn't raise."""
        import dataclasses

        c1 = _capsule(5.0)
        c2 = dataclasses.replace(_capsule(6.0), workflow_version="v2")
        validate_workflow_version_consistency([c1, c2], strict=False)

    def test_empty_capsules_passes(self):
        """Empty list passes without error."""
        validate_workflow_version_consistency([])


class TestComputePerZoneStats:
    def test_groups_by_zone(self):
        """Capsules from two zones produce two keys in result."""
        c1 = _capsule(5.0, zone="us-west-2a")
        c2 = _capsule(6.0, zone="us-east-1a")
        buckets = [_bucket()]
        result = compute_per_zone_stats([c1, c2], buckets=buckets)
        assert "us-west-2a" in result
        assert "us-east-1a" in result

    def test_none_zone_maps_to_unknown(self):
        """Capsule with zone=None is grouped under 'unknown'."""
        import dataclasses

        c = dataclasses.replace(_capsule(5.0), zone=None)
        buckets = [_bucket()]
        result = compute_per_zone_stats([c], buckets=buckets)
        assert "unknown" in result

    def test_empty_capsules_returns_empty_dict(self):
        """No capsules → empty dict."""
        result = compute_per_zone_stats([], buckets=[_bucket()])
        assert not result


class TestComputeGlobalStats:
    def test_returns_bucket_stats_dict(self):
        """Returns a dict mapping bucket_name → BucketStats."""
        capsules = _make_capsules([5.0, 6.0])
        buckets = [_bucket()]
        result = compute_global_stats(capsules, buckets=buckets)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, BucketStats)

    def test_empty_capsules_returns_empty(self):
        """No capsules → empty dict."""
        result = compute_global_stats([], buckets=[_bucket()])
        assert not result


class TestComputeWorstZoneP95:
    def test_returns_zone_with_highest_p95(self):
        """Zone with highest p95 across all buckets is returned."""
        stats_a = BucketStats(
            bucket_name="b",
            count=25,
            p50=5.0,
            p95=20.0,
            p99=25.0,
            mean=10.0,
            min=1.0,
            max=30.0,
            threshold_p50=10.0,
            threshold_p95=15.0,
            pass_fail="fail",
        )
        stats_b = BucketStats(
            bucket_name="b",
            count=25,
            p50=5.0,
            p95=10.0,
            p99=12.0,
            mean=6.0,
            min=1.0,
            max=15.0,
            threshold_p50=10.0,
            threshold_p95=15.0,
            pass_fail="pass",
        )
        per_zone = {"zone-a": {"b": stats_a}, "zone-b": {"b": stats_b}}
        worst_zone, worst_p95 = compute_worst_zone_p95(per_zone)
        assert worst_zone == "zone-a"
        assert worst_p95 == pytest.approx(20.0)

    def test_empty_stats_returns_none_none(self):
        """Empty per_zone_stats → (None, None)."""
        zone, p95 = compute_worst_zone_p95({})
        assert zone is None
        assert p95 is None
