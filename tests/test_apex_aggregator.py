"""Tests for APEX aggregator.

Tests validate:
- Per-zone statistics computation
- Global statistics computation
- Worst-zone p95 detection
- Bucket matching and filtering
"""

from transformation_portal.metrics.aggregator import (
    compute_bucket_stats,
    compute_global_stats,
    compute_per_zone_stats,
    compute_worst_zone_p95,
)
from transformation_portal.metrics.performance_capsule import PerformanceBucket, PerformanceCapsule


class TestAggregator:
    """Test APEX aggregator functionality."""

    def create_capsule(
        self,
        image_id: str,
        total_time: float,
        workflow_version: str = "v1",
        zone: str = "local",
        scene_type: str = "pool",
        device: str = "mps",
        pixel_count: int = 10_000_000,
    ) -> PerformanceCapsule:
        """Helper to create test capsule."""
        return PerformanceCapsule(
            image_id=image_id,
            image_path=f"/path/to/{image_id}.jpg",
            input_hash=f"hash_{image_id}",
            original_shape=(3000, 4000),
            enforced_shape=(3000, 4000),
            pixel_count=pixel_count,
            dimension_adjustment="exact",
            timings={"total": total_time, "inference": total_time * 0.8},
            workflow_version=workflow_version,
            zone=zone,
            scene_type=scene_type,
            device=device,
            backend_id="da3",
        )

    def test_compute_bucket_stats(self):
        """Test bucket statistics computation."""
        capsules = [
            self.create_capsule("img1", 8.0, scene_type="pool"),
            self.create_capsule("img2", 10.0, scene_type="pool"),
            self.create_capsule("img3", 12.0, scene_type="pool"),
            self.create_capsule("img4", 14.0, scene_type="pool"),
            self.create_capsule("img5", 16.0, scene_type="pool"),
        ]

        bucket = PerformanceBucket(
            name="pool_medium_mps",
            filters={"scene_type": "pool", "device": "mps"},
            p50_threshold_sec=11.0,
            p95_threshold_sec=15.0,
        )

        stats = compute_bucket_stats(capsules, bucket)

        assert stats is not None
        assert stats.count == 5
        assert stats.p50 == 12.0  # Median
        assert stats.p95 == 16.0  # 95th percentile (small sample)
        assert stats.min == 8.0
        assert stats.max == 16.0
        # With n=5 < min_samples=20, contract requires insufficient_data flag
        assert stats.is_insufficient_data is True
        assert stats.pass_fail == "pass"  # Nominal verdict (flag indicates insufficient data)

    def test_compute_bucket_stats_no_match(self):
        """Test bucket stats returns None when no capsules match."""
        capsules = [
            self.create_capsule("img1", 8.0, scene_type="interior"),
        ]

        bucket = PerformanceBucket(
            name="pool_medium_mps",
            filters={"scene_type": "pool"},
            p50_threshold_sec=11.0,
            p95_threshold_sec=15.0,
        )

        stats = compute_bucket_stats(capsules, bucket)
        assert stats is None

    def test_compute_per_zone_stats(self):
        """Test per-zone statistics computation."""
        capsules = [
            self.create_capsule("img1", 8.0, zone="us-west-2a", scene_type="pool"),
            self.create_capsule("img2", 10.0, zone="us-west-2a", scene_type="pool"),
            self.create_capsule("img3", 12.0, zone="us-west-2b", scene_type="pool"),
            self.create_capsule("img4", 14.0, zone="us-west-2b", scene_type="pool"),
        ]

        buckets = [
            PerformanceBucket(
                name="pool_medium_mps",
                filters={"scene_type": "pool", "device": "mps"},
                p50_threshold_sec=11.0,
                p95_threshold_sec=15.0,
            )
        ]

        per_zone = compute_per_zone_stats(capsules, buckets)

        assert "us-west-2a" in per_zone
        assert "us-west-2b" in per_zone
        assert "pool_medium_mps" in per_zone["us-west-2a"]
        assert "pool_medium_mps" in per_zone["us-west-2b"]

        # Zone A has lower p95
        assert per_zone["us-west-2a"]["pool_medium_mps"].p95 < per_zone["us-west-2b"]["pool_medium_mps"].p95

    def test_compute_global_stats(self):
        """Test global statistics computation (across all zones)."""
        capsules = [
            self.create_capsule("img1", 8.0, zone="us-west-2a", scene_type="pool"),
            self.create_capsule("img2", 10.0, zone="us-west-2a", scene_type="pool"),
            self.create_capsule("img3", 12.0, zone="us-west-2b", scene_type="pool"),
            self.create_capsule("img4", 14.0, zone="us-west-2b", scene_type="pool"),
        ]

        buckets = [
            PerformanceBucket(
                name="pool_medium_mps",
                filters={"scene_type": "pool", "device": "mps"},
                p50_threshold_sec=11.0,
                p95_threshold_sec=15.0,
            )
        ]

        global_stats = compute_global_stats(capsules, buckets)

        assert "pool_medium_mps" in global_stats
        assert global_stats["pool_medium_mps"].count == 4
        # Median of [8, 10, 12, 14] is (10+12)/2 = 11.0 (correct median for even-sized sample)
        assert global_stats["pool_medium_mps"].p50 == 11.0

    def test_compute_worst_zone_p95(self):
        """Test worst-zone p95 detection."""
        per_zone_stats = {
            "us-west-2a": {
                "pool_medium_mps": type("BucketStats", (), {"p95": 10.0})(),
            },
            "us-west-2b": {
                "pool_medium_mps": type("BucketStats", (), {"p95": 15.0})(),
            },
            "us-east-1a": {
                "pool_medium_mps": type("BucketStats", (), {"p95": 12.0})(),
            },
        }

        worst_zone, worst_p95 = compute_worst_zone_p95(per_zone_stats)

        assert worst_zone == "us-west-2b"
        assert worst_p95 == 15.0

    def test_compute_worst_zone_p95_empty(self):
        """Test worst-zone p95 with empty stats."""
        per_zone_stats = {}

        worst_zone, worst_p95 = compute_worst_zone_p95(per_zone_stats)

        assert worst_zone is None
        assert worst_p95 is None

    def test_workflow_version_filtering(self):
        """Test that workflow_version filters work correctly."""
        capsules = [
            self.create_capsule("img1", 8.0, workflow_version="v1", scene_type="pool"),
            self.create_capsule("img2", 10.0, workflow_version="v2", scene_type="pool"),
        ]

        # Bucket filters for v1 only
        bucket_v1 = PerformanceBucket(
            name="pool_v1",
            filters={"scene_type": "pool", "workflow_version": "v1"},
            p50_threshold_sec=11.0,
            p95_threshold_sec=15.0,
        )

        stats_v1 = compute_bucket_stats(capsules, bucket_v1)

        assert stats_v1 is not None
        assert stats_v1.count == 1  # Only v1 capsule matched

    def test_zone_filtering(self):
        """Test that zone filters work correctly."""
        capsules = [
            self.create_capsule("img1", 8.0, zone="us-west-2a", scene_type="pool"),
            self.create_capsule("img2", 10.0, zone="us-west-2b", scene_type="pool"),
        ]

        # Bucket filters for us-west-2a only
        bucket_zone_a = PerformanceBucket(
            name="pool_zone_a",
            filters={"scene_type": "pool", "zone": "us-west-2a"},
            p50_threshold_sec=11.0,
            p95_threshold_sec=15.0,
        )

        stats_zone_a = compute_bucket_stats(capsules, bucket_zone_a)

        assert stats_zone_a is not None
        assert stats_zone_a.count == 1  # Only zone A capsule matched


class TestValidateSingleRun:
    """Test single-run validation for contamination detection."""

    def create_capsule(
        self,
        image_id: str,
        workflow_version: str = "v1",
        zone: str = "local",
        run_id: str | None = None,
        commit_sha: str | None = None,
    ) -> PerformanceCapsule:
        """Helper to create test capsule with run metadata."""
        # Note: PerformanceCapsule doesn't have image_metadata field
        # Run context is detected from workflow_version + zone patterns
        return PerformanceCapsule(
            image_id=image_id,
            image_path=f"/path/to/{image_id}.jpg",
            input_hash=f"hash_{image_id}",
            original_shape=(3000, 4000),
            enforced_shape=(3000, 4000),
            pixel_count=10_000_000,
            dimension_adjustment="exact",
            timings={"total": 10.0},
            workflow_version=workflow_version,
            zone=zone,
            scene_type="pool",
            device="cpu",
            backend_id="da3",
        )

    def test_strict_raises_on_mixed_workflow_versions(self):
        """Test that strict=True raises ValueError on mixed workflow versions in same zone."""
        import pytest

        from transformation_portal.metrics.aggregator import validate_single_run_capsules

        capsules = [
            self.create_capsule("img1", workflow_version="v1", zone="local"),
            self.create_capsule("img2", workflow_version="v2", zone="local"),  # Mixed!
        ]

        with pytest.raises(ValueError, match="mixed workflow versions"):
            validate_single_run_capsules(capsules, strict=True)

    def test_non_strict_warns_on_mixed_workflow_versions(self):
        """Test that strict=False logs warning but doesn't raise on contamination."""
        import logging

        from transformation_portal.metrics.aggregator import validate_single_run_capsules

        capsules = [
            self.create_capsule("img1", workflow_version="v1", zone="local"),
            self.create_capsule("img2", workflow_version="v2", zone="local"),
        ]

        # Should not raise
        with caplog_context(logging.WARNING) as records:
            validate_single_run_capsules(capsules, strict=False)

        # Should have logged warning
        assert any("mixed workflow versions" in rec.message.lower() for rec in records)

    def test_happy_path_single_workflow_no_raise(self):
        """Test that clean single-workflow data doesn't raise or warn."""
        from transformation_portal.metrics.aggregator import validate_single_run_capsules

        capsules = [
            self.create_capsule("img1", workflow_version="v1", zone="local"),
            self.create_capsule("img2", workflow_version="v1", zone="local"),
            self.create_capsule("img3", workflow_version="v1", zone="local"),
        ]

        # Should not raise (no assertion needed - test passes if no exception)
        validate_single_run_capsules(capsules, strict=True)

    def test_multi_zone_same_workflow_allowed(self):
        """Test that different zones with same workflow version is allowed."""
        from transformation_portal.metrics.aggregator import validate_single_run_capsules

        capsules = [
            self.create_capsule("img1", workflow_version="v1", zone="zone-a"),
            self.create_capsule("img2", workflow_version="v1", zone="zone-b"),
            self.create_capsule("img3", workflow_version="v1", zone="zone-c"),
        ]

        # Should not raise - multiple zones is expected for matrix runs
        validate_single_run_capsules(capsules, strict=True)

    def test_empty_capsules_no_raise(self):
        """Test that empty capsule list doesn't raise."""
        from transformation_portal.metrics.aggregator import validate_single_run_capsules

        validate_single_run_capsules([], strict=True)


def caplog_context(level):
    """Helper context manager to capture log records."""
    import logging

    class LogCapture:
        def __init__(self, level):
            self.level = level
            self.records = []
            self.handler = None

        def __enter__(self):
            self.handler = logging.Handler()
            self.handler.setLevel(self.level)
            self.handler.emit = lambda record: self.records.append(record)
            logging.getLogger().addHandler(self.handler)
            return self.records

        def __exit__(self, *args):
            if self.handler:
                logging.getLogger().removeHandler(self.handler)

    return LogCapture(level)
