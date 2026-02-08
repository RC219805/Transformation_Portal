"""Tests for APEX workflow contracts.

Tests validate:
- RunSpec immutability and serialization
- Observation linkage to RunSpec
- Judgement construction and properties
- BucketStats and RegressionReport
"""

import pytest

from transformation_portal.metrics.contracts import BucketStats, Judgement, Observation, RegressionReport, RunSpec
from transformation_portal.metrics.performance_capsule import PerformanceCapsule


class TestRunSpec:
    """Test RunSpec contract."""

    def test_runspec_creation(self):
        """Test basic RunSpec creation."""
        spec = RunSpec(
            run_id="test123",
            commit_sha="abc123",
            workflow_version="v1",
            zones=["us-west-2a"],
            device="mps",
            backend_id="da3",
        )

        assert spec.run_id == "test123"
        assert spec.commit_sha == "abc123"
        assert spec.workflow_version == "v1"
        assert spec.zones == ["us-west-2a"]

    def test_runspec_immutable(self):
        """Test that RunSpec is immutable (frozen dataclass)."""
        spec = RunSpec(
            run_id="test123",
            commit_sha="abc123",
            workflow_version="v1",
            zones=["local"],
            device="cpu",
            backend_id="da3",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            spec.run_id = "modified"

    def test_runspec_serialization(self):
        """Test RunSpec serialization roundtrip."""
        spec = RunSpec(
            run_id="test123",
            commit_sha="abc123",
            workflow_version="v2",
            zones=["us-west-2a", "us-west-2b"],
            device="mps",
            backend_id="da3",
            scene_type="pool",
        )

        spec_dict = spec.to_dict()
        restored = RunSpec.from_dict(spec_dict)

        assert restored.run_id == spec.run_id
        assert restored.workflow_version == spec.workflow_version
        assert restored.zones == spec.zones

    def test_runspec_hash(self):
        """Test RunSpec deterministic hashing."""
        timestamp = "2026-02-07T12:00:00.000000+00:00"

        spec1 = RunSpec(
            run_id="test123",
            commit_sha="abc123",
            workflow_version="v1",
            zones=["local"],
            device="cpu",
            backend_id="da3",
            timestamp=timestamp,
        )

        spec2 = RunSpec(
            run_id="test123",
            commit_sha="abc123",
            workflow_version="v1",
            zones=["local"],
            device="cpu",
            backend_id="da3",
            timestamp=timestamp,
        )

        # Same spec should produce same hash
        assert spec1.to_hash() == spec2.to_hash()

        # Different spec should produce different hash
        spec3 = RunSpec(
            run_id="different",
            commit_sha="abc123",
            workflow_version="v1",
            zones=["local"],
            device="cpu",
            backend_id="da3",
            timestamp=timestamp,
        )
        assert spec1.to_hash() != spec3.to_hash()


class TestObservation:
    """Test Observation contract."""

    def test_observation_creation(self):
        """Test basic Observation creation."""
        spec = RunSpec(
            run_id="test123",
            commit_sha="abc123",
            workflow_version="v1",
            zones=["local"],
            device="cpu",
            backend_id="da3",
        )

        capsule = PerformanceCapsule(
            image_id="test_img",
            image_path="/path/to/test.jpg",
            input_hash="hash123",
            original_shape=(1000, 1500),
            enforced_shape=(1000, 1500),
            pixel_count=1_500_000,
            dimension_adjustment="exact",
            timings={"total": 5.0},
            workflow_version="v1",
            zone="local",
        )

        obs = Observation(
            run_spec=spec,
            zone="local",
            capsules=[capsule],
        )

        assert obs.run_spec.run_id == "test123"
        assert obs.zone == "local"
        assert len(obs.capsules) == 1
        assert not obs.has_errors

    def test_observation_with_errors(self):
        """Test Observation with errors."""
        spec = RunSpec(
            run_id="test123",
            commit_sha="abc123",
            workflow_version="v1",
            zones=["local"],
            device="cpu",
            backend_id="da3",
        )

        obs = Observation(
            run_spec=spec,
            zone="local",
            capsules=[],
            errors=["Model failed to load", "OOM error"],
        )

        assert obs.has_errors
        assert len(obs.errors) == 2
        assert obs.sample_count == 0


class TestJudgement:
    """Test Judgement contract."""

    def test_judgement_creation(self):
        """Test basic Judgement creation."""
        bucket_stats = {
            "test_bucket": BucketStats(
                bucket_name="test_bucket",
                count=10,
                p50=5.0,
                p95=8.0,
                p99=9.0,
                mean=5.5,
                min=3.0,
                max=10.0,
                threshold_p50=6.0,
                threshold_p95=10.0,
                pass_fail="pass",
            )
        }

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=None,
            pass_fail="pass",
            explanation="All thresholds met",
        )

        assert judgement.run_id == "test123"
        assert not judgement.is_blocking
        assert "test_bucket" in judgement.bucket_stats

    def test_judgement_blocking(self):
        """Test Judgement blocking property."""
        judgement_fail = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats={},
            regression_report=None,
            pass_fail="fail",
            explanation="Threshold exceeded",
        )

        assert judgement_fail.is_blocking

        judgement_warn = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats={},
            regression_report=None,
            pass_fail="warn",
            explanation="Close to threshold",
        )

        assert not judgement_warn.is_blocking


class TestBucketStats:
    """Test BucketStats."""

    def test_bucket_stats_creation(self):
        """Test BucketStats creation and serialization."""
        stats = BucketStats(
            bucket_name="pool_medium_mps",
            count=15,
            p50=11.0,
            p95=14.5,
            p99=15.2,
            mean=11.5,
            min=9.0,
            max=16.0,
            threshold_p50=11.0,
            threshold_p95=15.0,
            pass_fail="pass",
        )

        assert stats.bucket_name == "pool_medium_mps"
        assert stats.count == 15
        assert stats.pass_fail == "pass"

        # Test serialization
        stats_dict = stats.to_dict()
        restored = BucketStats.from_dict(stats_dict)

        assert restored.p95 == stats.p95


class TestRegressionReport:
    """Test RegressionReport."""

    def test_regression_report_creation(self):
        """Test RegressionReport creation."""
        report = RegressionReport(
            baseline_run_id="baseline123",
            baseline_commit_sha="abc123",
            current_run_id="current456",
            current_commit_sha="def456",
            bucket_regressions={"bucket1": 0.05, "bucket2": 0.18},
            max_regression=0.18,
            max_regression_bucket="bucket2",
            status="fail",
            explanation="Regression detected in bucket2",
        )

        assert report.max_regression == 0.18
        assert report.max_regression_bucket == "bucket2"
        assert report.status == "fail"

        # Test serialization
        report_dict = report.to_dict()
        restored = RegressionReport.from_dict(report_dict)

        assert restored.max_regression == report.max_regression
