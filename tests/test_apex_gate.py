"""Tests for APEX gate logic.

Tests validate:
- Gate rule evaluation
- Mode enforcement (enforce/shadow/disabled)
- Worst-zone p95 gating
- Regression gating
- Bucket threshold gating
"""

import pytest

from transformation_portal.metrics.contracts import (
    BucketStats,
    Judgement,
    RegressionReport,
)
from transformation_portal.metrics.gate import evaluate_gate, should_block


class TestGate:
    """Test APEX gate functionality."""

    def create_bucket_stats(
        self,
        name: str,
        p95: float,
        threshold_p95: float = 15.0,
        pass_fail: str = "pass",
    ) -> BucketStats:
        """Helper to create test BucketStats."""
        return BucketStats(
            bucket_name=name,
            count=10,
            p50=p95 * 0.7,
            p95=p95,
            p99=p95 * 1.1,
            mean=p95 * 0.75,
            min=p95 * 0.5,
            max=p95 * 1.2,
            threshold_p50=threshold_p95 * 0.7,
            threshold_p95=threshold_p95,
            pass_fail=pass_fail,
        )

    def test_gate_disabled_always_passes(self):
        """Test that disabled mode always passes."""
        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats={},
            regression_report=None,
            pass_fail="fail",  # Even with fail verdict
            explanation="Threshold exceeded",
        )

        result = evaluate_gate(judgement, mode="disabled")

        assert not result.should_block
        assert result.mode == "disabled"
        assert "disabled" in result.explanation.lower()

    def test_gate_bucket_threshold_violation(self):
        """Test gate blocks on bucket threshold violation."""
        bucket_stats = {"pool_medium": self.create_bucket_stats("pool_medium", p95=16.0, threshold_p95=15.0, pass_fail="fail")}

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=None,
            pass_fail="fail",
            explanation="Bucket threshold exceeded",
        )

        result = evaluate_gate(judgement, mode="enforce")

        assert result.should_block
        assert "threshold violation" in result.explanation.lower()
        assert len(result.reasons) > 0

    def test_gate_worst_zone_p95_violation(self):
        """Test gate blocks on worst-zone p95 violation."""
        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats={},
            regression_report=None,
            pass_fail="pass",
            explanation="Within thresholds",
            worst_zone_p95=18.0,
            worst_zone_name="us-west-2b",
        )

        result = evaluate_gate(
            judgement,
            worst_zone_p95_threshold=15.0,
            mode="enforce",
        )

        assert result.should_block
        assert "worst-zone p95" in result.explanation.lower()
        assert "us-west-2b" in result.explanation

    def test_gate_regression_violation(self):
        """Test gate blocks on regression violation."""
        regression_report = RegressionReport(
            baseline_run_id="baseline",
            baseline_commit_sha="abc",
            current_run_id="current",
            current_commit_sha="def",
            bucket_regressions={"pool_medium": 0.20},
            max_regression=0.20,
            max_regression_bucket="pool_medium",
            status="fail",
            explanation="Regression detected",
        )

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats={},
            regression_report=regression_report,
            pass_fail="pass",
            explanation="Regression detected",
        )

        result = evaluate_gate(
            judgement,
            max_regression_threshold=0.15,
            mode="enforce",
        )

        assert result.should_block
        assert "regression" in result.explanation.lower()
        assert "pool_medium" in result.explanation

    def test_gate_shadow_mode_warns_but_not_blocks(self):
        """Test shadow mode logs warnings but doesn't block."""
        judgement = Judgement(
            run_id="test123",
            workflow_version="v2",
            zone=None,
            bucket_stats={},
            regression_report=None,
            pass_fail="fail",
            explanation="Threshold exceeded",
            worst_zone_p95=18.0,
        )

        result = evaluate_gate(
            judgement,
            worst_zone_p95_threshold=15.0,
            mode="shadow",
        )

        assert not result.should_block  # Shadow mode doesn't block
        assert result.mode == "shadow"
        assert len(result.reasons) > 0  # But reasons are still recorded
        assert "shadow" in result.explanation.lower()

    def test_gate_passes_when_all_rules_pass(self):
        """Test gate passes when all rules pass."""
        bucket_stats = {"pool_medium": self.create_bucket_stats("pool_medium", p95=12.0, threshold_p95=15.0, pass_fail="pass")}

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=None,
            pass_fail="pass",
            explanation="All thresholds met",
            worst_zone_p95=12.0,
        )

        result = evaluate_gate(
            judgement,
            worst_zone_p95_threshold=15.0,
            max_regression_threshold=0.15,
            mode="enforce",
        )

        assert not result.should_block
        assert result.mode == "enforce"
        assert len(result.reasons) == 0
        assert "passed" in result.explanation.lower()

    def test_should_block_simple_api(self):
        """Test simple should_block API."""
        bucket_stats = {"pool_medium": self.create_bucket_stats("pool_medium", p95=16.0, threshold_p95=15.0, pass_fail="fail")}

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=None,
            pass_fail="fail",
            explanation="Threshold exceeded",
        )

        block, reason = should_block(judgement, mode="enforce")

        assert block
        assert "threshold" in reason.lower() or "bucket" in reason.lower()

    def test_multiple_violations_all_reported(self):
        """Test that multiple violations are all reported."""
        bucket_stats = {"pool_medium": self.create_bucket_stats("pool_medium", p95=16.0, threshold_p95=15.0, pass_fail="fail")}

        regression_report = RegressionReport(
            baseline_run_id="baseline",
            baseline_commit_sha="abc",
            current_run_id="current",
            current_commit_sha="def",
            bucket_regressions={"pool_medium": 0.20},
            max_regression=0.20,
            max_regression_bucket="pool_medium",
            status="fail",
            explanation="Regression detected",
        )

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=regression_report,
            pass_fail="fail",
            explanation="Multiple violations",
            worst_zone_p95=18.0,
        )

        result = evaluate_gate(
            judgement,
            worst_zone_p95_threshold=15.0,
            max_regression_threshold=0.15,
            mode="enforce",
        )

        assert result.should_block
        # Should have 3 reasons: bucket violation, worst-zone, regression
        assert len(result.reasons) == 3
