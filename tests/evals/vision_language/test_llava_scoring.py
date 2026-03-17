"""Tests for VQA scoring utilities."""

from __future__ import annotations

import pytest

from transformation_portal.evals.vision_language.llava_schema import VQAIssue, VQAResult
from transformation_portal.evals.vision_language.llava_scoring import (
    compute_quality_gate_pass,
    recompute_summary_score,
    severity_to_numeric,
)


class TestRecomputeSummaryScore:
    """Tests for recompute_summary_score function."""

    def test_no_issues(self) -> None:
        """Should return 1.0 for result with no issues."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.8,
            issues=[],
        )
        assert recompute_summary_score(result) == 1.0

    def test_low_severity_issue(self) -> None:
        """Should subtract 0.10 for low severity issue."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.8,
            issues=[VQAIssue("test", "low", "evidence")],
        )
        assert recompute_summary_score(result) == 0.9

    def test_medium_severity_issue(self) -> None:
        """Should subtract 0.25 for medium severity issue."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.8,
            issues=[VQAIssue("test", "medium", "evidence")],
        )
        assert recompute_summary_score(result) == 0.75

    def test_high_severity_issue(self) -> None:
        """Should subtract 0.50 for high severity issue."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.8,
            issues=[VQAIssue("test", "high", "evidence")],
        )
        assert recompute_summary_score(result) == 0.5

    def test_multiple_issues(self) -> None:
        """Should accumulate penalties for multiple issues."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.8,
            issues=[
                VQAIssue("test1", "low", "e1"),
                VQAIssue("test2", "medium", "e2"),
            ],
        )
        # 1.0 - 0.10 - 0.25 = 0.65
        assert recompute_summary_score(result) == 0.65

    def test_clamp_to_zero(self) -> None:
        """Should clamp score to 0.0 when penalties exceed 1.0."""
        result = VQAResult(
            passes_basic_quality=False,
            summary_score=0.0,
            issues=[
                VQAIssue("t1", "high", "e1"),
                VQAIssue("t2", "high", "e2"),
                VQAIssue("t3", "high", "e3"),
            ],
        )
        assert recompute_summary_score(result) == 0.0

    def test_unknown_severity_defaults_to_high(self) -> None:
        """Should use 0.50 penalty for unknown severity."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.8,
            issues=[VQAIssue("test", "critical", "evidence")],  # Unknown severity
        )
        assert recompute_summary_score(result) == 0.5


class TestComputeQualityGatePass:
    """Tests for compute_quality_gate_pass function."""

    def test_passes_with_good_result(self) -> None:
        """Should pass for result with good score and no issues."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.9,
            issues=[],
        )
        assert compute_quality_gate_pass(result) is True

    def test_fails_low_score(self) -> None:
        """Should fail when score is below threshold."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.5,
            issues=[],
        )
        assert compute_quality_gate_pass(result, min_score=0.75) is False

    def test_fails_high_severity_issues(self) -> None:
        """Should fail when high severity issues exceed threshold."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.9,
            issues=[VQAIssue("test", "high", "evidence")],
        )
        assert compute_quality_gate_pass(result, max_high_severity_issues=0) is False

    def test_fails_medium_severity_issues(self) -> None:
        """Should fail when medium severity issues exceed threshold."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.9,
            issues=[
                VQAIssue("t1", "medium", "e1"),
                VQAIssue("t2", "medium", "e2"),
                VQAIssue("t3", "medium", "e3"),
            ],
        )
        assert compute_quality_gate_pass(result, max_medium_severity_issues=2) is False

    def test_custom_thresholds(self) -> None:
        """Should use custom thresholds when provided."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.6,
            issues=[VQAIssue("test", "high", "e")],
        )
        # With relaxed thresholds, should pass
        assert (
            compute_quality_gate_pass(
                result,
                min_score=0.5,
                max_high_severity_issues=1,
            )
            is True
        )


class TestSeverityToNumeric:
    """Tests for severity_to_numeric function."""

    def test_low_severity(self) -> None:
        """Should return 0.25 for low severity."""
        assert severity_to_numeric("low") == 0.25

    def test_medium_severity(self) -> None:
        """Should return 0.50 for medium severity."""
        assert severity_to_numeric("medium") == 0.50

    def test_high_severity(self) -> None:
        """Should return 1.00 for high severity."""
        assert severity_to_numeric("high") == 1.00

    def test_case_insensitive(self) -> None:
        """Should handle different cases."""
        assert severity_to_numeric("LOW") == 0.25
        assert severity_to_numeric("Medium") == 0.50
        assert severity_to_numeric("HIGH") == 1.00

    def test_unknown_defaults_to_medium(self) -> None:
        """Should default to 0.50 for unknown severity."""
        assert severity_to_numeric("critical") == 0.50
        assert severity_to_numeric("unknown") == 0.50
