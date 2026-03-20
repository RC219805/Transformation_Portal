"""Tests for VQA result schema and parsing."""

from __future__ import annotations

import pytest

from transformation_portal.evals.vision_language.llava_schema import (
    VQAIssue,
    VQAParseError,
    VQAResult,
    parse_vqa_result,
)


class TestVQAIssue:
    """Tests for VQAIssue dataclass."""

    def test_basic_construction(self) -> None:
        """VQAIssue should store issue_type, severity, and evidence."""
        issue = VQAIssue(
            issue_type="mask_leakage",
            severity="high",
            evidence="Visible bleeding at edges",
        )
        assert issue.issue_type == "mask_leakage"
        assert issue.severity == "high"
        assert issue.evidence == "Visible bleeding at edges"

    def test_from_dict(self) -> None:
        """VQAIssue.from_dict should parse dictionary correctly."""
        data = {
            "issue_type": "texture_seam",
            "severity": "medium",
            "evidence": "Visible seam at wall junction",
        }
        issue = VQAIssue.from_dict(data)
        assert issue.issue_type == "texture_seam"
        assert issue.severity == "medium"

    def test_from_dict_defaults(self) -> None:
        """VQAIssue.from_dict should use defaults for missing fields."""
        data = {}
        issue = VQAIssue.from_dict(data)
        assert issue.issue_type == "unknown"
        assert issue.severity == "medium"
        assert issue.evidence == ""


class TestVQAResult:
    """Tests for VQAResult dataclass."""

    def test_basic_construction(self) -> None:
        """VQAResult should store all fields correctly."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.85,
            issues=[],
            raw_text="{}",
            model_key="test_model",
        )
        assert result.passes_basic_quality is True
        assert result.summary_score == 0.85
        assert result.issues == []
        assert result.model_key == "test_model"

    def test_to_dict(self) -> None:
        """VQAResult.to_dict should serialize correctly."""
        result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.75,
            issues=[VQAIssue("test", "low", "evidence")],
            model_key="test_model",
        )
        d = result.to_dict()
        assert d["passes_basic_quality"] is True
        assert d["summary_score"] == 0.75
        assert len(d["issues"]) == 1
        assert d["issues"][0]["issue_type"] == "test"


class TestParseVqaResult:
    """Tests for parse_vqa_result function."""

    def test_valid_json(self) -> None:
        """Should parse valid JSON response correctly."""
        raw_text = '{"passes_basic_quality": true, "summary_score": 0.9, "issues": []}'
        result = parse_vqa_result("test_model", raw_text)
        assert result.passes_basic_quality is True
        assert result.summary_score == 0.9
        assert result.issues == []
        assert result.parse_error is None

    def test_json_with_issues(self) -> None:
        """Should parse JSON with issues array."""
        raw_text = """
        {
            "passes_basic_quality": false,
            "summary_score": 0.5,
            "issues": [
                {"issue_type": "mask_leak", "severity": "high", "evidence": "edge bleeding"},
                {"issue_type": "texture", "severity": "low", "evidence": "minor seam"}
            ]
        }
        """
        result = parse_vqa_result("test_model", raw_text)
        assert result.passes_basic_quality is False
        assert len(result.issues) == 2
        assert result.issues[0].severity == "high"

    def test_json_in_markdown(self) -> None:
        """Should extract JSON from markdown code fence."""
        raw_text = """Here is my assessment:

```json
{"passes_basic_quality": true, "summary_score": 0.85, "issues": []}
```

That's my evaluation.
"""
        result = parse_vqa_result("test_model", raw_text)
        assert result.passes_basic_quality is True
        assert result.summary_score == 0.85
        assert result.parse_error is None

    def test_invalid_json(self) -> None:
        """Should handle invalid JSON gracefully."""
        raw_text = "This is not valid JSON at all."
        result = parse_vqa_result("test_model", raw_text)
        assert result.passes_basic_quality is False
        assert result.summary_score == 0.0
        assert result.parse_error is not None

    def test_score_clamping(self) -> None:
        """Should clamp summary_score to [0, 1] range."""
        raw_text = '{"passes_basic_quality": true, "summary_score": 1.5, "issues": []}'
        result = parse_vqa_result("test_model", raw_text)
        assert result.summary_score == 1.0

        raw_text = '{"passes_basic_quality": true, "summary_score": -0.5, "issues": []}'
        result = parse_vqa_result("test_model", raw_text)
        assert result.summary_score == 0.0
