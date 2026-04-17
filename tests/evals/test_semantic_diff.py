"""Tests for evals/semantic_diff.py module (Phase 5 coverage).

Tests for:
- SemanticChange dataclass
- SemanticDiffResult dataclass
- Semantic diff analysis flow
- Severity scoring

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.evals.semantic_diff import (
    SEVERITY_WEIGHTS,
    SemanticChange,
    SemanticDiffError,
    SemanticDiffResult,
    _parse_response,
    _validate_change,
    compute_diff_penalty,
    semantic_diff,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


class TestSemanticChange:
    """Test SemanticChange dataclass."""

    def test_basic_creation(self):
        """Test basic change creation."""
        change = SemanticChange(
            type="geometry",
            severity="medium",
            description="Wall angle slightly off",
        )

        assert change.type == "geometry"
        assert change.severity == "medium"
        assert change.description == "Wall angle slightly off"
        assert change.location is None
        assert change.confidence == 1.0

    def test_with_location(self):
        """Test change with location."""
        change = SemanticChange(
            type="texture",
            severity="low",
            description="Minor color shift",
            location="upper-left quadrant",
        )

        assert change.location == "upper-left quadrant"

    def test_with_confidence(self):
        """Test change with confidence."""
        change = SemanticChange(
            type="artifact",
            severity="high",
            description="Visible noise pattern",
            confidence=0.85,
        )

        assert change.confidence == 0.85

    def test_frozen_dataclass(self):
        """Test that SemanticChange is frozen (immutable)."""
        change = SemanticChange(
            type="test",
            severity="low",
            description="test",
        )

        with pytest.raises(AttributeError):
            change.severity = "high"


class TestSemanticDiffResult:
    """Test SemanticDiffResult dataclass."""

    def test_empty_result(self):
        """Test result with no changes."""
        result = SemanticDiffResult(
            summary="No significant differences detected.",
            changes=(),
            raw_text="{}",
            structured={},
        )

        assert not result.has_changes
        assert result.high_severity_count == 0
        assert len(result.change_types) == 0

    def test_result_with_changes(self):
        """Test result with changes."""
        changes = (
            SemanticChange(type="geometry", severity="high", description="Deformation"),
            SemanticChange(type="texture", severity="low", description="Color shift"),
            SemanticChange(type="geometry", severity="medium", description="Misalignment"),
        )

        result = SemanticDiffResult(
            summary="Multiple differences found.",
            changes=changes,
            raw_text="...",
            structured={"changes": []},
            image_a_hash="abc123",
            image_b_hash="def456",
        )

        assert result.has_changes
        assert result.high_severity_count == 1
        assert result.change_types == {"geometry", "texture"}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        changes = (
            SemanticChange(
                type="artifact",
                severity="high",
                description="Noise visible",
                location="center",
                confidence=0.9,
            ),
        )

        result = SemanticDiffResult(
            summary="Artifacts detected.",
            changes=changes,
            raw_text="raw response",
            structured={"key": "value"},
            image_a_hash="abc123",
            image_b_hash="def456",
        )

        d = result.to_dict()

        assert d["summary"] == "Artifacts detected."
        assert len(d["changes"]) == 1
        assert d["changes"][0]["type"] == "artifact"
        assert d["changes"][0]["severity"] == "high"
        assert d["changes"][0]["location"] == "center"
        assert d["changes"][0]["confidence"] == 0.9
        assert d["image_a_hash"] == "abc123"
        assert d["image_b_hash"] == "def456"
        assert d["high_severity_count"] == 1
        assert d["has_changes"] is True


class TestParseResponse:
    """Test _parse_response function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        raw = '{"summary": "Test summary", "changes": []}'
        parsed = _parse_response(raw)

        assert parsed["summary"] == "Test summary"
        assert parsed["changes"] == []

    def test_parse_json_with_code_block(self):
        """Test parsing JSON wrapped in code block."""
        raw = """```json
{"summary": "Test", "changes": []}
```"""
        parsed = _parse_response(raw)

        assert parsed["summary"] == "Test"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns error dict."""
        raw = "This is not valid JSON"
        parsed = _parse_response(raw)

        assert "parse_error" in parsed
        assert parsed["changes"] == []

    def test_parse_whitespace_handling(self):
        """Test parsing handles whitespace."""
        raw = '   \n\n{"summary": "Test", "changes": []}\n\n   '
        parsed = _parse_response(raw)

        assert parsed["summary"] == "Test"


class TestValidateChange:
    """Test _validate_change function."""

    def test_validate_valid_change(self):
        """Test validating a valid change."""
        change_dict = {
            "type": "geometry",
            "severity": "high",
            "description": "Wall deformed",
        }

        result = _validate_change(change_dict)

        assert result is not None
        assert result.type == "geometry"
        assert result.severity == "high"

    def test_validate_with_location(self):
        """Test validating change with location."""
        change_dict = {
            "type": "texture",
            "severity": "low",
            "description": "Color shift",
            "location": "bottom-right corner",
        }

        result = _validate_change(change_dict)

        assert result.location == "bottom-right corner"

    def test_validate_with_confidence(self):
        """Test validating change with confidence."""
        change_dict = {
            "type": "artifact",
            "severity": "medium",
            "description": "Minor artifact",
            "confidence": 0.75,
        }

        result = _validate_change(change_dict)

        assert result.confidence == 0.75

    def test_validate_invalid_type_defaults(self):
        """Test that invalid type defaults to semantic."""
        change_dict = {
            "type": "unknown_type",
            "severity": "high",
            "description": "Some issue",
        }

        result = _validate_change(change_dict)

        assert result.type == "semantic"

    def test_validate_invalid_severity_defaults(self):
        """Test that invalid severity defaults to medium."""
        change_dict = {
            "type": "geometry",
            "severity": "critical",  # Invalid
            "description": "Issue",
        }

        result = _validate_change(change_dict)

        assert result.severity == "medium"

    def test_validate_empty_description_returns_none(self):
        """Test that empty description returns None."""
        change_dict = {
            "type": "geometry",
            "severity": "high",
            "description": "",
        }

        result = _validate_change(change_dict)

        assert result is None

    def test_validate_missing_description_returns_none(self):
        """Test that missing description returns None."""
        change_dict = {
            "type": "geometry",
            "severity": "high",
        }

        result = _validate_change(change_dict)

        assert result is None


class TestSemanticDiff:
    """Test semantic_diff function."""

    def test_semantic_diff_with_mock_backend(self, tmp_path):
        """Test semantic diff with mocked backend."""
        # Create test images
        img_a = tmp_path / "image_a.png"
        img_b = tmp_path / "image_b.png"
        img_a.write_bytes(b"fake image data a")
        img_b.write_bytes(b"fake image data b")

        # Mock backend
        mock_backend = MagicMock()
        mock_backend.generate.return_value = """
{
    "summary": "Minor differences detected",
    "changes": [
        {"type": "texture", "severity": "low", "description": "Color shift"}
    ]
}
"""

        result = semantic_diff(backend=mock_backend, image_a=img_a, image_b=img_b)

        assert result.summary == "Minor differences detected"
        assert len(result.changes) == 1
        assert result.changes[0].type == "texture"
        mock_backend.generate.assert_called_once()

    def test_semantic_diff_missing_image_a_raises(self, tmp_path):
        """Test that missing image A raises error."""
        img_b = tmp_path / "image_b.png"
        img_b.write_bytes(b"data")

        mock_backend = MagicMock()

        with pytest.raises(SemanticDiffError, match="not found"):
            semantic_diff(
                backend=mock_backend,
                image_a=tmp_path / "nonexistent.png",
                image_b=img_b,
            )

    def test_semantic_diff_missing_image_b_raises(self, tmp_path):
        """Test that missing image B raises error."""
        img_a = tmp_path / "image_a.png"
        img_a.write_bytes(b"data")

        mock_backend = MagicMock()

        with pytest.raises(SemanticDiffError, match="not found"):
            semantic_diff(
                backend=mock_backend,
                image_a=img_a,
                image_b=tmp_path / "nonexistent.png",
            )

    def test_semantic_diff_backend_error_raises(self, tmp_path):
        """Test that backend error raises SemanticDiffError."""
        img_a = tmp_path / "image_a.png"
        img_b = tmp_path / "image_b.png"
        img_a.write_bytes(b"data a")
        img_b.write_bytes(b"data b")

        mock_backend = MagicMock()
        mock_backend.generate.side_effect = RuntimeError("Backend failed")

        with pytest.raises(SemanticDiffError, match="Inference failed"):
            semantic_diff(backend=mock_backend, image_a=img_a, image_b=img_b)

    def test_semantic_diff_computes_hashes(self, tmp_path):
        """Test that image hashes are computed."""
        img_a = tmp_path / "image_a.png"
        img_b = tmp_path / "image_b.png"
        img_a.write_bytes(b"unique content a")
        img_b.write_bytes(b"unique content b")

        mock_backend = MagicMock()
        mock_backend.generate.return_value = '{"summary": "Test", "changes": []}'

        result = semantic_diff(backend=mock_backend, image_a=img_a, image_b=img_b)

        assert result.image_a_hash != ""
        assert result.image_b_hash != ""
        assert len(result.image_a_hash) == 16
        assert len(result.image_b_hash) == 16
        assert result.image_a_hash != result.image_b_hash


class TestSeverityWeights:
    """Test severity weight constants."""

    def test_weights_exist(self):
        """Test all severity weights exist."""
        assert "low" in SEVERITY_WEIGHTS
        assert "medium" in SEVERITY_WEIGHTS
        assert "high" in SEVERITY_WEIGHTS

    def test_weights_ordering(self):
        """Test severity weights are ordered correctly."""
        assert SEVERITY_WEIGHTS["low"] < SEVERITY_WEIGHTS["medium"]
        assert SEVERITY_WEIGHTS["medium"] < SEVERITY_WEIGHTS["high"]


class TestComputeDiffPenalty:
    """Test compute_diff_penalty function."""

    def test_no_changes_no_penalty(self):
        """Test no penalty for empty changes."""
        result = SemanticDiffResult(
            summary="No differences",
            changes=(),
            raw_text="",
            structured={},
        )

        penalty = compute_diff_penalty(result)
        assert penalty == 0.0

    def test_single_low_severity_change(self):
        """Test penalty for single low severity change."""
        result = SemanticDiffResult(
            summary="Minor issue",
            changes=(SemanticChange(type="texture", severity="low", description="test"),),
            raw_text="",
            structured={},
        )

        penalty = compute_diff_penalty(result)
        assert penalty == pytest.approx(0.1)  # low weight * 1.0 confidence

    def test_single_high_severity_change(self):
        """Test penalty for single high severity change."""
        result = SemanticDiffResult(
            summary="Critical issue",
            changes=(SemanticChange(type="geometry", severity="high", description="test"),),
            raw_text="",
            structured={},
        )

        penalty = compute_diff_penalty(result)
        assert penalty == pytest.approx(0.6)  # high weight * 1.0 confidence

    def test_multiple_changes_accumulate(self):
        """Test penalties accumulate for multiple changes."""
        result = SemanticDiffResult(
            summary="Multiple issues",
            changes=(
                SemanticChange(type="geometry", severity="high", description="test1"),
                SemanticChange(type="texture", severity="low", description="test2"),
            ),
            raw_text="",
            structured={},
        )

        penalty = compute_diff_penalty(result)
        # 0.6 (high) + 0.1 (low) = 0.7
        assert penalty == pytest.approx(0.7)

    def test_penalty_capped_at_one(self):
        """Test penalty is capped at 1.0."""
        result = SemanticDiffResult(
            summary="Many issues",
            changes=(
                SemanticChange(type="geometry", severity="high", description="test1"),
                SemanticChange(type="artifact", severity="high", description="test2"),
                SemanticChange(type="missing", severity="high", description="test3"),
            ),
            raw_text="",
            structured={},
        )

        penalty = compute_diff_penalty(result)
        assert penalty == 1.0

    def test_confidence_affects_penalty(self):
        """Test confidence affects penalty calculation."""
        result = SemanticDiffResult(
            summary="Uncertain issue",
            changes=(
                SemanticChange(
                    type="geometry",
                    severity="high",
                    description="test",
                    confidence=0.5,
                ),
            ),
            raw_text="",
            structured={},
        )

        penalty = compute_diff_penalty(result)
        # 0.6 (high weight) * 0.5 (confidence) = 0.3
        assert penalty == pytest.approx(0.3)


class TestSemanticDiffFromHashes:
    """Test semantic_diff_from_hashes function."""

    def test_resolves_hashes_to_paths(self, tmp_path):
        """Test that hashes are resolved to CAS paths."""
        from transformation_portal.evals.semantic_diff import semantic_diff_from_hashes

        # Create CAS structure
        cas_root = tmp_path / "cas"
        objects_dir = cas_root / "objects"
        hash_a = "a" * 64
        hash_b = "b" * 64

        (objects_dir / hash_a[:2]).mkdir(parents=True)
        (objects_dir / hash_b[:2]).mkdir(parents=True)

        file_a = objects_dir / hash_a[:2] / hash_a
        file_b = objects_dir / hash_b[:2] / hash_b
        file_a.write_bytes(b"content a")
        file_b.write_bytes(b"content b")

        mock_backend = MagicMock()
        mock_backend.generate.return_value = '{"summary": "Test", "changes": []}'

        result = semantic_diff_from_hashes(
            backend=mock_backend,
            hash_a=hash_a,
            hash_b=hash_b,
            cas_root=cas_root,
        )

        assert result.summary == "Test"
        mock_backend.generate.assert_called_once()
