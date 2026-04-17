"""Tests for evals/self_healing.py module (Phase 5 coverage).

Tests for:
- FixSuggestion dataclass
- FixSuggestionSet collection
- Fix suggestion rules
- Metric-based suggestions

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

import pytest

from transformation_portal.evals.self_healing import (
    FixSuggestion,
    FixSuggestionSet,
    suggest_fixes,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


class TestFixSuggestion:
    """Test FixSuggestion dataclass."""

    def test_basic_creation(self):
        """Test basic fix suggestion creation."""
        fix = FixSuggestion(
            type="segmentation",
            target_node="sam2",
            action="increase_mask_coverage",
            params={"threshold": 0.3},
            confidence=0.8,
            rationale="Missing region detected",
        )

        assert fix.type == "segmentation"
        assert fix.target_node == "sam2"
        assert fix.action == "increase_mask_coverage"
        assert fix.params["threshold"] == 0.3
        assert fix.confidence == 0.8
        assert fix.priority == 0  # Default
        assert fix.reversible is True  # Default

    def test_with_priority(self):
        """Test fix suggestion with priority."""
        fix = FixSuggestion(
            type="reconstruction",
            target_node="nvdiffrec",
            action="increase_iterations",
            params={"steps": 500},
            confidence=0.75,
            rationale="Artifact detected",
            priority=9,
        )

        assert fix.priority == 9

    def test_non_reversible(self):
        """Test non-reversible fix suggestion."""
        fix = FixSuggestion(
            type="destructive",
            target_node="optimizer",
            action="reset_weights",
            params={},
            confidence=0.9,
            rationale="Diverged training",
            reversible=False,
        )

        assert fix.reversible is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        fix = FixSuggestion(
            type="material",
            target_node="material_backend",
            action="adjust_roughness",
            params={"bias": 0.1},
            confidence=0.7,
            rationale="Texture issue",
            priority=5,
            reversible=True,
        )

        d = fix.to_dict()

        assert d["type"] == "material"
        assert d["target_node"] == "material_backend"
        assert d["action"] == "adjust_roughness"
        assert d["params"]["bias"] == 0.1
        assert d["confidence"] == 0.7
        assert d["priority"] == 5
        assert d["reversible"] is True

    def test_frozen_dataclass(self):
        """Test that FixSuggestion is frozen (immutable)."""
        fix = FixSuggestion(
            type="test",
            target_node="test",
            action="test",
            params={},
            confidence=0.5,
            rationale="test",
        )

        with pytest.raises(AttributeError):
            fix.confidence = 0.9


class TestFixSuggestionSet:
    """Test FixSuggestionSet collection."""

    def test_empty_set(self):
        """Test empty suggestion set."""
        fss = FixSuggestionSet()

        assert len(fss.suggestions) == 0
        assert fss.source_diff_summary == ""
        assert fss.source_metrics == {}

    def test_add_suggestion(self):
        """Test adding suggestions."""
        fss = FixSuggestionSet()

        fix = FixSuggestion(
            type="test",
            target_node="node1",
            action="action1",
            params={},
            confidence=0.8,
            rationale="Test",
        )

        fss.add(fix)

        assert len(fss.suggestions) == 1
        assert fss.suggestions[0] == fix

    def test_high_confidence_filter(self):
        """Test high confidence filter."""
        fss = FixSuggestionSet()

        fix_high = FixSuggestion(
            type="test",
            target_node="node1",
            action="high_conf",
            params={},
            confidence=0.85,
            rationale="High confidence",
        )

        fix_low = FixSuggestion(
            type="test",
            target_node="node2",
            action="low_conf",
            params={},
            confidence=0.5,
            rationale="Low confidence",
        )

        fss.add(fix_high)
        fss.add(fix_low)

        high_conf = fss.high_confidence
        assert len(high_conf) == 1
        assert high_conf[0].action == "high_conf"

    def test_by_priority(self):
        """Test sorting by priority."""
        fss = FixSuggestionSet()

        fix_low = FixSuggestion(
            type="test",
            target_node="node1",
            action="low_priority",
            params={},
            confidence=0.8,
            rationale="Low priority",
            priority=3,
        )

        fix_high = FixSuggestion(
            type="test",
            target_node="node2",
            action="high_priority",
            params={},
            confidence=0.8,
            rationale="High priority",
            priority=10,
        )

        fss.add(fix_low)
        fss.add(fix_high)

        by_priority = fss.by_priority
        assert by_priority[0].priority == 10  # Highest first
        assert by_priority[1].priority == 3

    def test_by_node(self):
        """Test grouping by target node."""
        fss = FixSuggestionSet()

        fix1 = FixSuggestion(
            type="test",
            target_node="sam2",
            action="action1",
            params={},
            confidence=0.8,
            rationale="Test 1",
        )

        fix2 = FixSuggestion(
            type="test",
            target_node="sam2",
            action="action2",
            params={},
            confidence=0.7,
            rationale="Test 2",
        )

        fix3 = FixSuggestion(
            type="test",
            target_node="nvdiffrec",
            action="action3",
            params={},
            confidence=0.9,
            rationale="Test 3",
        )

        fss.add(fix1)
        fss.add(fix2)
        fss.add(fix3)

        by_node = fss.by_node
        assert len(by_node["sam2"]) == 2
        assert len(by_node["nvdiffrec"]) == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        fss = FixSuggestionSet(
            source_diff_summary="Test diff",
            source_metrics={"psnr": 25.0},
        )

        fix = FixSuggestion(
            type="test",
            target_node="node1",
            action="action1",
            params={},
            confidence=0.85,
            rationale="Test",
        )
        fss.add(fix)

        d = fss.to_dict()

        assert d["count"] == 1
        assert d["high_confidence_count"] == 1
        assert d["source_diff_summary"] == "Test diff"
        assert d["source_metrics"]["psnr"] == 25.0


class TestSuggestFixes:
    """Test suggest_fixes function."""

    def test_empty_diff(self):
        """Test with empty semantic diff."""
        diff = {"changes": []}
        result = suggest_fixes(diff)

        assert len(result.suggestions) == 0

    def test_missing_region_fixes(self):
        """Test fixes for missing regions."""
        diff = {
            "changes": [
                {
                    "type": "missing",
                    "severity": "high",
                    "description": "Region not captured",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should suggest segmentation improvements
        actions = [s.action for s in result.suggestions]
        assert "increase_mask_coverage" in actions or "expand_prompt_set" in actions

    def test_artifact_fixes(self):
        """Test fixes for artifacts."""
        diff = {
            "changes": [
                {
                    "type": "artifact",
                    "severity": "high",
                    "description": "Noise pattern visible",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should suggest reconstruction or postprocess improvements
        types = [s.type for s in result.suggestions]
        assert "reconstruction" in types or "postprocess" in types

    def test_texture_fixes(self):
        """Test fixes for texture issues."""
        diff = {
            "changes": [
                {
                    "type": "texture",
                    "severity": "medium",
                    "description": "Color tone mismatch",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should suggest material or color adjustments
        actions = [s.action for s in result.suggestions]
        assert any("roughness" in a or "tone" in a or "texture" in a for a in actions)

    def test_geometry_fixes(self):
        """Test fixes for geometry issues."""
        diff = {
            "changes": [
                {
                    "type": "geometry",
                    "severity": "high",
                    "description": "Structural deformation",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should suggest depth or reconstruction improvements
        types = [s.type for s in result.suggestions]
        assert "depth" in types or "reconstruction" in types

    def test_semantic_fixes(self):
        """Test fixes for semantic issues."""
        diff = {
            "changes": [
                {
                    "type": "semantic",
                    "severity": "high",
                    "description": "Object identity unclear",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should suggest review flag for semantic issues
        actions = [s.action for s in result.suggestions]
        assert "flag_for_review" in actions

    def test_metric_based_suggestions_low_psnr(self):
        """Test metric-based suggestions for low PSNR."""
        diff = {"changes": []}
        metrics = {"psnr": 20.0}  # Low PSNR

        result = suggest_fixes(diff, metrics)

        # Should suggest quality improvements
        types = [s.type for s in result.suggestions]
        assert "quality" in types

    def test_metric_based_suggestions_low_ssim(self):
        """Test metric-based suggestions for low SSIM."""
        diff = {"changes": []}
        metrics = {"ssim": 0.75}  # Low SSIM

        result = suggest_fixes(diff, metrics)

        # Should suggest structural improvements
        types = [s.type for s in result.suggestions]
        assert "structure" in types

    def test_metric_based_suggestions_high_lpips(self):
        """Test metric-based suggestions for high LPIPS."""
        diff = {"changes": []}
        metrics = {"lpips": 0.4}  # High LPIPS

        result = suggest_fixes(diff, metrics)

        # Should suggest perceptual improvements
        types = [s.type for s in result.suggestions]
        assert "perceptual" in types

    def test_seam_artifact_specific_fix(self):
        """Test specific fix for seam artifacts."""
        diff = {
            "changes": [
                {
                    "type": "artifact",
                    "severity": "medium",
                    "description": "Visible seam at edge",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should suggest seam blending
        actions = [s.action for s in result.suggestions]
        assert any("seam" in a for a in actions)

    def test_noise_artifact_specific_fix(self):
        """Test specific fix for noise artifacts."""
        diff = {
            "changes": [
                {
                    "type": "artifact",
                    "severity": "medium",
                    "description": "Grain noise visible",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should suggest denoising
        actions = [s.action for s in result.suggestions]
        assert any("denois" in a for a in actions)

    def test_metallic_material_fix(self):
        """Test specific fix for metallic material issues."""
        diff = {
            "changes": [
                {
                    "type": "texture",
                    "severity": "medium",
                    "description": "Metallic reflective surface incorrect",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should suggest metalness adjustment
        actions = [s.action for s in result.suggestions]
        assert any("metal" in a for a in actions)

    def test_multiple_changes(self):
        """Test with multiple change types."""
        diff = {
            "changes": [
                {"type": "missing", "severity": "medium", "description": "Region A"},
                {"type": "artifact", "severity": "high", "description": "Noise"},
                {"type": "texture", "severity": "low", "description": "Color"},
            ]
        }

        result = suggest_fixes(diff)

        # Should have suggestions for all types
        assert len(result.suggestions) >= 3

    def test_unknown_change_type_ignored(self):
        """Test that unknown change types are ignored."""
        diff = {
            "changes": [
                {
                    "type": "unknown_type",
                    "severity": "high",
                    "description": "Unknown issue",
                }
            ]
        }

        result = suggest_fixes(diff)

        # Should not crash, just ignore unknown type
        assert isinstance(result, FixSuggestionSet)

    def test_source_metrics_preserved(self):
        """Test that source metrics are preserved in result."""
        diff = {"changes": []}
        metrics = {"psnr": 30.0, "ssim": 0.95}

        result = suggest_fixes(diff, metrics)

        assert result.source_metrics == metrics

    def test_source_diff_summary_preserved(self):
        """Test that source diff summary is preserved."""
        diff = {"summary": "Test summary", "changes": []}

        result = suggest_fixes(diff)

        assert result.source_diff_summary == "Test summary"

    def test_priority_ordering(self):
        """Test that suggestions have appropriate priorities."""
        diff = {
            "changes": [
                {"type": "semantic", "severity": "high", "description": "Critical issue"},
            ]
        }

        result = suggest_fixes(diff)

        # Review flags should have high priority
        review_fixes = [s for s in result.suggestions if s.action == "flag_for_review"]
        if review_fixes:
            assert review_fixes[0].priority >= 8
