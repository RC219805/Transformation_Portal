"""Tests for execution_graph.patcher — pipeline fix application."""

from __future__ import annotations

import pytest

from transformation_portal.evals.self_healing import FixSuggestion
from transformation_portal.execution_graph.patcher import (
    PatchError,
    apply_fix,
    apply_fixes,
    generate_patch_diff,
)

pytestmark = pytest.mark.unit


def _fix(action: str, target: str = "sam2", params: dict | None = None) -> FixSuggestion:
    return FixSuggestion(
        type="config",
        target_node=target,
        action=action,
        params=params or {},
        confidence=0.9,
        rationale="test",
        priority=1,
        reversible=True,
    )


def _pipeline(nodes: list[dict] | None = None) -> dict:
    if nodes is None:
        nodes = [{"id": "sam2", "config": {"threshold": 0.5, "steps": 100, "seam_blending": False}}]
    return {"nodes": nodes}


class TestApplyFix:
    def test_increase_mask_coverage_updates_threshold(self):
        """increase_mask_coverage sets threshold in the node config."""
        pipeline = _pipeline()
        patched = apply_fix(pipeline, _fix("increase_mask_coverage", params={"threshold": 0.8}))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["threshold"] == pytest.approx(0.8)

    def test_original_pipeline_not_mutated(self):
        """apply_fix returns a deep copy; original is unchanged."""
        pipeline = _pipeline()
        apply_fix(pipeline, _fix("increase_mask_coverage", params={"threshold": 0.9}))
        node = next(n for n in pipeline["nodes"] if n["id"] == "sam2")
        assert node["config"]["threshold"] == pytest.approx(0.5)

    def test_unknown_node_raises_patch_error(self):
        """target_node not in pipeline raises PatchError."""
        with pytest.raises(PatchError, match="not found"):
            apply_fix(_pipeline(), _fix("apply_denoising", target="missing_node"))

    def test_unknown_action_raises_patch_error(self):
        """An unregistered action raises PatchError."""
        with pytest.raises(PatchError, match="Unknown action"):
            apply_fix(_pipeline(), _fix("explode_everything"))

    def test_increase_iterations_updates_steps(self):
        """increase_iterations sets the steps field."""
        patched = apply_fix(_pipeline(), _fix("increase_iterations", params={"steps": 200}))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["steps"] == 200

    def test_enable_seam_blending_sets_flag(self):
        """enable_seam_blending sets seam_blending=True."""
        patched = apply_fix(_pipeline(), _fix("enable_seam_blending"))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["seam_blending"] is True

    def test_apply_denoising_sets_flag(self):
        """apply_denoising sets denoising=True."""
        patched = apply_fix(_pipeline(), _fix("apply_denoising"))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["denoising"] is True

    def test_adjust_roughness_prior_accumulates(self):
        """adjust_roughness_prior adds bias to existing roughness_bias."""
        pipeline = _pipeline(nodes=[{"id": "sam2", "config": {"roughness_bias": 0.1}}])
        patched = apply_fix(pipeline, _fix("adjust_roughness_prior", params={"bias": 0.2}))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["roughness_bias"] == pytest.approx(0.3)

    def test_adjust_metalness_prior_accumulates(self):
        """adjust_metalness_prior adds bias to existing metalness_bias."""
        pipeline = _pipeline(nodes=[{"id": "sam2", "config": {"metalness_bias": 0.0}}])
        patched = apply_fix(pipeline, _fix("adjust_metalness_prior", params={"bias": 0.15}))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["metalness_bias"] == pytest.approx(0.15)

    def test_adjust_tone_curve_updates_contrast(self):
        """adjust_tone_curve sets the contrast field."""
        patched = apply_fix(_pipeline(), _fix("adjust_tone_curve", params={"contrast": 1.5}))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["contrast"] == pytest.approx(1.5)

    def test_increase_resolution_updates_scale(self):
        """increase_resolution sets resolution_scale."""
        patched = apply_fix(_pipeline(), _fix("increase_resolution", params={"scale": 2.0}))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["resolution_scale"] == pytest.approx(2.0)

    def test_flag_for_review_sets_needs_review(self):
        """flag_for_review sets needs_review=True."""
        patched = apply_fix(_pipeline(), _fix("flag_for_review"))
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        assert node["config"]["needs_review"] is True


class TestApplyFixes:
    def test_multiple_fixes_applied_in_order(self):
        """Second fix sees the result of the first fix."""
        pipeline = _pipeline(nodes=[{"id": "sam2", "config": {"steps": 100}}])
        fixes = [
            _fix("increase_iterations", params={"steps": 200}),
            _fix("increase_iterations", params={"steps": 300}),
        ]
        patched, patch_set = apply_fixes(pipeline, fixes)
        node = next(n for n in patched["nodes"] if n["id"] == "sam2")
        # Second fix overwrites: steps == 300
        assert node["config"]["steps"] == 300

    def test_failed_fix_does_not_stop_others(self):
        """A fix with a bad target node is recorded as failed; others still apply."""
        pipeline = _pipeline()
        fixes = [
            _fix("apply_denoising", target="missing"),  # will fail
            _fix("enable_seam_blending"),  # should succeed
        ]
        patched, patch_set = apply_fixes(pipeline, fixes)
        assert len(patch_set.failed) == 1
        assert len(patch_set.successful) == 1

    def test_patch_set_counts_correct(self):
        """successful + failed == total patches."""
        pipeline = _pipeline()
        fixes = [_fix("apply_denoising"), _fix("unknown_action")]
        _, patch_set = apply_fixes(pipeline, fixes)
        assert len(patch_set.patches) == 2
        assert len(patch_set.successful) + len(patch_set.failed) == 2

    def test_original_hash_non_empty(self):
        """PatchSet.original_hash is non-empty."""
        _, patch_set = apply_fixes(_pipeline(), [])
        assert patch_set.original_hash != ""

    def test_empty_fixes_returns_original(self):
        """No fixes → returned pipeline equals the original."""
        pipeline = _pipeline()
        patched, _ = apply_fixes(pipeline, [])
        assert patched == pipeline


class TestGeneratePatchDiff:
    def test_changed_key_appears_in_diff(self):
        """A modified config key appears in the diff with old/new values."""
        original = _pipeline(nodes=[{"id": "sam2", "config": {"threshold": 0.5}}])
        patched = _pipeline(nodes=[{"id": "sam2", "config": {"threshold": 0.8}}])
        diff = generate_patch_diff(original, patched)
        assert "sam2" in diff["nodes"]
        assert "threshold" in diff["nodes"]["sam2"]
        assert diff["nodes"]["sam2"]["threshold"]["old"] == pytest.approx(0.5)
        assert diff["nodes"]["sam2"]["threshold"]["new"] == pytest.approx(0.8)

    def test_unchanged_key_not_in_diff(self):
        """Keys with the same value are not included in the diff."""
        node_cfg = {"threshold": 0.5, "steps": 100}
        original = _pipeline(nodes=[{"id": "sam2", "config": dict(node_cfg)}])
        patched = _pipeline(nodes=[{"id": "sam2", "config": dict(node_cfg)}])
        diff = generate_patch_diff(original, patched)
        assert "sam2" not in diff["nodes"]

    def test_identical_pipelines_empty_diff(self):
        """Identical pipelines produce an empty nodes diff."""
        p = _pipeline()
        diff = generate_patch_diff(p, p)
        assert diff["nodes"] == {}

    def test_new_key_in_patched_appears_in_diff(self):
        """A key present only in the patched node is included."""
        original = _pipeline(nodes=[{"id": "sam2", "config": {}}])
        patched = _pipeline(nodes=[{"id": "sam2", "config": {"denoising": True}}])
        diff = generate_patch_diff(original, patched)
        assert "denoising" in diff["nodes"]["sam2"]
