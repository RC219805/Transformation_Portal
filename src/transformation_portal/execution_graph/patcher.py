"""DAG pipeline patcher for self-healing system.

This module provides utilities to apply fix suggestions to pipeline
configurations, generating patched versions that can be re-executed.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from transformation_portal.evals.self_healing import FixSuggestion

logger = logging.getLogger(__name__)


class PatchError(RuntimeError):
    """Error applying patch to pipeline."""


@dataclass(frozen=True)
class PatchResult:
    """Result of applying a patch."""

    success: bool
    node_id: str
    action: str
    changes: dict[str, Any]
    error: str | None = None


@dataclass
class PatchSet:
    """Collection of patches applied to a pipeline."""

    patches: list[PatchResult] = field(default_factory=list)
    original_hash: str = ""

    @property
    def successful(self) -> list[PatchResult]:
        """Get successful patches."""
        return [p for p in self.patches if p.success]

    @property
    def failed(self) -> list[PatchResult]:
        """Get failed patches."""
        return [p for p in self.patches if not p.success]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patches": [
                {
                    "success": p.success,
                    "node_id": p.node_id,
                    "action": p.action,
                    "changes": p.changes,
                    "error": p.error,
                }
                for p in self.patches
            ],
            "original_hash": self.original_hash,
            "total": len(self.patches),
            "successful": len(self.successful),
            "failed": len(self.failed),
        }


def _apply_increase_mask_coverage(node: dict, params: dict) -> dict[str, Any]:
    """Apply mask coverage increase."""
    config = node.setdefault("config", {})
    changes = {}

    if "threshold" in params:
        old = config.get("threshold")
        config["threshold"] = params["threshold"]
        changes["threshold"] = {"old": old, "new": params["threshold"]}

    if "iou_threshold" in params:
        old = config.get("iou_threshold")
        config["iou_threshold"] = params["iou_threshold"]
        changes["iou_threshold"] = {"old": old, "new": params["iou_threshold"]}

    return changes


def _apply_increase_iterations(node: dict, params: dict) -> dict[str, Any]:
    """Apply iteration increase."""
    config = node.setdefault("config", {})
    changes = {}

    if "steps" in params:
        old = config.get("steps")
        config["steps"] = params["steps"]
        changes["steps"] = {"old": old, "new": params["steps"]}

    if "refine" in params:
        old = config.get("refine")
        config["refine"] = params["refine"]
        changes["refine"] = {"old": old, "new": params["refine"]}

    return changes


def _apply_expand_prompt_set(node: dict, params: dict) -> dict[str, Any]:
    """Apply prompt set expansion."""
    config = node.setdefault("config", {})
    changes = {}

    if "include_negative" in params:
        old = config.get("include_negative")
        config["include_negative"] = params["include_negative"]
        changes["include_negative"] = {"old": old, "new": params["include_negative"]}

    return changes


def _apply_enable_seam_blending(node: dict, params: dict) -> dict[str, Any]:
    """Apply seam blending enable."""
    config = node.setdefault("config", {})
    changes = {}

    config["seam_blending"] = True
    changes["seam_blending"] = {"old": config.get("seam_blending"), "new": True}

    if "blend_radius" in params:
        old = config.get("blend_radius")
        config["blend_radius"] = params["blend_radius"]
        changes["blend_radius"] = {"old": old, "new": params["blend_radius"]}

    return changes


def _apply_apply_denoising(node: dict, params: dict) -> dict[str, Any]:
    """Apply denoising enable."""
    config = node.setdefault("config", {})
    changes = {}

    config["denoising"] = True
    changes["denoising"] = {"old": config.get("denoising"), "new": True}

    if "strength" in params:
        old = config.get("denoise_strength")
        config["denoise_strength"] = params["strength"]
        changes["denoise_strength"] = {"old": old, "new": params["strength"]}

    return changes


def _apply_adjust_roughness_prior(node: dict, params: dict) -> dict[str, Any]:
    """Apply roughness prior adjustment."""
    config = node.setdefault("config", {})
    changes = {}

    if "bias" in params:
        old = config.get("roughness_bias", 0.0)
        new_bias = old + params["bias"]
        config["roughness_bias"] = new_bias
        changes["roughness_bias"] = {"old": old, "new": new_bias}

    return changes


def _apply_adjust_metalness_prior(node: dict, params: dict) -> dict[str, Any]:
    """Apply metalness prior adjustment."""
    config = node.setdefault("config", {})
    changes = {}

    if "bias" in params:
        old = config.get("metalness_bias", 0.0)
        new_bias = old + params["bias"]
        config["metalness_bias"] = new_bias
        changes["metalness_bias"] = {"old": old, "new": new_bias}

    return changes


def _apply_adjust_tone_curve(node: dict, params: dict) -> dict[str, Any]:
    """Apply tone curve adjustment."""
    config = node.setdefault("config", {})
    changes = {}

    if "contrast" in params:
        old = config.get("contrast", 1.0)
        config["contrast"] = params["contrast"]
        changes["contrast"] = {"old": old, "new": params["contrast"]}

    return changes


def _apply_increase_resolution(node: dict, params: dict) -> dict[str, Any]:
    """Apply resolution increase."""
    config = node.setdefault("config", {})
    changes = {}

    if "scale" in params:
        old = config.get("resolution_scale", 1.0)
        config["resolution_scale"] = params["scale"]
        changes["resolution_scale"] = {"old": old, "new": params["scale"]}

    return changes


def _apply_increase_mesh_resolution(node: dict, params: dict) -> dict[str, Any]:
    """Apply mesh resolution increase."""
    config = node.setdefault("config", {})
    changes = {}

    if "subdivisions" in params:
        old = config.get("subdivisions", 0)
        config["subdivisions"] = params["subdivisions"]
        changes["subdivisions"] = {"old": old, "new": params["subdivisions"]}

    return changes


def _apply_flag_for_review(node: dict, params: dict) -> dict[str, Any]:
    """Apply review flag."""
    config = node.setdefault("config", {})
    changes = {}

    config["needs_review"] = True
    config["review_reason"] = params.get("reason", "Flagged by self-healing")
    changes["needs_review"] = {"old": False, "new": True}
    changes["review_reason"] = {"old": None, "new": config["review_reason"]}

    return changes


def _apply_adjust_texture_quality(node: dict, params: dict) -> dict[str, Any]:
    """Apply texture quality adjustment."""
    config = node.setdefault("config", {})
    changes = {}

    if "detail_level" in params:
        old = config.get("detail_level")
        config["detail_level"] = params["detail_level"]
        changes["detail_level"] = {"old": old, "new": params["detail_level"]}

    return changes


# Action handlers registry
_ACTION_HANDLERS = {
    "increase_mask_coverage": _apply_increase_mask_coverage,
    "increase_iterations": _apply_increase_iterations,
    "expand_prompt_set": _apply_expand_prompt_set,
    "enable_seam_blending": _apply_enable_seam_blending,
    "apply_denoising": _apply_apply_denoising,
    "adjust_roughness_prior": _apply_adjust_roughness_prior,
    "adjust_metalness_prior": _apply_adjust_metalness_prior,
    "adjust_tone_curve": _apply_adjust_tone_curve,
    "increase_resolution": _apply_increase_resolution,
    "increase_mesh_resolution": _apply_increase_mesh_resolution,
    "flag_for_review": _apply_flag_for_review,
    "adjust_texture_quality": _apply_adjust_texture_quality,
}


def apply_fix(pipeline: dict[str, Any], fix: FixSuggestion) -> dict[str, Any]:
    """Apply a single fix to a pipeline configuration.

    Args:
        pipeline: Pipeline configuration dict with "nodes" list
        fix: Fix suggestion to apply

    Returns:
        New pipeline configuration with fix applied

    Raises:
        PatchError: If fix cannot be applied
    """
    # Deep copy to avoid mutating original
    new_pipeline = copy.deepcopy(pipeline)

    # Find target node
    target_node = None
    for node in new_pipeline.get("nodes", []):
        if node.get("id") == fix.target_node:
            target_node = node
            break

    if target_node is None:
        raise PatchError(f"Target node not found: {fix.target_node}")

    # Get handler for action
    handler = _ACTION_HANDLERS.get(fix.action)
    if handler is None:
        raise PatchError(f"Unknown action: {fix.action}")

    # Apply fix
    try:
        handler(target_node, fix.params)
    except Exception as e:
        raise PatchError(f"Failed to apply {fix.action}: {e}") from e

    logger.info(
        "Applied fix: %s on %s with params %s",
        fix.action,
        fix.target_node,
        fix.params,
    )

    return new_pipeline


def apply_fixes(
    pipeline: dict[str, Any],
    fixes: list[FixSuggestion],
) -> tuple[dict[str, Any], PatchSet]:
    """Apply multiple fixes to a pipeline.

    Args:
        pipeline: Pipeline configuration
        fixes: List of fix suggestions to apply

    Returns:
        Tuple of (patched pipeline, patch set with results)
    """
    import hashlib
    import json

    # Compute original hash for tracking
    original_hash = hashlib.sha256(json.dumps(pipeline, sort_keys=True).encode()).hexdigest()[:16]

    patch_set = PatchSet(original_hash=original_hash)
    current = pipeline

    for fix in fixes:
        try:
            # Find handler
            handler = _ACTION_HANDLERS.get(fix.action)
            if handler is None:
                raise PatchError(f"Unknown action: {fix.action}")

            # Make deep copy
            new_pipeline = copy.deepcopy(current)

            # Find and modify node
            target_node = None
            for node in new_pipeline.get("nodes", []):
                if node.get("id") == fix.target_node:
                    target_node = node
                    break

            if target_node is None:
                raise PatchError(f"Node not found: {fix.target_node}")

            # Apply and record changes
            changes = handler(target_node, fix.params)

            patch_set.patches.append(
                PatchResult(
                    success=True,
                    node_id=fix.target_node,
                    action=fix.action,
                    changes=changes,
                )
            )

            current = new_pipeline

        except Exception as e:
            logger.error("Failed to apply fix %s: %s", fix.action, e)
            patch_set.patches.append(
                PatchResult(
                    success=False,
                    node_id=fix.target_node,
                    action=fix.action,
                    changes={},
                    error=str(e),
                )
            )

    return current, patch_set


def generate_patch_diff(
    original: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    """Generate a diff between original and patched pipelines.

    Args:
        original: Original pipeline config
        patched: Patched pipeline config

    Returns:
        Diff showing changes per node
    """
    diff: dict[str, Any] = {"nodes": {}}

    original_nodes = {n["id"]: n for n in original.get("nodes", [])}
    patched_nodes = {n["id"]: n for n in patched.get("nodes", [])}

    for node_id, patched_node in patched_nodes.items():
        original_node = original_nodes.get(node_id, {})

        original_config = original_node.get("config", {})
        patched_config = patched_node.get("config", {})

        # Find changed keys
        all_keys = set(original_config.keys()) | set(patched_config.keys())
        changes = {}

        for key in all_keys:
            old_val = original_config.get(key)
            new_val = patched_config.get(key)

            if old_val != new_val:
                changes[key] = {"old": old_val, "new": new_val}

        if changes:
            diff["nodes"][node_id] = changes

    return diff
