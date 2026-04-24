"""Shared decision logic for Materials V3 pixel operations."""

from __future__ import annotations

from typing import Any, Dict, List

from .pixel_ops_registry import OP_REGISTRY


def _material_confidence_threshold(material_key: str, config: Any) -> float:
    """Return the fail-closed confidence threshold for material pixel ops."""
    base_threshold = float(getattr(config, "min_mean_conf", 0.2))
    try:
        from .materials_v3_taxonomy import DEFAULT_MATERIAL_METADATA
    except ImportError:
        return base_threshold

    material_threshold = DEFAULT_MATERIAL_METADATA.get(material_key, {}).get("threshold")
    if material_threshold is None:
        return base_threshold
    return max(base_threshold, float(material_threshold))


def _pixel_ops_enabled(material_key: str, config: Any) -> tuple[bool, str]:
    if not bool(getattr(config, "apply_pixel_ops", False)):
        return False, "pixel_ops_disabled"
    flag_name = f"{material_key}_response_enabled"
    if hasattr(config, flag_name) and not bool(getattr(config, flag_name)):
        return False, f"{material_key}_response_disabled"
    return True, ""


def decide_pixel_ops(
    material_key: str,
    stats: Dict[str, Any],
    config: Any,
    registry: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Return normalized decision state for pixel ops.

    Args:
        material_key: Material identifier (e.g., "glass", "water")
        stats: Material statistics including coverage_px, mean_conf, edge_conf
        config: Configuration object with min_coverage_px threshold
        registry: Pixel operations registry (defaults to OP_REGISTRY)

    Returns:
        Decision state dict with eligible, enabled, implemented, will_apply, etc.
    """
    registry = registry or OP_REGISTRY
    ops_for_material = registry.get(material_key, {})
    recommended_ops = list(ops_for_material.keys())
    # Check if this material has any implemented operations in the registry
    implemented = bool(ops_for_material) and any(op.implemented for op in ops_for_material.values())

    eligible = False
    reason = "no_implementation"
    should_apply = False

    # If material has implemented operations in registry, evaluate eligibility
    if implemented:
        # Coverage threshold check - respect config.min_coverage_px instead of hard-coded value
        min_coverage = getattr(config, "min_coverage_px", 500)  # Default to 500 if not set
        confidence_threshold = _material_confidence_threshold(material_key, config)
        mean_conf = float(stats.get("mean_conf", 0.0))
        if stats["coverage_px"] < min_coverage:
            eligible = False
            reason = "below_coverage_threshold"
        elif mean_conf < confidence_threshold:
            eligible = False
            reason = "below_confidence_threshold"
        else:
            eligible = True
            # Material-specific recommendation logic
            if material_key == "glass":
                if mean_conf < 0.80:
                    should_apply = True
                    reason = "low_mean_confidence"
                elif stats.get("edge_conf", 1.0) < 0.55:
                    should_apply = True
                    reason = "low_edge_confidence"
                else:
                    reason = "confidence_already_high"
            else:
                # For other materials (stone, water, foliage), apply if present
                should_apply = True
                reason = "material_present_with_coverage"
    else:
        eligible = False
        reason = "no_implementation"

    enabled, disabled_reason = _pixel_ops_enabled(material_key, config)

    blocked_by: List[str] = []
    if not eligible:
        blocked_by.append(reason)
    if not enabled:
        blocked_by.append(disabled_reason)
    if not implemented:
        blocked_by.append("no_implementation")
    if eligible and implemented and enabled and not should_apply:
        blocked_by.append("not_recommended")

    will_apply = bool(enabled and eligible and implemented and should_apply)

    return {
        "eligible": eligible,
        "enabled": enabled,
        "implemented": implemented,
        "recommended_ops": recommended_ops,
        "should_apply": should_apply,
        "will_apply": will_apply,
        "blocked_by": blocked_by,
        "reason": reason,
    }
