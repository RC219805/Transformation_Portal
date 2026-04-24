"""Materials V3 Response Planner (PR-4C).

Separates decision logic from execution.
Computes objective edge signals to gate ML refinement.
"""

from typing import Any, Dict

import numpy as np
import scipy.ndimage

from .pixel_ops_decider import decide_pixel_ops
from .pixel_ops_registry import OP_REGISTRY


def compute_edge_signals(
    mask_np: np.ndarray,
    rgb_np: np.ndarray,
    grad_mag: np.ndarray | None = None,
) -> Dict[str, float]:
    """Computes objective boundary metrics using image gradients."""
    if mask_np is None or mask_np.sum() == 0:
        return {"boundary_pixels": 0, "edge_alignment": 0.0}

    # 1. Extract Boundary (Morphological Edge approx 3px wide)
    binary_mask = (mask_np > 0.5).astype(int)
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    dilated = scipy.ndimage.binary_dilation(
        binary_mask,
        structure=struct,
        iterations=1,
    )
    eroded = scipy.ndimage.binary_erosion(
        binary_mask,
        structure=struct,
        iterations=1,
    )
    boundary_mask = (dilated ^ eroded).astype(bool)

    boundary_pixels_count = int(np.sum(boundary_mask))
    if boundary_pixels_count == 0:
        return {"boundary_pixels": 0, "edge_alignment": 0.0}

    # 2. Use precomputed image gradients when available; otherwise compute
    # them for backward-compatible direct helper calls.
    if grad_mag is None:
        grad_mag = _normalized_gradient_magnitude(rgb_np)

    # 3. Compute Alignment (Mean gradient magnitude at boundary)
    alignment_score = float(np.mean(grad_mag[boundary_mask]))

    return {
        "boundary_pixels": boundary_pixels_count,
        "edge_alignment": round(alignment_score, 4),
    }


def _normalized_gradient_magnitude(rgb_np: np.ndarray) -> np.ndarray:
    if rgb_np.ndim == 3 and rgb_np.shape[2] == 3:
        gray = np.dot(rgb_np[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = rgb_np

    sx = scipy.ndimage.sobel(gray, axis=0)
    sy = scipy.ndimage.sobel(gray, axis=1)
    grad_mag = np.hypot(sx, sy)

    max_grad = np.max(grad_mag)
    if max_grad > 0:
        grad_mag /= max_grad
    return grad_mag


def _decide_refinement(
    material_key: str,
    stats: Dict,
    edge_signals: Dict,
    config: Any,
) -> Dict[str, Any]:
    """Decision Block A: EfficientSAM Refinement Gate."""
    canary_set = {"glass", "foliage", "water"}

    # Eligibility
    is_canary = material_key in canary_set
    sufficient_coverage = stats["coverage_px"] >= config.min_coverage_px
    sufficient_conf = stats["mean_conf"] >= config.min_mean_conf

    # PR-4C Safety Gates
    sufficient_boundary = edge_signals["boundary_pixels"] >= 250
    has_edge_support = edge_signals["edge_alignment"] >= 0.10

    eligible = is_canary and sufficient_coverage and sufficient_conf and sufficient_boundary and has_edge_support

    # Recommendation
    ambiguity_threshold = 0.90
    should_refine = eligible and (stats["mean_conf"] < ambiguity_threshold)

    reason = "eligible_candidate"
    if not is_canary:
        reason = "not_in_canary_set"
    elif not sufficient_coverage:
        reason = "insufficient_coverage"
    elif not sufficient_conf:
        reason = "insufficient_confidence"
    elif not sufficient_boundary:
        reason = "insufficient_boundary_pixels"
    elif not has_edge_support:
        reason = "poor_edge_alignment"
    elif stats["mean_conf"] >= ambiguity_threshold:
        reason = "confidence_already_high"

    return {
        "should_refine_edges": should_refine,
        "eligible": eligible,
        "reason": reason,
        "strategy": "canary",
    }


def _decide_pixel_ops(
    material_key: str,
    stats: Dict,
    config: Any,
) -> Dict[str, Any]:
    """Decision Block B: Pixel Ops Gate."""
    return decide_pixel_ops(material_key, stats, config, registry=OP_REGISTRY)


def generate_response_plan(
    per_class_stats: Dict[str, Any],
    rgb_image: np.ndarray,
    config: Any,
) -> Dict[str, Any]:
    """Generates Schema v3.1 Response Plan."""
    plan: Dict[str, Any] = {
        "version": "v3.1",
        "config_summary": {
            "strategy": str(config.refinement_strategy),
            "min_coverage": config.min_coverage_px,
        },
        "per_class": {},
        "summary": {
            "present_classes": [],
            "eligible_for_pixel_ops": [],
            "eligible_for_refinement": [],
            "skipped_reasons_histogram": {},
        },
    }

    histogram: Dict[str, int] = {}
    shared_grad_mag = _normalized_gradient_magnitude(rgb_image) if per_class_stats else None
    for mat_key, stats in per_class_stats.items():
        if not stats.get("present", False):
            continue
        plan["summary"]["present_classes"].append(mat_key)

        edge_signals = {"boundary_pixels": 0, "edge_alignment": 0.0}
        if "mask" in stats:
            edge_signals = compute_edge_signals(stats["mask"], rgb_image, shared_grad_mag)

        refinement = _decide_refinement(mat_key, stats, edge_signals, config)
        pixel_ops = _decide_pixel_ops(mat_key, stats, config)

        if refinement["eligible"]:
            plan["summary"]["eligible_for_refinement"].append(mat_key)
        if pixel_ops["eligible"]:
            plan["summary"]["eligible_for_pixel_ops"].append(mat_key)

        r_reason = pixel_ops["reason"]
        histogram[r_reason] = histogram.get(r_reason, 0) + 1

        plan_entry = {
            "present": True,
            "coverage_px": stats["coverage_px"],
            "mean_conf": stats["mean_conf"],
            "edge_conf": stats.get("edge_conf", 0.0),
            "bbox": stats.get("bbox"),
            "refinement": refinement,
            "pixel_ops": pixel_ops,
            "edge_signals": edge_signals,
        }
        if "material_confidence" in stats:
            plan_entry["material_confidence"] = stats["material_confidence"]
        plan["per_class"][mat_key] = plan_entry

    plan["summary"]["skipped_reasons_histogram"] = histogram
    return plan
