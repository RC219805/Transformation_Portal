"""Executor for Materials V3 pixel operations."""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np

from .pixel_ops_decider import decide_pixel_ops
from .pixel_ops_registry import OP_REGISTRY


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _compute_delta_stats(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    delta = np.abs(after.astype(np.float32) - before.astype(np.float32))
    mask = np.squeeze(mask) if mask.ndim == 3 else mask
    mask_bool = mask > 0.5
    inside = float(delta[mask_bool].mean()) if mask_bool.any() else 0.0
    outside = float(delta[~mask_bool].mean()) if (~mask_bool).any() else 0.0
    return {
        "inside_mask_mean_abs": round(inside, 6),
        "outside_mask_mean_abs": round(outside, 6),
    }


def apply_pixel_ops(
    image: np.ndarray,
    segmentation_result: Dict[str, Any],
    response_plan: Dict[str, Any],
    config: Any,
    registry: Dict[str, Dict[str, Any]] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply pixel ops and emit telemetry (never null)."""
    registry = registry or OP_REGISTRY
    telemetry = {
        "enabled": bool(getattr(config, "apply_pixel_ops", False)),
        "applied": [],
        "blocked": [],
        "timing_ms": {},
    }

    if not telemetry["enabled"]:
        return image, telemetry

    start_total = time.perf_counter()
    output = image.copy()
    materials = segmentation_result.get("materials", {})
    plan_per_class = response_plan.get("per_class", {})

    for material_key, mask in materials.items():
        plan_entry = plan_per_class.get(material_key, {})
        if not plan_entry:
            continue

        plan_decision = plan_entry.get("pixel_ops")
        decision = plan_decision or decide_pixel_ops(material_key, plan_entry, config, registry=registry)
        ops_for_material = registry.get(material_key, {})
        recommended_ops = decision.get("recommended_ops") or list(ops_for_material.keys())
        if not decision.get("will_apply", False):
            telemetry["blocked"].append(
                {
                    "material": material_key,
                    "reason": decision.get("reason", "not_recommended"),
                    "blocked_by": decision.get("blocked_by", []),
                    "recommended_ops": recommended_ops,
                }
            )
            continue

        implemented_ops = [
            op_name
            for op_name in recommended_ops
            if (op_def := ops_for_material.get(op_name)) and op_def.implemented
        ]
        if not implemented_ops:
            telemetry["blocked"].append(
                {
                    "material": material_key,
                    "reason": "no_implementation",
                    "blocked_by": ["no_implementation"],
                    "recommended_ops": recommended_ops,
                }
            )
            continue

        bbox = _bounding_box(mask)
        if bbox is None:
            telemetry["blocked"].append(
                {
                    "material": material_key,
                    "reason": "empty_mask",
                    "blocked_by": ["empty_mask"],
                    "recommended_ops": decision["recommended_ops"],
                }
            )
            continue

        x0, y0, x1, y1 = bbox
        mask_roi = mask[y0:y1, x0:x1]
        before = output[y0:y1, x0:x1]
        after = before.copy()

        start_material = time.perf_counter()
        applied_ops = []
        working = after
        original_dtype = before.dtype
        if original_dtype == np.uint8:
            working = before.astype(np.float32) / 255.0
        for op_name in recommended_ops:
            op_def = ops_for_material.get(op_name)
            if not op_def or not op_def.implemented:
                continue
            working = op_def.op(
                working,
                mask_roi,
                {"material": material_key, "normalized": working, "scale": 1.0},
            )
            applied_ops.append(op_name)

        if original_dtype == np.uint8:
            after = np.clip(working * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            after = working.astype(original_dtype)

        output[y0:y1, x0:x1] = after
        elapsed_ms = (time.perf_counter() - start_material) * 1000.0
        delta_stats = _compute_delta_stats(before, after, mask_roi)
        telemetry["applied"].append(
            {
                "material": material_key,
                "ops": applied_ops,
                "timing_ms": round(elapsed_ms, 3),
                "delta_stats": delta_stats,
            }
        )

    telemetry["timing_ms"]["total"] = round((time.perf_counter() - start_total) * 1000.0, 3)
    return output, telemetry
