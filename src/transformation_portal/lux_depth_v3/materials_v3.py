"""Materials V3 Engine.

Handles material segmentation, refinement planning, and pixel operations.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from .materials_v3_response import generate_response_plan
from .pixel_ops_executor import apply_pixel_ops

logger = logging.getLogger(__name__)


def _coerce_unit_confidence(value: Any) -> Optional[float]:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= confidence <= 1.0 and np.isfinite(confidence):
        return confidence
    return None


def _extract_material_confidences(segmentation_result: Dict[str, Any]) -> Dict[str, float]:
    confidences: Dict[str, float] = {}
    sources = [segmentation_result.get("material_confidences")]
    segmentation_metadata = segmentation_result.get("segmentation_metadata")
    if isinstance(segmentation_metadata, dict):
        sources.append(segmentation_metadata.get("material_confidences"))

    for source in sources:
        if not isinstance(source, dict):
            continue
        for material_key, confidence in source.items():
            coerced = _coerce_unit_confidence(confidence)
            if coerced is not None:
                confidences[str(material_key)] = coerced
    return confidences


def _split_mask_and_confidence(value: Any, fallback_confidence: Optional[float]) -> tuple[Any, Optional[float]]:
    if isinstance(value, tuple) and len(value) == 2:
        mask, confidence = value
        coerced = _coerce_unit_confidence(confidence)
        return mask, coerced if coerced is not None else fallback_confidence
    return value, fallback_confidence


class MaterialsV3Engine:
    def __init__(self, config: Any) -> None:
        self.config = config

    def _compute_mask_stats(self, mask: np.ndarray, material_confidence: Optional[float] = None) -> Dict[str, Any]:
        """Compute basic coverage/confidence stats."""
        mask_2d = np.asarray(mask, dtype=np.float32)
        if mask_2d.ndim == 3 and mask_2d.shape[-1] == 1:
            mask_2d = mask_2d.squeeze(axis=-1)
        elif mask_2d.ndim == 3 and mask_2d.shape[0] == 1:
            mask_2d = mask_2d.squeeze(axis=0)
        total_px = mask_2d.size
        active = mask_2d > 0.5
        coverage_px = int(np.count_nonzero(active))
        mean_conf = float(mask_2d.mean())
        bbox = None
        if coverage_px > 0:
            ys, xs = np.where(active)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        stats = {
            "present": coverage_px > 0,
            "coverage_px": coverage_px,
            "coverage_ratio": coverage_px / total_px if total_px else 0.0,
            "mean_conf": mean_conf,
            "edge_conf": 0.0,  # Placeholder, would be computed by edge extraction
            "bbox": bbox,
            "mask": mask_2d,
        }
        if material_confidence is not None:
            stats["material_confidence"] = material_confidence
        return stats

    def process(
        self, image: np.ndarray, segmentation_result: Dict[str, Any], depth_map: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Main entry point."""
        # Check if Materials V3 is enabled
        if not getattr(self.config, "enable_materials_v3", False):
            return {}

        timing_ms: Dict[str, float] = {}

        # 1. Stats
        t_stats = time.perf_counter()
        raw_materials = segmentation_result.get("materials") or segmentation_result.get("material_masks") or {}
        segmentation_metadata = segmentation_result.get("segmentation_metadata")
        material_confidences = _extract_material_confidences(segmentation_result)
        materials = {}
        for mat_key, value in raw_materials.items():
            material_key = str(mat_key)
            mask, confidence = _split_mask_and_confidence(value, material_confidences.get(material_key))
            materials[mat_key] = mask
            if confidence is not None:
                material_confidences[material_key] = confidence
        segmentation_result = {"materials": materials}

        per_class_stats = {}
        for mat_key, mask in segmentation_result.get("materials", {}).items():
            stats = self._compute_mask_stats(mask, material_confidences.get(str(mat_key)))
            per_class_stats[mat_key] = stats
        timing_ms["stats"] = round((time.perf_counter() - t_stats) * 1000.0, 3)

        # 2. Plan (Schema v3.1)
        t_plan = time.perf_counter()
        response_plan = generate_response_plan(per_class_stats, image, self.config)
        timing_ms["planning"] = round((time.perf_counter() - t_plan) * 1000.0, 3)

        # Clean up
        for mat_key in per_class_stats:
            if "mask" in per_class_stats[mat_key]:
                del per_class_stats[mat_key]["mask"]

        # 3. Execution (Pixel Ops)
        t_pixel_ops = time.perf_counter()
        enhanced_image, pixel_ops = apply_pixel_ops(image, segmentation_result, response_plan, self.config)
        timing_ms["pixel_ops"] = round((time.perf_counter() - t_pixel_ops) * 1000.0, 3)
        if isinstance(pixel_ops, dict):
            existing_timings = pixel_ops.get("timing_ms")
            if isinstance(existing_timings, dict):
                timing_ms["pixel_ops_executor_total"] = existing_timings.get("total", timing_ms["pixel_ops"])

        return {
            "enhanced_image": enhanced_image,  # Modified image with pixel ops applied
            "materials_v3_response_plan": response_plan,
            "materials_v3_pixel_ops": pixel_ops,
            "materials_v3_metadata": {
                "version": "3.1",
                "segmentation_metadata": segmentation_metadata if isinstance(segmentation_metadata, dict) else None,
                "timing_ms": timing_ms,
            },
            "material_masks": segmentation_result.get("materials", {}),
        }
