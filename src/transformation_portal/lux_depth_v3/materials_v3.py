"""Materials V3 Engine.

Handles material segmentation, refinement planning, and pixel operations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from .materials_v3_response import generate_response_plan
from .pixel_ops_executor import apply_pixel_ops

logger = logging.getLogger(__name__)


class MaterialsV3Engine:
    def __init__(self, config):
        self.config = config

    def _compute_mask_stats(self, mask: np.ndarray) -> Dict[str, Any]:
        """Compute basic coverage/confidence stats."""
        total_px = mask.size
        coverage_px = np.count_nonzero(mask > 0.5)
        mean_conf = float(mask.mean())
        return {
            "present": coverage_px > 0,
            "coverage_px": coverage_px,
            "coverage_ratio": coverage_px / total_px,
            "mean_conf": mean_conf,
            "edge_conf": 0.0,  # Placeholder, would be computed by edge extraction
        }

    def process(
        self, image: np.ndarray, segmentation_result: Dict[str, Any], depth_map: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Main entry point."""
        # Check if Materials V3 is enabled (support both .enabled and .enable_materials_v3)
        is_enabled = getattr(self.config, "enabled", None)
        if is_enabled is None:
            is_enabled = getattr(self.config, "enable_materials_v3", False)

        if not is_enabled:
            return {}

        # 1. Stats
        materials = segmentation_result.get("materials") or segmentation_result.get("material_masks") or {}
        segmentation_result = {"materials": materials}

        per_class_stats = {}
        for mat_key, mask in segmentation_result.get("materials", {}).items():
            stats = self._compute_mask_stats(mask)
            per_class_stats[mat_key] = stats
            # Attach mask for edge signal computation (PR-4C)
            per_class_stats[mat_key]["mask"] = mask

        # 2. Plan (Schema v3.1)
        response_plan = generate_response_plan(per_class_stats, image, self.config)

        # Clean up
        for mat_key in per_class_stats:
            if "mask" in per_class_stats[mat_key]:
                del per_class_stats[mat_key]["mask"]

        # 3. Execution (Pixel Ops)
        _, pixel_ops = apply_pixel_ops(image, segmentation_result, response_plan, self.config)

        return {
            "materials_v3_response_plan": response_plan,
            "materials_v3_pixel_ops": pixel_ops,
            "materials_v3_metadata": {"version": "3.1"},
            "material_masks": segmentation_result.get("materials", {}),
        }
