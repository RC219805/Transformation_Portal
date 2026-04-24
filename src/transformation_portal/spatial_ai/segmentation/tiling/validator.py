"""Default deterministic tiling merge validation."""

from __future__ import annotations

from typing import Any

from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig
from transformation_portal.spatial_ai.segmentation.tiling.types import TileManifest


class SeamMergeValidator:
    """Validate merged-tile seam metrics against the tiling contract."""

    def validate(
        self,
        *,
        manifest: TileManifest,
        merge_stats: dict,
        config: SegmentationTilingConfig,
    ) -> tuple[bool, dict[str, Any]]:
        del manifest
        metrics = dict(merge_stats.get("seam_metrics") or {})
        threshold = float(config.validation.seam_discontinuity_threshold)
        pair_count = int(metrics.get("merged_pair_count", 0))
        max_discontinuity = float(metrics.get("max_merged_discontinuity", 0.0))
        mean_discontinuity = float(metrics.get("mean_merged_discontinuity", 0.0))

        details: dict[str, Any] = {
            "merged_pair_count": pair_count,
            "max_merged_discontinuity": max_discontinuity,
            "mean_merged_discontinuity": mean_discontinuity,
            "threshold": threshold,
            "ok": True,
        }
        if pair_count <= 0 or max_discontinuity <= threshold:
            return True, details

        warning = "SAM2 tiling seam discontinuity " f"{max_discontinuity:.3f} exceeds threshold {threshold:.3f}"
        merge_stats.setdefault("warnings", []).append(warning)
        details["ok"] = False
        details["warning"] = warning
        return False, details
