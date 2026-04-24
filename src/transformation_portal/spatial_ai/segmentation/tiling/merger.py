"""Default deterministic tile merger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata
from transformation_portal.spatial_ai.segmentation.metadata import make_mask_metadata, mask_bbox_xywh
from transformation_portal.spatial_ai.segmentation.tiling.config import MergeConfig
from transformation_portal.spatial_ai.segmentation.tiling.types import (
    BBox,
    GlobalSeedHints,
    TileManifest,
    TileSegmentationResult,
)


@dataclass(frozen=True)
class _PatchCandidate:
    order: int
    mask_patch: np.ndarray
    prob_patch: np.ndarray
    bbox: BBox
    area: int
    score: float
    stability_score: float
    material_label: Optional[str]
    material_confidence: Optional[float]
    tile_id: str
    tile_bbox: BBox
    tile_overlap_px: int


class BinaryUnionTileMerger:
    """Merge tile instances without expanding every candidate to full image size."""

    @staticmethod
    def _labels_conflict(left: _PatchCandidate, right: _PatchCandidate) -> bool:
        return bool(left.material_label and right.material_label and left.material_label != right.material_label)

    @staticmethod
    def _bbox_overlap_window(left_bbox: BBox, right_bbox: BBox) -> Optional[tuple[int, int, int, int]]:
        x0 = max(left_bbox.x0, right_bbox.x0)
        y0 = max(left_bbox.y0, right_bbox.y0)
        x1 = min(left_bbox.x1, right_bbox.x1)
        y1 = min(left_bbox.y1, right_bbox.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _window_patch(candidate: _PatchCandidate, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        lx0 = x0 - candidate.bbox.x0
        ly0 = y0 - candidate.bbox.y0
        lx1 = x1 - candidate.bbox.x0
        ly1 = y1 - candidate.bbox.y0
        return candidate.mask_patch[ly0:ly1, lx0:lx1]

    def _iou(self, left: _PatchCandidate, right: _PatchCandidate, overlap_window: tuple[int, int, int, int]) -> float:
        left_band = self._window_patch(left, overlap_window)
        right_band = self._window_patch(right, overlap_window)
        intersection = int(np.count_nonzero(left_band & right_band))
        if intersection <= 0:
            return 0.0
        union = int(left.area) + int(right.area) - intersection
        return float(intersection / union) if union > 0 else 0.0

    def _overlap_band_continuity(
        self,
        left: _PatchCandidate,
        right: _PatchCandidate,
        overlap_window: tuple[int, int, int, int],
    ) -> float:
        left_band = self._window_patch(left, overlap_window)
        right_band = self._window_patch(right, overlap_window)
        left_area = int(np.count_nonzero(left_band))
        right_area = int(np.count_nonzero(right_band))
        denominator = min(left_area, right_area)
        if denominator <= 0:
            return 0.0
        overlap_area = int(np.count_nonzero(left_band & right_band))
        return float(overlap_area / denominator)

    @staticmethod
    def _weighted_mean(items: Sequence[_PatchCandidate], key: str) -> float:
        total_area = sum(int(item.area) for item in items)
        if total_area <= 0:
            return 0.0
        return float(sum(float(getattr(item, key)) * int(item.area) for item in items) / total_area)

    @staticmethod
    def _weighted_confidence(items: Sequence[_PatchCandidate], label: Optional[str]) -> Optional[float]:
        if not label:
            return None
        weighted = [
            (float(item.material_confidence), int(item.area))
            for item in items
            if item.material_label == label and item.material_confidence is not None
        ]
        total_area = sum(area for _, area in weighted)
        if total_area <= 0:
            return None
        return float(sum(conf * area for conf, area in weighted) / total_area)

    @staticmethod
    def _to_probabilities(values: np.ndarray, space: str) -> np.ndarray:
        probs = values.astype(np.float32, copy=False)
        if space == "logits":
            probs = 1.0 / (1.0 + np.exp(-probs))
        return probs

    @staticmethod
    def _candidate_weight(candidate: _PatchCandidate, *, W: int, H: int, mode: str) -> np.ndarray:
        overlap = max(0, int(candidate.tile_overlap_px))
        if overlap <= 0 or mode not in {"hann", "cosine", "linear"}:
            return np.ones_like(candidate.prob_patch, dtype=np.float32)

        height, width = candidate.prob_patch.shape
        ys = np.arange(candidate.bbox.y0, candidate.bbox.y1, dtype=np.float32) - float(candidate.tile_bbox.y0)
        xs = np.arange(candidate.bbox.x0, candidate.bbox.x1, dtype=np.float32) - float(candidate.tile_bbox.x0)
        tile_w = max(1, candidate.tile_bbox.w)
        tile_h = max(1, candidate.tile_bbox.h)

        x_ratio = np.ones((width,), dtype=np.float32)
        y_ratio = np.ones((height,), dtype=np.float32)
        if candidate.tile_bbox.x0 > 0:
            x_ratio = np.minimum(x_ratio, xs / float(overlap))
        if candidate.tile_bbox.x1 < W:
            x_ratio = np.minimum(x_ratio, (float(tile_w - 1) - xs) / float(overlap))
        if candidate.tile_bbox.y0 > 0:
            y_ratio = np.minimum(y_ratio, ys / float(overlap))
        if candidate.tile_bbox.y1 < H:
            y_ratio = np.minimum(y_ratio, (float(tile_h - 1) - ys) / float(overlap))

        ratio = np.clip(np.minimum(y_ratio[:, np.newaxis], x_ratio[np.newaxis, :]), 0.0, 1.0)
        if mode == "linear":
            weight = ratio
        elif mode == "cosine":
            weight = np.sin(ratio * (np.pi / 2.0))
        else:
            weight = 0.5 - 0.5 * np.cos(np.pi * ratio)
        return weight.astype(np.float32, copy=False)

    def _materialize_candidate(
        self,
        *,
        instance: Any,
        tile_result: TileSegmentationResult,
        W: int,
        H: int,
        order: int,
    ) -> Optional[_PatchCandidate]:
        probs = self._to_probabilities(instance.soft_mask.values, instance.soft_mask.space)
        local_bbox = instance.soft_mask.bbox
        x0 = tile_result.tile_spec.bbox.x0 + local_bbox.x0
        y0 = tile_result.tile_spec.bbox.y0 + local_bbox.y0
        x1 = min(tile_result.tile_spec.bbox.x0 + local_bbox.x1, W)
        y1 = min(tile_result.tile_spec.bbox.y0 + local_bbox.y1, H)
        if x1 <= x0 or y1 <= y0:
            return None

        region = probs[: max(0, y1 - y0), : max(0, x1 - x0)]
        if region.size == 0:
            return None

        mask_patch = region > 0.5
        area = int(np.count_nonzero(mask_patch))
        if area <= 0:
            return None

        return _PatchCandidate(
            order=order,
            mask_patch=mask_patch.astype(bool, copy=False),
            prob_patch=region.astype(np.float32, copy=False),
            bbox=BBox(int(x0), int(y0), int(x1), int(y1)),
            area=area,
            score=float(np.clip(instance.score, 0.0, 1.0)),
            stability_score=float(np.clip(instance.stability_score, 0.0, 1.0)),
            material_label=instance.material_label,
            material_confidence=instance.material_confidence,
            tile_id=tile_result.tile_id,
            tile_bbox=tile_result.tile_spec.bbox,
            tile_overlap_px=int(tile_result.tile_spec.overlap_px),
        )

    def _build_group_mask(
        self,
        group: Sequence[_PatchCandidate],
        *,
        W: int,
        H: int,
        window: str,
    ) -> np.ndarray:
        gx0 = min(item.bbox.x0 for item in group)
        gy0 = min(item.bbox.y0 for item in group)
        gx1 = max(item.bbox.x1 for item in group)
        gy1 = max(item.bbox.y1 for item in group)
        width = max(0, gx1 - gx0)
        height = max(0, gy1 - gy0)
        if width <= 0 or height <= 0:
            return np.zeros((H, W), dtype=bool)

        weighted = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        single_source = np.zeros((height, width), dtype=np.float32)
        contributions = np.zeros((height, width), dtype=np.uint16)

        for item in group:
            x0 = item.bbox.x0 - gx0
            y0 = item.bbox.y0 - gy0
            x1 = item.bbox.x1 - gx0
            y1 = item.bbox.y1 - gy0
            weight = self._candidate_weight(item, W=W, H=H, mode=window)
            weighted[y0:y1, x0:x1] += item.prob_patch * weight
            weights[y0:y1, x0:x1] += weight
            single_source[y0:y1, x0:x1] = np.maximum(single_source[y0:y1, x0:x1], item.prob_patch)
            contributions[y0:y1, x0:x1] += 1

        probabilities = np.zeros((height, width), dtype=np.float32)
        multi = (contributions > 1) & (weights > 0.0)
        probabilities[multi] = weighted[multi] / weights[multi]
        single = contributions == 1
        probabilities[single] = single_source[single]
        full_mask = np.zeros((H, W), dtype=bool)
        full_mask[gy0:gy1, gx0:gx1] = probabilities > 0.5
        return full_mask

    def merge(
        self,
        *,
        image_hash: str,
        W: int,
        H: int,
        manifest: TileManifest,
        tile_results: Sequence[TileSegmentationResult],
        global_hints: Optional[GlobalSeedHints],
        merge_config: MergeConfig,
    ) -> Tuple[np.ndarray, np.ndarray, list[MaskMetadata], dict]:
        del image_hash, manifest, global_hints
        candidates: list[_PatchCandidate] = []
        skipped_zero_area = 0
        next_order = 0
        for tile_result in tile_results:
            for instance in tile_result.instances:
                candidate = self._materialize_candidate(
                    instance=instance,
                    tile_result=tile_result,
                    W=W,
                    H=H,
                    order=next_order,
                )
                next_order += 1
                if candidate is None:
                    skipped_zero_area += 1
                    continue
                candidates.append(candidate)

        merge_stats: dict[str, Any] = {
            "unions_performed": 0,
            "instances_in": len(candidates),
            "instances_out": len(candidates),
            "skipped_zero_area": skipped_zero_area,
            "seam_metrics": {
                "merged_pair_count": 0,
                "max_merged_discontinuity": 0.0,
                "mean_merged_discontinuity": 0.0,
            },
            "warnings": [],
        }

        if not candidates:
            return (
                np.zeros((0, H, W), dtype=bool),
                np.zeros((0,), dtype=np.float32),
                [],
                merge_stats,
            )

        instance_merge = getattr(merge_config, "instance_merge", None)
        merge_enabled = bool(getattr(instance_merge, "enabled", False))
        merged_pair_discontinuities: list[float] = []

        if merge_enabled:
            parent = list(range(len(candidates)))
            iou_threshold = float(getattr(instance_merge, "iou_threshold", 0.35))
            border_only = bool(getattr(instance_merge, "border_only", True))
            continuity_threshold = 0.80

            def find(idx: int) -> int:
                while parent[idx] != idx:
                    parent[idx] = parent[parent[idx]]
                    idx = parent[idx]
                return idx

            def union(left_idx: int, right_idx: int) -> None:
                left_root = find(left_idx)
                right_root = find(right_idx)
                if left_root == right_root:
                    return
                parent[right_root] = left_root

            for left_idx in range(len(candidates)):
                for right_idx in range(left_idx + 1, len(candidates)):
                    left = candidates[left_idx]
                    right = candidates[right_idx]
                    if left.tile_id == right.tile_id:
                        continue
                    if self._labels_conflict(left, right):
                        continue
                    tile_overlap = self._bbox_overlap_window(left.tile_bbox, right.tile_bbox)
                    if tile_overlap is None:
                        continue
                    candidate_overlap = self._bbox_overlap_window(left.bbox, right.bbox)
                    if candidate_overlap is None:
                        continue
                    overlap_window = self._bbox_overlap_window(
                        BBox(*tile_overlap),
                        BBox(*candidate_overlap),
                    )
                    if overlap_window is None:
                        continue
                    continuity = self._overlap_band_continuity(left, right, overlap_window)
                    if border_only and continuity <= 0.0:
                        continue
                    should_merge = (
                        self._iou(left, right, overlap_window) >= iou_threshold or continuity >= continuity_threshold
                    )
                    if should_merge:
                        union(left_idx, right_idx)
                        merged_pair_discontinuities.append(float(1.0 - continuity))

            grouped: dict[int, list[_PatchCandidate]] = {}
            for idx, candidate in enumerate(candidates):
                grouped.setdefault(find(idx), []).append(candidate)
            groups = sorted(grouped.values(), key=lambda group: min(item.order for item in group))
        else:
            groups = [[candidate] for candidate in candidates]

        masks = []
        scores = []
        metadata = []
        unions_performed = 0
        for group in groups:
            union_mask = self._build_group_mask(group, W=W, H=H, window=merge_config.window)
            area = int(np.count_nonzero(union_mask))
            if area <= 0:
                continue

            labels = [item.material_label for item in group if item.material_label]
            material_label = labels[0] if labels and all(label == labels[0] for label in labels) else None
            material_confidence = self._weighted_confidence(group, material_label)
            score = float(np.clip(self._weighted_mean(group, "score"), 0.0, 1.0))
            stability_score = float(np.clip(self._weighted_mean(group, "stability_score"), 0.0, 1.0))

            metadata.append(
                make_mask_metadata(
                    area=area,
                    bbox=mask_bbox_xywh(union_mask),
                    stability_score=stability_score,
                    material_label=material_label,
                    material_confidence=material_confidence,
                )
            )
            masks.append(union_mask)
            scores.append(score)
            unions_performed += max(0, len(group) - 1)

        merge_stats["unions_performed"] = unions_performed
        merge_stats["instances_out"] = len(masks)
        if merged_pair_discontinuities:
            merge_stats["seam_metrics"] = {
                "merged_pair_count": len(merged_pair_discontinuities),
                "max_merged_discontinuity": float(max(merged_pair_discontinuities)),
                "mean_merged_discontinuity": float(np.mean(merged_pair_discontinuities)),
            }

        if not masks:
            return (
                np.zeros((0, H, W), dtype=bool),
                np.zeros((0,), dtype=np.float32),
                [],
                merge_stats,
            )
        return (
            np.stack(masks).astype(bool, copy=False),
            np.array(scores, dtype=np.float32),
            metadata,
            merge_stats,
        )
