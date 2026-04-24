"""Mask processing utilities for segmentation (Phase 2.1).

This module provides post-processing for SAM2 masks:
- Temporal consistency (tracking IDs across video frames)
- Mask refinement (morphological operations)
- Overlap resolution (handle competing masks)
- Quality filtering (score/area thresholds)

Architecture (ADR-027):
- Isolated processing logic (no model dependencies)
- Deterministic operations (fixed random seeds where applicable)
- Contract-preserving (input/output validation)

Example:
    >>> processor = MaskProcessor()
    >>> refined = processor.refine_masks(result.masks, min_area=100)
    >>> tracked = processor.track_temporal(current_masks, prev_masks, prev_ids)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationResult

logger = logging.getLogger(__name__)


class MaskProcessor:
    """Mask post-processing for temporal tracking and refinement.

    Attributes:
        min_area: Minimum mask area (pixels) to keep.
        min_stability: Minimum stability score to keep.
        iou_threshold: IoU threshold for temporal tracking (default: 0.5).
    """

    def __init__(
        self,
        min_area: int = 100,
        min_stability: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        """Initialize mask processor.

        Args:
            min_area: Minimum mask area in pixels.
            min_stability: Minimum stability score [0, 1].
            iou_threshold: IoU threshold for temporal matching.
        """
        if min_area <= 0:
            raise ValueError(f"min_area must be positive, got {min_area}")
        if not 0.0 <= min_stability <= 1.0:
            raise ValueError(f"min_stability must be in [0, 1], got {min_stability}")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

        self.min_area = min_area
        self.min_stability = min_stability
        self.iou_threshold = iou_threshold

        logger.info(
            f"MaskProcessor initialized: min_area={min_area}, " f"min_stability={min_stability}, iou_threshold={iou_threshold}"
        )

    def filter_masks(self, result: SegmentationResult) -> SegmentationResult:
        """Filter masks by area and stability thresholds.

        Args:
            result: Segmentation result to filter.

        Returns:
            Filtered segmentation result.
        """
        # Find masks that meet criteria
        valid_indices = []
        for i, metadata in enumerate(result.metadata):
            if metadata.area >= self.min_area and metadata.stability_score >= self.min_stability:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            logger.warning("No masks passed filtering criteria")
            # Return empty result
            H, W = result.masks.shape[1:3]
            return SegmentationResult(
                masks=np.zeros((0, H, W), dtype=bool),
                scores=np.array([], dtype=np.float32),
                metadata=[],
                temporal_ids=np.array([], dtype=np.int32) if result.temporal_ids is not None else None,
            )

        # Filter arrays
        filtered_masks = result.masks[valid_indices]
        filtered_scores = result.scores[valid_indices]
        filtered_metadata = [result.metadata[i] for i in valid_indices]
        filtered_temporal_ids = result.temporal_ids[valid_indices] if result.temporal_ids is not None else None

        logger.info(f"Filtered {len(result.masks)} masks to {len(filtered_masks)}")

        return SegmentationResult(
            masks=filtered_masks,
            scores=filtered_scores,
            metadata=filtered_metadata,
            temporal_ids=filtered_temporal_ids,
        )

    def refine_masks(self, masks: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Refine masks with morphological operations.

        Applies:
        1. Opening (erosion + dilation) to remove small noise
        2. Closing (dilation + erosion) to fill small holes

        Args:
            masks: Boolean masks (N, H, W).
            kernel_size: Structuring element size (default: 3x3).

        Returns:
            Refined masks (N, H, W) bool.
        """
        if masks.dtype != bool:
            raise ValueError(f"Masks must be bool, got {masks.dtype}")

        if masks.ndim != 3:
            raise ValueError(f"Masks must be (N, H, W), got shape {masks.shape}")

        # Create structuring element (square kernel)
        structure = np.ones((kernel_size, kernel_size), dtype=bool)

        refined = np.zeros_like(masks)
        for i, mask in enumerate(masks):
            # Opening (remove noise)
            opened = ndimage.binary_opening(mask, structure=structure)
            # Closing (fill holes)
            closed = ndimage.binary_closing(opened, structure=structure)
            refined[i] = closed

        return refined

    def track_temporal(
        self,
        current_masks: np.ndarray,
        prev_masks: np.ndarray,
        prev_ids: np.ndarray,
    ) -> np.ndarray:
        """Assign temporal tracking IDs to current masks.

        Matches current masks to previous frame masks using IoU.
        New masks get new IDs, matched masks inherit IDs.

        Args:
            current_masks: Current frame masks (N_cur, H, W) bool.
            prev_masks: Previous frame masks (N_prev, H, W) bool.
            prev_ids: Previous frame tracking IDs (N_prev,) int.

        Returns:
            Tracking IDs for current masks (N_cur,) int.
        """
        if current_masks.dtype != bool or prev_masks.dtype != bool:
            raise ValueError("Masks must be bool dtype")

        if current_masks.shape[1:] != prev_masks.shape[1:]:
            raise ValueError(
                f"Spatial dimensions must match: " f"current {current_masks.shape[1:]}, prev {prev_masks.shape[1:]}"
            )

        N_cur = current_masks.shape[0]
        N_prev = prev_masks.shape[0]

        if prev_ids.shape != (N_prev,):
            raise ValueError(f"prev_ids shape must be ({N_prev},), got {prev_ids.shape}")

        # Compute IoU matrix (N_cur x N_prev)
        iou_matrix = self._compute_iou_matrix(current_masks, prev_masks)

        # Assign IDs
        current_ids = np.zeros(N_cur, dtype=np.int32)
        if N_prev == 0:
            return np.arange(N_cur, dtype=np.int32)

        used_prev_ids = set()
        next_new_id = prev_ids.max() + 1 if len(prev_ids) > 0 else 0

        for i in range(N_cur):
            # Find best match in previous frame
            best_prev_idx = iou_matrix[i].argmax()
            best_iou = iou_matrix[i, best_prev_idx]

            if best_iou >= self.iou_threshold and best_prev_idx not in used_prev_ids:
                # Match found: inherit ID
                current_ids[i] = prev_ids[best_prev_idx]
                used_prev_ids.add(best_prev_idx)
            else:
                # No match: assign new ID
                current_ids[i] = next_new_id
                next_new_id += 1

        logger.debug(
            f"Temporal tracking: {N_cur} current masks, " f"{len(used_prev_ids)} matched, {N_cur - len(used_prev_ids)} new"
        )

        return current_ids

    def resolve_overlaps(self, masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Resolve overlapping masks (keep highest score).

        Args:
            masks: Boolean masks (N, H, W).
            scores: Confidence scores (N,).

        Returns:
            Non-overlapping masks (N, H, W) bool.
        """
        if masks.dtype != bool:
            raise ValueError(f"Masks must be bool, got {masks.dtype}")

        if masks.shape[0] != scores.shape[0]:
            raise ValueError(f"Masks and scores must have same N, got {masks.shape[0]} vs {scores.shape[0]}")

        N, H, W = masks.shape

        # Sort masks by score (descending)
        sorted_indices = np.argsort(-scores)  # Negative for descending

        # Create resolved masks (highest score wins per pixel)
        resolved = np.zeros((N, H, W), dtype=bool)
        occupied = np.zeros((H, W), dtype=bool)

        for idx in sorted_indices:
            mask = masks[idx]
            # Keep only non-occupied pixels
            resolved[idx] = mask & ~occupied
            # Mark these pixels as occupied
            occupied |= resolved[idx]

        return resolved

    def _compute_iou_matrix(self, masks_a: np.ndarray, masks_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of masks.

        Args:
            masks_a: Masks (N_a, H, W) bool.
            masks_b: Masks (N_b, H, W) bool.

        Returns:
            IoU matrix (N_a, N_b) float.
        """
        N_a = masks_a.shape[0]
        N_b = masks_b.shape[0]

        iou_matrix = np.zeros((N_a, N_b), dtype=np.float32)

        for i in range(N_a):
            for j in range(N_b):
                intersection = (masks_a[i] & masks_b[j]).sum()
                union = (masks_a[i] | masks_b[j]).sum()

                if union > 0:
                    iou_matrix[i, j] = intersection / union
                else:
                    iou_matrix[i, j] = 0.0

        return iou_matrix
