"""Data contracts for segmentation module (Phase 2.1).

Contract validation ensures:
- Gamma=1.0 enforcement (linear RGB only)
- Float32 dtype for RGB inputs
- Bool dtype for mask outputs
- Valid coordinate ranges for bounding boxes
- Confidence scores in [0, 1]

Architecture (ADR-027):
- SpatialCaptureV1 contract alignment (gamma=1.0)
- Explicit shape/dtype validation
- Runtime contract enforcement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


@dataclass
class SegmentationInput:
    """Input contract for SAM2 segmentation.

    Attributes:
        image: Linear RGB image (H, W, 3) float32, values in [0, ∞).
               For video mode, can be None (video_path used instead).
        gamma: Gamma value (must be 1.0 for linear).
        mode: Segmentation mode.
            - "auto": Automatic mask generation (entire image)
            - "points": Point prompts (requires prompts)
            - "bbox": Bounding box prompts (requires prompts)
            - "video": Temporal tracking (requires video_path + prompts)
        prompts: Optional prompts for interactive segmentation.
            Format: {"frame_idx": 0, "object_id": 1, "points": [[x,y]], "labels": [1]}
            or {"frame_idx": 0, "object_id": 1, "bbox": [x1,y1,x2,y2]}
        video_path: Path to video file (MP4/MOV) for video mode.
        prev_masks: Previous frame masks for video mode (N, H, W) bool (deprecated, use video_path).
        frame_idx: Frame index in video sequence (0-based, deprecated).
    """

    image: Optional[np.ndarray]
    gamma: float
    mode: Literal["auto", "points", "bbox", "video"]
    prompts: Optional[Dict] = None
    video_path: Optional[str] = None
    prev_masks: Optional[np.ndarray] = None
    frame_idx: Optional[int] = None
    # N-3 (audit finding #4 — duplicate hashing): optional SHA-256 hex
    # digest of ``image``, threaded from an upstream layer that has
    # already hashed the array (e.g. the lux_depth_v3 segmentation cache
    # building its cache key). When set, the backend reuses this digest
    # instead of recomputing one in ``_stable_image_hash``, ensuring a
    # given image is hashed at most once per pipeline run. The digest is
    # the same shape/dtype/raw-buffer formula returned by
    # ``spatial_ai.segmentation._content_digest.compute_array_sha256``.
    content_digest: Optional[str] = None

    def __post_init__(self):
        """Validate input contract."""
        # Gamma enforcement (SpatialCaptureV1 contract)
        if abs(self.gamma - 1.0) > 1e-6:
            raise ValueError(
                f"Segmentation requires gamma=1.0 (linear RGB), got {self.gamma}. "
                "This violates the SpatialCaptureV1 contract."
            )

        # Video mode has different validation
        if self.mode == "video":
            if not self.video_path:
                raise ValueError("Mode 'video' requires video_path (path to MP4/MOV file)")
            if not self.prompts:
                raise ValueError("Mode 'video' requires prompts (initial frame points/bbox)")
            # Image not required for video mode
            return

        # Image validation (required for non-video modes)
        if self.image is None:
            raise ValueError(f"Mode '{self.mode}' requires image (use video_path for video mode)")

        # Dtype enforcement
        if self.image.dtype != np.float32:
            raise ValueError(f"Image must be float32, got {self.image.dtype}. " "Convert to linear float32 first.")

        # Shape validation
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Image must be (H, W, 3), got shape {self.image.shape}")

        # Mode-specific validation
        if self.mode in ["points", "bbox"] and not self.prompts:
            raise ValueError(f"Mode '{self.mode}' requires prompts")

        # Prev masks validation
        if self.prev_masks is not None:
            if self.prev_masks.ndim != 3:
                raise ValueError(f"prev_masks must be (N, H, W), got shape {self.prev_masks.shape}")
            if self.prev_masks.dtype != bool:
                raise ValueError(f"prev_masks must be bool, got {self.prev_masks.dtype}")
            # Check spatial dimensions match
            if self.prev_masks.shape[1:] != self.image.shape[:2]:
                raise ValueError(
                    f"prev_masks spatial dims {self.prev_masks.shape[1:]} " f"must match image dims {self.image.shape[:2]}"
                )


@dataclass
class MaskMetadata:
    """Per-mask metadata.

    Attributes:
        area: Pixel count of mask.
        bbox: Bounding box (x, y, w, h) in image coordinates.
        stability_score: Mask stability score [0, 1] (higher = more stable).
        material_label: Optional material classification (e.g., "wood", "marble").
        material_confidence: Optional material classification confidence [0, 1].
        is_empty: True when metadata represents an intentionally empty mask
            placeholder, such as a missing video object in one frame.
    """

    area: int
    bbox: Tuple[int, int, int, int]
    stability_score: float
    material_label: Optional[str] = None
    material_confidence: Optional[float] = None
    is_empty: bool = False

    def __post_init__(self):
        """Validate metadata."""
        # Area must be positive
        if self.area <= 0:
            raise ValueError(f"Mask area must be positive, got {self.area}")

        # Stability score in [0, 1]
        if not 0.0 <= self.stability_score <= 1.0:
            raise ValueError(f"Stability score must be in [0, 1], got {self.stability_score}")

        # Material confidence in [0, 1] if provided
        if self.material_confidence is not None:
            if not 0.0 <= self.material_confidence <= 1.0:
                raise ValueError(f"Material confidence must be in [0, 1], got {self.material_confidence}")

        # Bbox validation
        x, y, w, h = self.bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"Bounding box width/height must be positive, got {self.bbox}")


@dataclass
class SegmentationResult:
    """Output contract for SAM2 segmentation.

    Attributes:
        masks: Boolean masks (N, H, W) where N is number of segments.
        scores: Confidence scores (N,) in [0, 1] for each mask.
        metadata: Per-mask metadata (N items).
        temporal_ids: Optional tracking IDs (N,) for video mode.
            Same ID across frames = same object tracked.
    """

    masks: np.ndarray
    scores: np.ndarray
    metadata: List[MaskMetadata]
    temporal_ids: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate output contract."""
        # Masks dtype and shape
        if self.masks.dtype != bool:
            raise ValueError(f"Masks must be bool dtype, got {self.masks.dtype}")

        if self.masks.ndim != 3:
            raise ValueError(f"Masks must be (N, H, W), got shape {self.masks.shape}")

        N = self.masks.shape[0]

        # Scores dtype and shape
        if self.scores.dtype not in [np.float32, np.float64]:
            raise ValueError(f"Scores must be float32/float64, got {self.scores.dtype}")

        if self.scores.shape != (N,):
            raise ValueError(f"Scores shape must be ({N},), got {self.scores.shape}")

        # Scores in [0, 1]
        if not np.all((self.scores >= 0.0) & (self.scores <= 1.0)):
            raise ValueError(f"All scores must be in [0, 1], got range [{self.scores.min()}, {self.scores.max()}]")

        # Metadata length
        if len(self.metadata) != N:
            raise ValueError(f"Metadata length must match N={N}, got {len(self.metadata)}")

        # Temporal IDs validation (if provided)
        if self.temporal_ids is not None:
            if self.temporal_ids.shape != (N,):
                raise ValueError(f"Temporal IDs shape must be ({N},), got {self.temporal_ids.shape}")
            if self.temporal_ids.dtype not in [np.int32, np.int64]:
                raise ValueError(f"Temporal IDs must be int32/int64, got {self.temporal_ids.dtype}")
