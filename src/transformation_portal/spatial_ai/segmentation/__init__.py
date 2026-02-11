"""Segmentation module for Spatial AI Foundation (Phase 2.1).

This module provides SAM2-based segmentation with temporal consistency
for material/object boundary detection in architectural visualization.

Architecture (ADR-027):
- Isolated from lux_depth_v3 (ADR-023 compliance)
- Contract-driven design (input/output validation)
- Optional CLIP material classification
- Video temporal tracking support

Components:
- contracts: SegmentationInput, SegmentationResult, MaskMetadata
- sam2_backend: SAM2 model wrapper with HF integration
- mask_processor: Temporal consistency and refinement
- material_classifier: Optional CLIP-based material labeling

Example:
    >>> from transformation_portal.spatial_ai.segmentation import SAM2Backend
    >>> backend = SAM2Backend(model_size="large", device="cuda")
    >>> result = backend.segment(linear_rgb, gamma=1.0, mode="auto")
    >>> assert result.masks.shape[0] > 0  # At least one mask
    >>> assert result.masks.dtype == bool
"""

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult
from transformation_portal.spatial_ai.segmentation.mask_processor import MaskProcessor
from transformation_portal.spatial_ai.segmentation.material_classifier import MaterialClassifier
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

__all__ = [
    "SegmentationInput",
    "SegmentationResult",
    "MaskMetadata",
    "SAM2Backend",
    "MaskProcessor",
    "MaterialClassifier",
]
