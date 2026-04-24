"""Shared mask metadata helpers for segmentation backends."""

from __future__ import annotations

import inspect
from typing import Any, Optional, Tuple, cast

import numpy as np

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata

_MASK_METADATA_FIELDS = frozenset(inspect.signature(MaskMetadata).parameters)


def mask_bbox_xywh(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return an xywh bounding box for a non-empty 2D mask."""
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 1, 1)
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def make_mask_metadata(
    *,
    area: int,
    bbox: Tuple[int, int, int, int],
    stability_score: float,
    material_label: Optional[str] = None,
    material_confidence: Optional[float] = None,
    is_empty: bool = False,
) -> MaskMetadata:
    """Build MaskMetadata while preserving positive-area constructor compatibility."""
    kwargs = {
        "area": max(int(area), 1),
        "bbox": bbox,
        "stability_score": float(stability_score),
        "material_label": material_label,
        "material_confidence": material_confidence,
        "is_empty": bool(is_empty),
    }
    filtered = {key: value for key, value in kwargs.items() if key in _MASK_METADATA_FIELDS}
    return cast(MaskMetadata, MaskMetadata(**cast(Any, filtered)))


def metadata_from_mask(
    mask: np.ndarray,
    *,
    stability_score: float,
    material_label: Optional[str] = None,
    material_confidence: Optional[float] = None,
    is_empty: Optional[bool] = None,
) -> MaskMetadata:
    """Create metadata from a 2D mask, marking zero-area masks explicitly."""
    area = int(np.count_nonzero(mask))
    empty = area <= 0 if is_empty is None else bool(is_empty)
    return make_mask_metadata(
        area=area,
        bbox=mask_bbox_xywh(mask),
        stability_score=stability_score,
        material_label=material_label,
        material_confidence=material_confidence,
        is_empty=empty,
    )
