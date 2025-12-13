"""Boundary-aware metrics for Materials V3 edge refinement evaluation.

These metrics answer: "Did EfficientSAM refinement improve edge quality?"

Unlike mean IoU (which treats all pixels equally), boundary metrics focus on
the edge band where refinement actually matters.

Key metrics:
- Boundary F1 (BF1): precision/recall on edge-band pixels
- Trimap IoU: separate IoU for core vs boundary vs background
- Edge alignment: correlation with image gradients / depth edges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion


@dataclass
class BoundaryMetrics:
    """Boundary-focused metrics for segmentation quality.
    
    Attributes
    ----------
    boundary_f1 : float
        F1 score on boundary band pixels (primary metric)
    boundary_precision : float
        Precision on boundary band
    boundary_recall : float
        Recall on boundary band
    trimap_iou_core : float
        IoU on core region (high-confidence interior)
    trimap_iou_boundary : float
        IoU on boundary band (transition zone)
    trimap_iou_background : float
        IoU on background
    edge_alignment : float
        Correlation with image gradients (0-1, higher is better)
    boundary_pixels : int
        Number of pixels in boundary band
    """
    
    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    trimap_iou_core: float
    trimap_iou_boundary: float
    trimap_iou_background: float
    edge_alignment: float = 0.0
    boundary_pixels: int = 0
    
    def to_dict(self) -> dict:
        return {
            "boundary_f1": float(self.boundary_f1),
            "boundary_precision": float(self.boundary_precision),
            "boundary_recall": float(self.boundary_recall),
            "trimap_iou_core": float(self.trimap_iou_core),
            "trimap_iou_boundary": float(self.trimap_iou_boundary),
            "trimap_iou_background": float(self.trimap_iou_background),
            "edge_alignment": float(self.edge_alignment),
            "boundary_pixels": int(self.boundary_pixels),
        }


def extract_boundary_band(
    mask: np.ndarray,
    *,
    band_width_px: int = 5,
    mode: str = "both",
) -> np.ndarray:
    """Extract boundary band from a binary mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask (bool or 0/1)
    band_width_px : int
        Width of boundary band in pixels
    mode : str
        'both' (inside+outside), 'inside' (dilation only), 'outside' (erosion only)
    
    Returns
    -------
    np.ndarray
        Boolean mask of boundary pixels
    """
    mask_bin = mask.astype(bool, copy=False) if mask.dtype != bool else mask
    
    if mode == "inside":
        dilated = binary_dilation(mask_bin, iterations=band_width_px)
        return dilated & ~mask_bin
    
    elif mode == "outside":
        eroded = binary_erosion(mask_bin, iterations=band_width_px)
        return mask_bin & ~eroded
    
    elif mode == "both":
        dilated = binary_dilation(mask_bin, iterations=band_width_px)
        eroded = binary_erosion(mask_bin, iterations=band_width_px)
        return (dilated & ~eroded)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_boundary_f1(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    *,
    band_width_px: int = 5,
    mode: str = "both",
) -> Tuple[float, float, float, int]:
    """Compute boundary F1 score.
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted mask (bool or float [0,1])
    ref_mask : np.ndarray
        Reference mask (bool or float [0,1])
    band_width_px : int
        Boundary band width in pixels
    mode : str
        Boundary extraction mode
    
    Returns
    -------
    boundary_f1 : float
    boundary_precision : float
    boundary_recall : float
    boundary_pixels : int
    """
    if pred_mask.shape != ref_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_mask.shape} vs ref {ref_mask.shape}"
        )
    
    # Binarize if needed
    pred_bin = (pred_mask >= 0.5) if pred_mask.dtype != bool else pred_mask
    ref_bin = (ref_mask >= 0.5) if ref_mask.dtype != bool else ref_mask
    
    # Extract boundary bands
    pred_boundary = extract_boundary_band(pred_bin, band_width_px=band_width_px, mode=mode)
    ref_boundary = extract_boundary_band(ref_bin, band_width_px=band_width_px, mode=mode)
    
    boundary_pixels = int(ref_boundary.sum())
    
    if boundary_pixels == 0:
        # Degenerate case: no reference boundary
        return 1.0, 1.0, 1.0, 0
    
    # Precision/recall on boundary pixels
    tp = (pred_boundary & ref_boundary).sum()
    fp = (pred_boundary & ~ref_boundary).sum()
    fn = (~pred_boundary & ref_boundary).sum()
    
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1, precision, recall, boundary_pixels


def compute_trimap_iou(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    *,
    band_width_px: int = 5,
) -> Tuple[float, float, float]:
    """Compute IoU separately for core, boundary, and background regions.
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted mask (bool or float [0,1])
    ref_mask : np.ndarray
        Reference mask (bool or float [0,1])
    band_width_px : int
        Boundary band width
    
    Returns
    -------
    iou_core : float
    iou_boundary : float
    iou_background : float
    """
    if pred_mask.shape != ref_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_mask.shape} vs ref {ref_mask.shape}"
        )
    
    # Binarize
    pred_bin = (pred_mask >= 0.5) if pred_mask.dtype != bool else pred_mask
    ref_bin = (ref_mask >= 0.5) if ref_mask.dtype != bool else ref_mask
    
    # Define trimap regions from reference
    ref_core = binary_erosion(ref_bin, iterations=band_width_px)
    ref_boundary = extract_boundary_band(ref_bin, band_width_px=band_width_px, mode="both")
    ref_background = ~(ref_core | ref_boundary)
    
    def _iou_region(pred: np.ndarray, ref: np.ndarray, region_mask: np.ndarray) -> float:
        """IoU restricted to a region."""
        pred_r = pred & region_mask
        ref_r = ref & region_mask
        inter = (pred_r & ref_r).sum()
        union = (pred_r | ref_r).sum()
        return float(inter) / float(union) if union > 0 else 1.0
    
    iou_core = _iou_region(pred_bin, ref_bin, ref_core)
    iou_boundary = _iou_region(pred_bin, ref_bin, ref_boundary)
    iou_background = _iou_region(pred_bin, ref_bin, ref_background)
    
    return iou_core, iou_boundary, iou_background


def compute_edge_alignment(
    pred_mask: np.ndarray,
    image_gradients: np.ndarray,
    *,
    band_width_px: int = 5,
) -> float:
    """Compute correlation between mask boundary and image gradients.
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted mask (bool or float [0,1])
    image_gradients : np.ndarray
        Image gradient magnitude (HxW, float)
    band_width_px : int
        Boundary band width
    
    Returns
    -------
    float
        Correlation coefficient [0,1] (1 = perfect alignment)
    """
    if pred_mask.shape != image_gradients.shape:
        raise ValueError(
            f"Shape mismatch: mask {pred_mask.shape} vs gradients {image_gradients.shape}"
        )
    
    pred_bin = (pred_mask >= 0.5) if pred_mask.dtype != bool else pred_mask
    boundary = extract_boundary_band(pred_bin, band_width_px=band_width_px, mode="both")
    
    if boundary.sum() == 0:
        return 0.0
    
    # Gradient values at boundary pixels
    grad_at_boundary = image_gradients[boundary]
    
    # Normalize to [0,1]
    grad_norm = grad_at_boundary / (grad_at_boundary.max() + 1e-8)
    
    # Mean gradient strength at boundary (higher is better)
    return float(grad_norm.mean())


def compute_full_boundary_metrics(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    image_gradients: Optional[np.ndarray] = None,
    *,
    band_width_px: int = 5,
) -> BoundaryMetrics:
    """Compute all boundary metrics in one call.
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted mask
    ref_mask : np.ndarray
        Reference mask
    image_gradients : Optional[np.ndarray]
        Image gradient magnitude (for edge alignment)
    band_width_px : int
        Boundary band width
    
    Returns
    -------
    BoundaryMetrics
    """
    f1, prec, rec, bpx = compute_boundary_f1(
        pred_mask, ref_mask, band_width_px=band_width_px
    )
    
    iou_core, iou_boundary, iou_bg = compute_trimap_iou(
        pred_mask, ref_mask, band_width_px=band_width_px
    )
    
    edge_align = 0.0
    if image_gradients is not None:
        edge_align = compute_edge_alignment(
            pred_mask, image_gradients, band_width_px=band_width_px
        )
    
    return BoundaryMetrics(
        boundary_f1=f1,
        boundary_precision=prec,
        boundary_recall=rec,
        trimap_iou_core=iou_core,
        trimap_iou_boundary=iou_boundary,
        trimap_iou_background=iou_bg,
        edge_alignment=edge_align,
        boundary_pixels=bpx,
    )
