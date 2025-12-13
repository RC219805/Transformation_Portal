# lux_depth_v2/segmentation_fusion.py
"""
Segmentation mask fusion utilities for EfficientSAM V3.

Provides IoU-gated confidence-weighted fusion between base segmentation
(SegFormer) and refined segmentation (EfficientSAM).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np


class FusionMode(str, Enum):
    NONE = "none"
    UNION = "union"
    INTERSECTION = "intersection"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass(frozen=True)
class FusionConfig:
    mode: FusionMode = FusionMode.CONFIDENCE_WEIGHTED

    # IoU gating: if refined mask disagrees too much with base, skip fusion
    min_iou: float = 0.30

    # Core / edge thresholds computed from base confidence map
    core_thresh: float = 0.70
    edge_low: float = 0.20
    edge_high: float = 0.70

    # Alpha blending on edges vs core for CONFIDENCE_WEIGHTED
    alpha_edge: float = 0.70
    alpha_core: float = 0.30

    clamp: bool = True


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def mask_iou(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    """IoU of two boolean masks."""
    if a_bin.dtype != bool:
        a_bin = a_bin.astype(bool)
    if b_bin.dtype != bool:
        b_bin = b_bin.astype(bool)

    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def compute_core_edge_bands(
    base_conf: np.ndarray,
    *,
    core_thresh: float,
    edge_low: float,
    edge_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a base confidence map in [0,1], compute:
      - core mask: base_conf >= core_thresh
      - edge band mask: edge_low < base_conf <= edge_high
    """
    if base_conf.ndim != 2:
        raise ValueError(f"Expected 2D base_conf, got {base_conf.shape}")

    core = base_conf >= core_thresh
    edge = (base_conf > edge_low) & (base_conf <= edge_high)
    return core, edge


def fuse_masks(
    base_conf: np.ndarray,
    refined_conf: np.ndarray,
    cfg: FusionConfig,
    *,
    base_bin_thresh: float = 0.50,
    refined_bin_thresh: float = 0.50,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fuse base (SegFormer) confidence with refined (EfficientSAM) confidence.

    Returns:
      fused_conf: float32 HxW in [0,1]
      stats: {'iou_base_vs_refined': float, 'fusion_applied': 0/1}
    """
    if base_conf.shape != refined_conf.shape:
        raise ValueError(
            f"Shape mismatch: base {base_conf.shape} vs refined {refined_conf.shape}"
        )
    if base_conf.ndim != 2:
        raise ValueError(f"Expected 2D masks, got {base_conf.shape}")

    base = base_conf.astype(np.float32, copy=False)
    refined = refined_conf.astype(np.float32, copy=False)

    if cfg.clamp:
        base = _clamp01(base)
        refined = _clamp01(refined)

    base_bin = base >= base_bin_thresh
    refined_bin = refined >= refined_bin_thresh
    iou = mask_iou(base_bin, refined_bin)

    stats: Dict[str, float] = {
        "iou_base_vs_refined": float(iou),
        "fusion_applied": 0.0,
    }

    # IoU gating
    if iou < cfg.min_iou:
        return base, stats

    mode = cfg.mode

    if mode == FusionMode.NONE:
        return base, stats

    if mode == FusionMode.UNION:
        stats["fusion_applied"] = 1.0
        return np.maximum(base, refined), stats

    if mode == FusionMode.INTERSECTION:
        stats["fusion_applied"] = 1.0
        return np.minimum(base, refined), stats

    if mode == FusionMode.CONFIDENCE_WEIGHTED:
        core, edge = compute_core_edge_bands(
            base,
            core_thresh=cfg.core_thresh,
            edge_low=cfg.edge_low,
            edge_high=cfg.edge_high,
        )
        fused = base.copy()

        if np.any(core):
            fused[core] = cfg.alpha_core * refined[core] + (1.0 - cfg.alpha_core) * base[core]
        if np.any(edge):
            fused[edge] = cfg.alpha_edge * refined[edge] + (1.0 - cfg.alpha_edge) * base[edge]

        if cfg.clamp:
            fused = _clamp01(fused)

        stats["fusion_applied"] = 1.0
        return fused, stats

    raise ValueError(f"Unknown FusionMode: {mode}")
