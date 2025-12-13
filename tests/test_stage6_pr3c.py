#!/usr/bin/env python3
"""Tests for Stage 6 PR-3C boundary metrics integration."""
from __future__ import annotations

import numpy as np
import pytest

from lux_depth_v2.metrics.boundary_metrics import (
    compute_full_boundary_metrics,
    extract_boundary_band,
    compute_boundary_f1,
    compute_trimap_iou,
)


def test_extract_boundary_band_both():
    """Test boundary band extraction (both sides)."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    
    boundary = extract_boundary_band(mask, band_width_px=2, mode="both")
    
    # Boundary should be a ring around the mask
    assert boundary.sum() > 0
    # Core should not be in boundary
    assert not boundary[12, 12]
    # Just outside should be in boundary
    assert boundary[6, 12] or boundary[12, 6]


def test_compute_boundary_f1_perfect_match():
    """Test boundary F1 with perfect match."""
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0
    
    f1, prec, rec, bpx = compute_boundary_f1(mask, mask, band_width_px=3)
    
    assert f1 == pytest.approx(1.0)
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
    assert bpx > 0


def test_compute_boundary_f1_shifted():
    """Test boundary F1 with shifted mask."""
    ref_mask = np.zeros((32, 32), dtype=np.float32)
    ref_mask[8:24, 8:24] = 1.0
    
    pred_mask = np.zeros((32, 32), dtype=np.float32)
    pred_mask[9:25, 9:25] = 1.0  # Shifted by 1px
    
    f1, prec, rec, bpx = compute_boundary_f1(pred_mask, ref_mask, band_width_px=3)
    
    # Should have some overlap but not perfect
    assert 0.3 < f1 < 1.0
    assert bpx > 0


def test_compute_trimap_iou():
    """Test trimap IoU computation."""
    ref_mask = np.zeros((32, 32), dtype=np.float32)
    ref_mask[8:24, 8:24] = 1.0
    
    pred_mask = ref_mask.copy()
    
    iou_core, iou_boundary, iou_bg = compute_trimap_iou(
        pred_mask, ref_mask, band_width_px=3
    )
    
    # Perfect match should have high IoU across all regions
    assert iou_core == pytest.approx(1.0)
    assert iou_boundary == pytest.approx(1.0)
    assert iou_bg == pytest.approx(1.0)


def test_compute_full_boundary_metrics():
    """Test full boundary metrics computation."""
    ref_mask = np.zeros((32, 32), dtype=np.float32)
    ref_mask[8:24, 8:24] = 1.0
    
    pred_mask = ref_mask.copy()
    
    # Add some noise to prediction
    pred_mask[10, 10] = 0.0
    pred_mask[25, 25] = 1.0
    
    metrics = compute_full_boundary_metrics(pred_mask, ref_mask, band_width_px=3)
    
    # Should have high but not perfect scores
    assert 0.8 < metrics.boundary_f1 <= 1.0
    assert metrics.boundary_pixels > 0
    assert 0.0 <= metrics.trimap_iou_core <= 1.0
    assert 0.0 <= metrics.trimap_iou_boundary <= 1.0


def test_boundary_metrics_with_gradients():
    """Test boundary metrics with image gradients."""
    ref_mask = np.zeros((32, 32), dtype=np.float32)
    ref_mask[8:24, 8:24] = 1.0
    
    pred_mask = ref_mask.copy()
    
    # Synthetic gradients (high at edges)
    gradients = np.zeros((32, 32), dtype=np.float32)
    gradients[7:9, :] = 1.0  # Top edge
    gradients[23:25, :] = 1.0  # Bottom edge
    gradients[:, 7:9] = 1.0  # Left edge
    gradients[:, 23:25] = 1.0  # Right edge
    
    metrics = compute_full_boundary_metrics(
        pred_mask, ref_mask, image_gradients=gradients, band_width_px=3
    )
    
    # Edge alignment should be high (mask aligns with gradients)
    assert metrics.edge_alignment > 0.3


def test_boundary_f1_degenerate_empty():
    """Test boundary F1 with empty masks."""
    empty_mask = np.zeros((32, 32), dtype=np.float32)
    
    f1, prec, rec, bpx = compute_boundary_f1(empty_mask, empty_mask, band_width_px=3)
    
    # Empty vs empty should be considered perfect (or 0 pixels)
    assert f1 == pytest.approx(1.0)
    assert bpx == 0


def test_shape_mismatch_raises():
    """Test that shape mismatch raises ValueError."""
    mask1 = np.zeros((32, 32), dtype=np.float32)
    mask2 = np.zeros((64, 64), dtype=np.float32)
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_boundary_f1(mask1, mask2)
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_trimap_iou(mask1, mask2)
