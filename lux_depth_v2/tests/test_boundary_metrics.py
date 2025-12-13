"""Unit tests for boundary metrics."""

import numpy as np
import pytest

from lux_depth_v2.metrics.boundary_metrics import (
    BoundaryMetrics,
    compute_boundary_f1,
    compute_trimap_iou,
    extract_boundary_band,
    compute_edge_alignment,
    compute_full_boundary_metrics,
)


def _create_circle_mask(h=64, w=64, r=20):
    """Create a circular mask for testing."""
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return mask.astype(np.float32)


def _create_square_mask(h=64, w=64, x0=20, y0=20, x1=44, y1=44):
    """Create a square mask."""
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def test_extract_boundary_band_both():
    """Test boundary extraction (both sides)."""
    mask = _create_square_mask()
    boundary = extract_boundary_band(mask, band_width_px=2, mode="both")
    
    # Boundary should form a band around the square
    assert boundary.sum() > 0
    # Should be roughly (perimeter * band_width)
    # For 24x24 square, perimeter ~ 96, band ~ 96*2 = 192
    assert 100 < boundary.sum() < 400


def test_extract_boundary_band_inside():
    """Test boundary extraction (inside only)."""
    mask = _create_square_mask()
    boundary = extract_boundary_band(mask, band_width_px=2, mode="inside")
    
    # Inside boundary should be within dilated region
    assert boundary.sum() > 0
    assert not np.any(boundary & (mask > 0.5))


def test_extract_boundary_band_outside():
    """Test boundary extraction (outside only)."""
    mask = _create_square_mask()
    boundary = extract_boundary_band(mask, band_width_px=2, mode="outside")
    
    # Outside boundary should be within original mask
    assert boundary.sum() > 0
    assert np.all(boundary <= (mask > 0.5))


def test_compute_boundary_f1_perfect_match():
    """Test boundary F1 with perfect match."""
    mask = _create_circle_mask()
    f1, prec, rec, bpx = compute_boundary_f1(mask, mask, band_width_px=3)
    
    assert f1 == pytest.approx(1.0)
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
    assert bpx > 0


def test_compute_boundary_f1_partial_match():
    """Test boundary F1 with partial overlap."""
    ref = _create_square_mask(x0=20, y0=20, x1=44, y1=44)
    pred = _create_square_mask(x0=22, y0=22, x1=46, y1=46)  # shifted
    
    f1, prec, rec, bpx = compute_boundary_f1(pred, ref, band_width_px=3)
    
    # Should have partial match (not perfect, not zero)
    assert 0.0 < f1 < 1.0
    assert 0.0 < prec < 1.0
    assert 0.0 < rec < 1.0
    assert bpx > 0


def test_compute_boundary_f1_no_overlap():
    """Test boundary F1 with disjoint masks."""
    ref = _create_square_mask(x0=10, y0=10, x1=30, y1=30)
    pred = _create_square_mask(x0=40, y0=40, x1=60, y1=60)
    
    f1, prec, rec, bpx = compute_boundary_f1(pred, ref, band_width_px=3)
    
    # Disjoint → F1 should be very low
    assert f1 < 0.2
    assert bpx > 0


def test_compute_trimap_iou_perfect():
    """Test trimap IoU with perfect match."""
    mask = _create_circle_mask()
    iou_core, iou_boundary, iou_bg = compute_trimap_iou(mask, mask, band_width_px=3)
    
    assert iou_core == pytest.approx(1.0)
    assert iou_boundary == pytest.approx(1.0)
    assert iou_bg == pytest.approx(1.0)


def test_compute_trimap_iou_partial():
    """Test trimap IoU with partial match."""
    ref = _create_circle_mask(r=20)
    pred = _create_circle_mask(r=18)  # smaller circle
    
    iou_core, iou_boundary, iou_bg = compute_trimap_iou(pred, ref, band_width_px=2)
    
    # Core should have decent overlap
    assert iou_core > 0.5
    # Boundary IoU can be low or zero for very different radii
    assert 0.0 <= iou_boundary <= iou_core
    # Background should be high
    assert iou_bg > 0.9


def test_compute_edge_alignment_synthetic():
    """Test edge alignment with synthetic gradients."""
    mask = _create_square_mask()
    
    # Create synthetic gradients (high at edges)
    gradients = np.zeros((64, 64), dtype=np.float32)
    # Draw gradient peaks at square edges
    gradients[19:45, 19:21] = 1.0  # left edge
    gradients[19:45, 43:45] = 1.0  # right edge
    gradients[19:21, 19:45] = 1.0  # top edge
    gradients[43:45, 19:45] = 1.0  # bottom edge
    
    alignment = compute_edge_alignment(mask, gradients, band_width_px=3)
    
    # Should have reasonable alignment
    assert 0.3 < alignment <= 1.0


def test_compute_full_boundary_metrics():
    """Test full metrics computation."""
    ref = _create_circle_mask(r=20)
    pred = _create_circle_mask(r=19)
    
    # Create synthetic gradients
    gradients = np.random.rand(64, 64).astype(np.float32) * 0.5
    
    metrics = compute_full_boundary_metrics(
        pred, ref, image_gradients=gradients, band_width_px=3
    )
    
    assert isinstance(metrics, BoundaryMetrics)
    assert 0.0 <= metrics.boundary_f1 <= 1.0
    assert 0.0 <= metrics.boundary_precision <= 1.0
    assert 0.0 <= metrics.boundary_recall <= 1.0
    assert 0.0 <= metrics.trimap_iou_core <= 1.0
    assert 0.0 <= metrics.trimap_iou_boundary <= 1.0
    assert 0.0 <= metrics.trimap_iou_background <= 1.0
    assert 0.0 <= metrics.edge_alignment <= 1.0
    assert metrics.boundary_pixels > 0


def test_boundary_metrics_to_dict():
    """Test BoundaryMetrics serialization."""
    metrics = BoundaryMetrics(
        boundary_f1=0.85,
        boundary_precision=0.90,
        boundary_recall=0.80,
        trimap_iou_core=0.92,
        trimap_iou_boundary=0.75,
        trimap_iou_background=0.98,
        edge_alignment=0.65,
        boundary_pixels=150,
    )
    
    d = metrics.to_dict()
    
    assert d["boundary_f1"] == 0.85
    assert d["boundary_pixels"] == 150
    assert all(isinstance(v, (int, float)) for v in d.values())


def test_shape_mismatch_errors():
    """Test that shape mismatches raise errors."""
    mask1 = np.zeros((32, 32), dtype=np.float32)
    mask2 = np.zeros((64, 64), dtype=np.float32)
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_boundary_f1(mask1, mask2)
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_trimap_iou(mask1, mask2)
    
    gradients = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_edge_alignment(mask1, gradients)


def test_degenerate_empty_mask():
    """Test metrics on empty masks (edge case)."""
    empty = np.zeros((32, 32), dtype=np.float32)
    nonempty = _create_square_mask(h=32, w=32, x0=10, y0=10, x1=20, y1=20)
    
    # Empty ref mask → should handle gracefully
    f1, prec, rec, bpx = compute_boundary_f1(nonempty, empty, band_width_px=2)
    # No reference boundary → perfect score by convention
    assert f1 == 1.0
    assert bpx == 0
