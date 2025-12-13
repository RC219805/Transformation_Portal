# lux_depth_v2/tests/test_segmentation_fusion.py
"""Tests for segmentation mask fusion utilities."""
import numpy as np
import pytest

from lux_depth_v2.segmentation_fusion import FusionConfig, FusionMode, fuse_masks, mask_iou


def _square(h=32, w=32, x0=8, y0=8, x1=24, y1=24, v=1.0):
    """Generate a square mask for testing."""
    m = np.zeros((h, w), dtype=np.float32)
    m[y0:y1, x0:x1] = v
    return m


def test_mask_iou_identical():
    """Test IoU of identical masks is 1.0."""
    a = _square()
    b = _square()
    assert mask_iou(a > 0.5, b > 0.5) == 1.0


def test_mask_iou_disjoint():
    """Test IoU of disjoint masks is 0.0."""
    a = _square(x0=0, y0=0, x1=8, y1=8)
    b = _square(x0=24, y0=24, x1=32, y1=32)
    assert mask_iou(a > 0.5, b > 0.5) == 0.0


def test_iou_gating_skips_when_disjoint():
    """Test fusion is skipped when IoU is below threshold."""
    base = _square()
    refined = _square(x0=0, y0=0, x1=6, y1=6)
    cfg = FusionConfig(mode=FusionMode.CONFIDENCE_WEIGHTED, min_iou=0.5)
    fused, stats = fuse_masks(base, refined, cfg)
    assert stats["fusion_applied"] == 0.0
    assert np.allclose(fused, base)


def test_union_mode():
    """Test UNION mode combines masks with max operation."""
    base = _square()
    refined = _square(x0=10, y0=10, x1=26, y1=26)
    cfg = FusionConfig(mode=FusionMode.UNION, min_iou=0.1)
    fused, stats = fuse_masks(base, refined, cfg)
    assert stats["fusion_applied"] == 1.0
    assert fused.sum() >= base.sum()
    assert fused.sum() >= refined.sum()


def test_intersection_mode():
    """Test INTERSECTION mode combines masks with min operation."""
    base = _square()
    refined = _square(x0=10, y0=10, x1=22, y1=22)
    cfg = FusionConfig(mode=FusionMode.INTERSECTION, min_iou=0.1)
    fused, stats = fuse_masks(base, refined, cfg)
    assert stats["fusion_applied"] == 1.0
    assert fused.sum() <= base.sum()
    assert fused.sum() <= refined.sum()


def test_confidence_weighted_core_vs_edge_behavior():
    """Test CONFIDENCE_WEIGHTED mode uses different alpha for core vs edge."""
    base = np.zeros((32, 32), dtype=np.float32)
    base[8:24, 8:24] = 0.9
    base[7:25, 7:25] = np.maximum(base[7:25, 7:25], 0.4)  # edge band

    refined = _square(x0=9, y0=9, x1=23, y1=23, v=1.0)

    cfg = FusionConfig(
        mode=FusionMode.CONFIDENCE_WEIGHTED,
        min_iou=0.1,
        core_thresh=0.7,
        edge_low=0.2,
        edge_high=0.7,
        alpha_edge=0.7,
        alpha_core=0.3,
    )
    fused, stats = fuse_masks(base, refined, cfg)
    assert stats["fusion_applied"] == 1.0

    # core pixel should be closer to base (alpha_core=0.3)
    core = (12, 12)
    assert abs(fused[core] - base[core]) < abs(fused[core] - refined[core])

    # edge pixel should lean refined (alpha_edge=0.7)
    edge = (7, 12)
    assert abs(fused[edge] - refined[edge]) < abs(fused[edge] - base[edge])


def test_none_mode_returns_base():
    """Test NONE mode returns base mask unchanged."""
    base = _square()
    refined = _square(x0=10, y0=10, x1=26, y1=26, v=0.5)
    cfg = FusionConfig(mode=FusionMode.NONE)
    fused, stats = fuse_masks(base, refined, cfg)
    assert stats["fusion_applied"] == 0.0
    assert np.allclose(fused, base)


def test_shape_mismatch_raises():
    """Test fusion raises ValueError on shape mismatch."""
    base = _square(h=32, w=32)
    refined = _square(h=16, w=16)
    cfg = FusionConfig()
    with pytest.raises(ValueError, match="Shape mismatch"):
        fuse_masks(base, refined, cfg)
