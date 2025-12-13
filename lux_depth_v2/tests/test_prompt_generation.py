# lux_depth_v2/tests/test_prompt_generation.py
"""
Tests for intelligent mask-driven prompt generation (PR-2).
"""

import numpy as np
import pytest

from lux_depth_v2.backends.prompt_generation import (
    PromptGenerationConfig,
    generate_prompts_from_mask,
    compute_roi_from_mask,
    farthest_point_sampling,
)


def _synthetic_mask(h=256, w=256, mode="single_blob"):
    """Generate synthetic masks for testing."""
    mask = np.zeros((h, w), dtype=np.float32)
    
    if mode == "single_blob":
        # Single high-confidence region
        y0, x0 = h // 4, w // 4
        y1, x1 = 3 * h // 4, 3 * w // 4
        mask[y0:y1, x0:x1] = 0.9
        
    elif mode == "tiny":
        # Very small region (should trigger skip)
        mask[h//2:h//2+10, w//2:w//2+10] = 0.8
        
    elif mode == "low_confidence":
        # Low confidence everywhere
        mask[:, :] = 0.3
        
    elif mode == "multi_blob":
        # Multiple disconnected regions
        mask[50:100, 50:100] = 0.9
        mask[150:200, 150:200] = 0.85
        
    return mask


def test_farthest_point_sampling_basic():
    """Test farthest-point sampling distributes points spatially."""
    points = np.array([
        [0, 0],
        [0, 1],
        [0, 2],
        [100, 100],
        [200, 200],
    ], dtype=np.float32)
    
    selected = farthest_point_sampling(points, n_samples=3)
    
    assert selected.shape == (3, 2)
    # Should select spatially distributed points, not clusters
    assert len(np.unique(selected, axis=0)) == 3


def test_farthest_point_sampling_deterministic_when_seeded():
    """Sampling should be deterministic with fixed seed."""
    np.random.seed(42)
    points = np.random.rand(100, 2) * 256
    
    np.random.seed(42)
    result1 = farthest_point_sampling(points, n_samples=5)
    
    np.random.seed(42)
    result2 = farthest_point_sampling(points, n_samples=5)
    
    np.testing.assert_array_equal(result1, result2)


def test_generate_prompts_single_blob():
    """Standard case: generate prompts from a clean mask."""
    mask = _synthetic_mask(mode="single_blob")
    cfg = PromptGenerationConfig(
        num_fg_points=4,
        num_bg_points=2,
        min_mask_pixels=100,
    )
    
    fg_points, bg_points, stats = generate_prompts_from_mask(mask, cfg)
    
    assert stats["skip_reason"] is None
    assert fg_points.shape[1] == 2  # (y, x)
    assert bg_points.shape[1] == 2
    
    # Should generate requested number of points
    assert len(fg_points) == 4
    assert len(bg_points) <= 2  # may be fewer if boundary is small
    
    # FG points should be in high-confidence region
    for y, x in fg_points:
        assert mask[int(y), int(x)] > 0.7


def test_generate_prompts_skips_tiny_mask():
    """Should skip when mask is too small."""
    mask = _synthetic_mask(mode="tiny")
    cfg = PromptGenerationConfig(min_mask_pixels=500)
    
    fg_points, bg_points, stats = generate_prompts_from_mask(mask, cfg)
    
    assert stats["skip_reason"] is not None
    assert "mask_too_small" in stats["skip_reason"]
    assert len(fg_points) == 0
    assert len(bg_points) == 0


def test_generate_prompts_skips_low_confidence():
    """Should skip when no high-confidence pixels exist."""
    mask = _synthetic_mask(mode="low_confidence")
    cfg = PromptGenerationConfig(
        fg_confidence_threshold=0.60,
        min_mask_pixels=10,
    )
    
    fg_points, bg_points, stats = generate_prompts_from_mask(mask, cfg)
    
    assert stats["skip_reason"] is not None
    assert len(fg_points) == 0


def test_generate_prompts_spatial_distribution():
    """FG points should be spatially distributed, not clustered."""
    mask = _synthetic_mask(mode="single_blob")
    cfg = PromptGenerationConfig(
        num_fg_points=4,
        enforce_spacing=True,
        min_spacing_pixels=50,
    )
    
    fg_points, _, _ = generate_prompts_from_mask(mask, cfg)
    
    # Compute pairwise distances
    dists = []
    for i in range(len(fg_points)):
        for j in range(i + 1, len(fg_points)):
            dist = np.linalg.norm(fg_points[i] - fg_points[j])
            dists.append(dist)
    
    # At least some pairs should be well-separated
    assert max(dists) > 50


def test_compute_roi_standard_case():
    """Compute ROI from a standard mask."""
    mask = _synthetic_mask(mode="single_blob")
    
    roi, stats = compute_roi_from_mask(mask, padding=10, max_side=1024)
    
    assert stats["skip_reason"] is None
    assert roi is not None
    
    y0, x0, y1, x1 = roi
    assert 0 <= y0 < y1 <= mask.shape[0]
    assert 0 <= x0 < x1 <= mask.shape[1]
    
    # ROI should contain the confident region
    confident_y, confident_x = np.where(mask > 0.5)
    assert y0 <= confident_y.min()
    assert y1 >= confident_y.max()
    assert x0 <= confident_x.min()
    assert x1 >= confident_x.max()


def test_compute_roi_skips_empty_mask():
    """ROI computation should fail gracefully on empty mask."""
    mask = np.zeros((256, 256), dtype=np.float32)
    
    roi, stats = compute_roi_from_mask(mask)
    
    assert roi is None
    assert stats["skip_reason"] == "empty_mask"


def test_compute_roi_skips_oversized():
    """ROI computation should reject when result would be too large."""
    mask = np.ones((6000, 6000), dtype=np.float32)
    
    roi, stats = compute_roi_from_mask(mask, padding=0, max_side=4096)
    
    assert roi is None
    assert "roi_too_large" in stats["skip_reason"]


def test_prompt_generation_config_defaults():
    """Verify default config values are reasonable."""
    cfg = PromptGenerationConfig()
    
    assert cfg.num_fg_points == 4
    assert cfg.num_bg_points == 2
    assert cfg.min_mask_pixels == 500
    assert cfg.enforce_spacing is True
    assert 0.0 < cfg.fg_confidence_threshold < 1.0
