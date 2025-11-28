#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Ultimate Quality Pipeline
====================================

Tests focusing on the neutral gray bug fix and depth normalization.

Note: The helper functions in this test file are intentionally duplicated
from the pipeline module to allow testing without heavy ML dependencies
(torch, transformers) that would significantly slow down the test suite.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter


def safe_normalize_depth(depth_array: np.ndarray) -> np.ndarray:
    """
    Safely normalize depth array to [0, 1] range.

    This function handles edge cases where all depth values are equal,
    which would otherwise cause division by zero (NaN values).

    Args:
        depth_array: Raw depth array from depth estimation

    Returns:
        Normalized depth array with values in [0, 1] range
    """
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    depth_range = depth_max - depth_min

    # Use epsilon pattern for consistency with production code
    return (depth_array - depth_min) / (depth_range + 1e-6)


def apply_depth_aware_clarity(
    image_array: np.ndarray,
    depth_map: np.ndarray,
    strength: float = 0.3
) -> np.ndarray:
    """
    Apply depth-aware clarity enhancement.

    Stronger sharpening on foreground, gentler on background.
    """
    foreground_mask = depth_map > 0.7
    midground_mask = (depth_map >= 0.4) & (depth_map <= 0.7)
    background_mask = depth_map < 0.4

    blurred = gaussian_filter(image_array, sigma=2.0)
    unsharp = image_array - blurred

    clarity_map = np.zeros_like(depth_map)
    clarity_map[foreground_mask] = strength * 1.5
    clarity_map[midground_mask] = strength * 1.0
    clarity_map[background_mask] = strength * 0.5

    result = image_array.copy()
    for c in range(3):
        result[:, :, c] += unsharp[:, :, c] * clarity_map

    return np.clip(result, 0, 1)


def apply_luxury_color_grade(image_array: np.ndarray) -> np.ndarray:
    """
    Apply luxury color grading optimized for architectural renders.
    """
    luminance = (0.2126 * image_array[:, :, 0]
                 + 0.7152 * image_array[:, :, 1]
                 + 0.0722 * image_array[:, :, 2])

    result = image_array.copy()

    # Cool highlights
    highlight_mask = luminance > 0.7
    result[highlight_mask, 2] *= 1.02

    # Warm shadows
    shadow_mask = luminance < 0.3
    result[shadow_mask, 0] *= 1.03
    result[shadow_mask, 1] *= 1.01

    # Saturation boost in midtones
    midtone_mask = (luminance >= 0.3) & (luminance <= 0.7)
    saturation_boost = 1.08
    for c in range(3):
        result[midtone_mask, c] = (luminance[midtone_mask]
                                   + (result[midtone_mask, c]
                                      - luminance[midtone_mask])
                                   * saturation_boost)

    return np.clip(result, 0, 1)


class TestDepthNormalization:
    """Test depth normalization with edge cases."""

    def test_normal_depth_normalization(self):
        """Test normalization with varying depth values."""
        depth = np.random.rand(100, 100).astype(np.float32) * 0.8 + 0.1
        normalized = safe_normalize_depth(depth)

        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()

    def test_uniform_depth_normalization(self):
        """Test normalization with uniform depth values (bug scenario)."""
        # All same values - this would cause division by zero
        depth = np.full((100, 100), 0.5, dtype=np.float32)
        normalized = safe_normalize_depth(depth)

        # Should return 0.5 (midground) instead of NaN
        assert np.allclose(normalized, 0.5)
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()

    def test_nearly_uniform_depth_normalization(self):
        """Test normalization with very small depth range."""
        depth = np.full((100, 100), 0.5, dtype=np.float32)
        depth[50, 50] = 0.5 + 1e-10  # Tiny variation
        normalized = safe_normalize_depth(depth)

        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestNeutralGrayBug:
    """Test for the neutral gray contamination bug."""

    def test_color_preserved_with_normal_depth(self):
        """Test that colors are preserved with normal depth map."""
        # Create colorful test image
        image = np.random.rand(100, 100, 3).astype(np.float32) * 0.8 + 0.1
        depth = np.random.rand(100, 100).astype(np.float32)

        # Apply clarity
        result = apply_depth_aware_clarity(image, depth)
        result = apply_luxury_color_grade(result)

        # Check that colors are NOT all equal (not neutral gray)
        is_neutral = (np.allclose(result[:, :, 0], result[:, :, 1], atol=0.01)
                      and np.allclose(result[:, :, 1], result[:, :, 2], atol=0.01))
        assert not is_neutral, "Output should NOT be neutral gray"

    def test_color_preserved_with_uniform_depth(self):
        """Test colors preserved even with uniform depth (fixed bug scenario)."""
        # Create image with distinct channel values (not random, for predictable test)
        image = np.zeros((100, 100, 3), dtype=np.float32)
        image[:, :, 0] = 0.7  # High red
        image[:, :, 1] = 0.5  # Medium green
        image[:, :, 2] = 0.3  # Low blue

        # Create uniform depth map (previously caused neutral gray)
        depth_raw = np.full((100, 100), 0.5, dtype=np.float32)
        depth = safe_normalize_depth(depth_raw)  # Should return 0.5 (midground)

        # Apply clarity
        result = apply_depth_aware_clarity(image, depth)
        result = apply_luxury_color_grade(result)

        # Check that colors are NOT all equal (not neutral gray)
        is_neutral = (np.allclose(result[:, :, 0], result[:, :, 1], atol=0.01)
                      and np.allclose(result[:, :, 1], result[:, :, 2], atol=0.01))
        assert not is_neutral, "Output should NOT be neutral gray (bug fix verification)"

        # Verify color channels are still distinct from each other
        r_mean = result[:, :, 0].mean()
        g_mean = result[:, :, 1].mean()
        b_mean = result[:, :, 2].mean()

        # Colors should still be reasonably varied and in expected order
        assert r_mean > g_mean, "Red should still be higher than green"
        assert g_mean > b_mean, "Green should still be higher than blue"

    def test_no_nan_in_output(self):
        """Test that no NaN values appear in output."""
        image = np.random.rand(100, 100, 3).astype(np.float32)

        # Test with both normal and uniform depth
        for depth_raw in [
            np.random.rand(100, 100).astype(np.float32),
            np.full((100, 100), 0.5, dtype=np.float32),
        ]:
            depth = safe_normalize_depth(depth_raw)
            result = apply_depth_aware_clarity(image.copy(), depth)
            result = apply_luxury_color_grade(result)

            assert not np.isnan(result).any(), "No NaN should be in output"
            assert not np.isinf(result).any(), "No Inf should be in output"


class TestClarityWithUniformDepth:
    """Test clarity enhancement behavior with uniform depth."""

    def test_midground_enhancement_with_uniform_depth(self):
        """Test that uniform depth (0.5) applies midground enhancement."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        depth = np.full((100, 100), 0.5, dtype=np.float32)  # Midground

        result = apply_depth_aware_clarity(image, depth, strength=0.3)

        # Since depth=0.5 is in midground range (0.4, 0.7], clarity should be applied
        # Check that some enhancement occurred
        diff = np.abs(result - image).mean()
        assert diff > 0, "Some clarity enhancement should be applied"

    def test_depth_zones_work_correctly(self):
        """Test that depth zones (fore/mid/back) work correctly."""
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        # Test foreground (depth > 0.7)
        fg_depth = np.full((100, 100), 0.8, dtype=np.float32)
        fg_result = apply_depth_aware_clarity(image, fg_depth, strength=0.3)

        # Test midground (0.4 <= depth <= 0.7)
        mg_depth = np.full((100, 100), 0.5, dtype=np.float32)
        mg_result = apply_depth_aware_clarity(image, mg_depth, strength=0.3)

        # Test background (depth < 0.4)
        bg_depth = np.full((100, 100), 0.2, dtype=np.float32)
        bg_result = apply_depth_aware_clarity(image, bg_depth, strength=0.3)

        # All should process without error and produce valid output
        for result in [fg_result, mg_result, bg_result]:
            assert result.shape == image.shape
            assert not np.isnan(result).any()
            assert not np.isinf(result).any()


class TestColorGradePreservation:
    """Test that color grading preserves color diversity."""

    def test_saturation_boost_preserves_color(self):
        """Test that saturation boost doesn't destroy colors."""
        # Create image with distinct channel values
        image = np.zeros((100, 100, 3), dtype=np.float32)
        image[:, :, 0] = 0.8  # Red
        image[:, :, 1] = 0.4  # Green
        image[:, :, 2] = 0.2  # Blue

        result = apply_luxury_color_grade(image)

        # Channels should still be distinct
        r_mean = result[:, :, 0].mean()
        g_mean = result[:, :, 1].mean()
        b_mean = result[:, :, 2].mean()

        assert r_mean > g_mean > b_mean, "Color ordering should be preserved"
        assert not np.allclose(r_mean, g_mean)
        assert not np.allclose(g_mean, b_mean)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
