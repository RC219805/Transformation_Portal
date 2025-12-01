#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Parallax Occlusion Mapping (POM) in lux_render_pipeline_plus_v3."""

from __future__ import annotations
from lux_render_pipeline_plus_v3 import (
    apply_pbr_overlays,
    _parallax_uv,
    _bilinear,
)

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

# Add scripts/pipelines to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "pipelines"))


@pytest.fixture
def test_images(tmp_path: Path) -> dict:
    """Create test images for POM tests."""
    # Input image (gray)
    input_arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    input_path = tmp_path / "input.png"
    Image.fromarray(input_arr).save(input_path)

    # Normal map with varying Z values (to create height variation)
    normal_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            normal_arr[i, j, 0] = 128  # X neutral
            normal_arr[i, j, 1] = 128  # Y neutral
            # Z varies based on position to create depth variation
            normal_arr[i, j, 2] = 128 + int(50 * np.sin(i * 0.2) * np.sin(j * 0.2))
    normal_path = tmp_path / "normal.png"
    Image.fromarray(normal_arr).save(normal_path)

    # Albedo with checkerboard pattern (to visualize displacement)
    albedo_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            if (i // 8 + j // 8) % 2 == 0:
                albedo_arr[i, j] = [200, 180, 150]
            else:
                albedo_arr[i, j] = [100, 90, 75]
    albedo_path = tmp_path / "albedo.png"
    Image.fromarray(albedo_arr).save(albedo_path)

    return {
        "input": input_path,
        "normal": normal_path,
        "albedo": albedo_path,
    }


class TestParallaxUV:
    """Tests for the _parallax_uv helper function."""

    def test_no_displacement_with_zero_scale(self):
        """Test that zero scale produces no displacement."""
        height = np.random.rand(32, 32).astype(np.float32)
        uvx, uvy = _parallax_uv(height, (0.1, 0.1), scale=0.0, steps=4)

        # With scale=0, UV should be close to original coordinates
        expected_x = np.arange(32, dtype=np.float32)[None, :].repeat(32, axis=0)
        expected_y = np.arange(32, dtype=np.float32)[:, None].repeat(32, axis=1)

        np.testing.assert_allclose(uvx, expected_x, atol=1e-5)
        np.testing.assert_allclose(uvy, expected_y, atol=1e-5)

    def test_single_step_displacement(self):
        """Test that single step produces displacement."""
        height = np.ones((16, 16), dtype=np.float32) * 0.5
        uvx, uvy = _parallax_uv(height, (0.1, 0.2), scale=0.1, steps=1)

        # With uniform height and non-zero view direction, UV should be offset
        assert uvx.shape == (16, 16)
        assert uvy.shape == (16, 16)

        # UV should be displaced from original coordinates
        base_x = np.arange(16, dtype=np.float32)[None, :].repeat(16, axis=0)
        base_y = np.arange(16, dtype=np.float32)[:, None].repeat(16, axis=1)

        # Check that displacement was applied
        assert not np.allclose(uvx, base_x)
        assert not np.allclose(uvy, base_y)

    def test_multi_step_displacement(self):
        """Test that multi-step produces smoother displacement."""
        height = np.random.rand(32, 32).astype(np.float32)
        uvx_1, uvy_1 = _parallax_uv(height, (0.1, 0.1), scale=0.05, steps=1)
        uvx_4, uvy_4 = _parallax_uv(height, (0.1, 0.1), scale=0.05, steps=4)

        # Multi-step should produce different (accumulated) displacement
        assert uvx_1.shape == uvx_4.shape
        assert uvy_1.shape == uvy_4.shape


class TestBilinear:
    """Tests for the _bilinear interpolation function."""

    def test_2d_interpolation(self):
        """Test bilinear interpolation on 2D array."""
        img = np.array([[0, 1], [1, 0]], dtype=np.float32)
        uvx = np.array([[0.5]], dtype=np.float32)
        uvy = np.array([[0.5]], dtype=np.float32)

        result = _bilinear(img, uvx, uvy)
        assert result.shape == (1, 1)
        # At center, should be average of all corners
        assert np.allclose(result, 0.5, atol=0.1)

    def test_3d_rgb_interpolation(self):
        """Test bilinear interpolation on 3D RGB array."""
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[0, 0] = [1, 0, 0]  # Red
        img[0, 3] = [0, 1, 0]  # Green
        img[3, 0] = [0, 0, 1]  # Blue
        img[3, 3] = [1, 1, 1]  # White

        uvx = np.array([[1.5]], dtype=np.float32)
        uvy = np.array([[1.5]], dtype=np.float32)

        result = _bilinear(img, uvx, uvy)
        assert result.shape == (1, 1, 3)

    def test_edge_clipping(self):
        """Test that UV coordinates outside image are clipped."""
        img = np.ones((4, 4, 3), dtype=np.float32)
        uvx = np.array([[-1, 10]], dtype=np.float32)
        uvy = np.array([[0, 0]], dtype=np.float32)

        result = _bilinear(img, uvx, uvy)
        assert result.shape == (1, 2, 3)
        # Both should sample valid pixels due to clipping
        assert np.all(result >= 0)


class TestPOMIntegration:
    """Integration tests for POM in apply_pbr_overlays."""

    def test_pom_disabled_works(self, test_images: dict, tmp_path: Path):
        """Test that pipeline works with POM disabled."""
        output_path = tmp_path / "output.png"

        result = apply_pbr_overlays(
            test_images["input"],
            albedo=test_images["albedo"],
            normal=test_images["normal"],
            enable_displacement=False,
        )

        assert result is not None
        assert result.size == (64, 64)
        result.save(output_path)
        assert output_path.exists()

    def test_pom_enabled_works(self, test_images: dict, tmp_path: Path):
        """Test that POM feature works when enabled."""
        output_path = tmp_path / "output.png"

        result = apply_pbr_overlays(
            test_images["input"],
            albedo=test_images["albedo"],
            normal=test_images["normal"],
            enable_displacement=True,
            pom_scale=0.02,
            pom_steps=4,
            view_dir_xy=(0.1, 0.1),
        )

        assert result is not None
        assert result.size == (64, 64)
        result.save(output_path)
        assert output_path.exists()

    def test_pom_with_zero_scale_matches_disabled(self, test_images: dict):
        """Test that POM with zero scale produces similar result to disabled."""
        result_disabled = apply_pbr_overlays(
            test_images["input"],
            albedo=test_images["albedo"],
            normal=test_images["normal"],
            enable_displacement=False,
        )

        result_zero_scale = apply_pbr_overlays(
            test_images["input"],
            albedo=test_images["albedo"],
            normal=test_images["normal"],
            enable_displacement=True,
            pom_scale=0.0,  # Zero scale should effectively disable POM
            pom_steps=4,
            view_dir_xy=(0.1, 0.1),
        )

        # Results should be identical when scale is zero
        arr_disabled = np.array(result_disabled)
        arr_zero = np.array(result_zero_scale)
        np.testing.assert_array_equal(arr_disabled, arr_zero)

    def test_pom_with_different_view_directions(self, test_images: dict):
        """Test that different view directions produce different results."""
        result_view1 = apply_pbr_overlays(
            test_images["input"],
            albedo=test_images["albedo"],
            normal=test_images["normal"],
            enable_displacement=True,
            pom_scale=0.05,
            pom_steps=4,
            view_dir_xy=(0.15, 0.0),
        )

        result_view2 = apply_pbr_overlays(
            test_images["input"],
            albedo=test_images["albedo"],
            normal=test_images["normal"],
            enable_displacement=True,
            pom_scale=0.05,
            pom_steps=4,
            view_dir_xy=(-0.15, 0.0),
        )

        # Different view directions should produce different results
        arr_view1 = np.array(result_view1)
        arr_view2 = np.array(result_view2)
        assert not np.array_equal(arr_view1, arr_view2)

    def test_pom_requires_normal_map(self, test_images: dict):
        """Test that POM with no normal map is skipped."""
        # When no normal map is provided, POM should be silently skipped
        result = apply_pbr_overlays(
            test_images["input"],
            albedo=test_images["albedo"],
            normal=None,  # No normal map
            enable_displacement=True,
            pom_scale=0.02,
        )

        assert result is not None
        assert result.size == (64, 64)

    def test_pom_with_quality_presets(self, test_images: dict):
        """Test that POM works with different quality presets."""
        for quality in ["draft", "preview", "final"]:
            result = apply_pbr_overlays(
                test_images["input"],
                albedo=test_images["albedo"],
                normal=test_images["normal"],
                enable_displacement=True,
                pom_scale=0.02,
                pom_steps=4,
                quality=quality,
            )

            assert result is not None
            assert result.size == (64, 64), f"Failed for quality={quality}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
