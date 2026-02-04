"""Tests for PBR map generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.pbr import (
    PBRConfig,
    generate_pbr_maps,
)
from transformation_portal.lux_depth_v3.pbr_writer import write_pbr_maps


def test_pbr_flat_depth_normal_is_up():
    """Flat depth surface should produce Z-up normals (blue: RGB = 128, 128, 255)."""
    # Create perfectly flat depth map
    depth = np.full((256, 256), 0.5, dtype=np.float32)

    config = PBRConfig(normal_strength=1.0, normal_blur_radius=0)
    normal_map, _, _ = generate_pbr_maps(depth, config)

    # Flat surface: gradients are zero, so normal = (0, 0, 1)
    # Encoded as RGB: (0+1)*127.5 = 127.5 ≈ 128 for X and Y
    # (1+1)*127.5 = 255 for Z

    # Check central region (avoid edge artifacts)
    center = normal_map[64:192, 64:192]

    # X and Y channels should be near 128 (neutral)
    assert np.abs(center[:, :, 0].mean() - 128) < 5, "X channel should be ~128 (neutral)"
    assert np.abs(center[:, :, 1].mean() - 128) < 5, "Y channel should be ~128 (neutral)"

    # Z channel should be near 255 (pointing up)
    assert center[:, :, 2].mean() > 250, "Z channel should be ~255 (up vector)"


@pytest.mark.parametrize(
    "shape",
    [
        (256, 256),
        (512, 512),
        (128, 256),  # Non-square
        (100, 100),  # Small
        (1024, 768),  # HD-like
    ],
)
def test_pbr_output_shapes_match_input(shape):
    """CRITICAL: Output maps must have SAME dimensions as input depth.

    This test catches the integral image padding bug.
    """
    h, w = shape
    depth = np.random.rand(h, w).astype(np.float32)

    config = PBRConfig(
        normal_blur_radius=3,
        roughness_blur_radius=5,
        ao_blur_radius=7,
    )

    normal_map, roughness_map, ao_map = generate_pbr_maps(depth, config)

    # Check shapes
    assert normal_map.shape == (h, w, 3), f"Normal map shape mismatch for ({h}, {w})"
    assert roughness_map.shape == (h, w), f"Roughness map shape mismatch for ({h}, {w})"
    assert ao_map.shape == (h, w), f"AO map shape mismatch for ({h}, {w})"


def test_pbr_outputs_uint8():
    """All PBR maps should be uint8 in range [0, 255]."""
    depth = np.random.rand(128, 128).astype(np.float32)

    normal_map, roughness_map, ao_map = generate_pbr_maps(depth)

    # Check data types
    assert normal_map.dtype == np.uint8
    assert roughness_map.dtype == np.uint8
    assert ao_map.dtype == np.uint8

    # Check value ranges
    assert normal_map.min() >= 0 and normal_map.max() <= 255
    assert roughness_map.min() >= 0 and roughness_map.max() <= 255
    assert ao_map.min() >= 0 and ao_map.max() <= 255


def test_pbr_ao_independent_of_normal_strength():
    """CRITICAL: AO must be independent of normal_strength parameter.

    This test validates the decoupling fix - AO uses raw gradients,
    not scaled by normal_strength.
    """
    depth = np.random.rand(256, 256).astype(np.float32)

    # Generate AO with different normal_strength values
    config_weak = PBRConfig(normal_strength=0.5, ao_strength=1.0, ao_blur_radius=5)
    config_strong = PBRConfig(normal_strength=2.0, ao_strength=1.0, ao_blur_radius=5)

    _, _, ao_weak = generate_pbr_maps(depth, config_weak)
    _, _, ao_strong = generate_pbr_maps(depth, config_strong)

    # AO maps should be IDENTICAL despite different normal_strength
    np.testing.assert_array_equal(ao_weak, ao_strong, err_msg="AO must be independent of normal_strength")


def test_pbr_input_validation_2d():
    """Reject non-2D depth inputs."""
    depth_1d = np.random.rand(256)
    depth_3d = np.random.rand(256, 256, 3)

    with pytest.raises(ValueError, match="Depth must be 2D"):
        generate_pbr_maps(depth_1d)

    with pytest.raises(ValueError, match="Depth must be 2D"):
        generate_pbr_maps(depth_3d)


def test_pbr_input_validation_nan_inf():
    """Reject depth with NaN or Inf values."""
    depth_nan = np.full((256, 256), np.nan, dtype=np.float32)
    depth_inf = np.full((256, 256), np.inf, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN or Inf"):
        generate_pbr_maps(depth_nan)

    with pytest.raises(ValueError, match="NaN or Inf"):
        generate_pbr_maps(depth_inf)


def test_pbr_ao_bias_effect():
    """Verify ao_bias parameter controls AO brightness floor."""
    depth = np.random.rand(256, 256).astype(np.float32)

    # Low bias (darker AO allowed)
    config_low = PBRConfig(ao_bias=0.0, ao_strength=1.0)
    _, _, ao_low = generate_pbr_maps(depth, config_low)

    # High bias (AO clamped to brighter values)
    config_high = PBRConfig(ao_bias=0.8, ao_strength=1.0)
    _, _, ao_high = generate_pbr_maps(depth, config_high)

    # High bias should produce brighter AO overall
    assert ao_high.mean() > ao_low.mean(), "Higher ao_bias should produce brighter AO"
    assert ao_high.min() >= 0.8 * 255 - 5, "High bias should clamp minimum AO brightness"


def test_write_pbr_maps_creates_pngs_and_no_tmp_files():
    """Verify write_pbr_maps creates PNGs atomically without temp file leaks."""
    depth = np.random.rand(128, 128).astype(np.float32)
    normal_map, roughness_map, ao_map = generate_pbr_maps(depth)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write PBR maps
        paths = write_pbr_maps(normal_map, roughness_map, ao_map, output_dir, "test_render")

        # Verify all maps were written
        assert "normal" in paths
        assert "roughness" in paths
        assert "ao" in paths

        # Verify files exist
        assert paths["normal"].exists()
        assert paths["roughness"].exists()
        assert paths["ao"].exists()

        # Verify no temp files remain (new atomic write helper should clean up)
        temp_files = list(output_dir.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Found temp files: {temp_files}"

        # Verify file names
        assert paths["normal"].name == "test_render_normal.png"
        assert paths["roughness"].name == "test_render_roughness.png"
        assert paths["ao"].name == "test_render_ao.png"


def test_write_pbr_maps_partial_failure_cleanup():
    """Verify that even with partial failures, no temp files are left behind."""
    depth = np.random.rand(128, 128).astype(np.float32)
    normal_map, roughness_map, ao_map = generate_pbr_maps(depth)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create invalid map to trigger failure for one map
        # (but valid maps should still succeed)
        invalid_map = np.zeros((128, 128, 5), dtype=np.uint8)  # Invalid shape

        try:
            # This should partially succeed (normal is invalid, but roughness and ao should work)
            write_pbr_maps(invalid_map, roughness_map, ao_map, output_dir, "partial_test")
        except Exception:
            # Some failure expected due to invalid normal map
            pass

        # Verify no temp files remain even after partial failure
        temp_files = list(output_dir.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Temp files leaked after partial failure: {temp_files}"
