"""Tests for PBR map generation."""

import numpy as np

from transformation_portal.lux_depth_v3.pbr import (
    PBRConfig,
    generate_pbr_maps,
)


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


def test_pbr_output_shapes_match_input():
    """CRITICAL: Output maps must have SAME dimensions as input depth.

    This test catches the integral image padding bug.
    """
    test_cases = [
        (256, 256),
        (512, 512),
        (128, 256),  # Non-square
        (100, 100),  # Small
    ]

    config = PBRConfig(
        normal_blur_radius=3,
        roughness_blur_radius=5,
        ao_blur_radius=7,
    )

    for h, w in test_cases:
        depth = np.random.rand(h, w).astype(np.float32)

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
