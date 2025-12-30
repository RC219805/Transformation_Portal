#!/usr/bin/env python3
"""Tests for conditional inference resolution policy."""

import numpy as np
import pytest
from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator
from high_fidelity_depth.depth_estimator import DepthConfig


def test_patch_multiple_computation():
    """Verify target sizes are multiples of patch size (14)."""
    config = DepthConfig()
    estimator = HighFidelityDepthEstimator(config)

    test_cases = [
        # (original_shape, input_size, (expected_h, expected_w))
        ((512, 512), 1022, (1022, 1022)),  # Small square: both get same size
        ((1920, 1080), 518, (910, 518)),  # HD landscape: width=518 (shorter), height scaled
        ((1080, 1920), 518, (518, 910)),  # HD portrait: height=518 (shorter), width scaled
        ((4000, 3000), 518, (686, 518)),  # Large landscape: width=518 (shorter), height scaled
    ]

    for (h, w), input_size, expected in test_cases:
        target_h, target_w = estimator._compute_target_size(h, w, input_size)

        # Check multiples of 14
        assert target_h % 14 == 0, f"Height {target_h} not multiple of 14"
        assert target_w % 14 == 0, f"Width {target_w} not multiple of 14"

        # Check expected values (allow some tolerance for rounding)
        assert target_h == expected[0], f"Expected height {expected[0]}, got {target_h}"
        assert target_w == expected[1], f"Expected width {expected[1]}, got {target_w}"

        # Verify shortest side matches input_size (rounded to patch multiple)
        expected_short = (input_size // 14) * 14
        actual_short = min(target_h, target_w)
        assert actual_short == expected_short, f"Shortest side should be {expected_short}, got {actual_short}"


def test_small_image_policy():
    """Verify small images trigger high input_size."""
    config = DepthConfig()
    estimator = HighFidelityDepthEstimator(config)

    # Small image (512×512)
    small_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    preprocessed, metadata = estimator.preprocess_for_inference(small_image)

    assert metadata["policy"] == "small_image_boost"
    assert metadata["input_size"] == 1022  # High resolution
    assert metadata["original_shape"] == (512, 512)

    # Large image (2000×2000)
    large_image = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
    preprocessed, metadata = estimator.preprocess_for_inference(large_image)

    assert metadata["policy"] == "default"
    assert metadata["input_size"] == 518  # Default resolution


def test_aspect_ratio_preserved():
    """Verify aspect ratio is preserved during preprocessing."""
    config = DepthConfig()
    estimator = HighFidelityDepthEstimator(config)

    # Wide image (1920×1080)
    wide_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    preprocessed, metadata = estimator.preprocess_for_inference(wide_image)

    original_aspect = 1920 / 1080
    preprocessed_aspect = preprocessed.shape[1] / preprocessed.shape[0]

    # Allow 5% tolerance for patch alignment (rounding both dimensions can shift aspect ratio)
    aspect_error = abs(preprocessed_aspect - original_aspect) / original_aspect
    assert aspect_error < 0.05, f"Aspect ratio error {aspect_error:.3f} exceeds 5% tolerance"


def test_roundtrip_dimension_preservation():
    """Verify preprocess → postprocess preserves original dimensions."""
    config = DepthConfig()
    estimator = HighFidelityDepthEstimator(config)

    test_sizes = [(512, 512), (1920, 1080), (4000, 3000), (1023, 1023)]

    for h, w in test_sizes:
        image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Preprocess
        preprocessed, metadata = estimator.preprocess_for_inference(image)

        # Simulate depth inference (random depth map)
        depth = np.random.rand(*preprocessed.shape[:2]).astype(np.float32)

        # Postprocess
        depth_restored = estimator.postprocess_depth(depth, metadata)

        # Verify original dimensions restored
        assert depth_restored.shape == (h, w), f"Expected {(h, w)}, got {depth_restored.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
