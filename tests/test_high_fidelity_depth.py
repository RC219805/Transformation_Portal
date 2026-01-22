#!/usr/bin/env python3
"""
Tests for High-Fidelity Depth Pipeline
=======================================
"""

import pytest
import numpy as np
from pathlib import Path
import importlib.util

# Skip if cv2 not available (vision dependency)
cv2 = pytest.importorskip("cv2")

from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator, DepthConfig
from high_fidelity_depth.validation import (
    validate_depth_quality,
    detect_edges,
    compute_edge_alignment,
)
from high_fidelity_depth.isolation_tests import run_isolation_tests

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def create_synthetic_image(size: int = 512) -> np.ndarray:
    """Create synthetic RGB image with clear edges."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Add some geometric shapes
    img[100:200, 100:200] = [255, 0, 0]  # Red square
    img[300:400, 300:400] = [0, 255, 0]  # Green square
    img[:, size // 2 - 5 : size // 2 + 5] = [255, 255, 255]  # White vertical line

    return img


def test_depth_config():
    """Test DepthConfig initialization."""
    config = DepthConfig()

    assert config.model_name == "depth-anything/Depth-Anything-V2-Large-hf"
    assert config.tile_size == 1024
    assert config.overlap == 192  # Updated from 128 → 192 for texture-heavy scenes
    assert config.reconcile_scales == True


@pytest.mark.skipif(
    not (HAS_TORCH and HAS_TRANSFORMERS),
    reason="PyTorch and transformers not available",
)
def test_depth_estimator_initialization():
    """Test HighFidelityDepthEstimator initialization."""
    config = DepthConfig()
    estimator = HighFidelityDepthEstimator(config)

    assert estimator.config == config
    assert estimator.device in ["cuda", "mps", "cpu"]


def test_edge_detection():
    """Test edge detection on synthetic image."""
    img = create_synthetic_image(256)
    gray = np.mean(img, axis=2).astype(np.uint8)

    edges = detect_edges(gray)

    assert edges.shape == gray.shape
    assert edges.dtype == np.uint8
    assert (edges == 0).any()  # Has non-edge pixels
    assert (edges > 0).any()  # Has edge pixels


def test_edge_alignment_perfect():
    """Test edge alignment with perfect correlation."""
    size = 256
    edges1 = np.zeros((size, size), dtype=np.uint8)
    edges1[100:200, 100:200] = 255

    edges2 = edges1.copy()

    alignment = compute_edge_alignment(edges1, edges2)

    assert alignment > 0.99  # Should be near-perfect


def test_edge_alignment_random():
    """Test edge alignment with random edges."""
    size = 256
    np.random.seed(42)
    edges1 = (np.random.rand(size, size) > 0.9).astype(np.uint8) * 255
    edges2 = (np.random.rand(size, size) > 0.9).astype(np.uint8) * 255

    alignment = compute_edge_alignment(edges1, edges2)

    assert -0.2 < alignment < 0.2  # Should be near zero


def test_validation_metrics():
    """Test validation metrics on synthetic data."""
    img = create_synthetic_image(256)

    # Create synthetic depth with some noise
    depth = np.random.rand(256, 256).astype(np.float32)
    depth[100:200, 100:200] = 0.8  # Match red square
    depth[300:400, 300:400] = 0.2  # Match green square (inverted)

    metrics = validate_depth_quality(img, depth, dilation=3)

    # Use edge_alignment_corr (correlation metric) - edge_alignment was renamed
    assert 0.0 <= metrics.edge_alignment_corr <= 1.0
    assert 0.0 <= metrics.edge_overlap <= 1.0
    assert metrics.edge_width > 0
    assert metrics.edge_count_ratio > 0


@pytest.mark.skipif(
    not (HAS_TORCH and HAS_TRANSFORMERS),
    reason="PyTorch and transformers not available",
)
def test_tile_extraction():
    """Test tile extraction."""
    config = DepthConfig(tile_size=128, overlap=32)
    estimator = HighFidelityDepthEstimator(config)

    img = create_synthetic_image(256)
    tiles = estimator._extract_tiles(img)

    assert len(tiles) > 0
    for tile, y0, y1, x0, x1 in tiles:
        assert tile.shape[0] <= 128
        assert tile.shape[1] <= 128
        assert y1 > y0
        assert x1 > x0


@pytest.mark.skipif(
    not (HAS_TORCH and HAS_TRANSFORMERS),
    reason="PyTorch and transformers not available",
)
def test_blend_window():
    """Test blend window creation."""
    config = DepthConfig(tile_size=128, overlap=32)
    estimator = HighFidelityDepthEstimator(config)

    window = estimator._make_blend_window(128, 32)

    assert window.shape == (128, 128)
    assert window.min() >= 0.0
    assert window.max() <= 1.0
    # Check edges are ramped
    assert window[0, 64] < window[64, 64]  # Top edge is ramped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
