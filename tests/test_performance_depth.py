#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance benchmarks for depth processing pipeline.

Tests using pytest-benchmark to measure depth estimation performance.
These tests are discovered by the performance-monitor.yml workflow.

Requirements:
    pip install pytest-benchmark

Local usage:
    pytest tests/test_performance_depth.py -k performance --benchmark-only
    
CI usage (performance-monitor.yml):
    pytest tests/ -k performance --benchmark-only --benchmark-json=results.json
"""
import numpy as np
import pytest

# Conditional imports for optional dependencies
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Skip all tests if PIL not available
pytestmark = pytest.mark.skipif(not HAS_PIL, reason="PIL not available")

# Pillow version compatibility: Image.Resampling.* only exists in Pillow 10+
# Fallback to Image.* constants for older versions
_RESAMPLING = getattr(Image, "Resampling", Image)
LANCZOS = getattr(_RESAMPLING, "LANCZOS", getattr(Image, "LANCZOS", 1))
BILINEAR = getattr(_RESAMPLING, "BILINEAR", getattr(Image, "BILINEAR", 2))


@pytest.fixture
def benchmark_fallback(request):
    """Fallback benchmark fixture for local testing without pytest-benchmark.
    
    Returns a simple callable that executes the function once.
    """
    try:
        # Try to get the real benchmark fixture if pytest-benchmark is installed
        return request.getfixturevalue('benchmark')
    except Exception:
        # Fallback: just execute once
        def fake_benchmark(func, *args, **kwargs):
            return func(*args, **kwargs)
        return fake_benchmark


@pytest.fixture
def synthetic_image():
    """Create synthetic test image for benchmarking."""
    # 512x512 RGB image (small enough for fast CI)
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img)


@pytest.fixture
def synthetic_image_hd():
    """Create HD synthetic test image for benchmarking."""
    # 1920x1080 RGB image
    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    return Image.fromarray(img)


class TestDepthProcessingPerformance:
    """Benchmark tests for depth processing operations."""

    @pytest.mark.performance
    def test_image_loading_performance(self, benchmark_fallback, tmp_path, synthetic_image):
        """Benchmark image loading from disk."""
        # Save image to temporary file
        img_path = tmp_path / "test_image.png"
        synthetic_image.save(img_path)
        
        # Benchmark loading (with proper file handle closing)
        def load_image():
            with Image.open(img_path) as im:
                return im.copy()  # Force load and close file handle
        
        result = benchmark_fallback(load_image)
        assert result is not None
        assert result.size == (512, 512)

    @pytest.mark.performance
    def test_image_resize_performance(self, benchmark_fallback, synthetic_image):
        """Benchmark image resizing operation."""
        target_size = (384, 384)
        
        def resize_image():
            return synthetic_image.resize(target_size, LANCZOS)
        
        result = benchmark_fallback(resize_image)
        assert result.size == target_size

    @pytest.mark.performance
    def test_numpy_conversion_performance(self, benchmark_fallback, synthetic_image):
        """Benchmark PIL to NumPy conversion."""
        def convert_to_numpy():
            return np.array(synthetic_image)
        
        result = benchmark_fallback(convert_to_numpy)
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    @pytest.mark.performance
    def test_normalize_performance(self, benchmark_fallback, synthetic_image):
        """Benchmark image normalization (common preprocessing step)."""
        arr = np.array(synthetic_image)
        
        def normalize():
            # Standard normalization: [0, 255] -> [0, 1]
            return arr.astype(np.float32) / 255.0
        
        result = benchmark_fallback(normalize)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    @pytest.mark.performance
    @pytest.mark.slow
    def test_hd_image_processing_performance(self, benchmark_fallback, synthetic_image_hd):
        """Benchmark HD image processing (tagged as slow for optional execution)."""
        def process_hd():
            # Simulate typical preprocessing pipeline
            arr = np.array(synthetic_image_hd)
            normalized = arr.astype(np.float32) / 255.0
            resized = Image.fromarray((normalized * 255).astype(np.uint8)).resize(
                (512, 512), LANCZOS
            )
            return np.array(resized)
        
        result = benchmark_fallback(process_hd)
        assert result.shape == (512, 512, 3)


class TestMaterialSegmentationPerformance:
    """Benchmark tests for material segmentation (if available)."""

    @pytest.mark.performance
    def test_color_clustering_performance(self, benchmark_fallback, synthetic_image):
        """Benchmark basic color clustering (material segmentation proxy)."""
        arr = np.array(synthetic_image)
        
        def cluster_colors():
            # Simple k-means-like operation on colors
            # Flatten to (N, 3) for color analysis
            pixels = arr.reshape(-1, 3)
            # Simple binning as proxy for clustering
            bins = pixels // 32  # 8 bins per channel (8^3 = 512 colors)
            return bins
        
        result = benchmark_fallback(cluster_colors)
        assert result.shape[0] == 512 * 512

    @pytest.mark.performance
    def test_edge_detection_performance(self, benchmark_fallback, synthetic_image):
        """Benchmark edge detection (used in material segmentation)."""
        arr = np.array(synthetic_image)
        gray = arr.mean(axis=2).astype(np.uint8)
        
        def sobel_edges():
            # Simple Sobel-like edge detection
            gx = np.diff(gray, axis=1)
            gy = np.diff(gray, axis=0)
            return gx, gy
        
        result = benchmark_fallback(sobel_edges)
        assert len(result) == 2


class TestUpscalingPerformance:
    """Benchmark tests for upscaling operations."""

    @pytest.mark.performance
    def test_bilinear_upscale_2x_performance(self, benchmark_fallback, synthetic_image):
        """Benchmark 2x bilinear upscaling."""
        target_size = (1024, 1024)
        
        def upscale_bilinear():
            return synthetic_image.resize(target_size, Image.Resampling.BILINEAR)
        
        result = benchmark_fallback(upscale_bilinear)
        assert result.size == target_size

    @pytest.mark.performance
    def test_lanczos_upscale_2x_performance(self, benchmark_fallback, synthetic_image):
        """Benchmark 2x Lanczos upscaling (higher quality, slower)."""
        target_size = (1024, 1024)
        
        def upscale_lanczos():
            return synthetic_image.resize(target_size, LANCZOS)
        
        result = benchmark_fallback(upscale_lanczos)
        assert result.size == target_size
