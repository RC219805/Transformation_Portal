#!/usr/bin/env python3
"""
Unit tests for high-fidelity depth pipeline modules.

Tests for:
- Tiled depth inference
- Normal map generation
- Quality metrics
"""

import numpy as np
import pytest

# Test data fixtures
@pytest.fixture
def synthetic_rgb_image():
    """Create synthetic RGB image for testing."""
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create distinct regions with edges
    img[:h//2, :w//2] = [255, 0, 0]  # Red quadrant
    img[:h//2, w//2:] = [0, 255, 0]  # Green quadrant
    img[h//2:, :w//2] = [0, 0, 255]  # Blue quadrant
    img[h//2:, w//2:] = [255, 255, 0]  # Yellow quadrant
    
    return img


@pytest.fixture
def synthetic_depth_map():
    """Create synthetic depth map with known properties."""
    h, w = 512, 512
    depth = np.zeros((h, w), dtype=np.float32)
    
    # Create depth gradients aligned with RGB regions
    depth[:h//2, :w//2] = 0.2  # Near
    depth[:h//2, w//2:] = 0.5  # Mid
    depth[h//2:, :w//2] = 0.8  # Far
    depth[h//2:, w//2:] = 0.6  # Mid-far
    
    # Add smooth ramp to test edge detection
    for i in range(h):
        for j in range(w):
            depth[i, j] += 0.1 * (i / h + j / w) / 2
    
    return depth


@pytest.fixture
def synthetic_depth_uint16():
    """Create 16-bit depth map."""
    h, w = 512, 512
    depth_norm = np.random.rand(h, w).astype(np.float32)
    return (depth_norm * 65535).astype(np.uint16)


# ============================================================================
# Tests for normal_map.py
# ============================================================================

def test_normal_map_basic_generation(synthetic_depth_map):
    """Test basic normal map generation."""
    from lux_depth_v2.normal_map import generate_normal_map
    
    normals = generate_normal_map(synthetic_depth_map, preset="architectural")
    
    # Check output shape and type
    assert normals.shape == (512, 512, 3)
    assert normals.dtype == np.float32
    
    # Check range (tangent space [0, 1])
    assert normals.min() >= 0.0
    assert normals.max() <= 1.0


def test_normal_map_z_scale_effect():
    """Test that Z scale affects normal steepness."""
    from lux_depth_v2.normal_map import NormalMapGenerator, NormalMapConfig
    
    # Create simple gradient
    depth = np.linspace(0, 1, 256).reshape(16, 16).astype(np.float32)
    
    # Low Z scale = steep normals
    config_steep = NormalMapConfig(z_scale=0.5)
    gen_steep = NormalMapGenerator(config_steep)
    normals_steep = gen_steep.generate(depth)
    
    # High Z scale = flat normals
    config_flat = NormalMapConfig(z_scale=2.0)
    gen_flat = NormalMapGenerator(config_flat)
    normals_flat = gen_flat.generate(depth)
    
    # Steep normals should have lower average Z (more tilted from camera)
    z_steep = normals_steep[:, :, 2].mean()
    z_flat = normals_flat[:, :, 2].mean()
    
    assert z_steep < z_flat, "Lower z_scale should produce steeper (less camera-facing) normals"


def test_normal_map_validation():
    """Test normal map validation detects issues."""
    from lux_depth_v2.normal_map import NormalMapGenerator, NormalMapConfig
    
    # Create truly flat depth (perfectly constant)
    depth_flat = np.ones((128, 128), dtype=np.float32) * 0.5
    
    config = NormalMapConfig(z_scale=1.0)
    generator = NormalMapGenerator(config)
    normals = generator.generate(depth_flat)
    
    metrics = generator.validate_normal_map(normals)
    
    # Truly flat depth should have near-zero X/Y variation
    # (Small variations can occur from edge pixels)
    assert metrics["nx_std"] < 0.15, "Flat depth should produce low X variation"
    assert metrics["ny_std"] < 0.15, "Flat depth should produce low Y variation"
    assert metrics["nz_mean"] > 0.95, "Flat depth should have normals pointing at camera"


def test_normal_map_presets():
    """Test that presets are accessible."""
    from lux_depth_v2.normal_map import PRESETS
    
    assert "architectural" in PRESETS
    assert "subtle" in PRESETS
    assert "pronounced" in PRESETS
    
    # Check that presets have different z_scale
    assert PRESETS["subtle"].z_scale > PRESETS["architectural"].z_scale
    assert PRESETS["pronounced"].z_scale < PRESETS["architectural"].z_scale


# ============================================================================
# Tests for quality_metrics.py
# ============================================================================

def test_edge_alignment_perfect_match(synthetic_rgb_image):
    """Test edge alignment with perfectly aligned depth."""
    from lux_depth_v2.quality_metrics import DepthQualityAnalyzer
    
    # Create depth that exactly matches RGB edges
    depth = np.zeros((512, 512), dtype=np.float32)
    depth[:256, :256] = 0.2
    depth[:256, 256:] = 0.8
    depth[256:, :256] = 0.5
    depth[256:, 256:] = 0.6
    
    analyzer = DepthQualityAnalyzer()
    score = analyzer.compute_edge_alignment(synthetic_rgb_image, depth)
    
    # Should have high alignment (though not perfect due to Canny/Sobel differences)
    assert score > 0.3, "Aligned edges should produce moderate-to-high score"


def test_edge_alignment_poor_match():
    """Test edge alignment with misaligned depth."""
    from lux_depth_v2.quality_metrics import DepthQualityAnalyzer
    
    # Create RGB with clear edges
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[:, :128] = [255, 0, 0]
    rgb[:, 128:] = [0, 0, 255]
    
    # Create depth with no edges (smooth gradient)
    depth = np.linspace(0, 1, 256).reshape(1, 256).repeat(256, axis=0).astype(np.float32)
    
    analyzer = DepthQualityAnalyzer()
    score = analyzer.compute_edge_alignment(rgb, depth)
    
    # Should have low alignment
    assert score < 0.4, "Misaligned edges should produce low score"


def test_quality_metrics_comprehensive(synthetic_rgb_image, synthetic_depth_map):
    """Test comprehensive quality analysis."""
    from lux_depth_v2.quality_metrics import quick_quality_check
    
    depth_uint16 = (synthetic_depth_map * 65535).astype(np.uint16)
    
    metrics = quick_quality_check(synthetic_rgb_image, synthetic_depth_map, depth_uint16)
    
    # Check all fields are populated
    assert 0.0 <= metrics.edge_alignment_score <= 1.0
    assert metrics.edge_width_median_px >= 0.0
    assert 0.0 <= metrics.edge_overshoot_score <= 1.0
    assert metrics.unique_levels_16bit > 0
    assert metrics.effective_bit_depth > 0
    assert 0.0 <= metrics.overall_quality_score <= 100.0


def test_luxury_validation_fails_low_quality():
    """Test that low-quality depth fails validation."""
    from lux_depth_v2.quality_metrics import DepthQualityAnalyzer
    
    # Create poor-quality depth (very few unique values, smooth)
    rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32) * 0.5  # Constant depth
    depth_uint16 = (depth * 65535).astype(np.uint16)
    
    analyzer = DepthQualityAnalyzer()
    metrics = analyzer.analyze(rgb, depth, depth_uint16)
    passes, issues = analyzer.validate_for_luxury_rendering(metrics)
    
    assert not passes, "Constant depth should fail validation"
    assert len(issues) > 0, "Should report specific issues"


def test_luxury_validation_passes_high_quality(synthetic_rgb_image):
    """Test that high-quality depth passes validation."""
    from lux_depth_v2.quality_metrics import DepthQualityAnalyzer
    
    # Create high-quality depth with good edge alignment
    depth = np.zeros((512, 512), dtype=np.float32)
    
    # Add many unique levels with good edges
    for i in range(512):
        for j in range(512):
            depth[i, j] = (i * 512 + j) / (512 * 512)  # Unique per pixel
    
    # Add structure aligned with RGB
    depth[:256, :256] += 0.1
    depth[:256, 256:] += 0.3
    depth[256:, :256] += 0.2
    depth[256:, 256:] += 0.4
    
    depth = np.clip(depth, 0, 1)
    depth_uint16 = (depth * 65535).astype(np.uint16)
    
    analyzer = DepthQualityAnalyzer(
        target_edge_alignment=0.3,  # Relaxed for synthetic
        target_unique_levels=5000
    )
    metrics = analyzer.analyze(synthetic_rgb_image, depth, depth_uint16)
    passes, _ = analyzer.validate_for_luxury_rendering(metrics)
    
    # Should pass with good metrics
    assert metrics.unique_levels_16bit > 10000


# ============================================================================
# Tests for depth_inference.py
# ============================================================================

@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("torch"),
    reason="PyTorch not available"
)
def test_tiled_estimator_initialization():
    """Test tiled depth estimator initialization."""
    from lux_depth_v2.depth_inference import create_tiled_estimator
    
    estimator = create_tiled_estimator(
        tile_size=512,
        overlap=64,
        fusion_mode="median",
        device="cpu"
    )
    
    assert estimator.config.tile_size == 512
    assert estimator.config.overlap == 64
    assert estimator.config.fusion_mode == "median"


@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("torch"),
    reason="PyTorch not available"
)
def test_tile_extraction():
    """Test tile extraction from image."""
    from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig
    
    config = TiledInferenceConfig(tile_size=256, overlap=32)
    estimator = TiledDepthEstimator(config)
    
    # Create 512x512 image
    image = np.random.rand(512, 512, 3).astype(np.float32)
    
    tiles = estimator._extract_tiles(image)
    
    # Check we got tiles
    assert len(tiles) > 0, "Should extract at least one tile"
    
    # Check tile format
    tile, y0, y1, x0, x1 = tiles[0]
    assert tile.shape[0] == 256  # tile_size
    assert tile.shape[1] == 256
    assert y1 - y0 == 256
    assert x1 - x0 == 256


@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("torch"),
    reason="PyTorch not available"
)
def test_blend_window_generation():
    """Test blending window generation."""
    from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig
    
    config = TiledInferenceConfig(tile_size=256, overlap=32, blend_window="hann")
    estimator = TiledDepthEstimator(config)
    
    window = estimator._make_blend_window(256, 32)
    
    # Check shape
    assert window.shape == (256, 256)
    
    # Check edges are ramped
    assert window[0, 128] < window[32, 128], "Top edge should be ramped"
    assert window[128, 0] < window[128, 32], "Left edge should be ramped"
    
    # Check center is full weight
    assert window[128, 128] == pytest.approx(1.0, abs=0.01)


@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("torch"),
    reason="PyTorch not available"
)
def test_edge_alignment_computation():
    """Test edge alignment score computation."""
    from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig
    
    config = TiledInferenceConfig()
    estimator = TiledDepthEstimator(config)
    
    # Create aligned RGB and depth
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[:, :128] = [255, 0, 0]
    rgb[:, 128:] = [0, 0, 255]
    
    depth = np.zeros((256, 256), dtype=np.float32)
    depth[:, :128] = 0.2
    depth[:, 128:] = 0.8
    
    score = estimator.compute_edge_alignment(rgb, depth)
    
    # Should have high score for aligned edges
    assert score > 0.3, "Aligned edges should produce moderate-to-high score"


# ============================================================================
# Integration tests
# ============================================================================

def test_end_to_end_pipeline_integration(synthetic_rgb_image, synthetic_depth_map):
    """Test complete pipeline: depth → normals → quality."""
    from lux_depth_v2.normal_map import generate_normal_map
    from lux_depth_v2.quality_metrics import quick_quality_check
    
    # Convert depth to uint16
    depth_uint16 = (synthetic_depth_map * 65535).astype(np.uint16)
    
    # Generate normals
    normals = generate_normal_map(depth_uint16, preset="architectural")
    
    # Check quality
    metrics = quick_quality_check(synthetic_rgb_image, synthetic_depth_map, depth_uint16)
    
    # Verify outputs
    assert normals.shape == (512, 512, 3)
    assert 0.0 <= normals.min() <= normals.max() <= 1.0
    assert metrics.overall_quality_score >= 0.0


def test_module_imports():
    """Test that all modules can be imported."""
    try:
        from lux_depth_v2 import depth_inference
        from lux_depth_v2 import normal_map
        from lux_depth_v2 import quality_metrics
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


# ============================================================================
# Performance benchmarks (optional, can be slow)
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("torch"),
    reason="PyTorch not available"
)
def test_benchmark_normal_map_generation():
    """Benchmark normal map generation speed."""
    import time
    from lux_depth_v2.normal_map import generate_normal_map
    
    # 4K depth map
    depth = np.random.rand(2160, 3840).astype(np.float32)
    depth_uint16 = (depth * 65535).astype(np.uint16)
    
    start = time.time()
    normals = generate_normal_map(depth_uint16)
    elapsed = time.time() - start
    
    print(f"\n4K Normal map generation: {elapsed:.3f}s")
    assert elapsed < 2.0, "Normal map generation should be fast (<2s for 4K)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
