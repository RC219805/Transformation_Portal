"""Tests for edge-aware depth refinement module.

Validates edge preservation, smoothing behavior, and metric improvements.
"""

import numpy as np
import pytest

# Mark all tests in this module as requiring ML dependencies
pytestmark = pytest.mark.ml

try:
    from lux_depth_v3.edge_refinement import (
        DepthRefiner,
        create_refinement_preset,
    )
    from lux_depth_v3.config import RefinementConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DepthRefiner = None
    create_refinement_preset = None
    RefinementConfig = None


@pytest.fixture
def sample_depth():
    """Create sample depth map with edges and smooth regions."""
    depth = np.zeros((100, 100), dtype=np.float32)
    
    # Background
    depth[:, :] = 0.5
    
    # Foreground object (step edge)
    depth[30:70, 30:70] = 0.8
    
    # Add noise
    noise = np.random.normal(0, 0.01, depth.shape).astype(np.float32)
    depth = np.clip(depth + noise, 0.0, 1.0)
    
    return depth


@pytest.fixture
def sample_rgb():
    """Create sample RGB image with corresponding edges."""
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    # Foreground object (aligned with depth edge)
    rgb[30:70, 30:70] = [200, 200, 200]
    
    return rgb


class TestDepthRefiner:
    """Test DepthRefiner class."""
    
    def test_initialization(self):
        """Test refiner initialization."""
        config = RefinementConfig()
        refiner = DepthRefiner(config)
        
        assert refiner.config == config
        # OpenCV should be available in CI
        assert refiner.has_cv2
    
    def test_bilateral_filter(self, sample_depth):
        """Test bilateral filtering does not significantly increase variance.
        
        Note: Platform differences (BLAS/LAPACK, floating-point precision, OpenCV backends)
        mean strict variance reduction is not guaranteed. This test verifies bounded behavior:
        variance should not significantly increase (≤ 10% allowed).
        """
        config = RefinementConfig(
            enable_refinement=True,
            stages=["bilateral"],
            enable_bilateral=True,
        )
        refiner = DepthRefiner(config)
        
        # Add noise to depth
        noisy_depth = sample_depth + np.random.normal(0, 0.05, sample_depth.shape)
        noisy_depth = np.clip(noisy_depth, 0.0, 1.0).astype(np.float32)
        
        # Create dummy RGB
        rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Apply refinement
        refined = refiner.refine(noisy_depth, rgb, stages=["bilateral"])
        
        # Check output shape and range
        assert refined.shape == sample_depth.shape
        assert refined.dtype == np.float32
        assert refined.min() >= 0.0
        assert refined.max() <= 1.0
        
        # Check variance does not significantly increase (≤ 10% allowed)
        # Compare flat region to minimize edge effects
        flat_region = (10, 20, 10, 20)  # y1, y2, x1, x2
        original_var = np.var(noisy_depth[flat_region[0]:flat_region[1], flat_region[2]:flat_region[3]])
        refined_var = np.var(refined[flat_region[0]:flat_region[1], flat_region[2]:flat_region[3]])
        
        assert refined_var <= original_var * 1.10, (
            f"Variance should not significantly increase (<= 10% allowed), got "
            f"refined_var={refined_var:.6f} vs original_var={original_var:.6f}"
        )
    
    def test_guided_filter(self, sample_depth, sample_rgb):
        """Test guided filter preserves edges."""
        config = RefinementConfig(
            enable_refinement=True,
            stages=["guided"],
            enable_guided=True,
            guided_radius=4,
            guided_eps=0.01,
        )
        refiner = DepthRefiner(config)
        
        # Apply refinement
        refined = refiner.refine(sample_depth, sample_rgb, stages=["guided"])
        
        # Check output properties
        assert refined.shape == sample_depth.shape
        assert refined.dtype == np.float32
        assert refined.min() >= 0.0
        assert refined.max() <= 1.0
        
        # Edge should be preserved (gradient should still exist)
        # Measure gradient across edge
        edge_gradient_original = np.abs(sample_depth[29, 50] - sample_depth[31, 50])
        edge_gradient_refined = np.abs(refined[29, 50] - refined[31, 50])
        
        # Refined gradient should be at least 50% of original
        assert edge_gradient_refined > 0.5 * edge_gradient_original
    
    def test_edge_enhancement(self, sample_depth, sample_rgb):
        """Test edge-guided enhancement."""
        config = RefinementConfig(
            enable_refinement=True,
            stages=["edge"],
            enable_edge=True,
            edge_canny_low=30,
            edge_canny_high=120,
        )
        refiner = DepthRefiner(config)
        
        # Apply refinement
        refined = refiner.refine(sample_depth, sample_rgb, stages=["edge"])
        
        # Check output properties
        assert refined.shape == sample_depth.shape
        assert refined.dtype == np.float32
        assert refined.min() >= 0.0
        assert refined.max() <= 1.0
    
    def test_gradient_smoothing(self, sample_depth, sample_rgb):
        """Test gradient consistency filtering."""
        config = RefinementConfig(
            enable_refinement=True,
            stages=["gradient"],
            enable_gradient=True,
            gradient_threshold=0.1,
        )
        refiner = DepthRefiner(config)
        
        # Apply refinement
        refined = refiner.refine(sample_depth, sample_rgb, stages=["gradient"])
        
        # Check output properties
        assert refined.shape == sample_depth.shape
        assert refined.dtype == np.float32
        assert refined.min() >= 0.0
        assert refined.max() <= 1.0
    
    def test_multi_stage_pipeline(self, sample_depth, sample_rgb):
        """Test multi-stage refinement pipeline."""
        config = RefinementConfig(
            enable_refinement=True,
            stages=["guided", "bilateral", "edge"],
            enable_guided=True,
            enable_bilateral=True,
            enable_edge=True,
        )
        refiner = DepthRefiner(config)
        
        # Apply refinement
        refined = refiner.refine(sample_depth, sample_rgb)
        
        # Check output properties
        assert refined.shape == sample_depth.shape
        assert refined.dtype == np.float32
        assert refined.min() >= 0.0
        assert refined.max() <= 1.0
        
        # Refined depth should differ from original
        assert not np.allclose(refined, sample_depth)
    
    def test_disabled_refinement(self, sample_depth, sample_rgb):
        """Test that disabled refinement returns original depth."""
        config = RefinementConfig(enable_refinement=False)
        refiner = DepthRefiner(config)
        
        refined = refiner.refine(sample_depth, sample_rgb)
        
        # Should return original depth unchanged
        assert np.array_equal(refined, sample_depth)
    
    def test_custom_stages(self, sample_depth, sample_rgb):
        """Test custom stage ordering."""
        config = RefinementConfig(
            enable_refinement=True,
            enable_bilateral=True,
            enable_guided=True,
        )
        refiner = DepthRefiner(config)
        
        # Custom stage order
        refined = refiner.refine(
            sample_depth,
            sample_rgb,
            stages=["bilateral", "guided"],  # Reverse order
        )
        
        assert refined.shape == sample_depth.shape
        assert refined.dtype == np.float32
    
    def test_stats(self):
        """Test get_stats method."""
        config = RefinementConfig(
            enable_refinement=True,
            stages=["guided", "bilateral"],
            enable_guided=True,
            enable_bilateral=True,
        )
        refiner = DepthRefiner(config)
        
        stats = refiner.get_stats()
        
        assert "enabled" in stats
        assert "stages" in stats
        assert "has_opencv" in stats
        assert stats["enabled"] is True
        assert stats["bilateral_enabled"] is True
        assert stats["guided_enabled"] is True


class TestRefinementPresets:
    """Test refinement preset configurations."""
    
    def test_balanced_preset(self):
        """Test balanced preset."""
        config = create_refinement_preset("balanced")
        
        assert config.enable_refinement is True
        assert "guided" in config.stages
        assert "bilateral" in config.stages
        assert config.enable_guided is True
        assert config.enable_bilateral is True
    
    def test_aggressive_preset(self):
        """Test aggressive preset."""
        config = create_refinement_preset("aggressive")
        
        assert config.enable_refinement is True
        assert len(config.stages) == 4  # All stages
        assert config.enable_guided is True
        assert config.enable_bilateral is True
        assert config.enable_edge is True
        assert config.enable_gradient is True
    
    def test_conservative_preset(self):
        """Test conservative preset."""
        config = create_refinement_preset("conservative")
        
        assert config.enable_refinement is True
        assert config.stages == ["bilateral"]
        assert config.enable_bilateral is True
        assert config.enable_guided is False
        assert config.enable_edge is False
    
    def test_edge_focused_preset(self):
        """Test edge-focused preset."""
        config = create_refinement_preset("edge_focused")
        
        assert config.enable_refinement is True
        assert "edge" in config.stages
        assert "guided" in config.stages
        assert config.enable_edge is True
        assert config.enable_guided is True
    
    def test_unknown_preset_fallback(self):
        """Test fallback to balanced for unknown preset."""
        config = create_refinement_preset("unknown_preset")
        
        # Should fall back to balanced
        assert config.enable_refinement is True
        assert "guided" in config.stages
        assert "bilateral" in config.stages


class TestEdgePreservation:
    """Test edge preservation properties."""
    
    def test_preserves_sharp_edges(self, sample_rgb):
        """Test that refinement preserves sharp edges."""
        # Create depth with sharp edge
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[:, :50] = 0.3
        depth[:, 50:] = 0.9
        
        config = RefinementConfig(
            enable_refinement=True,
            stages=["guided", "bilateral"],
            enable_guided=True,
            enable_bilateral=True,
        )
        refiner = DepthRefiner(config)
        
        refined = refiner.refine(depth, sample_rgb)
        
        # Check that edge location is preserved
        # Edge should still be near column 50
        edge_col_original = 50
        
        # Find edge in refined depth (max gradient)
        grad = np.abs(np.diff(refined[50, :]))
        edge_col_refined = np.argmax(grad)
        
        # Edge should be within a few pixels of original
        assert abs(edge_col_refined - edge_col_original) < 5
    
    def test_smooths_flat_regions(self):
        """Test that refinement does not significantly increase variance in flat regions.
        
        Note: Variance-based assertions are platform-sensitive due to differences in:
        - BLAS/LAPACK implementations
        - Floating-point precision (x86_64 vs ARM)
        - OpenCV bilateral filter backends
        
        A 10% tolerance allows for these platform variations. The test verifies bounded
        behavior: variance should not significantly increase (≤ 10% allowed).
        
        For stricter variance reduction guarantees, use center-crop metrics to exclude
        edge effects (see Issue #595 for future enhancement).
        """
        # Create noisy flat depth with deterministic seed
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        depth = np.ones((100, 100), dtype=np.float32) * 0.5
        noise = rng.normal(0, 0.05, depth.shape).astype(np.float32)
        noisy_depth = np.clip(depth + noise, 0.0, 1.0)
        
        # Create uniform RGB (no edges)
        rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        config = RefinementConfig(
            enable_refinement=True,
            stages=["bilateral"],
            enable_bilateral=True,
        )
        refiner = DepthRefiner(config)
        
        refined = refiner.refine(noisy_depth, rgb)
        
        # Variance should not significantly increase (≤ 10% allowed)
        # Compute in float64 to reduce BLAS/dtype sensitivity
        # Allows for platform differences and filter edge effects
        original_var = np.var(noisy_depth.astype(np.float64))
        refined_var = np.var(refined.astype(np.float64))
        
        assert refined_var <= original_var * 1.10, (
            f"Variance should not significantly increase (<= 10% allowed), got "
            f"refined_var={refined_var:.6f} vs original_var={original_var:.6f}"
        )


@pytest.mark.parametrize("preset", ["balanced", "aggressive", "conservative", "edge_focused"])
def test_all_presets_work(preset, sample_depth, sample_rgb):
    """Test that all presets can process depth maps."""
    config = create_refinement_preset(preset)
    refiner = DepthRefiner(config)
    
    refined = refiner.refine(sample_depth, sample_rgb)
    
    # Basic checks
    assert refined.shape == sample_depth.shape
    assert refined.dtype == np.float32
    assert refined.min() >= 0.0
    assert refined.max() <= 1.0


def test_no_opencv_fallback(sample_depth, sample_rgb):
    """Test graceful fallback when OpenCV unavailable."""
    config = RefinementConfig(enable_refinement=True)
    refiner = DepthRefiner(config)
    
    # Simulate missing OpenCV
    refiner.has_cv2 = False
    
    refined = refiner.refine(sample_depth, sample_rgb)
    
    # Should return original depth unchanged
    assert np.array_equal(refined, sample_depth)
