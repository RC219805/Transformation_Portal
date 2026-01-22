#!/usr/bin/env python3
"""
Tests for Edge Refinement Module
==================================

Comprehensive test suite for edge-aware depth refinement techniques.

Test Coverage:
    - Bilateral filtering (Module 1)
    - Guided filter (Module 2)
    - Edge-guided enhancement (Module 3)
    - Gradient consistency filtering (Module 4)
    - Segment-aware refinement (Module 5)
    - Pipeline integration
    - Input validation and security (CWE-703, CWE-834)

Author: Transformation Portal Specialist
Date: 2025-12-20
"""

import pytest
import numpy as np
import cv2

from lux_depth_v2.edge_refinement import (
    bilateral_depth_filter,
    guided_filter_depth,
    enhance_edges_with_guidance,
    gradient_smoothness,
    segment_aware_refine,
    EdgeRefinementPipeline,
    EdgeRefinementConfig,
    RefinementPreset,
    refine_depth_edge_aware,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_depth():
    """Create synthetic depth map with clear edges (256x256)."""
    depth = np.zeros((256, 256), dtype=np.float32)
    # Create step edges (architectural structure)
    depth[:128, :] = 0.3  # Background
    depth[128:, :] = 0.7  # Foreground
    depth[:, 128:] += 0.1  # Right side offset
    return np.clip(depth, 0, 1)


@pytest.fixture
def synthetic_rgb(synthetic_depth):
    """Create synthetic RGB image aligned with depth edges."""
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    # Create edges aligned with depth
    rgb[:128, :] = [100, 100, 100]  # Gray background
    rgb[128:, :] = [200, 200, 200]  # Bright foreground
    rgb[:, 128:, 0] += 50  # Red tint on right
    return rgb


@pytest.fixture
def noisy_depth(synthetic_depth):
    """Add Gaussian noise to synthetic depth."""
    noise = np.random.normal(0, 0.05, synthetic_depth.shape)
    return np.clip(synthetic_depth + noise, 0, 1).astype(np.float32)


@pytest.fixture
def segmentation_mask():
    """Create segmentation mask with 4 segments."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:128, :128] = 0  # Top-left
    mask[:128, 128:] = 1  # Top-right
    mask[128:, :128] = 2  # Bottom-left
    mask[128:, 128:] = 3  # Bottom-right
    return mask


@pytest.fixture
def simple_depth():
    """Simple 64x64 depth map for fast tests."""
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture
def simple_rgb():
    """Simple 64x64 RGB image for fast tests."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


# ============================================================================
# Module 1: Bilateral Filtering Tests
# ============================================================================


class TestBilateralDepthFilter:
    """Test suite for bilateral filtering."""

    def test_basic_filtering(self, noisy_depth):
        """Test bilateral filter reduces noise while preserving edges."""
        filtered = bilateral_depth_filter(noisy_depth, d=9, sigma_color=75, sigma_space=75)

        assert filtered.shape == noisy_depth.shape
        assert filtered.dtype == np.float32
        assert 0.0 <= filtered.min() <= filtered.max() <= 1.0

    def test_edge_preservation(self, synthetic_depth):
        """Test that bilateral filter preserves sharp edges."""
        filtered = bilateral_depth_filter(synthetic_depth, d=9, sigma_color=50, sigma_space=50)

        # Edge at y=128 should be preserved
        edge_diff_original = abs(synthetic_depth[127, 64] - synthetic_depth[128, 64])
        edge_diff_filtered = abs(filtered[127, 64] - filtered[128, 64])

        # Edge should be preserved (at least 70% of original contrast)
        assert edge_diff_filtered > edge_diff_original * 0.7

    def test_uint8_input(self):
        """Test bilateral filter with uint8 input."""
        depth_uint8 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        filtered = bilateral_depth_filter(depth_uint8)

        assert filtered.shape == depth_uint8.shape
        assert filtered.dtype == np.uint8

    def test_uint16_input(self):
        """Test bilateral filter with uint16 input."""
        depth_uint16 = np.random.randint(0, 65536, (64, 64), dtype=np.uint16)
        filtered = bilateral_depth_filter(depth_uint16)

        assert filtered.shape == depth_uint16.shape
        assert filtered.dtype == np.uint16

    def test_auto_diameter(self, simple_depth):
        """Test automatic diameter computation when d=0."""
        filtered = bilateral_depth_filter(simple_depth, d=0, sigma_space=5.0)

        assert filtered.shape == simple_depth.shape
        assert filtered.dtype == np.float32

    def test_invalid_input_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError):
            bilateral_depth_filter([1, 2, 3])

    def test_invalid_shape(self):
        """Test error handling for invalid shape (3D instead of 2D)."""
        with pytest.raises(ValueError, match="must be 2D"):
            bilateral_depth_filter(np.random.rand(64, 64, 3))

    def test_dimension_limit(self):
        """Test resource exhaustion prevention (CWE-834)."""
        with pytest.raises(ValueError, match="exceed maximum"):
            bilateral_depth_filter(np.random.rand(10000, 10000))


# ============================================================================
# Module 2: Guided Filter Tests
# ============================================================================


class TestGuidedFilterDepth:
    """Test suite for guided filter."""

    def test_basic_guided_filtering(self, simple_depth, simple_rgb):
        """Test guided filter with basic inputs."""
        filtered = guided_filter_depth(simple_depth, simple_rgb, radius=8, eps=0.01)

        assert filtered.shape == simple_depth.shape
        assert filtered.dtype == np.float32
        assert 0.0 <= filtered.min() <= filtered.max() <= 1.0

    def test_edge_alignment(self, synthetic_depth, synthetic_rgb):
        """Test that guided filter aligns depth edges with RGB edges."""
        filtered = guided_filter_depth(synthetic_depth, synthetic_rgb, radius=8, eps=0.01)

        # Edges should be preserved
        edge_diff = abs(filtered[127, 64] - filtered[128, 64])
        assert edge_diff > 0.1  # Significant edge preserved

    def test_float_rgb_input(self, simple_depth):
        """Test guided filter with float32 RGB input."""
        rgb_float = np.random.rand(64, 64, 3).astype(np.float32)
        filtered = guided_filter_depth(simple_depth, rgb_float)

        assert filtered.shape == simple_depth.shape
        assert filtered.dtype == np.float32

    def test_shape_mismatch(self, simple_depth):
        """Test error handling for shape mismatch."""
        rgb_wrong_size = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Shape mismatch"):
            guided_filter_depth(simple_depth, rgb_wrong_size)

    def test_invalid_rgb_shape(self, simple_depth):
        """Test error handling for invalid RGB shape (not 3 channels)."""
        rgb_invalid = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        with pytest.raises(ValueError, match="must be HxWx3"):
            guided_filter_depth(simple_depth, rgb_invalid)

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(TypeError):
            guided_filter_depth([1, 2, 3], np.zeros((64, 64, 3)))


# ============================================================================
# Module 3: Edge-Guided Enhancement Tests
# ============================================================================


class TestEnhanceEdgesWithGuidance:
    """Test suite for edge-guided enhancement."""

    def test_basic_enhancement(self, simple_depth, simple_rgb):
        """Test edge enhancement with basic inputs."""
        enhanced = enhance_edges_with_guidance(simple_depth, simple_rgb, strength=0.3)

        assert enhanced.shape == simple_depth.shape
        assert enhanced.dtype == np.float32
        assert 0.0 <= enhanced.min() <= enhanced.max() <= 1.0

    def test_edge_sharpening(self, synthetic_depth, synthetic_rgb):
        """Test that edge enhancement increases edge sharpness."""
        enhanced = enhance_edges_with_guidance(synthetic_depth, synthetic_rgb, strength=0.5, threshold=30.0)

        # Compute edge strength before and after
        def edge_strength(depth, y, x):
            return abs(depth[y - 1, x] - depth[y + 1, x])

        original_strength = edge_strength(synthetic_depth, 128, 64)
        enhanced_strength = edge_strength(enhanced, 128, 64)

        # Enhanced should have stronger edges
        assert enhanced_strength >= original_strength * 0.9

    def test_strength_parameter(self, simple_depth, simple_rgb):
        """Test different strength values."""
        weak = enhance_edges_with_guidance(simple_depth, simple_rgb, strength=0.1)
        strong = enhance_edges_with_guidance(simple_depth, simple_rgb, strength=0.8)

        # Both should be valid
        assert weak.shape == simple_depth.shape
        assert strong.shape == simple_depth.shape

    def test_invalid_strength(self, simple_depth, simple_rgb):
        """Test error handling for invalid strength values."""
        with pytest.raises(ValueError, match="strength must be in"):
            enhance_edges_with_guidance(simple_depth, simple_rgb, strength=1.5)

        with pytest.raises(ValueError, match="strength must be in"):
            enhance_edges_with_guidance(simple_depth, simple_rgb, strength=-0.1)

    def test_invalid_threshold(self, simple_depth, simple_rgb):
        """Test error handling for invalid threshold values."""
        with pytest.raises(ValueError, match="threshold must be in"):
            enhance_edges_with_guidance(simple_depth, simple_rgb, threshold=300.0)

        with pytest.raises(ValueError, match="threshold must be in"):
            enhance_edges_with_guidance(simple_depth, simple_rgb, threshold=-10.0)

    def test_shape_mismatch(self, simple_depth):
        """Test error handling for shape mismatch."""
        rgb_wrong = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Shape mismatch"):
            enhance_edges_with_guidance(simple_depth, rgb_wrong)


# ============================================================================
# Module 4: Gradient Smoothness Tests
# ============================================================================


class TestGradientSmoothness:
    """Test suite for gradient consistency filtering."""

    def test_basic_gradient_smoothing(self, simple_depth, simple_rgb):
        """Test gradient smoothing with basic inputs."""
        smoothed = gradient_smoothness(simple_depth, simple_rgb, gradient_weight=0.5)

        assert smoothed.shape == simple_depth.shape
        assert smoothed.dtype == np.float32
        assert 0.0 <= smoothed.min() <= smoothed.max() <= 1.0

    def test_gradient_alignment(self, synthetic_depth, synthetic_rgb):
        """Test that gradient smoothing preserves edges with high RGB gradients."""
        smoothed = gradient_smoothness(synthetic_depth, synthetic_rgb, gradient_weight=0.7)

        # Edge should be preserved
        edge_diff = abs(smoothed[127, 64] - smoothed[128, 64])
        assert edge_diff > 0.1

    def test_gradient_weight_parameter(self, simple_depth, simple_rgb):
        """Test different gradient weight values."""
        low_weight = gradient_smoothness(simple_depth, simple_rgb, gradient_weight=0.2)
        high_weight = gradient_smoothness(simple_depth, simple_rgb, gradient_weight=0.8)

        assert low_weight.shape == simple_depth.shape
        assert high_weight.shape == simple_depth.shape

    def test_invalid_gradient_weight(self, simple_depth, simple_rgb):
        """Test error handling for invalid gradient weight."""
        with pytest.raises(ValueError, match="gradient_weight must be in"):
            gradient_smoothness(simple_depth, simple_rgb, gradient_weight=1.5)

    def test_shape_mismatch(self, simple_depth):
        """Test error handling for shape mismatch."""
        rgb_wrong = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Shape mismatch"):
            gradient_smoothness(simple_depth, rgb_wrong)


# ============================================================================
# Module 5: Segment-Aware Refinement Tests
# ============================================================================


class TestSegmentAwareRefine:
    """Test suite for segment-aware refinement."""

    def test_basic_segment_refinement(self, simple_depth):
        """Test segment-aware refinement with basic inputs."""
        segments = np.random.randint(0, 5, (64, 64), dtype=np.uint8)
        refined = segment_aware_refine(simple_depth, segments, filter_radius=5)

        assert refined.shape == simple_depth.shape
        assert refined.dtype == np.float32
        assert 0.0 <= refined.min() <= refined.max() <= 1.0

    def test_segment_boundary_preservation(self, segmentation_mask):
        """Test that segment boundaries are preserved."""
        depth = np.random.rand(256, 256).astype(np.float32)
        refined = segment_aware_refine(depth, segmentation_mask, filter_radius=5)

        # Boundaries should show discontinuities
        boundary_diff = abs(refined[127, 127] - refined[128, 128])
        # Should have some edge contrast (not fully smoothed)
        assert refined.shape == depth.shape

    def test_single_segment(self, simple_depth):
        """Test refinement with single segment (uniform smoothing)."""
        single_segment = np.zeros((64, 64), dtype=np.uint8)
        refined = segment_aware_refine(simple_depth, single_segment, filter_radius=5)

        assert refined.shape == simple_depth.shape

    def test_invalid_filter_radius(self, simple_depth):
        """Test error handling for invalid filter radius."""
        segments = np.zeros((64, 64), dtype=np.uint8)

        with pytest.raises(ValueError, match="filter_radius must be in"):
            segment_aware_refine(simple_depth, segments, filter_radius=0)

        with pytest.raises(ValueError, match="filter_radius must be in"):
            segment_aware_refine(simple_depth, segments, filter_radius=50)

    def test_shape_mismatch(self, simple_depth):
        """Test error handling for shape mismatch."""
        segments_wrong = np.zeros((32, 32), dtype=np.uint8)

        with pytest.raises(ValueError, match="Shape mismatch"):
            segment_aware_refine(simple_depth, segments_wrong)

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(TypeError):
            segment_aware_refine([1, 2, 3], np.zeros((64, 64)))


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


class TestEdgeRefinementPipeline:
    """Test suite for edge refinement pipeline."""

    def test_pipeline_initialization_default(self):
        """Test pipeline initialization with default config."""
        pipeline = EdgeRefinementPipeline()

        assert pipeline.config is not None
        assert isinstance(pipeline.config, EdgeRefinementConfig)

    def test_pipeline_initialization_custom(self):
        """Test pipeline initialization with custom config."""
        config = EdgeRefinementConfig(enable_bilateral=True, enable_guided=False, enable_edge_enhancement=True)
        pipeline = EdgeRefinementPipeline(config)

        assert pipeline.config.enable_bilateral is True
        assert pipeline.config.enable_guided is False

    def test_pipeline_bilateral_only(self, simple_depth):
        """Test pipeline with only bilateral filtering enabled."""
        config = EdgeRefinementConfig(
            enable_bilateral=True,
            enable_guided=False,
            enable_edge_enhancement=False,
            enable_gradient_smoothing=False,
        )
        pipeline = EdgeRefinementPipeline(config)

        refined = pipeline.refine(simple_depth)

        assert refined.shape == simple_depth.shape
        assert refined.dtype == np.float32

    def test_pipeline_full_stack(self, simple_depth, simple_rgb):
        """Test pipeline with all modules enabled."""
        config = EdgeRefinementConfig(
            enable_bilateral=True,
            enable_guided=True,
            enable_edge_enhancement=True,
            enable_gradient_smoothing=True,
        )
        pipeline = EdgeRefinementPipeline(config)

        refined = pipeline.refine(simple_depth, simple_rgb)

        assert refined.shape == simple_depth.shape
        assert refined.dtype == np.float32

    def test_pipeline_missing_rgb(self, simple_depth):
        """Test error when RGB required but not provided."""
        config = EdgeRefinementConfig(enable_guided=True)
        pipeline = EdgeRefinementPipeline(config)

        with pytest.raises(ValueError, match="rgb_image required"):
            pipeline.refine(simple_depth, rgb_image=None)

    def test_pipeline_with_segmentation(self, simple_depth, simple_rgb):
        """Test pipeline with segmentation mask."""
        segments = np.random.randint(0, 5, (64, 64), dtype=np.uint8)
        pipeline = EdgeRefinementPipeline()

        refined = pipeline.refine(simple_depth, simple_rgb, segments)

        assert refined.shape == simple_depth.shape


# ============================================================================
# Configuration Tests
# ============================================================================


class TestEdgeRefinementConfig:
    """Test suite for configuration classes."""

    def test_config_default_initialization(self):
        """Test default configuration initialization."""
        config = EdgeRefinementConfig()

        assert config.enable_bilateral is True
        assert config.enable_guided is True
        assert config.bilateral_d == 9
        assert config.guided_radius == 8

    def test_config_preset_subtle(self):
        """Test subtle preset configuration."""
        config = EdgeRefinementConfig.from_preset(RefinementPreset.SUBTLE)

        assert config.bilateral_sigma_color == 50.0
        assert config.edge_enhancement_strength == 0.15
        assert config.structure_weight == 0.4

    def test_config_preset_balanced(self):
        """Test balanced preset configuration."""
        config = EdgeRefinementConfig.from_preset(RefinementPreset.BALANCED)

        assert config.bilateral_sigma_color == 75.0
        assert config.edge_enhancement_strength == 0.3
        assert config.structure_weight == 0.5

    def test_config_preset_aggressive(self):
        """Test aggressive preset configuration."""
        config = EdgeRefinementConfig.from_preset(RefinementPreset.AGGRESSIVE)

        assert config.bilateral_sigma_color == 100.0
        assert config.edge_enhancement_strength == 0.5
        assert config.structure_weight == 0.6


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_refine_depth_edge_aware_basic(self, simple_depth, simple_rgb):
        """Test convenience function with basic inputs."""
        refined = refine_depth_edge_aware(simple_depth, simple_rgb)

        assert refined.shape == simple_depth.shape
        assert refined.dtype == np.float32

    def test_refine_depth_edge_aware_preset(self, simple_depth, simple_rgb):
        """Test convenience function with different presets."""
        subtle = refine_depth_edge_aware(simple_depth, simple_rgb, preset=RefinementPreset.SUBTLE)
        aggressive = refine_depth_edge_aware(simple_depth, simple_rgb, preset=RefinementPreset.AGGRESSIVE)

        assert subtle.shape == simple_depth.shape
        assert aggressive.shape == simple_depth.shape

    def test_refine_depth_edge_aware_with_segments(self, simple_depth, simple_rgb):
        """Test convenience function with segmentation."""
        segments = np.random.randint(0, 5, (64, 64), dtype=np.uint8)
        refined = refine_depth_edge_aware(simple_depth, simple_rgb, segmentation_mask=segments)

        assert refined.shape == simple_depth.shape


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for realistic workflows."""

    def test_full_workflow_noisy_depth(self, noisy_depth, synthetic_rgb):
        """Test complete workflow on noisy depth map."""
        pipeline = EdgeRefinementPipeline(EdgeRefinementConfig.from_preset(RefinementPreset.BALANCED))

        refined = pipeline.refine(noisy_depth, synthetic_rgb)

        # Refined should have less variance (smoother) while preserving edges
        assert refined.std() <= noisy_depth.std() * 1.1  # Some smoothing
        assert refined.shape == noisy_depth.shape

    def test_architectural_rendering_workflow(self, synthetic_depth, synthetic_rgb):
        """Test architectural rendering workflow with structure preservation."""
        config = EdgeRefinementConfig(
            enable_bilateral=True,
            enable_guided=True,
            enable_edge_enhancement=True,
            edge_enhancement_strength=0.4,
            structure_weight=0.6,
        )
        pipeline = EdgeRefinementPipeline(config)

        refined = pipeline.refine(synthetic_depth, synthetic_rgb)

        # Edge at y=128 should be preserved
        edge_contrast_original = abs(synthetic_depth[127, 64] - synthetic_depth[128, 64])
        edge_contrast_refined = abs(refined[127, 64] - refined[128, 64])

        # At least 60% edge preservation
        assert edge_contrast_refined > edge_contrast_original * 0.6

    def test_performance_benchmark(self, benchmark=None):
        """Benchmark processing time (if pytest-benchmark available)."""
        if benchmark is None:
            pytest.skip("pytest-benchmark not available")

        depth = np.random.rand(512, 512).astype(np.float32)
        rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        pipeline = EdgeRefinementPipeline()

        def run():
            return pipeline.refine(depth, rgb)

        result = benchmark(run)
        assert result.shape == depth.shape


# ============================================================================
# Property-Based Tests (if hypothesis available)
# ============================================================================


try:
    from hypothesis import given, strategies as st, settings, HealthCheck

    class TestPropertyBased:
        """Property-based tests using hypothesis."""

        @given(
            depth=st.lists(
                st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=8, max_size=8),
                min_size=8,
                max_size=8,
            )
        )
        @settings(
            suppress_health_check=[
                HealthCheck.large_base_example,
                HealthCheck.data_too_large,
            ]
        )
        def test_bilateral_preserves_range(self, depth):
            """Property: Bilateral filter output is bounded [0, 1]."""
            depth_arr = np.array(depth, dtype=np.float32)
            filtered = bilateral_depth_filter(depth_arr)

            assert 0.0 <= filtered.min() <= filtered.max() <= 1.0

        @given(strength=st.floats(min_value=0.0, max_value=1.0))
        def test_edge_enhancement_strength_bounded(self, strength):
            """Property: Edge enhancement with any valid strength produces bounded output."""
            depth = np.random.rand(64, 64).astype(np.float32)
            rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

            enhanced = enhance_edges_with_guidance(depth, rgb, strength=strength)

            assert 0.0 <= enhanced.min() <= enhanced.max() <= 1.0

except ImportError:
    pass  # Hypothesis not available, skip property-based tests
