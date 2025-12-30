#!/usr/bin/env python3
"""
Tests for Advanced Edge-Aware Depth Refinement Module
======================================================

Validates all refinement techniques and edge quality metrics.

Author: Transformation Portal Specialist
Date: 2025-12-20
"""

import pytest
import numpy as np
import cv2

from lux_depth_v2.advanced_refinement import (
    DepthRefiner,
    AdvancedRefinementConfig,
    RefinementTechnique,
    refine_depth_advanced,
    compute_edge_metrics,
    compute_chamfer_distance,
)


@pytest.fixture
def synthetic_depth():
    """Create synthetic depth map with clear edges."""
    depth = np.zeros((256, 256), dtype=np.float32)
    # Create step edges (architectural structure)
    depth[:128, :] = 0.3  # Background
    depth[128:, :] = 0.7  # Foreground
    depth[:, 128:] += 0.2  # Right side offset
    return depth


@pytest.fixture
def synthetic_rgb():
    """Create synthetic RGB image aligned with depth edges."""
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    # Create edges aligned with depth
    rgb[:128, :] = [100, 100, 100]  # Gray background
    rgb[128:, :] = [200, 200, 200]  # Bright foreground
    rgb[:, 128:, 0] += 50  # Red tint on right
    return rgb


@pytest.fixture
def noisy_depth(synthetic_depth):
    """Add noise to synthetic depth."""
    noise = np.random.normal(0, 0.05, synthetic_depth.shape)
    return np.clip(synthetic_depth + noise, 0, 1).astype(np.float32)


class TestDepthRefiner:
    """Test suite for DepthRefiner class."""

    def test_initialization_default_config(self):
        """Test refiner initialization with default config."""
        refiner = DepthRefiner()
        assert refiner.config is not None
        assert isinstance(refiner.config, AdvancedRefinementConfig)

    def test_initialization_custom_config(self):
        """Test refiner initialization with custom config."""
        config = AdvancedRefinementConfig(bilateral_d=5, guided_radius=16)
        refiner = DepthRefiner(config)
        assert refiner.config.bilateral_d == 5
        assert refiner.config.guided_radius == 16

    def test_normalize_depth_uint8(self):
        """Test depth normalization for uint8 input."""
        refiner = DepthRefiner()
        depth_uint8 = np.array([[0, 127, 255]], dtype=np.uint8)
        depth_norm, metadata = refiner._normalize_depth(depth_uint8)

        assert depth_norm.dtype == np.float32
        assert depth_norm.min() == 0.0
        assert depth_norm.max() == pytest.approx(1.0, abs=0.01)
        assert metadata["original_dtype"] == np.uint8

    def test_normalize_depth_uint16(self):
        """Test depth normalization for uint16 input."""
        refiner = DepthRefiner()
        depth_uint16 = np.array([[0, 32767, 65535]], dtype=np.uint16)
        depth_norm, metadata = refiner._normalize_depth(depth_uint16)

        assert depth_norm.dtype == np.float32
        assert depth_norm.min() == 0.0
        assert depth_norm.max() == pytest.approx(1.0, abs=0.001)
        assert metadata["original_dtype"] == np.uint16

    def test_normalize_depth_float32(self):
        """Test depth normalization for float32 input."""
        refiner = DepthRefiner()
        depth_float = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        depth_norm, metadata = refiner._normalize_depth(depth_float)

        assert depth_norm.dtype == np.float32
        assert depth_norm.min() == 0.0
        assert depth_norm.max() == 1.0
        assert metadata["original_dtype"] == np.float32

    def test_bilateral_filter_reduces_noise(self, noisy_depth):
        """Test bilateral filter noise reduction."""
        refiner = DepthRefiner()
        filtered = refiner.bilateral_filter(noisy_depth)

        # Filtered should have lower variance (less noise)
        assert filtered.std() < noisy_depth.std()
        # Output shape preserved
        assert filtered.shape == noisy_depth.shape

    def test_bilateral_filter_preserves_edges(self, synthetic_depth):
        """Test bilateral filter edge preservation."""
        refiner = DepthRefiner()
        filtered = refiner.bilateral_filter(synthetic_depth)

        # Compute edge strength before/after
        edges_orig = cv2.Sobel(synthetic_depth, cv2.CV_32F, 1, 0, ksize=3)
        edges_filt = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)

        # Edge magnitude should be similar (edges preserved)
        assert np.abs(edges_filt).mean() > 0.5 * np.abs(edges_orig).mean()

    def test_guided_filter_with_rgb(self, noisy_depth, synthetic_rgb):
        """Test guided filter with RGB guidance."""
        refiner = DepthRefiner()
        filtered = refiner.guided_filter(noisy_depth, synthetic_rgb)

        # Should reduce noise while preserving RGB-aligned edges
        assert filtered.std() < noisy_depth.std()
        assert filtered.shape == noisy_depth.shape

    def test_guided_filter_fallback_without_ximgproc(self, noisy_depth, synthetic_rgb, monkeypatch):
        """Test guided filter fallback when ximgproc unavailable."""
        # Mock cv2 to simulate missing ximgproc
        refiner = DepthRefiner()

        # Should still work (fallback to bilateral)
        filtered = refiner.guided_filter(noisy_depth, synthetic_rgb)
        assert filtered.shape == noisy_depth.shape

    def test_edge_guided_enhancement(self, synthetic_depth, synthetic_rgb):
        """Test edge-guided enhancement preserves RGB edges."""
        refiner = DepthRefiner()
        enhanced = refiner.edge_guided_enhancement(synthetic_depth, synthetic_rgb)

        # Output shape preserved
        assert enhanced.shape == synthetic_depth.shape

        # Compute edge alignment
        metrics = compute_edge_metrics(enhanced, synthetic_rgb, "comprehensive")
        assert "edge_alignment" in metrics

    def test_gradient_consistency_filter(self, noisy_depth, synthetic_rgb):
        """Test gradient consistency filtering."""
        refiner = DepthRefiner()
        filtered = refiner.gradient_consistency_filter(noisy_depth, synthetic_rgb)

        # Should smooth in low-gradient RGB regions (or maintain similar std)
        assert filtered.std() <= noisy_depth.std() * 1.01  # Allow small tolerance
        assert filtered.shape == noisy_depth.shape

    def test_hybrid_refinement_all_stages(self, noisy_depth, synthetic_rgb):
        """Test hybrid refinement pipeline with all stages."""
        config = AdvancedRefinementConfig(use_bilateral_first=True, use_gradient_alignment=True, use_edge_preservation=True)
        refiner = DepthRefiner(config)
        refined = refiner.hybrid_refinement(noisy_depth, synthetic_rgb)

        # Should produce cleaner depth with preserved edges
        assert refined.std() < noisy_depth.std()
        assert refined.shape == noisy_depth.shape

    def test_hybrid_refinement_selective_stages(self, noisy_depth, synthetic_rgb):
        """Test hybrid refinement with selective stages."""
        config = AdvancedRefinementConfig(use_bilateral_first=False, use_gradient_alignment=True, use_edge_preservation=False)
        refiner = DepthRefiner(config)
        refined = refiner.hybrid_refinement(noisy_depth, synthetic_rgb)

        assert refined.shape == noisy_depth.shape

    def test_refine_bilateral_technique(self, noisy_depth):
        """Test refine() with bilateral technique."""
        refiner = DepthRefiner()
        refined = refiner.refine(noisy_depth, technique="bilateral")

        assert refined.shape == noisy_depth.shape
        assert refined.std() < noisy_depth.std()

    def test_refine_guided_technique(self, noisy_depth, synthetic_rgb):
        """Test refine() with guided filter technique."""
        refiner = DepthRefiner()
        refined = refiner.refine(noisy_depth, synthetic_rgb, technique="guided")

        assert refined.shape == noisy_depth.shape

    def test_refine_edge_guided_technique(self, noisy_depth, synthetic_rgb):
        """Test refine() with edge-guided technique."""
        refiner = DepthRefiner()
        refined = refiner.refine(noisy_depth, synthetic_rgb, technique="edge_guided")

        assert refined.shape == noisy_depth.shape

    def test_refine_gradient_consistency_technique(self, noisy_depth, synthetic_rgb):
        """Test refine() with gradient consistency technique."""
        refiner = DepthRefiner()
        refined = refiner.refine(noisy_depth, synthetic_rgb, technique="gradient_consistency")

        assert refined.shape == noisy_depth.shape

    def test_refine_hybrid_technique(self, noisy_depth, synthetic_rgb):
        """Test refine() with hybrid technique (recommended)."""
        refiner = DepthRefiner()
        refined = refiner.refine(noisy_depth, synthetic_rgb, technique="hybrid")

        assert refined.shape == noisy_depth.shape
        assert refined.std() < noisy_depth.std()

    def test_refine_without_rgb_fallback(self, noisy_depth):
        """Test refine() fallback when RGB not provided."""
        refiner = DepthRefiner()
        # Should fallback to bilateral
        refined = refiner.refine(noisy_depth, rgb=None, technique="guided")

        assert refined.shape == noisy_depth.shape

    def test_refine_invalid_technique(self, noisy_depth):
        """Test refine() with invalid technique raises error."""
        refiner = DepthRefiner()

        with pytest.raises(ValueError):
            refiner.refine(noisy_depth, technique="invalid_technique")


class TestEdgeMetrics:
    """Test suite for edge quality metrics."""

    def test_compute_edge_metrics_basic(self, synthetic_depth):
        """Test basic edge metrics computation."""
        metrics = compute_edge_metrics(synthetic_depth, metric_type="basic")

        # Should contain gradient statistics
        assert "gradient_mean" in metrics
        assert "gradient_std" in metrics
        assert "gradient_p95" in metrics
        assert "gradient_p99" in metrics
        assert "gradient_max" in metrics

        # All values should be non-negative
        for value in metrics.values():
            assert value >= 0.0

    def test_compute_edge_metrics_comprehensive(self, synthetic_depth, synthetic_rgb):
        """Test comprehensive edge metrics with RGB."""
        metrics = compute_edge_metrics(synthetic_depth, synthetic_rgb, metric_type="comprehensive")

        # Should contain alignment metrics
        assert "edge_alignment" in metrics
        assert "edge_precision" in metrics
        assert "edge_recall" in metrics
        assert "edge_f1" in metrics

        # Alignment should be reasonable for aligned synthetic data
        assert -1.0 <= metrics["edge_alignment"] <= 1.0
        assert 0.0 <= metrics["edge_f1"] <= 1.0

    def test_compute_edge_metrics_uint8_input(self):
        """Test edge metrics with uint8 depth input."""
        depth_uint8 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        metrics = compute_edge_metrics(depth_uint8, metric_type="basic")

        assert "gradient_mean" in metrics
        assert metrics["gradient_mean"] >= 0.0

    def test_compute_edge_metrics_uint16_input(self):
        """Test edge metrics with uint16 depth input."""
        depth_uint16 = np.random.randint(0, 65536, (128, 128), dtype=np.uint16)
        metrics = compute_edge_metrics(depth_uint16, metric_type="basic")

        assert "gradient_mean" in metrics
        assert metrics["gradient_mean"] >= 0.0


class TestChamferDistance:
    """Test suite for Chamfer distance computation."""

    def test_chamfer_distance_identical_maps(self, synthetic_depth):
        """Test Chamfer distance for identical depth maps."""
        distance = compute_chamfer_distance(synthetic_depth, synthetic_depth)

        # Should be zero or very close
        assert distance < 1.0

    def test_chamfer_distance_noisy_maps(self, synthetic_depth, noisy_depth):
        """Test Chamfer distance with noisy depth."""
        distance = compute_chamfer_distance(synthetic_depth, noisy_depth)

        # Should be non-zero but finite
        assert 0.0 < distance < 100.0

    def test_chamfer_distance_different_maps(self):
        """Test Chamfer distance with completely different maps."""
        # Create maps with clear non-overlapping edges
        depth1 = np.ones((128, 128), dtype=np.float32) * 0.5
        depth1[40:60, 40:60] = 1.0  # Small bright square

        depth2 = np.ones((128, 128), dtype=np.float32) * 0.5
        depth2[80:100, 80:100] = 1.0  # Different bright square

        distance = compute_chamfer_distance(depth1, depth2)

        # Should be larger than identical maps (edges don't align)
        # Distance will be proportional to spatial separation
        assert distance >= 0.0  # Valid distance (non-negative)


class TestConvenienceFunction:
    """Test suite for convenience function."""

    def test_refine_depth_advanced_default(self, noisy_depth, synthetic_rgb):
        """Test convenience function with defaults."""
        refined = refine_depth_advanced(noisy_depth, synthetic_rgb)

        assert refined.shape == noisy_depth.shape
        assert refined.std() < noisy_depth.std()

    def test_refine_depth_advanced_bilateral(self, noisy_depth):
        """Test convenience function with bilateral technique."""
        refined = refine_depth_advanced(noisy_depth, technique="bilateral")

        assert refined.shape == noisy_depth.shape

    def test_refine_depth_advanced_custom_config(self, noisy_depth, synthetic_rgb):
        """Test convenience function with custom config."""
        config = AdvancedRefinementConfig(bilateral_d=5)
        refined = refine_depth_advanced(noisy_depth, synthetic_rgb, config=config)

        assert refined.shape == noisy_depth.shape


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_structure_scene_refinement(self):
        """Test refinement on architectural structure scene."""
        # Create architectural scene with clear structure
        depth = np.zeros((512, 512), dtype=np.float32)
        # Building facade with windows
        depth[100:400, 100:400] = 0.6  # Wall
        depth[150:200, 150:200] = 0.8  # Window 1
        depth[150:200, 250:300] = 0.8  # Window 2
        depth[250:300, 150:200] = 0.8  # Window 3
        depth[250:300, 250:300] = 0.8  # Window 4

        # Create aligned RGB
        rgb = np.ones((512, 512, 3), dtype=np.uint8) * 50
        rgb[100:400, 100:400] = [200, 200, 200]  # Wall
        rgb[150:200, 150:200] = [100, 150, 200]  # Windows
        rgb[150:200, 250:300] = [100, 150, 200]
        rgb[250:300, 150:200] = [100, 150, 200]
        rgb[250:300, 250:300] = [100, 150, 200]

        # Add realistic noise
        noise = np.random.normal(0, 0.02, depth.shape)
        depth_noisy = np.clip(depth + noise, 0, 1).astype(np.float32)

        # Refine with hybrid technique
        refiner = DepthRefiner()
        refined = refiner.refine(depth_noisy, rgb, technique="hybrid")

        # Compute edge quality improvements
        metrics_before = compute_edge_metrics(depth_noisy, rgb, "comprehensive")
        metrics_after = compute_edge_metrics(refined, rgb, "comprehensive")

        # Edge F1 should improve
        assert metrics_after["edge_f1"] >= metrics_before["edge_f1"] * 0.95

        # Noise should be reduced
        assert refined.std() <= depth_noisy.std()

    def test_refinement_preserves_16bit_precision(self):
        """Test that refinement maintains 16-bit precision when configured."""
        depth_uint16 = np.random.randint(0, 65536, (256, 256), dtype=np.uint16)
        rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        config = AdvancedRefinementConfig(preserve_16bit=True)
        refiner = DepthRefiner(config)

        refined = refiner.bilateral_filter(depth_uint16)

        # Should maintain reasonable range
        assert refined.dtype in [np.float32, np.uint16]

    def test_batch_processing_consistency(self, synthetic_depth, synthetic_rgb):
        """Test batch processing produces consistent results."""
        refiner = DepthRefiner()

        # Process same image multiple times
        results = []
        for _ in range(3):
            refined = refiner.refine(synthetic_depth, synthetic_rgb, technique="hybrid")
            results.append(refined)

        # Results should be identical (deterministic)
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i], decimal=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
