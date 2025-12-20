"""Tests for metric depth conversion utilities."""

import pytest
import numpy as np
from pathlib import Path

# Skip all tests in this module if lux_depth_v3 dependencies are missing
pytest.importorskip("cv2", reason="OpenCV not installed")
pytest.importorskip("torch", reason="PyTorch not installed")

from lux_depth_v3.metric_depth import (
    MetricDepthConverter,
    MetricDepthResult,
    convert_to_metric_depth,
    depth_to_meters,
    get_depth_statistics
)


class TestMetricDepthConverter:
    """Test MetricDepthConverter class."""
    
    def test_metric_conversion_with_intrinsics(self):
        """Test conversion using camera intrinsics."""
        depth = np.random.rand(480, 640) * 10.0
        intrinsics = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        converter = MetricDepthConverter("DA3METRIC-LARGE")
        result = converter.convert(depth, intrinsics=intrinsics)
        
        assert result.depth_meters.shape == depth.shape
        assert result.focal_length_px == 500.0
        assert result.scale_factor == 500.0 / 300.0
        assert not result.already_metric
        assert result.source_model == "DA3METRIC-LARGE"
    
    def test_metric_conversion_with_focal(self):
        """Test conversion using explicit focal length."""
        depth = np.random.rand(480, 640) * 10.0
        focal = 600.0
        
        result = convert_to_metric_depth(
            depth,
            model_name="DA3METRIC-LARGE",
            focal_length_px=focal
        )
        
        assert result.focal_length_px == focal
        assert result.scale_factor == focal / 300.0
        assert not result.already_metric
        
        # Check that conversion was applied
        expected_depth = depth * (focal / 300.0)
        np.testing.assert_array_almost_equal(
            result.depth_meters,
            expected_depth
        )
    
    def test_nested_model_no_conversion(self):
        """Test that nested models don't convert (already metric)."""
        depth = np.random.rand(480, 640) * 10.0
        
        converter = MetricDepthConverter("DA3NESTED-GIANT-LARGE-1.1")
        result = converter.convert(depth)
        
        assert result.already_metric
        np.testing.assert_array_equal(result.depth_meters, depth)
        assert result.scale_factor == 1.0
        assert result.focal_length_px == 0.0
        assert result.source_model == "DA3NESTED-GIANT-LARGE-1.1"
    
    def test_nested_model_v1_no_conversion(self):
        """Test v1.0 nested model (also metric)."""
        depth = np.random.rand(480, 640) * 10.0
        
        converter = MetricDepthConverter("DA3NESTED-GIANT-LARGE")
        result = converter.convert(depth)
        
        assert result.already_metric
        np.testing.assert_array_equal(result.depth_meters, depth)
    
    def test_fov_estimation(self):
        """Test FOV-based focal length estimation."""
        depth = np.random.rand(480, 640) * 10.0
        
        result = convert_to_metric_depth(
            depth,
            model_name="DA3METRIC-LARGE",
            image_width=640,
            fov_degrees=60.0
        )
        
        # Verify estimation is reasonable
        # For 640px width and 60° FOV:
        # focal = (640/2) / tan(60°/2) ≈ 554px
        assert 500.0 < result.focal_length_px < 600.0
        assert result.scale_factor > 0.0
    
    def test_multiple_intrinsics(self):
        """Test with multiple intrinsics matrices (batch)."""
        depth = np.random.rand(2, 480, 640) * 10.0
        intrinsics = np.array([
            [[500.0, 0.0, 320.0],
             [0.0, 500.0, 240.0],
             [0.0, 0.0, 1.0]],
            [[600.0, 0.0, 320.0],
             [0.0, 600.0, 240.0],
             [0.0, 0.0, 1.0]]
        ])
        
        converter = MetricDepthConverter("DA3METRIC-LARGE")
        result = converter.convert(depth, intrinsics=intrinsics)
        
        # Should use first camera's intrinsics
        assert result.focal_length_px == 500.0
    
    def test_conversion_error_no_focal(self):
        """Test that conversion raises error without focal information."""
        depth = np.random.rand(480, 640)
        
        converter = MetricDepthConverter("DA3METRIC-LARGE")
        
        with pytest.raises(ValueError, match="No focal length information"):
            converter.convert(depth)
    
    def test_invalid_intrinsics_shape(self):
        """Test with invalid intrinsics shape."""
        depth = np.random.rand(480, 640) * 10.0
        invalid_intrinsics = np.random.rand(2, 2)  # Wrong shape
        
        # Should fail to extract focal length and raise error
        with pytest.raises(ValueError, match="No focal length information"):
            convert_to_metric_depth(depth, intrinsics=invalid_intrinsics)


class TestMetricDepthResult:
    """Test MetricDepthResult dataclass."""
    
    def test_save_load_metric_result(self, tmp_path):
        """Test saving and loading metric depth results."""
        depth = np.random.rand(100, 100)
        result = convert_to_metric_depth(depth, focal_length_px=500.0)
        
        # Save
        save_path = tmp_path / "metric_depth.npz"
        result.save(save_path)
        
        assert save_path.exists()
        
        # Load
        loaded = MetricDepthResult.load(save_path)
        
        np.testing.assert_array_equal(loaded.depth_meters, result.depth_meters)
        assert loaded.focal_length_px == result.focal_length_px
        assert loaded.scale_factor == result.scale_factor
        assert loaded.source_model == result.source_model
        assert loaded.already_metric == result.already_metric
    
    def test_save_load_preserves_metadata(self, tmp_path):
        """Test that save/load preserves all metadata."""
        depth = np.random.rand(50, 50)
        
        original = MetricDepthResult(
            depth_meters=depth,
            focal_length_px=750.0,
            scale_factor=2.5,
            source_model="DA3METRIC-LARGE",
            already_metric=False
        )
        
        save_path = tmp_path / "test.npz"
        original.save(save_path)
        loaded = MetricDepthResult.load(save_path)
        
        assert loaded.focal_length_px == 750.0
        assert loaded.scale_factor == 2.5
        assert loaded.source_model == "DA3METRIC-LARGE"
        assert loaded.already_metric == False


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_depth_to_meters(self):
        """Test quick conversion function."""
        depth = np.random.rand(100, 100) * 10.0
        focal = 500.0
        
        result = depth_to_meters(depth, focal)
        
        expected = depth * (focal / 300.0)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_depth_statistics(self):
        """Test depth statistics computation."""
        depth_meters = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        stats = get_depth_statistics(depth_meters)
        
        assert stats['min_m'] == 1.0
        assert stats['max_m'] == 6.0
        assert stats['mean_m'] == 3.5
        assert stats['median_m'] == 3.5
        assert stats['range_m'] == 5.0
        assert stats['std_m'] > 0.0
    
    def test_depth_statistics_with_mask(self):
        """Test depth statistics with mask."""
        depth_meters = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        # Mask out values > 3
        mask = depth_meters <= 3.0
        
        stats = get_depth_statistics(depth_meters, mask=mask)
        
        assert stats['min_m'] == 1.0
        assert stats['max_m'] == 3.0
        assert stats['mean_m'] == 2.0
        assert stats['range_m'] == 2.0


class TestScaleConstants:
    """Test scale constants for different models."""
    
    def test_metric_large_scale_constant(self):
        """Test DA3METRIC-LARGE scale constant."""
        converter = MetricDepthConverter("DA3METRIC-LARGE")
        assert converter.scale_constant == 300.0
    
    def test_nested_scale_constant(self):
        """Test nested model scale constant."""
        converter = MetricDepthConverter("DA3NESTED-GIANT-LARGE-1.1")
        assert converter.scale_constant == 1.0
    
    def test_unknown_model_defaults_to_metric(self):
        """Test unknown model defaults to DA3METRIC constant."""
        converter = MetricDepthConverter("UNKNOWN-MODEL")
        assert converter.scale_constant == 300.0
        assert not converter.is_metric_model


class TestBatchProcessing:
    """Test batch processing scenarios."""
    
    def test_batch_depth_conversion(self):
        """Test converting batch of depth maps."""
        batch_depth = np.random.rand(4, 480, 640) * 10.0
        focal = 500.0
        
        # Convert batch
        results = []
        for depth in batch_depth:
            result = convert_to_metric_depth(
                depth,
                focal_length_px=focal
            )
            results.append(result.depth_meters)
        
        # Verify batch consistency
        assert len(results) == 4
        for result in results:
            assert result.shape == (480, 640)
    
    def test_varying_focal_lengths(self):
        """Test with different focal lengths per image."""
        depth = np.random.rand(100, 100)
        focals = [400.0, 500.0, 600.0]
        
        results = []
        for focal in focals:
            result = convert_to_metric_depth(depth, focal_length_px=focal)
            results.append(result)
        
        # Verify different scale factors
        assert results[0].scale_factor < results[1].scale_factor < results[2].scale_factor
        
        # Verify depths are scaled accordingly
        assert np.mean(results[0].depth_meters) < np.mean(results[2].depth_meters)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_depth(self):
        """Test with zero depth values."""
        depth = np.zeros((100, 100))
        result = convert_to_metric_depth(depth, focal_length_px=500.0)
        
        assert np.all(result.depth_meters == 0.0)
        assert result.scale_factor > 0.0
    
    def test_single_pixel(self):
        """Test with single pixel depth."""
        depth = np.array([[5.0]])
        result = convert_to_metric_depth(depth, focal_length_px=500.0)
        
        assert result.depth_meters.shape == (1, 1)
        assert result.depth_meters[0, 0] > 0.0
    
    def test_very_large_focal(self):
        """Test with very large focal length."""
        depth = np.random.rand(100, 100) * 10.0
        focal = 10000.0
        
        result = convert_to_metric_depth(depth, focal_length_px=focal)
        
        assert result.scale_factor == focal / 300.0
        assert np.all(result.depth_meters > depth)  # Should be scaled up
    
    def test_very_small_focal(self):
        """Test with very small focal length."""
        depth = np.random.rand(100, 100) * 10.0
        focal = 50.0
        
        result = convert_to_metric_depth(depth, focal_length_px=focal)
        
        assert result.scale_factor == focal / 300.0
        assert np.all(result.depth_meters < depth)  # Should be scaled down


class TestRealWorldScenarios:
    """Test real-world use cases."""
    
    def test_architectural_measurement(self):
        """Test architectural measurement scenario."""
        # Simulate depth map of a room (5-15 meters)
        depth = np.random.uniform(5.0, 15.0, (1080, 1920))
        
        # Typical camera parameters for architectural photography
        focal_fx = 2000.0  # 4K camera
        focal_fy = 2000.0
        intrinsics = np.array([
            [focal_fx, 0.0, 960.0],
            [0.0, focal_fy, 540.0],
            [0.0, 0.0, 1.0]
        ])
        
        result = convert_to_metric_depth(
            depth,
            model_name="DA3METRIC-LARGE",
            intrinsics=intrinsics
        )
        
        stats = get_depth_statistics(result.depth_meters)
        
        # Verify reasonable architectural distances
        assert stats['min_m'] > 0.0
        assert stats['max_m'] < 100.0  # Room shouldn't be > 100m
        assert stats['mean_m'] > 0.0
    
    def test_exterior_scene(self):
        """Test exterior scene (larger distances)."""
        # Simulate exterior scene (1-50 meters)
        depth = np.random.uniform(1.0, 50.0, (720, 1280))
        
        # Wide-angle lens (24mm equivalent)
        fov = 84.0  # degrees
        
        result = convert_to_metric_depth(
            depth,
            model_name="DA3METRIC-LARGE",
            image_width=1280,
            fov_degrees=fov
        )
        
        stats = get_depth_statistics(result.depth_meters)
        
        # Verify exterior scene distances
        assert stats['min_m'] > 0.0
        assert stats['max_m'] < 200.0
    
    def test_nested_model_architectural(self):
        """Test nested model (already metric)."""
        # Nested model outputs metric depth directly
        metric_depth = np.random.uniform(2.0, 20.0, (480, 640))
        
        result = convert_to_metric_depth(
            metric_depth,
            model_name="DA3NESTED-GIANT-LARGE-1.1"
        )
        
        # Should return unchanged
        np.testing.assert_array_equal(result.depth_meters, metric_depth)
        assert result.already_metric
        
        # Statistics should match input
        stats = get_depth_statistics(result.depth_meters)
        assert stats['min_m'] == np.min(metric_depth)
        assert stats['max_m'] == np.max(metric_depth)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
