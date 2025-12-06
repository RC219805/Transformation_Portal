"""
Tests for Depth Processor Integration Module
=============================================
"""

import numpy as np
import pytest
from pathlib import Path

try:
    from utils.depth_processor import (
        DepthProcessor,
        DepthConfig,
        create_depth_processor
    )
    DEPTH_PROCESSOR_AVAILABLE = True
except ImportError:
    DEPTH_PROCESSOR_AVAILABLE = False


@pytest.mark.skipif(not DEPTH_PROCESSOR_AVAILABLE, reason="Depth processor not available")
class TestDepthConfig:
    """Test depth configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DepthConfig()
        assert config.model_name == "depth_anything_v2"
        assert config.tile_size == 518
        assert config.enable_zone_processing is True
        assert config.device in ("auto", "cpu", "cuda", "mps")
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DepthConfig(
            model_name="custom_model",
            tile_size=1024,
            enable_zone_processing=False,
            foreground_boost=1.5,
            device="cpu"
        )
        assert config.model_name == "custom_model"
        assert config.tile_size == 1024
        assert config.enable_zone_processing is False
        assert config.foreground_boost == 1.5
        assert config.device == "cpu"


@pytest.mark.skipif(not DEPTH_PROCESSOR_AVAILABLE, reason="Depth processor not available")
class TestDepthProcessor:
    """Test depth processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create test processor."""
        config = DepthConfig(device="cpu")
        return DepthProcessor(config)
    
    @pytest.fixture
    def sample_image(self):
        """Create sample RGB image."""
        return np.random.rand(256, 256, 3).astype(np.float32)
    
    def test_processor_initialization(self, processor):
        """Test processor initializes correctly."""
        assert processor.config.device == "cpu"
        assert processor.depth_model is None  # Lazy loading
    
    def test_create_zone_masks(self, processor):
        """Test zone mask creation."""
        depth_map = np.random.rand(256, 256).astype(np.float32)
        
        foreground, midground, background = processor.create_zone_masks(depth_map)
        
        assert foreground.shape == depth_map.shape
        assert midground.shape == depth_map.shape
        assert background.shape == depth_map.shape
        
        # Check masks sum to ~1
        total = foreground + midground + background
        assert np.allclose(total, 1.0, atol=0.01)
        
        # Check value ranges
        assert np.all(foreground >= 0) and np.all(foreground <= 1)
        assert np.all(midground >= 0) and np.all(midground <= 1)
        assert np.all(background >= 0) and np.all(background <= 1)
    
    def test_apply_zone_adjustments(self, processor, sample_image):
        """Test zone-based adjustments."""
        depth_map = np.random.rand(256, 256).astype(np.float32)
        
        enhanced = processor.apply_zone_adjustments(sample_image, depth_map)
        
        assert enhanced.shape == sample_image.shape
        assert np.all(enhanced >= 0) and np.all(enhanced <= 1)
    
    def test_apply_zone_adjustments_disabled(self, sample_image):
        """Test with zone processing disabled."""
        config = DepthConfig(enable_zone_processing=False, device="cpu")
        processor = DepthProcessor(config)
        
        depth_map = np.random.rand(256, 256).astype(np.float32)
        enhanced = processor.apply_zone_adjustments(sample_image, depth_map)
        
        # Should return unchanged image
        assert np.array_equal(enhanced, sample_image)


@pytest.mark.skipif(not DEPTH_PROCESSOR_AVAILABLE, reason="Depth processor not available")
class TestDepthProcessorConvenience:
    """Test convenience functions."""
    
    def test_create_depth_processor(self):
        """Test convenience function."""
        processor = create_depth_processor(
            model_name="depth_anything_v2",
            enable_zone_processing=True,
            device="cpu"
        )
        
        assert isinstance(processor, DepthProcessor)
        assert processor.config.model_name == "depth_anything_v2"
        assert processor.config.enable_zone_processing is True
        assert processor.config.device == "cpu"


@pytest.mark.skipif(not DEPTH_PROCESSOR_AVAILABLE, reason="Depth processor not available")
class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_image(self):
        """Test with minimal image."""
        processor = create_depth_processor(device="cpu")
        image = np.zeros((1, 1, 3), dtype=np.float32)
        
        # Should not crash
        enhanced, depth_map = processor.process(image)
        assert enhanced.shape == image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
