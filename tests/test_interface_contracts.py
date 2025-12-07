"""
Tests for interface contracts.

Validates that interface definitions are correct and can be instantiated
for testing purposes.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from transformation_portal.interfaces import (
    ImageProcessor, VideoProcessor, ProcessingError,
    Pipeline, PipelineStage, BatchPipeline, PipelineError,
    Enhancer, AdaptiveEnhancer, EnhancementError,
    Segmenter, MaterialSegmenter, SemanticSegmenter, MaterialType, SegmentationError,
    DepthEstimator, NormalEstimator, UnifiedEstimator, EstimationError
)


# Mock implementations for testing
class MockImageProcessor(ImageProcessor):
    """Mock processor for testing."""
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return image * 1.1
    
    def get_config(self) -> Dict[str, Any]:
        return {"type": "mock", "multiplier": 1.1}


class MockEnhancer(Enhancer):
    """Mock enhancer for testing."""
    
    def enhance(self, image: np.ndarray, strength: float = 1.0, **kwargs) -> np.ndarray:
        self.validate_strength(strength)
        return image * (1.0 + 0.2 * strength)
    
    def get_config(self) -> Dict[str, Any]:
        return {"type": "mock_enhancer"}


class MockSegmenter(Segmenter):
    """Mock segmenter for testing."""
    
    def segment(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        h, w = image.shape[:2]
        return {
            "object1": np.ones((h, w), dtype=bool),
            "object2": np.zeros((h, w), dtype=bool)
        }
    
    def get_supported_categories(self) -> List[str]:
        return ["object1", "object2"]
    
    def get_config(self) -> Dict[str, Any]:
        return {"type": "mock_segmenter"}


class MockDepthEstimator(DepthEstimator):
    """Mock depth estimator for testing."""
    
    def estimate_depth(self, image: np.ndarray, normalize: bool = True, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        depth = np.random.rand(h, w).astype(np.float32)
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        return depth
    
    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "MockDepth", "version": "1.0"}
    
    def get_config(self) -> Dict[str, Any]:
        return {"type": "mock_depth"}


class TestImageProcessorInterface:
    """Test ImageProcessor interface."""
    
    def test_processor_creation(self):
        """Test that mock processor can be created."""
        processor = MockImageProcessor()
        assert isinstance(processor, ImageProcessor)
    
    def test_processor_process(self):
        """Test process method."""
        processor = MockImageProcessor()
        image = np.random.rand(100, 100, 3).astype(np.float32)
        result = processor.process(image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
        np.testing.assert_allclose(result, image * 1.1)
    
    def test_processor_config(self):
        """Test get_config method."""
        processor = MockImageProcessor()
        config = processor.get_config()
        
        assert isinstance(config, dict)
        assert "type" in config


class TestEnhancerInterface:
    """Test Enhancer interface."""
    
    def test_enhancer_creation(self):
        """Test that mock enhancer can be created."""
        enhancer = MockEnhancer()
        assert isinstance(enhancer, Enhancer)
    
    def test_enhancer_enhance(self):
        """Test enhance method with valid strength."""
        enhancer = MockEnhancer()
        image = np.random.rand(50, 50, 3).astype(np.float32)
        
        # Test with different strength values
        result1 = enhancer.enhance(image, strength=0.0)
        result2 = enhancer.enhance(image, strength=1.0)
        
        assert result1.shape == image.shape
        assert result2.shape == image.shape
        np.testing.assert_allclose(result1, image)  # strength=0 -> no change
        np.testing.assert_allclose(result2, image * 1.2)  # strength=1 -> 20% boost
    
    def test_enhancer_strength_validation(self):
        """Test that invalid strength raises error."""
        enhancer = MockEnhancer()
        image = np.random.rand(50, 50, 3).astype(np.float32)
        
        # Invalid strengths
        with pytest.raises(ValueError, match="Strength must be in"):
            enhancer.enhance(image, strength=-0.1)
        
        with pytest.raises(ValueError, match="Strength must be in"):
            enhancer.enhance(image, strength=1.5)


class TestSegmenterInterface:
    """Test Segmenter interface."""
    
    def test_segmenter_creation(self):
        """Test that mock segmenter can be created."""
        segmenter = MockSegmenter()
        assert isinstance(segmenter, Segmenter)
    
    def test_segmenter_segment(self):
        """Test segment method."""
        segmenter = MockSegmenter()
        image = np.random.rand(100, 100, 3).astype(np.float32)
        
        masks = segmenter.segment(image)
        
        assert isinstance(masks, dict)
        assert "object1" in masks
        assert masks["object1"].shape == (100, 100)
        assert masks["object1"].dtype == bool
    
    def test_segmenter_categories(self):
        """Test get_supported_categories method."""
        segmenter = MockSegmenter()
        categories = segmenter.get_supported_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(cat, str) for cat in categories)


class TestMaterialType:
    """Test MaterialType enum."""
    
    def test_material_types_exist(self):
        """Test that expected material types are defined."""
        assert hasattr(MaterialType, 'WOOD')
        assert hasattr(MaterialType, 'METAL')
        assert hasattr(MaterialType, 'GLASS')
        assert hasattr(MaterialType, 'STONE')
        assert hasattr(MaterialType, 'MARBLE')
        assert hasattr(MaterialType, 'FABRIC')
        assert hasattr(MaterialType, 'WATER')
        assert hasattr(MaterialType, 'VEGETATION')
        assert hasattr(MaterialType, 'SKY')
    
    def test_material_values(self):
        """Test material type values."""
        assert MaterialType.WOOD.value == "wood"
        assert MaterialType.METAL.value == "metal"
        assert MaterialType.GLASS.value == "glass"


class TestDepthEstimatorInterface:
    """Test DepthEstimator interface."""
    
    def test_estimator_creation(self):
        """Test that mock depth estimator can be created."""
        estimator = MockDepthEstimator()
        assert isinstance(estimator, DepthEstimator)
    
    def test_estimator_estimate_depth(self):
        """Test estimate_depth method."""
        estimator = MockDepthEstimator()
        image = np.random.rand(100, 100, 3).astype(np.float32)
        
        depth = estimator.estimate_depth(image)
        
        assert isinstance(depth, np.ndarray)
        assert depth.shape == (100, 100)
        assert depth.dtype == np.float32
        assert 0.0 <= depth.min() <= depth.max() <= 1.0
    
    def test_estimator_normalize_flag(self):
        """Test normalize parameter."""
        estimator = MockDepthEstimator()
        image = np.random.rand(50, 50, 3).astype(np.float32)
        
        depth_normalized = estimator.estimate_depth(image, normalize=True)
        depth_raw = estimator.estimate_depth(image, normalize=False)
        
        assert depth_normalized.shape == depth_raw.shape
    
    def test_estimator_model_info(self):
        """Test get_model_info method."""
        estimator = MockDepthEstimator()
        info = estimator.get_model_info()
        
        assert isinstance(info, dict)
        assert "name" in info or "version" in info
    
    def test_estimator_invert_depth(self):
        """Test depth inversion utility method."""
        estimator = MockDepthEstimator()
        image = np.random.rand(50, 50, 3).astype(np.float32)
        
        depth = estimator.estimate_depth(image)
        inverted = estimator.invert_depth(depth)
        
        np.testing.assert_allclose(inverted, 1.0 - depth)


class TestInterfaceExceptions:
    """Test interface-specific exceptions."""
    
    def test_exceptions_exist(self):
        """Test that custom exception classes are defined."""
        assert issubclass(ProcessingError, Exception)
        assert issubclass(PipelineError, Exception)
        assert issubclass(EnhancementError, Exception)
        assert issubclass(SegmentationError, Exception)
        assert issubclass(EstimationError, Exception)
    
    def test_exception_can_be_raised(self):
        """Test that exceptions can be raised and caught."""
        with pytest.raises(ProcessingError):
            raise ProcessingError("Test error")
        
        with pytest.raises(EnhancementError):
            raise EnhancementError("Enhancement failed")


class TestInterfaceImports:
    """Test that all interfaces can be imported."""
    
    def test_all_exports_available(self):
        """Test that __all__ exports are accessible."""
        from transformation_portal import interfaces
        
        # Check core interfaces
        assert hasattr(interfaces, 'ImageProcessor')
        assert hasattr(interfaces, 'Pipeline')
        assert hasattr(interfaces, 'Enhancer')
        assert hasattr(interfaces, 'Segmenter')
        assert hasattr(interfaces, 'DepthEstimator')
        
        # Check extended interfaces
        assert hasattr(interfaces, 'AdaptiveEnhancer')
        assert hasattr(interfaces, 'MaterialSegmenter')
        assert hasattr(interfaces, 'SemanticSegmenter')
        assert hasattr(interfaces, 'NormalEstimator')
        assert hasattr(interfaces, 'UnifiedEstimator')
        
        # Check enums
        assert hasattr(interfaces, 'MaterialType')
        
        # Check exceptions
        assert hasattr(interfaces, 'ProcessingError')
        assert hasattr(interfaces, 'PipelineError')
        assert hasattr(interfaces, 'EnhancementError')
        assert hasattr(interfaces, 'SegmentationError')
        assert hasattr(interfaces, 'EstimationError')
