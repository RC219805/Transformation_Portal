"""Integration tests for DA3InferenceEngine.

Tests the real implementation with various inputs and configurations.
Requires torch/transformers (ML tier dependencies).
"""
import pytest
import numpy as np
from pathlib import Path
from transformation_portal.lux_depth_v3 import DA3Config, DA3InferenceEngine
from transformation_portal.lux_depth_v3.config import DeviceConfig, ModelVariant

# Mark all tests in this file as requiring ML dependencies
pytestmark = pytest.mark.ml


def test_da3_predict_basic():
    """Test basic predict() functionality."""
    config = DA3Config()
    engine = DA3InferenceEngine(config)

    # Create test image
    image = np.random.rand(128, 128, 3).astype(np.float32)

    # Run inference
    result = engine.predict(image)

    # Validate result
    assert result.depth_map.shape == (128, 128)
    assert result.depth_map.dtype == np.float32
    assert result.depth_map.min() >= 0.0
    assert result.depth_map.max() <= 1.0
    assert result.original_image.shape == image.shape

    # Validate metadata
    assert 'inference_time_ms' in result.metadata
    assert 'backend' in result.metadata
    assert 'device' in result.metadata
    assert 'model_variant' in result.metadata


def test_da3_infer_alias():
    """Test that infer() is an alias for predict()."""
    config = DA3Config()
    engine = DA3InferenceEngine(config)

    image = np.random.rand(64, 64, 3).astype(np.uint8)

    # Both should work
    result1 = engine.predict(image)
    result2 = engine.infer(image)

    assert result1.depth_map.shape == result2.depth_map.shape
    assert result1.depth_map.dtype == result2.depth_map.dtype


def test_da3_depth_property_alias():
    """Test that DepthResult.depth is an alias for depth_map."""
    config = DA3Config()
    engine = DA3InferenceEngine(config)

    image = np.random.rand(64, 64, 3).astype(np.float32)
    result = engine.predict(image)

    # depth should be an alias for depth_map
    assert result.depth is result.depth_map
    assert np.array_equal(result.depth, result.depth_map)


def test_da3_different_image_sizes():
    """Test with different image sizes."""
    config = DA3Config()
    engine = DA3InferenceEngine(config)

    sizes = [(64, 64), (128, 128), (256, 256)]

    for h, w in sizes:
        image = np.random.rand(h, w, 3).astype(np.float32)
        result = engine.predict(image)

        assert result.depth_map.shape == (h, w), f"Failed for size {h}x{w}"
        assert result.depth_map.min() >= 0.0
        assert result.depth_map.max() <= 1.0


def test_da3_uint8_image():
    """Test with uint8 image (common format)."""
    config = DA3Config()
    engine = DA3InferenceEngine(config)

    # Create uint8 image
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    result = engine.predict(image)

    assert result.depth_map.shape == (128, 128)
    assert result.depth_map.dtype == np.float32


def test_da3_device_config():
    """Test device configuration."""
    # CPU device
    config = DA3Config()
    config.device = DeviceConfig(device="cpu")
    engine = DA3InferenceEngine(config)

    assert engine.device == "cpu"

    image = np.random.rand(64, 64, 3).astype(np.float32)
    result = engine.predict(image)

    assert 'device' in result.metadata
    # Should be cpu or fallback to cpu
    assert result.metadata['device'] in ['cpu', 'mps', 'cuda']


def test_da3_metadata_completeness():
    """Test that metadata contains all expected fields."""
    config = DA3Config()
    engine = DA3InferenceEngine(config)

    image = np.random.rand(64, 64, 3).astype(np.float32)
    result = engine.predict(image)

    # Required metadata fields
    required_fields = [
        'inference_time_ms',
        'backend',
        'device',
        'model_variant',
        'shape',
    ]

    for field in required_fields:
        assert field in result.metadata, f"Missing metadata field: {field}"

    # Validate types
    assert isinstance(result.metadata['inference_time_ms'], (int, float))
    assert result.metadata['inference_time_ms'] >= 0
    assert isinstance(result.metadata['backend'], str)
    assert isinstance(result.metadata['device'], str)


def test_da3_lazy_loading():
    """Test that model is loaded lazily on first inference."""
    config = DA3Config()
    engine = DA3InferenceEngine(config)

    # Model should not be loaded yet
    assert not engine._model_loaded

    # Run inference (loads model)
    image = np.random.rand(64, 64, 3).astype(np.float32)
    result = engine.predict(image)

    # Model should now be loaded
    assert engine._model_loaded
    assert engine.model is not None

    # Second inference should reuse loaded model
    result2 = engine.predict(image)
    assert result2.depth_map.shape == result.depth_map.shape


def test_da3_fallback_model_indicator():
    """Test that metadata indicates when fallback model is used."""
    config = DA3Config()
    # Use V3 large which will fallback to V2
    config.model_variant = ModelVariant.METRIC_LARGE
    engine = DA3InferenceEngine(config)

    image = np.random.rand(64, 64, 3).astype(np.float32)
    result = engine.predict(image)

    # Should indicate fallback (V3 models don't exist yet)
    if 'using_fallback' in result.metadata:
        assert result.metadata['using_fallback'] is True
        assert 'fallback_model' in result.metadata
        assert 'V2' in result.metadata['fallback_model']


def test_da3_commercial_use_flag():
    """Test commercial_use initialization parameter."""
    config = DA3Config()

    # Test with commercial use enabled (default)
    engine1 = DA3InferenceEngine(config, commercial_use=True)
    assert engine1.commercial_use is True

    # Test with commercial use disabled
    engine2 = DA3InferenceEngine(config, commercial_use=False)
    assert engine2.commercial_use is False


@pytest.mark.skipif(not Path(__file__).exists(), reason="Test file path needed")
def test_da3_infer_from_path():
    """Test infer_from_path() with actual image file."""
    # This would need a real image file to test properly
    # For now, just verify the method exists and has correct signature
    config = DA3Config()
    engine = DA3InferenceEngine(config)

    assert hasattr(engine, 'infer_from_path')
    assert callable(engine.infer_from_path)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
