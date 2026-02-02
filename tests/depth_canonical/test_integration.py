"""Integration tests for Phase 2 depth estimation."""

import os
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth_canonical import DepthPipeline
from transformation_portal.depth_canonical.config import (
    UnifiedDepthConfig,
    ModelConfig,
    ModelVariant,
    DeviceType,
    ProcessingConfig,
    PBRConfig,
)


# Check if we should mock transformers (when TRANSFORMERS_OFFLINE=1)
TRANSFORMERS_OFFLINE = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'


@pytest.fixture(autouse=True)
def mock_transformers_pipeline():
    """Mock transformers pipeline when running offline.

    This prevents tests from downloading models during CI runs
    while still testing the pipeline logic.
    """
    if not TRANSFORMERS_OFFLINE:
        yield None
        return

    # Create a mock pipeline that returns depth map dict
    def mock_pipeline_factory(task=None, model=None, device=None):
        """Factory that creates a mock HF pipeline."""
        def mock_pipe(image):
            """Mock pipeline call that returns depth dict."""
            # Return a dict with 'depth' key containing a numpy array
            # Size should match input image
            if isinstance(image, Image.Image):
                size = image.size
                depth = np.random.rand(size[1], size[0]).astype(np.float32)
            else:
                depth = np.random.rand(512, 512).astype(np.float32)

            return {"depth": depth}

        return mock_pipe

    # Patch the pipeline import in the da3_wrapper module
    with patch('transformers.pipeline', side_effect=mock_pipeline_factory):
        yield


@pytest.fixture
def test_image():
    """Create a simple test image."""
    # Create a 512x512 RGB gradient image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        img[i, :, 0] = int(i / 512 * 255)  # Red gradient
        img[:, i, 1] = int(i / 512 * 255)  # Green gradient
    img[:, :, 2] = 128  # Constant blue
    return Image.fromarray(img)


@pytest.fixture
def simple_config():
    """Create a simple config for testing."""
    return UnifiedDepthConfig(
        model=ModelConfig(
            variant=ModelVariant.DA3_SMALL,  # Use small for speed
            device=DeviceType.CPU,
        ),
        processing=ProcessingConfig(
            pbr=PBRConfig(enabled=False)  # Disable PBR for pure depth tests
        )
    )


@pytest.fixture
def pbr_config():
    """Create config with PBR enabled."""
    return UnifiedDepthConfig(
        model=ModelConfig(
            variant=ModelVariant.DA3_SMALL,
            device=DeviceType.CPU,
        ),
        processing=ProcessingConfig(
            pbr=PBRConfig(
                enabled=True,
                normal_strength=1.0,
                roughness_strength=1.0,
                ao_strength=1.0,
            )
        )
    )


@pytest.mark.slow
@pytest.mark.integration
def test_depth_estimation_from_pil_image(test_image, simple_config):
    """Test depth estimation from PIL Image."""
    pipeline = DepthPipeline(simple_config)

    result = pipeline.process(image=test_image)

    # Check depth map was generated
    assert result.depth_map is not None
    assert isinstance(result.depth_map, np.ndarray)
    assert result.depth_map.ndim == 2
    assert result.depth_map.shape[0] > 0
    assert result.depth_map.shape[1] > 0

    # Check normalization [0, 1]
    assert result.depth_map.min() >= 0.0
    assert result.depth_map.max() <= 1.0


@pytest.mark.slow
@pytest.mark.integration
def test_depth_estimation_from_numpy_array(simple_config):
    """Test depth estimation from numpy array."""
    # Create test image as numpy array
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    pipeline = DepthPipeline(simple_config)
    result = pipeline.process(image=img_array)

    assert result.depth_map is not None
    assert isinstance(result.depth_map, np.ndarray)
    assert result.depth_map.ndim == 2


@pytest.mark.slow
@pytest.mark.integration
def test_depth_estimation_with_pbr_generation(test_image, pbr_config, tmp_path):
    """Test full pipeline: depth estimation + PBR generation."""
    pipeline = DepthPipeline(pbr_config)

    result = pipeline.process(
        image=test_image,
        output_dir=tmp_path,
        basename="test"
    )

    # Check depth map
    assert result.depth_map is not None

    # Check PBR maps generated
    assert result.pbr_maps is not None
    assert "normal" in result.pbr_maps
    assert "roughness" in result.pbr_maps
    assert "ao" in result.pbr_maps

    # Check PBR files saved
    assert result.pbr_paths is not None
    assert (tmp_path / "test_normal.png").exists()
    assert (tmp_path / "test_roughness.png").exists()
    assert (tmp_path / "test_ao.png").exists()


@pytest.mark.slow
@pytest.mark.integration
def test_depth_caching(test_image, simple_config):
    """Test that depth maps are cached."""
    pipeline = DepthPipeline(simple_config)

    # First call - should estimate
    result1 = pipeline.process(image=test_image)
    depth1 = result1.depth_map

    # Second call with same image - should use cache
    result2 = pipeline.process(image=test_image)
    depth2 = result2.depth_map

    # Results should be identical (from cache)
    assert np.allclose(depth1, depth2)


@pytest.mark.slow
@pytest.mark.integration
def test_batch_processing(simple_config, tmp_path):
    """Test batch processing of multiple images."""
    # Create multiple test images
    images = []
    for i in range(3):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_{i}.png"
        Image.fromarray(img).save(img_path)
        images.append(img_path)

    pipeline = DepthPipeline(simple_config)

    results = pipeline.batch_process(
        images=images,
        output_dir=tmp_path / "output"
    )

    # Check all results
    assert len(results) == 3
    for result in results:
        assert result.depth_map is not None
        assert result.depth_map.ndim == 2


@pytest.mark.integration
def test_backward_compatibility_with_provided_depth(pbr_config, tmp_path):
    """Test backward compatibility: providing depth_map still works."""
    # Create pre-computed depth map
    depth_map = np.random.rand(512, 512).astype(np.float32)

    pipeline = DepthPipeline(pbr_config)

    result = pipeline.process(
        depth_map=depth_map,
        output_dir=tmp_path,
        basename="test"
    )

    # Should use provided depth map
    assert np.array_equal(result.depth_map, depth_map)

    # PBR maps should still be generated
    assert result.pbr_maps is not None


def test_cache_key_generation(simple_config):
    """Test cache key generation for different images."""
    pipeline = DepthPipeline(simple_config)

    # Different images should have different cache keys
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

    key1 = pipeline._generate_cache_key(img1)
    key2 = pipeline._generate_cache_key(img2)

    assert key1 != key2


def test_error_handling_no_input(simple_config):
    """Test error handling when no input provided."""
    pipeline = DepthPipeline(simple_config)

    with pytest.raises(ValueError, match="Either image or depth_map must be provided"):
        pipeline.process()
