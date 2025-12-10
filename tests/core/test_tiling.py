"""Tests for tiled processing."""

import pytest

torch = pytest.importorskip("torch", reason="torch required for tiling tests")

from src.transformation_portal.core.processing.tiling import (
    TiledProcessor,
    TileConfig
)


def test_tile_config_validation():
    """Test tile configuration validation."""
    # Valid config
    config = TileConfig(tile_size=512, overlap=64)
    assert config.tile_size == 512
    assert config.overlap == 64
    
    # Invalid: overlap >= tile_size
    with pytest.raises(ValueError, match="Overlap.*must be less than"):
        TileConfig(tile_size=128, overlap=128)
    
    # Invalid: tile_size too small
    with pytest.raises(ValueError, match="at least"):
        TileConfig(tile_size=100, overlap=10)
    
    # Invalid blend mode
    with pytest.raises(ValueError, match="Invalid blend_mode"):
        TileConfig(tile_size=512, overlap=64, blend_mode="invalid")


def test_tiled_processor_small_image():
    """Test that small images are processed directly."""
    processor = TiledProcessor(tile_size=512, overlap=64)
    
    # Small image that fits in one tile
    image = torch.randn(1, 3, 256, 256)
    
    def identity_fn(x):
        return x
    
    result = processor.process(image, identity_fn)
    
    assert result.shape == image.shape
    assert torch.allclose(result, image)


def test_tiled_processor_large_image():
    """Test processing large image with tiling."""
    processor = TiledProcessor(tile_size=256, overlap=32)
    
    # Large image requiring tiling
    image = torch.randn(1, 3, 512, 512)
    
    def identity_fn(x):
        return x * 2
    
    result = processor.process(image, identity_fn)
    
    assert result.shape == image.shape
    # Result should be approximately 2x input (with blending, edges may differ)
    # Check center region where blending is minimal
    center_result = result[:, :, 100:400, 100:400]
    center_expected = (image * 2)[:, :, 100:400, 100:400]
    assert torch.allclose(center_result, center_expected, atol=0.1)


def test_tiled_processor_single_image_input():
    """Test processing single image (no batch dimension)."""
    processor = TiledProcessor(tile_size=256, overlap=32)
    
    # Single image [C, H, W]
    image = torch.randn(3, 256, 256)
    
    def identity_fn(x):
        return x
    
    result = processor.process(image, identity_fn)
    
    # Output should have same shape (no batch dimension)
    assert result.shape == image.shape


def test_tiled_processor_estimate_tiles():
    """Test tile count estimation."""
    processor = TiledProcessor(tile_size=256, overlap=32)
    
    # Small image: 1 tile
    assert processor.estimate_tiles(100, 100) == 1
    
    # Large image: multiple tiles
    num_tiles = processor.estimate_tiles(512, 512)
    assert num_tiles > 1


def test_tiled_processor_blend_modes():
    """Test different blending modes."""
    image = torch.randn(1, 3, 512, 512)
    
    def identity_fn(x):
        return x
    
    # Linear blending
    processor_linear = TiledProcessor(tile_size=256, overlap=32, blend_mode="linear")
    result_linear = processor_linear.process(image, identity_fn)
    assert result_linear.shape == image.shape
    
    # Gaussian blending
    processor_gauss = TiledProcessor(tile_size=256, overlap=32, blend_mode="gaussian")
    result_gauss = processor_gauss.process(image, identity_fn)
    assert result_gauss.shape == image.shape
    
    # No blending
    processor_none = TiledProcessor(tile_size=256, overlap=32, blend_mode="none")
    result_none = processor_none.process(image, identity_fn)
    assert result_none.shape == image.shape


def test_tiled_processor_calculate_tiles():
    """Test tile calculation."""
    processor = TiledProcessor(tile_size=256, overlap=32)
    
    # Calculate tiles for 512x512 image
    tiles = processor._calculate_tiles(512, 512)
    
    # Verify all tiles are valid
    for y1, y2, x1, x2 in tiles:
        assert 0 <= y1 < y2 <= 512
        assert 0 <= x1 < x2 <= 512
        assert (y2 - y1) <= 256
        assert (x2 - x1) <= 256


def test_tiled_processor_create_blend_weight():
    """Test blend weight creation."""
    processor = TiledProcessor(tile_size=256, overlap=32)
    
    # Linear weight
    weight = processor._create_blend_weight(256, 256, torch.device("cpu"))
    assert weight.shape == (1, 1, 256, 256)
    assert weight.min() >= 0
    assert weight.max() <= 1
    
    # Gaussian weight
    processor.config.blend_mode = "gaussian"
    weight = processor._create_blend_weight(256, 256, torch.device("cpu"))
    assert weight.shape == (1, 1, 256, 256)
    assert weight.min() >= 0
    assert weight.max() <= 1
    
    # No blending
    processor.config.blend_mode = "none"
    weight = processor._create_blend_weight(256, 256, torch.device("cpu"))
    assert weight.shape == (1, 1, 256, 256)
    assert torch.all(weight == 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tiled_processor_cuda():
    """Test tiled processing on CUDA."""
    processor = TiledProcessor(tile_size=256, overlap=32)
    
    # Move image to CUDA
    image = torch.randn(1, 3, 512, 512, device="cuda")
    
    def identity_fn(x):
        assert x.device.type == "cuda"
        return x * 2
    
    result = processor.process(image, identity_fn)
    
    assert result.device.type == "cuda"
    assert result.shape == image.shape
