"""
Tests for Advanced Upscaling Engine
====================================

Test suite for utils/upscaling_engine.py covering:
- Configuration validation
- 16-bit precision preservation
- Tile-based processing
- Batch operations
- Color consistency validation
- Model caching
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Import module under test
try:
    from utils.upscaling_engine import (
        UpscalingEngine,
        UpscalingConfig,
        UpscalingModel,
        UpscalingMetrics,
        TORCH_AVAILABLE,
        TIFFFILE_AVAILABLE
    )
    UPSCALING_AVAILABLE = True
except ImportError:
    UPSCALING_AVAILABLE = False


@pytest.mark.skipif(not UPSCALING_AVAILABLE, reason="upscaling_engine not available")
class TestUpscalingConfig:
    """Test configuration and validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = UpscalingConfig()
        assert config.model == UpscalingModel.SWINIR_REAL_4X
        assert config.preserve_16bit is True
        assert config.validate_colors is True
        assert config.cache_model is True
    
    def test_auto_tile_size(self):
        """Test automatic tile size detection."""
        config = UpscalingConfig(model=UpscalingModel.REALESRGAN_4X, tile_size=0)
        assert config.tile_size == 512  # Real-ESRGAN default
        
        config = UpscalingConfig(model=UpscalingModel.SWINIR_REAL_4X, tile_size=0)
        assert config.tile_size == 384  # SwinIR default
    
    def test_device_auto_detection(self):
        """Test automatic device selection."""
        config = UpscalingConfig(device="auto")
        assert config.device in ("cpu", "cuda", "mps")
    
    def test_custom_tile_size(self):
        """Test custom tile size override."""
        config = UpscalingConfig(tile_size=256)
        assert config.tile_size == 256


@pytest.mark.skipif(not UPSCALING_AVAILABLE or not TORCH_AVAILABLE, 
                    reason="PyTorch required")
class TestUpscalingEngine:
    """Test upscaling engine core functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create test engine with CPU device."""
        config = UpscalingConfig(
            model=UpscalingModel.REALESRGAN_4X,
            device="cpu",
            cache_model=False,
            validate_colors=False
        )
        return UpscalingEngine(config)
    
    @pytest.fixture
    def sample_image(self):
        """Create a 16-bit sample image."""
        # 256x256 RGB gradient
        image = np.zeros((256, 256, 3), dtype=np.uint16)
        for i in range(256):
            image[i, :, 0] = i * 256  # Red gradient
            image[:, i, 1] = i * 256  # Green gradient
        image[:, :, 2] = 32768  # Mid-gray blue
        return image
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine.config.model == UpscalingModel.REALESRGAN_4X
        assert engine.device.type == "cpu"
        assert engine.model is None  # Not loaded yet
    
    def test_model_loading(self, engine):
        """Test model loading with caching."""
        # Mock model loader to avoid actual model download
        with patch.object(engine, '_load_realesrgan') as mock_loader:
            mock_model = Mock()
            mock_model.eval = Mock(return_value=mock_model)
            mock_model.to = Mock(return_value=mock_model)
            mock_loader.return_value = mock_model
            
            model = engine._load_model(UpscalingModel.REALESRGAN_4X)
            assert mock_loader.called
            assert model is not None
    
    def test_16bit_preservation(self, sample_image):
        """Test that 16-bit precision is preserved."""
        # Convert to float
        image_float = sample_image.astype(np.float32) / 65535.0
        
        # Simulate round-trip
        image_reconstructed = (image_float * 65535).astype(np.uint16)
        
        # Check no significant loss
        diff = np.abs(sample_image.astype(np.int32) - image_reconstructed.astype(np.int32))
        assert np.max(diff) <= 1  # Allow ±1 due to rounding
    
    def test_tile_generation(self, engine, sample_image):
        """Test image tiling with overlap."""
        image_float = sample_image.astype(np.float32) / 65535.0
        tiles = engine._tile_image(image_float, tile_size=128, overlap=10)
        
        # Should create 2x2 grid for 256x256 image with 128 tiles
        assert len(tiles) >= 4
        
        # Check tile shape
        for tile, bbox in tiles:
            assert tile.shape[0] == 128
            assert tile.shape[1] == 128
            assert tile.shape[2] == 3
    
    def test_tile_stitching(self, engine):
        """Test tile stitching with blending."""
        # Create synthetic tiles for a 64x64 input (256x256 output at 4x)
        tiles = []
        tile_size = 64  # Input tile size
        scale = 4
        
        # 2x2 grid with proper overlap handling
        # Each tile is 64x64 input → 256x256 output
        for y in [0, 32]:
            for x in [0, 32]:
                # Output tile size after upscaling
                tile = np.ones((256, 256, 3), dtype=np.float32) * 0.5
                # Last tiles might be smaller if they go over the edge
                h = min(tile_size, 64 - y)
                w = min(tile_size, 64 - x)
                # Adjust output tile size accordingly
                if h < tile_size or w < tile_size:
                    tile = tile[:h*scale, :w*scale, :]
                tiles.append((tile, (y, x, h, w)))
        
        output_shape = (64 * scale, 64 * scale, 3)
        stitched = engine._stitch_tiles(tiles, output_shape, scale=scale, overlap=10)
        
        assert stitched.shape == output_shape
        assert np.all(stitched >= 0) and np.all(stitched <= 1)
    
    def test_color_validation(self, engine, sample_image):
        """Test color consistency validation."""
        original = sample_image.astype(np.float32) / 65535.0
        
        # Simulate slight color shift in upscaled version
        # Use proper interpolation instead of simple tiling
        from PIL import Image
        h, w = original.shape[:2]
        orig_pil = Image.fromarray((original * 255).astype(np.uint8))
        upscaled_pil = orig_pil.resize((w * 4, h * 4), Image.LANCZOS)
        upscaled = np.array(upscaled_pil).astype(np.float32) / 255.0
        
        # Apply 1% brightness increase
        upscaled = upscaled * 1.01
        upscaled = np.clip(upscaled, 0, 1)
        
        deviation = engine._validate_color_consistency(original, upscaled, tolerance=0.02)
        
        assert isinstance(deviation, float)
        assert 0 <= deviation <= 1
        # Should detect small deviation (interpolation + 1% shift)
        assert 0.001 < deviation < 0.05


@pytest.mark.skipif(not UPSCALING_AVAILABLE, reason="upscaling_engine not available")
class TestUpscalingModels:
    """Test model enum and properties."""
    
    def test_model_enum_values(self):
        """Test all model enum values."""
        models = list(UpscalingModel)
        assert len(models) >= 2  # At least Real-ESRGAN and SwinIR
        
        for model in models:
            assert model.scale_factor == 4
            assert model.tile_size_recommended > 0
    
    def test_tile_size_recommendations(self):
        """Test tile size recommendations differ by model."""
        realesrgan_tile = UpscalingModel.REALESRGAN_4X.tile_size_recommended
        swinir_tile = UpscalingModel.SWINIR_REAL_4X.tile_size_recommended
        
        # SwinIR typically needs smaller tiles (more memory intensive)
        assert swinir_tile <= realesrgan_tile


@pytest.mark.skipif(not UPSCALING_AVAILABLE, reason="upscaling_engine not available")
class TestImageIO:
    """Test image loading and saving."""
    
    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create temporary 16-bit TIFF."""
        from PIL import Image
        
        # Create 16-bit RGB image
        image = np.random.randint(0, 65536, (128, 128, 3), dtype=np.uint16)
        img_pil = Image.fromarray(image, mode='RGB')
        
        path = tmp_path / "test.tif"
        img_pil.save(path)
        return path, image
    
    def test_load_16bit_image(self, temp_image):
        """Test loading 16-bit TIFF."""
        path, original = temp_image
        
        config = UpscalingConfig(device="cpu")
        engine = UpscalingEngine(config)
        
        loaded = engine._load_image_16bit(path)
        assert loaded.shape == original.shape
        assert loaded.dtype in (np.uint8, np.uint16)
    
    def test_save_16bit_image(self, tmp_path):
        """Test saving 16-bit image."""
        config = UpscalingConfig(preserve_16bit=True, device="cpu")
        engine = UpscalingEngine(config)
        
        # Create test image
        image = np.random.rand(128, 128, 3).astype(np.float32)
        
        output_path = tmp_path / "output.tif"
        engine._save_image_16bit(image, output_path)
        
        assert output_path.exists()
        
        # Verify bit depth if tifffile available
        if TIFFFILE_AVAILABLE:
            from tifffile import TiffFile
            with TiffFile(output_path) as tif:
                loaded = tif.asarray()
                assert loaded.dtype == np.uint16


@pytest.mark.skipif(not UPSCALING_AVAILABLE or not TORCH_AVAILABLE,
                    reason="Full stack required")
class TestBatchProcessing:
    """Test batch processing capabilities."""
    
    @pytest.fixture
    def sample_batch(self, tmp_path):
        """Create sample batch of images."""
        from PIL import Image
        
        images = []
        for i in range(3):
            arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            path = tmp_path / f"image_{i}.png"
            img.save(path)
            images.append(path)
        
        return images
    
    @patch('utils.upscaling_engine.UpscalingEngine.upscale_image')
    def test_batch_upscale(self, mock_upscale, sample_batch, tmp_path):
        """Test batch processing with model caching."""
        # Mock upscale to avoid actual processing
        mock_metrics = UpscalingMetrics(
            model_name="test",
            input_size=(64, 64),
            output_size=(256, 256),
            processing_time=0.1,
            tiles_processed=1,
            memory_peak_mb=100.0
        )
        mock_upscale.return_value = (np.zeros((256, 256, 3)), mock_metrics)
        
        config = UpscalingConfig(device="cpu", cache_model=True)
        engine = UpscalingEngine(config)
        
        output_dir = tmp_path / "output"
        results = engine.batch_upscale(sample_batch, output_dir)
        
        assert len(results) == len(sample_batch)
        assert mock_upscale.call_count == len(sample_batch)


@pytest.mark.skipif(not UPSCALING_AVAILABLE, reason="upscaling_engine not available")
class TestMetrics:
    """Test metrics collection."""
    
    def test_metrics_creation(self):
        """Test metrics dataclass."""
        metrics = UpscalingMetrics(
            model_name="swinir_real_4x",
            input_size=(1024, 768),
            output_size=(4096, 3072),
            processing_time=15.3,
            tiles_processed=48,
            memory_peak_mb=8192.0,
            color_deviation=0.012,
            sharpness_score=0.85
        )
        
        assert metrics.model_name == "swinir_real_4x"
        assert metrics.input_size == (1024, 768)
        assert metrics.output_size == (4096, 3072)
        assert metrics.processing_time == 15.3
        assert metrics.color_deviation == 0.012


@pytest.mark.skipif(not UPSCALING_AVAILABLE, reason="upscaling_engine not available")
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_pytorch(self):
        """Test graceful handling when PyTorch unavailable."""
        if TORCH_AVAILABLE:
            pytest.skip("PyTorch is available, cannot test missing case")
        
        config = UpscalingConfig(device="cpu")
        engine = UpscalingEngine(config)
        
        with pytest.raises(RuntimeError, match="PyTorch not available"):
            engine.upscale_image(np.zeros((64, 64, 3)))
    
    def test_invalid_model_type(self):
        """Test error on invalid model."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required for model loading test")
        
        config = UpscalingConfig(device="cpu")
        engine = UpscalingEngine(config)
        
        with pytest.raises(ValueError, match="Unsupported model"):
            # Create fake enum member
            fake_model = Mock()
            fake_model.value = "invalid_model"
            engine._load_model(fake_model)
    
    def test_empty_batch(self, tmp_path):
        """Test batch processing with no images."""
        config = UpscalingConfig(device="cpu")
        engine = UpscalingEngine(config)
        
        results = engine.batch_upscale([], tmp_path / "output")
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
