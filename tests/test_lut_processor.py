"""
Tests for LUT Processor module.

Tests .cube LUT loading, application, and color grading functionality.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from utils.lut_processor import (
    LUTProcessor,
    LUTConfig,
    LUTCategory,
    create_lut_processor,
    discover_luts
)


@pytest.fixture
def sample_3d_lut_file():
    """Create a temporary 3D LUT file for testing."""
    content = """# Test 3D LUT
TITLE "Test_LUT"
LUT_3D_SIZE 3
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0

0.0 0.0 0.0
0.0 0.0 0.5
0.0 0.0 1.0
0.0 0.5 0.0
0.0 0.5 0.5
0.0 0.5 1.0
0.0 1.0 0.0
0.0 1.0 0.5
0.0 1.0 1.0
0.5 0.0 0.0
0.5 0.0 0.5
0.5 0.0 1.0
0.5 0.5 0.0
0.5 0.5 0.5
0.5 0.5 1.0
0.5 1.0 0.0
0.5 1.0 0.5
0.5 1.0 1.0
1.0 0.0 0.0
1.0 0.0 0.5
1.0 0.0 1.0
1.0 0.5 0.0
1.0 0.5 0.5
1.0 0.5 1.0
1.0 1.0 0.0
1.0 1.0 0.5
1.0 1.0 1.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False) as f:
        f.write(content)
        return Path(f.name)


@pytest.fixture
def sample_1d_lut_file():
    """Create a temporary 1D LUT file for testing."""
    content = """# Test 1D LUT
TITLE "Test_1D_LUT"
LUT_1D_SIZE 5

0.0 0.0 0.0
0.3 0.3 0.3
0.5 0.5 0.5
0.7 0.7 0.7
1.0 1.0 1.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False) as f:
        f.write(content)
        return Path(f.name)


@pytest.fixture
def test_image():
    """Create a test image."""
    # Create simple gradient image
    image = np.zeros((64, 64, 3), dtype=np.float32)
    for i in range(64):
        image[i, :, :] = i / 63.0
    return image


class TestLUTConfig:
    """Test LUT configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LUTConfig()
        assert config.lut_path is None
        assert config.strength == 0.7
        assert config.category == LUTCategory.FILM_EMULATION
        assert config.preserve_highlights is True
        assert config.preserve_blacks is True
    
    def test_custom_config(self, sample_3d_lut_file):
        """Test custom configuration."""
        config = LUTConfig(
            lut_path=sample_3d_lut_file,
            strength=0.5,
            category=LUTCategory.LOCATION_AESTHETIC
        )
        assert config.lut_path == sample_3d_lut_file
        assert config.strength == 0.5
        assert config.category == LUTCategory.LOCATION_AESTHETIC
    
    def test_invalid_strength(self):
        """Test invalid strength values."""
        with pytest.raises(ValueError):
            LUTConfig(strength=1.5)
        with pytest.raises(ValueError):
            LUTConfig(strength=-0.1)
    
    def test_invalid_lut_path(self):
        """Test invalid LUT path."""
        with pytest.raises(FileNotFoundError):
            LUTConfig(lut_path="/nonexistent/path.cube")


class TestLUTProcessor:
    """Test LUT processor functionality."""
    
    def test_load_3d_lut(self, sample_3d_lut_file):
        """Test loading 3D LUT file."""
        config = LUTConfig(lut_path=sample_3d_lut_file)
        processor = LUTProcessor(config)
        
        assert processor.is_3d is True
        assert processor.lut_size == 3
        assert processor.lut_data.shape == (3, 3, 3, 3)
        assert processor.title == "Test_LUT"
    
    def test_load_1d_lut(self, sample_1d_lut_file):
        """Test loading 1D LUT file."""
        config = LUTConfig(lut_path=sample_1d_lut_file)
        processor = LUTProcessor(config)
        
        assert processor.is_3d is False
        assert processor.lut_size == 5
        assert processor.lut_data.shape == (5, 3)
        assert processor.title == "Test_1D_LUT"
    
    def test_apply_3d_lut(self, sample_3d_lut_file, test_image):
        """Test applying 3D LUT to image."""
        config = LUTConfig(lut_path=sample_3d_lut_file, strength=1.0)
        processor = LUTProcessor(config)
        
        result = processor.apply(test_image)
        
        assert result.shape == test_image.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
    
    def test_apply_1d_lut(self, sample_1d_lut_file, test_image):
        """Test applying 1D LUT to image."""
        config = LUTConfig(lut_path=sample_1d_lut_file, strength=1.0)
        processor = LUTProcessor(config)
        
        result = processor.apply(test_image)
        
        assert result.shape == test_image.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
    
    def test_strength_control(self, sample_3d_lut_file, test_image):
        """Test LUT strength/opacity control."""
        config = LUTConfig(lut_path=sample_3d_lut_file)
        processor = LUTProcessor(config)
        
        # Apply with different strengths
        result_0 = processor.apply(test_image, strength=0.0)
        result_05 = processor.apply(test_image, strength=0.5)
        result_1 = processor.apply(test_image, strength=1.0)
        
        # Strength 0 should match original
        np.testing.assert_allclose(result_0, test_image, atol=1e-5)
        
        # Strength 0.5 should be between original and full LUT
        assert np.all(np.abs(result_05 - test_image) <= np.abs(result_1 - test_image) + 1e-5)
    
    def test_preserve_highlights(self, sample_3d_lut_file):
        """Test highlight preservation."""
        config = LUTConfig(
            lut_path=sample_3d_lut_file,
            preserve_highlights=True
        )
        processor = LUTProcessor(config)
        
        # Create image with bright highlights
        image = np.ones((64, 64, 3), dtype=np.float32) * 0.95
        result = processor.apply(image, strength=1.0)
        
        # Highlights should be relatively preserved
        assert np.mean(result) > 0.8  # Still bright
    
    def test_preserve_blacks(self, sample_3d_lut_file):
        """Test black level preservation."""
        config = LUTConfig(
            lut_path=sample_3d_lut_file,
            preserve_blacks=True
        )
        processor = LUTProcessor(config)
        
        # Create image with deep blacks
        image = np.ones((64, 64, 3), dtype=np.float32) * 0.05
        result = processor.apply(image, strength=1.0)
        
        # Blacks should be relatively preserved
        assert np.mean(result) < 0.2  # Still dark
    
    def test_no_lut_loaded(self):
        """Test applying without loaded LUT."""
        config = LUTConfig()
        processor = LUTProcessor(config)
        
        image = np.random.rand(64, 64, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="No LUT loaded"):
            processor.apply(image)
    
    def test_load_lut_method(self, sample_3d_lut_file, sample_1d_lut_file):
        """Test loading LUT via method."""
        config = LUTConfig()
        processor = LUTProcessor(config)
        
        # Load 3D LUT
        processor.load_lut(sample_3d_lut_file)
        assert processor.is_3d is True
        assert processor.lut_size == 3
        
        # Load 1D LUT (replace)
        processor.load_lut(sample_1d_lut_file)
        assert processor.is_3d is False
        assert processor.lut_size == 5
    
    def test_value_clamping(self, sample_3d_lut_file):
        """Test input value clamping."""
        config = LUTConfig(lut_path=sample_3d_lut_file)
        processor = LUTProcessor(config)
        
        # Create image with out-of-range values
        image = np.random.rand(64, 64, 3).astype(np.float32) * 2.0 - 0.5
        result = processor.apply(image)
        
        # Result should be clamped to [0, 1]
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


class TestConvenienceFunction:
    """Test convenience creation function."""
    
    def test_create_lut_processor(self, sample_3d_lut_file):
        """Test convenience function."""
        processor = create_lut_processor(
            lut_path=sample_3d_lut_file,
            strength=0.8,
            category=LUTCategory.MATERIAL_RESPONSE
        )
        
        assert isinstance(processor, LUTProcessor)
        assert processor.config.strength == 0.8
        assert processor.config.category == LUTCategory.MATERIAL_RESPONSE
        assert processor.lut_data is not None


class TestLUTDiscovery:
    """Test LUT discovery functionality."""
    
    def test_discover_luts_real_directory(self):
        """Test discovering LUTs in real assets directory."""
        base_path = Path("/Users/rc/Transformation_Portal/assets/luts")
        
        if base_path.exists():
            luts = discover_luts(base_path)
            
            assert isinstance(luts, dict)
            assert LUTCategory.FILM_EMULATION in luts
            assert LUTCategory.LOCATION_AESTHETIC in luts
            assert LUTCategory.MATERIAL_RESPONSE in luts
            
            # Should find at least some LUTs
            total_luts = sum(len(v) for v in luts.values())
            assert total_luts > 0
    
    def test_discover_luts_empty_directory(self):
        """Test discovering LUTs in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            luts = discover_luts(Path(tmpdir))
            
            # Should return empty lists for all categories
            for category_luts in luts.values():
                assert len(category_luts) == 0


class TestRealLUTFiles:
    """Test with real LUT files if available."""
    
    def test_kodak_lut(self, test_image):
        """Test loading and applying real Kodak LUT."""
        lut_path = Path("/Users/rc/Transformation_Portal/assets/luts/film_emulation/Kodak/Kodak_2393_D55.cube")
        
        if lut_path.exists():
            processor = create_lut_processor(lut_path=lut_path, strength=0.7)
            result = processor.apply(test_image)
            
            assert result.shape == test_image.shape
            assert np.all(result >= 0.0) and np.all(result <= 1.0)
            
            # Should produce some visible change
            assert not np.allclose(result, test_image)
    
    def test_location_lut(self, test_image):
        """Test loading and applying real location aesthetic LUT."""
        lut_path = Path("/Users/rc/Transformation_Portal/assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube")
        
        if lut_path.exists():
            processor = create_lut_processor(lut_path=lut_path, strength=0.7)
            result = processor.apply(test_image)
            
            assert result.shape == test_image.shape
            assert np.all(result >= 0.0) and np.all(result <= 1.0)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_pixel_image(self, sample_3d_lut_file):
        """Test applying LUT to single pixel."""
        processor = create_lut_processor(lut_path=sample_3d_lut_file)
        image = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        
        result = processor.apply(image)
        
        assert result.shape == (1, 1, 3)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
    
    def test_large_image(self, sample_3d_lut_file):
        """Test applying LUT to large image."""
        processor = create_lut_processor(lut_path=sample_3d_lut_file)
        image = np.random.rand(1024, 1024, 3).astype(np.float32)
        
        result = processor.apply(image)
        
        assert result.shape == image.shape
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
    
    def test_pure_black(self, sample_3d_lut_file):
        """Test LUT on pure black image."""
        processor = create_lut_processor(lut_path=sample_3d_lut_file)
        image = np.zeros((64, 64, 3), dtype=np.float32)
        
        result = processor.apply(image)
        
        assert result.shape == image.shape
        assert np.all(result >= 0.0)
    
    def test_pure_white(self, sample_3d_lut_file):
        """Test LUT on pure white image."""
        processor = create_lut_processor(lut_path=sample_3d_lut_file)
        image = np.ones((64, 64, 3), dtype=np.float32)
        
        result = processor.apply(image)
        
        assert result.shape == image.shape
        assert np.all(result <= 1.0)
