"""
Tests for Material Responder Integration Module
================================================
"""

import numpy as np
import pytest
from pathlib import Path

try:
    from utils.material_responder import (
        MaterialResponder,
        MaterialResponseConfig,
        SurfaceType,
        create_material_responder
    )
    MATERIAL_RESPONDER_AVAILABLE = True
except ImportError:
    MATERIAL_RESPONDER_AVAILABLE = False


@pytest.mark.skipif(not MATERIAL_RESPONDER_AVAILABLE, reason="Material responder not available")
class TestMaterialResponseConfig:
    """Test material response configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MaterialResponseConfig()
        assert config.strength == 0.75
        assert config.surface_types == ["wood", "metal", "glass", "stone"]
        assert config.depth_aware is True
        assert config.preserve_highlights is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MaterialResponseConfig(
            strength=0.9,
            surface_types=["wood", "fabric"],
            depth_aware=False,
            preserve_highlights=False
        )
        assert config.strength == 0.9
        assert config.surface_types == ["wood", "fabric"]
        assert config.depth_aware is False
        assert config.preserve_highlights is False


@pytest.mark.skipif(not MATERIAL_RESPONDER_AVAILABLE, reason="Material responder not available")
class TestSurfaceType:
    """Test surface type enum."""
    
    def test_surface_types(self):
        """Test all surface types are defined."""
        expected = ["wood", "metal", "glass", "stone", "fabric", "concrete", "ceramic", "water"]
        
        for surface in expected:
            assert any(s.value == surface for s in SurfaceType)


@pytest.mark.skipif(not MATERIAL_RESPONDER_AVAILABLE, reason="Material responder not available")
class TestMaterialResponder:
    """Test material responder functionality."""
    
    @pytest.fixture
    def responder(self):
        """Create test responder."""
        config = MaterialResponseConfig(strength=0.75)
        return MaterialResponder(config)
    
    @pytest.fixture
    def sample_image(self):
        """Create sample RGB image with wood-like region."""
        image = np.random.rand(256, 256, 3).astype(np.float32)
        
        # Add brown region (simulated wood)
        image[50:150, 50:150, 0] = 0.6  # R
        image[50:150, 50:150, 1] = 0.4  # G
        image[50:150, 50:150, 2] = 0.2  # B
        
        return image
    
    def test_responder_initialization(self, responder):
        """Test responder initializes correctly."""
        assert responder.config.strength == 0.75
        assert len(responder.profiles) > 0
        
        # Check key profiles exist
        assert SurfaceType.WOOD in responder.profiles
        assert SurfaceType.METAL in responder.profiles
        assert SurfaceType.GLASS in responder.profiles
    
    def test_detect_materials(self, responder, sample_image):
        """Test material detection."""
        material_maps = responder.detect_materials(sample_image)
        
        # Should return dict with material types
        assert isinstance(material_maps, dict)
        assert len(material_maps) > 0
        
        # Check maps have correct shape
        for surface_type, confidence_map in material_maps.items():
            assert confidence_map.shape == sample_image.shape[:2]
            assert np.all(confidence_map >= 0) and np.all(confidence_map <= 1)
    
    def test_enhance_surface(self, responder, sample_image):
        """Test surface-specific enhancement."""
        confidence_map = np.ones(sample_image.shape[:2], dtype=np.float32) * 0.5
        
        enhanced = responder.enhance_surface(
            sample_image,
            SurfaceType.WOOD,
            confidence_map,
            depth_map=None
        )
        
        assert enhanced.shape == sample_image.shape
        assert np.all(enhanced >= 0) and np.all(enhanced <= 1)
    
    def test_enhance_with_depth(self, responder, sample_image):
        """Test enhancement with depth map."""
        depth_map = np.random.rand(256, 256).astype(np.float32)
        
        enhanced = responder.enhance(
            sample_image,
            surfaces=["wood"],
            depth_map=depth_map
        )
        
        assert enhanced.shape == sample_image.shape
        assert np.all(enhanced >= 0) and np.all(enhanced <= 1)
    
    def test_enhance_multiple_surfaces(self, responder, sample_image):
        """Test enhancement with multiple surfaces."""
        enhanced = responder.enhance(
            sample_image,
            surfaces=["wood", "metal", "glass"]
        )
        
        assert enhanced.shape == sample_image.shape
        assert np.all(enhanced >= 0) and np.all(enhanced <= 1)
    
    def test_enhance_unknown_surface(self, responder, sample_image):
        """Test with unknown surface type."""
        # Should handle gracefully
        enhanced = responder.enhance(
            sample_image,
            surfaces=["unknown_material"]
        )
        
        # Should return unchanged or slightly modified
        assert enhanced.shape == sample_image.shape


@pytest.mark.skipif(not MATERIAL_RESPONDER_AVAILABLE, reason="Material responder not available")
class TestMaterialResponderConvenience:
    """Test convenience functions."""
    
    def test_create_material_responder(self):
        """Test convenience function."""
        responder = create_material_responder(
            strength=0.8,
            surfaces=["wood", "metal"],
            depth_aware=True
        )
        
        assert isinstance(responder, MaterialResponder)
        assert responder.config.strength == 0.8
        assert responder.config.surface_types == ["wood", "metal"]
        assert responder.config.depth_aware is True


@pytest.mark.skipif(not MATERIAL_RESPONDER_AVAILABLE, reason="Material responder not available")
class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_strength(self):
        """Test with zero enhancement strength."""
        responder = create_material_responder(strength=0.0)
        image = np.random.rand(128, 128, 3).astype(np.float32)
        
        enhanced = responder.enhance(image, surfaces=["wood"])
        
        # Should be nearly unchanged
        assert np.allclose(enhanced, image, atol=0.05)
    
    def test_maximum_strength(self):
        """Test with maximum enhancement strength."""
        responder = create_material_responder(strength=1.0)
        image = np.random.rand(128, 128, 3).astype(np.float32)
        
        enhanced = responder.enhance(image, surfaces=["wood"])
        
        assert enhanced.shape == image.shape
        assert np.all(enhanced >= 0) and np.all(enhanced <= 1)
    
    def test_empty_surfaces_list(self):
        """Test with empty surfaces list."""
        responder = create_material_responder()
        image = np.random.rand(128, 128, 3).astype(np.float32)
        
        enhanced = responder.enhance(image, surfaces=[])
        
        # Should return nearly unchanged
        assert np.allclose(enhanced, image, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
