#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Material Response Engine.

Tests for the MaterialResponseEngine class and related functionality.
"""

import numpy as np
import pytest
from PIL import Image


# Check if scipy is available
HAS_SCIPY = False
try:
    import scipy.ndimage  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not HAS_SCIPY,
    reason="scipy is required for material response engine"
)


@pytest.fixture
def sample_image():
    """Create sample RGB image for testing."""
    arr = np.zeros((600, 800, 3), dtype=np.uint8)
    # Add gradient
    for i in range(600):
        arr[i, :, 0] = int(255 * i / 600)
    for j in range(800):
        arr[:, j, 1] = int(255 * j / 800)
    arr[:, :, 2] = 128
    return Image.fromarray(arr, 'RGB')


@pytest.fixture
def interior_image():
    """Create image with interior characteristics."""
    # Create image with varied brightness and low saturation in upper half
    arr = np.random.randint(50, 150, (600, 800, 3), dtype=np.uint8)
    # Make upper portion wall-like (low saturation, moderate brightness)
    arr[:300, :, :] = np.clip(arr[:300, :, :].mean(axis=2, keepdims=True), 80, 180)
    return Image.fromarray(arr, 'RGB')


class TestMaterialResponseConfig:
    """Test MaterialResponseConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseConfig
        )

        config = MaterialResponseConfig()

        assert config.profile == "luxury_interior"
        assert config.texture_boost == 0.25
        assert config.ambient_occlusion == 0.12
        assert config.highlight_warmth == 0.08

    def test_value_clamping(self):
        """Test that values are clamped to valid ranges."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseConfig
        )

        config = MaterialResponseConfig(
            texture_boost=2.0,  # Should clamp to 1.0
            ambient_occlusion=-0.5,  # Should clamp to 0.0
        )

        assert config.texture_boost == 1.0
        assert config.ambient_occlusion == 0.0


class TestMaterialResponseEngine:
    """Test MaterialResponseEngine class."""

    def test_engine_initialization(self):
        """Test engine can be initialized."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine, MaterialResponseConfig
        )

        config = MaterialResponseConfig()
        engine = MaterialResponseEngine(config)

        assert engine.config.profile == "luxury_interior"

    def test_from_config_dict(self):
        """Test creating engine from config dictionary."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        engine = MaterialResponseEngine.from_config({
            'profile': 'luxury_interior',
            'texture_boost': 0.3,
            'ambient_occlusion': 0.15,
        })

        assert engine.config.texture_boost == 0.3
        assert engine.config.ambient_occlusion == 0.15

    def test_apply_returns_image(self, sample_image):
        """Test apply method returns PIL Image."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        engine = MaterialResponseEngine.from_config({
            'profile': 'luxury_interior',
            'texture_boost': 0.25,
        })

        result = engine.apply(sample_image)

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
        assert result.mode == 'RGB'

    def test_apply_with_strength(self, sample_image):
        """Test apply method with strength parameter."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        engine = MaterialResponseEngine.from_config({'profile': 'luxury_interior'})

        # Apply with different strengths
        result_full = engine.apply(sample_image, strength=1.0)
        result_half = engine.apply(sample_image, strength=0.5)

        # Both should be valid images
        assert isinstance(result_full, Image.Image)
        assert isinstance(result_half, Image.Image)

    def test_enhance_floor(self, sample_image):
        """Test floor enhancement method."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        engine = MaterialResponseEngine.from_config({
            'floor_plank_contrast': 0.2,
            'floor_specular': 0.2,
        })

        # Convert to array for testing
        rgb = np.array(sample_image).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]

        # Create simple masks
        y_norm = np.linspace(0, 1, h).reshape(-1, 1)
        y_norm = np.broadcast_to(y_norm, (h, w)).astype(np.float32)
        floor_mask = np.clip((y_norm - 0.55) / 0.45, 0.0, 1.0)
        wood_mask = floor_mask * 0.5

        result = engine.enhance_floor(rgb, floor_mask, wood_mask)

        assert result.shape == rgb.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_enhance_textiles(self, sample_image):
        """Test textile enhancement method."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        engine = MaterialResponseEngine.from_config({'textile_contrast': 0.2})

        rgb = np.array(sample_image).astype(np.float32) / 255.0
        textile_mask = np.ones(rgb.shape[:2], dtype=np.float32) * 0.5

        result = engine.enhance_textiles(rgb, textile_mask)

        assert result.shape == rgb.shape

    def test_enhance_metals(self, sample_image):
        """Test metal enhancement method."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        engine = MaterialResponseEngine.from_config({})

        rgb = np.array(sample_image).astype(np.float32) / 255.0
        metal_mask = np.ones(rgb.shape[:2], dtype=np.float32) * 0.5

        result = engine.enhance_metals(rgb, metal_mask)

        assert result.shape == rgb.shape

    def test_add_atmospheric_effects(self, sample_image):
        """Test atmospheric effects method."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        engine = MaterialResponseEngine.from_config({'haze_strength': 0.1})

        rgb = np.array(sample_image).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]

        result = engine.add_atmospheric_effects(rgb, h, w)

        assert result.shape == rgb.shape

    def test_grayscale_conversion(self):
        """Test that grayscale images are converted to RGB."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        # Create grayscale image
        gray_arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray_image = Image.fromarray(gray_arr, 'L')

        engine = MaterialResponseEngine.from_config({'profile': 'luxury_interior'})
        result = engine.apply(gray_image)

        assert result.mode == 'RGB'


class TestMaterialProfiles:
    """Test material profile functionality."""

    def test_get_profile(self):
        """Test getting a profile by name."""
        from transformation_portal.processors.material_response.profiles import (
            get_profile
        )

        profile = get_profile('luxury_interior')

        assert profile['name'] == 'luxury_interior'
        assert 'texture_boost' in profile
        assert 'ambient_occlusion' in profile

    def test_list_profiles(self):
        """Test listing all profiles."""
        from transformation_portal.processors.material_response.profiles import (
            list_profiles
        )

        profiles = list_profiles()

        assert 'luxury_interior' in profiles
        assert 'wood_floor_oak' in profiles
        assert 'marble_stone' in profiles
        assert 'textile_linen' in profiles
        assert 'metal_brushed' in profiles
        assert 'glass_window' in profiles

    def test_invalid_profile(self):
        """Test error on invalid profile name."""
        from transformation_portal.processors.material_response.profiles import (
            get_profile
        )

        with pytest.raises(KeyError):
            get_profile('nonexistent_profile')

    def test_profile_info(self):
        """Test getting profile display info."""
        from transformation_portal.processors.material_response.profiles import (
            get_profile_info
        )

        info = get_profile_info('luxury_interior')

        assert 'display_name' in info
        assert 'description' in info
        assert info['name'] == 'luxury_interior'

    def test_all_profiles_valid(self):
        """Test that all profiles have required fields."""
        from transformation_portal.processors.material_response.profiles import (
            get_all_profiles
        )

        profiles = get_all_profiles()
        required_fields = ['name', 'texture_boost', 'ambient_occlusion']

        for profile_name, profile_data in profiles.items():
            for field in required_fields:
                assert field in profile_data, f"Profile {profile_name} missing field {field}"


class TestMaterialMask:
    """Test material mask computation."""

    def test_material_mask_creation(self, interior_image):
        """Test material mask computation."""
        from transformation_portal.processors.material_response.engine import (
            MaterialResponseEngine
        )

        engine = MaterialResponseEngine.from_config({'profile': 'luxury_interior'})
        rgb = np.array(interior_image).astype(np.float32) / 255.0

        masks = engine._compute_material_masks(rgb)

        # Check all masks exist
        assert masks.floor is not None
        assert masks.wall is not None
        assert masks.textile is not None
        assert masks.wood is not None
        assert masks.metal is not None
        assert masks.highlight is not None
        assert masks.midtone is not None

        # Check mask shapes
        h, w = rgb.shape[:2]
        assert masks.floor.shape == (h, w)
        assert masks.wall.shape == (h, w)

        # Check mask value ranges
        assert masks.floor.min() >= 0.0
        assert masks.floor.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
