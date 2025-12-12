"""
Tests for scene-aware preset auto-selection.

Tests cover:
    - Scene classification (interior/exterior, subtype)
    - Preset mapping logic
    - Fallback behavior for low confidence
    - Quality tier selection
    - Integration with CLIPClassifier
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lux_depth_v2.config import Preset

try:
    from lux_depth_v2.preset_selector import (
        PresetSelector,
        SceneType,
        QualityTier,
        SceneClassification,
        auto_select_preset
    )
    PRESET_SELECTOR_AVAILABLE = True
except ImportError:
    PRESET_SELECTOR_AVAILABLE = False


@pytest.mark.skipif(not PRESET_SELECTOR_AVAILABLE, reason="Preset selector not available")
class TestSceneClassification:
    """Test scene classification dataclass."""
    
    def test_confident_classification(self):
        """Test confident classification (>= 0.5)."""
        scene = SceneClassification(
            scene_type=SceneType.INTERIOR,
            scene_subtype="kitchen",
            confidence=0.85
        )
        assert scene.is_confident is True
    
    def test_low_confidence_classification(self):
        """Test low confidence classification (< 0.5)."""
        scene = SceneClassification(
            scene_type=SceneType.INTERIOR,
            scene_subtype="kitchen",
            confidence=0.35
        )
        assert scene.is_confident is False
    
    def test_threshold_confidence(self):
        """Test confidence exactly at threshold."""
        scene = SceneClassification(
            scene_type=SceneType.EXTERIOR,
            scene_subtype="pool",
            confidence=0.5
        )
        assert scene.is_confident is True


@pytest.mark.skipif(not PRESET_SELECTOR_AVAILABLE, reason="Preset selector not available")
class TestPresetMapping:
    """Test preset mapping logic."""
    
    def test_interior_kitchen_apex_mapping(self):
        """Test interior kitchen + APEX → interior_luxury_apex_quality."""
        key = (SceneType.INTERIOR, "kitchen", QualityTier.APEX)
        expected = Preset.INTERIOR_LUXURY_APEX_QUALITY
        assert PresetSelector.PRESET_MAP[key] == expected
    
    def test_exterior_pool_apex_mapping(self):
        """Test exterior pool + APEX → exterior_pool_apex_quality."""
        key = (SceneType.EXTERIOR, "pool", QualityTier.APEX)
        expected = Preset.EXTERIOR_POOL_APEX_QUALITY
        assert PresetSelector.PRESET_MAP[key] == expected
    
    def test_interior_max_quality_mapping(self):
        """Test interior + MAX → interior_luxury_max_quality."""
        key = (SceneType.INTERIOR, "bathroom", QualityTier.MAX)
        expected = Preset.INTERIOR_LUXURY_MAX_QUALITY
        assert PresetSelector.PRESET_MAP[key] == expected
    
    def test_exterior_standard_mapping(self):
        """Test exterior + STANDARD → exterior_showcase."""
        key = (SceneType.EXTERIOR, "courtyard", QualityTier.STANDARD)
        expected = Preset.EXTERIOR_SHOWCASE
        assert PresetSelector.PRESET_MAP[key] == expected
    
    def test_all_tiers_have_mappings(self):
        """Test that all quality tiers have at least one mapping."""
        tiers_present = set(key[2] for key in PresetSelector.PRESET_MAP.keys())
        assert QualityTier.STANDARD in tiers_present
        assert QualityTier.MAX in tiers_present
        assert QualityTier.APEX in tiers_present


@pytest.mark.skipif(not PRESET_SELECTOR_AVAILABLE, reason="Preset selector not available")
class TestFallbackBehavior:
    """Test fallback preset selection."""
    
    def test_fallback_interior_apex(self):
        """Test fallback for interior + APEX."""
        key = (SceneType.INTERIOR, QualityTier.APEX)
        expected = Preset.INTERIOR_LUXURY_APEX_QUALITY
        assert PresetSelector.FALLBACK_PRESETS[key] == expected
    
    def test_fallback_exterior_standard(self):
        """Test fallback for exterior + STANDARD."""
        key = (SceneType.EXTERIOR, QualityTier.STANDARD)
        expected = Preset.EXTERIOR_SHOWCASE
        assert PresetSelector.FALLBACK_PRESETS[key] == expected
    
    def test_fallback_unknown_scene(self):
        """Test fallback for unknown scene type."""
        key = (SceneType.UNKNOWN, QualityTier.MAX)
        expected = Preset.INTERIOR_LUXURY_MAX_QUALITY
        assert PresetSelector.FALLBACK_PRESETS[key] == expected
    
    def test_all_scene_types_have_fallbacks(self):
        """Test that all scene types have fallback presets."""
        scene_types = set(key[0] for key in PresetSelector.FALLBACK_PRESETS.keys())
        assert SceneType.INTERIOR in scene_types
        assert SceneType.EXTERIOR in scene_types
        assert SceneType.UNKNOWN in scene_types


@pytest.mark.skipif(not PRESET_SELECTOR_AVAILABLE, reason="Preset selector not available")
class TestPresetSelector:
    """Test PresetSelector with mocked CLIP."""
    
    @pytest.fixture
    def mock_clip(self):
        """Mock CLIP classifier."""
        with patch('lux_depth_v2.preset_selector.CLIPClassifier') as mock:
            yield mock
    
    @pytest.fixture
    def selector(self, mock_clip):
        """Create PresetSelector with mocked CLIP."""
        mock_instance = Mock()
        mock_clip.return_value = mock_instance
        return PresetSelector()
    
    def test_classify_scene_interior_kitchen(self, selector, mock_clip):
        """Test classification of interior kitchen scene."""
        # Mock CLIP responses
        mock_instance = mock_clip.return_value
        
        # First call: scene type (interior vs exterior)
        # Second call: scene subtype (kitchen, bathroom, etc.)
        mock_instance.classify_image.side_effect = [
            # Scene type probabilities (4 interior prompts, 4 exterior prompts)
            np.array([0.8, 0.85, 0.75, 0.82, 0.1, 0.15, 0.12, 0.08]),
            # Scene subtype probabilities (21 total: 7 subtypes × 3 prompts each)
            # kitchen (0-2), bathroom (3-5), bedroom (6-8), living (9-11), pool (12-14), courtyard (15-17), facade (18-20)
            np.array([0.9, 0.88, 0.92,  # kitchen - highest
                     0.3, 0.35, 0.32,  # bathroom
                     0.25, 0.28, 0.22,  # bedroom
                     0.2, 0.18, 0.23,  # living_room
                     0.15, 0.12, 0.18,  # pool
                     0.1, 0.08, 0.12,  # courtyard
                     0.05, 0.06, 0.04])  # facade
        ]
        
        # Create a simple test image
        from PIL import Image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        scene = selector.classify_scene(test_image)
        
        assert scene.scene_type == SceneType.INTERIOR
        assert scene.scene_subtype == "kitchen"
        assert scene.confidence > 0.5
    
    def test_classify_scene_exterior_pool(self, selector, mock_clip):
        """Test classification of exterior pool scene."""
        mock_instance = mock_clip.return_value
        
        # Mock CLIP to prefer exterior and pool
        mock_instance.classify_image.side_effect = [
            # Scene type: exterior dominant
            np.array([0.15, 0.12, 0.18, 0.10, 0.85, 0.88, 0.82, 0.90]),
            # Scene subtype: pool dominant (21 prompts total)
            np.array([0.2, 0.18, 0.22,  # kitchen
                     0.15, 0.12, 0.18,  # bathroom
                     0.1, 0.08, 0.12,  # bedroom
                     0.05, 0.06, 0.04,  # living_room
                     0.92, 0.88, 0.95,  # pool - highest
                     0.3, 0.28, 0.32,  # courtyard
                     0.25, 0.22, 0.28])  # facade
        ]
        
        from PIL import Image
        test_image = Image.new('RGB', (100, 100), color='blue')
        
        scene = selector.classify_scene(test_image)
        
        assert scene.scene_type == SceneType.EXTERIOR
        assert scene.scene_subtype == "pool"
    
    def test_select_preset_high_confidence(self, selector, mock_clip):
        """Test preset selection with high confidence."""
        mock_instance = mock_clip.return_value
        
        # Mock interior kitchen with high confidence
        mock_instance.classify_image.side_effect = [
            np.array([0.9, 0.88, 0.92, 0.87, 0.05, 0.08, 0.06, 0.04]),
            np.array([0.95, 0.92, 0.93,  # kitchen
                     0.1, 0.08, 0.12,  # bathroom
                     0.05, 0.06, 0.04,  # bedroom
                     0.08, 0.06, 0.10,  # living_room
                     0.05, 0.03, 0.08,  # pool
                     0.03, 0.02, 0.05,  # courtyard
                     0.01, 0.02, 0.01])  # facade
        ]
        
        from PIL import Image
        test_image = Image.new('RGB', (100, 100))
        
        recommendation = selector.select_preset(test_image, QualityTier.APEX)
        
        assert recommendation.preset == Preset.INTERIOR_LUXURY_APEX_QUALITY
        assert recommendation.fallback_used is False
        assert "kitchen" in recommendation.reason.lower()
    
    def test_select_preset_low_confidence_fallback(self, selector, mock_clip):
        """Test preset selection falls back on low confidence."""
        mock_instance = mock_clip.return_value
        
        # Mock low confidence classification
        mock_instance.classify_image.side_effect = [
            np.array([0.4, 0.42, 0.38, 0.35, 0.45, 0.48, 0.43, 0.40]),
            np.array([0.35, 0.38, 0.32,  # kitchen
                     0.40, 0.42, 0.38,  # bathroom
                     0.35, 0.33, 0.36,  # bedroom
                     0.39, 0.37, 0.34,  # living_room
                     0.31, 0.30, 0.33,  # pool
                     0.28, 0.30, 0.27,  # courtyard
                     0.25, 0.28, 0.26])  # facade
        ]
        
        from PIL import Image
        test_image = Image.new('RGB', (100, 100))
        
        recommendation = selector.select_preset(test_image, QualityTier.MAX)
        
        assert recommendation.fallback_used is True
        assert "fallback" in recommendation.reason.lower()
        assert recommendation.scene.confidence < 0.5
    
    def test_select_preset_from_path(self, selector, mock_clip, tmp_path):
        """Test simplified path-based interface."""
        mock_instance = mock_clip.return_value
        
        # Mock interior bedroom
        mock_instance.classify_image.side_effect = [
            np.array([0.85, 0.88, 0.82, 0.87, 0.1, 0.12, 0.08, 0.15]),
            np.array([0.2, 0.18, 0.22,  # kitchen
                     0.15, 0.12, 0.18,  # bathroom
                     0.92, 0.88, 0.95,  # bedroom - highest
                     0.1, 0.08, 0.12,  # living_room
                     0.05, 0.03, 0.08,  # pool
                     0.03, 0.02, 0.05,  # courtyard
                     0.02, 0.01, 0.03])  # facade
        ]
        
        # Create temp image file
        from PIL import Image
        test_image_path = tmp_path / "bedroom.jpg"
        Image.new('RGB', (100, 100)).save(test_image_path)
        
        preset = selector.select_preset_from_path(test_image_path, QualityTier.APEX)
        
        assert preset == Preset.INTERIOR_LUXURY_APEX_QUALITY


@pytest.mark.skipif(not PRESET_SELECTOR_AVAILABLE, reason="Preset selector not available")
def test_auto_select_preset_convenience_function(tmp_path):
    """Test convenience function with mocked CLIP."""
    with patch('lux_depth_v2.preset_selector.PresetSelector') as mock_selector_class:
        mock_selector = Mock()
        mock_selector_class.return_value = mock_selector
        
        # Mock the return value
        mock_selector.select_preset_from_path.return_value = Preset.INTERIOR_LUXURY_APEX_QUALITY
        
        # Create temp image
        from PIL import Image
        test_image_path = tmp_path / "test.jpg"
        Image.new('RGB', (100, 100)).save(test_image_path)
        
        # Call convenience function
        preset = auto_select_preset(test_image_path, quality_tier="apex")
        
        assert preset == Preset.INTERIOR_LUXURY_APEX_QUALITY
        mock_selector.select_preset_from_path.assert_called_once()


@pytest.mark.skipif(not PRESET_SELECTOR_AVAILABLE, reason="Preset selector not available")
def test_quality_tier_string_mapping():
    """Test quality tier string to enum mapping."""
    from lux_depth_v2.preset_selector import auto_select_preset
    
    with patch('lux_depth_v2.preset_selector.PresetSelector') as mock_selector_class:
        mock_selector = Mock()
        mock_selector_class.return_value = mock_selector
        mock_selector.select_preset_from_path.return_value = Preset.INTERIOR_LUXURY
        
        from PIL import Image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            Image.new('RGB', (100, 100)).save(f.name)
            
            # Test each tier string
            for tier_str in ["standard", "max", "apex"]:
                auto_select_preset(f.name, quality_tier=tier_str)
                
                # Verify correct QualityTier was passed
                call_args = mock_selector.select_preset_from_path.call_args
                passed_tier = call_args[0][1]
                
                if tier_str == "standard":
                    assert passed_tier == QualityTier.STANDARD
                elif tier_str == "max":
                    assert passed_tier == QualityTier.MAX
                elif tier_str == "apex":
                    assert passed_tier == QualityTier.APEX


@pytest.mark.skipif(not PRESET_SELECTOR_AVAILABLE, reason="Preset selector not available")
def test_preset_map_completeness():
    """Test that preset map covers common scenarios."""
    # Interior scenes with all tiers
    for subtype in ["kitchen", "bathroom", "bedroom", "living_room"]:
        for tier in [QualityTier.STANDARD, QualityTier.MAX, QualityTier.APEX]:
            key = (SceneType.INTERIOR, subtype, tier)
            assert key in PresetSelector.PRESET_MAP, f"Missing mapping for {key}"
    
    # Exterior scenes with all tiers
    for subtype in ["pool", "courtyard"]:
        for tier in [QualityTier.STANDARD, QualityTier.MAX, QualityTier.APEX]:
            key = (SceneType.EXTERIOR, subtype, tier)
            assert key in PresetSelector.PRESET_MAP, f"Missing mapping for {key}"
