"""
Tests for auto-preset canary-blocking safety features.

Validates:
- Canary presets blocked by default (allow_canary=False)
- Canary presets allowed with explicit flag (allow_canary=True)
- Non-canary equivalents returned when blocking
- Fallback reasoning includes canary-blocked message
"""

import numpy as np
import pytest
from PIL import Image

from lux_depth_v2.config import Preset
from lux_depth_v2.preset_selector import (
    PresetSelector,
    Intent,
    QualityTier,
    SceneType,
    SceneClassification,
)


@pytest.fixture
def mock_selector(monkeypatch):
    """Preset selector with mocked CLIP (to avoid model downloads)."""
    # Mock classify_scene to return a controlled result
    def mock_classify(self, image):
        # Always return interior kitchen with high confidence
        return SceneClassification(
            scene_type=SceneType.INTERIOR,
            scene_subtype="kitchen",
            confidence=0.85
        )
    
    monkeypatch.setattr(PresetSelector, "classify_scene", mock_classify)
    
    selector = PresetSelector.__new__(PresetSelector)
    selector.confidence_threshold = 0.5
    return selector


class TestCanaryBlockingDefault:
    """Test that canary presets are blocked by default."""
    
    def test_canary_blocked_returns_non_canary_equivalent(self, mock_selector):
        """When canary preset selected but allow_canary=False, fallback to non-canary."""
        # Create a dummy image
        img_array = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Manually force recommendation to be canary
        # (In practice, PRESET_MAP wouldn't return canary, but we're testing the guard)
        canary_preset = Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM
        
        # The _is_canary_preset check should detect it
        assert mock_selector._is_canary_preset(canary_preset) is True
        
        # Get non-canary equivalent
        fallback = mock_selector._get_non_canary_equivalent(canary_preset)
        assert fallback == Preset.INTERIOR_LUXURY_APEX_QUALITY
    
    def test_select_preset_with_auto_tier_blocks_canary(self, mock_selector, monkeypatch):
        """select_preset_with_auto_tier blocks canary when allow_canary=False."""
        img_array = np.zeros((800, 800, 3), dtype=np.uint8)
        
        # Mock the base select_preset to return a canary preset
        def mock_select_preset(self, image, quality_tier):
            from lux_depth_v2.preset_selector import PresetRecommendation
            return PresetRecommendation(
                preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
                scene=SceneClassification(
                    scene_type=SceneType.INTERIOR,
                    scene_subtype="kitchen",
                    confidence=0.90
                ),
                fallback_used=False,
                reason="Interior kitchen scene detected"
            )
        
        monkeypatch.setattr(PresetSelector, "select_preset", mock_select_preset)
        
        # Call with allow_canary=False (default)
        recommendation = mock_selector.select_preset_with_auto_tier(
            img_array,
            quality_tier=QualityTier.APEX,
            allow_canary=False
        )
        
        # Should fallback to non-canary
        assert recommendation.preset == Preset.INTERIOR_LUXURY_APEX_QUALITY
        assert recommendation.fallback_used is True
        assert "Canary blocked" in recommendation.reason


class TestCanaryAllowedExplicitly:
    """Test that canary presets are allowed when explicitly requested."""
    
    def test_canary_allowed_with_flag(self, mock_selector, monkeypatch):
        """select_preset_with_auto_tier allows canary when allow_canary=True."""
        img_array = np.zeros((800, 800, 3), dtype=np.uint8)
        
        # Mock select_preset to return canary
        def mock_select_preset(self, image, quality_tier):
            from lux_depth_v2.preset_selector import PresetRecommendation
            return PresetRecommendation(
                preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
                scene=SceneClassification(
                    scene_type=SceneType.INTERIOR,
                    scene_subtype="kitchen",
                    confidence=0.90
                ),
                fallback_used=False,
                reason="Interior kitchen scene detected"
            )
        
        monkeypatch.setattr(PresetSelector, "select_preset", mock_select_preset)
        
        # Call with allow_canary=True
        recommendation = mock_selector.select_preset_with_auto_tier(
            img_array,
            quality_tier=QualityTier.APEX,
            allow_canary=True
        )
        
        # Should preserve canary preset
        assert recommendation.preset == Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM
        assert recommendation.fallback_used is False
        assert "Canary blocked" not in recommendation.reason


class TestCanaryDetection:
    """Test canary preset identification."""
    
    @pytest.mark.parametrize("preset,expected_is_canary", [
        (Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM, True),
        (Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM, True),
        (Preset.INTERIOR_LUXURY_APEX_QUALITY, False),
        (Preset.INTERIOR_LUXURY_MAX_QUALITY, False),
        (Preset.INTERIOR_LUXURY, False),
        (Preset.EXTERIOR_POOL_APEX_QUALITY, False),
        (Preset.EXTERIOR_SHOWCASE, False),
    ])
    def test_is_canary_preset_detection(self, preset, expected_is_canary):
        """_is_canary_preset correctly identifies canary presets."""
        assert PresetSelector._is_canary_preset(preset) == expected_is_canary
    
    def test_get_non_canary_equivalent_mappings(self):
        """_get_non_canary_equivalent returns correct non-canary presets."""
        # Interior canary → non-canary
        assert (PresetSelector._get_non_canary_equivalent(
            Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM
        ) == Preset.INTERIOR_LUXURY_APEX_QUALITY)
        
        # Exterior canary → non-canary
        assert (PresetSelector._get_non_canary_equivalent(
            Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM
        ) == Preset.EXTERIOR_POOL_APEX_QUALITY)
        
        # Non-canary → unchanged
        assert (PresetSelector._get_non_canary_equivalent(
            Preset.INTERIOR_LUXURY_MAX_QUALITY
        ) == Preset.INTERIOR_LUXURY_MAX_QUALITY)


class TestAutoTierWithComplexity:
    """Test auto-tier selection with complexity scoring."""
    
    def test_high_complexity_upgrades_client_to_apex(self, mock_selector):
        """High complexity + CLIENT intent → APEX tier."""
        from lux_depth_v2.complexity_scorer import ComplexityScore
        
        high_complexity = ComplexityScore(
            gradient_energy=0.25,
            edge_density=0.30,
            megapixels=15.0,
            complexity_class="high"
        )
        
        tier = mock_selector.select_quality_tier(
            intent=Intent.CLIENT,
            complexity=high_complexity
        )
        
        assert tier == QualityTier.APEX
    
    def test_low_complexity_client_stays_max(self, mock_selector):
        """Low complexity + CLIENT intent → MAX tier."""
        from lux_depth_v2.complexity_scorer import ComplexityScore
        
        low_complexity = ComplexityScore(
            gradient_energy=0.05,
            edge_density=0.08,
            megapixels=5.0,
            complexity_class="low"
        )
        
        tier = mock_selector.select_quality_tier(
            intent=Intent.CLIENT,
            complexity=low_complexity
        )
        
        assert tier == QualityTier.MAX
    
    def test_large_megapixels_upgrades_to_apex(self, mock_selector):
        """Large image (≥20 MP) + CLIENT intent → APEX tier."""
        tier = mock_selector.select_quality_tier(
            intent=Intent.CLIENT,
            megapixels=25.0
        )
        
        assert tier == QualityTier.APEX
    
    def test_preview_intent_always_standard(self, mock_selector):
        """PREVIEW intent → STANDARD tier regardless of complexity."""
        from lux_depth_v2.complexity_scorer import ComplexityScore
        
        high_complexity = ComplexityScore(
            gradient_energy=0.25,
            edge_density=0.30,
            megapixels=25.0,
            complexity_class="high"
        )
        
        tier = mock_selector.select_quality_tier(
            intent=Intent.PREVIEW,
            complexity=high_complexity
        )
        
        assert tier == QualityTier.STANDARD
    
    def test_hero_intent_always_apex(self, mock_selector):
        """HERO intent → APEX tier regardless of complexity."""
        from lux_depth_v2.complexity_scorer import ComplexityScore
        
        low_complexity = ComplexityScore(
            gradient_energy=0.02,
            edge_density=0.05,
            megapixels=2.0,
            complexity_class="low"
        )
        
        tier = mock_selector.select_quality_tier(
            intent=Intent.HERO,
            complexity=low_complexity
        )
        
        assert tier == QualityTier.APEX


class TestCLIIntegration:
    """Test CLI flag integration (requires --allow-canary to select canary)."""
    
    def test_cli_defaults_block_canary(self, monkeypatch):
        """CLI without --allow-canary should not select canary presets."""
        # This is validated in test_cli_auto_preset.py integration tests
        # Here we just verify the default
        from lux_depth_v2.preset_selector import PresetSelector
        
        # Default allow_canary in select_preset_with_auto_tier is False
        import inspect
        sig = inspect.signature(PresetSelector.select_preset_with_auto_tier)
        assert sig.parameters["allow_canary"].default is False
