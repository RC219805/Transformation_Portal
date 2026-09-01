"""Tests for PBR EnhanceConfig presets."""

import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.lux_depth_v3.pbr_presets import (
    FABRIC_OPTIMIZED,
    FAST_PREVIEW,
    GLASS_OPTIMIZED,
    METAL_OPTIMIZED,
    PREMIUM_QUALITY,
    STANDARD_QUALITY,
    STONE_OPTIMIZED,
    WOOD_OPTIMIZED,
    get_preset,
    list_presets,
)

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]


class TestPresetConfiguration:
    """Test preset configuration validity."""

    @pytest.mark.parametrize(
        "preset",
        [
            STANDARD_QUALITY,
            PREMIUM_QUALITY,
            FAST_PREVIEW,
            WOOD_OPTIMIZED,
            METAL_OPTIMIZED,
            GLASS_OPTIMIZED,
            STONE_OPTIMIZED,
            FABRIC_OPTIMIZED,
        ],
    )
    def test_preset_is_enhance_config(self, preset):
        """All presets should be EnhanceConfig instances."""
        assert isinstance(preset, EnhanceConfig)

    @pytest.mark.parametrize(
        "preset",
        [
            STANDARD_QUALITY,
            PREMIUM_QUALITY,
            FAST_PREVIEW,
            WOOD_OPTIMIZED,
            METAL_OPTIMIZED,
            GLASS_OPTIMIZED,
            STONE_OPTIMIZED,
            FABRIC_OPTIMIZED,
        ],
    )
    def test_preset_enables_pbr(self, preset):
        """All presets should enable PBR generation."""
        assert preset.generate_pbr is True

    @pytest.mark.parametrize(
        "preset",
        [
            STANDARD_QUALITY,
            PREMIUM_QUALITY,
            FAST_PREVIEW,
            WOOD_OPTIMIZED,
            METAL_OPTIMIZED,
            GLASS_OPTIMIZED,
            STONE_OPTIMIZED,
            FABRIC_OPTIMIZED,
        ],
    )
    def test_preset_has_explicit_commercial_model_selector(self, preset):
        """All maintained presets should avoid implicit research selection."""
        assert preset.model_key in {"da3-metric", "da3-base"}
        assert preset.model_variant is None

    @pytest.mark.parametrize(
        ("preset", "expected_model_key"),
        [
            (STANDARD_QUALITY, "da3_metric"),
            (PREMIUM_QUALITY, "da3_metric"),
            (FAST_PREVIEW, "da3_base"),
            (WOOD_OPTIMIZED, "da3_metric"),
            (METAL_OPTIMIZED, "da3_metric"),
            (GLASS_OPTIMIZED, "da3_metric"),
            (STONE_OPTIMIZED, "da3_metric"),
            (FABRIC_OPTIMIZED, "da3_metric"),
        ],
    )
    def test_preset_prepares_without_research_acknowledgement(
        self,
        preset,
        expected_model_key,
        tmp_path,
    ):
        """Maintained PBR presets must compile under commercial-safe policy."""

        input_root = tmp_path / "inputs"
        input_root.mkdir()
        image = input_root / "scene.jpg"
        image.write_bytes(b"plan preparation does not decode inputs")

        prepared = prepare_lux_execution(preset, input_root, [image])

        assert prepared.plan.resolved_model is not None
        assert prepared.plan.resolved_model.canonical_key == expected_model_key
        assert prepared.plan.license_acknowledgements.non_commercial_ok is False


class TestPresetParameterRanges:
    """Test that preset parameters are in valid ranges."""

    @pytest.mark.parametrize(
        "preset",
        [
            STANDARD_QUALITY,
            PREMIUM_QUALITY,
            FAST_PREVIEW,
            WOOD_OPTIMIZED,
            METAL_OPTIMIZED,
            GLASS_OPTIMIZED,
            STONE_OPTIMIZED,
            FABRIC_OPTIMIZED,
        ],
    )
    def test_normal_strength_positive(self, preset):
        """Normal strength should be positive."""
        assert preset.pbr_normal_strength > 0
        assert preset.pbr_normal_strength <= 2.0  # Reasonable upper bound

    @pytest.mark.parametrize(
        "preset",
        [
            STANDARD_QUALITY,
            PREMIUM_QUALITY,
            FAST_PREVIEW,
            WOOD_OPTIMIZED,
            METAL_OPTIMIZED,
            GLASS_OPTIMIZED,
            STONE_OPTIMIZED,
            FABRIC_OPTIMIZED,
        ],
    )
    def test_blur_radii_non_negative(self, preset):
        """Blur radii should be non-negative integers."""
        assert preset.pbr_normal_blur_radius >= 0
        assert preset.pbr_roughness_blur_radius >= 0
        assert preset.pbr_ao_blur_radius >= 0
        assert isinstance(preset.pbr_normal_blur_radius, int)
        assert isinstance(preset.pbr_roughness_blur_radius, int)
        assert isinstance(preset.pbr_ao_blur_radius, int)

    @pytest.mark.parametrize(
        "preset",
        [
            STANDARD_QUALITY,
            PREMIUM_QUALITY,
            FAST_PREVIEW,
            WOOD_OPTIMIZED,
            METAL_OPTIMIZED,
            GLASS_OPTIMIZED,
            STONE_OPTIMIZED,
            FABRIC_OPTIMIZED,
        ],
    )
    def test_ao_bias_in_range(self, preset):
        """AO bias should be in [0.0, 1.0] range."""
        assert 0.0 <= preset.pbr_ao_bias <= 1.0


class TestStandardQualityPreset:
    """Test Standard Quality preset configuration."""

    def test_standard_enables_float_depth(self):
        """Standard should enable float depth for quality."""
        assert STANDARD_QUALITY.save_float_depth is True

    def test_standard_uses_large_model(self):
        """Standard should use large model for quality."""
        assert STANDARD_QUALITY.model_key == "da3-metric"

    def test_standard_balanced_parameters(self):
        """Standard should have balanced parameters."""
        assert STANDARD_QUALITY.pbr_normal_strength == 1.2
        assert STANDARD_QUALITY.pbr_normal_blur_radius == 1
        assert STANDARD_QUALITY.pbr_roughness_strength == 1.0
        assert STANDARD_QUALITY.pbr_roughness_blur_radius == 3
        assert STANDARD_QUALITY.pbr_ao_strength == 1.0
        assert STANDARD_QUALITY.pbr_ao_blur_radius == 5
        assert STANDARD_QUALITY.pbr_ao_bias == 0.45


class TestPremiumQualityPreset:
    """Test Premium Quality preset configuration."""

    def test_premium_enables_float_depth(self):
        """Premium MUST enable float depth."""
        assert PREMIUM_QUALITY.save_float_depth is True

    def test_premium_uses_large_model(self):
        """Premium should use large model for best quality."""
        assert PREMIUM_QUALITY.model_key == "da3-metric"

    def test_premium_no_normal_blur(self):
        """Premium should preserve all normal detail."""
        assert PREMIUM_QUALITY.pbr_normal_blur_radius == 0

    def test_premium_high_strength_parameters(self):
        """Premium should have high strength for maximum detail."""
        assert PREMIUM_QUALITY.pbr_normal_strength > STANDARD_QUALITY.pbr_normal_strength
        assert PREMIUM_QUALITY.pbr_roughness_strength > STANDARD_QUALITY.pbr_roughness_strength
        assert PREMIUM_QUALITY.pbr_ao_strength > STANDARD_QUALITY.pbr_ao_strength

    def test_premium_darker_ao_bias(self):
        """Premium should have darker AO for dramatic depth."""
        assert PREMIUM_QUALITY.pbr_ao_bias < STANDARD_QUALITY.pbr_ao_bias


class TestFastPreviewPreset:
    """Test Fast Preview preset configuration."""

    def test_draft_disables_float_depth(self):
        """Draft should disable float depth for speed."""
        assert FAST_PREVIEW.save_float_depth is False

    def test_draft_uses_base_model(self):
        """Draft should use base model for speed."""
        assert FAST_PREVIEW.model_key == "da3-base"

    def test_draft_blur_budget_stays_below_standard(self):
        """Draft should keep a lighter blur budget than standard for speed."""
        assert FAST_PREVIEW.pbr_normal_blur_radius <= STANDARD_QUALITY.pbr_normal_blur_radius
        assert FAST_PREVIEW.pbr_roughness_blur_radius <= STANDARD_QUALITY.pbr_roughness_blur_radius
        assert FAST_PREVIEW.pbr_ao_blur_radius < STANDARD_QUALITY.pbr_ao_blur_radius

    def test_draft_lower_strength_parameters(self):
        """Draft should have lower strength for speed."""
        assert FAST_PREVIEW.pbr_normal_strength < STANDARD_QUALITY.pbr_normal_strength
        assert FAST_PREVIEW.pbr_roughness_strength < STANDARD_QUALITY.pbr_roughness_strength
        assert FAST_PREVIEW.pbr_ao_strength < STANDARD_QUALITY.pbr_ao_strength


class TestMaterialOptimizedPresets:
    """Test material-specific preset configurations."""

    def test_wood_emphasizes_normal_detail(self):
        """Wood preset should capture grain texture."""
        assert WOOD_OPTIMIZED.pbr_normal_strength > STANDARD_QUALITY.pbr_normal_strength
        assert WOOD_OPTIMIZED.pbr_normal_blur_radius == 0  # Preserve grain

    def test_metal_reduces_roughness(self):
        """Metal preset should have lower roughness for polished surfaces."""
        assert METAL_OPTIMIZED.pbr_roughness_strength < STANDARD_QUALITY.pbr_roughness_strength

    def test_glass_heavy_smoothing(self):
        """Glass preset should have heavy smoothing for flat surfaces."""
        assert GLASS_OPTIMIZED.pbr_normal_blur_radius >= 3
        assert GLASS_OPTIMIZED.pbr_roughness_blur_radius >= 6

    def test_glass_low_strength(self):
        """Glass preset should have low strength for smooth surfaces."""
        assert GLASS_OPTIMIZED.pbr_normal_strength < STANDARD_QUALITY.pbr_normal_strength
        assert GLASS_OPTIMIZED.pbr_roughness_strength < STANDARD_QUALITY.pbr_roughness_strength

    def test_glass_bright_ao_bias(self):
        """Glass should have bright AO (transmissive material)."""
        assert GLASS_OPTIMIZED.pbr_ao_bias > STANDARD_QUALITY.pbr_ao_bias

    def test_stone_high_detail(self):
        """Stone preset should capture texture detail."""
        assert STONE_OPTIMIZED.pbr_normal_strength > STANDARD_QUALITY.pbr_normal_strength
        assert STONE_OPTIMIZED.pbr_roughness_strength > STANDARD_QUALITY.pbr_roughness_strength
        assert STONE_OPTIMIZED.pbr_normal_blur_radius == 0  # Preserve texture

    def test_stone_dark_ao_bias(self):
        """Stone should have darker AO for grout/joint shadows."""
        assert STONE_OPTIMIZED.pbr_ao_bias < STANDARD_QUALITY.pbr_ao_bias

    def test_fabric_moderate_parameters(self):
        """Fabric preset should have moderate parameters."""
        # Fabric is between smooth and highly textured
        assert FABRIC_OPTIMIZED.pbr_normal_strength > GLASS_OPTIMIZED.pbr_normal_strength
        assert FABRIC_OPTIMIZED.pbr_normal_strength < WOOD_OPTIMIZED.pbr_normal_strength


class TestPresetRegistry:
    """Test preset registry and lookup functions."""

    def test_get_preset_by_name(self):
        """Test getting preset by name."""
        assert get_preset("standard") == STANDARD_QUALITY
        assert get_preset("premium") == PREMIUM_QUALITY
        assert get_preset("draft") == FAST_PREVIEW
        assert get_preset("wood") == WOOD_OPTIMIZED
        assert get_preset("metal") == METAL_OPTIMIZED
        assert get_preset("glass") == GLASS_OPTIMIZED
        assert get_preset("stone") == STONE_OPTIMIZED
        assert get_preset("fabric") == FABRIC_OPTIMIZED

    def test_get_preset_case_insensitive(self):
        """Test preset lookup is case-insensitive."""
        assert get_preset("STANDARD") == STANDARD_QUALITY
        assert get_preset("Premium") == PREMIUM_QUALITY
        assert get_preset("DRAFT") == FAST_PREVIEW

    def test_get_preset_invalid_name(self):
        """Test get_preset raises ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("invalid_preset")

    def test_list_presets_returns_all(self):
        """Test list_presets returns all preset names."""
        presets = list_presets()
        assert "standard" in presets
        assert "premium" in presets
        assert "draft" in presets
        assert "wood" in presets
        assert "metal" in presets
        assert "glass" in presets
        assert "stone" in presets
        assert "fabric" in presets
        assert len(presets) == 8


class TestPresetConsistency:
    """Test consistency across presets."""

    def test_all_presets_enable_pbr(self):
        """All presets should enable PBR generation."""
        for preset_name in list_presets():
            preset = get_preset(preset_name)
            assert preset.generate_pbr is True, f"{preset_name} should enable PBR"

    def test_quality_tier_uses_same_model(self):
        """Standard and Premium should use the same model selector."""
        assert STANDARD_QUALITY.model_key == PREMIUM_QUALITY.model_key

    def test_material_presets_enable_float_depth(self):
        """All material-optimized presets should enable float depth."""
        for preset in [WOOD_OPTIMIZED, METAL_OPTIMIZED, GLASS_OPTIMIZED, STONE_OPTIMIZED, FABRIC_OPTIMIZED]:
            assert preset.save_float_depth is True

    def test_material_presets_use_large_model(self):
        """All material-optimized presets should use large model."""
        for preset in [WOOD_OPTIMIZED, METAL_OPTIMIZED, GLASS_OPTIMIZED, STONE_OPTIMIZED, FABRIC_OPTIMIZED]:
            assert preset.model_key == "da3-metric"


class TestPresetPerformanceCharacteristics:
    """Test performance-related preset characteristics."""

    @staticmethod
    def _blur_work_score(config: EnhanceConfig) -> int:
        return sum(
            (2 * radius) + 1
            for radius in (
                config.pbr_normal_blur_radius,
                config.pbr_roughness_blur_radius,
                config.pbr_ao_blur_radius,
            )
            if radius > 0
        )

    def test_draft_optimized_for_speed(self):
        """Draft should prioritize speed over quality."""
        # Smaller model
        assert FAST_PREVIEW.model_key == "da3-base"
        # No float depth (faster I/O)
        assert FAST_PREVIEW.save_float_depth is False
        # Lower strength (less computation)
        assert FAST_PREVIEW.pbr_normal_strength < 1.0
        assert FAST_PREVIEW.pbr_roughness_strength < 1.0
        assert self._blur_work_score(FAST_PREVIEW) < self._blur_work_score(STANDARD_QUALITY)

    def test_premium_optimized_for_quality(self):
        """Premium should prioritize quality over speed."""
        # Largest model
        assert PREMIUM_QUALITY.model_key == "da3-metric"
        # Float depth for precision
        assert PREMIUM_QUALITY.save_float_depth is True
        # No normal blur (preserve detail)
        assert PREMIUM_QUALITY.pbr_normal_blur_radius == 0
        # High strength
        assert PREMIUM_QUALITY.pbr_normal_strength >= 1.5

    def test_standard_balanced(self):
        """Standard should be balanced between draft and premium."""
        # Parameters should be between draft and premium
        assert FAST_PREVIEW.pbr_normal_strength < STANDARD_QUALITY.pbr_normal_strength < PREMIUM_QUALITY.pbr_normal_strength
        assert (
            FAST_PREVIEW.pbr_roughness_strength
            < STANDARD_QUALITY.pbr_roughness_strength
            < PREMIUM_QUALITY.pbr_roughness_strength
        )


class TestPresetPBRConfigConversion:
    """Test conversion from EnhanceConfig to PBRConfig."""

    @pytest.mark.parametrize(
        "preset",
        [
            STANDARD_QUALITY,
            PREMIUM_QUALITY,
            FAST_PREVIEW,
        ],
    )
    def test_preset_converts_to_pbr_config(self, preset):
        """Test that presets can be converted to PBRConfig."""
        pbr_config = preset.to_pbr_config()

        # Verify parameter transfer
        assert pbr_config.normal_strength == preset.pbr_normal_strength
        assert pbr_config.normal_blur_radius == preset.pbr_normal_blur_radius
        assert pbr_config.roughness_strength == preset.pbr_roughness_strength
        assert pbr_config.roughness_blur_radius == preset.pbr_roughness_blur_radius
        assert pbr_config.ao_strength == preset.pbr_ao_strength
        assert pbr_config.ao_blur_radius == preset.pbr_ao_blur_radius
        assert pbr_config.ao_bias == preset.pbr_ao_bias


class TestPresetDocumentation:
    """Test that presets are well-documented."""

    def test_module_has_docstring(self):
        """Module should have comprehensive docstring."""
        import transformation_portal.lux_depth_v3.pbr_presets as module

        assert module.__doc__ is not None
        assert "STANDARD_QUALITY" in module.__doc__
        assert "PREMIUM_QUALITY" in module.__doc__
        assert "FAST_PREVIEW" in module.__doc__

    def test_get_preset_has_docstring(self):
        """get_preset function should have docstring."""
        assert get_preset.__doc__ is not None

    def test_list_presets_has_docstring(self):
        """list_presets function should have docstring."""
        assert list_presets.__doc__ is not None
