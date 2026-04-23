"""Unit tests for V2 Enhancement Presets.

Tests preset configuration loading, validation, and parameter ranges.
"""

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.v2_presets import PRESETS, V2EnhancementConfig, get_preset_description, list_presets


class TestV2EnhancementConfig:
    """Test V2EnhancementConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = V2EnhancementConfig()

        assert config.preset == "default"
        assert config.enhancement_strength == 0.7
        assert config.clarity_strength == 0.5
        assert config.material_strength == 0.6
        assert config.depth_aware_tone_mapping is True
        assert config.atmospheric_effects is True
        # Banding-mitigation guards default ON — see ADR-022 addendum.
        assert config.tone_depth_smoothing is True
        assert config.tone_low_tex_strength == 0.6
        assert config.version == "1.1.0"

    def test_custom_config(self):
        """Test custom configuration."""
        config = V2EnhancementConfig(
            preset="custom",
            enhancement_strength=0.9,
            clarity_strength=0.8,
            material_strength=0.7,
            depth_aware_tone_mapping=False,
            atmospheric_effects=False,
        )

        assert config.preset == "custom"
        assert config.enhancement_strength == 0.9
        assert config.clarity_strength == 0.8
        assert config.material_strength == 0.7
        assert config.depth_aware_tone_mapping is False
        assert config.atmospheric_effects is False

    def test_validation_enhancement_strength(self):
        """Test validation of enhancement_strength parameter."""
        # Valid ranges
        V2EnhancementConfig(enhancement_strength=0.0)
        V2EnhancementConfig(enhancement_strength=0.5)
        V2EnhancementConfig(enhancement_strength=1.0)

        # Invalid ranges
        with pytest.raises(ValueError, match="enhancement_strength must be in"):
            V2EnhancementConfig(enhancement_strength=-0.1)
        with pytest.raises(ValueError, match="enhancement_strength must be in"):
            V2EnhancementConfig(enhancement_strength=1.5)

    def test_validation_clarity_strength(self):
        """Test validation of clarity_strength parameter."""
        # Valid ranges
        V2EnhancementConfig(clarity_strength=0.0)
        V2EnhancementConfig(clarity_strength=1.0)

        # Invalid ranges
        with pytest.raises(ValueError, match="clarity_strength must be in"):
            V2EnhancementConfig(clarity_strength=-0.1)
        with pytest.raises(ValueError, match="clarity_strength must be in"):
            V2EnhancementConfig(clarity_strength=1.1)

    def test_validation_material_strength(self):
        """Test validation of material_strength parameter."""
        # Valid ranges
        V2EnhancementConfig(material_strength=0.0)
        V2EnhancementConfig(material_strength=1.0)

        # Invalid ranges
        with pytest.raises(ValueError, match="material_strength must be in"):
            V2EnhancementConfig(material_strength=-0.1)
        with pytest.raises(ValueError, match="material_strength must be in"):
            V2EnhancementConfig(material_strength=1.5)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = V2EnhancementConfig(preset="test")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["preset"] == "test"
        assert config_dict["enhancement_strength"] == 0.7
        assert config_dict["clarity_strength"] == 0.5
        assert config_dict["material_strength"] == 0.6
        assert config_dict["depth_aware_tone_mapping"] is True
        assert config_dict["atmospheric_effects"] is True
        assert config_dict["tone_depth_smoothing"] is True
        assert config_dict["tone_low_tex_strength"] == 0.6
        assert config_dict["version"] == "1.1.0"


class TestPresetLoading:
    """Test preset loading functionality."""

    def test_load_default_preset(self):
        """Test loading default preset."""
        config = V2EnhancementConfig.from_preset("default")

        assert config.preset == "default"
        assert config.enhancement_strength == 0.7
        assert config.clarity_strength == 0.5
        assert config.material_strength == 0.6
        assert config.depth_aware_tone_mapping is True
        assert config.atmospheric_effects is True

    def test_load_luxury_estate_preset(self):
        """Test loading luxury_estate preset."""
        config = V2EnhancementConfig.from_preset("luxury_estate")

        assert config.preset == "luxury_estate"
        assert config.enhancement_strength == 0.8
        assert config.clarity_strength == 0.6
        assert config.material_strength == 0.7
        assert config.depth_aware_tone_mapping is True
        assert config.atmospheric_effects is True

    def test_load_architectural_preset(self):
        """Test loading architectural preset."""
        config = V2EnhancementConfig.from_preset("architectural")

        assert config.preset == "architectural"
        assert config.enhancement_strength == 0.6
        assert config.clarity_strength == 0.7
        assert config.material_strength == 0.5
        assert config.depth_aware_tone_mapping is True
        assert config.atmospheric_effects is False  # No atmosphere for technical viz

    def test_load_none_preset(self):
        """Test loading 'none' preset (skip enhancement)."""
        config = V2EnhancementConfig.from_preset("none")

        assert config.preset == "none"
        assert config.enhancement_strength == 0.0
        assert config.clarity_strength == 0.0
        assert config.material_strength == 0.0
        assert config.depth_aware_tone_mapping is False
        assert config.atmospheric_effects is False

    def test_load_unknown_preset(self):
        """Test loading unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            V2EnhancementConfig.from_preset("nonexistent")

        with pytest.raises(ValueError, match="Available presets"):
            V2EnhancementConfig.from_preset("invalid")


class TestPresetHelpers:
    """Test preset helper functions."""

    def test_get_preset_description(self):
        """Test getting preset descriptions."""
        desc = get_preset_description("default")
        assert desc == "Balanced enhancement for general use"

        desc = get_preset_description("luxury_estate")
        assert desc == "Premium marketing aesthetic"

        desc = get_preset_description("nonexistent")
        assert desc is None

    def test_list_presets(self):
        """Test listing all presets."""
        presets = list_presets()

        assert isinstance(presets, dict)
        assert "default" in presets
        assert "luxury_estate" in presets
        assert "architectural" in presets
        assert "none" in presets

        # Check descriptions are present
        assert presets["default"] == "Balanced enhancement for general use"
        assert presets["luxury_estate"] == "Premium marketing aesthetic"

    def test_all_presets_valid(self):
        """Test that all defined presets can be loaded."""
        for preset_name in PRESETS.keys():
            config = V2EnhancementConfig.from_preset(preset_name)
            assert config.preset == preset_name


class TestPresetConsistency:
    """Test preset consistency and completeness."""

    def test_preset_keys_match(self):
        """Test that PRESETS dict has all expected keys."""
        expected_presets = {"default", "luxury_estate", "architectural", "none"}
        actual_presets = set(PRESETS.keys())

        assert actual_presets == expected_presets

    def test_preset_schema_consistency(self):
        """Test that all presets have consistent schema."""
        required_keys = {
            "description",
            "enhancement_strength",
            "clarity_strength",
            "material_strength",
            "depth_aware_tone_mapping",
            "atmospheric_effects",
            "use_case",
        }

        for preset_name, preset_config in PRESETS.items():
            assert set(preset_config.keys()) == required_keys, f"Preset '{preset_name}' has inconsistent schema"

    def test_preset_strength_ranges(self):
        """Test that all presets have valid strength values."""
        for preset_name, preset_config in PRESETS.items():
            assert 0.0 <= preset_config["enhancement_strength"] <= 1.0, f"Invalid enhancement_strength in {preset_name}"
            assert 0.0 <= preset_config["clarity_strength"] <= 1.0, f"Invalid clarity_strength in {preset_name}"
            assert 0.0 <= preset_config["material_strength"] <= 1.0, f"Invalid material_strength in {preset_name}"

    def test_preset_boolean_flags(self):
        """Test that boolean flags are actually booleans."""
        for preset_name, preset_config in PRESETS.items():
            assert isinstance(
                preset_config["depth_aware_tone_mapping"], bool
            ), f"depth_aware_tone_mapping must be bool in {preset_name}"
            assert isinstance(preset_config["atmospheric_effects"], bool), f"atmospheric_effects must be bool in {preset_name}"
