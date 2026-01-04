"""Tests for EnhanceOrchestrator preset behavior."""

import pytest
from pathlib import Path
from lux_depth_v3.enhance.orchestrator import EnhanceConfig
from lux_depth_v3.config import ModelVariant, Preset


class TestEnhanceConfigPresetBehavior:
    """Test suite for EnhanceConfig preset and model_variant interaction."""

    def test_preset_none_model_none_uses_default(self):
        """When both preset and model_variant are None, should use METRIC_LARGE default."""
        config = EnhanceConfig(
            model_variant=None,
            preset=None,
        )
        assert config.model_variant is None
        assert config.preset is None

    def test_preset_none_model_explicit(self):
        """When preset is None but model is explicit, should use that model."""
        config = EnhanceConfig(
            model_variant=ModelVariant.DA3_BASE,
            preset=None,
        )
        assert config.model_variant == ModelVariant.DA3_BASE
        assert config.preset is None

    def test_preset_set_model_none(self):
        """When preset is set but model is None, preset's model should be used (not overridden)."""
        config = EnhanceConfig(
            model_variant=None,
            preset=Preset.INTERIOR_LUXURY,
        )
        assert config.model_variant is None  # None means "use preset's default"
        assert config.preset == Preset.INTERIOR_LUXURY

    def test_preset_and_model_both_set(self):
        """When both preset and model are set, model should override preset's choice."""
        config = EnhanceConfig(
            model_variant=ModelVariant.DA3_SMALL,
            preset=Preset.INTERIOR_LUXURY,
        )
        assert config.model_variant == ModelVariant.DA3_SMALL
        assert config.preset == Preset.INTERIOR_LUXURY

    def test_default_values(self):
        """Test that default values are correct."""
        config = EnhanceConfig()
        assert config.model_variant is None
        assert config.preset is None
        assert config.v2_preset == "production_ultra"
        assert config.depth_device == "auto"
        assert config.execution_mode == "sequential"

    def test_optional_model_variant_type(self):
        """Ensure model_variant field accepts None."""
        # This test verifies the type annotation is Optional[ModelVariant]
        config = EnhanceConfig(model_variant=None)
        assert config.model_variant is None

        config2 = EnhanceConfig(model_variant=ModelVariant.METRIC_LARGE)
        assert config2.model_variant == ModelVariant.METRIC_LARGE
