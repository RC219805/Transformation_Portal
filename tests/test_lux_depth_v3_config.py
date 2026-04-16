"""Tests for lux_depth_v3 configuration and validation.

Tests validation logic for depth_fallback and other config fields.
"""

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.security import validate_depth_fallback


class TestDepthFallbackValidation:
    """Test depth_fallback validation logic."""

    @pytest.mark.parametrize("mode", ["fail", "skip", "v2-auto"])
    def test_depth_fallback_accepts_valid_values(self, mode):
        """Test that validate_depth_fallback accepts valid values."""
        result = validate_depth_fallback(mode)
        assert result == mode

    @pytest.mark.parametrize("mode", ["FAIL", "Skip", "V2-AUTO", "  fail  "])
    def test_depth_fallback_normalizes_case_and_whitespace(self, mode):
        """Test that validate_depth_fallback normalizes case and whitespace."""
        result = validate_depth_fallback(mode)
        assert result in ["fail", "skip", "v2-auto"]

    def test_depth_fallback_accepts_none(self):
        """Test that validate_depth_fallback accepts None."""
        result = validate_depth_fallback(None)
        assert result is None

    @pytest.mark.parametrize(
        "invalid_mode",
        [
            "nope",
            "invalid",
            "none",  # Old value, should be rejected
            "zeros",  # Old value, should be rejected
            "previous",  # Old value, should be rejected
            "interpolate",  # Old value, should be rejected
        ],
    )
    def test_depth_fallback_rejects_invalid_values(self, invalid_mode):
        """Test that validate_depth_fallback rejects invalid values."""
        with pytest.raises(ValueError, match="Invalid depth fallback"):
            validate_depth_fallback(invalid_mode)


class TestEnhanceConfig:
    """Test EnhanceConfig dataclass."""

    def test_enhance_config_has_depth_fallback_field(self):
        """Test that EnhanceConfig has depth_fallback field with correct default."""
        config = EnhanceConfig()
        assert hasattr(config, "depth_fallback")
        assert config.depth_fallback == "fail"

    def test_enhance_config_has_verify_depth_writes_field(self):
        """Test that EnhanceConfig has verify_depth_writes field."""
        config = EnhanceConfig()
        assert hasattr(config, "verify_depth_writes")
        assert config.verify_depth_writes is True  # PR #751 uses True (safer default)

    def test_enhance_config_has_force_v2_field(self):
        """Test that EnhanceConfig has force_v2 field."""
        config = EnhanceConfig()
        assert hasattr(config, "force_v2")
        assert config.force_v2 is False

    def test_enhance_config_has_v2_timeout_field(self):
        """Test that EnhanceConfig has v2_timeout field."""
        config = EnhanceConfig()
        assert hasattr(config, "v2_timeout")
        assert config.v2_timeout == 300  # int type

    def test_enhance_config_has_reconstruction_fields(self):
        """Test that Phase B reconstruction config fields are present with safe defaults."""
        config = EnhanceConfig()
        assert hasattr(config, "enable_reconstruction")
        assert config.enable_reconstruction is False
        assert hasattr(config, "grouping_mode")
        assert config.grouping_mode == "single"
        assert hasattr(config, "cameras_sidecar_path")
        assert config.cameras_sidecar_path is None
        assert hasattr(config, "reconstruction_iterations")
        assert config.reconstruction_iterations == 1000
        assert hasattr(config, "reconstruction_tier")
        assert config.reconstruction_tier == "apex_research"
        assert hasattr(config, "emit_scene_debug_bundle")
        assert config.emit_scene_debug_bundle is False

    def test_enhance_config_has_raw_ingest_fields(self):
        """Test that Phase C RAW ingest knobs are present with safe defaults."""
        config = EnhanceConfig()
        assert hasattr(config, "raw_ingest_mode")
        assert config.raw_ingest_mode == "auto"
        assert hasattr(config, "raw_wb_mode")
        assert config.raw_wb_mode == "camera"
        assert hasattr(config, "raw_demosaic")
        assert config.raw_demosaic == "AHD"

    @pytest.mark.parametrize("mode", ["fail", "skip", "v2-auto"])
    def test_enhance_config_accepts_valid_depth_fallback(self, mode):
        """Test that EnhanceConfig can be instantiated with valid depth_fallback values."""
        config = EnhanceConfig(depth_fallback=mode)
        assert config.depth_fallback == mode

    def test_enhance_config_normalizes_legacy_depth_backend_alias(self):
        """Legacy backend aliases should normalize to canonical IDs."""
        with pytest.warns(FutureWarning, match="depth_anything_v3"):
            config = EnhanceConfig(depth_backend="depth_anything_v3")

        assert config.depth_backend == "da3"

    def test_enhance_config_normalizes_fallback_chain_aliases(self):
        """Fallback chains should normalize aliases and deduplicate canonical IDs."""
        with pytest.warns(FutureWarning, match="depth-anything-v3"):
            config = EnhanceConfig(
                depth_operational_fallback_chain=("depth-anything-v3", "da2", "da3"),
            )

        assert config.depth_operational_fallback_chain == ("da3", "da2")

    def test_enhance_config_all_required_fields_present(self):
        """Test that EnhanceConfig has all fields used by orchestrator."""
        config = EnhanceConfig()

        # Depth configuration
        assert hasattr(config, "model_variant")
        assert hasattr(config, "preset")
        assert hasattr(config, "depth_device")
        assert hasattr(config, "depth_quantization")
        assert hasattr(config, "depth_fallback")
        assert hasattr(config, "verify_depth_writes")

    def test_enhance_config_clamps_negative_apex_scaled_saturation_margin(self):
        """Negative scaled saturation margins should normalize to the effective runtime floor."""
        config = EnhanceConfig(apex_depth_scaled_saturation_margin=-0.25)

        assert config.apex_depth_scaled_saturation_margin == 0.0

    def test_enhance_config_clamps_negative_apex_low_saturation_warning_band(self):
        """Negative warning bands should normalize to the effective runtime floor."""
        config = EnhanceConfig(apex_depth_low_saturation_warning_band=-0.25)

        assert config.apex_depth_low_saturation_warning_band == 0.0

        # V2 configuration
        assert hasattr(config, "v2_preset")
        assert hasattr(config, "v2_device")
        assert hasattr(config, "v2_upscaler_backend")
        assert hasattr(config, "v2_timeout")

        # Flags
        assert hasattr(config, "force_depth")
        assert hasattr(config, "force_v2")
        assert hasattr(config, "non_commercial_ok")
        assert hasattr(config, "hash_mode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
