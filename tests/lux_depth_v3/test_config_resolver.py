"""Unit tests for ConfigResolver.

Tests the configuration resolution logic extracted from orchestrator.py
as part of ADR-043 decomposition.

These tests verify:
1. Preset discovery
2. Preset to DA3Config resolution
3. Configuration fingerprint computation
4. ResolvedConfig interface
5. Backward compatibility with orchestrator imports
"""

from __future__ import annotations

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]


class TestConfigResolverImports:
    """Test that imports work from the new module."""

    def test_import_from_config_resolver(self):
        """Test that we can import from the new config_resolver module."""
        from transformation_portal.lux_depth_v3.config_resolver import (
            ConfigResolver,
            PresetInfo,
            ResolvedConfig,
            build_apex_depth_gate_fingerprint_payload,
            build_materials_fingerprint_payload,
            build_pbr_fingerprint_payload,
            build_run_card_config_fingerprint,
            compute_config_fingerprint,
            discover_presets,
            resolve_preset,
        )

        assert ConfigResolver is not None
        assert ResolvedConfig is not None
        assert PresetInfo is not None
        assert callable(discover_presets)
        assert callable(resolve_preset)
        assert callable(compute_config_fingerprint)
        assert callable(build_materials_fingerprint_payload)
        assert callable(build_pbr_fingerprint_payload)
        assert callable(build_apex_depth_gate_fingerprint_payload)
        assert callable(build_run_card_config_fingerprint)


class TestDiscoverPresets:
    """Test preset discovery functionality."""

    def test_discover_presets_returns_list(self):
        """Test that discover_presets returns a list."""
        from transformation_portal.lux_depth_v3.config_resolver import discover_presets

        presets = discover_presets()
        assert isinstance(presets, list)

    def test_discover_presets_contains_known_presets(self):
        """Test that known presets are discoverable."""
        from transformation_portal.lux_depth_v3.config_resolver import discover_presets

        presets = discover_presets()
        names = [p.name for p in presets]

        assert "ARCHITECTURAL_INTERIOR" in names
        assert "ARCHITECTURAL_EXTERIOR" in names
        assert "LUXURY_ESTATE" in names
        assert "DEFAULT" in names

    def test_preset_info_has_required_fields(self):
        """Test that PresetInfo has all required fields."""
        from transformation_portal.lux_depth_v3.config_resolver import discover_presets

        presets = discover_presets()
        for preset in presets:
            assert hasattr(preset, "name")
            assert hasattr(preset, "value")
            assert hasattr(preset, "display_name")
            assert hasattr(preset, "tier")

    def test_discover_presets_unknown_pipeline_returns_empty(self):
        """Test that unknown pipeline returns empty list."""
        from transformation_portal.lux_depth_v3.config_resolver import discover_presets

        presets = discover_presets("unknown_pipeline")
        assert presets == []


class TestResolvePreset:
    """Test preset resolution functionality."""

    def test_resolve_preset_with_none(self):
        """Test resolving None preset returns default config."""
        from transformation_portal.lux_depth_v3.config import ModelVariant
        from transformation_portal.lux_depth_v3.config_resolver import resolve_preset

        da3_config, model = resolve_preset(None)

        assert model == ModelVariant.METRIC_LARGE
        assert da3_config.model_variant == ModelVariant.METRIC_LARGE

    def test_resolve_preset_with_preset(self):
        """Test resolving a valid preset."""
        from transformation_portal.lux_depth_v3.config import ModelVariant, Preset
        from transformation_portal.lux_depth_v3.config_resolver import resolve_preset

        da3_config, model = resolve_preset(Preset.ARCHITECTURAL_INTERIOR)

        assert model == ModelVariant.METRIC_LARGE
        assert da3_config.model_variant == ModelVariant.METRIC_LARGE
        # Should have bilateral filter enabled
        assert da3_config.postprocessing.apply_bilateral_filter is True

    def test_resolve_preset_with_model_override(self):
        """Test that model override takes precedence."""
        from transformation_portal.lux_depth_v3.config import ModelVariant, Preset
        from transformation_portal.lux_depth_v3.config_resolver import resolve_preset

        da3_config, model = resolve_preset(
            Preset.ARCHITECTURAL_INTERIOR,
            ModelVariant.METRIC_SMALL,
        )

        assert model == ModelVariant.METRIC_SMALL
        assert da3_config.model_variant == ModelVariant.METRIC_SMALL

    def test_resolve_preset_luxury_estate(self):
        """Test resolving LUXURY_ESTATE preset."""
        from transformation_portal.lux_depth_v3.config import ModelVariant, Preset
        from transformation_portal.lux_depth_v3.config_resolver import resolve_preset

        da3_config, model = resolve_preset(Preset.LUXURY_ESTATE)

        assert model == ModelVariant.METRIC_LARGE
        assert da3_config.postprocessing.apply_bilateral_filter is True
        # Luxury estate has tighter edge threshold
        assert da3_config.postprocessing.edge_threshold == 0.03


class TestComputeConfigFingerprint:
    """Test configuration fingerprint computation."""

    def test_fingerprint_is_config_fingerprint_type(self):
        """Test that fingerprint is of correct type."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            compute_config_fingerprint,
        )
        from transformation_portal.lux_depth_v3.manifest import ConfigFingerprint

        config = EnhanceConfig()
        fingerprint = compute_config_fingerprint(config)

        assert isinstance(fingerprint, ConfigFingerprint)

    def test_fingerprint_contains_expected_fields(self):
        """Test that fingerprint contains expected fields."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            compute_config_fingerprint,
        )

        config = EnhanceConfig()
        fingerprint = compute_config_fingerprint(config)

        assert fingerprint.model_variant is not None
        assert fingerprint.depth_quantization == "none"
        assert fingerprint.depth_device == "cpu"

    def test_fingerprint_includes_materials_config(self):
        """Test that fingerprint includes materials config."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            compute_config_fingerprint,
        )

        config = EnhanceConfig(enable_materials_v3=True)
        fingerprint = compute_config_fingerprint(config)

        assert fingerprint.materials_config is not None
        assert fingerprint.materials_config["enable_materials_v3"] is True

    def test_fingerprint_deterministic(self):
        """Test that fingerprint is deterministic."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            compute_config_fingerprint,
        )

        config = EnhanceConfig()
        fp1 = compute_config_fingerprint(config)
        fp2 = compute_config_fingerprint(config)

        assert fp1.to_sha256() == fp2.to_sha256()

    def test_fingerprint_changes_with_config(self):
        """Test that fingerprint changes when config changes."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            compute_config_fingerprint,
        )

        config1 = EnhanceConfig(depth_device="cpu")
        config2 = EnhanceConfig(depth_device="mps")

        fp1 = compute_config_fingerprint(config1)
        fp2 = compute_config_fingerprint(config2)

        assert fp1.to_sha256() != fp2.to_sha256()


class TestBuildFingerprintPayloads:
    """Test fingerprint payload building functions."""

    def test_materials_fingerprint_payload(self):
        """Test materials fingerprint payload structure."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_materials_fingerprint_payload,
        )

        config = EnhanceConfig(
            enable_materials_v3=True,
            apply_pixel_ops=False,
        )
        payload = build_materials_fingerprint_payload(config)

        assert payload["enable_materials_v3"] is True
        assert payload["apply_pixel_ops"] is False
        assert "mask_feather_sigma_default" in payload
        assert "sam2_model_size" in payload

    def test_pbr_fingerprint_payload(self):
        """Test PBR fingerprint payload structure."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_pbr_fingerprint_payload,
        )

        config = EnhanceConfig(
            generate_pbr=True,
            pbr_normal_strength=0.8,
        )
        payload = build_pbr_fingerprint_payload(config)

        assert payload["generate_pbr"] is True
        assert payload["normal_strength"] == 0.8
        assert "roughness_strength" in payload
        assert "ao_bias" in payload

    def test_apex_depth_gate_fingerprint_payload(self):
        """Test APEX depth gate fingerprint payload structure."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_apex_depth_gate_fingerprint_payload,
        )

        config = EnhanceConfig(quality_tier="apex")
        payload = build_apex_depth_gate_fingerprint_payload(config)

        assert payload["quality_tier"] == "apex"
        assert "min_finite_pct" in payload
        assert "min_gradient_energy" in payload


class TestBuildRunCardConfigFingerprint:
    """Test run card config fingerprint building."""

    def test_run_card_fingerprint_has_sha256(self):
        """Test that run card fingerprint includes SHA256."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_run_card_config_fingerprint,
        )

        config = EnhanceConfig()
        fingerprint = build_run_card_config_fingerprint(config)

        assert "sha256" in fingerprint
        assert len(fingerprint["sha256"]) == 64

    def test_run_card_fingerprint_has_canonical_json(self):
        """Test that run card fingerprint includes canonical JSON."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_run_card_config_fingerprint,
        )

        config = EnhanceConfig()
        fingerprint = build_run_card_config_fingerprint(config)

        assert "canonical_json" in fingerprint
        assert "hash_algorithm" in fingerprint
        assert fingerprint["hash_algorithm"] == "sha256"

    def test_run_card_fingerprint_includes_resolution_metadata(self):
        """Test that run card fingerprint includes resolution metadata."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_run_card_config_fingerprint,
        )

        config = EnhanceConfig(preset=Preset.LUXURY_ESTATE)
        fingerprint = build_run_card_config_fingerprint(config)

        assert "preset_requested" in fingerprint
        assert "preset_resolved" in fingerprint
        assert "backend_requested" in fingerprint
        assert "quality_tier" in fingerprint


class TestConfigResolverClass:
    """Test the ConfigResolver class interface."""

    def test_resolver_init(self):
        """Test resolver initialization."""
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        resolver = ConfigResolver()
        assert resolver is not None

    def test_resolver_resolve(self):
        """Test resolver resolve method."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            ConfigResolver,
            ResolvedConfig,
        )

        resolver = ConfigResolver()
        config = EnhanceConfig()
        resolved = resolver.resolve(config)

        assert isinstance(resolved, ResolvedConfig)
        assert resolved.enhance_config is config
        assert resolved.da3_config is not None
        assert resolved.fingerprint is not None

    def test_resolver_resolve_with_preset(self):
        """Test resolver resolve with preset."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        resolver = ConfigResolver()
        config = EnhanceConfig(preset=Preset.ARCHITECTURAL_INTERIOR)
        resolved = resolver.resolve(config)

        assert resolved.preset_resolved == "architectural_interior"
        assert resolved.da3_config.postprocessing.apply_bilateral_filter is True

    def test_resolver_discover_presets(self):
        """Test resolver discover_presets method."""
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        resolver = ConfigResolver()
        presets = resolver.discover_presets()

        assert len(presets) > 0
        assert any(p.name == "LUXURY_ESTATE" for p in presets)

    def test_resolver_get_preset_config(self):
        """Test resolver get_preset_config caching."""
        from transformation_portal.lux_depth_v3.config import Preset
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        resolver = ConfigResolver()
        config1 = resolver.get_preset_config(Preset.DEFAULT)
        config2 = resolver.get_preset_config(Preset.DEFAULT)

        # Should be cached (same instance)
        assert config1 is config2

    def test_resolver_compute_fingerprint(self):
        """Test resolver compute_fingerprint method."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        resolver = ConfigResolver()
        config = EnhanceConfig()
        fingerprint = resolver.compute_fingerprint(config)

        assert fingerprint is not None
        assert len(fingerprint.to_sha256()) == 64

    def test_resolver_build_run_card_fingerprint(self):
        """Test resolver build_run_card_fingerprint method."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        resolver = ConfigResolver()
        config = EnhanceConfig()
        fingerprint = resolver.build_run_card_fingerprint(config)

        assert "sha256" in fingerprint
        assert "canonical_json" in fingerprint


class TestResolvedConfig:
    """Test ResolvedConfig data class."""

    def test_resolved_config_fields(self):
        """Test ResolvedConfig has expected fields."""
        from transformation_portal.lux_depth_v3.config import DA3Config, EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import ResolvedConfig

        config = EnhanceConfig()
        da3 = DA3Config()

        resolved = ResolvedConfig(
            enhance_config=config,
            da3_config=da3,
            preset_requested=None,
            preset_resolved="default",
            quality_tier="standard",
        )

        assert resolved.enhance_config is config
        assert resolved.da3_config is da3
        assert resolved.quality_tier == "standard"

    def test_resolved_config_from_resolver(self):
        """Test ResolvedConfig from resolver."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        resolver = ConfigResolver()
        config = EnhanceConfig()
        resolved = resolver.resolve(config)

        # Model variant should be set after resolution
        assert resolved.model_variant is not None
        assert config.model_variant is not None  # Mutated by resolve


class TestPresetInfo:
    """Test PresetInfo data class."""

    def test_preset_info_creation(self):
        """Test PresetInfo creation."""
        from transformation_portal.lux_depth_v3.config_resolver import PresetInfo

        info = PresetInfo(
            name="TEST",
            value="test",
            display_name="Test Preset",
            description="A test preset",
            tier="premium",
        )

        assert info.name == "TEST"
        assert info.value == "test"
        assert info.display_name == "Test Preset"
        assert info.tier == "premium"

    def test_preset_info_defaults(self):
        """Test PresetInfo default values."""
        from transformation_portal.lux_depth_v3.config_resolver import PresetInfo

        info = PresetInfo(
            name="TEST",
            value="test",
            display_name="Test",
        )

        assert info.description is None
        assert info.default_model is None
        assert info.tier == "standard"
