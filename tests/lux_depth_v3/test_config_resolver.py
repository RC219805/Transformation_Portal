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
            build_depth_cache_fingerprint,
            build_materials_fingerprint_payload,
            build_orchestrator_run_card_config_fingerprint,
            build_pbr_fingerprint_payload,
            build_run_card_config_fingerprint,
            compute_config_fingerprint,
            discover_presets,
            finalize_run_card_config_fingerprint,
            require_model_variant,
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
        assert callable(require_model_variant)
        assert callable(build_depth_cache_fingerprint)
        assert callable(finalize_run_card_config_fingerprint)
        assert callable(build_orchestrator_run_card_config_fingerprint)


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
        assert not presets


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


class TestEffectiveDa3RuntimeResolution:
    """Test effective DA3 runtime resolution."""

    def test_repo_local_runtime_path_discovers_repo_root_via_markers(self, monkeypatch, tmp_path):
        """Repo-local path discovery should use repo markers instead of fixed depth."""
        from transformation_portal.lux_depth_v3 import config_resolver

        module_path = tmp_path / "packages" / "src" / "transformation_portal" / "lux_depth_v3" / "config_resolver.py"
        module_path.parent.mkdir(parents=True)
        module_path.write_text("# test helper\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires=[]\n", encoding="utf-8")
        (tmp_path / "src").mkdir(exist_ok=True)

        monkeypatch.setattr(config_resolver, "__file__", str(module_path))

        assert config_resolver._repo_local_da3_python_path() == (
            tmp_path / ".runtime" / "Depth-Anything-3" / ".venv-da3" / "bin" / "python"
        )

    def test_prefers_explicit_config(self, monkeypatch, tmp_path):
        """Explicit config should win over env and repo-local discovery."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import resolve_effective_da3_python_executable

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "/env/python")
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_da3_python_path",
            lambda: discovered_python,
        )

        config = EnhanceConfig(da3_python_executable="/config/python")

        assert resolve_effective_da3_python_executable(config) == "/config/python"

    def test_uses_env_when_config_unset(self, monkeypatch, tmp_path):
        """Environment override should win when config is unset."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import resolve_effective_da3_python_executable

        missing_python = tmp_path / "missing" / "python"
        monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "/env/python")
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_da3_python_path",
            lambda: missing_python,
        )

        assert resolve_effective_da3_python_executable(EnhanceConfig()) == "/env/python"

    def test_auto_discovers_repo_local_contract(self, monkeypatch, tmp_path):
        """Repo-local DA3 runtime should resolve to the stable contract path."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            REPO_LOCAL_DA3_PYTHON,
            resolve_effective_da3_python_executable,
        )

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_DA3_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_da3_python_path",
            lambda: discovered_python,
        )

        assert resolve_effective_da3_python_executable(EnhanceConfig()) == REPO_LOCAL_DA3_PYTHON


class TestEffectiveDepthProRuntimeResolution:
    """Test effective Depth Pro runtime resolution."""

    def test_auto_discovers_repo_local_contract(self, monkeypatch, tmp_path):
        """Repo-local Depth Pro runtime should resolve to the stable contract path."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            REPO_LOCAL_DEPTH_PRO_PYTHON,
            resolve_effective_depth_pro_python_executable,
        )

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_depth_pro_python_path",
            lambda: discovered_python,
        )

        assert resolve_effective_depth_pro_python_executable(EnhanceConfig()) == REPO_LOCAL_DEPTH_PRO_PYTHON


class TestEffectiveRawRuntimeResolution:
    """Test effective RAW runtime resolution."""

    def test_auto_discovers_repo_local_contract(self, monkeypatch, tmp_path):
        """Repo-local RAW runtime should resolve to the stable contract path."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            REPO_LOCAL_RAW_PYTHON,
            resolve_effective_raw_python_executable,
        )

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_raw_python_path",
            lambda: discovered_python,
        )

        assert resolve_effective_raw_python_executable(EnhanceConfig()) == REPO_LOCAL_RAW_PYTHON

    def test_prefers_explicit_config(self, monkeypatch, tmp_path):
        """Explicit RAW config should win over env and repo-local discovery."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import resolve_effective_raw_python_executable

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("TRANSFORMATION_PORTAL_RAW_PYTHON", "/env/raw-python")
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_raw_python_path",
            lambda: discovered_python,
        )

        config = EnhanceConfig(raw_python_executable="/config/raw-python")

        assert resolve_effective_raw_python_executable(config) == "/config/raw-python"

    def test_uses_env_when_config_unset(self, monkeypatch, tmp_path):
        """Environment override should win when RAW config is unset."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import resolve_effective_raw_python_executable

        missing_python = tmp_path / "missing" / "python"
        monkeypatch.setenv("TRANSFORMATION_PORTAL_RAW_PYTHON", "/env/raw-python")
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_raw_python_path",
            lambda: missing_python,
        )

        assert resolve_effective_raw_python_executable(EnhanceConfig()) == "/env/raw-python"


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

    def test_depth_pro_model_identity_normalized_in_fingerprint(self):
        """Explicit Depth Pro runs should not serialize a stale DA3 model variant."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_depth_cache_payload,
            compute_config_fingerprint,
        )

        config = EnhanceConfig(depth_backend="depth_pro")
        fingerprint = compute_config_fingerprint(config)
        cache_payload = build_depth_cache_payload(config)

        assert fingerprint.model_variant == "apple/ml-depth-pro"
        assert cache_payload["model_variant"] == "apple/ml-depth-pro"
        assert compute_config_fingerprint(config).to_sha256() == fingerprint.to_sha256()

    def test_non_depth_pro_model_identity_remains_da3_variant(self):
        """DA3 paths retain the existing model-variant fingerprint contract."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_depth_cache_payload,
            compute_config_fingerprint,
        )

        config = EnhanceConfig(depth_backend="da3", model_variant=ModelVariant.METRIC_SMALL)
        fingerprint = compute_config_fingerprint(config)
        cache_payload = build_depth_cache_payload(config)

        assert fingerprint.model_variant == ModelVariant.METRIC_SMALL.value.name
        assert cache_payload["model_variant"] == ModelVariant.METRIC_SMALL.value.name

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

    def test_fingerprint_records_auto_discovered_da3_runtime(self, monkeypatch, tmp_path):
        """Fingerprint should capture the effective repo-local DA3 runtime."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            REPO_LOCAL_DA3_PYTHON,
            compute_config_fingerprint,
        )

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_DA3_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_da3_python_path",
            lambda: discovered_python,
        )

        fingerprint = compute_config_fingerprint(EnhanceConfig())

        assert fingerprint.da3_python_executable == REPO_LOCAL_DA3_PYTHON

    def test_fingerprint_records_auto_discovered_raw_runtime(self, monkeypatch, tmp_path):
        """Fingerprint should capture the effective repo-local RAW runtime."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            REPO_LOCAL_RAW_PYTHON,
            compute_config_fingerprint,
        )

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_raw_python_path",
            lambda: discovered_python,
        )

        fingerprint = compute_config_fingerprint(EnhanceConfig())

        assert fingerprint.raw_python_executable == REPO_LOCAL_RAW_PYTHON


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

    def test_materials_fingerprint_payload_includes_low_texture_guard_knobs(self):
        """Materials cache fingerprint must include every seam-safe guard knob."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_materials_fingerprint_payload,
        )

        config = EnhanceConfig(
            pixel_ops_low_grad_threshold=0.02,
            pixel_ops_low_tex_min_bbox_frac=0.10,
            pixel_ops_low_tex_feather_multiplier=4.0,
            pixel_ops_low_tex_delta_ceiling=0.02,
        )
        payload = build_materials_fingerprint_payload(config)

        assert payload["pixel_ops_low_grad_threshold"] == pytest.approx(0.02)
        assert payload["pixel_ops_low_tex_min_bbox_frac"] == pytest.approx(0.10)
        assert payload["pixel_ops_low_tex_feather_multiplier"] == pytest.approx(4.0)
        assert payload["pixel_ops_low_tex_delta_ceiling"] == pytest.approx(0.02)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("pixel_ops_low_grad_threshold", 0.02),
            ("pixel_ops_low_tex_min_bbox_frac", 0.10),
            ("pixel_ops_low_tex_feather_multiplier", 4.0),
            ("pixel_ops_low_tex_delta_ceiling", 0.02),
        ],
    )
    def test_config_fingerprint_changes_with_low_texture_guard_knobs(self, field, value):
        """Changing a seam-safe guard knob must invalidate Stage A reuse."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            compute_config_fingerprint,
        )

        baseline = EnhanceConfig()
        changed = EnhanceConfig(**{field: value})

        assert compute_config_fingerprint(baseline).to_sha256() != compute_config_fingerprint(changed).to_sha256()

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
        assert payload["low_saturation_warning_band"] == pytest.approx(0.0075)
        assert payload["threshold_epsilon"] == pytest.approx(1e-6)
        # APEX tier auto-upgrades depth_fallback "fail" -> "v2-auto"; that
        # policy must be in the gate fingerprint so cache replays don't serve
        # outputs from the previous fail-closed regime.
        assert payload["depth_fallback"] == "v2-auto"

    def test_apex_depth_gate_fingerprint_distinguishes_apex_strict_from_default(self):
        """`apex-strict` opt-out produces a different gate fingerprint than the
        default APEX run (auto-upgraded to v2-auto), so caches do not collide."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_apex_depth_gate_fingerprint_payload,
        )

        default_apex = EnhanceConfig(quality_tier="apex")
        strict_apex = EnhanceConfig(quality_tier="apex", depth_fallback="apex-strict")

        default_payload = build_apex_depth_gate_fingerprint_payload(default_apex)
        strict_payload = build_apex_depth_gate_fingerprint_payload(strict_apex)

        assert default_payload["depth_fallback"] == "v2-auto"
        assert strict_payload["depth_fallback"] == "fail"
        assert default_payload != strict_payload

    def test_materials_fingerprint_includes_pixel_ops_strict_policy_version(self):
        """The materials soft-passthrough is a global behavior change with no
        config knob; the policy version in the fingerprint marks the regime
        and bumps when blocker semantics shift, invalidating stale caches."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_materials_fingerprint_payload,
        )

        payload = build_materials_fingerprint_payload(EnhanceConfig(enable_materials_v3=True))

        assert payload["pixel_ops_strict_policy_version"] == "v2"

    def test_fingerprint_changes_when_low_saturation_warning_band_changes(self):
        """APEX gate fingerprint should change when the warning band changes."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            compute_config_fingerprint,
        )

        config_a = EnhanceConfig(apex_depth_low_saturation_warning_band=0.0075)
        config_b = EnhanceConfig(apex_depth_low_saturation_warning_band=0.01)

        fingerprint_a = compute_config_fingerprint(config_a)
        fingerprint_b = compute_config_fingerprint(config_b)

        assert fingerprint_a.to_sha256() != fingerprint_b.to_sha256()

    def test_apex_depth_gate_fingerprint_normalizes_negative_warning_band(self):
        """Negative warning bands should fingerprint as the effective non-negative policy."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_apex_depth_gate_fingerprint_payload,
            compute_config_fingerprint,
        )

        negative_config = EnhanceConfig(apex_depth_low_saturation_warning_band=-0.01)
        zero_config = EnhanceConfig(apex_depth_low_saturation_warning_band=0.0)

        negative_payload = build_apex_depth_gate_fingerprint_payload(negative_config)
        zero_payload = build_apex_depth_gate_fingerprint_payload(zero_config)

        assert negative_payload["low_saturation_warning_band"] == pytest.approx(0.0)
        assert negative_payload == zero_payload
        assert compute_config_fingerprint(negative_config).to_sha256() == compute_config_fingerprint(zero_config).to_sha256()

    def test_depth_cache_payload_ignores_low_saturation_warning_band(self):
        """Depth cache payload should not depend on gate demotion policy."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_depth_cache_payload,
        )

        payload_a = build_depth_cache_payload(
            EnhanceConfig(apex_depth_low_saturation_warning_band=0.0075),
        )
        payload_b = build_depth_cache_payload(
            EnhanceConfig(apex_depth_low_saturation_warning_band=0.01),
        )

        assert payload_a == payload_b

    def test_apex_depth_gate_fingerprint_normalizes_negative_threshold_epsilon(self):
        """Negative threshold epsilon should fingerprint as the effective non-negative policy."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_apex_depth_gate_fingerprint_payload,
            compute_config_fingerprint,
        )

        negative_config = EnhanceConfig(apex_depth_threshold_epsilon=-1e-3)
        zero_config = EnhanceConfig(apex_depth_threshold_epsilon=0.0)

        negative_payload = build_apex_depth_gate_fingerprint_payload(negative_config)
        zero_payload = build_apex_depth_gate_fingerprint_payload(zero_config)

        assert negative_payload["threshold_epsilon"] == pytest.approx(0.0)
        assert negative_payload == zero_payload
        assert compute_config_fingerprint(negative_config).to_sha256() == compute_config_fingerprint(zero_config).to_sha256()

    def test_fingerprint_changes_when_threshold_epsilon_changes(self):
        """APEX gate fingerprint should change when threshold epsilon changes."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            compute_config_fingerprint,
        )

        config_a = EnhanceConfig(apex_depth_threshold_epsilon=1e-6)
        config_b = EnhanceConfig(apex_depth_threshold_epsilon=1e-4)

        fingerprint_a = compute_config_fingerprint(config_a)
        fingerprint_b = compute_config_fingerprint(config_b)

        assert fingerprint_a.to_sha256() != fingerprint_b.to_sha256()

    def test_depth_cache_payload_ignores_threshold_epsilon(self):
        """Depth cache payload should not change when only gate epsilon changes."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_depth_cache_payload,
        )

        payload_a = build_depth_cache_payload(
            EnhanceConfig(apex_depth_threshold_epsilon=1e-6),
        )
        payload_b = build_depth_cache_payload(
            EnhanceConfig(apex_depth_threshold_epsilon=1e-4),
        )

        assert payload_a == payload_b


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

    def test_run_card_fingerprint_records_auto_discovered_da3_runtime(self, monkeypatch, tmp_path):
        """Run-card fingerprint should record the effective repo-local DA3 runtime."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            REPO_LOCAL_DA3_PYTHON,
            build_run_card_config_fingerprint,
        )

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_DA3_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_da3_python_path",
            lambda: discovered_python,
        )

        fingerprint = build_run_card_config_fingerprint(EnhanceConfig())

        assert fingerprint["da3_python_executable"] == REPO_LOCAL_DA3_PYTHON

    def test_run_card_fingerprint_records_auto_discovered_raw_runtime(self, monkeypatch, tmp_path):
        """Run-card fingerprint should record the effective repo-local RAW runtime."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            REPO_LOCAL_RAW_PYTHON,
            build_run_card_config_fingerprint,
        )

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_raw_python_path",
            lambda: discovered_python,
        )

        fingerprint = build_run_card_config_fingerprint(EnhanceConfig())

        assert fingerprint["raw_python_executable"] == REPO_LOCAL_RAW_PYTHON


class TestOrchestratorFingerprintHelpers:
    """Test orchestrator-facing config fingerprint helpers."""

    def test_require_model_variant_returns_resolved_variant(self):
        """Resolved model variants are exposed for orchestrator delegates."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.config_resolver import require_model_variant

        config = EnhanceConfig(model_variant=ModelVariant.METRIC_SMALL)

        assert require_model_variant(config) is ModelVariant.METRIC_SMALL

    def test_build_depth_cache_fingerprint_is_backend_and_units_scoped(self):
        """Depth cache fingerprint should change with backend output units."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.config_resolver import build_depth_cache_fingerprint

        config = EnhanceConfig(depth_backend="da3", model_variant=ModelVariant.METRIC_LARGE)

        relative = build_depth_cache_fingerprint(
            config,
            ModelVariant.METRIC_LARGE,
            "da3",
            "relative",
        )
        metric = build_depth_cache_fingerprint(
            config,
            ModelVariant.METRIC_LARGE,
            "da3",
            "meters",
        )

        assert len(relative) == 64
        assert relative != metric

    def test_finalize_run_card_config_fingerprint_recomputes_canonical_hash(self):
        """Finalization strips stale hash fields before canonicalization."""
        from transformation_portal.lux_depth_v3.config_resolver import finalize_run_card_config_fingerprint

        finalized = finalize_run_card_config_fingerprint(
            {
                "b": 2,
                "a": 1,
                "hash_algorithm": "stale",
                "canonical_json": "stale",
                "sha256": "stale",
            }
        )

        assert finalized["hash_algorithm"] == "sha256"
        assert finalized["canonical_json"] == '{"a":1,"b":2}'
        assert len(finalized["sha256"]) == 64

    def test_orchestrator_run_card_fingerprint_applies_depth_pro_overrides(self):
        """The extracted orchestrator helper preserves Depth Pro run-card overrides."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_orchestrator_run_card_config_fingerprint,
        )
        from transformation_portal.lux_depth_v3.manifest import BackendSelectionMetadata

        metadata = BackendSelectionMetadata(
            requested_backend="depth_pro",
            resolved_backend="depth_pro",
            resolution_status="success",
            resolution_reason="Depth Pro backend ready",
            model_id="apple/ml-depth-pro",
            device="cpu",
            attempts=[],
        )

        fingerprint = build_orchestrator_run_card_config_fingerprint(
            EnhanceConfig(depth_backend="depth_pro", model_variant=ModelVariant.METRIC_LARGE),
            ModelVariant.METRIC_LARGE,
            metadata,
        )

        assert fingerprint["model_variant"] == "apple/ml-depth-pro"
        assert fingerprint["preset_resolved"] == "backend:depth_pro"
        assert fingerprint["output_depth_units"] == "meters"
        assert len(fingerprint["sha256"]) == 64


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

    def test_resolver_persists_auto_discovered_da3_runtime(self, monkeypatch, tmp_path):
        """Resolver should persist the effective DA3 runtime on config."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import REPO_LOCAL_DA3_PYTHON, ConfigResolver

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_DA3_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_da3_python_path",
            lambda: discovered_python,
        )

        config = EnhanceConfig()
        resolved = ConfigResolver().resolve(config)

        assert resolved.enhance_config.da3_python_executable == REPO_LOCAL_DA3_PYTHON
        assert resolved.enhance_config is config
        assert resolved.da3_config is not None
        assert resolved.fingerprint is not None

    def test_resolver_persists_auto_discovered_raw_runtime(self, monkeypatch, tmp_path):
        """Resolver should persist the effective RAW runtime on config."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import REPO_LOCAL_RAW_PYTHON, ConfigResolver

        discovered_python = tmp_path / "bin" / "python"
        discovered_python.parent.mkdir(parents=True)
        discovered_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.config_resolver._repo_local_raw_python_path",
            lambda: discovered_python,
        )

        config = EnhanceConfig()
        resolved = ConfigResolver().resolve(config)

        assert resolved.enhance_config.raw_python_executable == REPO_LOCAL_RAW_PYTHON
        assert resolved.enhance_config is config

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
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
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
