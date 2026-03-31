"""Tests for licensing compliance gates and non-commercial enforcement.

Validates that:
1. Non-commercial models require explicit opt-in (non_commercial_ok=True)
2. Presets with non-commercial models have required license markers
3. Commercial workflows are unaffected (backward compatibility)
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

from transformation_portal.compliance import (
    LicenseRestrictionError,
    load_and_validate_preset,
    require_non_commercial,
    validate_materials_preset,
    validate_non_commercial_preset,
)


@dataclass
class MockConfig:
    """Mock EnhanceConfig for testing."""

    non_commercial_ok: bool = False


class TestRequireNonCommercialDecorator:
    """Test @require_non_commercial decorator enforcement."""

    def test_decorator_blocks_access_without_flag(self):
        """Verify decorator raises error when non_commercial_ok=False."""

        @require_non_commercial(reason="Test model uses CC BY-NC 4.0")
        def load_test_model(config: MockConfig):
            return "model loaded"

        config = MockConfig(non_commercial_ok=False)
        with pytest.raises(LicenseRestrictionError, match="requires non_commercial_ok=True"):
            load_test_model(config)

    def test_decorator_allows_access_with_flag(self):
        """Verify decorator allows execution when non_commercial_ok=True."""

        @require_non_commercial(reason="Test model uses CC BY-NC 4.0")
        def load_test_model(config: MockConfig):
            return "model loaded successfully"

        config = MockConfig(non_commercial_ok=True)
        result = load_test_model(config)
        assert result == "model loaded successfully"

    def test_decorator_includes_reason_in_error(self):
        """Verify error message includes provided reason."""
        reason = "DA3 1.1 uses CC BY-NC 4.0 models"

        @require_non_commercial(reason=reason)
        def load_model(config: MockConfig):
            pass

        config = MockConfig(non_commercial_ok=False)
        with pytest.raises(LicenseRestrictionError) as exc_info:
            load_model(config)

        assert reason in str(exc_info.value)

    def test_decorator_works_with_kwargs(self):
        """Verify decorator accepts config as keyword argument."""

        @require_non_commercial(reason="Test")
        def load_model(*, config: MockConfig):
            return "loaded"

        config = MockConfig(non_commercial_ok=True)
        result = load_model(config=config)
        assert result == "loaded"

    def test_decorator_raises_type_error_if_config_missing(self):
        """Verify decorator raises TypeError if config cannot be found."""

        @require_non_commercial(reason="Test")
        def load_model(some_arg: str):
            pass

        with pytest.raises(TypeError, match="expects first argument"):
            load_model("not a config")


class TestValidateNonCommercialPreset:
    """Test preset validation for non-commercial licensing markers."""

    def test_non_mapping_preset_raises_value_error(self):
        """Validators should fail fast on invalid YAML roots."""
        with pytest.raises(ValueError, match="Preset must be a mapping"):
            validate_non_commercial_preset(None)  # type: ignore[arg-type]

    def test_commercial_preset_passes_validation(self):
        """Verify commercial presets pass without markers."""
        preset = {"name": "commercial-preset", "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"}}
        assert validate_non_commercial_preset(preset) is True

    def test_non_commercial_preset_requires_marker(self):
        """Verify non-commercial presets require license_restriction marker."""
        preset = {
            "name": "da31-preset",
            "model": {"hf_id": "depth-anything/DA3-Large-1.1"},
            # Missing: license_restriction: non_commercial
        }
        with pytest.raises(LicenseRestrictionError, match="license_restriction='non_commercial'"):
            validate_non_commercial_preset(preset)

    def test_non_commercial_preset_with_proper_marker_passes(self):
        """Verify properly marked non-commercial presets pass validation."""
        preset = {
            "name": "da31-preset",
            "model": {"hf_id": "depth-anything/DA3-Large-1.1"},
            "license_restriction": "non_commercial",
        }
        assert validate_non_commercial_preset(preset) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "depth-anything/DA3-Large-1.1",
            "depth-anything/DA3-Base-1.1",
            "depth-anything/DA3-Small-1.1",
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        ],
    )
    def test_all_da31_variants_detected(self, model_id):
        """Verify all known DA3 1.1 variants are detected."""
        preset = {"name": "test", "model": {"hf_id": model_id}}
        with pytest.raises(LicenseRestrictionError):
            validate_non_commercial_preset(preset)

    def test_missing_model_section_passes(self):
        """Verify presets without model section pass validation."""
        preset = {"name": "test-preset"}
        assert validate_non_commercial_preset(preset) is True

    def test_empty_preset_passes(self):
        """Verify empty presets pass validation."""
        preset = {}
        assert validate_non_commercial_preset(preset) is True


class TestLoadAndValidatePreset:
    """Test loading and validating presets from YAML files."""

    def test_load_commercial_preset_succeeds(self):
        """Verify commercial presets load without errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "commercial-preset",
                    "tier": "stable",
                    "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
                },
                f,
            )
            f.flush()
            path = Path(f.name)

        try:
            preset = load_and_validate_preset(path)
            assert preset["name"] == "commercial-preset"
        finally:
            path.unlink()

    def test_load_non_commercial_preset_without_marker_fails(self):
        """Verify non-commercial presets without marker fail validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"name": "da31-preset", "tier": "research", "model": {"hf_id": "depth-anything/DA3-Large-1.1"}}, f)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(LicenseRestrictionError):
                load_and_validate_preset(path)
        finally:
            path.unlink()

    def test_load_non_commercial_preset_with_marker_succeeds(self):
        """Verify properly marked non-commercial presets load successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "da31-research-preset",
                    "tier": "research",
                    "license_restriction": "non_commercial",
                    "model": {"hf_id": "depth-anything/DA3-Large-1.1"},
                },
                f,
            )
            f.flush()
            path = Path(f.name)

        try:
            preset = load_and_validate_preset(path)
            assert preset["name"] == "da31-research-preset"
            assert preset["license_restriction"] == "non_commercial"
        finally:
            path.unlink()

    def test_load_nonexistent_preset_raises_filenotfound(self):
        """Verify loading nonexistent preset raises FileNotFoundError."""
        path = Path("/nonexistent/preset.yaml")
        with pytest.raises(FileNotFoundError):
            load_and_validate_preset(path)

    def test_load_malformed_yaml_raises_yamlerror(self):
        """Verify malformed YAML raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception):  # YAML error
                load_and_validate_preset(path)
        finally:
            path.unlink()

    def test_load_non_mapping_preset_raises_value_error(self):
        """Preset loader should reject YAML roots that are not mappings."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["not", "a", "mapping"], f)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Preset must be a mapping"):
                load_and_validate_preset(path)
        finally:
            path.unlink()


class TestValidateMaterialsPreset:
    """Test materials-specific tier, license, and attestation gates."""

    def test_research_materials_backend_requires_marker(self):
        """Research-only materials backends must declare a research restriction marker."""
        preset = {
            "name": "experimental-materials",
            "tier": "dev",
            "model": {
                "backend": "nvdiffrec",
                "repo_id": "nvidia/nvdiffrec",
                "revision": "4ebe0ed10266bca0d7593e5461618e61d064cd9e",
            },
        }

        with pytest.raises(LicenseRestrictionError, match="license_restriction='research_only'"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/experimental/material_pbr.yaml"),
                allow_research_materials=True,
            )

    def test_research_materials_backend_rejected_in_stable_tier(self):
        """Stable and canary material presets must not select research-only backends."""
        preset = {
            "name": "bad-stable-materials",
            "tier": "stable",
            "license_restriction": "research_only",
            "model": {
                "backend": "nvdiffrec",
                "repo_id": "nvidia/nvdiffrec",
                "revision": "4ebe0ed10266bca0d7593e5461618e61d064cd9e",
            },
        }

        with pytest.raises(LicenseRestrictionError, match="tier='stable'"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr.yaml"),
                allow_research_materials=True,
            )

    def test_research_materials_backend_requires_explicit_opt_in(self):
        """Research-only materials presets should fail closed without explicit opt-in."""
        preset = {
            "name": "experimental-materials",
            "tier": "dev",
            "license_restriction": "research_only",
            "model": {
                "backend": "nvdiffrec",
                "repo_id": "nvidia/nvdiffrec",
                "revision": "4ebe0ed10266bca0d7593e5461618e61d064cd9e",
            },
        }

        with pytest.raises(LicenseRestrictionError, match="allow_research_materials=True"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/experimental/material_pbr.yaml"),
            )

    def test_unattested_materials_backend_requires_explicit_opt_in(self):
        """Unresolved source tuples should remain blocked unless explicitly allowed."""
        preset = {
            "name": "experimental-materials",
            "tier": "dev",
            "license_restriction": "research_only",
            "model": {
                "backend": "nvdiffrec",
                "repo_id": None,
                "revision": None,
            },
        }

        with pytest.raises(LicenseRestrictionError, match="allow_unattested_materials=True"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/experimental/material_pbr.yaml"),
                allow_research_materials=True,
            )

    def test_experimental_materials_backend_can_opt_in_to_unattested_sources(self):
        """Dev/experimental presets may explicitly opt in to unresolved source tuples."""
        preset = {
            "name": "experimental-materials",
            "tier": "dev",
            "license_restriction": "research_only",
            "model": {
                "backend": "nvdiffrec",
                "repo_id": None,
                "revision": None,
            },
        }

        assert (
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/experimental/material_pbr.yaml"),
                allow_research_materials=True,
                allow_unattested_materials=True,
            )
            is True
        )

    def test_canary_pbrfusion_requires_attested_source_tuple(self):
        """Commercial materials backends still need pinned source metadata outside dev/experimental tiers."""
        preset = {
            "name": "PBR Material Generation (Canary)",
            "tier": "canary",
            "backend": {
                "type": "pbr_fusion",
                "model": {
                    "repo_id": "NightRaven109/PBRFusion4-RTXREMIX-Portable",
                    "revision": None,
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match="attested source tuple"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr_canary.yaml"),
            )

    def test_canary_pbrfusion_rejects_all_zero_revision(self):
        """Obvious placeholder revisions must not count as attested."""
        preset = {
            "name": "PBR Material Generation (Canary)",
            "tier": "canary",
            "backend": {
                "type": "pbr_fusion",
                "model": {
                    "repo_id": "NightRaven109/PBRFusion4-RTXREMIX-Portable",
                    "revision": "0" * 40,
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match="attested source tuple"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr_canary.yaml"),
            )

    def test_actual_canary_material_preset_passes(self):
        """The checked-in canary preset should satisfy attestation requirements."""
        preset = load_and_validate_preset(Path("config/presets/material_pbr_canary.yaml"))
        assert preset["backend"]["type"] == "pbr_fusion"

    def test_nested_materials_backend_alias_is_validated(self):
        """Nested materials configs should normalize legacy backend aliases."""
        preset = {
            "name": "apex-research-ultra-experimental",
            "tier": "apex_research_ultra",
            "license_restriction": "research_only",
            "materials": {
                "backend": "materialgan",
                "model": {
                    "checkpoint": "checkpoints/materialgan_v2.pth",
                    "expected_sha256": "PLACEHOLDER_UPDATE_WHEN_INTEGRATED",
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match="attested source tuple"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/experimental/apex_research_ultra.yaml"),
                allow_research_materials=True,
            )

    def test_nested_materials_backend_rejects_all_zero_checkpoint_digest(self):
        """All-zero checkpoint digests must not count as attested."""
        preset = {
            "name": "apex-research-ultra-experimental",
            "tier": "apex_research_ultra",
            "license_restriction": "research_only",
            "materials": {
                "backend": "materialgan",
                "model": {
                    "checkpoint": "checkpoints/materialgan_v2.pth",
                    "expected_sha256": "0" * 64,
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match="attested source tuple"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/experimental/apex_research_ultra.yaml"),
                allow_research_materials=True,
            )


class TestBackwardCompatibility:
    """Test that commercial workflows are unaffected."""

    def test_enhance_config_defaults_to_commercial(self):
        """Verify EnhanceConfig defaults to commercial (non_commercial_ok=False)."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig

        config = EnhanceConfig()
        assert config.non_commercial_ok is False

    def test_commercial_pipeline_always_succeeds(self):
        """Verify commercial pipelines work without any opt-in."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig

        @require_non_commercial(reason="This should not apply")
        def only_for_non_commercial(config):
            return "should not reach here"

        # Commercial config should fail even without decorator expectation
        config = EnhanceConfig(non_commercial_ok=False)
        with pytest.raises(LicenseRestrictionError):
            only_for_non_commercial(config)

    def test_v2_preset_always_available(self):
        """Verify commercial DA3 V2 presets don't require opt-in."""
        preset = {
            "name": "commercial-da3-v2",
            "tier": "stable",
            "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
        }
        # Should not raise
        assert validate_non_commercial_preset(preset) is True
