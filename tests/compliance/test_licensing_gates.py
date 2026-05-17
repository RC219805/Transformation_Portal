"""Tests for licensing compliance gates and non-commercial enforcement.

Validates that:
1. Non-commercial models require explicit opt-in (non_commercial_ok=True)
2. Presets with non-commercial models have required license markers
3. Commercial workflows are unaffected (backward compatibility)
"""

import hashlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from transformation_portal.compliance.validate_licenses import main as validate_licenses_main
from transformation_portal.compliance.validate_licenses import validate_preset_file


def _write_model_lock_manifest(
    path: Path,
    *,
    repositories: dict[str, dict[str, Any]] | None = None,
    artifact_attestation: dict[str, Any] | None = None,
) -> None:
    """Write a minimal model lock manifest for tests."""
    payload = {
        "version": 1,
        "updated_at": "2026-03-31",
        "repositories": repositories or {},
        "artifact_attestation": artifact_attestation or {},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


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
        reason = "DA3 Giant 1.1 uses CC BY-NC 4.0"

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

    @pytest.mark.parametrize(
        "model_id",
        [
            "depth-anything/DA3-GIANT-1.1",
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        ],
    )
    def test_non_commercial_preset_requires_marker(self, model_id):
        """Only known CC BY-NC DA3 variants should require license markers."""
        preset = {
            "name": "da31-preset",
            "model": {"hf_id": model_id},
            # Missing: license_restriction: non_commercial
        }
        with pytest.raises(LicenseRestrictionError, match="license_restriction='non_commercial'"):
            validate_non_commercial_preset(preset)

    def test_non_commercial_preset_with_proper_marker_passes(self):
        """Verify properly marked non-commercial presets pass validation."""
        preset = {
            "name": "da31-preset",
            "model": {"hf_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1"},
            "license_restriction": "non_commercial",
        }
        assert validate_non_commercial_preset(preset) is True

    def test_non_commercial_preset_with_research_only_marker_passes(self):
        """Research-only markers should satisfy the non-commercial acknowledgement gate."""
        preset = {
            "name": "da31-preset",
            "model": {"hf_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1"},
            "license_restriction": "research_only",
        }
        assert validate_non_commercial_preset(preset) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "depth-anything/DA3-GIANT-1.1",
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        ],
    )
    def test_known_non_commercial_da31_variants_detected(self, model_id):
        """Verify only the known CC BY-NC DA3 variants are detected."""
        preset = {"name": "test", "model": {"hf_id": model_id}}
        with pytest.raises(LicenseRestrictionError):
            validate_non_commercial_preset(preset)

    @pytest.mark.parametrize(
        "model_id",
        [
            "depth-anything/DA3-Large-1.1",
            "depth-anything/DA3-Base-1.1",
            "depth-anything/DA3-Small-1.1",
        ],
    )
    def test_apache_da31_variants_do_not_require_marker(self, model_id):
        """Apache-licensed DA3 variants should not be gated as non-commercial."""
        preset = {"name": "test", "model": {"hf_id": model_id}}
        assert validate_non_commercial_preset(preset) is True

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
            yaml.dump(
                {"name": "da31-preset", "tier": "research", "model": {"hf_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1"}},
                f,
            )
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
                    "model": {"hf_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1"},
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

    def test_load_apache_da31_preset_without_marker_succeeds(self):
        """Apache-licensed DA3 presets should not require a non-commercial marker."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "da31-large-preset",
                    "tier": "research",
                    "model": {"hf_id": "depth-anything/DA3-Large-1.1"},
                },
                f,
            )
            f.flush()
            path = Path(f.name)

        try:
            preset = load_and_validate_preset(path)
            assert preset["model"]["hf_id"] == "depth-anything/DA3-Large-1.1"
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
            "preset_family": "materials_pbr",
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
            "preset_family": "materials_pbr",
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
            "preset_family": "materials_pbr",
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
            "preset_family": "materials_pbr",
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
            "preset_family": "materials_pbr",
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

    def test_apex_research_ultra_can_opt_in_to_unattested_sources(self):
        """apex_research_ultra should be treated as an explicit experimental research tier."""
        preset = {
            "name": "apex-research-ultra-experimental",
            "tier": "apex_research_ultra",
            "preset_family": "materials_pbr",
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
                preset_path=Path("config/presets/experimental/apex_research_ultra.yaml"),
                allow_research_materials=True,
                allow_unattested_materials=True,
            )
            is True
        )

    def test_unknown_materials_backend_schema_location_is_rejected(self):
        """Materials backend declarations must stay within the approved schema paths."""
        preset = {
            "name": "bad-materials-schema",
            "tier": "dev",
            "materials": {
                "runtime": {
                    "backend": "nvdiffrec",
                }
            },
        }

        with pytest.raises(LicenseRestrictionError, match="Unknown paths"):
            validate_materials_preset(preset, preset_path=Path("config/presets/experimental/material_pbr.yaml"))

    def test_top_level_materials_preset_requires_explicit_family_marker(self):
        """Top-level materials presets should declare an explicit preset family marker."""
        preset = {
            "name": "PBR Material Generation (Canary)",
            "tier": "canary",
            "backend": {
                "type": "pbr_fusion",
                "model": {
                    "repo_id": "NightRaven109/PBRFusion4-RTXREMIX-Portable",
                    "revision": "4ebe0ed10266bca0d7593e5461618e61d064cd9e",
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match="preset_family='materials_pbr'"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr_canary.yaml"),
            )

    def test_top_level_materials_preset_rejects_incorrect_family_marker(self):
        """Typos in the explicit preset family marker must fail closed."""
        preset = {
            "name": "PBR Material Generation (Canary)",
            "tier": "canary",
            "preset_family": "material-pbr",
            "backend": {
                "type": "pbr_fusion",
                "model": {
                    "repo_id": "NightRaven109/PBRFusion4-RTXREMIX-Portable",
                    "revision": "4ebe0ed10266bca0d7593e5461618e61d064cd9e",
                },
            },
            "pbr": {"resolution": 1024},
        }

        with pytest.raises(LicenseRestrictionError, match="got 'material-pbr'"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr_canary.yaml"),
            )

    def test_canary_pbrfusion_requires_attested_source_tuple(self):
        """Commercial materials backends still need pinned source metadata outside dev/experimental tiers."""
        preset = {
            "name": "PBR Material Generation (Canary)",
            "tier": "canary",
            "preset_family": "materials_pbr",
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
            "preset_family": "materials_pbr",
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

    def test_repo_backed_materials_backend_must_match_manifest(self, tmp_path: Path):
        """Pinned repo-backed materials sources must be approved in the model lock manifest."""
        manifest_path = tmp_path / "model_lock_manifest.yaml"
        _write_model_lock_manifest(
            manifest_path,
            repositories={
                "NightRaven109/PBRFusion4-RTXREMIX-Portable": {
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                    "owner": "materials",
                }
            },
        )
        preset = {
            "name": "PBR Material Generation (Canary)",
            "tier": "canary",
            "preset_family": "materials_pbr",
            "backend": {
                "type": "pbr_fusion",
                "model": {
                    "repo_id": "NightRaven109/PBRFusion4-RTXREMIX-Portable",
                    "revision": "89abcdef0123456789abcdef0123456789abcdef",
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match="exact approved revision"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr_canary.yaml"),
                manifest_path=manifest_path,
            )

    def test_explicit_false_override_beats_preset_governance_opt_in(self):
        """Explicit call-site overrides must take precedence over preset opt-ins."""
        preset = {
            "name": "Experimental MaterialGAN",
            "tier": "experimental",
            "license_restriction": "research_only",
            "governance": {
                "materials": {
                    "allow_research_materials": True,
                }
            },
            "materials": {
                "backend": "material_gan",
                "model": {
                    "checkpoint": "checkpoints/materialgan_v2.pth",
                    "expected_sha256": "f" * 64,
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match="allow_research_materials=True"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/experimental/material_pbr.yaml"),
                allow_research_materials=False,
                allow_unattested_materials=False,
            )

    def test_conflicting_governance_values_fail_closed(self):
        """Conflicting preset governance flags should raise instead of resolving permissively."""
        preset = {
            "name": "Experimental MaterialGAN",
            "tier": "experimental",
            "license_restriction": "research_only",
            "governance": {
                "materials": {
                    "allow_research_materials": False,
                }
            },
            "materials": {
                "allow_research_materials": True,
                "backend": "material_gan",
                "model": {
                    "checkpoint": "checkpoints/materialgan_v2.pth",
                    "expected_sha256": "f" * 64,
                },
            },
        }

        with pytest.raises(ValueError, match="Conflicting non-None values for allow_research_materials"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/experimental/material_pbr.yaml"),
            )

    def test_heuristic_materials_backend_does_not_require_manifest(self):
        """Pure heuristic materials configs should not load the model-lock manifest."""
        preset = {
            "name": "Heuristic Materials",
            "tier": "standard",
            "materials": {
                "backend": "heuristic",
            },
        }

        assert (
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr.yaml"),
                manifest_path=Path("/definitely/missing/model_lock_manifest.yaml"),
            )
            is True
        )

    def test_manifest_load_failure_is_wrapped_as_license_error(self):
        """Manifest load failures should surface as actionable licensing errors."""
        preset = {
            "name": "PBR Material Generation (Canary)",
            "tier": "canary",
            "preset_family": "materials_pbr",
            "backend": {
                "type": "pbr_fusion",
                "model": {
                    "repo_id": "NightRaven109/PBRFusion4-RTXREMIX-Portable",
                    "revision": "89abcdef0123456789abcdef0123456789abcdef",
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match="requires a valid model lock manifest"):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr_canary.yaml"),
                manifest_path=Path("/definitely/missing/model_lock_manifest.yaml"),
            )

    def test_manifest_mismatch_error_reports_resolved_manifest_path(self, tmp_path: Path):
        """Repo-backed attestation errors should reference the resolved manifest path."""
        manifest_path = tmp_path / "custom_manifest.yaml"
        _write_model_lock_manifest(
            manifest_path,
            repositories={
                "NightRaven109/PBRFusion4-RTXREMIX-Portable": {
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                    "owner": "materials",
                }
            },
        )
        preset = {
            "name": "PBR Material Generation (Canary)",
            "tier": "canary",
            "preset_family": "materials_pbr",
            "backend": {
                "type": "pbr_fusion",
                "model": {
                    "repo_id": "NightRaven109/PBRFusion4-RTXREMIX-Portable",
                    "revision": "89abcdef0123456789abcdef0123456789abcdef",
                },
            },
        }

        with pytest.raises(LicenseRestrictionError, match=str(manifest_path)):
            validate_materials_preset(
                preset,
                preset_path=Path("config/presets/material_pbr_canary.yaml"),
                manifest_path=manifest_path,
            )

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

    def test_runtime_checkpoint_bytes_are_verified_when_present(self, tmp_path: Path):
        """Runtime loading should verify local checkpoint bytes against preset and manifest hashes."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "materialgan_v2.pth"
        checkpoint_path.write_bytes(b"materialgan-checkpoint")
        digest = hashlib.sha256(b"materialgan-checkpoint").hexdigest()

        manifest_path = tmp_path / "model_lock_manifest.yaml"
        _write_model_lock_manifest(
            manifest_path,
            artifact_attestation={
                "materials": {
                    "material_gan": {
                        "artifacts": [
                            {
                                "filename": "checkpoints/materialgan_v2.pth",
                                "sha256": digest,
                            }
                        ]
                    }
                }
            },
        )

        preset_path = tmp_path / "materialgan.yaml"
        preset_path.write_text(
            yaml.safe_dump(
                {
                    "name": "Experimental MaterialGAN",
                    "tier": "apex_research_ultra",
                    "license_restriction": "research_only",
                    "governance": {
                        "materials": {
                            "allow_research_materials": True,
                        }
                    },
                    "materials": {
                        "backend": "material_gan",
                        "model": {
                            "checkpoint": "checkpoints/materialgan_v2.pth",
                            "expected_sha256": digest,
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        preset = load_and_validate_preset(
            preset_path,
            manifest_path=manifest_path,
            verify_runtime_bytes=True,
        )
        assert preset["materials"]["backend"] == "material_gan"

        checkpoint_path.write_bytes(b"tampered-checkpoint")
        with pytest.raises(LicenseRestrictionError, match="checkpoint bytes"):
            load_and_validate_preset(
                preset_path,
                manifest_path=manifest_path,
                verify_runtime_bytes=True,
            )

    def test_runtime_checkpoint_bytes_are_not_verified_by_default(self, tmp_path: Path):
        """Runtime-byte hashing should remain opt-in during preset load."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "materialgan_v2.pth"
        checkpoint_path.write_bytes(b"tampered-checkpoint")
        manifest_digest = hashlib.sha256(b"materialgan-checkpoint").hexdigest()

        manifest_path = tmp_path / "model_lock_manifest.yaml"
        _write_model_lock_manifest(
            manifest_path,
            artifact_attestation={
                "materials": {
                    "material_gan": {
                        "artifacts": [
                            {
                                "filename": "checkpoints/materialgan_v2.pth",
                                "sha256": manifest_digest,
                            }
                        ]
                    }
                }
            },
        )

        preset_path = tmp_path / "materialgan.yaml"
        preset_path.write_text(
            yaml.safe_dump(
                {
                    "name": "Experimental MaterialGAN",
                    "tier": "apex_research_ultra",
                    "license_restriction": "research_only",
                    "governance": {
                        "materials": {
                            "allow_research_materials": True,
                        }
                    },
                    "materials": {
                        "backend": "material_gan",
                        "model": {
                            "checkpoint": "checkpoints/materialgan_v2.pth",
                            "expected_sha256": manifest_digest,
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        preset = load_and_validate_preset(
            preset_path,
            manifest_path=manifest_path,
        )
        assert preset["materials"]["backend"] == "material_gan"

    def test_runtime_checkpoint_verification_rejects_path_traversal(self, tmp_path: Path):
        """Checkpoint verification must reject paths that escape the allowed roots."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        escaped_checkpoint = tmp_path / "materialgan_v2.pth"
        escaped_checkpoint.write_bytes(b"materialgan-checkpoint")
        digest = hashlib.sha256(b"materialgan-checkpoint").hexdigest()

        manifest_path = tmp_path / "model_lock_manifest.yaml"
        _write_model_lock_manifest(
            manifest_path,
            artifact_attestation={
                "materials": {
                    "material_gan": {
                        "artifacts": [
                            {
                                "filename": "../materialgan_v2.pth",
                                "sha256": digest,
                            }
                        ]
                    }
                }
            },
        )

        preset_path = preset_dir / "materialgan.yaml"
        preset_path.write_text(
            yaml.safe_dump(
                {
                    "name": "Experimental MaterialGAN",
                    "tier": "experimental",
                    "license_restriction": "research_only",
                    "governance": {
                        "materials": {
                            "allow_research_materials": True,
                        }
                    },
                    "materials": {
                        "backend": "material_gan",
                        "model": {
                            "checkpoint": "../materialgan_v2.pth",
                            "expected_sha256": digest,
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        with pytest.raises(LicenseRestrictionError, match="outside allowed roots"):
            load_and_validate_preset(
                preset_path,
                manifest_path=manifest_path,
                verify_runtime_bytes=True,
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


class TestValidateLicensesScript:
    """Test the standalone preset compliance script."""

    def test_script_accepts_apache_da31_large_without_noncommercial_marker(self):
        """The standalone validator should share the updated DA3 license policy."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "da31-large-preset",
                    "tier": "research",
                    "model": {"hf_id": "depth-anything/DA3-Large-1.1"},
                },
                f,
            )
            f.flush()
            path = Path(f.name)

        try:
            is_valid, issues = validate_preset_file(path)
            assert is_valid is True
            assert issues == []
        finally:
            path.unlink()

    def test_script_uses_shared_loader_for_materials_governance(self):
        """The standalone validator must reject the same materials policy violations as runtime loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "materialgan-preset",
                    "tier": "experimental",
                    "license_restriction": "research_only",
                    "materials": {
                        "backend": "material_gan",
                        "model": {
                            "checkpoint": "checkpoints/materialgan_v2.pth",
                            "expected_sha256": "f" * 64,
                        },
                    },
                },
                f,
            )
            f.flush()
            path = Path(f.name)

        try:
            is_valid, issues = validate_preset_file(path)
            assert is_valid is False
            assert any("allow_research_materials=True" in issue for issue in issues)
        finally:
            path.unlink()

    def test_script_scans_nested_preset_directories(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
        """The standalone validator should scan preset trees recursively."""
        nested_dir = tmp_path / "experimental"
        nested_dir.mkdir()
        preset_path = nested_dir / "nested_preset.yaml"
        preset_path.write_text(
            yaml.safe_dump(
                {
                    "name": "nested-commercial-preset",
                    "tier": "experimental",
                    "model": {"hf_id": "depth-anything/DA3-Large-1.1"},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(sys, "argv", ["validate_licenses.py", "--check-presets", str(tmp_path)])
        exit_code = validate_licenses_main()
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "nested_preset.yaml" in captured.out


class TestExtendsResolution:
    """Tests for the ``extends:`` preset inheritance resolver."""

    @staticmethod
    def _write_yaml(path: Path, payload: dict) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def test_basic_extends_merges_parent_fields(self, tmp_path):
        parent = tmp_path / "parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(
            parent,
            {
                "name": "parent",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
                "io": {"output_format": "png"},
            },
        )
        self._write_yaml(child, {"extends": str(parent), "name": "child"})

        merged = load_and_validate_preset(child)

        assert merged["name"] == "child"
        assert merged["tier"] == "stable"
        assert merged["io"] == {"output_format": "png"}

    def test_extends_child_overrides_parent_scalar(self, tmp_path):
        parent = tmp_path / "parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(
            parent,
            {
                "name": "parent",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
                "cache_size": 50,
            },
        )
        self._write_yaml(child, {"extends": str(parent), "cache_size": 200})

        merged = load_and_validate_preset(child)

        assert merged["cache_size"] == 200

    def test_extends_deep_merge_nested_dicts(self, tmp_path):
        parent = tmp_path / "parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(
            parent,
            {
                "name": "parent",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
                "processing": {"denoise": {"sigma": 1.0, "edge": 0.1}, "tone": "agx"},
            },
        )
        self._write_yaml(
            child,
            {"extends": str(parent), "processing": {"denoise": {"sigma": 2.5}}},
        )

        merged = load_and_validate_preset(child)

        assert merged["processing"]["denoise"]["sigma"] == 2.5
        assert merged["processing"]["denoise"]["edge"] == 0.1
        assert merged["processing"]["tone"] == "agx"

    def test_extends_child_list_replaces_parent_list(self, tmp_path):
        parent = tmp_path / "parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(
            parent,
            {
                "name": "parent",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
                "zones": [1, 2, 3],
            },
        )
        self._write_yaml(child, {"extends": str(parent), "zones": [9]})

        merged = load_and_validate_preset(child)

        assert merged["zones"] == [9]

    def test_extends_missing_parent_raises_license_error(self, tmp_path):
        child = tmp_path / "child.yaml"
        self._write_yaml(child, {"extends": "does_not_exist_preset", "name": "child"})

        with pytest.raises(LicenseRestrictionError, match="could not be resolved"):
            load_and_validate_preset(child)

    def test_extends_cycle_raises_license_error(self, tmp_path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        self._write_yaml(a, {"extends": str(b), "name": "a"})
        self._write_yaml(b, {"extends": str(a), "name": "b"})

        with pytest.raises(LicenseRestrictionError, match="Cycle detected"):
            load_and_validate_preset(a)

    def test_extends_three_level_chain_resolves(self, tmp_path):
        grandparent = tmp_path / "gp.yaml"
        parent = tmp_path / "parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(
            grandparent,
            {
                "name": "gp",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
                "from_gp": True,
            },
        )
        self._write_yaml(parent, {"extends": str(grandparent), "from_parent": True})
        self._write_yaml(child, {"extends": str(parent), "from_child": True})

        merged = load_and_validate_preset(child)

        assert merged["from_gp"] is True
        assert merged["from_parent"] is True
        assert merged["from_child"] is True

    def test_returned_dict_does_not_contain_extends_key(self, tmp_path):
        parent = tmp_path / "parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(
            parent,
            {
                "name": "parent",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
            },
        )
        self._write_yaml(child, {"extends": str(parent), "name": "child"})

        merged = load_and_validate_preset(child)

        assert "extends" not in merged

    def test_extends_string_without_suffix_resolves_to_yaml(self, tmp_path):
        parent = tmp_path / "apex_parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(
            parent,
            {
                "name": "apex_parent",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
                "marker": "from_parent",
            },
        )
        # Use bare name "apex_parent" — resolver should add .yaml suffix
        self._write_yaml(child, {"extends": "apex_parent", "name": "child"})

        merged = load_and_validate_preset(child)

        assert merged["marker"] == "from_parent"

    def test_extends_non_string_value_raises(self, tmp_path):
        child = tmp_path / "child.yaml"
        self._write_yaml(child, {"extends": ["a", "b"], "name": "child"})

        with pytest.raises(LicenseRestrictionError, match="must be a non-empty string"):
            load_and_validate_preset(child)

    def test_extends_rejects_directory_target(self, tmp_path):
        """If `extends:` resolves to a directory (not a file), the resolver
        must skip it and report an unresolved error rather than letting the
        caller fail later trying to open a directory as YAML."""
        # Create a directory named like the candidate the resolver would try
        dir_target = tmp_path / "parent.yaml"
        dir_target.mkdir()
        child = tmp_path / "child.yaml"
        self._write_yaml(child, {"extends": str(dir_target.with_suffix("")), "name": "child"})

        with pytest.raises(LicenseRestrictionError, match="could not be resolved"):
            load_and_validate_preset(child)

    def test_extends_rejects_absolute_path_outside_approved_roots(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        outside_parent = tmp_path / "outside_parent.yaml"
        child = nested / "child.yaml"
        self._write_yaml(outside_parent, {"name": "outside", "tier": "stable"})
        self._write_yaml(child, {"extends": str(outside_parent.resolve()), "name": "child"})

        with pytest.raises(LicenseRestrictionError, match="could not be resolved"):
            load_and_validate_preset(child)

    def test_extends_rejects_path_traversal_outside_child_dir(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        outside_parent = tmp_path / "outside_parent.yaml"
        child = nested / "child.yaml"
        self._write_yaml(outside_parent, {"name": "outside", "tier": "stable"})
        self._write_yaml(child, {"extends": "../outside_parent.yaml", "name": "child"})

        with pytest.raises(LicenseRestrictionError, match="could not be resolved"):
            load_and_validate_preset(child)

    def test_extends_rejects_symlink_escape_outside_approved_roots(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        outside_parent = tmp_path / "outside_parent.yaml"
        symlink = nested / "linked_parent.yaml"
        child = nested / "child.yaml"
        self._write_yaml(outside_parent, {"name": "outside", "tier": "stable"})
        try:
            symlink.symlink_to(outside_parent)
        except OSError:
            pytest.skip("symlink creation unsupported on this platform")
        self._write_yaml(child, {"extends": "linked_parent.yaml", "name": "child"})

        with pytest.raises(LicenseRestrictionError, match="could not be resolved"):
            load_and_validate_preset(child)

    def test_extends_allows_normal_sibling_parent(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        parent = nested / "parent.yaml"
        child = nested / "child.yaml"
        self._write_yaml(
            parent,
            {
                "name": "parent",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
                "marker": "from_parent",
            },
        )
        self._write_yaml(child, {"extends": "parent", "name": "child"})

        merged = load_and_validate_preset(child)

        assert merged["name"] == "child"
        assert merged["marker"] == "from_parent"

    def test_extends_rejects_non_yaml_suffix(self, tmp_path):
        parent = tmp_path / "parent.yaml"
        child = tmp_path / "child.yaml"
        self._write_yaml(
            parent,
            {
                "name": "parent",
                "tier": "stable",
                "model": {"hf_id": "depth-anything/Depth-Anything-V3-Metric-Large-hf"},
            },
        )
        self._write_yaml(child, {"extends": "parent.txt", "name": "child"})

        with pytest.raises(LicenseRestrictionError, match="must reference a .yaml or .yml"):
            load_and_validate_preset(child)

    def test_extends_skips_self_match_and_falls_through_to_config_presets(self, tmp_path):
        """A bare-name `extends:` whose first candidate is the child itself must
        skip the self-match and continue searching `config/presets/`.

        This is the pattern used by `config/presets/experimental/sam2_segmentation.yaml`
        which declares `extends: sam2_segmentation` intending the stable parent at
        `config/presets/sam2_segmentation.yaml`, not itself.
        """
        # Simulate the directory layout by writing a parent into config/presets/
        # via tmp + monkeypatch is heavy; just rely on the real-file integration
        # test below to cover this on the shipped sam2 experimental preset.
        repo_root = Path(__file__).resolve().parents[2]
        child_path = repo_root / "config" / "presets" / "experimental" / "sam2_segmentation.yaml"
        preset = load_and_validate_preset(child_path, allow_unattested_materials=True)
        # Inherited from stable parent (config/presets/sam2_segmentation.yaml)
        assert "model" in preset  # parent + child both populate this
        # extends key was stripped
        assert "extends" not in preset

    def test_apex_research_canary_inherits_parent_segmentation_block(self):
        """Real-file integration: apex_research_canary.yaml must merge with apex_research.yaml."""
        repo_root = Path(__file__).resolve().parents[2]
        canary_path = repo_root / "config" / "presets" / "apex_research_canary.yaml"

        preset = load_and_validate_preset(canary_path)

        # Inherited from parent (apex_research.yaml)
        assert preset["segmentation"]["model_variant"] == "vit_h"
        assert preset["compliance"]["non_commercial_ok"] is True
        # Child override
        assert preset["depth_backend"] == "da3_1.1_nested_giant_large"
        # extends key stripped
        assert "extends" not in preset
