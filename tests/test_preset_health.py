"""Tests for preset health validation (ADR-026 §M1.1).

Covers:
- Placeholder string detection (NEEDS_VERIFICATION_*, PLACEHOLDER_*)
- Depth backend ID validation against registry
- Pipeline stage status reporting
- Health report serialization
- Ultra preset specific validation
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from transformation_portal.core.config.preset_health import HealthIssue, PresetHealthReport, StageStatus, validate_preset


class TestPlaceholderDetection:
    """Placeholder strings in preset values must be flagged as errors."""

    def test_detects_needs_verification_placeholder(self, tmp_path):
        """NEEDS_VERIFICATION_* strings must produce an error."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {
                        "backend": "da3",
                        "models": [
                            {
                                "name": "da3",
                                "revision": "NEEDS_VERIFICATION_0000000000000000000000",
                            }
                        ],
                    },
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3", "ensemble"])

        assert not report.healthy
        placeholder_issues = [i for i in report.issues if i.category == "placeholder"]
        assert len(placeholder_issues) >= 1
        assert "NEEDS_VERIFICATION" in placeholder_issues[0].message

    def test_detects_placeholder_update_string(self, tmp_path):
        """PLACEHOLDER_UPDATE_* strings must produce an error."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {"backend": "da3"},
                    "segmentation": {
                        "backend": "sam2_hiera_large",
                        "model": {"expected_sha256": "PLACEHOLDER_UPDATE_WHEN_INTEGRATED"},
                    },
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])

        assert not report.healthy
        placeholder_issues = [i for i in report.issues if i.category == "placeholder"]
        assert len(placeholder_issues) >= 1

    def test_detects_pending_verification_placeholder(self, tmp_path):
        """PENDING_VERIFICATION strings should be caught by the shared placeholder policy."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "materials": {
                        "backend": "heuristic",
                        "model": {
                            "revision": "PENDING_VERIFICATION",
                        },
                    },
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])

        placeholder_issues = [i for i in report.issues if i.category == "placeholder"]
        assert len(placeholder_issues) >= 1
        assert "PENDING_VERIFICATION" in placeholder_issues[0].message

    def test_detects_empty_string_placeholder(self, tmp_path):
        """Empty-string placeholders should align with compliance placeholder scanning."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "materials": {
                        "backend": "heuristic",
                        "model": {
                            "license": "",
                        },
                    },
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])

        placeholder_issues = [i for i in report.issues if i.category == "placeholder"]
        assert len(placeholder_issues) >= 1
        assert "Placeholder value detected" in placeholder_issues[0].message

    def test_clean_preset_has_no_placeholder_errors(self, tmp_path):
        """A preset without placeholders should have no placeholder errors."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "clean-preset",
                    "depth": {
                        "backend": "da3",
                        "models": [
                            {
                                "name": "da3",
                                "revision": "abc123def456",
                                "expected_sha256": "deadbeef" * 8,
                            }
                        ],
                    },
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])

        placeholder_issues = [i for i in report.issues if i.category == "placeholder"]
        assert len(placeholder_issues) == 0


class TestBackendIdValidation:
    """Depth model names must exist in the backend registry."""

    def test_unknown_backend_id_flagged(self, tmp_path):
        """A model name not in the registry should produce an error."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {
                        "backend": "ensemble",
                        "models": [
                            {"name": "depth_pro"},
                            {"name": "unknown_backend_xyz"},
                        ],
                    },
                }
            )
        )

        report = validate_preset(
            preset,
            available_depth_backend_ids=["da3", "depth_pro", "ensemble", "depthcrafter", "synthetic"],
        )

        backend_issues = [i for i in report.issues if i.category == "backend_id"]
        assert len(backend_issues) == 1
        assert "unknown_backend_xyz" in backend_issues[0].message

    def test_valid_backend_ids_pass(self, tmp_path):
        """All valid backend IDs should produce no backend_id errors."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {
                        "backend": "ensemble",
                        "models": [
                            {"name": "depth_pro"},
                            {"name": "da3"},
                            {"name": "depthcrafter"},
                        ],
                    },
                }
            )
        )

        report = validate_preset(
            preset,
            available_depth_backend_ids=["da3", "depth_pro", "ensemble", "depthcrafter", "synthetic"],
        )

        backend_issues = [i for i in report.issues if i.category == "backend_id"]
        assert len(backend_issues) == 0

    def test_old_da3_variant_name_flagged(self, tmp_path):
        """The old 'da3_1.1_nested_giant_large' name should be flagged (§1.1 regression guard)."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {
                        "backend": "ensemble",
                        "models": [
                            {"name": "da3_1.1_nested_giant_large"},
                        ],
                    },
                }
            )
        )

        report = validate_preset(
            preset,
            available_depth_backend_ids=["da3", "depth_pro", "ensemble", "depthcrafter", "synthetic"],
        )

        backend_issues = [i for i in report.issues if i.category == "backend_id"]
        assert len(backend_issues) == 1
        assert "da3_1.1_nested_giant_large" in backend_issues[0].message

    def test_unknown_top_level_backend_flagged(self, tmp_path):
        """Top-level depth.backend must also be valid."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {"backend": "nonexistent_backend"},
                }
            )
        )

        report = validate_preset(
            preset,
            available_depth_backend_ids=["da3", "depth_pro", "ensemble"],
        )

        backend_issues = [i for i in report.issues if i.category == "backend_id"]
        assert len(backend_issues) == 1

    def test_registry_unavailable_reports_warning(self, tmp_path, monkeypatch):
        """When registry lookup fails, report a warning instead of silently passing."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {"backend": "da3"},
                }
            )
        )

        module_name = "transformation_portal.depth.backends.registry"
        fake_registry_module = types.ModuleType(module_name)

        class FailingDepthBackendRegistry:
            def __init__(self):
                raise RuntimeError("registry unavailable")

        fake_registry_module.DepthBackendRegistry = FailingDepthBackendRegistry
        monkeypatch.setitem(sys.modules, module_name, fake_registry_module)

        report = validate_preset(preset, available_depth_backend_ids=None)
        registry_issues = [i for i in report.issues if i.category == "backend_registry_unavailable"]
        assert len(registry_issues) == 1
        assert registry_issues[0].severity == "warning"


class TestStageStatusReporting:
    """Pipeline stage status should be explicitly reported."""

    def test_declared_stages_marked(self, tmp_path):
        """Stages present in the preset should be marked as declared."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {"backend": "da3"},
                    "enhancement": {"tone_mapping": {"method": "aces_filmic"}},
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])

        stage_map = {s.name: s for s in report.stages}
        assert stage_map["depth"].declared is True
        assert stage_map["enhancement"].declared is True

    def test_missing_stages_explicitly_skipped(self, tmp_path):
        """Stages NOT in the preset should have a skip reason."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {"backend": "da3"},
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])

        stage_map = {s.name: s for s in report.stages}
        # Segmentation, materials, reconstruction, validation should be skipped
        assert stage_map["segmentation"].declared is False
        assert stage_map["segmentation"].skipped_reason is not None
        assert stage_map["materials"].skipped_reason is not None
        assert stage_map["reconstruction"].skipped_reason is not None
        assert stage_map["validation"].skipped_reason is not None

    def test_all_six_stages_reported(self, tmp_path):
        """All six ADR-026 pipeline stages should appear in the report."""
        preset = tmp_path / "test.yaml"
        preset.write_text(yaml.dump({"name": "test-preset"}))

        report = validate_preset(preset, available_depth_backend_ids=[])

        stage_names = [s.name for s in report.stages]
        assert "depth" in stage_names
        assert "segmentation" in stage_names
        assert "materials" in stage_names
        assert "reconstruction" in stage_names
        assert "enhancement" in stage_names
        assert "validation" in stage_names

    def test_depth_backend_availability_is_reported(self, tmp_path):
        """Depth stage should include backend availability when IDs are known."""
        preset = tmp_path / "test.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "test-preset",
                    "depth": {"backend": "da3"},
                }
            )
        )

        report_valid = validate_preset(
            preset,
            available_depth_backend_ids=["da3", "ensemble"],
        )
        stage_map_valid = {s.name: s for s in report_valid.stages}
        assert stage_map_valid["depth"].backend_available is True

        report_invalid = validate_preset(
            preset,
            available_depth_backend_ids=["ensemble"],
        )
        stage_map_invalid = {s.name: s for s in report_invalid.stages}
        assert stage_map_invalid["depth"].backend_available is False


class TestMaterialsGovernanceParity:
    """Preset health should report the same materials schema/family issues as runtime validation."""

    def test_missing_materials_preset_family_is_reported(self, tmp_path):
        preset = tmp_path / "material_pbr.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "PBR Material Generation (Stable)",
                    "tier": "stable",
                    "backend": {"type": "heuristic"},
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])
        family_issues = [i for i in report.issues if i.category == "preset_family"]
        assert len(family_issues) == 1
        assert "preset_family='materials_pbr'" in family_issues[0].message

    def test_incorrect_materials_preset_family_is_reported(self, tmp_path):
        preset = tmp_path / "material_pbr.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "PBR Material Generation (Stable)",
                    "tier": "stable",
                    "preset_family": "material-pbr",
                    "backend": {"type": "heuristic"},
                    "pbr": {"resolution": "match_input"},
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])
        family_issues = [i for i in report.issues if i.category == "preset_family"]
        assert len(family_issues) == 1
        assert "got 'material-pbr'" in family_issues[0].message

    def test_unknown_materials_schema_path_is_reported(self, tmp_path):
        preset = tmp_path / "bad_materials.yaml"
        preset.write_text(
            yaml.dump(
                {
                    "name": "Bad Materials Schema",
                    "tier": "experimental",
                    "preset_family": "materials_pbr",
                    "materials": {
                        "runtime": {
                            "backend": "nvdiffrec",
                        }
                    },
                }
            )
        )

        report = validate_preset(preset, available_depth_backend_ids=["da3"])
        schema_issues = [i for i in report.issues if i.category == "materials_schema"]
        assert len(schema_issues) == 1
        assert schema_issues[0].path == "materials.runtime.backend"


class TestHealthReportSerialization:
    """Health report should serialize to JSON."""

    def test_to_dict_includes_all_fields(self):
        report = PresetHealthReport(
            preset_path="/tmp/test.yaml",
            preset_name="test",
            issues=[
                HealthIssue(severity="error", category="placeholder", message="test", path="a.b"),
                HealthIssue(severity="warning", category="backend_id", message="warn", path="c"),
            ],
            stages=[
                StageStatus(name="depth", declared=True, backend="da3"),
            ],
        )
        d = report.to_dict()
        assert d["healthy"] is False
        assert d["error_count"] == 1
        assert d["warning_count"] == 1
        assert len(d["issues"]) == 2
        assert len(d["stages"]) == 1

    def test_save_writes_json(self, tmp_path):
        report = PresetHealthReport(
            preset_path="/tmp/test.yaml",
            preset_name="test",
        )
        out = tmp_path / "preset_health.json"
        report.save(out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["healthy"] is True
        assert data["preset_name"] == "test"

    def test_healthy_property(self):
        report = PresetHealthReport(preset_path="", preset_name="ok")
        assert report.healthy is True

        report.issues.append(HealthIssue(severity="warning", category="test", message="w"))
        assert report.healthy is True  # Warnings don't block

        report.issues.append(HealthIssue(severity="error", category="test", message="e"))
        assert report.healthy is False  # Errors block


class TestUltraPresetValidation:
    """Integration test: validate the actual Ultra preset file."""

    def test_ultra_preset_da3_model_id_is_valid(self):
        """After §1.1 fix, da3 model name should be 'da3' (not the old variant name)."""
        preset_path = Path("config/presets/experimental/apex_research_ultra.yaml")
        if not preset_path.exists():
            pytest.skip("Ultra preset file not found")

        with open(preset_path) as f:
            data = yaml.safe_load(f)

        depth_models = data.get("depth", {}).get("models", [])
        da3_models = [m for m in depth_models if "da3" in m.get("name", "").lower()]
        assert len(da3_models) >= 1, "Expected at least one DA3 model in ultra preset"

        for model in da3_models:
            assert model["name"] == "da3", f"DA3 model name should be 'da3' (stable registry key), " f"got '{model['name']}'"

    def test_ultra_preset_has_placeholders(self):
        """Ultra preset currently has known placeholders; verify they're detected."""
        preset_path = Path("config/presets/experimental/apex_research_ultra.yaml")
        if not preset_path.exists():
            pytest.skip("Ultra preset file not found")

        # Use registry-based validation
        report = validate_preset(
            preset_path,
            available_depth_backend_ids=["da3", "depth_pro", "ensemble", "depthcrafter", "synthetic"],
        )

        placeholder_issues = [i for i in report.issues if i.category == "placeholder"]
        # Ultra preset still has NEEDS_VERIFICATION and PLACEHOLDER strings
        assert len(placeholder_issues) >= 1, "Ultra preset should flag placeholders until revisions/hashes are pinned"

    def test_ultra_preset_depth_backend_ids_valid(self):
        """After §1.1 fix, all depth model names should be valid registry IDs."""
        preset_path = Path("config/presets/experimental/apex_research_ultra.yaml")
        if not preset_path.exists():
            pytest.skip("Ultra preset file not found")

        report = validate_preset(
            preset_path,
            available_depth_backend_ids=["da3", "depth_pro", "ensemble", "depthcrafter", "synthetic"],
        )

        backend_issues = [i for i in report.issues if i.category == "backend_id"]
        assert len(backend_issues) == 0, (
            f"All depth model names should be valid registry IDs. " f"Issues: {[i.message for i in backend_issues]}"
        )


class TestFileNotFound:
    """Validation of missing preset files."""

    def test_missing_file_returns_unhealthy_report(self, tmp_path):
        report = validate_preset(tmp_path / "nonexistent.yaml")
        assert not report.healthy
        assert report.issues[0].category == "file_missing"


class TestShippedExperimentalPresetsPlaceholderInventory:
    """Regression: only known-flagged experimental files may carry PENDING_VERIFICATION.

    The audit-sandbox network policy blocked upstream SHA256/commit verification
    for several presets, so those entries were migrated from the bogus
    ``NEEDS_VERIFICATION_0000…`` / ``PLACEHOLDER_UPDATE_WHEN_INTEGRATED`` markers
    to explicit ``PENDING_VERIFICATION`` (consistent with model_lock_manifest).
    This test pins which files are still allowed to carry placeholders, so the
    list shrinks (not grows) as verifications get unblocked.

    Both checks parse YAML and walk string *values* so that comments mentioning
    the sentinel strings don't produce false positives.
    """

    # Files explicitly allowed to ship with PENDING_VERIFICATION markers until
    # the corresponding upstream artifact is fetched and pinned.
    _ALLOWED_PLACEHOLDER_FILES = {
        "apex_research_ultra.yaml",  # SAM2 + DepthCrafter + MaterialGAN
    }

    @staticmethod
    def _iter_string_values(node):
        """Yield every string scalar reached by walking a parsed YAML tree."""
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from TestShippedExperimentalPresetsPlaceholderInventory._iter_string_values(v)
        elif isinstance(node, list):
            for v in node:
                yield from TestShippedExperimentalPresetsPlaceholderInventory._iter_string_values(v)

    def test_experimental_presets_have_no_legacy_placeholder_strings(self):
        """No experimental preset may still carry the legacy NEEDS_VERIFICATION_0000…
        or PLACEHOLDER_UPDATE_WHEN_INTEGRATED markers in any string *value* — they
        were migrated to the unified PENDING_VERIFICATION sentinel. Comments are
        ignored (YAML safe_load discards them)."""
        repo_root = Path(__file__).resolve().parents[1]
        experimental_dir = repo_root / "config" / "presets" / "experimental"
        legacy_markers = ("NEEDS_VERIFICATION_0", "PLACEHOLDER_UPDATE_WHEN_INTEGRATED")

        offenders: list[tuple[str, str]] = []
        for preset_path in sorted(experimental_dir.glob("*.yaml")):
            data = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
            for value in self._iter_string_values(data):
                for marker in legacy_markers:
                    if marker in value:
                        offenders.append((preset_path.name, marker))
                        break

        assert not offenders, (
            f"Legacy placeholder markers still present in experimental presets: {offenders}. "
            f"Migrate them to PENDING_VERIFICATION (matching model_lock_manifest)."
        )

    def test_only_allowlisted_presets_carry_pending_verification(self):
        """The set of presets carrying PENDING_VERIFICATION in their parsed YAML
        *values* (not in comments) must not grow without an explicit allowlist update."""
        repo_root = Path(__file__).resolve().parents[1]
        scan_dirs = [
            repo_root / "config" / "presets",
            repo_root / "config" / "presets" / "experimental",
        ]
        offenders: list[str] = []
        for d in scan_dirs:
            if not d.exists():
                continue
            for preset_path in sorted(d.glob("*.yaml")):
                data = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
                has_sentinel = any(value.strip().upper() == "PENDING_VERIFICATION" for value in self._iter_string_values(data))
                if has_sentinel and preset_path.name not in self._ALLOWED_PLACEHOLDER_FILES:
                    offenders.append(preset_path.name)

        assert not offenders, (
            f"Unexpected presets carrying PENDING_VERIFICATION: {offenders}. "
            f"Either pin the real value or add to _ALLOWED_PLACEHOLDER_FILES."
        )


# Pytest markers
pytestmark = [
    pytest.mark.unit,
]
