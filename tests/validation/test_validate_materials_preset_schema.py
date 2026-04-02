"""Tests for materials preset schema-location validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_materials_preset_schema.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_materials_preset_schema", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_material_presets_use_known_backend_paths():
    module = _load_module()

    assert module.check_preset(PROJECT_ROOT / "config/presets/experimental/material_pbr.yaml") == []
    assert module.check_preset(PROJECT_ROOT / "config/presets/material_pbr_canary.yaml") == []
    assert module.check_preset(PROJECT_ROOT / "config/presets/experimental/apex_research_ultra.yaml") == []


def test_unknown_material_backend_schema_path_is_rejected(tmp_path: Path):
    module = _load_module()
    preset_path = tmp_path / "bad_materials.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "name": "Bad Materials Schema",
                "tier": "experimental",
                "materials": {
                    "runtime": {
                        "backend": "nvdiffrec",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    issues = module.check_preset(preset_path)
    assert len(issues) == 1
    assert "materials.runtime.backend" in issues[0]


def test_top_level_materials_preset_requires_explicit_family_marker(tmp_path: Path):
    module = _load_module()
    preset_path = tmp_path / "material_pbr.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "name": "PBR Material Generation (Stable)",
                "tier": "stable",
                "backend": {
                    "type": "heuristic",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    issues = module.check_preset(preset_path)
    assert len(issues) == 1
    assert "preset_family='materials_pbr'" in issues[0]


def test_incorrect_top_level_materials_preset_family_is_rejected(tmp_path: Path):
    module = _load_module()
    preset_path = tmp_path / "material_pbr.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "name": "PBR Material Generation (Stable)",
                "tier": "stable",
                "preset_family": "material-pbr",
                "backend": {
                    "type": "heuristic",
                },
                "pbr": {
                    "resolution": "match_input",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    issues = module.check_preset(preset_path)
    assert len(issues) == 1
    assert "preset_family='materials_pbr'" in issues[0]
    assert "got 'material-pbr'" in issues[0]


def test_non_materials_preset_is_ignored(tmp_path: Path):
    module = _load_module()
    preset_path = tmp_path / "reconstruction_only.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "name": "NVDIFFREC Reconstruction",
                "tier": "apex_research",
                "pipeline": {
                    "reconstruction": {
                        "backend": "nvdiffrec",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert module.check_preset(preset_path) == []
