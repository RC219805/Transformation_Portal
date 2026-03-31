"""Tests for YAML governance boundary validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from textwrap import dedent
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_yaml_governance_boundary.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_yaml_governance_boundary", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_runtime_sources_pass_yaml_governance_boundary():
    module = _load_module()
    assert module.find_violations([PROJECT_ROOT / "src"]) == []


def test_unmarked_raw_safe_load_is_rejected(tmp_path: Path):
    module = _load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    bad_file = source_dir / "bad_loader.py"
    bad_file.write_text(
        dedent("""
            import yaml

            def load_payload(path):
                with open(path) as handle:
                    return yaml.safe_load(handle)
            """).strip() + "\n",
        encoding="utf-8",
    )

    violations = module.find_violations([source_dir])
    assert len(violations) == 1
    assert "bad_loader.py" in violations[0]


def test_exempt_raw_safe_load_is_allowed(tmp_path: Path):
    module = _load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    exempt_file = source_dir / "recipe_loader.py"
    exempt_file.write_text(
        dedent("""
            import yaml

            def load_recipe(path):
                with open(path) as handle:
                    # YAML_GOVERNANCE_EXEMPT: internal recipe loader.
                    return yaml.safe_load(handle)
            """).strip() + "\n",
        encoding="utf-8",
    )

    assert module.find_violations([source_dir]) == []


def test_authority_raw_safe_load_is_allowed(tmp_path: Path):
    module = _load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    authority_file = source_dir / "licensing_loader.py"
    authority_file.write_text(
        dedent("""
            import yaml

            def load_and_validate_preset(path):
                with open(path) as handle:
                    # YAML_GOVERNANCE_AUTHORITY: shared preset loader.
                    return yaml.safe_load(handle)
            """).strip() + "\n",
        encoding="utf-8",
    )

    assert module.find_violations([source_dir]) == []
