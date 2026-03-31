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


def test_single_python_file_path_is_scanned(tmp_path: Path):
    module = _load_module()
    bad_file = tmp_path / "single_loader.py"
    bad_file.write_text(
        dedent("""
            import yaml as y

            def load_payload(path):
                with open(path) as handle:
                    return y.safe_load(handle)
            """).strip() + "\n",
        encoding="utf-8",
    )

    violations = module.find_violations([bad_file])
    assert len(violations) == 1
    assert "single_loader.py" in violations[0]


def test_direct_safe_load_import_is_rejected(tmp_path: Path):
    module = _load_module()
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    bad_file = source_dir / "direct_loader.py"
    bad_file.write_text(
        dedent("""
            from yaml import safe_load

            def load_payload(path):
                with open(path) as handle:
                    return safe_load(handle)
            """).strip() + "\n",
        encoding="utf-8",
    )

    violations = module.find_violations([source_dir])
    assert len(violations) == 1
    assert "direct_loader.py" in violations[0]


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
