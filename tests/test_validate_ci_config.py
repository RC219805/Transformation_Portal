from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validate_ci_config.py"
BUILD_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build.yml"
SPEC = importlib.util.spec_from_file_location("validate_ci_config", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
validate_ci_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_ci_config)


def _load_config(path: Path) -> tuple[object, dict]:
    validator = validate_ci_config.CIValidator(PROJECT_ROOT)
    config = validator.validate_yaml_syntax(path)
    assert config is not None
    return validator, config


def _mutated_build_workflow(tmp_path: Path, old: str, new: str) -> Path:
    source = BUILD_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / "build.yml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return path


def test_build_workflow_coverage_contract_passes_repo_config() -> None:
    validator, config = _load_config(BUILD_WORKFLOW_PATH)

    assert validator.validate_build_coverage_contract(BUILD_WORKFLOW_PATH, config) is True
    assert validator.errors == []


def test_build_workflow_coverage_upload_must_be_core_only(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "if: always() && matrix.test-type == 'core'",
        "if: always()",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("Coverage upload step must be guarded" in error for error in validator.errors)


def test_build_workflow_ml_leg_keeps_no_cov_fast_path(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(tmp_path, 'COV_FLAGS="--no-cov"', 'COV_FLAGS=""')
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any('ML test leg must keep COV_FLAGS="--no-cov"' in error for error in validator.errors)


def test_build_workflow_coverage_upload_keeps_html_artifact_path(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(tmp_path, "            htmlcov/\n", "")
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("Coverage upload path must include 'htmlcov/'" in error for error in validator.errors)
