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


def _build_workflow_without_upload_path(tmp_path: Path, upload_path: str) -> Path:
    lines = BUILD_WORKFLOW_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    filtered_lines = [line for line in lines if line.strip() != upload_path]
    assert len(filtered_lines) == len(lines) - 1

    path = tmp_path / "build.yml"
    path.write_text("".join(filtered_lines), encoding="utf-8")
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
    assert any("must scope '--no-cov' to the ML matrix leg" in error for error in validator.errors)


def test_build_workflow_ml_no_cov_must_be_scoped_to_ml_branch(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        'if [ "${{ matrix.test-type }}" = "ml" ]; then',
        "if true; then",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("must scope '--no-cov' to the ML matrix leg" in error for error in validator.errors)


def test_build_workflow_core_leg_keeps_xml_coverage_generation(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(tmp_path, "--cov-report=xml ", "")
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any(
        "Core test leg must retain coverage generation flags: '--cov-report=xml'" in error for error in validator.errors
    )


def test_build_workflow_core_leg_keeps_branch_coverage_enforcement(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "python scripts/ci/check_per_package_branch_coverage.py coverage.xml || rc=$?",
        "echo branch coverage enforcement removed",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("must retain branch coverage enforcement check" in error for error in validator.errors)


def test_build_workflow_core_leg_rejects_branch_coverage_dry_run(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "python scripts/ci/check_per_package_branch_coverage.py coverage.xml || rc=$?",
        "python scripts/ci/check_per_package_branch_coverage.py coverage.xml --dry-run || rc=$?",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("must enforce branch coverage without --dry-run" in error for error in validator.errors)


def test_build_workflow_core_leg_rejects_branch_coverage_dry_run_with_spacing(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "python scripts/ci/check_per_package_branch_coverage.py coverage.xml || rc=$?",
        "python scripts/ci/check_per_package_branch_coverage.py coverage.xml  \\\n" "            --dry-run || rc=$?",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("must enforce branch coverage without --dry-run" in error for error in validator.errors)


def test_build_workflow_coverage_upload_keeps_html_artifact_path(tmp_path: Path) -> None:
    workflow_path = _build_workflow_without_upload_path(tmp_path, "htmlcov/")
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("Coverage upload path must include 'htmlcov/'" in error for error in validator.errors)
