from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validate_ci_config.py"
BUILD_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build.yml"
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
FIREWALL_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci-quality-firewall.yml"
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
    return _mutated_workflow(BUILD_WORKFLOW_PATH, tmp_path, old, new)


def _mutated_workflow(source_path: Path, tmp_path: Path, old: str, new: str) -> Path:
    source = source_path.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / source_path.name
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


def test_mypy_config_remains_root_linting_config() -> None:
    mypy_config = PROJECT_ROOT / "mypy.ini"
    assert mypy_config.exists()
    assert not (PROJECT_ROOT / "src" / "mypy.ini").exists()

    mypy_config_text = mypy_config.read_text(encoding="utf-8")
    for required_option in (
        "show_error_codes = True",
        "warn_redundant_casts = True",
        "no_implicit_optional = True",
        "strict_equality = True",
    ):
        assert required_option in mypy_config_text

    organization_doc = (PROJECT_ROOT / "docs" / "governance" / "REPO_ORGANIZATION.md").read_text(encoding="utf-8")
    assert "- **Testing and linting configuration**: `pyproject.toml`, `.pylintrc`, `mypy.ini`" in organization_doc
    assert ".flake8" not in organization_doc


def test_mypy_policy_contract_passes_repo_workflows() -> None:
    for workflow_path in (BUILD_WORKFLOW_PATH, CI_WORKFLOW_PATH, FIREWALL_WORKFLOW_PATH):
        validator, config = _load_config(workflow_path)

        assert validator.validate_mypy_policy_contract(workflow_path, config) is True
        assert validator.errors == []


def test_mypy_policy_contract_rejects_workflow_whitelist_drift(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        CI_WORKFLOW_PATH,
        tmp_path,
        "            src/transformation_portal/api/ \\\n",
        "",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_mypy_policy_contract(workflow_path, config) is False
    assert any("mypy whitelist must match" in error for error in validator.errors)


def test_mypy_policy_contract_requires_api_runtime_dependency(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        CI_WORKFLOW_PATH,
        tmp_path,
        ' types-PyYAML "pydantic==2.13.3"',
        " types-PyYAML",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_mypy_policy_contract(workflow_path, config) is False
    assert any("API mypy whitelist requires 'pydantic==2.13.3'" in error for error in validator.errors)


def test_mypy_policy_contract_requires_root_mypy_ini_config(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        BUILD_WORKFLOW_PATH,
        tmp_path,
        "          mypy --config-file=mypy.ini \\\n",
        "          mypy --config-file=pyproject.toml \\\n",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_mypy_policy_contract(workflow_path, config) is False
    assert any("must use --config-file=mypy.ini" in error for error in validator.errors)


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
        "python scripts/ci/check_per_package_branch_coverage.py coverage.xml  \\\n            --dry-run || rc=$?",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("must enforce branch coverage without --dry-run" in error for error in validator.errors)


def test_build_workflow_core_leg_keeps_cold_zone_touched_file_evidence(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "python scripts/ci/check_cold_zone_touched_files.py coverage.xml --compare-ref origin/main || rc=$?",
        "echo cold-zone touched-file evidence removed",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("must retain cold-zone touched-file coverage evidence check" in error for error in validator.errors)


def test_build_workflow_core_leg_requires_touched_file_compare_ref(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "python scripts/ci/check_cold_zone_touched_files.py coverage.xml --compare-ref origin/main || rc=$?",
        "python scripts/ci/check_cold_zone_touched_files.py coverage.xml || rc=$?",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("must retain cold-zone touched-file coverage evidence check" in error for error in validator.errors)


def test_build_workflow_core_leg_accepts_equals_form_compare_ref(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "python scripts/ci/check_cold_zone_touched_files.py coverage.xml --compare-ref origin/main || rc=$?",
        "python scripts/ci/check_cold_zone_touched_files.py coverage.xml --compare-ref=origin/main || rc=$?",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is True
    assert validator.errors == []


def test_build_workflow_core_leg_fetches_cold_zone_compare_ref(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "git fetch --no-tags --depth=1 origin main:refs/remotes/origin/main || rc=$?",
        "echo origin main fetch removed",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("must fetch origin/main before cold-zone touched-file evidence" in error for error in validator.errors)


def test_build_workflow_coverage_upload_keeps_html_artifact_path(tmp_path: Path) -> None:
    workflow_path = _build_workflow_without_upload_path(tmp_path, "htmlcov/")
    validator, config = _load_config(workflow_path)

    assert validator.validate_build_coverage_contract(workflow_path, config) is False
    assert any("Coverage upload path must include 'htmlcov/'" in error for error in validator.errors)
