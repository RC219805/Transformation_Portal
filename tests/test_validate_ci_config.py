from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validate_ci_config.py"
BUILD_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build.yml"
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
FIREWALL_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci-quality-firewall.yml"
DIAGNOSTIC_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "diagnostic-trial.yml"
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


FIREWALL_SENSITIVE_STEP_CASES = (
    pytest.param(
        {"uses": "actions/checkout@pinned", "with": {"ref": "${{ needs.preflight.outputs.head_sha }}"}},
        id="checkout",
    ),
    pytest.param(
        {"uses": "actions/setup-python@pinned", "with": {"cache": "pip"}},
        id="setup-python-cache",
    ),
    pytest.param(
        {"uses": "actions/setup-node@pinned", "with": {"cache": "npm"}},
        id="setup-node-cache",
    ),
    pytest.param({"uses": "actions/cache@pinned"}, id="cache"),
    pytest.param({"uses": "actions/cache/restore@pinned"}, id="cache-restore"),
    pytest.param({"uses": "actions/cache/save@pinned"}, id="cache-save"),
    pytest.param({"uses": "actions/upload-artifact@pinned"}, id="artifact-upload"),
    pytest.param({"uses": "codecov/codecov-action@pinned"}, id="codecov-upload"),
)


FIREWALL_NON_PRODUCER_STEP_CASES = (
    pytest.param(
        {"uses": "actions/setup-python@pinned", "with": {"python-version": "3.12"}},
        id="setup-python-without-cache",
    ),
    pytest.param(
        {"uses": "actions/setup-node@pinned", "with": {"node-version": "22"}},
        id="setup-node-without-cache",
    ),
    pytest.param({"uses": "actions/download-artifact@pinned"}, id="artifact-download"),
    pytest.param({"run": "echo report-only"}, id="run-step"),
)


def test_build_workflow_coverage_contract_passes_repo_config() -> None:
    validator, config = _load_config(BUILD_WORKFLOW_PATH)

    assert validator.validate_build_coverage_contract(BUILD_WORKFLOW_PATH, config) is True
    assert validator.errors == []


def test_build_workflow_ci_gate_contract_passes_repo_config() -> None:
    validator, config = _load_config(BUILD_WORKFLOW_PATH)

    assert validator.validate_ci_gate_contract(BUILD_WORKFLOW_PATH, config) is True
    assert validator.errors == []


def test_build_workflow_ci_gate_rejects_duplicate_check_publisher(tmp_path: Path) -> None:
    workflow_path = _mutated_build_workflow(
        tmp_path,
        "    permissions: {}\n\n    steps:",
        """    permissions:
      checks: write

    steps:
      - name: Publish Dedicated CI Gate Check
        uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3
        with:
          script: |
            await github.rest.checks.create({name: 'CI Gate'});
""",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_ci_gate_contract(workflow_path, config) is False
    assert any("must not request token permissions" in error for error in validator.errors)
    assert any("Must not publish a duplicate dedicated CI Gate check" in error for error in validator.errors)
    assert any("Must not call github.rest.checks.create" in error for error in validator.errors)


def test_firewall_checkout_trust_contract_passes_repo_config() -> None:
    validator, config = _load_config(FIREWALL_WORKFLOW_PATH)

    assert validator.validate_firewall_checkout_trust_contract(FIREWALL_WORKFLOW_PATH, config) is True
    assert validator.errors == []


def test_firewall_upstream_workflow_identity_passes_repo_config() -> None:
    for workflow_path in (BUILD_WORKFLOW_PATH, FIREWALL_WORKFLOW_PATH, DIAGNOSTIC_WORKFLOW_PATH):
        validator, config = _load_config(workflow_path)

        assert validator.validate_firewall_upstream_workflow_identity(workflow_path, config) is True
        assert validator.errors == []


def test_firewall_upstream_workflow_identity_rejects_renamed_build(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        BUILD_WORKFLOW_PATH,
        tmp_path,
        "name: CI (Lint, Tests & Manifest)\n",
        "name: Alternate CI\n",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_upstream_workflow_identity(workflow_path, config) is False
    assert any("Workflow name must match the reserved firewall upstream identity" in error for error in validator.errors)


def test_firewall_upstream_workflow_identity_rejects_duplicate_name(tmp_path: Path) -> None:
    source = DIAGNOSTIC_WORKFLOW_PATH.read_text(encoding="utf-8")
    original_name = next(line for line in source.splitlines() if line.startswith("name: "))
    workflow_path = tmp_path / DIAGNOSTIC_WORKFLOW_PATH.name
    workflow_path.write_text(
        source.replace(original_name, "name: CI (Lint, Tests & Manifest)", 1),
        encoding="utf-8",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_upstream_workflow_identity(workflow_path, config) is False
    assert any(
        "Non-build workflow must not claim the reserved firewall upstream identity" in error for error in validator.errors
    )


def test_firewall_upstream_identity_diagnostics_redact_reserved_name(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    sensitive_name = "SENSITIVE_WORKFLOW_IDENTITY_SENTINEL"
    monkeypatch.setattr(
        validate_ci_config.CIValidator,
        "FIREWALL_TRUSTED_UPSTREAM_WORKFLOW_NAME",
        sensitive_name,
    )

    build_validator, build_config = _load_config(BUILD_WORKFLOW_PATH)
    assert build_validator.validate_firewall_upstream_workflow_identity(BUILD_WORKFLOW_PATH, build_config) is False
    assert sensitive_name not in "\n".join(build_validator.errors)

    non_build_validator, non_build_config = _load_config(DIAGNOSTIC_WORKFLOW_PATH)
    non_build_config["name"] = sensitive_name
    assert (
        non_build_validator.validate_firewall_upstream_workflow_identity(
            DIAGNOSTIC_WORKFLOW_PATH,
            non_build_config,
        )
        is False
    )
    assert sensitive_name not in "\n".join(non_build_validator.errors)

    workflow_path = tmp_path / "build.yml"
    workflow_path.write_text(BUILD_WORKFLOW_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["validate_ci_config.py", str(workflow_path)])

    with pytest.raises(SystemExit) as exc_info:
        validate_ci_config.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    rendered_output = captured.out + captured.err
    assert "build.yml: Workflow name must match the reserved firewall upstream identity" in rendered_output
    assert sensitive_name not in rendered_output


def test_firewall_checkout_trust_contract_rejects_cache_write_trigger(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        "    types: [completed]\n",
        "    types: [completed]\n  workflow_dispatch:\n",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("Must remain workflow_run-only" in error for error in validator.errors)


@pytest.mark.parametrize(
    ("old", "new", "expected_error"),
    (
        pytest.param(
            '    workflows: ["CI (Lint, Tests & Manifest)"]\n',
            '    workflows: ["Untrusted Alternate Workflow"]\n',
            "Upstream workflow must remain exactly CI (Lint, Tests & Manifest)",
            id="alternate-upstream-workflow",
        ),
        pytest.param(
            "    branches: [main, develop]\n",
            "    branches: [develop, main]\n",
            "Branch allowlist must remain the ordered main and develop list",
            id="reordered-branches",
        ),
        pytest.param(
            "    types: [completed]\n",
            "    types: [requested]\n",
            "Event type must remain exactly completed",
            id="pre-completion-event",
        ),
        pytest.param(
            "    types: [completed]\n",
            "    types: [completed]\n    secrets: inherit\n",
            "Trigger must define only workflows, branches, and types",
            id="unexpected-trigger-key",
        ),
    ),
)
def test_firewall_checkout_trust_contract_requires_exact_workflow_run_trigger(
    tmp_path: Path,
    old: str,
    new: str,
    expected_error: str,
) -> None:
    workflow_path = _mutated_workflow(FIREWALL_WORKFLOW_PATH, tmp_path, old, new)
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any(expected_error in error for error in validator.errors)


def test_firewall_checkout_trust_contract_rejects_mixed_trigger_ref_resolution(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        '          echo "head_sha=${{ github.event.workflow_run.head_sha }}" >> $GITHUB_OUTPUT\n',
        '          echo "head_sha=${{ github.sha }}" >> $GITHUB_OUTPUT\n',
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("resolve-ref must write only the trusted workflow_run" in error for error in validator.errors)
    assert any("must not mix direct-trigger and workflow_run refs" in error for error in validator.errors)


def test_firewall_checkout_trust_contract_rejects_duplicate_head_sha_assignment(tmp_path: Path) -> None:
    trusted_assignment = '          echo "head_sha=${{ github.event.workflow_run.head_sha }}" >> $GITHUB_OUTPUT\n'
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        trusted_assignment,
        trusted_assignment + '          echo "head_sha=${MALICIOUS_SHA}" >> $GITHUB_OUTPUT\n',
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("resolve-ref must write only the trusted workflow_run" in error for error in validator.errors)


def test_firewall_checkout_trust_contract_rejects_repository_override(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        "          ref: ${{ needs.preflight.outputs.head_sha }}\n",
        "          repository: attacker/untrusted\n" "          ref: ${{ needs.preflight.outputs.head_sha }}\n",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("Checkout must not override the trusted repository" in error for error in validator.errors)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        pytest.param(
            "      should_run: ${{ steps.check.outputs.should_run }}\n",
            "      should_run: true\n",
            id="literal-should-run",
        ),
        pytest.param(
            "      head_sha: ${{ steps.resolve-ref.outputs.head_sha }}\n",
            "      head_sha: ${{ github.sha }}\n",
            id="untrusted-head-sha",
        ),
        pytest.param(
            "      head_branch: ${{ steps.resolve-ref.outputs.head_branch }}\n",
            "      head_branch: ${{ github.ref_name }}\n",
            id="untrusted-head-branch",
        ),
    ),
)
def test_firewall_checkout_trust_contract_requires_exact_preflight_output_mappings(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    workflow_path = _mutated_workflow(FIREWALL_WORKFLOW_PATH, tmp_path, old, new)
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("Outputs must map exactly to the trusted check and resolve-ref steps" in error for error in validator.errors)


def test_firewall_checkout_trust_contract_rejects_alternate_output_write(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        '          echo "head_branch=${{ github.event.workflow_run.head_branch }}" >> $GITHUB_OUTPUT\n',
        '          echo "head_branch=${{ github.event.workflow_run.head_branch }}" >> $GITHUB_OUTPUT\n'
        '          printf \'%s=%s\\n\' head_sha "${MALICIOUS_SHA}" >> "$GITHUB_OUTPUT"\n',
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("resolve-ref must write only the trusted workflow_run" in error for error in validator.errors)


def test_firewall_checkout_trust_contract_rejects_obfuscated_extra_ref_write(tmp_path: Path) -> None:
    trusted_log = (
        '          echo "Testing commit: ${{ github.event.workflow_run.head_sha }} '
        '(branch: ${{ github.event.workflow_run.head_branch }})"\n'
    )
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        trusted_log,
        trusted_log
        + '          python3 -c \'import os; open(os.environ["GITHUB_"+"OUTPUT"], "a").write("head_sha=deadbeef\\\\n")\'\n',
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("resolve-ref must use only the trusted ref resolution script" in error for error in validator.errors)


def test_firewall_checkout_trust_contract_rejects_inverted_upstream_result_gate(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        '          if [ "$upstream_result" != "success" ]; then\n',
        '          if [ "$upstream_result" == "success" ]; then\n',
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("check must exactly gate should_run on a successful upstream result" in error for error in validator.errors)


@pytest.mark.parametrize(
    ("step_id", "replacement", "expected_error"),
    (
        pytest.param("check", "", "check must explicitly use the trusted bash shell", id="check-missing-shell"),
        pytest.param(
            "check",
            "        shell: /bin/true {0}\n",
            "check must explicitly use the trusted bash shell",
            id="check-custom-shell",
        ),
        pytest.param(
            "resolve-ref",
            "",
            "resolve-ref must explicitly use the trusted bash shell",
            id="resolve-missing-shell",
        ),
        pytest.param(
            "resolve-ref",
            "        shell: /bin/true {0}\n",
            "resolve-ref must explicitly use the trusted bash shell",
            id="resolve-custom-shell",
        ),
    ),
)
def test_firewall_checkout_trust_contract_requires_explicit_bash_shell(
    tmp_path: Path,
    step_id: str,
    replacement: str,
    expected_error: str,
) -> None:
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        f"        id: {step_id}\n        shell: bash\n",
        f"        id: {step_id}\n{replacement}",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any(expected_error in error for error in validator.errors)


@pytest.mark.parametrize(
    ("old", "new", "expected_error"),
    (
        pytest.param(
            'env:\n  PYTHON_VERSION_LINT: "3.12"\n',
            'defaults:\n  run:\n    shell: /bin/true {0}\n\nenv:\n  PYTHON_VERSION_LINT: "3.12"\n',
            "Must not define workflow run defaults",
            id="workflow-shell-default",
        ),
        pytest.param(
            'env:\n  PYTHON_VERSION_LINT: "3.12"\n',
            'env:\n  BASH_ENV: /tmp/untrusted-bash-env\n  PYTHON_VERSION_LINT: "3.12"\n',
            "Workflow environment must match",
            id="workflow-bash-env",
        ),
        pytest.param(
            "    runs-on: ubuntu-latest\n    outputs:\n",
            "    runs-on: ubuntu-latest\n    defaults:\n      run:\n        shell: /bin/true {0}\n    outputs:\n",
            "Must not define execution overrides: defaults",
            id="preflight-shell-default",
        ),
        pytest.param(
            "        shell: bash\n        run: |\n",
            "        shell: bash\n        env:\n          BASH_ENV: /tmp/untrusted-bash-env\n        run: |\n",
            "check must not define execution overrides: env",
            id="check-bash-env",
        ),
    ),
)
def test_firewall_checkout_trust_contract_rejects_execution_overrides(
    tmp_path: Path,
    old: str,
    new: str,
    expected_error: str,
) -> None:
    workflow_path = _mutated_workflow(FIREWALL_WORKFLOW_PATH, tmp_path, old, new)
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any(expected_error in error for error in validator.errors)


def test_firewall_checkout_trust_contract_rejects_extra_preflight_step(tmp_path: Path) -> None:
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        "    steps:\n      - name: Check upstream CI result\n",
        "    steps:\n"
        "      - name: Prepare untrusted shell environment\n"
        "        shell: bash\n"
        "        run: echo 'BASH_ENV=/tmp/untrusted' >> \"$GITHUB_ENV\"\n"
        "      - name: Check upstream CI result\n",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("Must contain only the ordered check and resolve-ref steps" in error for error in validator.errors)


@pytest.mark.parametrize(
    ("old", "new", "expected_error"),
    (
        pytest.param(
            "      github.event.workflow_run.head_repository.full_name == github.repository &&\n",
            "",
            "Must admit only same-repository push or manual upstream runs",
            id="wrong-repository",
        ),
        pytest.param(
            "        github.event.workflow_run.event == 'workflow_dispatch'\n",
            "        github.event.workflow_run.event == 'pull_request'\n",
            "Must admit only same-repository push or manual upstream runs",
            id="wrong-upstream-event",
        ),
        pytest.param(
            "    branches: [main, develop]\n",
            "    branches: [main, develop, feature]\n",
            "Branch allowlist must remain the ordered main and develop list",
            id="wrong-branch",
        ),
        pytest.param(
            "          ref: ${{ needs.preflight.outputs.head_sha }}\n",
            "          ref: ${{ github.sha }}\n",
            "Checkout must use the trusted preflight head_sha",
            id="wrong-checkout-ref",
        ),
        pytest.param(
            """  group: >-
    ci-firewall-${{ github.event.workflow_run.head_repository.id }}-${{
    github.event.workflow_run.event }}-${{ github.event.workflow_run.head_branch }}
""",
            "  group: ci-firewall-${{ github.event.workflow_run.head_branch }}\n",
            "Group must isolate upstream repository, event, and branch",
            id="wrong-concurrency-domain",
        ),
        pytest.param(
            "    needs: [preflight, test-core, test-ml]\n",
            "    needs: [test-core, test-ml]\n",
            "Checkout consumers must depend directly on preflight",
            id="missing-direct-preflight-need",
        ),
    ),
)
def test_firewall_checkout_trust_contract_rejects_trust_boundary_mutations(
    tmp_path: Path,
    old: str,
    new: str,
    expected_error: str,
) -> None:
    workflow_path = _mutated_workflow(FIREWALL_WORKFLOW_PATH, tmp_path, old, new)
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any(expected_error in error for error in validator.errors)


@pytest.mark.parametrize(
    "condition",
    (
        pytest.param("always()", id="missing-should-run"),
        pytest.param(
            "always() || needs.preflight.outputs.should_run == 'true'",
            id="bypassable-should-run",
        ),
    ),
)
def test_firewall_checkout_trust_contract_requires_non_bypassable_should_run_gate(
    tmp_path: Path,
    condition: str,
) -> None:
    workflow_path = _mutated_workflow(
        FIREWALL_WORKFLOW_PATH,
        tmp_path,
        "    if: needs.preflight.outputs.should_run == 'true'\n",
        f"    if: {condition}\n",
    )
    validator, config = _load_config(workflow_path)

    assert validator.validate_firewall_checkout_trust_contract(workflow_path, config) is False
    assert any("must require the trusted preflight should_run output" in error for error in validator.errors)


@pytest.mark.parametrize("sensitive_step", FIREWALL_SENSITIVE_STEP_CASES)
def test_firewall_checkout_trust_contract_enforces_provenance_for_all_sensitive_steps(sensitive_step: dict) -> None:
    validator, config = _load_config(FIREWALL_WORKFLOW_PATH)
    config["jobs"]["missing-direct-need"] = {
        "if": "needs.preflight.outputs.should_run == 'true'",
        "steps": [sensitive_step],
    }
    config["jobs"]["bypassable-producer"] = {
        "needs": "preflight",
        "if": "always()",
        "steps": [sensitive_step],
    }

    assert validator.validate_firewall_checkout_trust_contract(FIREWALL_WORKFLOW_PATH, config) is False
    assert any("must depend directly on preflight" in error for error in validator.errors)
    assert any("must require the trusted preflight should_run output" in error for error in validator.errors)


@pytest.mark.parametrize("non_producer_step", FIREWALL_NON_PRODUCER_STEP_CASES)
def test_firewall_checkout_trust_contract_ignores_non_producers(non_producer_step: dict) -> None:
    validator, config = _load_config(FIREWALL_WORKFLOW_PATH)
    config["jobs"]["report-only"] = {"steps": [non_producer_step]}

    assert validator.validate_firewall_checkout_trust_contract(FIREWALL_WORKFLOW_PATH, config) is True
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
