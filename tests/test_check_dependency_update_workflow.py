import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_dependency_update_workflow.py"
SPEC = importlib.util.spec_from_file_location("check_dependency_update_workflow", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow_contract)


def valid_workflow_text() -> str:
    required_targets = "\n".join(workflow_contract.REQUIRED_AUDIT_TARGETS)
    required_pr_refs = "\n".join(f"          {ref}" for ref in workflow_contract.REQUIRED_PR_BODY_REFERENCES)
    required_pr_snippets = "\n".join(f"          {snippet}" for snippet in workflow_contract.REQUIRED_PR_BODY_SNIPPETS)
    required_workflow_snippets = "\n".join(f"        {snippet}" for snippet in workflow_contract.REQUIRED_WORKFLOW_SNIPPETS)
    required_install_snippets = "\n".join(
        f"        {snippet}" for snippet in workflow_contract.REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS
    )
    required_audit_report_snippets = "\n".join(
        f"        {snippet}" for snippet in workflow_contract.REQUIRED_AUDIT_REPORT_SNIPPETS[:-1]
    )
    required_upload_path_snippet = workflow_contract.REQUIRED_AUDIT_REPORT_SNIPPETS[-1]
    return f"""
    - name: Install lock generation tools
      run: |
{required_install_snippets}
    - name: Update dependencies
      env:
        PIP_NO_CACHE_DIR: "1"
      run: |
{required_workflow_snippets}
    - name: Check for vulnerabilities
      run: |
{required_audit_report_snippets}
        audit_targets=(
          {required_targets}
        )
    - name: Create Pull Request
      with:
        body: |
          {required_pr_refs}
          {required_pr_snippets}
          Confirm target-owned ML lock contracts
    - name: Upload pip-audit report
      with:
        {required_upload_path_snippet}
    """


def remove_from_pr_body(text: str, needle: str) -> str:
    before_body, body = text.split("        body: |\n", maxsplit=1)
    body_content, after_body = body.split("    - name: Upload pip-audit report\n", maxsplit=1)
    body_without_ref = body_content.replace(f"          {needle}\n", "", 1)
    return f"{before_body}        body: |\n{body_without_ref}    - name: Upload pip-audit report\n{after_body}"


def remove_from_audit_targets(text: str, needle: str) -> str:
    before_block, rest = text.split("        audit_targets=(\n", maxsplit=1)
    audit_targets_block, after_block = rest.split("        )\n", maxsplit=1)
    audit_targets_block = audit_targets_block.replace(f"{needle}\n", "", 1)
    return f"{before_block}        audit_targets=(\n{audit_targets_block}        )\n{after_block}"


def remove_audit_targets_block(text: str) -> str:
    before_block, rest = text.split("        audit_targets=(\n", maxsplit=1)
    _, after_block = rest.split("        )\n", maxsplit=1)
    return f"{before_block}{after_block}"


def add_to_pr_body(text: str, line: str) -> str:
    before_body, body = text.split("        body: |\n", maxsplit=1)
    body_content, after_body = body.split("    - name: Upload pip-audit report\n", maxsplit=1)
    body_with_ref = body_content + f"          {line}\n"
    return f"{before_body}        body: |\n{body_with_ref}    - name: Upload pip-audit report\n{after_body}"


def remove_workflow_snippet(text: str, snippet: str) -> str:
    return text.replace(f"        {snippet}\n", "", 1)


def test_valid_dependency_update_workflow_contract_passes() -> None:
    assert workflow_contract.validate_dependency_update_workflow(valid_workflow_text()) == []


def test_missing_required_audit_target_is_reported() -> None:
    broken = remove_from_audit_targets(valid_workflow_text(), "requirements/security.txt")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update workflow must audit governed lockfile target " "'requirements/security.txt'") in errors


def test_stale_noncontract_ml_reference_is_reported() -> None:
    broken = add_to_pr_body(valid_workflow_text(), "requirements/ml.txt")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update workflow still references non-contract ML lockfile " "'requirements/ml.txt'") in errors


def test_missing_required_pr_body_reference_is_reported() -> None:
    broken = remove_from_pr_body(valid_workflow_text(), "requirements/security.txt")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update PR body must reference checked-in contract file " "'requirements/security.txt'") in errors


def test_missing_required_workflow_snippet_is_reported() -> None:
    broken = remove_workflow_snippet(valid_workflow_text(), "make update-generic LOCK_PYTHON_VERSION=3.11")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update workflow must include snippet " "'make update-generic LOCK_PYTHON_VERSION=3.11'") in errors


def test_missing_fresh_index_contract_is_reported() -> None:
    broken = remove_workflow_snippet(valid_workflow_text(), 'PIP_NO_CACHE_DIR: "1"')
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update Update dependencies step must set env.PIP_NO_CACHE_DIR to '1'" in errors


def test_fresh_index_setting_in_run_script_is_reported() -> None:
    broken = valid_workflow_text().replace(
        '      env:\n        PIP_NO_CACHE_DIR: "1"\n      run: |',
        '      run: |\n        PIP_NO_CACHE_DIR: "1"',
        1,
    )
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update Update dependencies step must set env.PIP_NO_CACHE_DIR to '1'" in errors


def test_fresh_index_setting_on_wrong_step_is_reported() -> None:
    broken = valid_workflow_text().replace('      env:\n        PIP_NO_CACHE_DIR: "1"\n', "", 1)
    broken = broken.replace(
        "    - name: Install lock generation tools\n      run: |",
        '    - name: Install lock generation tools\n      env:\n        PIP_NO_CACHE_DIR: "1"\n      run: |',
        1,
    )
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update Update dependencies step must set env.PIP_NO_CACHE_DIR to '1'" in errors


def test_missing_required_install_toolchain_snippet_is_reported() -> None:
    broken = remove_workflow_snippet(valid_workflow_text(), 'python -m pip install --upgrade "pip==26.2.1"')
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert (
        "dependency-update workflow must include install-tool snippet 'python -m pip install --upgrade \"pip==26.2.1\"'"
        in errors
    )


def test_missing_required_audit_report_temp_dir_snippet_is_reported() -> None:
    broken = remove_workflow_snippet(
        valid_workflow_text(), 'audit_reports_dir="${{ runner.temp }}/dependency-update-audit-reports"'
    )
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert (
        "dependency-update workflow must include audit-report snippet "
        "'audit_reports_dir=\"${{ runner.temp }}/dependency-update-audit-reports\"'"
    ) in errors


def test_missing_required_audit_report_upload_path_is_reported() -> None:
    broken = remove_workflow_snippet(valid_workflow_text(), "path: ${{ runner.temp }}/dependency-update-audit-reports/")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert (
        "dependency-update workflow must include audit-report snippet "
        "'path: ${{ runner.temp }}/dependency-update-audit-reports/'"
    ) in errors


def test_forbidden_target_agnostic_update_command_is_reported() -> None:
    broken = valid_workflow_text() + "\n        make update LOCK_PYTHON_VERSION=3.11\n"
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must not include snippet 'make update LOCK_PYTHON_VERSION=3.11'" in errors


def test_forbidden_linux_target_owned_update_command_is_reported() -> None:
    broken = valid_workflow_text() + "\n        make update-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11\n"
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must not include snippet 'make update-ml-linux-x86_64'" in errors


def test_repo_local_audit_reports_usage_is_reported() -> None:
    broken = valid_workflow_text() + "\n        mkdir -p audit-reports\n        path: audit-reports/\n"
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must not include snippet 'mkdir -p audit-reports'" in errors
    assert "dependency-update workflow must not include snippet 'path: audit-reports/'" in errors


def test_retired_pygments_exception_is_reported() -> None:
    broken = (
        valid_workflow_text()
        + "\n        # CVE-2026-4539 (pygments): No fix available yet - temporary exception\n"
        + "        pip-audit --ignore-vuln CVE-2026-4539 -r requirements/security.txt\n"
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update workflow must not include snippet '--ignore-vuln CVE-2026-4539'" in errors
    assert (
        "dependency-update workflow must not include snippet " "'CVE-2026-4539 (pygments): No fix available yet'"
    ) in errors


def test_missing_audit_targets_block_is_reported_independently_of_pr_body_references() -> None:
    broken = remove_audit_targets_block(valid_workflow_text())
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must define an audit_targets block" in errors
    assert ("dependency-update PR body must reference checked-in contract file " "'requirements/security.txt'") not in errors
