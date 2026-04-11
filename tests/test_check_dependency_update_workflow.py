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
    return f"""
    - name: Install lock generation tools
      run: |
{required_install_snippets}
    - name: Update dependencies
      run: |
{required_workflow_snippets}
    - name: Check for vulnerabilities
      run: |
        audit_targets=(
          {required_targets}
        )
    - name: Create Pull Request
      with:
        body: |
          {required_pr_refs}
          {required_pr_snippets}
          Confirm target-owned ML lock contracts
    """


def remove_from_pr_body(text: str, needle: str) -> str:
    before_body, body = text.split("        body: |\n", maxsplit=1)
    body_without_ref = body.replace(f"          {needle}\n", "", 1)
    return f"{before_body}        body: |\n{body_without_ref}"


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
    body_with_ref = body + f"          {line}\n"
    return f"{before_body}        body: |\n{body_with_ref}"


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
    broken = remove_workflow_snippet(valid_workflow_text(), "make update-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert (
        "dependency-update workflow must include snippet " "'make update-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11'"
    ) in errors


def test_missing_required_install_toolchain_snippet_is_reported() -> None:
    broken = remove_workflow_snippet(valid_workflow_text(), 'python -m pip install --upgrade "pip<26"')
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert (
        "dependency-update workflow must include install-tool snippet 'python -m pip install --upgrade \"pip<26\"'" in errors
    )


def test_forbidden_target_agnostic_update_command_is_reported() -> None:
    broken = valid_workflow_text() + "\n        make update LOCK_PYTHON_VERSION=3.11\n"
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must not include snippet 'make update LOCK_PYTHON_VERSION=3.11'" in errors


def test_missing_audit_targets_block_is_reported_independently_of_pr_body_references() -> None:
    broken = remove_audit_targets_block(valid_workflow_text())
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must define an audit_targets block" in errors
    assert ("dependency-update PR body must reference checked-in contract file " "'requirements/security.txt'") not in errors
