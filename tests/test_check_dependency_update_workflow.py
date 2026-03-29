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
    return f"""
    - name: Check for vulnerabilities
      run: |
        audit_targets=(
          {required_targets}
        )
    - name: Create Pull Request
      with:
        body: |
          {required_pr_refs}
          Confirm platform ML core contracts
    """


def remove_from_pr_body(text: str, needle: str) -> str:
    before_body, body = text.split("        body: |\n", maxsplit=1)
    body_without_ref = body.replace(f"          {needle}\n", "", 1)
    return f"{before_body}        body: |\n{body_without_ref}"


def remove_from_audit_targets(text: str, needle: str) -> str:
    return text.replace(f"{needle}\n", "", 1)


def add_to_pr_body(text: str, line: str) -> str:
    before_body, body = text.split("        body: |\n", maxsplit=1)
    body_with_ref = body + f"          {line}\n"
    return f"{before_body}        body: |\n{body_with_ref}"


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


def test_missing_audit_target_is_reported_even_if_pr_body_still_mentions_it() -> None:
    broken = remove_from_audit_targets(valid_workflow_text(), "requirements/security.txt")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update workflow must audit governed lockfile target " "'requirements/security.txt'") in errors
