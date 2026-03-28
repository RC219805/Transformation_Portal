import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_dependabot_config.py"
SPEC = importlib.util.spec_from_file_location("check_dependabot_config", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
dependabot_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dependabot_contract)


def valid_dependabot_text() -> str:
    return """
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    target-branch: "main"
    open-pull-requests-limit: 5
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    target-branch: "main"
    open-pull-requests-limit: 5
    schedule:
      interval: "weekly"
"""


def test_valid_dependabot_config_passes() -> None:
    assert dependabot_contract.validate_dependabot_config(valid_dependabot_text()) == []


def test_missing_target_branch_is_reported() -> None:
    broken = valid_dependabot_text().replace('    target-branch: "main"\n', "", 1)
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot update ('pip', '/') must target branch 'main'" in errors


def test_unsupported_update_target_is_reported() -> None:
    broken = valid_dependabot_text().replace(
        'package-ecosystem: "github-actions"',
        'package-ecosystem: "npm"',
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert (
        "dependabot config contains unsupported update target ('npm', '/'); "
        "expected only [('github-actions', '/'), ('pip', '/')]"
    ) in errors


def test_missing_open_pr_limit_is_reported() -> None:
    broken = valid_dependabot_text().replace("    open-pull-requests-limit: 5\n", "", 1)
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert ("dependabot update ('pip', '/') must set open-pull-requests-limit to 5") in errors
