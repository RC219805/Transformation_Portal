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
    exclude-paths:
      - "requirements/ml-core-linux.*"
      - "requirements/ml-core-darwin-x86_64.*"
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


def test_missing_required_exclude_path_is_reported() -> None:
    broken = valid_dependabot_text().replace('      - "requirements/ml-core-linux.*"\n', "", 1)
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot update ('pip', '/') must exclude unsupported manifest 'requirements/ml-core-linux.*'" in errors


def test_invalid_yaml_is_reported() -> None:
    errors = dependabot_contract.validate_dependabot_config("updates: [")
    assert len(errors) == 1
    assert errors[0].startswith("invalid YAML:")


def test_non_mapping_yaml_root_is_reported() -> None:
    errors = dependabot_contract.validate_dependabot_config("- package-ecosystem: pip")
    assert errors == ["dependabot config must be a YAML mapping"]


def test_missing_package_ecosystem_is_reported() -> None:
    broken = """
version: 2
updates:
  - directory: "/"
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
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "updates[0] package-ecosystem must be a non-empty string" in errors


def test_missing_directory_is_reported() -> None:
    broken = """
version: 2
updates:
  - package-ecosystem: "pip"
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
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "updates[0] directory must be a non-empty string" in errors


def test_duplicate_update_target_is_reported() -> None:
    broken = """
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    target-branch: "main"
    open-pull-requests-limit: 5
    schedule:
      interval: "weekly"
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
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot config contains duplicate update target ('pip', '/')" in errors
