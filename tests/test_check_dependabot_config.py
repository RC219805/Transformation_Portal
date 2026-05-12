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
  - package-ecosystem: "npm"
    directory: "/"
    target-branch: "main"
    open-pull-requests-limit: 5
    schedule:
      interval: "weekly"
    groups:
      root-node-tooling:
        applies-to: "version-updates"
        patterns:
          - "*"
          - "@*/*"
        update-types:
          - "minor"
          - "patch"
  - package-ecosystem: "npm"
    directory: "/web/secure-landing"
    target-branch: "main"
    open-pull-requests-limit: 5
    schedule:
      interval: "weekly"
    groups:
      frontdoor-node:
        applies-to: "version-updates"
        patterns:
          - "*"
          - "@*/*"
        update-types:
          - "minor"
          - "patch"
  - package-ecosystem: "npm"
    directory: "/cloudflare/transformationportal-worker"
    target-branch: "main"
    open-pull-requests-limit: 5
    schedule:
      interval: "weekly"
    groups:
      cloudflare-worker-node:
        applies-to: "version-updates"
        patterns:
          - "*"
          - "@*/*"
        update-types:
          - "minor"
          - "patch"
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
        'package-ecosystem: "gomod"',
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert (
        "dependabot config contains unsupported update target ('gomod', '/'); "
        "expected only [('github-actions', '/'), ('npm', '/'), "
        "('npm', '/cloudflare/transformationportal-worker'), ('npm', '/web/secure-landing'), ('pip', '/')]"
    ) in errors


def test_missing_open_pr_limit_is_reported() -> None:
    broken = valid_dependabot_text().replace("    open-pull-requests-limit: 5\n", "", 1)
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert ("dependabot update ('pip', '/') must set open-pull-requests-limit to 5") in errors


def test_pip_exclude_paths_are_not_required_after_retired_manifests_are_removed() -> None:
    errors = dependabot_contract.validate_dependabot_config(valid_dependabot_text())
    assert not any("must exclude unsupported manifest" in error for error in errors)


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


def test_missing_npm_group_is_reported() -> None:
    broken = valid_dependabot_text().replace(
        """    groups:
      frontdoor-node:
        applies-to: "version-updates"
        patterns:
          - "*"
          - "@*/*"
        update-types:
          - "minor"
          - "patch"
""",
        "",
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot update ('npm', '/web/secure-landing') must define npm version-update groups" in errors


def test_npm_group_must_keep_major_updates_separate() -> None:
    broken = valid_dependabot_text().replace(
        """        update-types:
          - "minor"
          - "patch"
""",
        """        update-types:
          - "major"
          - "minor"
          - "patch"
""",
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot npm group 'root-node-tooling' must group only ['minor', 'patch'] updates" in errors
