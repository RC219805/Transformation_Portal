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
    return (PROJECT_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")


def test_valid_dependabot_config_passes() -> None:
    assert dependabot_contract.validate_dependabot_config(valid_dependabot_text()) == []


def test_declared_codeql_action_group_has_live_repository_references() -> None:
    assert dependabot_contract.validate_repository_references(valid_dependabot_text(), PROJECT_ROOT) == []


def test_declared_codeql_action_group_is_rejected_when_action_family_is_absent(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "build.yml").write_text(
        "steps:\n  - uses: actions/checkout@0123456789abcdef\n",
        encoding="utf-8",
    )

    errors = dependabot_contract.validate_repository_references(valid_dependabot_text(), tmp_path)

    assert errors == [
        "dependabot group 'codeql-actions' is stale: no github/codeql-action/* uses remain under .github/workflows"
    ]


def test_codeql_text_inside_run_scalar_is_not_a_live_action_reference(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "decoy.yml").write_text(
        """jobs:
  decoy:
    runs-on: ubuntu-latest
    steps:
      - run: |
          echo 'uses: github/codeql-action/init@deadbeef'
""",
        encoding="utf-8",
    )

    errors = dependabot_contract.validate_repository_references(valid_dependabot_text(), tmp_path)

    assert errors == [
        "dependabot group 'codeql-actions' is stale: no github/codeql-action/* uses remain under .github/workflows"
    ]


def test_quoted_codeql_step_uses_is_a_live_action_reference(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "codeql.yml").write_text(
        """jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: "github/codeql-action/init@deadbeef"
""",
        encoding="utf-8",
    )

    assert dependabot_contract.validate_repository_references(valid_dependabot_text(), tmp_path) == []


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


def test_missing_labels_are_reported() -> None:
    broken = valid_dependabot_text().replace(
        """    labels:
      - "dependencies"
      - "automated"
""",
        "",
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot update ('pip', '/') must define labels as a list" in errors


def test_staggered_schedule_drift_is_reported() -> None:
    broken = valid_dependabot_text().replace('      time: "10:45"', '      time: "10:30"', 1)
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot update ('npm', '/web/secure-landing') must set schedule 'time' to '10:45'" in errors


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
    assert "updates[0] directories must be a non-empty list of strings" in errors


def test_frontdoor_target_branch_override_is_rejected() -> None:
    broken = valid_dependabot_text().replace(
        '    directory: "/web/secure-landing"\n',
        '    directory: "/web/secure-landing"\n    target-branch: "main"\n',
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot frontdoor update must omit target-branch so security-update grouping applies" in errors


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
        "      frontdoor-node:\n",
        "      missing-frontdoor-node:\n",
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot update ('npm', '/web/secure-landing') must define npm group 'frontdoor-node'" in errors


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
    assert "dependabot group 'worker-node-tooling' must group only ['minor', 'patch'] updates" in errors


def test_wrangler_group_must_include_worker_types() -> None:
    broken = valid_dependabot_text().replace(
        '          - "@cloudflare/workers-types"\n',
        "",
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert ("dependabot group 'wrangler-sync' must atomically match " "['@cloudflare/workers-types', 'wrangler']") in errors


def test_wrangler_group_must_not_split_updates_by_dependency_name() -> None:
    broken = valid_dependabot_text().replace(
        '          - "@cloudflare/workers-types"\n',
        '          - "@cloudflare/workers-types"\n        group-by: "dependency-name"\n',
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot group 'wrangler-sync' must omit group-by so Wrangler and Worker types stay coupled" in errors


def test_codeql_actions_must_be_grouped() -> None:
    broken = valid_dependabot_text().replace("      codeql-actions:\n", "      split-codeql-actions:\n", 1)
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot github-actions update must define group 'codeql-actions'" in errors


def test_redis_major_updates_remain_ignored() -> None:
    broken = valid_dependabot_text().replace(
        '          - "version-update:semver-major"',
        '          - "version-update:semver-minor"',
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert ("dependabot pip ignore 'redis' must use update-types " "['version-update:semver-major']") in errors


def test_transformers_minor_updates_remain_ignored() -> None:
    broken = valid_dependabot_text().replace(
        '          - "version-update:semver-minor"',
        '          - "version-update:semver-major"',
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert ("dependabot pip ignore 'transformers' must use update-types " "['version-update:semver-minor']") in errors


def test_frontdoor_security_updates_must_be_grouped() -> None:
    broken = valid_dependabot_text().replace(
        "      frontdoor-security:\n",
        "      split-frontdoor-security:\n",
        1,
    )
    errors = dependabot_contract.validate_dependabot_config(broken)
    assert "dependabot frontdoor update must define group 'frontdoor-security'" in errors
