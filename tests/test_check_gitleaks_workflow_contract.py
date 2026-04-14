import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_gitleaks_workflow_contract.py"
SPEC = importlib.util.spec_from_file_location("check_gitleaks_workflow_contract", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
gitleaks_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gitleaks_contract)


def valid_config_text() -> str:
    return f"""
title = "Transformation Portal gitleaks config"

[extend]
useDefault = true

[[rules]]
id = "{gitleaks_contract.EXPECTED_RULE_ID}"

[[rules.allowlists]]
description = "Ignore the generated portal auth failure branch false positive"
condition = "AND"
regexTarget = "secret"
regexes = ['''{gitleaks_contract.EXPECTED_SECRET_REGEX}''']
paths = ['''{gitleaks_contract.EXPECTED_PATH_REGEX}''']
"""


def valid_ci_workflow_text() -> str:
    return """
      - name: Check for secrets with gitleaks
        uses: gitleaks/gitleaks-action@sha
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_CONFIG: .gitleaks.toml
"""


def valid_firewall_workflow_text() -> str:
    return """
          "${RUNNER_TEMP}/gitleaks-bin/gitleaks" detect --config .gitleaks.toml --source . --verbose --no-git --exit-code 1 || {
            echo "⚠️  Secrets detected - review gitleaks output above"
            exit 1
          }
"""


def test_repo_gitleaks_workflow_contract_passes() -> None:
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=(PROJECT_ROOT / ".gitleaks.toml").read_text(encoding="utf-8"),
        ci_workflow_text=(PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        firewall_workflow_text=(PROJECT_ROOT / ".github" / "workflows" / "ci-quality-firewall.yml").read_text(
            encoding="utf-8"
        ),
    )
    assert errors == []


def test_missing_ci_config_reference_is_reported() -> None:
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=valid_config_text(),
        ci_workflow_text=valid_ci_workflow_text().replace("          GITLEAKS_CONFIG: .gitleaks.toml\n", "", 1),
        firewall_workflow_text=valid_firewall_workflow_text(),
    )
    assert "ci workflow must set GITLEAKS_CONFIG to .gitleaks.toml" in errors


def test_missing_firewall_config_reference_is_reported() -> None:
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=valid_config_text(),
        ci_workflow_text=valid_ci_workflow_text(),
        firewall_workflow_text=valid_firewall_workflow_text().replace("--config .gitleaks.toml ", "", 1),
    )
    assert "ci-quality-firewall workflow must pass --config .gitleaks.toml to gitleaks detect" in errors


def test_global_portal_allowlist_is_reported() -> None:
    config_text = valid_config_text() + """
[[allowlists]]
description = "too broad"
paths = ['''^public/portal-assets/''']
"""
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=config_text,
        ci_workflow_text=valid_ci_workflow_text(),
        firewall_workflow_text=valid_firewall_workflow_text(),
    )
    assert "gitleaks config must not define a global allowlist for portal assets" in errors


def test_broad_rule_path_allowlist_is_reported() -> None:
    config_text = valid_config_text().replace(
        f"paths = ['''{gitleaks_contract.EXPECTED_PATH_REGEX}''']",
        "paths = ['''^public/portal-assets/''']",
        1,
    )
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=config_text,
        ci_workflow_text=valid_ci_workflow_text(),
        firewall_workflow_text=valid_firewall_workflow_text(),
    )
    assert "generic-api-key allowlist must be scoped only to public/portal-assets/portal.js" in errors


def test_wrong_secret_regex_is_reported() -> None:
    config_text = valid_config_text().replace(
        f"regexes = ['''{gitleaks_contract.EXPECTED_SECRET_REGEX}''']",
        "regexes = ['''auth_failure''']",
        1,
    )
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=config_text,
        ci_workflow_text=valid_ci_workflow_text(),
        firewall_workflow_text=valid_firewall_workflow_text(),
    )
    assert "generic-api-key allowlist must match only the generated auth failure false positive" in errors
