#!/usr/bin/env python3
"""Validate the shared gitleaks workflow contract."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".gitleaks.toml"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
FIREWALL_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci-quality-firewall.yml"

EXPECTED_RULE_ID = "generic-api-key"
EXPECTED_PATH_REGEX = r"(^|/)public/portal-assets/portal\.js$"
EXPECTED_SECRET_REGEX = r'^(uth_failure"\|\|normalizedReason==="a|normalizedReason===)$'
EXPECTED_CI_SNIPPET = "GITLEAKS_CONFIG: .gitleaks.toml"
EXPECTED_FIREWALL_SNIPPET = "detect --config .gitleaks.toml --source . --verbose --no-git --exit-code 1"


def validate_gitleaks_contract(
    config_text: str,
    ci_workflow_text: str,
    firewall_workflow_text: str,
) -> list[str]:
    """Return contract violations for the shared gitleaks setup."""
    errors: list[str] = []

    try:
        config = tomllib.loads(config_text)
    except tomllib.TOMLDecodeError as exc:
        return [f"gitleaks config is not valid TOML: {exc}"]

    errors.extend(_validate_config(config))

    if EXPECTED_CI_SNIPPET not in ci_workflow_text:
        errors.append("ci workflow must set GITLEAKS_CONFIG to .gitleaks.toml")

    if EXPECTED_FIREWALL_SNIPPET not in firewall_workflow_text:
        errors.append("ci-quality-firewall workflow must pass --config .gitleaks.toml to gitleaks detect")

    return errors


def _validate_config(config: dict[str, object]) -> list[str]:
    errors: list[str] = []

    extend = config.get("extend")
    if not isinstance(extend, dict) or extend.get("useDefault") is not True:
        errors.append("gitleaks config must extend the default ruleset with useDefault = true")

    disabled_rules = extend.get("disabledRules", []) if isinstance(extend, dict) else []
    if EXPECTED_RULE_ID in disabled_rules:
        errors.append("gitleaks config must not disable the generic-api-key rule")

    global_allowlists = config.get("allowlists", [])
    if isinstance(global_allowlists, list):
        for allowlist in global_allowlists:
            if not isinstance(allowlist, dict):
                continue
            for path_regex in allowlist.get("paths", []):
                if "public/portal-assets" in str(path_regex):
                    errors.append("gitleaks config must not define a global allowlist for portal assets")

    matching_rules = [
        rule for rule in config.get("rules", []) if isinstance(rule, dict) and rule.get("id") == EXPECTED_RULE_ID
    ]
    if len(matching_rules) != 1:
        return errors + ["gitleaks config must define exactly one generic-api-key rule extension"]

    allowlists = matching_rules[0].get("allowlists", [])
    if not isinstance(allowlists, list) or len(allowlists) != 1:
        return errors + ["generic-api-key rule extension must define exactly one allowlist"]

    allowlist = allowlists[0]
    if not isinstance(allowlist, dict):
        return errors + ["generic-api-key allowlist must be a TOML table"]

    if allowlist.get("condition") != "AND":
        errors.append("generic-api-key allowlist must require AND semantics")

    if allowlist.get("regexTarget") != "secret":
        errors.append("generic-api-key allowlist must target the extracted secret")

    path_patterns = allowlist.get("paths", [])
    if path_patterns != [EXPECTED_PATH_REGEX]:
        errors.append("generic-api-key allowlist must be scoped only to public/portal-assets/portal.js")

    secret_patterns = allowlist.get("regexes", [])
    if secret_patterns != [EXPECTED_SECRET_REGEX]:
        errors.append("generic-api-key allowlist must match only the generated auth failure false positive")

    return errors


def main() -> int:
    missing_paths = [path for path in (CONFIG_PATH, CI_WORKFLOW_PATH, FIREWALL_WORKFLOW_PATH) if not path.is_file()]
    if missing_paths:
        print("ERROR: gitleaks workflow contract files are missing:", file=sys.stderr)
        for path in missing_paths:
            print(f"  - {path}", file=sys.stderr)
        return 1

    errors = validate_gitleaks_contract(
        config_text=CONFIG_PATH.read_text(encoding="utf-8"),
        ci_workflow_text=CI_WORKFLOW_PATH.read_text(encoding="utf-8"),
        firewall_workflow_text=FIREWALL_WORKFLOW_PATH.read_text(encoding="utf-8"),
    )
    if errors:
        print("ERROR: gitleaks workflow contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("gitleaks workflow contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
