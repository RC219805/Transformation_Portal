import importlib.util
import re
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

[allowlist]
description = "Ignore generated local artifact directories already excluded from version control"
paths = ['''{gitleaks_contract.EXPECTED_ARTIFACT_PATH_REGEX}''']

[[rules]]
id = "{gitleaks_contract.EXPECTED_RULE_ID}"

[[rules.allowlists]]
description = "Ignore the generated portal auth failure branch false positive"
condition = "AND"
regexTarget = "{gitleaks_contract.EXPECTED_REGEX_TARGET}"
regexes = ['''{gitleaks_contract.EXPECTED_MATCH_REGEX}''']
paths = ['''{gitleaks_contract.EXPECTED_PATH_REGEX}''']

[[rules.allowlists]]
description = "Ignore the SAM2 tiling UI flag false positive"
condition = "AND"
regexTarget = "{gitleaks_contract.EXPECTED_REGEX_TARGET}"
regexes = ['''{gitleaks_contract.EXPECTED_SAM2_TILING_MATCH_REGEX}''']
paths = [
  '''{gitleaks_contract.EXPECTED_SAM2_TILING_PATH_REGEXES[0]}''',
  '''{gitleaks_contract.EXPECTED_SAM2_TILING_PATH_REGEXES[1]}''',
  '''{gitleaks_contract.EXPECTED_SAM2_TILING_PATH_REGEXES[2]}''',
]
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
    assert "gitleaks global allowlist must use singular [allowlist] syntax for v8.21.x" in errors


def test_missing_generated_artifact_allowlist_is_reported() -> None:
    config_text = valid_config_text().replace(
        f"""
[allowlist]
description = "Ignore generated local artifact directories already excluded from version control"
paths = ['''{gitleaks_contract.EXPECTED_ARTIFACT_PATH_REGEX}''']
""",
        "",
        1,
    )
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=config_text,
        ci_workflow_text=valid_ci_workflow_text(),
        firewall_workflow_text=valid_firewall_workflow_text(),
    )
    assert "gitleaks config must define exactly one ignored generated artifact directory allowlist" in errors


def test_broad_generated_artifact_allowlist_is_reported() -> None:
    config_text = valid_config_text().replace(
        f"paths = ['''{gitleaks_contract.EXPECTED_ARTIFACT_PATH_REGEX}''']",
        "paths = ['''.*''']",
        1,
    )
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=config_text,
        ci_workflow_text=valid_ci_workflow_text(),
        firewall_workflow_text=valid_firewall_workflow_text(),
    )
    assert "gitleaks global allowlist must be scoped only to ignored generated artifact directories" in errors


def test_generated_artifact_allowlist_matches_frontdoor_tmp_output() -> None:
    pattern = re.compile(gitleaks_contract.EXPECTED_ARTIFACT_PATH_REGEX)

    assert pattern.search("web/secure-landing/tmp/dev-auth-state.json")
    assert pattern.search("/repo/web/secure-landing/tmp/preview-keys.json")
    assert pattern.search(".runtime/fastvlm/runtime.json")
    assert pattern.search(".venv/lib/python3.11/site-packages/generated.py")
    assert pattern.search(".venv-da3/lib/python3.11/site-packages/generated.py")
    assert pattern.search(".coverage")
    assert pattern.search(".coverage.core-py3.12")
    assert pattern.search("coverage.xml")
    assert pattern.search("coverage.json")
    assert pattern.search("htmlcov/index.html")
    assert pattern.search("node_modules/example/index.js")
    assert pattern.search("web/secure-landing/node_modules/example/index.js")
    assert pattern.search("cloudflare/transformationportal-worker/.wrangler/state/v3/cache")
    assert pattern.search("archive_reports/fixity/hash_manifest.csv.gz")
    assert pattern.search("archive/experiments/local-run/report.json")
    assert pattern.search("artifact_store/local/blob.bin")
    assert not pattern.search("web/secure-landing/portal-src/tmp/dev-auth-state.json")
    assert not pattern.search("input_images/client/secret.txt")
    assert not pattern.search("data/sample_images/README.md")
    assert not pattern.search("Architectural_Plans/client.pdf")
    assert not pattern.search("external/vendor/secret.txt")


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
    assert "generic-api-key allowlists must be scoped only to approved false-positive patterns" in errors
    assert "generic-api-key allowlist must match only the generated auth failure false-positive branch" in errors


def test_wrong_regex_target_is_reported() -> None:
    config_text = valid_config_text().replace(
        f'regexTarget = "{gitleaks_contract.EXPECTED_REGEX_TARGET}"',
        'regexTarget = "secret"',
        1,
    )
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=config_text,
        ci_workflow_text=valid_ci_workflow_text(),
        firewall_workflow_text=valid_firewall_workflow_text(),
    )
    assert "generic-api-key allowlists must target source lines" in errors


def test_wrong_match_regex_is_reported() -> None:
    config_text = valid_config_text().replace(
        f"regexes = ['''{gitleaks_contract.EXPECTED_MATCH_REGEX}''']",
        "regexes = ['''normalizedReason===auth_failure''']",
        1,
    )
    errors = gitleaks_contract.validate_gitleaks_contract(
        config_text=config_text,
        ci_workflow_text=valid_ci_workflow_text(),
        firewall_workflow_text=valid_firewall_workflow_text(),
    )
    assert "generic-api-key allowlist must match only the generated auth failure false-positive branch" in errors
