from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "secure-install-pilot.yml"
REQUIREMENTS_README_PATH = REPO_ROOT / "requirements" / "README.md"
DEPENDABOT_GOVERNANCE_PATH = REPO_ROOT / "docs" / "governance" / "DEPENDABOT_PR_GOVERNANCE.md"
ROADMAP_PATH = REPO_ROOT / "docs" / "architecture" / "transformation_portal_roadmap_rereview_2026-04-07.md"
CHECKOUT_SHA = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
SETUP_PYTHON_SHA = "a309ff8b426b58ec0e2a45f0f869d46889d02405"
UPLOAD_ARTIFACT_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_secure_install_pilot_workflow_exists() -> None:
    assert WORKFLOW_PATH.exists()


def test_secure_install_pilot_workflow_triggers_on_dependency_surface_pull_requests() -> None:
    workflow = _load_workflow()
    pull_request = workflow["on"]["pull_request"]

    assert pull_request["branches"] == ["main"]
    assert set(pull_request["paths"]) >= {
        "requirements/**",
        "requirements.txt",
        "requirements-ci.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "scripts/validation/**",
        ".github/workflows/secure-install-pilot.yml",
        ".github/workflows/build.yml",
        ".github/workflows/dependency-update.yml",
    }


def test_secure_install_pilot_workflow_is_advisory_and_uses_pinned_actions() -> None:
    workflow = _load_workflow()
    job = workflow["jobs"]["secure-install-pilot"]
    steps = job["steps"]
    checkout_step = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    setup_python_step = next(step for step in steps if step.get("uses", "").startswith("actions/setup-python@"))
    upload_step = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))

    assert job["continue-on-error"] == "true"
    assert workflow["permissions"]["contents"] == "read"
    assert checkout_step["uses"] == f"actions/checkout@{CHECKOUT_SHA}"
    assert setup_python_step["uses"] == f"actions/setup-python@{SETUP_PYTHON_SHA}"
    assert upload_step["uses"] == f"actions/upload-artifact@{UPLOAD_ARTIFACT_SHA}"


def test_secure_install_pilot_workflow_runs_makefile_targets_and_uploads_artifacts() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["secure-install-pilot"]["steps"]
    install_step = next(step for step in steps if step.get("name") == "Install lock generation tools")
    compile_step = next(step for step in steps if step.get("name") == "Compile secure-install pilot lockfiles")
    check_step = next(step for step in steps if step.get("name") == "Validate secure-install pilot lockfiles")
    upload_step = next(step for step in steps if step.get("name") == "Upload secure-install pilot artifacts")

    assert 'python -m pip install --upgrade "pip<26"' in install_step["run"]
    assert 'python -m pip install "pip-tools==7.5.2"' in install_step["run"]
    assert compile_step["working-directory"] == "requirements"
    assert "make compile-hash-pilot" in compile_step["run"]
    assert 'HASH_PILOT_OUT_DIR="${GITHUB_WORKSPACE}/requirements/.hash-pilot"' in compile_step["run"]
    assert check_step["working-directory"] == "requirements"
    assert "make check-hash-pilot" in check_step["run"]
    assert 'HASH_PILOT_OUT_DIR="${GITHUB_WORKSPACE}/requirements/.hash-pilot"' in check_step["run"]
    assert upload_step["if"] == "always()"
    assert upload_step["with"]["name"] == "secure-install-pilot-locks"
    assert upload_step["with"]["path"] == "requirements/.hash-pilot/"


def test_secure_install_pilot_readme_mentions_local_toolchain_requirement() -> None:
    readme = REQUIREMENTS_README_PATH.read_text(encoding="utf-8")

    assert 'python -m pip install --upgrade "pip<26"' in readme
    assert 'python -m pip install "pip-tools==7.5.2"' in readme
    assert ".github/workflows/secure-install-pilot.yml" in readme


def test_secure_install_pilot_readme_records_explicit_hash_policy() -> None:
    readme = REQUIREMENTS_README_PATH.read_text(encoding="utf-8")

    assert "CI-only" in readme
    assert "advisory control for the non-ML checked-in layered locks" in readme
    assert "standard local install flows remain" in readme
    assert "pinned-without-hashes" in readme
    assert "`requirements.txt`, `requirements-ci.txt`, and" in readme
    assert "`requirements-dev.txt` remain outside this hash-enforced policy decision" in readme
    assert "Promotion to mandatory `--require-hashes` enforcement requires a separate" in readme
    assert "policy decision." in readme


def test_requirements_readme_records_current_curated_web_runtime_baseline() -> None:
    readme = REQUIREMENTS_README_PATH.read_text(encoding="utf-8")

    assert "| FastAPI | `requirements/base.in` | `0.136.1` |" in readme
    assert "| Starlette | `requirements/base.in` + `pyproject.toml` bound | `1.0.1` |" in readme
    assert "| Uvicorn | `requirements/base.in` | `0.46.0` |" in readme
    assert "Starlette `PYSEC-2026-161` patch" in readme


def test_dependabot_governance_doc_includes_dep_pin_changed_checklist() -> None:
    governance_doc = DEPENDABOT_GOVERNANCE_PATH.read_text(encoding="utf-8")

    assert '## "Dep Pin Changed" Checklist' in governance_doc
    assert "Regenerate only the affected governed lockfiles through the existing lock" in governance_doc
    assert "make check-requirements-lock-contract" in governance_doc
    assert "runtime/toolchain requirements stay current" in governance_doc
    assert "make ci" in governance_doc


def test_hash_policy_roadmap_refresh_closes_csp_unlock_and_records_policy_decision() -> None:
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")

    assert "The direct-debug portal CSP unlock is complete" in roadmap
    assert "The hash strategy decision is now explicit" in roadmap
    assert "CI-only advisory control for the" in roadmap
    assert "non-ML layered locks" in roadmap
    assert "No new immediate remediation lane is promoted after the CSP unlock and" in roadmap
    assert "hash-policy closure work." in roadmap
