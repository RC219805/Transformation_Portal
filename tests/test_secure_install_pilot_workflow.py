from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "secure-install-pilot.yml"
CHECKOUT_SHA = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
SETUP_PYTHON_SHA = "a309ff8b426b58ec0e2a45f0f869d46889d02405"
UPLOAD_ARTIFACT_SHA = "bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"


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

    assert job["continue-on-error"] == "true"
    assert workflow["permissions"]["contents"] == "read"
    assert steps[0]["uses"] == f"actions/checkout@{CHECKOUT_SHA}"
    assert steps[1]["uses"] == f"actions/setup-python@{SETUP_PYTHON_SHA}"
    upload_step = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
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
