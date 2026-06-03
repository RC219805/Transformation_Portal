"""Keep local pre-commit hooks independent of a bare `python` executable."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
MAKEFILE = REPO_ROOT / "Makefile"
CLAUDE_GUIDE = REPO_ROOT / "CLAUDE.md"
REPO_PYTHON_RUNNER = "scripts/setup/run_repo_python.sh"
PYTHON_HOOK_IDS = {
    "auto-format-staged",
    "check-design-tokens-doc",
    "check-script-topology",
    "check-test-markers",
    "check-unsafe-torch-load",
    "ban-tautological-tests",
}


def _load_pre_commit_config() -> dict:
    assert PRE_COMMIT_CONFIG.exists(), f"missing pre-commit config at {PRE_COMMIT_CONFIG}"
    return yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))


def test_python_pre_commit_hooks_use_repo_python_runner() -> None:
    config = _load_pre_commit_config()
    hooks = {
        str(hook.get("id")): hook
        for repo in config.get("repos", [])
        for hook in repo.get("hooks", [])
        if hook.get("id") in PYTHON_HOOK_IDS
    }

    assert set(hooks) == PYTHON_HOOK_IDS
    for hook_id, hook in hooks.items():
        entry = str(hook.get("entry", ""))
        assert entry.startswith(f"{REPO_PYTHON_RUNNER} "), f"{hook_id} must resolve Python through the repo runner"
        assert not entry.startswith("python "), f"{hook_id} must not depend on a bare python executable"


def test_repo_python_runner_is_executable() -> None:
    runner = REPO_ROOT / REPO_PYTHON_RUNNER
    assert runner.exists(), f"missing repo Python runner at {runner}"
    assert os.access(runner, os.X_OK), f"{runner} must be executable for pre-commit system hooks"


def test_install_hooks_target_documents_all_installed_hook_types() -> None:
    config = _load_pre_commit_config()
    installed_hook_types = set(config.get("default_install_hook_types") or [])
    makefile = MAKEFILE.read_text(encoding="utf-8")

    assert installed_hook_types == {"pre-commit", "pre-push"}
    assert "Install git pre-commit and pre-push hooks" in makefile
    assert "Installing git pre-commit and pre-push hooks" in makefile
    assert "Install git pre-commit hook" not in makefile
    assert "Installing git pre-commit hook" not in makefile

    claude_guide = CLAUDE_GUIDE.read_text(encoding="utf-8")
    assert "make install-hooks         # install git pre-commit and pre-push hooks" in claude_guide
    assert "make install-hooks         # install git pre-commit hook" not in claude_guide


def test_pre_commit_config_operator_guidance_prefers_make_targets() -> None:
    config_text = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")

    assert "#   make install-hooks" in config_text
    assert "#   make pre-commit" in config_text
    assert "`make install-hooks` invokes the repo-managed `.venv` pre-commit" in config_text
    assert "#   pre-commit install -f" not in config_text
    assert "#   pre-commit run --all-files" not in config_text
    assert "`pre-commit install -f` (invoked by `make install-hooks`)" not in config_text


def test_dependency_constraints_hook_runs_for_root_dependency_metadata() -> None:
    config = _load_pre_commit_config()
    hooks = {str(hook.get("id")): hook for repo in config.get("repos", []) for hook in repo.get("hooks", [])}
    hook = hooks["validate-dependency-constraints"]
    pattern = re.compile(str(hook["files"]))

    expected_matches = {
        "requirements/base.in",
        "requirements/base.txt",
        "requirements/dev.in",
        "requirements/dev.txt",
        "pyproject.toml",
        "scripts/validate_dependency_constraints.sh",
    }
    for path in expected_matches:
        assert pattern.fullmatch(path), f"dependency constraints hook must run for {path}"

    for path in ("README.md", "CONTRIBUTING.md", "scripts/validation/other_check.py"):
        assert not pattern.fullmatch(path), f"dependency constraints hook should not run for {path}"


def test_script_topology_hook_runs_for_wrapper_and_canonical_surfaces() -> None:
    config = _load_pre_commit_config()
    hooks = {str(hook.get("id")): hook for repo in config.get("repos", []) for hook in repo.get("hooks", [])}
    hook = hooks["check-script-topology"]
    pattern = re.compile(str(hook["files"]))

    assert hook["entry"] == "scripts/setup/run_repo_python.sh scripts/governance/check_script_topology.py"
    assert hook["pass_filenames"] is False

    expected_matches = {
        "scripts/install_models.py",
        "scripts/setup/install_models.py",
        "scripts/pipelines/lux_render_pipeline.py",
        "src/transformation_portal/pipelines/lux_render_pipeline.py",
        "src/transformation_portal/perceptual/synthetic_viewer.py",
        "archive/scripts/legacy-organization/organize_outputs.sh",
    }
    for path in expected_matches:
        assert pattern.fullmatch(path), f"script topology hook must run for {path}"

    for path in ("README.md", "docs/guides/SETUP_GUIDE.md", "src/transformation_portal/app.py"):
        assert not pattern.fullmatch(path), f"script topology hook should not run for {path}"
