"""Keep local pre-commit hooks independent of a bare `python` executable."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
REPO_PYTHON_RUNNER = "scripts/setup/run_repo_python.sh"
PYTHON_HOOK_IDS = {"auto-format-staged", "check-test-markers", "ban-tautological-tests"}


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
