"""Regression tests for the shared lint runner shell script."""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "lint_runner.sh"


def _write_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    _run_git(repo_root, "init", "-b", "main")
    _run_git(repo_root, "config", "user.name", "Lint Runner Test")
    _run_git(repo_root, "config", "user.email", "lint-runner@example.com")
    (repo_root / "README.md").write_text("base\n", encoding="utf-8")
    _run_git(repo_root, "add", "README.md")
    _run_git(repo_root, "commit", "-m", "base")


def _write_fake_python(path: Path, log_path: Path) -> Path:
    script = (
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"printf '%s\\n' \"$*\" >> {shlex.quote(str(log_path))}",
                "exit 0",
            ]
        )
        + "\n"
    )
    return _write_executable(path, script)


def _run_lint_runner(repo_root: Path, fake_python: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(fake_python)
    env["LINT_RUNNER_GITHUB_EVENT_NAME"] = "pull_request"
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), "pr"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_failing_fake_python(path: Path, log_path: Path) -> Path:
    """Interpreter stub whose flake8 / pylint statuses come from FAKE_FLAKE8_EXIT / FAKE_PYLINT_EXIT."""
    script = (
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"printf '%s\\n' \"$*\" >> {shlex.quote(str(log_path))}",
                'if [ "${3:-}" = "--version" ]; then exit 0; fi',
                'if [ "${2:-}" = "flake8" ]; then exit "${FAKE_FLAKE8_EXIT:-0}"; fi',
                'if [ "${2:-}" = "pylint" ]; then exit "${FAKE_PYLINT_EXIT:-0}"; fi',
                "exit 0",
            ]
        )
        + "\n"
    )
    return _write_executable(path, script)


def _run_lint_runner_mode(
    tmp_path: Path, mode: str, *, flake8_exit: int = 0, pylint_exit: int = 0
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    log_path = tmp_path / "lint.log"
    fake_python = _write_failing_fake_python(tmp_path / "fake-python", log_path)

    env = os.environ.copy()
    env["PYTHON_BIN"] = str(fake_python)
    env["LINT_RUNNER_GITHUB_EVENT_NAME"] = "pull_request"
    env["FAKE_FLAKE8_EXIT"] = str(flake8_exit)
    env["FAKE_PYLINT_EXIT"] = str(pylint_exit)
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), mode],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    linter_runs = [
        line.split()[1]
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("-m ") and not line.endswith("--version")
    ]
    return result, linter_runs


@pytest.mark.parametrize(
    ("flake8_exit", "pylint_exit"),
    [(1, 0), (0, 1), (0, 2), (0, 32), (1, 1)],
    ids=["flake8-fails", "pylint-fatal", "pylint-error", "pylint-usage", "both-fail"],
)
def test_advisory_mode_never_blocks_and_runs_both_linters(tmp_path: Path, flake8_exit: int, pylint_exit: int) -> None:
    result, linter_runs = _run_lint_runner_mode(tmp_path, "advisory", flake8_exit=flake8_exit, pylint_exit=pylint_exit)

    assert result.returncode == 0, result.stdout + result.stderr
    assert linter_runs == ["flake8", "pylint"]
    if flake8_exit:
        assert f"flake8 failed (exit={flake8_exit})" in result.stdout
    if pylint_exit:
        assert f"pylint found blocking issues (exit={pylint_exit})" in result.stdout


@pytest.mark.parametrize("mode", ["local", "pr"])
def test_blocking_modes_still_run_pylint_after_flake8_fails(tmp_path: Path, mode: str) -> None:
    result, linter_runs = _run_lint_runner_mode(tmp_path, mode, flake8_exit=1)

    assert result.returncode == 1, result.stdout + result.stderr
    assert linter_runs == ["flake8", "pylint"]
    assert "flake8 failed (exit=1)" in result.stdout
    assert "pylint passed" in result.stdout


@pytest.mark.parametrize(
    ("pylint_exit", "expected_returncode"),
    [(1, 1), (2, 1), (32, 1), (4, 0), (8, 0), (16, 0), (28, 0), (30, 1)],
)
def test_blocking_modes_fail_only_on_fatal_error_or_usage_pylint_bits(
    tmp_path: Path, pylint_exit: int, expected_returncode: int
) -> None:
    result, linter_runs = _run_lint_runner_mode(tmp_path, "local", pylint_exit=pylint_exit)

    assert result.returncode == expected_returncode, result.stdout + result.stderr
    assert linter_runs == ["flake8", "pylint"]


def test_pr_mode_falls_back_when_diff_range_is_unavailable(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    log_path = tmp_path / "lint.log"
    fake_python = _write_fake_python(tmp_path / "fake-python", log_path)

    result = _run_lint_runner(repo_root, fake_python)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pylint candidates=3" in result.stdout
    assert "fallback pylint" in result.stdout

    pylint_invocation = next(
        line for line in log_path.read_text(encoding="utf-8").splitlines() if line.startswith("-m pylint --jobs=1")
    )
    assert "src/tp/phase4/verify_phase4_chain.py" in pylint_invocation
    assert "tests/test_material_response.py" in pylint_invocation
    assert "tests/test_depth_tools.py" in pylint_invocation


def test_pr_mode_uses_changed_python_files_when_origin_main_is_available(tmp_path: Path) -> None:
    origin_root = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(origin_root)],
        capture_output=True,
        text=True,
        check=True,
    )

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    _run_git(repo_root, "remote", "add", "origin", str(origin_root))
    _run_git(repo_root, "push", "-u", "origin", "main")
    _run_git(repo_root, "checkout", "-b", "feature/lint-runner")

    changed_python = repo_root / "tests" / "changed_for_lint_runner.py"
    changed_python.parent.mkdir(parents=True, exist_ok=True)
    changed_python.write_text("def lint_target() -> int:\n    return 1\n", encoding="utf-8")
    _run_git(repo_root, "add", str(changed_python.relative_to(repo_root)))
    _run_git(repo_root, "commit", "-m", "add python change")

    log_path = tmp_path / "lint.log"
    fake_python = _write_fake_python(tmp_path / "fake-python", log_path)

    result = _run_lint_runner(repo_root, fake_python)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "fallback pylint" not in result.stdout
    assert "pylint candidates=1" in result.stdout

    pylint_invocation = next(
        line for line in log_path.read_text(encoding="utf-8").splitlines() if line.startswith("-m pylint --jobs=1")
    )
    assert "tests/changed_for_lint_runner.py" in pylint_invocation
