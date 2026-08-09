"""Tests for the pip-tools cache tracking guardrail."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit


def _load_guard_module() -> ModuleType:
    """Load the validation script without crossing the scripts import boundary."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "validation" / "check_piptools_cache_tracked.py"
    spec = importlib.util.spec_from_file_location("check_piptools_cache_tracked_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard_module() -> ModuleType:
    """Return the dynamically loaded guard module."""
    return _load_guard_module()


def test_tracked_cache_files_uses_exact_pathspec_and_sorts_nul_output(
    monkeypatch: pytest.MonkeyPatch,
    guard_module: ModuleType,
) -> None:
    """The git query is bounded to the cache path and returns stable ordering."""
    observed: dict[str, object] = {}

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        observed.update(
            args=args,
            check=check,
            capture_output=capture_output,
            text=text,
        )
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="requirements/.pip-tools-cache/z.json\0requirements/.pip-tools-cache/a.json\0\0",
            stderr="",
        )

    monkeypatch.setattr(guard_module.subprocess, "run", fake_run)

    assert guard_module.tracked_cache_files() == [
        "requirements/.pip-tools-cache/a.json",
        "requirements/.pip-tools-cache/z.json",
    ]
    assert observed == {
        "args": ["git", "ls-files", "-z", "--", "requirements/.pip-tools-cache"],
        "check": False,
        "capture_output": True,
        "text": True,
    }


def test_tracked_cache_files_fails_closed_when_git_query_fails(
    monkeypatch: pytest.MonkeyPatch,
    guard_module: ModuleType,
) -> None:
    """A broken git query cannot be mistaken for an empty tracked-file set."""
    failed = subprocess.CompletedProcess(
        ["git", "ls-files"],
        128,
        stdout="",
        stderr="  fatal: not a git repository  \n",
    )
    monkeypatch.setattr(guard_module.subprocess, "run", lambda *_args, **_kwargs: failed)

    with pytest.raises(RuntimeError, match=r"git ls-files failed \(128\): fatal: not a git repository"):
        guard_module.tracked_cache_files()


def test_main_reports_clean_repository(
    monkeypatch: pytest.MonkeyPatch,
    guard_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No tracked cache artifacts is the successful CLI outcome."""
    monkeypatch.setattr(guard_module, "tracked_cache_files", lambda: [])

    assert guard_module.main() == 0

    captured = capsys.readouterr()
    assert captured.out == "pip-tools cache guardrail passed: no tracked files under requirements/.pip-tools-cache.\n"
    assert captured.err == ""


def test_main_lists_tracked_cache_files_and_remediation(
    monkeypatch: pytest.MonkeyPatch,
    guard_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Policy failures list every tracked artifact and the recovery command."""
    tracked = [
        "requirements/.pip-tools-cache/a.json",
        "requirements/.pip-tools-cache/nested/b.json",
    ]
    monkeypatch.setattr(guard_module, "tracked_cache_files", lambda: tracked)

    assert guard_module.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "ERROR: tracked pip-tools cache files detected:",
        "  - requirements/.pip-tools-cache/a.json",
        "  - requirements/.pip-tools-cache/nested/b.json",
        "Remediation: remove cache artifacts from git tracking "
        "(for example: git rm --cached -r requirements/.pip-tools-cache).",
    ]


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (FileNotFoundError(), "ERROR: git executable not found"),
        (RuntimeError("git ls-files failed (3): broken worktree"), "ERROR: git ls-files failed (3): broken worktree"),
    ],
)
def test_main_reports_operational_errors_separately_from_policy_failures(
    monkeypatch: pytest.MonkeyPatch,
    guard_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    expected_message: str,
) -> None:
    """Missing or broken git returns the distinct operational exit code."""

    def fail() -> list[str]:
        raise error

    monkeypatch.setattr(guard_module, "tracked_cache_files", fail)

    assert guard_module.main() == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{expected_message}\n"
