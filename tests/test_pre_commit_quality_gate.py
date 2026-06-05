from __future__ import annotations

import importlib.util
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_quality_gate() -> ModuleType:
    module_path = REPO_ROOT / "scripts" / "utilities" / "pre-commit-quality-check.py"
    spec = importlib.util.spec_from_file_location("pre_commit_quality_check", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QUALITY_GATE = _load_quality_gate()


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _init_repo(repo_root: Path) -> None:
    result = _run(["git", "init", "-q"], repo_root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _run(["git", "config", "user.email", "test@example.com"], repo_root).returncode == 0
    assert _run(["git", "config", "user.name", "Test User"], repo_root).returncode == 0
    assert _run(["git", "config", "commit.gpgsign", "false"], repo_root).returncode == 0


def _write(path: Path, content: str = "test\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_repo_file(relative_path: str, repo_root: Path, *, executable: bool = False) -> Path:
    source = REPO_ROOT / relative_path
    destination = repo_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if executable:
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return destination


def test_git_paths_includes_staged_renames_by_default(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)

    _write(repo_root / "old_name.py", "print('old')\n")
    assert _run(["git", "add", "old_name.py"], repo_root).returncode == 0
    commit_result = _run(["git", "commit", "-m", "initial"], repo_root)
    assert commit_result.returncode == 0, commit_result.stdout + commit_result.stderr

    rename_result = _run(["git", "mv", "old_name.py", "new_name.py"], repo_root)
    assert rename_result.returncode == 0, rename_result.stdout + rename_result.stderr

    staged_paths = QUALITY_GATE.git_paths(repo_root, all_files=False)

    assert repo_root / "new_name.py" in staged_paths


def test_root_file_placement_uses_canonical_shell_allowlist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_file("scripts/setup/pre-commit-check.sh", repo_root, executable=True)

    _write(repo_root / "SECURITY.md", "# Security\n")
    add_result = _run(["git", "add", "SECURITY.md"], repo_root)
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    outcome = QUALITY_GATE.check_root_file_placement(repo_root, all_files=False)

    assert outcome.ok


def test_choose_python_prefers_lint_venv_for_ci_parity(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    lint_python = repo_root / ".venv-lint" / "bin" / "python"
    venv_python = repo_root / ".venv" / "bin" / "python"
    _write(lint_python, "")
    _write(venv_python, "")

    monkeypatch.setattr(QUALITY_GATE.sys, "executable", str(repo_root / "system-python"))
    monkeypatch.setattr(
        QUALITY_GATE,
        "python_can_import",
        lambda python_bin, modules: python_bin == str(lint_python),
    )

    assert QUALITY_GATE.choose_python(repo_root) == str(lint_python)


def test_import_heuristics_ignores_assignment_targets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    sample = repo_root / "sample.py"
    _write(sample, "def assign_only():\n    cv2 = None\n    iio = None\n")

    outcome = QUALITY_GATE.check_import_heuristics([sample], repo_root)

    assert outcome.ok


def test_import_heuristics_still_flags_module_like_usage(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    sample = repo_root / "sample.py"
    _write(sample, "def read_image(path):\n    return cv2.imread(path)\n")

    outcome = QUALITY_GATE.check_import_heuristics([sample], repo_root)

    assert not outcome.ok


def test_pre_commit_hook_uses_python_for_non_executable_quality_gate(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    hook = _copy_repo_file("scripts/pre_commit_hook.sh", repo_root, executable=True)
    _copy_repo_file("scripts/maintenance/pre_commit_hook.sh", repo_root, executable=True)

    quality_gate = repo_root / "scripts" / "utilities" / "pre-commit-quality-check.py"
    _write(
        quality_gate,
        ("#!/usr/bin/env python3\n" "import sys\n" "print('quality gate invoked')\n" "sys.exit(0)\n"),
    )
    quality_gate.chmod(0o644)

    result = _run(["bash", str(hook.relative_to(repo_root))], repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "quality gate invoked" in result.stdout
