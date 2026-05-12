from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
MISSING_ROOT_DOC = "/".join(("docs", "MISSING.md"))


def _copy_repo_script(repo_root: Path) -> Path:
    source_dir = REPO_ROOT / "scripts" / "governance"
    destination_dir = repo_root / "scripts" / "governance"
    source = source_dir / "check_stale_docs_paths.py"
    destination = destination_dir / "check_stale_docs_paths.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    mode = destination.stat().st_mode
    destination.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return destination


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _init_repo(repo_root: Path) -> None:
    result = _run(["git", "init", "-q"], repo_root)
    assert result.returncode == 0, result.stdout + result.stderr
    for key, value in (
        ("commit.gpgsign", "false"),
        ("user.email", "ci@example.com"),
        ("user.name", "CI"),
    ):
        config = _run(["git", "config", key, value], repo_root)
        assert config.returncode == 0, config.stdout + config.stderr


def _write(path: Path, content: str = "test\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _track(repo_root: Path, *paths: str) -> None:
    result = _run(["git", "add", *paths], repo_root)
    assert result.returncode == 0, result.stdout + result.stderr


def _commit(repo_root: Path, message: str = "baseline") -> None:
    result = _run(["git", "commit", "-qm", message], repo_root)
    assert result.returncode == 0, result.stdout + result.stderr


def _run_checker(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, "scripts/governance/check_stale_docs_paths.py"],
        repo_root,
    )


def _assert_stale_reference_detected(
    repo_root: Path,
    reference_text: str,
) -> None:
    _write(repo_root / "notes.md", f"See {reference_text}\n")
    _track(repo_root, "notes.md")

    result = _run_checker(repo_root)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"notes.md: references missing {MISSING_ROOT_DOC}" in result.stdout


def test_checker_fails_for_changed_file_with_missing_root_doc_reference(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _track(
        repo_root,
        "docs/README.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _assert_stale_reference_detected(repo_root, MISSING_ROOT_DOC)


def test_checker_fails_for_changed_file_with_dot_slash_root_doc_reference(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _track(
        repo_root,
        "docs/README.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _assert_stale_reference_detected(repo_root, f"./{MISSING_ROOT_DOC}")


def test_checker_fails_for_changed_file_with_dot_dot_slash_root_doc_reference(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _track(
        repo_root,
        "docs/README.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _assert_stale_reference_detected(repo_root, f"../{MISSING_ROOT_DOC}")


def test_checker_allows_existing_root_doc_reference(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _track(
        repo_root,
        "docs/README.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _write(repo_root / "notes.md", "See docs/README.md\n")
    _track(repo_root, "notes.md")

    result = _run_checker(repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No stale docs path references detected" in result.stdout


def test_checker_ignores_docs_subdirectory_references(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _track(
        repo_root,
        "docs/README.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _write(repo_root / "notes.md", "See docs/guides/TROUBLESHOOTING.md\n")
    _track(repo_root, "notes.md")

    result = _run_checker(repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No stale docs path references detected" in result.stdout


@pytest.mark.parametrize(
    "archive_path",
    [
        "docs/historical/example.md",
        "docs/pr_archive/example.md",
    ],
)
def test_checker_allows_explicit_missing_root_doc_reference_in_archive_docs(
    tmp_path: Path,
    archive_path: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _track(
        repo_root,
        "docs/README.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _write(
        repo_root / archive_path,
        f"Historical broken reference: {MISSING_ROOT_DOC} does not exist.\n",
    )
    _track(repo_root, archive_path)

    result = _run_checker(repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No stale docs path references detected" in result.stdout


def test_checker_still_flags_unqualified_missing_root_doc_reference_in_archive_docs(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _track(
        repo_root,
        "docs/README.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _write(
        repo_root / "docs" / "pr_archive" / "example.md",
        f"See {MISSING_ROOT_DOC}\n",
    )
    _track(repo_root, "docs/pr_archive/example.md")

    result = _run_checker(repo_root)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"docs/pr_archive/example.md: references missing {MISSING_ROOT_DOC}" in result.stdout


def test_checker_ignores_unchanged_files_with_stale_references(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _write(repo_root / "legacy.md", f"Legacy {MISSING_ROOT_DOC}\n")
    _track(
        repo_root,
        "docs/README.md",
        "legacy.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _write(repo_root / "fresh.md", "Fresh content\n")
    _track(repo_root, "fresh.md")

    result = _run_checker(repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "legacy.md" not in result.stdout


def test_checker_ignores_cache_and_binary_like_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _track(
        repo_root,
        "docs/README.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _write_bytes(
        repo_root / "__pycache__" / "module.cpython-311.pyc",
        b"docs/" + b"MISSING.md\0binary",
    )
    _track(repo_root, "__pycache__/module.cpython-311.pyc")

    result = _run_checker(repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MISSING.md" not in result.stdout


def test_checker_uses_git_diff_fallbacks_when_origin_main_is_missing(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "docs" / "README.md")
    _write(repo_root / "notes.md", "Baseline\n")
    _track(
        repo_root,
        "docs/README.md",
        "notes.md",
        "scripts/governance/check_stale_docs_paths.py",
    )
    _commit(repo_root)

    _assert_stale_reference_detected(repo_root, f"../{MISSING_ROOT_DOC}")

    result = _run_checker(repo_root)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Unable to determine changed files" not in result.stdout
