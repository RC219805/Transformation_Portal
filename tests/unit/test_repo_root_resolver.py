from pathlib import Path

import pytest

from scripts.lib.repo_root import RepoRootError, resolve_repo_root


def _expected_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    while True:
        if (current / "pyproject.toml").is_file() and (current / ".github" / "workflows").is_dir():
            return current
        if current.parent == current:
            raise AssertionError("No test repo root anchors found")
        current = current.parent


def test_resolve_repo_root_walks_up_from_file_path() -> None:
    root = resolve_repo_root(start=Path(__file__))
    assert root == _expected_repo_root(Path(__file__))


def test_resolve_repo_root_honors_valid_repo_override(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    assert resolve_repo_root(repo=tmp_path) == tmp_path


def test_resolve_repo_root_rejects_invalid_repo_override(tmp_path: Path) -> None:
    with pytest.raises(RepoRootError, match="Invalid --repo path"):
        resolve_repo_root(repo=tmp_path)


def test_resolve_repo_root_fails_when_anchors_missing(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    with pytest.raises(RepoRootError, match="Unable to locate repository root"):
        resolve_repo_root(start=deep)
