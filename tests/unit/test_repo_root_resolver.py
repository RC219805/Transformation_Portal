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


def test_resolve_repo_root_through_symlinked_repo_path(tmp_path: Path) -> None:
    repo_real = tmp_path / "repo_real"
    (repo_real / ".github" / "workflows").mkdir(parents=True)
    (repo_real / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (repo_real / "scripts").mkdir(parents=True)
    (repo_real / "scripts" / "tool.py").write_text("print('ok')\n", encoding="utf-8")

    repo_link = tmp_path / "repo_link"
    try:
        repo_link.symlink_to(repo_real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation unsupported in test environment: {exc}")

    root = resolve_repo_root(start=repo_link / "scripts" / "tool.py")
    assert root == repo_real.resolve()
