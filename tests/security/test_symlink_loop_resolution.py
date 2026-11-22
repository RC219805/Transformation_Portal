"""Version-independent symlink-loop detection for non-strict path resolution.

Python 3.13 stopped raising ``RuntimeError`` from ``Path.resolve()`` for a
symlink loop. ``resolve_rejecting_symlink_loops`` restores one contract on
every supported interpreter.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from transformation_portal.core.security.path_safety import SymlinkLoopError, resolve_rejecting_symlink_loops

pytestmark = pytest.mark.security


def _assert_loop(path: Path) -> SymlinkLoopError:
    with pytest.raises(SymlinkLoopError) as caught:
        resolve_rejecting_symlink_loops(path)
    error = caught.value
    # Existing handlers catch RuntimeError (Python <= 3.12) or OSError.
    assert isinstance(error, RuntimeError)
    assert isinstance(error, OSError)
    assert error.errno == errno.ELOOP
    assert error.filename == os.fspath(path)
    return error


def test_loop_in_final_component(tmp_path):
    (tmp_path / "sub").mkdir()
    os.symlink("../sub/loop", tmp_path / "sub" / "loop")
    _assert_loop(tmp_path / "sub" / "loop")


def test_loop_in_middle_component_with_missing_tail(tmp_path):
    os.symlink("loop", tmp_path / "loop")
    _assert_loop(tmp_path / "loop" / "missing" / "tail.jpg")


@pytest.mark.parametrize("prefix", [".", "missing/.."])
def test_parent_component_cannot_discard_a_loop(tmp_path, prefix):
    os.symlink("loop", tmp_path / "loop")
    safe = tmp_path / "safe.jpg"
    safe.write_bytes(b"safe")
    _assert_loop(tmp_path / prefix / "loop" / ".." / safe.name)


@pytest.mark.parametrize("prefix", [".", "missing/.."])
def test_parent_component_cannot_discard_a_loop_inside_a_link_target(tmp_path, prefix):
    os.symlink("cycle", tmp_path / "cycle")
    os.symlink(f"{prefix}/cycle/../safe.jpg", tmp_path / "link")
    (tmp_path / "safe.jpg").write_bytes(b"safe")
    _assert_loop(tmp_path / "link")


def test_cycle_inside_a_target_with_an_initial_missing_component(tmp_path):
    os.symlink("missing/../cycle/../safe.jpg", tmp_path / "cycle")
    _assert_loop(tmp_path / "cycle")


def test_self_referential_link(tmp_path):
    os.symlink(tmp_path / "self", tmp_path / "self")
    _assert_loop(tmp_path / "self")


def test_two_link_cycle(tmp_path):
    os.symlink("second", tmp_path / "first")
    os.symlink("first", tmp_path / "second")
    _assert_loop(tmp_path / "first")
    _assert_loop(tmp_path / "second" / "child")


def test_string_argument_is_accepted(tmp_path):
    os.symlink("loop", tmp_path / "loop")
    with pytest.raises(SymlinkLoopError) as caught:
        resolve_rejecting_symlink_loops(str(tmp_path / "loop"))
    assert caught.value.filename == str(tmp_path / "loop")


def test_ordinary_symlink_resolves_to_its_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "photo.jpg").write_bytes(b"x")
    os.symlink(target, tmp_path / "link")
    resolved = resolve_rejecting_symlink_loops(tmp_path / "link" / "photo.jpg")
    assert resolved == (target / "photo.jpg").resolve(strict=True)
    assert resolved == (tmp_path / "link" / "photo.jpg").resolve()


def test_dangling_symlink_is_not_a_loop(tmp_path):
    os.symlink(tmp_path / "absent", tmp_path / "dangling")
    resolved = resolve_rejecting_symlink_loops(tmp_path / "dangling")
    assert resolved == (tmp_path / "absent").resolve()
    assert resolved == (tmp_path / "dangling").resolve()


def test_missing_path_resolves_lexically(tmp_path):
    missing = tmp_path / "missing" / "nested" / "file.txt"
    resolved = resolve_rejecting_symlink_loops(missing)
    assert resolved == missing.resolve()
    assert resolved == tmp_path.resolve() / "missing" / "nested" / "file.txt"
    assert not resolved.exists()


def test_repeated_link_and_missing_tail_preserve_normal_resolution(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    os.symlink("target", tmp_path / "link", target_is_directory=True)
    path = tmp_path / "link" / ".." / "link" / "missing" / "image.jpg"
    assert resolve_rejecting_symlink_loops(path) == target.resolve() / "missing" / "image.jpg"


def test_plain_path_matches_path_resolve(tmp_path):
    plain = tmp_path / "dir" / "file.txt"
    plain.parent.mkdir()
    plain.write_text("x", encoding="utf-8")
    assert resolve_rejecting_symlink_loops(plain) == plain.resolve(strict=True)
    assert resolve_rejecting_symlink_loops(tmp_path / "dir" / ".." / "dir") == plain.parent.resolve()
    assert resolve_rejecting_symlink_loops(plain / ".." / "absent") == plain.parent.resolve() / "absent"


def test_relative_path_resolves_against_working_directory(tmp_path, monkeypatch):
    (tmp_path / "dir").mkdir()
    os.symlink("loop", tmp_path / "loop")
    monkeypatch.chdir(tmp_path)
    assert resolve_rejecting_symlink_loops(Path("dir")) == (tmp_path / "dir").resolve()
    _assert_loop(Path("loop"))
