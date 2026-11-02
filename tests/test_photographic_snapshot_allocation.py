"""Snapshot reads reserve observed file bytes while preserving exact authority."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager

import pytest

from transformation_portal.photographic_delivery import native

pytestmark = pytest.mark.unit


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def _observe_reads(monkeypatch, callback):
    original = native.os.fdopen

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            callback(size)
            return self.stream.read(size)

    monkeypatch.setattr(native.os, "fdopen", lambda *args, **kwargs: Reader(original(*args, **kwargs)))
    yield


@pytest.mark.parametrize("maximum", [None, 37, 1024, native.MAX_SOURCE_BYTES])
def test_regular_snapshot_read_is_bounded_by_observed_size_not_global_ceiling(tmp_path, monkeypatch, maximum):
    raw = b"a hash-bound retained source snapshot"
    path = tmp_path / "source.bin"
    path.write_bytes(raw)
    observed = []

    def bounded(size):
        observed.append(size)
        assert size == len(raw) + 1, "A small retained source requested a ceiling-sized bytes allocation"

    with _observe_reads(monkeypatch, bounded):
        value = native.read_snapshot(None, path, _sha(raw), **({} if maximum is None else {"maximum": maximum}))
    assert value == raw and observed == [len(raw) + 1]


@pytest.mark.parametrize("mutation", ["grow", "shrink", "replace_bytes"])
def test_mutation_after_fstat_never_inherits_original_snapshot_authority(tmp_path, monkeypatch, mutation):
    raw = b"the exact immutable original source"
    path = tmp_path / "source.bin"
    path.write_bytes(raw)

    def mutate(size):
        assert size == len(raw) + 1
        if mutation == "grow":
            with path.open("ab") as stream:
                stream.write(b"unapproved growth" * 100)
        elif mutation == "shrink":
            path.write_bytes(raw[:-1])
        else:
            path.write_bytes(b"X" + raw[1:])

    with _observe_reads(monkeypatch, mutate), pytest.raises(ValueError, match="SHA-256 changed"):
        native.read_snapshot(None, path, _sha(raw))


@pytest.mark.parametrize("kind", ["oversized", "empty", "fifo"])
def test_invalid_file_size_or_kind_fails_before_any_bytes_allocation(tmp_path, monkeypatch, kind):
    path = tmp_path / "source.bin"
    raw = b"retained source"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(b"" if kind == "empty" else raw)

    def forbidden(_size):
        pytest.fail("An ineligible file reached snapshot allocation")

    with _observe_reads(monkeypatch, forbidden), pytest.raises(ValueError, match="bounded regular file"):
        native.read_snapshot(None, path, _sha(raw), maximum=len(raw) - 1)


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "parent_traversal"])
def test_source_access_still_rejects_symlink_and_parent_traversal(tmp_path, kind):
    raw = b"retained source"
    source = tmp_path / "source.bin"
    source.write_bytes(raw)
    if kind == "file_symlink":
        path = tmp_path / "linked.bin"
        path.symlink_to(source)
    elif kind == "directory_symlink":
        directory = tmp_path / "linked"
        directory.symlink_to(tmp_path, target_is_directory=True)
        path = directory / source.name
    else:
        directory = tmp_path / "nested"
        directory.mkdir()
        path = directory / ".." / source.name
    with pytest.raises(ValueError, match="symlink"):
        native.read_snapshot(None, path, _sha(raw))


def test_byte_backed_snapshot_retains_its_exact_object_and_digest_contract():
    raw = b"already owned immutable snapshot"
    assert native.read_snapshot(raw, None, _sha(raw), maximum=len(raw)) is raw
    with pytest.raises(ValueError, match="SHA-256 changed"):
        native.read_snapshot(raw, None, "a" * 64)
