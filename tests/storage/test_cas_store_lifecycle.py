"""Lifecycle + branch coverage for the content-addressable store.

Complements the existing atomic-write and hash-mismatch suites by exercising
the deduplication fast paths, ``materialize`` modes, garbage collection
(`gc` / `gc_quarantine`), the `CASFileLock` stale/timeout paths, and the
cross-platform `_fsync_parent_directory` branches. All deterministic: a
``tmp_path`` CAS root, byte payloads, and explicit ``mtime`` manipulation —
no network, no ML, no real concurrency.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.storage import cas_store
from transformation_portal.storage.cas_store import (
    ArtifactStore,
    CASError,
    CASFileLock,
)


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "cas")


# --------------------------------------------------------------------------- #
# Deduplication fast paths
# --------------------------------------------------------------------------- #


def test_add_file_dedup_returns_existing_when_verified(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")

    first = store.add_file(src)
    second = store.add_file(src, verify=True)  # fast-path hit, verified

    assert second.sha256 == first.sha256
    assert second.path == first.path


def test_add_file_dedup_fast_path_without_verify(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    first = store.add_file(src)

    # verify=False takes the unverified fast-path return.
    second = store.add_file(src, verify=False)
    assert second.sha256 == first.sha256


def test_add_file_reads_through_corrupt_object_and_readds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    obj = store.add_file(src)

    # Corrupt the stored object so the verified fast path detects a mismatch
    # and falls through to re-add under lock.
    obj.path.write_bytes(b"tampered")
    readded = store.add_file(src, verify=True)
    assert readded.sha256 == obj.sha256
    # Re-add restored the correct content.
    assert store.verify_object(obj.sha256) is True


def test_add_bytes_dedup_fast_path_without_verify(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.add_bytes(b"payload")
    second = store.add_bytes(b"payload", verify=False)
    assert second.sha256 == first.sha256
    assert second.size_bytes == first.size_bytes


def test_get_object_returns_none_for_missing_hash(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get_object("0" * 64) is None


# --------------------------------------------------------------------------- #
# materialize modes
# --------------------------------------------------------------------------- #


def test_materialize_symlink_default(tmp_path: Path) -> None:
    store = _store(tmp_path)
    obj = store.add_bytes(b"hello")
    dest = tmp_path / "out" / "model.bin"

    result = store.materialize(obj.sha256, dest)
    assert result == dest
    assert dest.is_symlink()
    assert dest.read_bytes() == b"hello"


def test_materialize_copy_mode(tmp_path: Path) -> None:
    store = _store(tmp_path)
    obj = store.add_bytes(b"hello")
    dest = tmp_path / "out" / "model.bin"

    store.materialize(obj.sha256, dest, use_symlink=False)
    assert dest.is_file() and not dest.is_symlink()
    assert dest.read_bytes() == b"hello"


def test_materialize_overwrite_false_raises_when_dest_exists(tmp_path: Path) -> None:
    store = _store(tmp_path)
    obj = store.add_bytes(b"hello")
    dest = tmp_path / "out.bin"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"existing")

    with pytest.raises(CASError, match="overwrite=False"):
        store.materialize(obj.sha256, dest, overwrite=False)


def test_materialize_overwrites_existing_dest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    obj = store.add_bytes(b"fresh")
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"stale")

    store.materialize(obj.sha256, dest, use_symlink=False, overwrite=True)
    assert dest.read_bytes() == b"fresh"


def test_materialize_verify_false_emits_warning_and_skips_check(tmp_path: Path, caplog) -> None:
    store = _store(tmp_path)
    obj = store.add_bytes(b"hello")
    dest = tmp_path / "out.bin"

    import logging

    with caplog.at_level(logging.WARNING):
        store.materialize(obj.sha256, dest, use_symlink=False, verify=False)
    assert any("verify=False" in r.message for r in caplog.records)
    assert dest.read_bytes() == b"hello"


def test_materialize_quarantines_corrupt_object(tmp_path: Path) -> None:
    store = _store(tmp_path)
    obj = store.add_bytes(b"hello")
    # Tamper the stored object so the read-time verify fails.
    obj.path.write_bytes(b"corrupt-bytes")

    dest = tmp_path / "out.bin"
    with pytest.raises(CASError, match="quarantined"):
        store.materialize(obj.sha256, dest, verify=True)

    quarantine = tmp_path / "cas" / "quarantine"
    assert quarantine.is_dir()
    assert list(quarantine.iterdir())  # the corrupt object was moved here
    assert not obj.path.exists()  # original removed


# --------------------------------------------------------------------------- #
# gc
# --------------------------------------------------------------------------- #


def test_gc_dry_run_lists_without_deleting(tmp_path: Path) -> None:
    store = _store(tmp_path)
    keep = store.add_bytes(b"keep")
    drop = store.add_bytes(b"drop")

    would_delete = store.gc({keep.sha256}, dry_run=True)
    assert drop.sha256 in would_delete
    assert keep.sha256 not in would_delete
    # Nothing actually deleted.
    assert store.has_object(drop.sha256)


def test_gc_deletes_unreferenced_objects(tmp_path: Path) -> None:
    store = _store(tmp_path)
    keep = store.add_bytes(b"keep")
    drop = store.add_bytes(b"drop")

    deleted = store.gc({keep.sha256.upper()}, dry_run=False)  # case-insensitive
    assert drop.sha256 in deleted
    assert not store.has_object(drop.sha256)
    assert store.has_object(keep.sha256)


def test_gc_skips_non_directory_entries(tmp_path: Path) -> None:
    store = _store(tmp_path)
    obj = store.add_bytes(b"keep")
    # A stray file directly under objects/ must be ignored (not a prefix dir).
    (store.objects_dir / "stray.txt").write_text("ignore me")

    deleted = store.gc({obj.sha256}, dry_run=False)
    assert deleted == []
    assert store.has_object(obj.sha256)


# --------------------------------------------------------------------------- #
# gc_quarantine
# --------------------------------------------------------------------------- #


def _make_quarantine_file(store: ArtifactStore, name: str, size: int, age_seconds: float) -> Path:
    qdir = store.root / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    p = qdir / name
    p.write_bytes(b"x" * size)
    mtime = time.time() - age_seconds
    os.utime(p, (mtime, mtime))
    return p


def test_gc_quarantine_no_dir_returns_zeroed_report(tmp_path: Path) -> None:
    store = _store(tmp_path)
    report = store.gc_quarantine()
    assert report == {
        "deleted": [],
        "retained": [],
        "total_size_before": 0,
        "total_size_after": 0,
    }


def test_gc_quarantine_age_based_cleanup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _make_quarantine_file(store, "old", size=10, age_seconds=10 * 86400)
    _make_quarantine_file(store, "new", size=10, age_seconds=60)

    report = store.gc_quarantine(max_age_seconds=7 * 86400, dry_run=False)
    assert "old" in report["deleted"]
    assert "new" in report["retained"]
    assert not (store.root / "quarantine" / "old").exists()
    assert (store.root / "quarantine" / "new").exists()


def test_gc_quarantine_size_based_cleanup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Two recent files; cap total size so the oldest is evicted by size policy.
    _make_quarantine_file(store, "older", size=100, age_seconds=120)
    _make_quarantine_file(store, "newer", size=100, age_seconds=10)

    report = store.gc_quarantine(max_age_seconds=7 * 86400, max_size_bytes=150, dry_run=False)
    # Oldest-first eviction drops "older", keeps "newer" under the 150B cap.
    assert "older" in report["deleted"]
    assert "newer" in report["retained"]


def test_gc_quarantine_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _make_quarantine_file(store, "old", size=10, age_seconds=10 * 86400)

    report = store.gc_quarantine(max_age_seconds=7 * 86400, dry_run=True)
    assert "old" in report["deleted"]
    assert (store.root / "quarantine" / "old").exists()  # dry run kept it
    assert report["total_size_before"] == 10


# --------------------------------------------------------------------------- #
# CASFileLock
# --------------------------------------------------------------------------- #


def test_file_lock_acquire_release_roundtrip(tmp_path: Path) -> None:
    lock = CASFileLock(tmp_path / "a.lock", timeout=1.0)
    assert lock.acquire() is True
    assert lock.lock_path.exists()
    lock.release()
    assert not lock.lock_path.exists()


def test_file_lock_times_out_when_held(tmp_path: Path) -> None:
    held = CASFileLock(tmp_path / "b.lock", timeout=0.2)
    assert held.acquire() is True
    # Fresh (non-stale) timestamp written by the holder → contender times out.
    contender = CASFileLock(tmp_path / "b.lock", timeout=0.2)
    assert contender.acquire() is False
    held.release()


def test_file_lock_breaks_stale_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "c.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a clearly-stale timestamp (> timeout*2 in the past).
    lock_path.write_text(str(time.time() - 1000))

    lock = CASFileLock(lock_path, timeout=0.2)
    assert lock.acquire() is True  # stale lock removed, then acquired
    lock.release()


def test_file_lock_times_out_on_unparseable_lock_body(tmp_path: Path) -> None:
    lock_path = tmp_path / "garbage.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Non-numeric body: stale-detection float() raises, the contender cannot
    # decide it is stale, and acquisition times out.
    lock_path.write_text("not-a-timestamp")

    contender = CASFileLock(lock_path, timeout=0.2)
    assert contender.acquire() is False


def test_file_lock_context_manager_raises_on_timeout(tmp_path: Path) -> None:
    holder = CASFileLock(tmp_path / "d.lock", timeout=0.2)
    holder.acquire()
    with pytest.raises(TimeoutError, match="Could not acquire CAS lock"):
        with CASFileLock(tmp_path / "d.lock", timeout=0.2):
            pass
    holder.release()


# --------------------------------------------------------------------------- #
# _fsync_parent_directory cross-platform branches
# --------------------------------------------------------------------------- #


def test_fsync_parent_noop_on_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cas_store, "_IS_WINDOWS", True)
    # Should return immediately without touching the filesystem fd APIs.
    cas_store._fsync_parent_directory(tmp_path / "x")


def test_fsync_parent_noop_without_o_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cas_store, "_IS_WINDOWS", False)
    monkeypatch.setattr(cas_store, "_HAS_O_DIRECTORY", False)
    cas_store._fsync_parent_directory(tmp_path / "x")


def test_fsync_parent_swallows_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cas_store, "_IS_WINDOWS", False)
    monkeypatch.setattr(cas_store, "_HAS_O_DIRECTORY", True)

    def _boom(*args, **kwargs):
        raise OSError("directory fsync unsupported")

    monkeypatch.setattr(cas_store.os, "open", _boom)
    # OSError from the directory fsync must be swallowed (consistency still holds
    # via the atomic rename), not propagated.
    cas_store._fsync_parent_directory(tmp_path / "x")
