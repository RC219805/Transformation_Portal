"""Containment of manifest logical paths in :mod:`storage.dataset_cas` (STOR-03).

A dataset manifest names files by ``logical_path`` and ``materialize`` joins it
under a caller-chosen directory. A manifest is untrusted input, so a path that
escapes that directory (``..``, absolute, or through a directory symlink
already inside the target) must be refused before anything is written.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from transformation_portal.storage.cas_store import ArtifactStore
from transformation_portal.storage.dataset_cas import DatasetCAS, DatasetCASError, DatasetFile, DatasetManifest

pytestmark = [pytest.mark.unit, pytest.mark.security]


def _dataset(tmp_path: Path) -> tuple[DatasetCAS, str]:
    cas = ArtifactStore(tmp_path / "cas")
    src = tmp_path / "payload.bin"
    src.write_bytes(b"dataset-payload")
    obj = cas.add_file(src)
    return DatasetCAS(cas, tmp_path / "datasets"), obj.sha256


def _manifest(sha: str, *logical_paths: str) -> DatasetManifest:
    return DatasetManifest(
        name="d",
        version="1",
        created_at="2026-01-01T00:00:00+00:00",
        files=[DatasetFile(logical_path=p, sha256=sha, size_bytes=15) for p in logical_paths],
    )


def test_materialize_writes_nested_logical_paths_inside_target(tmp_path: Path) -> None:
    dataset, sha = _dataset(tmp_path)
    target = tmp_path / "runtime" / "dataset"

    dataset.materialize(_manifest(sha, "train/a.bin", "b.bin"), target, use_symlinks=False)

    assert (target / "train" / "a.bin").read_bytes() == b"dataset-payload"
    assert (target / "b.bin").read_bytes() == b"dataset-payload"


@pytest.mark.parametrize("hostile", ["../../outside/victim.txt", "train/../../../outside/victim.txt"])
def test_materialize_rejects_parent_traversal(tmp_path: Path, hostile: str) -> None:
    dataset, sha = _dataset(tmp_path)
    victim = tmp_path / "outside" / "victim.txt"
    victim.parent.mkdir()
    victim.write_text("original")
    target = tmp_path / "runtime" / "dataset"

    with pytest.raises(DatasetCASError):
        dataset.materialize(_manifest(sha, "ok.bin", hostile), target, use_symlinks=False)

    assert victim.read_text() == "original"
    # Validation runs before any write: no partial materialization either.
    assert list(target.rglob("*")) == []


def test_materialize_rejects_absolute_logical_path(tmp_path: Path) -> None:
    dataset, sha = _dataset(tmp_path)
    absolute = tmp_path / "outside" / "absolute.bin"
    absolute.parent.mkdir()

    with pytest.raises(DatasetCASError):
        dataset.materialize(_manifest(sha, str(absolute)), tmp_path / "runtime", use_symlinks=False)

    assert not absolute.exists()


def test_materialize_rejects_directory_symlink_escape(tmp_path: Path) -> None:
    dataset, sha = _dataset(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "runtime"
    target.mkdir()
    os.symlink(outside, target / "link", target_is_directory=True)

    with pytest.raises(DatasetCASError):
        dataset.materialize(_manifest(sha, "link/victim.bin"), target, use_symlinks=False)

    assert not (outside / "victim.bin").exists()


@pytest.mark.parametrize("hostile", ["", ".", "a\x00b"])
def test_materialize_rejects_degenerate_logical_paths(tmp_path: Path, hostile: str) -> None:
    dataset, sha = _dataset(tmp_path)

    with pytest.raises(DatasetCASError):
        dataset.materialize(_manifest(sha, hostile), tmp_path / "runtime", use_symlinks=False)


def test_manifest_load_rejects_escaping_logical_path(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _manifest("0" * 64, "../escape.bin").save(path)

    with pytest.raises(DatasetCASError):
        DatasetManifest.load(path)


def test_ingest_files_rejects_escaping_logical_path_before_adding_objects(tmp_path: Path) -> None:
    dataset, _sha = _dataset(tmp_path)
    other = tmp_path / "other.bin"
    other.write_bytes(b"not-yet-ingested")
    objects_before = sorted(p.name for p in (tmp_path / "cas").rglob("*") if p.is_file())

    with pytest.raises(DatasetCASError):
        dataset.ingest_files("d", [("ok.bin", other), ("/etc/escape.bin", other)])

    assert sorted(p.name for p in (tmp_path / "cas").rglob("*") if p.is_file()) == objects_before
    assert dataset.get_manifest("d", "1.0.0") is None
