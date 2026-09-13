"""Bounded source validation for immutable artifact generation staging."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO contract")
def test_generation_fifo_source_is_rejected_without_waiting_for_a_writer(tmp_path: Path):
    source = tmp_path / "artifact.fifo"
    os.mkfifo(source)
    program = """
from pathlib import Path
import sys
from transformation_portal.orchestrator.artifact_store.generation import _snapshot
from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
try:
    _snapshot(Path(sys.argv[1]), Path(sys.argv[2]), 1024)
except ArtifactStoreError:
    raise SystemExit(0)
raise SystemExit(2)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(source), str(tmp_path / "snapshot")], capture_output=True, timeout=5, check=False
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "snapshot").exists()


def test_attempt_cleanup_removes_only_its_recorded_private_directory(tmp_path: Path):
    from transformation_portal.orchestrator.artifact_store.generation_cleanup import remove_owned_attempt_directory

    requested = tmp_path / "requested"
    requested.mkdir()
    keep = requested / "user-output.txt"
    keep.write_text("keep this existing output")
    attempt = requested / ".tp-attempts" / ("job_test-" + "a" * 32)
    attempt.mkdir(parents=True)
    (attempt / "result.txt").write_text("owned transient copy")
    remove_owned_attempt_directory(job_id="job_test", output_root=str(attempt), requested_output_root=str(requested))
    assert not attempt.exists()
    assert keep.read_text() == "keep this existing output"


@pytest.mark.parametrize("target_kind", ["requested_root", "sibling", "symlink"])
def test_attempt_cleanup_rejects_unowned_or_substituted_roots(tmp_path: Path, target_kind: str):
    from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
    from transformation_portal.orchestrator.artifact_store.generation_cleanup import remove_owned_attempt_directory

    requested = tmp_path / "requested"
    requested.mkdir()
    victim = tmp_path / "user-data"
    victim.mkdir()
    (victim / "keep.txt").write_text("preserve")
    if target_kind == "requested_root":
        candidate = requested
    elif target_kind == "sibling":
        candidate = victim
    else:
        candidate = requested / ".tp-attempts" / ("job_test-" + "a" * 32)
        candidate.parent.mkdir()
        candidate.symlink_to(victim, target_is_directory=True)
    with pytest.raises(ArtifactStoreError):
        remove_owned_attempt_directory(job_id="job_test", output_root=str(candidate), requested_output_root=str(requested))
    assert (victim / "keep.txt").read_text() == "preserve"


def test_attempt_cleanup_remains_anchored_when_requested_ancestor_is_swapped(tmp_path: Path, monkeypatch):
    import shutil

    from transformation_portal.orchestrator.artifact_store.generation_cleanup import remove_owned_attempt_directory

    requested = tmp_path / "requested"
    attempt_name = "job_test-" + "a" * 32
    attempt = requested / ".tp-attempts" / attempt_name
    attempt.mkdir(parents=True)
    (attempt / "owned.txt").write_text("remove this owned copy")
    victim = tmp_path / "victim"
    victim_attempt = victim / ".tp-attempts" / attempt_name
    victim_attempt.mkdir(parents=True)
    (victim_attempt / "keep.txt").write_text("preserve victim")
    saved = tmp_path / "original-requested"
    real_rmtree = shutil.rmtree

    def swap_before_removal(path, **kwargs):
        requested.rename(saved)
        requested.symlink_to(victim, target_is_directory=True)
        return real_rmtree(path, **kwargs)

    swap_before_removal.avoids_symlink_attacks = real_rmtree.avoids_symlink_attacks
    monkeypatch.setattr(shutil, "rmtree", swap_before_removal)
    remove_owned_attempt_directory(job_id="job_test", output_root=str(attempt), requested_output_root=str(requested))
    assert (victim_attempt / "keep.txt").read_text() == "preserve victim"
    assert not (saved / ".tp-attempts" / attempt_name).exists()


@pytest.mark.parametrize("swap_before_open", [False, True])
def test_generation_snapshot_never_follows_a_substituted_ancestor(tmp_path: Path, monkeypatch, swap_before_open: bool):
    from transformation_portal.orchestrator.artifact_store import _filesystem
    from transformation_portal.orchestrator.artifact_store.generation import _snapshot

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "artifact.txt").write_bytes(b"owned bytes")
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "artifact.txt").write_bytes(b"private victim bytes")
    saved = tmp_path / "original-source"
    real_open = os.open
    swapped = False

    def swapped_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        trigger = "source" if swap_before_open else "artifact.txt"
        if path == trigger and not swapped:
            source_root.rename(saved)
            source_root.symlink_to(victim, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(_filesystem.os, "open", swapped_open)
    snapshot = tmp_path / "snapshot"
    if swap_before_open:
        with pytest.raises(OSError):
            _snapshot(source_root / "artifact.txt", snapshot, 1024)
        assert not snapshot.exists()
    else:
        size, _ = _snapshot(source_root / "artifact.txt", snapshot, 1024)
        assert size == len(b"owned bytes")
        assert snapshot.read_bytes() == b"owned bytes"
    assert swapped
    assert (victim / "artifact.txt").read_bytes() == b"private victim bytes"
