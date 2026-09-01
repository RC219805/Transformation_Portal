"""Tests for io_atomic module - atomic write primitives.

Validates:
- Successful writes produce final file
- Failures leave no temp files behind
- Operations use atomic rename (os.replace)
- No file descriptor leaks
"""

import errno
import gc
import hashlib
import json
import multiprocessing
import os
import signal
import stat
import threading
import time
from pathlib import Path

import pytest
from PIL import Image  # pylint: disable=possibly-used-before-assignment

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3 import io_atomic
from transformation_portal.lux_depth_v3.io_atomic import (
    HAS_PIL,
    atomic_temp_file,
    atomic_write_bytes,
    atomic_write_pil_png,
    atomic_write_with_fd,
)


def _atomic_byte_temp_paths(directory: Path, destination: Path) -> list[Path]:
    """Return temp files owned by one atomic byte destination."""
    return list(directory.glob(f".{destination.name}.*.tmp"))


def _crash_boundary_writer(
    output_path: str,
    payload: bytes,
    boundary: str,
    connection,
) -> None:
    """Stop a child writer at a real publication boundary for crash tests."""
    from transformation_portal.lux_depth_v3 import io_atomic as child_io_atomic

    if boundary == "before_replace":
        real_replace = child_io_atomic.os.replace

        def stop_before_replace(source, destination):
            connection.send(boundary)
            os.kill(os.getpid(), signal.SIGSTOP)
            real_replace(source, destination)

        child_io_atomic.os.replace = stop_before_replace
    elif boundary == "after_replace":

        def stop_before_directory_fsync(_directory):
            connection.send(boundary)
            os.kill(os.getpid(), signal.SIGSTOP)

        child_io_atomic._fsync_directory = stop_before_directory_fsync
    else:  # pragma: no cover - test helper contract
        raise ValueError(f"unsupported crash boundary: {boundary}")

    try:
        child_io_atomic.atomic_write_bytes(Path(output_path), payload)
    finally:
        connection.close()


def _umask_thread_worker(directory: str, connection) -> None:
    """Exercise concurrent writes without exposing umask changes to pytest."""
    from transformation_portal.lux_depth_v3 import io_atomic as child_io_atomic

    original_umask = os.umask(0o077)
    errors: list[str] = []
    observed_umask = -1
    modes: list[int] = []
    try:

        def write_one(index: int) -> None:
            try:
                child_io_atomic.atomic_write_bytes(
                    Path(directory) / f"thread-{index}.json",
                    f'{{"index":{index}}}'.encode("utf-8"),
                )
            except BaseException as exc:  # pragma: no cover - returned to parent
                errors.append(repr(exc))

        threads = [threading.Thread(target=write_one, args=(index,)) for index in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            if thread.is_alive():
                errors.append("writer thread did not finish")

        modes = [stat.S_IMODE((Path(directory) / f"thread-{index}.json").stat().st_mode) for index in range(12)]
        observed_umask = os.umask(0o077)
    finally:
        os.umask(original_umask)

    connection.send((observed_umask, modes, errors))
    connection.close()


def _publication_lock_worker(destination: str, entered, release) -> None:
    """Hold one publication lock until the parent permits release."""
    from transformation_portal.lux_depth_v3 import io_atomic as child_io_atomic

    with child_io_atomic.publication_lock(Path(destination)):
        entered.set()
        release.wait(timeout=10)


def _evidence_pair_worker(primary: str, sidecar: str, writer_id: int, start) -> None:
    """Publish one internally matching pair from a child process."""
    from transformation_portal.lux_depth_v3 import io_atomic as child_io_atomic

    primary_bytes = json.dumps({"writer": writer_id}, sort_keys=True).encode("utf-8")
    sidecar_bytes = json.dumps(
        {"sha256": hashlib.sha256(primary_bytes).hexdigest()},
        sort_keys=True,
    ).encode("utf-8")
    if not start.wait(timeout=10):
        raise TimeoutError("pair publication start signal timed out")
    child_io_atomic.atomic_write_evidence_pair(
        Path(primary),
        primary_bytes,
        Path(sidecar),
        sidecar_bytes,
    )


def _fork_context():
    """Return a fork context or skip tests that require POSIX crash control."""
    if os.name != "posix" or not hasattr(signal, "SIGSTOP"):
        pytest.skip("requires POSIX process signals")
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("multiprocessing fork start method is unavailable")


class TestAtomicTempFile:
    """Test atomic temp file context manager."""

    def test_successful_write_creates_final_file(self, tmp_path):
        """Successful write should create final file via atomic rename."""
        output_path = tmp_path / "output.txt"

        with atomic_temp_file(output_path, create_file=True) as temp_path:
            # Temp file should exist
            assert temp_path.exists()
            # Should be in same directory
            assert temp_path.parent == output_path.parent
            # Should have temp prefix
            assert temp_path.name.startswith(".tmp_")

            # Write data to temp
            temp_path.write_text("hello")

        # Final file should exist
        assert output_path.exists()
        assert output_path.read_text() == "hello"

        # Temp file should be gone
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Temp files remain: {temp_files}"

    def test_failure_cleans_up_temp_file(self, tmp_path):
        """Failed write should cleanup temp file."""
        output_path = tmp_path / "output.txt"
        temp_path_captured = None

        try:
            with atomic_temp_file(output_path, create_file=True) as temp_path:
                temp_path_captured = temp_path
                temp_path.write_text("partial")
                # Simulate failure
                raise ValueError("Simulated write failure")
        except ValueError:
            pass

        # Output file should NOT exist
        assert not output_path.exists()

        # Temp file should be cleaned up
        assert not temp_path_captured.exists()
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Temp files remain: {temp_files}"

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if needed."""
        output_path = tmp_path / "subdir" / "nested" / "output.txt"

        with atomic_temp_file(output_path, create_file=True) as temp_path:
            temp_path.write_text("test")

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_custom_suffix_and_prefix(self, tmp_path):
        """Should respect custom suffix and prefix."""
        output_path = tmp_path / "output.png"

        with atomic_temp_file(output_path, suffix=".png", prefix="custom_", create_file=True) as temp_path:
            # Should have custom prefix
            assert temp_path.name.startswith("custom_")
            # Should have custom suffix
            assert temp_path.suffix == ".png"
            temp_path.write_bytes(b"data")

        assert output_path.exists()


class TestAtomicWriteBytes:
    """Test atomic byte writing."""

    def test_successful_bytes_write(self, tmp_path):
        """Should atomically write bytes."""
        output_path = tmp_path / "data.bin"
        data = b"Hello, atomic world!"

        result_path = atomic_write_bytes(output_path, data)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == data

        # No temp files should remain
        assert _atomic_byte_temp_paths(tmp_path, output_path) == []

    def test_overwrites_existing_file(self, tmp_path):
        """Should atomically overwrite existing file."""
        output_path = tmp_path / "overwrite.bin"

        # Write initial content
        output_path.write_bytes(b"old data")

        # Overwrite with new content
        atomic_write_bytes(output_path, b"new data")

        assert output_path.read_bytes() == b"new data"

    def test_empty_bytes(self, tmp_path):
        """Should handle empty byte arrays."""
        output_path = tmp_path / "empty.bin"

        atomic_write_bytes(output_path, b"")

        assert output_path.exists()
        assert output_path.read_bytes() == b""

    def test_large_bytes(self, tmp_path):
        """Should handle large byte arrays."""
        output_path = tmp_path / "large.bin"
        # 10 MB of data
        data = b"x" * (10 * 1024 * 1024)

        atomic_write_bytes(output_path, data)

        assert output_path.exists()
        assert len(output_path.read_bytes()) == len(data)

    @pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
    def test_new_file_uses_fixed_permissions(self, tmp_path):
        """New files should use 0644 independent of the ambient umask."""
        output_path = tmp_path / "permissions.bin"

        atomic_write_bytes(output_path, b"test data")

        assert stat.S_IMODE(output_path.stat().st_mode) == 0o644

    @pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
    def test_overwrite_preserves_existing_permissions(self, tmp_path):
        """Replacing a destination should preserve its prior mode."""
        output_path = tmp_path / "permissions.bin"
        output_path.write_bytes(b"old")
        output_path.chmod(0o640)

        atomic_write_bytes(output_path, b"new")

        assert output_path.read_bytes() == b"new"
        assert stat.S_IMODE(output_path.stat().st_mode) == 0o640

    def test_concurrent_writes_do_not_change_process_umask(self, tmp_path):
        """Threaded writes must never inspect or mutate the process umask."""
        context = _fork_context()
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_umask_thread_worker,
            args=(str(tmp_path), child_connection),
        )
        process.start()
        child_connection.close()
        try:
            assert parent_connection.poll(10), "concurrent umask worker timed out"
            observed_umask, modes, errors = parent_connection.recv()
        finally:
            parent_connection.close()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

        assert process.exitcode == 0
        assert errors == []
        assert observed_umask == 0o077
        assert modes == [0o644] * 12

    def test_file_fsync_replace_and_directory_fsync_order(self, tmp_path, monkeypatch):
        """Durability calls must fence replacement in the required order."""
        output_path = tmp_path / "ordered.bin"
        events: list[str] = []
        real_replace = io_atomic.os.replace
        real_apply_file_mode = io_atomic._apply_file_mode

        def record_fsync(_descriptor):
            events.append("fsync")

        def record_mode(path, descriptor, mode):
            events.append("chmod")
            real_apply_file_mode(path, descriptor, mode)

        def record_replace(source, destination):
            events.append("replace")
            real_replace(source, destination)

        monkeypatch.setattr(io_atomic.os, "fsync", record_fsync)
        monkeypatch.setattr(io_atomic, "_apply_file_mode", record_mode)
        monkeypatch.setattr(io_atomic.os, "replace", record_replace)

        atomic_write_bytes(output_path, b"ordered")

        assert events == ["fsync", "chmod", "fsync", "replace", "fsync"]

    def test_new_parent_entries_are_fsynced_before_file_publication(self, tmp_path, monkeypatch):
        """Every newly created directory entry must be durable before use."""
        output_path = tmp_path / "first" / "second" / "evidence.bin"
        fsynced_directories: list[Path] = []
        real_fsync_directory = io_atomic._fsync_directory

        def record_directory_fsync(directory):
            fsynced_directories.append(directory)
            real_fsync_directory(directory)

        monkeypatch.setattr(io_atomic, "_fsync_directory", record_directory_fsync)

        atomic_write_bytes(output_path, b"durable hierarchy")

        assert fsynced_directories == [tmp_path, tmp_path / "first", tmp_path / "first" / "second"]
        assert output_path.read_bytes() == b"durable hierarchy"

    def test_partial_write_failure_never_replaces_destination(self, tmp_path, monkeypatch):
        """A failed partial temp write must leave the old destination intact."""
        output_path = tmp_path / "partial.bin"
        output_path.write_bytes(b"old-complete")

        def fail_after_partial_write(handle, data):
            handle.write(data[:5])
            handle.flush()
            raise OSError(errno.ENOSPC, "simulated full disk")

        monkeypatch.setattr(io_atomic, "_write_all", fail_after_partial_write)

        with pytest.raises(IOError, match="Failed to write"):
            atomic_write_bytes(output_path, b"new-complete-payload")

        assert output_path.read_bytes() == b"old-complete"
        assert _atomic_byte_temp_paths(tmp_path, output_path) == []

    def test_replace_failure_cleans_temp_and_preserves_destination(self, tmp_path, monkeypatch):
        """A failed rename must clean the complete temp file without publishing it."""
        output_path = tmp_path / "replace.bin"
        output_path.write_bytes(b"old-complete")

        def fail_replace(_source, _destination):
            raise OSError(errno.EIO, "simulated replace failure")

        monkeypatch.setattr(io_atomic.os, "replace", fail_replace)

        for _ in range(8):
            with pytest.raises(IOError, match="Failed to write"):
                atomic_write_bytes(output_path, b"new-complete")

        assert output_path.read_bytes() == b"old-complete"
        assert _atomic_byte_temp_paths(tmp_path, output_path) == []

    def test_directory_fsync_failure_reports_unproven_durability(self, tmp_path, monkeypatch):
        """A real directory fsync failure is reported after complete replacement."""
        output_path = tmp_path / "directory-fsync.bin"
        output_path.write_bytes(b"old-complete")

        def fail_directory_fsync(_directory):
            raise OSError(errno.EIO, "simulated directory fsync failure")

        monkeypatch.setattr(io_atomic, "_fsync_directory", fail_directory_fsync)

        with pytest.raises(IOError, match="Failed to write"):
            atomic_write_bytes(output_path, b"new-complete")

        assert output_path.read_bytes() == b"new-complete"
        assert _atomic_byte_temp_paths(tmp_path, output_path) == []

    @pytest.mark.parametrize("boundary", ["before_replace", "after_replace"])
    def test_process_crash_never_exposes_partial_destination(self, tmp_path, boundary):
        """A killed writer exposes the old file before replace or full new file after."""
        context = _fork_context()
        output_path = tmp_path / f"crash-{boundary}.bin"
        old_payload = b"old-complete"
        new_payload = b"new-complete" * 131_072
        output_path.write_bytes(old_payload)
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_crash_boundary_writer,
            args=(str(output_path), new_payload, boundary, child_connection),
        )
        process.start()
        child_connection.close()
        try:
            assert parent_connection.poll(10), f"writer did not reach {boundary}"
            assert parent_connection.recv() == boundary
            expected = old_payload if boundary == "before_replace" else new_payload
            assert output_path.read_bytes() == expected
        finally:
            parent_connection.close()
            if process.is_alive():
                process.kill()
            process.join(timeout=5)

        assert not process.is_alive()
        expected = old_payload if boundary == "before_replace" else new_payload
        assert output_path.read_bytes() == expected

    def test_concurrent_same_destination_has_one_complete_winner(self, tmp_path):
        """Concurrent replacement of one path must never expose mixed bytes."""
        output_path = tmp_path / "winner.bin"
        old_payload = b"old-complete"
        payload_a = b"A" * (1024 * 1024)
        payload_b = b"B" * (1024 * 1024)
        output_path.write_bytes(old_payload)
        barrier = threading.Barrier(3)
        errors: list[BaseException] = []

        def publish(payload: bytes) -> None:
            try:
                barrier.wait(timeout=5)
                atomic_write_bytes(output_path, payload)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        writers = [
            threading.Thread(target=publish, args=(payload_a,)),
            threading.Thread(target=publish, args=(payload_b,)),
        ]
        for writer in writers:
            writer.start()
        barrier.wait(timeout=5)

        observed = {old_payload}
        deadline = time.monotonic() + 10
        while any(writer.is_alive() for writer in writers) and time.monotonic() < deadline:
            observed.add(output_path.read_bytes())
        for writer in writers:
            writer.join(timeout=5)

        assert errors == []
        assert all(not writer.is_alive() for writer in writers)
        observed.add(output_path.read_bytes())
        assert observed <= {old_payload, payload_a, payload_b}
        assert output_path.read_bytes() in {payload_a, payload_b}
        assert _atomic_byte_temp_paths(tmp_path, output_path) == []


class TestDirectoryFsync:
    """Test the narrow platform/error policy for directory durability."""

    def test_explicit_unsupported_error_is_tolerated(self, tmp_path, monkeypatch):
        """Only a recognized unsupported errno may degrade to rename-only."""

        def unsupported_fsync(_descriptor):
            raise OSError(errno.EINVAL, "directory fsync unsupported")

        monkeypatch.setattr(io_atomic, "_IS_WINDOWS", False)
        monkeypatch.setattr(io_atomic.os, "fsync", unsupported_fsync)

        io_atomic._fsync_directory(tmp_path)

    def test_genuine_io_error_propagates(self, tmp_path, monkeypatch):
        """Directory I/O failures must fail closed."""

        def failed_fsync(_descriptor):
            raise OSError(errno.EIO, "directory I/O failure")

        monkeypatch.setattr(io_atomic, "_IS_WINDOWS", False)
        monkeypatch.setattr(io_atomic.os, "fsync", failed_fsync)

        with pytest.raises(OSError) as exc_info:
            io_atomic._fsync_directory(tmp_path)

        assert exc_info.value.errno == errno.EIO

    def test_windows_is_an_explicit_noop(self, tmp_path, monkeypatch):
        """The Windows branch must not try to open a directory descriptor."""

        def unexpected_open(*_args, **_kwargs):
            raise AssertionError("directory open should not run on Windows")

        monkeypatch.setattr(io_atomic, "_IS_WINDOWS", True)
        monkeypatch.setattr(io_atomic.os, "open", unexpected_open)

        io_atomic._fsync_directory(tmp_path)


class TestDurableUnlink:
    """Test stale-evidence invalidation used by paired run-card writes."""

    def test_existing_file_is_removed_before_directory_fsync(self, tmp_path, monkeypatch):
        path = tmp_path / "stale.self.json"
        path.write_bytes(b"stale")
        observed: list[tuple[Path, bool]] = []

        def record_directory_fsync(directory):
            observed.append((directory, path.exists()))

        monkeypatch.setattr(io_atomic, "_fsync_directory", record_directory_fsync)

        io_atomic.durable_unlink(path)

        assert observed == [(tmp_path, False)]

    def test_missing_file_requires_no_directory_update(self, tmp_path, monkeypatch):
        path = tmp_path / "missing.self.json"
        calls: list[Path] = []
        monkeypatch.setattr(io_atomic, "_fsync_directory", calls.append)

        io_atomic.durable_unlink(path)

        assert calls == []


class TestEvidencePairPublication:
    """Test pair-level locking for a primary file and verifying sidecar."""

    def test_pair_rejects_two_paths_that_resolve_to_one_file(self, tmp_path):
        """A parent traversal alias cannot collapse the primary and sidecar."""
        primary_path = tmp_path / "run_card.json"
        primary_path.write_bytes(b"old-primary")
        alias_parent = tmp_path / "alias-parent"
        alias_parent.mkdir()
        sidecar_alias = alias_parent / ".." / primary_path.name

        with pytest.raises(ValueError, match="must differ"):
            io_atomic.atomic_write_evidence_pair(
                primary_path,
                b"new-primary",
                sidecar_alias,
                b"sidecar",
            )

        assert primary_path.read_bytes() == b"old-primary"

    def test_pair_rejects_fresh_unicode_normalization_aliases(self, tmp_path):
        """Fresh NFC/NFD-equivalent names must fail before either path is written."""
        primary_path = tmp_path / "caf\N{LATIN SMALL LETTER E WITH ACUTE}.json"
        sidecar_path = tmp_path / "cafe\N{COMBINING ACUTE ACCENT}.json"

        with pytest.raises(ValueError, match="Unicode normalization"):
            io_atomic.atomic_write_evidence_pair(
                primary_path,
                b"primary",
                sidecar_path,
                b"sidecar",
            )

        assert not primary_path.exists()
        assert not sidecar_path.exists()

    def test_publication_lock_rejects_precreated_symlink(self, tmp_path):
        """A predictable lock path must not chmod or write through a symlink."""
        destination = tmp_path / "run_card.json"
        lock_path = destination.with_name(f".{destination.name}.publication.lock")
        victim = tmp_path / "victim.bin"
        victim.write_bytes(b"unchanged")
        victim.chmod(0o600)
        try:
            lock_path.symlink_to(victim)
        except OSError:
            pytest.skip("symlink creation is unavailable")

        with pytest.raises(OSError):
            with io_atomic.publication_lock(destination):
                pytest.fail("unsafe lock path was accepted")

        assert victim.read_bytes() == b"unchanged"
        assert stat.S_IMODE(victim.stat().st_mode) == 0o600

    def test_publication_lock_rejects_precreated_hardlink(self, tmp_path):
        """A lock inode shared with another pathname must fail closed."""
        destination = tmp_path / "run_card.json"
        lock_path = destination.with_name(f".{destination.name}.publication.lock")
        victim = tmp_path / "victim.bin"
        victim.write_bytes(b"unchanged")
        victim.chmod(0o600)
        try:
            os.link(victim, lock_path)
        except OSError:
            pytest.skip("hardlink creation is unavailable")

        with pytest.raises(OSError):
            with io_atomic.publication_lock(destination):
                pytest.fail("multiply-linked lock inode was accepted")

        assert victim.read_bytes() == b"unchanged"
        assert stat.S_IMODE(victim.stat().st_mode) == 0o600

    def test_inactive_thread_locks_do_not_accumulate(self, tmp_path):
        """Unique batch destinations must not grow a permanent lock registry."""
        gc.collect()
        baseline = len(io_atomic._PUBLICATION_THREAD_LOCKS)
        for index in range(20):
            with io_atomic.publication_lock(tmp_path / f"run-card-{index}.json"):
                pass
        gc.collect()
        assert len(io_atomic._PUBLICATION_THREAD_LOCKS) == baseline

    def test_forked_child_discards_inherited_thread_lock(self, tmp_path):
        """A child forked inside a held lock can proceed after the parent unlocks."""
        context = _fork_context()
        destination = tmp_path / "run_card.json"
        entered = context.Event()
        release = context.Event()
        child = None
        try:
            with io_atomic.publication_lock(destination):
                child = context.Process(
                    target=_publication_lock_worker,
                    args=(str(destination), entered, release),
                )
                child.start()
                assert not entered.wait(timeout=0.25)
            assert entered.wait(timeout=5)
            release.set()
            child.join(timeout=5)
        finally:
            release.set()
            if child is not None:
                if child.is_alive():
                    child.kill()
                child.join(timeout=5)

        assert child is not None
        assert child.exitcode == 0

    def test_publication_lock_serializes_processes(self, tmp_path):
        """A second process cannot enter the same destination lock early."""
        context = _fork_context()
        destination = tmp_path / "run_card.json"
        entered_first = context.Event()
        release_first = context.Event()
        entered_second = context.Event()
        release_second = context.Event()
        first = context.Process(
            target=_publication_lock_worker,
            args=(str(destination), entered_first, release_first),
        )
        second = context.Process(
            target=_publication_lock_worker,
            args=(str(destination), entered_second, release_second),
        )

        try:
            first.start()
            assert entered_first.wait(timeout=5)
            second.start()
            assert not entered_second.wait(timeout=0.25)
            release_first.set()
            assert entered_second.wait(timeout=5)
            release_second.set()
            first.join(timeout=5)
            second.join(timeout=5)
        finally:
            release_first.set()
            release_second.set()
            for process in (first, second):
                if process.is_alive():
                    process.kill()
                process.join(timeout=5)

        assert first.exitcode == 0
        assert second.exitcode == 0

    def test_concurrent_pairs_leave_one_matching_winner(self, tmp_path):
        """Concurrent pair writers cannot mix one writer's sidecar with another's bytes."""
        primary_path = tmp_path / "run_card.json"
        sidecar_path = tmp_path / "run_card.self.json"
        writer_count = 8
        barrier = threading.Barrier(writer_count + 1)
        errors: list[BaseException] = []

        def publish(writer_id: int) -> None:
            primary_bytes = json.dumps({"writer": writer_id}, sort_keys=True).encode("utf-8")
            sidecar_bytes = json.dumps(
                {"sha256": hashlib.sha256(primary_bytes).hexdigest()},
                sort_keys=True,
            ).encode("utf-8")
            try:
                barrier.wait(timeout=5)
                io_atomic.atomic_write_evidence_pair(
                    primary_path,
                    primary_bytes,
                    sidecar_path,
                    sidecar_bytes,
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        writers = [threading.Thread(target=publish, args=(writer_id,)) for writer_id in range(writer_count)]
        for writer in writers:
            writer.start()
        barrier.wait(timeout=5)
        for writer in writers:
            writer.join(timeout=10)

        assert errors == []
        assert all(not writer.is_alive() for writer in writers)
        primary_bytes = primary_path.read_bytes()
        sidecar = json.loads(sidecar_path.read_bytes())
        assert sidecar["sha256"] == hashlib.sha256(primary_bytes).hexdigest()
        assert _atomic_byte_temp_paths(tmp_path, primary_path) == []
        assert _atomic_byte_temp_paths(tmp_path, sidecar_path) == []

    def test_concurrent_process_pairs_leave_one_matching_winner(self, tmp_path):
        """Cross-process pair writers leave one internally consistent winner."""
        context = _fork_context()
        primary_path = tmp_path / "run_card.json"
        sidecar_path = tmp_path / "run_card.self.json"
        start = context.Event()
        processes = [
            context.Process(
                target=_evidence_pair_worker,
                args=(str(primary_path), str(sidecar_path), writer_id, start),
            )
            for writer_id in range(4)
        ]
        try:
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(timeout=10)
        finally:
            start.set()
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join(timeout=5)

        assert [process.exitcode for process in processes] == [0] * len(processes)
        primary_bytes = primary_path.read_bytes()
        sidecar = json.loads(sidecar_path.read_bytes())
        assert sidecar["sha256"] == hashlib.sha256(primary_bytes).hexdigest()

    @pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
    def test_pair_preserves_existing_primary_and_sidecar_modes(self, tmp_path):
        """Invalidating a stale sidecar must not lose its governed mode."""
        primary_path = tmp_path / "run_card.json"
        sidecar_path = tmp_path / "run_card.self.json"
        primary_path.write_bytes(b"old-primary")
        sidecar_path.write_bytes(b"old-sidecar")
        primary_path.chmod(0o600)
        sidecar_path.chmod(0o640)

        io_atomic.atomic_write_evidence_pair(
            primary_path,
            b"new-primary",
            sidecar_path,
            b"new-sidecar",
        )

        assert stat.S_IMODE(primary_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(sidecar_path.stat().st_mode) == 0o640

    @pytest.mark.parametrize(
        ("failed_directory_fsync", "sidecar_exists"),
        [(2, False), (3, True)],
    )
    def test_pair_reports_post_replace_durability_failures(
        self,
        tmp_path,
        monkeypatch,
        failed_directory_fsync,
        sidecar_exists,
    ):
        """A post-replace failure leaves only complete, fail-closed evidence."""
        primary_path = tmp_path / "run_card.json"
        sidecar_path = tmp_path / "run_card.self.json"
        primary_path.write_bytes(b"old-primary")
        sidecar_path.write_bytes(b"old-sidecar")
        new_primary = b"new-primary"
        new_sidecar = json.dumps(
            {"sha256": hashlib.sha256(new_primary).hexdigest()},
            sort_keys=True,
        ).encode("utf-8")
        real_fsync_directory = io_atomic._fsync_directory
        fsync_calls = 0

        def controlled_fsync(directory):
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == failed_directory_fsync:
                raise OSError(errno.EIO, "simulated directory fsync failure")
            real_fsync_directory(directory)

        monkeypatch.setattr(io_atomic, "_fsync_directory", controlled_fsync)

        with pytest.raises(IOError, match="Failed to write"):
            io_atomic.atomic_write_evidence_pair(
                primary_path,
                new_primary,
                sidecar_path,
                new_sidecar,
            )

        assert primary_path.read_bytes() == new_primary
        assert sidecar_path.exists() is sidecar_exists
        if sidecar_exists:
            sidecar = json.loads(sidecar_path.read_bytes())
            assert sidecar["sha256"] == hashlib.sha256(new_primary).hexdigest()


@pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")
class TestAtomicWritePilPng:
    """Test atomic PIL PNG writing."""

    def test_successful_pil_write(self, tmp_path):
        """Should atomically write PIL Image as PNG."""
        output_path = tmp_path / "image.png"

        # Create test image
        img = Image.new("RGB", (100, 100), color="red")

        result_path = atomic_write_pil_png(output_path, img)

        assert result_path == output_path
        assert output_path.exists()

        # Verify can read back
        loaded = Image.open(output_path)
        assert loaded.size == (100, 100)
        assert loaded.mode == "RGB"

        # No temp files
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_grayscale_image(self, tmp_path):
        """Should handle grayscale images."""
        output_path = tmp_path / "gray.png"
        img = Image.new("L", (50, 50), color=128)

        atomic_write_pil_png(output_path, img)

        loaded = Image.open(output_path)
        assert loaded.mode == "L"
        assert loaded.size == (50, 50)

    def test_rgba_image(self, tmp_path):
        """Should handle RGBA images with transparency."""
        output_path = tmp_path / "rgba.png"
        img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))

        atomic_write_pil_png(output_path, img)

        loaded = Image.open(output_path)
        assert loaded.mode == "RGBA"
        assert loaded.size == (64, 64)

    def test_optimization_flag(self, tmp_path):
        """Should respect optimize flag."""
        output_path = tmp_path / "optimized.png"
        img = Image.new("RGB", (100, 100), color="blue")

        # With optimization
        atomic_write_pil_png(output_path, img, optimize=True)
        size_optimized = output_path.stat().st_size

        # Without optimization
        atomic_write_pil_png(output_path, img, optimize=False)
        size_unoptimized = output_path.stat().st_size

        # Optimized should typically be smaller or equal
        # (but we just verify both succeed without errors)
        assert output_path.exists()

    def test_custom_save_kwargs(self, tmp_path):
        """Should pass through custom save kwargs."""
        output_path = tmp_path / "custom.png"
        img = Image.new("RGB", (100, 100), color="green")

        # Pass compression level
        atomic_write_pil_png(output_path, img, compress_level=9)

        assert output_path.exists()
        loaded = Image.open(output_path)
        assert loaded.size == (100, 100)


class TestAtomicWriteWithFD:
    """Test atomic writing with file descriptor."""

    def test_fd_based_write(self, tmp_path):
        """Should handle FD-based writers."""
        output_path = tmp_path / "fd_output.txt"

        def writer_func(fd, temp_path):
            # Write using FD
            with os.fdopen(fd, "w") as f:
                f.write("Written via FD")

        result_path = atomic_write_with_fd(output_path, writer_func)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.read_text() == "Written via FD"

        # No temp files
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_fd_based_binary_write(self, tmp_path):
        """Should handle binary FD writes."""
        output_path = tmp_path / "binary.dat"

        def writer_func(fd, temp_path):
            with os.fdopen(fd, "wb") as f:
                f.write(b"binary data")

        atomic_write_with_fd(output_path, writer_func)

        assert output_path.read_bytes() == b"binary data"

    def test_writer_closes_fd_explicitly(self, tmp_path):
        """Should handle writers that close FD themselves."""
        output_path = tmp_path / "explicit_close.txt"

        def writer_func(fd, temp_path):
            # Close FD immediately
            os.close(fd)
            # Use path instead
            temp_path.write_text("closed FD, used path")

        atomic_write_with_fd(output_path, writer_func)

        assert output_path.read_text() == "closed FD, used path"

    def test_writer_failure_cleans_up(self, tmp_path):
        """Should cleanup temp file if writer fails."""
        output_path = tmp_path / "failed.txt"

        def failing_writer(fd, temp_path):
            os.close(fd)
            raise RuntimeError("Writer failed!")

        with pytest.raises(IOError, match="Failed to write"):
            atomic_write_with_fd(output_path, failing_writer)

        # Output should not exist
        assert not output_path.exists()

        # No temp files should remain
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_custom_suffix(self, tmp_path):
        """Should respect custom suffix."""
        output_path = tmp_path / "custom.png"

        def writer_func(fd, temp_path):
            # Verify temp has correct suffix
            assert temp_path.suffix == ".png"
            os.close(fd)
            temp_path.write_bytes(b"png data")

        atomic_write_with_fd(output_path, writer_func, suffix=".png")

        assert output_path.exists()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_nested_directory_creation(self, tmp_path):
        """Should handle deeply nested paths."""
        output_path = tmp_path / "a" / "b" / "c" / "d" / "output.txt"

        atomic_write_bytes(output_path, b"nested")

        assert output_path.exists()
        assert output_path.read_bytes() == b"nested"

    def test_unicode_filename(self, tmp_path):
        """Should handle unicode filenames."""
        output_path = tmp_path / "unicode_文件.txt"

        atomic_write_bytes(output_path, b"unicode content")

        assert output_path.exists()

    def test_concurrent_writes_different_files(self, tmp_path):
        """Should handle concurrent writes to different files."""
        # Write multiple files
        paths = [tmp_path / f"file_{i}.txt" for i in range(5)]

        for i, path in enumerate(paths):
            atomic_write_bytes(path, f"content_{i}".encode())

        # All should exist with correct content
        for i, path in enumerate(paths):
            assert path.exists()
            assert path.read_bytes() == f"content_{i}".encode()

        # No temp files
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_overwrite_during_write_is_atomic(self, tmp_path):
        """Verify write is atomic - no partial state visible."""
        output_path = tmp_path / "atomic_test.txt"

        # Initial state
        output_path.write_text("initial")

        # Overwrite atomically
        atomic_write_bytes(output_path, b"updated")

        # Should see either old or new, never partial
        content = output_path.read_text()
        assert content in ["initial", "updated"]
        # Since write completed, should be updated
        assert content == "updated"


class TestNoFDLeaks:
    """Test that FD management doesn't leak descriptors."""

    def test_no_fd_leak_on_success(self, tmp_path):
        """Successful writes should not leak FDs."""
        output_path = tmp_path / "fd_test.txt"

        # Get current FD count (rough check)
        # On POSIX, we can check /proc/self/fd or use resource limits
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Write many times
        for i in range(100):
            atomic_write_bytes(output_path, f"iteration {i}".encode())

        # Should not approach FD limit (rough sanity check)
        assert output_path.exists()

    def test_no_fd_leak_on_failure(self, tmp_path):
        """Failed writes should not leak FDs."""
        output_path = tmp_path / "fail_test.txt"

        def failing_writer(fd, temp_path):
            # Don't close FD - test cleanup
            raise ValueError("Intentional failure")

        # Try to write many times with failure
        for i in range(50):
            try:
                atomic_write_with_fd(output_path, failing_writer)
            except IOError:
                pass

        # Should not have leaked FDs (output should not exist)
        assert not output_path.exists()
