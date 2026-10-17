"""Tests for worker module - task handling and error recovery.

This module tests:
- Spawn-safe execution via run_spawned
- Error propagation from child processes
- Timeout handling
- GPU integration via run_with_gpu
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.runtime.worker import (
    SpawnError,
    _worker_entry,
    run_spawned,
    run_with_gpu,
)

# --- Test Helper Functions (module-level for pickle compatibility) ---


def simple_function(x: int, y: int) -> int:
    """Simple function for testing."""
    return x + y


def failing_function() -> None:
    """Function that always raises."""
    raise ValueError("Intentional test failure")


def slow_function(delay: float = 0.0) -> str:
    """Function that sleeps then returns."""
    time.sleep(delay)
    return "completed"


def gpu_function(device_id: int, data: str) -> dict:
    """Simulated GPU function that receives device_id."""
    return {"device_id": device_id, "processed": data.upper()}


def greet_function(name: str, greeting: str = "Hello") -> str:
    """Greeting function for testing kwargs."""
    return f"{greeting}, {name}!"


def void_function() -> None:
    """Function returning None for testing."""
    pass


def get_pid_function() -> int:
    """Get current process ID."""
    return os.getpid()


def get_executable_function() -> str:
    """Get Python executable path."""
    import sys

    return sys.executable


def no_args_function() -> str:
    """Function with no arguments."""
    return "no args"


def large_return_function() -> list:
    """Function returning large data."""
    return list(range(10000))


def large_binary_function() -> bytes:
    """Return enough data to require the parent to drain the result pipe."""
    return b"0123456789abcdef" * 131072


def large_error_function() -> None:
    """Raise an error whose traceback cannot fit in a result pipe buffer."""
    raise ValueError("large-error:" + "x" * 1048576 + ":end")


def exit_without_result_function(code: int) -> None:
    """Exit without running the worker's result publisher."""
    os._exit(code)


def unpicklable_result_function():
    """Return an object that cannot be transported to the parent."""
    return threading.Lock()


def fork_descendant_function(mode: str, ready_path: str, stop_path: str) -> None:
    """Keep descriptors inherited by fork open until the fixture releases them."""
    descendant = os.fork()
    if descendant == 0:
        ready = Path(ready_path)
        ready.with_suffix(".writing").write_text(str(os.getpid()), encoding="utf-8")
        os.replace(ready.with_suffix(".writing"), ready)
        while not Path(stop_path).exists():
            time.sleep(0.01)
        os._exit(0)
    deadline = time.monotonic() + 5
    while not Path(ready_path).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Fixture descendant did not become ready")
        time.sleep(0.01)
    if mode == "timeout":
        time.sleep(60)
    os._exit(0 if mode == "clean" else 17)


def inherited_descriptor_scenario(mode: str, ready_path: str, stop_path: str) -> None:
    """Run in an externally bounded process group, without caller thread leaks."""
    # Initialize multiprocessing's persistent resource tracker before FD counts.
    assert run_spawned(no_args_function, timeout=5) == "no args"
    existing_children = {child.pid for child in mp.active_children()}
    existing_threads = {thread.ident for thread in threading.enumerate()}
    fd_root = Path("/dev/fd")
    before_fds = len(list(fd_root.iterdir())) if fd_root.is_dir() else None
    expected = {"clean": "did not return a result", "error": "exited with code 17", "timeout": "timed out after 2.0s"}
    try:
        with pytest.raises(SpawnError, match=expected[mode]):
            run_spawned(fork_descendant_function, mode, ready_path, stop_path, timeout=2.0 if mode == "timeout" else None)
        # The descendant still positively exists when the direct call returns.
        descendant = int(Path(ready_path).read_text(encoding="utf-8"))
        os.kill(descendant, 0)
        assert {child.pid for child in mp.active_children()} == existing_children
        assert {thread.ident for thread in threading.enumerate()} == existing_threads
        if before_fds is not None:
            assert len(list(fd_root.iterdir())) == before_fds
    finally:
        Path(stop_path).touch()


def partial_frame_function(fault: str, marker_path: str) -> bytes:
    """Interrupt a real partial result file before atomic publication."""
    from transformation_portal.runtime import worker

    class InterruptedPickler:
        def __init__(self, stream):
            self.stream = stream

        def dump(self, value):
            self.stream.write(b"x" * 65536)
            self.stream.flush()
            Path(marker_path).write_text("65536", encoding="utf-8")
            if fault == "exit":
                os._exit(17)
            time.sleep(60)

    worker.ForkingPickler = InterruptedPickler
    return b"x" * 2097152


def ignore_termination_function(pid_path: str) -> None:
    """Require the timeout cleanup to escalate beyond SIGTERM on POSIX."""
    import signal
    from pathlib import Path

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(pid_path).write_text(str(os.getpid()), encoding="utf-8")
    while True:
        time.sleep(1)


def binary_return_function() -> bytes:
    """Function returning binary data."""
    return b"\x00\x01\x02\x03"


def special_error_function() -> None:
    """Function that raises with special characters."""
    raise ValueError("Error with 'quotes' and \"double quotes\"")


def complex_return_function() -> dict:
    """Function returning complex nested data."""
    return {
        "list": [1, 2, 3],
        "nested": {"key": "value"},
        "tuple": (1, 2),
    }


# --- Test Classes ---


class TestWorkerEntry:
    """Tests for _worker_entry internal function."""

    def test_worker_entry_success(self) -> None:
        """_worker_entry puts success result in queue."""
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        # Use a simple function
        _worker_entry(simple_function, (2, 3), {}, result_queue)

        status, payload = result_queue.get(timeout=1)

        assert status == "ok"
        assert payload == 5

    def test_worker_entry_with_kwargs(self) -> None:
        """_worker_entry handles kwargs correctly."""
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        def fn_with_kwargs(a: int, b: int = 10) -> int:
            return a * b

        _worker_entry(fn_with_kwargs, (5,), {"b": 20}, result_queue)

        status, payload = result_queue.get(timeout=1)

        assert status == "ok"
        assert payload == 100

    def test_worker_entry_exception(self) -> None:
        """_worker_entry captures exceptions."""
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        _worker_entry(failing_function, (), {}, result_queue)

        status, payload = result_queue.get(timeout=1)

        assert status == "error"
        assert "ValueError" in payload
        assert "Intentional test failure" in payload


class TestRunSpawned:
    """Tests for run_spawned function."""

    def test_run_spawned_success(self) -> None:
        """run_spawned returns result from function."""
        result = run_spawned(simple_function, 10, 20)

        assert result == 30

    def test_run_spawned_with_kwargs(self) -> None:
        """run_spawned handles kwargs."""
        result = run_spawned(greet_function, "World", greeting="Hi")

        assert result == "Hi, World!"

    def test_run_spawned_exception(self) -> None:
        """run_spawned raises SpawnError on function exception."""
        with pytest.raises(SpawnError) as exc_info:
            run_spawned(failing_function)

        assert "Worker process failed" in str(exc_info.value)
        assert "ValueError" in str(exc_info.value)

    def test_run_spawned_timeout(self) -> None:
        """run_spawned raises SpawnError on timeout."""
        # Use a very short timeout to ensure it triggers
        with pytest.raises(SpawnError) as exc_info:
            run_spawned(slow_function, 10.0, timeout=0.1)

        # Check that the error message indicates a timeout occurred
        assert "timed out" in str(exc_info.value).lower()

    def test_run_spawned_returns_none(self) -> None:
        """run_spawned handles functions returning None."""
        result = run_spawned(void_function)

        assert result is None

    def test_run_spawned_complex_return(self) -> None:
        """run_spawned handles complex return types."""
        result = run_spawned(complex_return_function)

        assert result["list"] == [1, 2, 3]
        assert result["nested"]["key"] == "value"


class TestRunSpawnedProcessBehavior:
    """Tests for process behavior in run_spawned."""

    def test_spawn_context_used(self) -> None:
        """run_spawned uses spawn multiprocessing context."""
        # This test verifies spawn context is used by checking
        # that the function runs in a separate process
        child_pid = run_spawned(get_pid_function)
        parent_pid = os.getpid()

        assert child_pid != parent_pid

    def test_process_terminated_after_timeout(self) -> None:
        """Process is terminated after timeout."""
        existing_children = {child.pid for child in mp.active_children()}
        with pytest.raises(SpawnError):
            run_spawned(slow_function, 60.0, timeout=0.1)

        assert {child.pid for child in mp.active_children()} == existing_children

    def test_no_result_from_process(self) -> None:
        """SpawnError raised when process completes without result."""
        with pytest.raises(SpawnError, match="Worker process did not return a result"):
            run_spawned(exit_without_result_function, 0, timeout=5)

    def test_abnormal_exit_without_result(self) -> None:
        """A dead child reports its exit code instead of waiting for a result."""
        with pytest.raises(SpawnError, match="Worker process exited with code 17"):
            run_spawned(exit_without_result_function, 17, timeout=5)

    def test_result_wait_and_exit_share_timeout(self) -> None:
        """A 1.5s publication plus 1s shutdown cannot fit in a 2s budget."""
        from types import SimpleNamespace

        from transformation_portal.runtime import worker

        clock = [0.0]
        context = MagicMock()
        process = context.Process.return_value
        process.exitcode = None
        process.is_alive.side_effect = lambda: process.exitcode is None

        def join_process(timeout=None):
            if process.exitcode is not None:
                return
            if timeout is None or timeout >= 2.5:
                clock[0] += 2.5
                process.exitcode = 0
            else:
                clock[0] += timeout

        def terminate_process():
            process.exitcode = -15

        process.join.side_effect = join_process
        process.terminate.side_effect = terminate_process
        with (
            patch.object(worker, "mp", SimpleNamespace(get_context=lambda _: context)),
            patch.object(worker, "time", SimpleNamespace(monotonic=lambda: clock[0])),
            pytest.raises(SpawnError, match="timed out after 2.0s"),
        ):
            run_spawned(no_args_function, timeout=2.0)

        assert not process.is_alive()

    @pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM escalation and waitpid reaping")
    def test_timeout_reaps_child_that_ignores_termination(self, tmp_path) -> None:
        """Timeout cleanup also reaps a child that needs the kill fallback."""
        pid_path = tmp_path / "worker.pid"
        existing_children = {child.pid for child in mp.active_children()}

        with pytest.raises(SpawnError, match="timed out after 2.0s"):
            run_spawned(ignore_termination_function, str(pid_path), timeout=2.0)

        child_pid = int(pid_path.read_text(encoding="utf-8"))
        assert {child.pid for child in mp.active_children()} == existing_children
        with pytest.raises(ChildProcessError):
            os.waitpid(child_pid, os.WNOHANG)

    @pytest.mark.parametrize("fault, message", [("exit", "exited with code 17"), ("hang", "timed out after 5s")])
    def test_partial_result_frame_is_bounded(self, fault, message, tmp_path) -> None:
        """An interrupted partial publication cannot be read or trap the caller."""
        marker_path = tmp_path / "partial-frame.marker"
        existing_children = {child.pid for child in mp.active_children()}
        with pytest.raises(SpawnError, match=message):
            run_spawned(partial_frame_function, fault, str(marker_path), timeout=5)

        assert marker_path.read_text(encoding="utf-8") == "65536"
        assert {child.pid for child in mp.active_children()} == existing_children
        assert not any(thread.name == "tp-worker-result" for thread in threading.enumerate())

    def test_unpicklable_result_is_refused_and_receiver_closes(self) -> None:
        """Serialization failure retains the existing no-result error."""
        with pytest.raises(SpawnError, match="Worker process did not return a result"):
            run_spawned(unpicklable_result_function, timeout=5)

        assert not any(thread.name == "tp-worker-result" for thread in threading.enumerate())

    @pytest.mark.skipif(os.name != "posix", reason="POSIX fork descriptor inheritance and owned process groups")
    @pytest.mark.parametrize("mode", ["clean", "error", "timeout"])
    def test_descendant_inherited_descriptors_do_not_block_result(self, mode, tmp_path) -> None:
        """A descendant cannot hold the caller or leak its receiver after exit."""
        ready_path, stop_path = tmp_path / "descendant.ready", tmp_path / "descendant.stop"
        transport = tmp_path / "transport"
        transport.mkdir(mode=0o700)
        environment = os.environ.copy()
        environment["TMPDIR"] = str(transport)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).parent), str(Path(__file__).parents[2] / "src"), environment.get("PYTHONPATH", "")]
        )
        code = (
            "from test_worker import inherited_descriptor_scenario; import sys; inherited_descriptor_scenario(*sys.argv[1:])"
        )
        with (tmp_path / "scenario.log").open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", code, mode, str(ready_path), str(stop_path)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=15)
            finally:
                stop_path.touch()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Signal only this fixture's positively owned live group.
                    assert os.getpgid(process.pid) == process.pid
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                if ready_path.exists():
                    descendant = int(ready_path.read_text(encoding="utf-8"))
                    deadline = time.monotonic() + 5
                    while True:
                        try:
                            os.kill(descendant, 0)
                        except ProcessLookupError:
                            break
                        if time.monotonic() >= deadline:
                            pytest.fail("Fixture descendant did not close after its stop marker")
                        time.sleep(0.01)
        assert process.returncode == 0, (tmp_path / "scenario.log").read_text(encoding="utf-8")
        assert list(transport.iterdir()) == []

    @pytest.mark.parametrize("mode", ["success", "error", "unpicklable", "clean", "abnormal", "timeout"])
    def test_owned_temporary_result_is_removed(self, mode, tmp_path) -> None:
        """Complete, failed and interrupted results leave no transport files."""
        from transformation_portal.runtime import worker

        original_temporary_directory = worker.TemporaryDirectory

        def temporary_directory(**kwargs):
            return original_temporary_directory(dir=tmp_path, **kwargs)

        calls = {
            "success": (no_args_function, (), 5),
            "error": (failing_function, (), 5),
            "unpicklable": (unpicklable_result_function, (), 5),
            "clean": (exit_without_result_function, (0,), 5),
            "abnormal": (exit_without_result_function, (17,), 5),
            "timeout": (slow_function, (60,), 0.1),
        }
        function, args, timeout = calls[mode]
        with patch.object(worker, "TemporaryDirectory", side_effect=temporary_directory):
            if mode == "success":
                assert run_spawned(function, *args, timeout=timeout) == "no args"
            else:
                with pytest.raises(SpawnError):
                    run_spawned(function, *args, timeout=timeout)
        assert list(tmp_path.iterdir()) == []


class TestAtomicResultPublisher:
    """Transport failures must never expose an incomplete result."""

    def test_only_closed_complete_private_file_is_published(self, tmp_path) -> None:
        import pickle
        import stat

        from transformation_portal.runtime import worker

        result = tmp_path / "result.pickle"
        original_replace = os.replace
        original_pickler = worker.ForkingPickler
        streams = []
        value = ("ok", {"payload": b"x" * 2097152})

        def pickler(stream):
            streams.append(stream)
            return original_pickler(stream)

        def replace(partial, destination):
            assert not result.exists()
            assert streams and streams[0].closed
            if os.name == "posix":
                assert stat.S_IMODE(Path(partial).stat().st_mode) == 0o600
            with Path(partial).open("rb") as stream:
                assert pickle.load(stream) == value
            original_replace(partial, destination)

        with (
            patch.object(worker.os, "replace", side_effect=replace),
            patch.object(worker, "ForkingPickler", side_effect=pickler),
        ):
            worker._ResultFilePublisher(result).put(value)
        assert result.is_file()
        assert not result.with_suffix(".partial").exists()

    @pytest.mark.parametrize("failure", ["open", "write", "replace"])
    def test_disk_failure_never_publishes_partial_result(self, failure, tmp_path) -> None:
        from transformation_portal.runtime import worker

        private_directory = worker.TemporaryDirectory(dir=tmp_path)
        result = Path(private_directory.name) / "result.pickle"
        streams = []

        class FailingPickler:
            def __init__(self, stream):
                self.stream = stream
                streams.append(stream)

            def dump(self, value):
                self.stream.write(b"partial")
                raise OSError("injected disk-write failure")

        if failure == "write":
            target, attribute, options = worker, "ForkingPickler", {"side_effect": FailingPickler}
        else:
            target, attribute, options = worker.os, failure, {"side_effect": OSError("injected disk failure")}
        try:
            with patch.object(target, attribute, **options):
                worker._ResultFilePublisher(result).put(("ok", "value"))
            assert not result.exists()
            assert all(stream.closed for stream in streams)
        finally:
            private_directory.cleanup()
        assert list(tmp_path.iterdir()) == []


class TestRunWithGPU:
    """Tests for run_with_gpu function."""

    def test_run_with_gpu_acquires_slot(self) -> None:
        """run_with_gpu acquires GPU slot before running."""
        from transformation_portal.runtime.gpu_semaphore import GPUSemaphore

        # Create mock semaphore
        mock_semaphore = MagicMock(spec=GPUSemaphore)

        # Create mock slot context manager
        from transformation_portal.runtime.gpu_semaphore import GPUSlot

        mock_slot = GPUSlot(device_id=0)

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_slot)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_semaphore.acquire.return_value = mock_context

        # Mock run_spawned to capture args
        with patch("transformation_portal.runtime.worker.run_spawned") as mock_run:
            mock_run.return_value = {"result": "ok"}

            run_with_gpu(mock_semaphore, gpu_function, "test_data")

            # Verify semaphore was used
            mock_semaphore.acquire.assert_called_once()

            # Verify device_id was prepended
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == gpu_function  # Function
            assert call_args[0][1] == 0  # device_id from slot

    def test_run_with_gpu_releases_on_success(self) -> None:
        """run_with_gpu releases GPU slot after success."""
        from transformation_portal.runtime.gpu_semaphore import GPUSemaphore, GPUSlot

        mock_semaphore = MagicMock(spec=GPUSemaphore)
        mock_slot = GPUSlot(device_id=1)

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_slot)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_semaphore.acquire.return_value = mock_context

        with patch("transformation_portal.runtime.worker.run_spawned") as mock_run:
            mock_run.return_value = "result"

            run_with_gpu(mock_semaphore, gpu_function, "data")

            # Context manager __exit__ should have been called
            mock_context.__exit__.assert_called_once()

    def test_run_with_gpu_releases_on_failure(self) -> None:
        """run_with_gpu releases GPU slot after failure."""
        from transformation_portal.runtime.gpu_semaphore import GPUSemaphore, GPUSlot

        mock_semaphore = MagicMock(spec=GPUSemaphore)
        mock_slot = GPUSlot(device_id=2)

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_slot)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_semaphore.acquire.return_value = mock_context

        with patch("transformation_portal.runtime.worker.run_spawned") as mock_run:
            mock_run.side_effect = SpawnError("Test failure")

            with pytest.raises(SpawnError):
                run_with_gpu(mock_semaphore, gpu_function, "data")

            # Context manager __exit__ should still have been called
            mock_context.__exit__.assert_called_once()

    def test_run_with_gpu_passes_timeout(self) -> None:
        """run_with_gpu passes timeout to run_spawned."""
        from transformation_portal.runtime.gpu_semaphore import GPUSemaphore, GPUSlot

        mock_semaphore = MagicMock(spec=GPUSemaphore)
        mock_slot = GPUSlot(device_id=0)

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_slot)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_semaphore.acquire.return_value = mock_context

        with patch("transformation_portal.runtime.worker.run_spawned") as mock_run:
            mock_run.return_value = "result"

            run_with_gpu(mock_semaphore, gpu_function, "data", timeout=30.0)

            # Check timeout was passed
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("timeout") == 30.0


class TestSpawnError:
    """Tests for SpawnError exception class."""

    def test_spawn_error_is_runtime_error(self) -> None:
        """SpawnError inherits from RuntimeError."""
        assert issubclass(SpawnError, RuntimeError)

    def test_spawn_error_message(self) -> None:
        """SpawnError preserves error message."""
        error = SpawnError("Custom error message")

        assert str(error) == "Custom error message"

    def test_spawn_error_in_except_clause(self) -> None:
        """SpawnError can be caught in except clause."""
        try:
            raise SpawnError("Test")
        except SpawnError as e:
            assert "Test" in str(e)


class TestWorkerIsolation:
    """Tests for process isolation in worker module."""

    def test_spawn_context_isolates_state(self) -> None:
        """Each spawned process gets fresh state."""
        # Simply verify that running same function twice works
        result1 = run_spawned(simple_function, 1, 2)
        result2 = run_spawned(simple_function, 3, 4)

        assert result1 == 3
        assert result2 == 7

    def test_imports_fresh_in_child(self) -> None:
        """Child process gets fresh imports."""
        result = run_spawned(get_executable_function)

        # Result should be a path string
        assert isinstance(result, str)
        assert len(result) > 0


class TestRunSpawnedEdgeCases:
    """Tests for edge cases in run_spawned."""

    def test_empty_args_and_kwargs(self) -> None:
        """run_spawned works with no args/kwargs."""
        result = run_spawned(no_args_function)
        assert result == "no args"

    def test_large_return_value(self) -> None:
        """run_spawned handles large return values."""
        result = run_spawned(large_return_function)

        assert len(result) == 10000
        assert result[0] == 0
        assert result[-1] == 9999

    def test_large_binary_result_does_not_wait_for_child_exit(self) -> None:
        """The parent drains a real result larger than an OS pipe buffer."""
        result = run_spawned(large_binary_function, timeout=5)

        assert result == b"0123456789abcdef" * 131072

    def test_large_error_preserves_payload_without_deadlock(self) -> None:
        """Error propagation drains the complete traceback before joining."""
        with pytest.raises(SpawnError, match="Worker process failed: ValueError") as exc_info:
            run_spawned(large_error_function, timeout=5)

        assert "large-error:" + "x" * 1048576 + ":end" in str(exc_info.value)

    def test_binary_return_value(self) -> None:
        """run_spawned handles binary return values."""
        result = run_spawned(binary_return_function)

        assert result == b"\x00\x01\x02\x03"

    def test_exception_with_special_characters(self) -> None:
        """run_spawned handles exceptions with special characters."""
        with pytest.raises(SpawnError) as exc_info:
            run_spawned(special_error_function)

        error_str = str(exc_info.value)
        assert "quotes" in error_str
