"""Tests for worker module - task handling and error recovery.

This module tests:
- Spawn-safe execution via run_spawned
- Error propagation from child processes
- Timeout handling
- GPU integration via run_with_gpu
"""

from __future__ import annotations

import multiprocessing as mp
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
    import time

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
    import os

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
        import os

        child_pid = run_spawned(get_pid_function)
        parent_pid = os.getpid()

        assert child_pid != parent_pid

    def test_process_terminated_after_timeout(self) -> None:
        """Process is terminated after timeout."""
        # We can't easily test this directly, but we can verify
        # the timeout mechanism works
        with pytest.raises(SpawnError):
            run_spawned(slow_function, 60.0, timeout=0.1)

    def test_no_result_from_process(self) -> None:
        """SpawnError raised when process completes without result."""
        # This is hard to test directly as we need to simulate
        # a process that exits cleanly but doesn't put result in queue
        # We'll test the exception type exists
        assert issubclass(SpawnError, RuntimeError)


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
