"""Tests for ProcessExecutor - process lifecycle management.

This module tests:
- Process spawn, monitor, and termination
- Timeout handling and cleanup
- Result queue communication
- Error propagation from child processes
- Batch execution
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.runtime.process_executor import (
    ProcessExecutor,
    ProcessExecutorError,
    ProcessResult,
    ProcessTask,
)

# --- Mock Node Classes for Testing ---


class SimpleNode:
    """Simple node that returns inputs."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        """Return inputs with optional prefix."""
        return {k: f"{self.prefix}{v}" for k, v in inputs.items()}


class FailingNode:
    """Node that raises an exception."""

    def __init__(self, error_message: str = "Intentional failure"):
        self.error_message = error_message

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        """Raise an exception."""
        raise RuntimeError(self.error_message)


class SlowNode:
    """Node that takes time to complete."""

    def __init__(self, sleep_seconds: float = 0.0):
        self.sleep_seconds = sleep_seconds

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        """Execute with optional delay."""
        import time

        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return {"completed": True}


# --- Test Classes ---


class TestProcessTask:
    """Tests for ProcessTask dataclass."""

    def test_task_creation(self) -> None:
        """ProcessTask stores all required fields."""
        task = ProcessTask(
            node_cls=SimpleNode,
            node_kwargs={"prefix": "test_"},
            inputs={"key": "value"},
            sandbox_config={
                "node_id": "test_001",
                "workspace_root": "/tmp/ws",
                "cas_root": "/tmp/cas",
            },
        )

        assert task.node_cls == SimpleNode
        assert task.node_kwargs == {"prefix": "test_"}
        assert task.inputs == {"key": "value"}
        assert task.sandbox_config["node_id"] == "test_001"

    def test_task_with_gpu_config(self) -> None:
        """ProcessTask can include GPU configuration."""
        task = ProcessTask(
            node_cls=SimpleNode,
            node_kwargs={},
            inputs={},
            sandbox_config={
                "node_id": "gpu_task",
                "workspace_root": "/tmp/ws",
                "cas_root": "/tmp/cas",
                "gpu_id": 0,
            },
        )

        assert task.sandbox_config["gpu_id"] == 0


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_success_property_true(self) -> None:
        """ProcessResult.success is True when no error."""
        result = ProcessResult(
            outputs={"sha": "abc123"},
            manifest={"node_id": "test"},
        )

        assert result.success is True
        assert result.error is None
        assert result.traceback is None

    def test_success_property_false(self) -> None:
        """ProcessResult.success is False when error present."""
        result = ProcessResult(
            outputs={},
            error="Something failed",
            traceback="Traceback (most recent call last)...",
        )

        assert result.success is False
        assert result.error == "Something failed"
        assert "Traceback" in result.traceback

    def test_default_empty_outputs(self) -> None:
        """ProcessResult has empty dict defaults."""
        result = ProcessResult()

        assert result.outputs == {}
        assert result.manifest == {}


class TestProcessExecutor:
    """Tests for ProcessExecutor class."""

    def test_executor_initialization_default(self) -> None:
        """ProcessExecutor initializes with defaults."""
        executor = ProcessExecutor()

        assert executor.timeout is None
        assert executor.execution_count == 0

    def test_executor_initialization_with_timeout(self) -> None:
        """ProcessExecutor accepts timeout parameter."""
        executor = ProcessExecutor(timeout=30.0)

        assert executor.timeout == 30.0

    def test_executor_initialization_spawn_method(self) -> None:
        """ProcessExecutor uses spawn start method."""
        executor = ProcessExecutor(start_method="spawn")

        # Verify context was created (we can't easily inspect it)
        assert executor.ctx is not None

    def test_execution_count_increments(self) -> None:
        """Execution count increments with each run."""
        executor = ProcessExecutor()

        assert executor.execution_count == 0

        # Mock the actual execution to avoid subprocess
        with patch.object(executor.ctx, "Queue") as mock_queue, patch.object(executor.ctx, "Process") as mock_process:
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = False
            mock_proc.exitcode = 0
            mock_process.return_value = mock_proc

            q = MagicMock()
            q.empty.return_value = False
            q.get_nowait.return_value = ProcessResult(outputs={"test": "value"})
            mock_queue.return_value = q

            task = ProcessTask(
                node_cls=SimpleNode,
                node_kwargs={},
                inputs={},
                sandbox_config={"node_id": "test", "workspace_root": "/tmp", "cas_root": "/tmp"},
            )

            executor.run(task)
            assert executor.execution_count == 1

            executor.run(task)
            assert executor.execution_count == 2


class TestProcessExecutorTimeout:
    """Tests for timeout handling in ProcessExecutor."""

    def test_timeout_terminates_process(self) -> None:
        """Process is terminated when timeout expires."""
        executor = ProcessExecutor(timeout=0.1)

        with patch.object(executor.ctx, "Queue") as mock_queue, patch.object(executor.ctx, "Process") as mock_process:
            mock_proc = MagicMock()
            # Simulate a process that ignores terminate() and only dies on kill()
            alive = {"value": True}
            mock_proc.is_alive.side_effect = lambda: alive["value"]
            mock_proc.kill.side_effect = lambda: alive.update(value=False)
            mock_process.return_value = mock_proc

            q = MagicMock()
            q.empty.return_value = True  # The child never produces a result
            mock_queue.return_value = q

            task = ProcessTask(
                node_cls=SlowNode,
                node_kwargs={"sleep_seconds": 10.0},
                inputs={},
                sandbox_config={"node_id": "slow", "workspace_root": "/tmp", "cas_root": "/tmp"},
            )

            result = executor.run(task, timeout=0.1)

            assert result.success is False
            assert "timeout" in result.error.lower()
            mock_proc.terminate.assert_called()

    def test_effective_timeout_from_parameter(self) -> None:
        """Per-call timeout overrides default."""
        executor = ProcessExecutor(timeout=60.0)

        with patch.object(executor.ctx, "Queue") as mock_queue, patch.object(executor.ctx, "Process") as mock_process:
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = False
            mock_proc.exitcode = 0
            mock_process.return_value = mock_proc

            q = MagicMock()
            q.empty.return_value = False
            q.get_nowait.return_value = ProcessResult(outputs={})
            mock_queue.return_value = q

            task = ProcessTask(
                node_cls=SimpleNode,
                node_kwargs={},
                inputs={},
                sandbox_config={"node_id": "test", "workspace_root": "/tmp", "cas_root": "/tmp"},
            )

            executor.run(task, timeout=5.0)

            # join should have been called with the override timeout
            mock_proc.join.assert_called()


class TestProcessExecutorErrorHandling:
    """Tests for error handling in ProcessExecutor."""

    def test_process_exit_nonzero(self) -> None:
        """Non-zero exit code is reported as error."""
        executor = ProcessExecutor()

        with patch.object(executor.ctx, "Queue") as mock_queue, patch.object(executor.ctx, "Process") as mock_process:
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = False
            mock_proc.exitcode = 1  # Non-zero exit
            mock_process.return_value = mock_proc

            q = MagicMock()
            q.empty.return_value = True  # No result in queue
            mock_queue.return_value = q

            task = ProcessTask(
                node_cls=FailingNode,
                node_kwargs={},
                inputs={},
                sandbox_config={"node_id": "fail", "workspace_root": "/tmp", "cas_root": "/tmp"},
            )

            result = executor.run(task)

            assert result.success is False
            assert "exit" in result.error.lower()
            assert "1" in result.error

    def test_process_no_result(self) -> None:
        """Process completion without result is an error."""
        executor = ProcessExecutor()

        with patch.object(executor.ctx, "Queue") as mock_queue, patch.object(executor.ctx, "Process") as mock_process:
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = False
            mock_proc.exitcode = 0  # Clean exit but no result
            mock_process.return_value = mock_proc

            q = MagicMock()
            q.empty.return_value = True  # Queue is empty
            mock_queue.return_value = q

            task = ProcessTask(
                node_cls=SimpleNode,
                node_kwargs={},
                inputs={},
                sandbox_config={"node_id": "empty", "workspace_root": "/tmp", "cas_root": "/tmp"},
            )

            result = executor.run(task)

            assert result.success is False
            assert "without result" in result.error.lower()

    def test_cleanup_on_exception(self) -> None:
        """Process is cleaned up even if exception occurs."""
        executor = ProcessExecutor()

        with patch.object(executor.ctx, "Queue") as mock_queue, patch.object(executor.ctx, "Process") as mock_process:
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = True  # Still alive at cleanup
            mock_process.return_value = mock_proc

            mock_queue.return_value = MagicMock()

            # Make join raise to simulate unexpected error
            mock_proc.join.side_effect = [Exception("Unexpected"), None]

            task = ProcessTask(
                node_cls=SimpleNode,
                node_kwargs={},
                inputs={},
                sandbox_config={"node_id": "cleanup", "workspace_root": "/tmp", "cas_root": "/tmp"},
            )

            try:
                executor.run(task)
            except Exception:
                pass

            # terminate should have been called in finally block
            mock_proc.terminate.assert_called()


class LargeOutputNode:
    """Node that finishes instantly with an output far larger than a pipe buffer."""

    def __init__(self, size: int = 1_000_000):
        self.size = size

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        return {"blob": "x" * self.size}


def _run_with_watchdog(executor: ProcessExecutor, task: ProcessTask, watchdog_s: float) -> ProcessResult:
    """Run ``executor.run`` in a thread and fail (rather than hang) if it never returns."""
    box: Dict[str, Any] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            box["result"] = executor.run(task)
        finally:
            done.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    if not done.wait(watchdog_s):
        for child in mp.active_children():  # do not leave a deadlocked child behind
            child.kill()
        thread.join(timeout=5)
        pytest.fail(f"ProcessExecutor.run did not return within {watchdog_s}s")
    return box["result"]


class TestProcessExecutorRealChild:
    """Real spawned children (T-05): results larger than the pipe buffer must not deadlock."""

    @pytest.mark.parametrize("size", [70_000, 1_000_000])
    def test_large_result_is_received_without_timeout(self, tmp_path: Path, size: int) -> None:
        executor = ProcessExecutor(timeout=20.0)
        task = ProcessTask(
            node_cls=LargeOutputNode,
            node_kwargs={"size": size},
            inputs={},
            sandbox_config={
                "node_id": f"large_{size}",
                "workspace_root": tmp_path / "ws",
                "cas_root": tmp_path / "cas",
            },
        )

        started = time.monotonic()
        result = _run_with_watchdog(executor, task, watchdog_s=30.0)
        elapsed = time.monotonic() - started

        assert result.error is None, result.error
        assert result.outputs["blob"] == "x" * size
        # Without draining before join, the run only ends via the 20s executor timeout.
        assert elapsed < 15.0

    def test_no_timeout_large_result_returns(self, tmp_path: Path) -> None:
        """With the default timeout=None the old join-first flow hung forever."""
        executor = ProcessExecutor()
        task = ProcessTask(
            node_cls=LargeOutputNode,
            node_kwargs={"size": 500_000},
            inputs={},
            sandbox_config={
                "node_id": "large_no_timeout",
                "workspace_root": tmp_path / "ws",
                "cas_root": tmp_path / "cas",
            },
        )

        result = _run_with_watchdog(executor, task, watchdog_s=20.0)

        assert result.success is True
        assert len(result.outputs["blob"]) == 500_000


class TestProcessExecutorBatch:
    """Tests for batch execution in ProcessExecutor."""

    def test_run_batch_sequential(self) -> None:
        """run_batch executes tasks sequentially."""
        executor = ProcessExecutor()

        with patch.object(executor, "run") as mock_run:
            mock_run.return_value = ProcessResult(outputs={"result": "ok"})

            tasks = [
                ProcessTask(
                    node_cls=SimpleNode,
                    node_kwargs={},
                    inputs={"idx": i},
                    sandbox_config={"node_id": f"batch_{i}", "workspace_root": "/tmp", "cas_root": "/tmp"},
                )
                for i in range(3)
            ]

            results = executor.run_batch(tasks)

            assert len(results) == 3
            assert mock_run.call_count == 3

    def test_run_batch_with_timeout(self) -> None:
        """run_batch passes timeout to each run."""
        executor = ProcessExecutor()

        with patch.object(executor, "run") as mock_run:
            mock_run.return_value = ProcessResult(outputs={})

            tasks = [
                ProcessTask(
                    node_cls=SimpleNode,
                    node_kwargs={},
                    inputs={},
                    sandbox_config={"node_id": f"t_{i}", "workspace_root": "/tmp", "cas_root": "/tmp"},
                )
                for i in range(2)
            ]

            executor.run_batch(tasks, timeout=30.0)

            for call in mock_run.call_args_list:
                assert call.kwargs.get("timeout") == 30.0

    def test_run_batch_empty_list(self) -> None:
        """run_batch handles empty task list."""
        executor = ProcessExecutor()

        results = executor.run_batch([])

        assert results == []


class TestProcessExecutorProcessNaming:
    """Tests for process naming in ProcessExecutor."""

    def test_process_named_by_node_id(self) -> None:
        """Spawned process is named after node_id."""
        executor = ProcessExecutor()

        with patch.object(executor.ctx, "Queue") as mock_queue, patch.object(executor.ctx, "Process") as mock_process:
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = False
            mock_proc.exitcode = 0
            mock_process.return_value = mock_proc

            q = MagicMock()
            q.empty.return_value = False
            q.get_nowait.return_value = ProcessResult(outputs={})
            mock_queue.return_value = q

            task = ProcessTask(
                node_cls=SimpleNode,
                node_kwargs={},
                inputs={},
                sandbox_config={"node_id": "my_special_node", "workspace_root": "/tmp", "cas_root": "/tmp"},
            )

            executor.run(task)

            # Check that Process was called with the correct name
            call_kwargs = mock_process.call_args.kwargs
            assert "sandbox-my_special_node" in call_kwargs.get("name", "")


class TestProcessResultIntegration:
    """Integration-style tests using ProcessResult patterns."""

    def test_result_with_manifest(self) -> None:
        """ProcessResult correctly stores manifest data."""
        manifest = {
            "node_id": "test_node",
            "workspace": "/tmp/workspace/test_node",
            "inputs": {"sha1": "/path/to/input"},
            "outputs": {"/path/to/output": "sha2"},
            "metrics": {
                "inputs_materialized": 1,
                "outputs_persisted": 1,
                "bytes_read": 1024,
                "bytes_written": 2048,
            },
        }

        result = ProcessResult(
            outputs={"result_sha": "abc123"},
            manifest=manifest,
        )

        assert result.manifest["node_id"] == "test_node"
        assert result.manifest["metrics"]["bytes_written"] == 2048

    def test_result_error_with_traceback(self) -> None:
        """ProcessResult stores full traceback for debugging."""
        result = ProcessResult(
            outputs={},
            error="ValueError: Invalid input",
            traceback="""Traceback (most recent call last):
  File "worker.py", line 10, in run
    raise ValueError("Invalid input")
ValueError: Invalid input""",
        )

        assert result.success is False
        assert "ValueError" in result.error
        assert "worker.py" in result.traceback
        assert "line 10" in result.traceback
