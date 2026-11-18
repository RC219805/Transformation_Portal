"""Tests for execution_graph.distributed_executor — local and Ray backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.execution_graph.distributed_executor import (
    DistributedDAGExecutor,
    DistributedExecutorError,
    ExecutionConfig,
    create_executor,
)
from transformation_portal.execution_graph.scheduler import PriorityDAGScheduler

pytestmark = pytest.mark.unit


def _simple_scheduler(return_value: dict | None = None) -> PriorityDAGScheduler:
    """Return a scheduler with a single node that returns return_value."""
    s = PriorityDAGScheduler()
    node = MagicMock()
    node.run.return_value = return_value or "result"
    s.add_node("a", node)
    return s


class TestCreateExecutor:
    def test_local_backend_returns_executor(self):
        """create_executor with 'local' backend returns DistributedDAGExecutor."""
        s = _simple_scheduler()
        executor = create_executor(s, backend="local")
        assert isinstance(executor, DistributedDAGExecutor)

    def test_default_backend_is_local(self):
        """create_executor without backend defaults to local."""
        s = _simple_scheduler()
        executor = create_executor(s)
        assert executor.config.backend == "local"


class TestDistributedDAGExecutorLocal:
    def test_run_local_delegates_to_scheduler(self):
        """run() with local backend calls scheduler.run() and returns results."""
        s = _simple_scheduler(return_value="node_output")
        executor = DistributedDAGExecutor(s, config=ExecutionConfig(backend="local"))
        results = executor.run()
        assert results["a"] == "node_output"

    def test_run_with_inputs_passes_through(self):
        """Inputs dict is forwarded to scheduler.run()."""
        s = PriorityDAGScheduler()
        node = MagicMock()
        node.run.return_value = "ok"
        s.add_node("n", node)
        executor = DistributedDAGExecutor(s, config=ExecutionConfig(backend="local"))
        results = executor.run(inputs={"extra": 42})
        assert "extra" in results

    def test_shutdown_no_error(self):
        """shutdown() completes without raising on local backend."""
        s = _simple_scheduler()
        executor = DistributedDAGExecutor(s, config=ExecutionConfig(backend="local"))
        executor.shutdown()  # must not raise


class TestDistributedDAGExecutorRay:
    def test_ray_backend_raises_when_ray_unavailable(self):
        """When ray is not installed, constructing with 'ray' backend raises DistributedExecutorError."""
        s = _simple_scheduler()
        with patch(
            "transformation_portal.execution_graph.distributed_executor.RAY_AVAILABLE",
            False,
        ):
            with pytest.raises(DistributedExecutorError, match="[Rr]ay"):
                DistributedDAGExecutor(s, config=ExecutionConfig(backend="ray"))

    def test_run_ray_calls_ray_remote(self):
        """With ray available and mocked, ray.remote is called during run()."""
        s = _simple_scheduler()
        mock_ray = MagicMock()
        mock_remote_fn = MagicMock()
        mock_remote_fn.remote.return_value = MagicMock()
        mock_ray.remote.return_value = mock_remote_fn
        mock_ray.get.return_value = "ray_result"
        mock_ray.is_initialized.return_value = True

        with (
            patch("transformation_portal.execution_graph.distributed_executor.RAY_AVAILABLE", True),
            patch("transformation_portal.execution_graph.distributed_executor.ray", mock_ray),
        ):
            executor = DistributedDAGExecutor(s, config=ExecutionConfig(backend="ray"))
            executor.run()

        mock_ray.remote.assert_called()


class _StubTask:
    """Outcome of a task that ran in-process when it was submitted."""

    def __init__(self, fn, args):
        self.error = None
        self.value = None
        try:
            self.value = fn(*args)
        except Exception as exc:  # noqa: BLE001 - re-raised by _StubRay.get, like a Ray task error
            self.error = exc


class _StubRay:
    """In-process stand-in for the parts of the Ray API the executor uses."""

    def __init__(self) -> None:
        self.get_timeouts: list[float | None] = []

    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def remote(*args, **_kwargs):
        def wrap(fn):
            submit = MagicMock()
            submit.remote.side_effect = lambda *call_args: _StubTask(fn, call_args)
            return submit

        return wrap(args[0]) if args and callable(args[0]) else wrap

    def get(self, task: _StubTask, timeout: float | None = None):
        self.get_timeouts.append(timeout)
        if task.error is not None:
            raise task.error
        return task.value


def _recording_node(calls: list[str], name: str, *, fail: bool = False) -> MagicMock:
    node = MagicMock()

    def run(**inputs):
        calls.append(name)
        if fail:
            raise RuntimeError(f"{name} lost its worker")
        return {name: sorted(inputs)}

    node.run.side_effect = run
    return node


@pytest.fixture
def stub_ray():
    ray = _StubRay()
    with (
        patch("transformation_portal.execution_graph.distributed_executor.RAY_AVAILABLE", True),
        patch("transformation_portal.execution_graph.distributed_executor.ray", ray),
    ):
        yield ray


class TestRayBackendFailsLikeTheLocalBackend:
    def test_valid_dag_gives_the_same_results_as_the_local_backend(self, stub_ray):
        def scheduler() -> PriorityDAGScheduler:
            s = PriorityDAGScheduler()
            s.add_node("ingest", _recording_node([], "ingest"))
            s.add_node("segment", _recording_node([], "segment"), deps=["ingest"])
            s.add_node("report", _recording_node([], "report"), deps=["ingest", "segment"])
            return s

        local = DistributedDAGExecutor(scheduler(), config=ExecutionConfig(backend="local")).run()
        distributed = DistributedDAGExecutor(scheduler(), config=ExecutionConfig(backend="ray")).run()

        assert distributed == local
        assert distributed["report"] == {"report": ["ingest", "segment"]}

    def test_missing_dependency_is_an_error_instead_of_a_skipped_node(self, stub_ray):
        calls: list[str] = []
        s = PriorityDAGScheduler()
        s.add_node("ingest", _recording_node(calls, "ingest"))
        s.add_node("quality_gate", _recording_node(calls, "quality_gate"), deps=["ingets"])  # misspelled dependency

        with pytest.raises(DistributedExecutorError, match="DAG validation failed.*'quality_gate'.*'ingets'"):
            DistributedDAGExecutor(s, config=ExecutionConfig(backend="ray")).run()

        assert calls == []

    def test_cycle_is_an_error_instead_of_an_empty_result(self, stub_ray):
        calls: list[str] = []
        s = PriorityDAGScheduler()
        s.add_node("a", _recording_node(calls, "a"), deps=["b"])
        s.add_node("b", _recording_node(calls, "b"), deps=["a"])

        with pytest.raises(DistributedExecutorError, match="Cycle detected"):
            DistributedDAGExecutor(s, config=ExecutionConfig(backend="ray")).run()

        assert calls == []

    def test_failed_dependency_is_reported_with_its_node_name(self, stub_ray):
        calls: list[str] = []
        s = PriorityDAGScheduler()
        s.add_node("ingest", _recording_node(calls, "ingest", fail=True))
        s.add_node("segment", _recording_node(calls, "segment"), deps=["ingest"])

        with pytest.raises(DistributedExecutorError, match="Node 'ingest' execution failed: ingest lost its worker"):
            DistributedDAGExecutor(s, config=ExecutionConfig(backend="ray")).run()

        assert calls == ["ingest"]

    def test_node_timeout_also_bounds_the_wait_for_a_dependency(self, stub_ray):
        s = PriorityDAGScheduler()
        s.add_node("ingest", _recording_node([], "ingest"))
        s.add_node("segment", _recording_node([], "segment"), deps=["ingest"])

        DistributedDAGExecutor(s, config=ExecutionConfig(backend="ray", timeout_per_node=7.5)).run()

        assert stub_ray.get_timeouts == [7.5, 7.5, 7.5]


class TestExecutionConfig:
    def test_default_config(self):
        """Default ExecutionConfig has backend='local' and no GPU tasks."""
        cfg = ExecutionConfig()
        assert cfg.backend == "local"
        assert cfg.num_gpus_per_task == 0
        assert cfg.max_retries == 0

    def test_custom_config_stored(self):
        """Custom values are stored on ExecutionConfig."""
        cfg = ExecutionConfig(backend="ray", num_gpus_per_task=1, max_retries=3)
        assert cfg.backend == "ray"
        assert cfg.num_gpus_per_task == 1
        assert cfg.max_retries == 3
