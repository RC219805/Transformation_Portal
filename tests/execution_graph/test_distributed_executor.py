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
