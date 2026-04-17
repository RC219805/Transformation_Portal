"""Tests for SandboxExecutor - execution isolation and security boundaries.

This module tests:
- Sandbox creation and lifecycle management
- GPU semaphore integration
- Execution result tracking
- Batch execution
- Error handling and recovery
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.runtime.sandbox import SandboxMetrics
from transformation_portal.runtime.sandbox_executor import (
    DAGNodeProtocol,
    ExecutionResult,
    ExecutorConfig,
    SandboxExecutor,
)

# --- Mock DAG Nodes for Testing ---


@dataclass
class MockSuccessNode:
    """A mock DAG node that always succeeds."""

    output_value: str = "test_output"

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        """Execute and return mock output."""
        return {"result_sha": self.output_value, "inputs_received": inputs}


@dataclass
class MockFailingNode:
    """A mock DAG node that always raises an exception."""

    error_message: str = "Node execution failed"

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        """Execute and raise an error."""
        raise RuntimeError(self.error_message)


@dataclass
class MockSlowNode:
    """A mock DAG node that simulates slow execution."""

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        """Execute with delay (mocked)."""
        return {"status": "completed"}


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_property_true_when_no_error(self) -> None:
        """ExecutionResult.success is True when error is None."""
        metrics = SandboxMetrics()
        result = ExecutionResult(
            node_id="test_node",
            outputs={"sha": "abc123"},
            metrics=metrics,
            error=None,
        )

        assert result.success is True

    def test_success_property_false_when_error(self) -> None:
        """ExecutionResult.success is False when error is set."""
        metrics = SandboxMetrics()
        result = ExecutionResult(
            node_id="test_node",
            outputs={},
            metrics=metrics,
            error="Something went wrong",
        )

        assert result.success is False

    def test_result_with_gpu_slot(self) -> None:
        """ExecutionResult correctly stores GPU slot info."""
        from transformation_portal.runtime.gpu_semaphore import GPUSlot

        metrics = SandboxMetrics()
        gpu_slot = GPUSlot(device_id=1)

        result = ExecutionResult(
            node_id="gpu_node",
            outputs={"result": "data"},
            metrics=metrics,
            gpu_slot=gpu_slot,
        )

        assert result.gpu_slot is not None
        assert result.gpu_slot.device_id == 1
        assert result.gpu_slot.device_string == "cuda:1"


class TestExecutorConfig:
    """Tests for ExecutorConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """ExecutorConfig has sensible defaults."""
        config = ExecutorConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
        )

        assert config.max_concurrent_gpu == 1
        assert config.cleanup_sandboxes is True

    def test_custom_values(self, tmp_path: Path) -> None:
        """ExecutorConfig accepts custom values."""
        config = ExecutorConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            max_concurrent_gpu=4,
            cleanup_sandboxes=False,
        )

        assert config.max_concurrent_gpu == 4
        assert config.cleanup_sandboxes is False


class TestSandboxExecutor:
    """Tests for SandboxExecutor class."""

    @pytest.fixture
    def executor_config(self, tmp_path: Path) -> ExecutorConfig:
        """Create executor config for tests."""
        return ExecutorConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            max_concurrent_gpu=0,  # Disable GPU for unit tests
            cleanup_sandboxes=True,
        )

    @pytest.fixture
    def mock_fs(self):
        """Create mock FSGuard."""
        fs = MagicMock()
        fs.read_text.return_value = "test content"
        fs.read_bytes.return_value = b"test bytes"
        return fs

    @pytest.fixture
    def mock_cas(self, tmp_path: Path):
        """Create mock CAS store."""
        cas = MagicMock()
        cas.has_object.return_value = True
        return cas

    def test_executor_initialization(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """SandboxExecutor initializes correctly."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        assert executor.config == executor_config
        assert executor.fs == mock_fs
        assert executor.cas == mock_cas
        assert executor.gpu_semaphore is None
        assert executor.execution_count == 0
        assert executor.results == []

    def test_executor_creates_workspace_dir(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """SandboxExecutor creates workspace directory."""
        SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        assert executor_config.workspace_root.exists()

    def test_create_sandbox(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """create_sandbox returns configured Sandbox."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        sandbox = executor.create_sandbox("test_node_001")

        assert sandbox.node_id == "test_node_001"
        assert sandbox.workspace.exists()
        assert sandbox.workspace.parent == executor_config.workspace_root

    def test_create_sandbox_with_custom_cleanup(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """create_sandbox respects cleanup_on_exit override."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        sandbox = executor.create_sandbox("test_node", cleanup_on_exit=False)

        assert sandbox.config.cleanup_on_exit is False

    def test_run_node_success(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """run_node returns successful result for working node."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        sandbox = executor.create_sandbox("success_node")
        node = MockSuccessNode(output_value="output_sha_abc")

        result = executor.run_node(
            node=node,
            sandbox=sandbox,
            use_gpu=False,
            input_data="test_input",
        )

        assert result.success is True
        assert result.node_id == "success_node"
        assert result.outputs["result_sha"] == "output_sha_abc"
        assert result.outputs["inputs_received"]["input_data"] == "test_input"
        assert result.error is None

    def test_run_node_failure(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """run_node returns failure result for failing node."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        sandbox = executor.create_sandbox("failing_node")
        node = MockFailingNode(error_message="Test failure")

        result = executor.run_node(
            node=node,
            sandbox=sandbox,
            use_gpu=False,
        )

        assert result.success is False
        assert result.node_id == "failing_node"
        assert "Test failure" in result.error
        assert result.outputs == {}

    def test_run_node_increments_count(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """run_node increments execution count."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        assert executor.execution_count == 0

        sandbox = executor.create_sandbox("node_1")
        executor.run_node(node=MockSuccessNode(), sandbox=sandbox)
        assert executor.execution_count == 1

        sandbox2 = executor.create_sandbox("node_2")
        executor.run_node(node=MockSuccessNode(), sandbox=sandbox2)
        assert executor.execution_count == 2

    def test_run_node_stores_results(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """run_node stores results in results list."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        sandbox = executor.create_sandbox("tracked_node")
        result = executor.run_node(node=MockSuccessNode(), sandbox=sandbox)

        assert len(executor.results) == 1
        assert executor.results[0] == result

    def test_execute_convenience_method(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """execute() creates sandbox and runs node in one call."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        result = executor.execute(
            node_id="convenience_node",
            node=MockSuccessNode(output_value="convenience_output"),
            use_gpu=False,
        )

        assert result.success is True
        assert result.node_id == "convenience_node"
        assert result.outputs["result_sha"] == "convenience_output"

    def test_execute_batch(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """execute_batch runs multiple nodes sequentially."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        nodes = [
            ("batch_1", MockSuccessNode(output_value="out1"), {"key": "val1"}),
            ("batch_2", MockSuccessNode(output_value="out2"), {"key": "val2"}),
            ("batch_3", MockSuccessNode(output_value="out3"), {"key": "val3"}),
        ]

        results = executor.execute_batch(nodes, use_gpu=False)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].outputs["result_sha"] == "out1"
        assert results[1].outputs["result_sha"] == "out2"
        assert results[2].outputs["result_sha"] == "out3"

    def test_execute_batch_with_failure(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """execute_batch continues after individual failures."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        nodes = [
            ("batch_1", MockSuccessNode(), {}),
            ("batch_2", MockFailingNode(), {}),
            ("batch_3", MockSuccessNode(), {}),
        ]

        results = executor.execute_batch(nodes, use_gpu=False)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    def test_get_summary(self, executor_config: ExecutorConfig, mock_fs, mock_cas) -> None:
        """get_summary returns execution statistics."""
        executor = SandboxExecutor(
            config=executor_config,
            fs=mock_fs,
            cas=mock_cas,
            gpu_semaphore=None,
        )

        # Run some executions
        executor.execute("node_1", MockSuccessNode())
        executor.execute("node_2", MockFailingNode())
        executor.execute("node_3", MockSuccessNode())

        summary = executor.get_summary()

        assert summary["total_executions"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["gpu_enabled"] is False
        assert "total_duration_seconds" in summary


class TestSandboxExecutorWithGPU:
    """Tests for SandboxExecutor with GPU semaphore."""

    @pytest.fixture
    def gpu_executor_config(self, tmp_path: Path) -> ExecutorConfig:
        """Create executor config with GPU enabled."""
        return ExecutorConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            max_concurrent_gpu=2,
            cleanup_sandboxes=True,
        )

    def test_executor_with_gpu_semaphore(self, gpu_executor_config: ExecutorConfig) -> None:
        """SandboxExecutor works with GPU semaphore provided."""
        from transformation_portal.runtime.gpu_semaphore import GPUSemaphore

        semaphore = GPUSemaphore(num_devices=2)

        executor = SandboxExecutor(
            config=gpu_executor_config,
            gpu_semaphore=semaphore,
        )

        assert executor.gpu_semaphore is not None
        assert executor.gpu_semaphore.num_devices == 2

    def test_run_node_with_gpu(self, gpu_executor_config: ExecutorConfig) -> None:
        """run_node acquires GPU slot when use_gpu=True."""
        from transformation_portal.runtime.gpu_semaphore import GPUSemaphore

        semaphore = GPUSemaphore(num_devices=1)

        executor = SandboxExecutor(
            config=gpu_executor_config,
            gpu_semaphore=semaphore,
        )

        result = executor.execute(
            node_id="gpu_node",
            node=MockSuccessNode(),
            use_gpu=True,
        )

        assert result.success is True
        # GPU slot should have been acquired and released
        assert result.gpu_slot is not None
        assert result.gpu_slot.device_id == 0

    def test_gpu_summary_shows_enabled(self, gpu_executor_config: ExecutorConfig) -> None:
        """get_summary shows gpu_enabled=True when semaphore present."""
        from transformation_portal.runtime.gpu_semaphore import GPUSemaphore

        semaphore = GPUSemaphore(num_devices=1)

        executor = SandboxExecutor(
            config=gpu_executor_config,
            gpu_semaphore=semaphore,
        )

        summary = executor.get_summary()
        assert summary["gpu_enabled"] is True


class TestSandboxExecutorSecurityBoundaries:
    """Tests for security boundaries in SandboxExecutor."""

    @pytest.fixture
    def executor(self, tmp_path: Path):
        """Create executor for security tests."""
        config = ExecutorConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            max_concurrent_gpu=0,
        )
        return SandboxExecutor(config=config)

    def test_sandbox_isolation_per_node(self, executor: SandboxExecutor) -> None:
        """Each sandbox has isolated workspace."""
        sandbox1 = executor.create_sandbox("node_1")
        sandbox2 = executor.create_sandbox("node_2")

        # Workspaces should be separate
        assert sandbox1.workspace != sandbox2.workspace
        assert sandbox1.workspace.name == "node_1"
        assert sandbox2.workspace.name == "node_2"

    def test_invalid_node_id_rejected(self, executor: SandboxExecutor) -> None:
        """Invalid node IDs are rejected."""
        from transformation_portal.runtime.sandbox import SandboxError

        with pytest.raises(SandboxError):
            executor.create_sandbox("../escape_attempt")

        with pytest.raises(SandboxError):
            executor.create_sandbox("node/with/slashes")

    def test_node_execution_errors_contained(self, executor: SandboxExecutor) -> None:
        """Errors in node execution are contained and don't propagate."""
        # This should not raise - error should be captured in result
        result = executor.execute(
            node_id="error_node",
            node=MockFailingNode(error_message="Contained error"),
        )

        assert result.success is False
        assert "Contained error" in result.error
        # Executor should still be functional
        assert executor.execution_count == 1
