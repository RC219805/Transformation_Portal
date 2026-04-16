"""Tests for ExecutionEngine - dispatch and coordination.

This module tests:
- Engine initialization and configuration
- Node execution and dispatch
- GPU pool integration
- Merkle DAG provenance tracking
- Input registration and lineage
- Pipeline execution
- Summary and statistics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.runtime.engine import (
    EngineConfig,
    ExecutionEngine,
    ExecutionEngineError,
    ExecutionRecord,
)


# --- Mock Node Classes for Testing ---


class MockProcessNode:
    """Mock processing node for tests."""

    def __init__(self, output_sha: str = "output_sha_default"):
        self.output_sha = output_sha

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        """Return mock output."""
        return {"result_sha": self.output_sha}


class MockFailingNode:
    """Mock node that always fails."""

    def __init__(self, error_message: str = "Node failed"):
        self.error_message = error_message

    def run(self, *, sandbox, **inputs) -> Dict[str, Any]:
        """Raise an exception."""
        raise RuntimeError(self.error_message)


# --- Test Classes ---


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """EngineConfig has sensible defaults."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
        )

        assert config.gpu_devices is None
        assert config.process_timeout is None
        assert config.cleanup_workspaces is True
        assert config.enable_provenance is True

    def test_custom_values(self, tmp_path: Path) -> None:
        """EngineConfig accepts custom values."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            gpu_devices=[0, 1, 2],
            process_timeout=120.0,
            cleanup_workspaces=False,
            enable_provenance=False,
        )

        assert config.gpu_devices == [0, 1, 2]
        assert config.process_timeout == 120.0
        assert config.cleanup_workspaces is False
        assert config.enable_provenance is False


class TestExecutionRecord:
    """Tests for ExecutionRecord dataclass."""

    def test_successful_record(self) -> None:
        """ExecutionRecord captures successful execution."""
        record = ExecutionRecord(
            node_id="process_001",
            node_type="MockProcessNode",
            merkle_hash="abc123def456",
            inputs={"image_sha": "input_sha"},
            outputs={"result_sha": "output_sha"},
            duration_seconds=1.5,
            gpu_device=0,
            success=True,
        )

        assert record.node_id == "process_001"
        assert record.success is True
        assert record.error is None
        assert record.gpu_device == 0

    def test_failed_record(self) -> None:
        """ExecutionRecord captures failed execution."""
        record = ExecutionRecord(
            node_id="failed_001",
            node_type="MockFailingNode",
            merkle_hash=None,
            inputs={},
            outputs={},
            duration_seconds=0.1,
            success=False,
            error="Node failed with error",
        )

        assert record.success is False
        assert "failed" in record.error.lower()
        assert record.merkle_hash is None


class TestExecutionEngine:
    """Tests for ExecutionEngine class."""

    @pytest.fixture
    def engine_config(self, tmp_path: Path) -> EngineConfig:
        """Create engine config for tests."""
        return EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            gpu_devices=None,  # No GPU for unit tests
            enable_provenance=True,
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock ProcessExecutor."""
        executor = MagicMock()
        from transformation_portal.runtime.process_executor import ProcessResult

        executor.run.return_value = ProcessResult(
            outputs={"result_sha": "abc123"},
            manifest={"node_id": "test"},
        )
        return executor

    @pytest.fixture
    def mock_gpu_pool(self):
        """Create mock GPUPool."""
        pool = MagicMock()
        pool.total_devices = 0
        return pool

    def test_engine_initialization(self, engine_config: EngineConfig) -> None:
        """ExecutionEngine initializes correctly."""
        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec:
            mock_exec.return_value = MagicMock()

            engine = ExecutionEngine(engine_config)

            assert engine.config == engine_config
            assert engine.cas is not None
            assert engine.dag is not None  # Provenance enabled
            assert engine.executions == []

    def test_engine_creates_directories(self, engine_config: EngineConfig) -> None:
        """ExecutionEngine creates workspace and CAS directories."""
        with patch("transformation_portal.runtime.engine.ProcessExecutor"):
            ExecutionEngine(engine_config)

            assert engine_config.workspace_root.exists()
            assert engine_config.cas_root.exists()

    def test_engine_provenance_disabled(self, tmp_path: Path) -> None:
        """ExecutionEngine can disable provenance tracking."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            enable_provenance=False,
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor"):
            engine = ExecutionEngine(config)

            assert engine.dag is None

    def test_register_input(self, engine_config: EngineConfig) -> None:
        """register_input adds artifact to Merkle DAG."""
        with patch("transformation_portal.runtime.engine.ProcessExecutor"):
            engine = ExecutionEngine(engine_config)

            merkle_hash = engine.register_input(
                sha="a" * 64,
                artifact_type="image",
                metadata={"filename": "test.png"},
            )

            assert len(merkle_hash) == 64
            # Registering same SHA again should return same hash
            merkle_hash2 = engine.register_input(sha="a" * 64)
            assert merkle_hash == merkle_hash2

    def test_register_input_no_dag(self, tmp_path: Path) -> None:
        """register_input returns SHA when provenance disabled."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            enable_provenance=False,
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor"):
            engine = ExecutionEngine(config)

            result = engine.register_input(sha="b" * 64)
            assert result == "b" * 64

    def test_run_node_success(self, engine_config: EngineConfig) -> None:
        """run_node returns merkle hash and outputs on success."""
        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
            from transformation_portal.runtime.process_executor import ProcessResult

            mock_executor = MagicMock()
            mock_executor.run.return_value = ProcessResult(
                outputs={"result_sha": "output_sha_123"},
                manifest={"node_id": "test_node"},
            )
            mock_exec_cls.return_value = mock_executor

            engine = ExecutionEngine(engine_config)

            merkle_hash, outputs = engine.run_node(
                MockProcessNode,
                inputs={"data": "value"},
                node_id="test_node_001",
            )

            assert merkle_hash is not None
            assert len(merkle_hash) == 64
            assert outputs["result_sha"] == "output_sha_123"

    def test_run_node_failure_raises(self, engine_config: EngineConfig) -> None:
        """run_node raises ExecutionEngineError on failure."""
        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
            from transformation_portal.runtime.process_executor import ProcessResult

            mock_executor = MagicMock()
            mock_executor.run.return_value = ProcessResult(
                outputs={},
                error="Node execution failed",
            )
            mock_exec_cls.return_value = mock_executor

            engine = ExecutionEngine(engine_config)

            with pytest.raises(ExecutionEngineError) as exc_info:
                engine.run_node(
                    MockFailingNode,
                    inputs={},
                    node_id="failing_node",
                )

            assert "failed" in str(exc_info.value).lower()

    def test_run_node_tracks_execution(self, engine_config: EngineConfig) -> None:
        """run_node adds ExecutionRecord to executions list."""
        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
            from transformation_portal.runtime.process_executor import ProcessResult

            mock_executor = MagicMock()
            mock_executor.run.return_value = ProcessResult(
                outputs={"result": "value"},
                manifest={},
            )
            mock_exec_cls.return_value = mock_executor

            engine = ExecutionEngine(engine_config)

            engine.run_node(
                MockProcessNode,
                inputs={"key": "val"},
                node_id="tracked_node",
            )

            assert len(engine.executions) == 1
            record = engine.executions[0]
            assert record.node_id == "tracked_node"
            assert record.node_type == "MockProcessNode"
            assert record.success is True

    def test_run_node_with_input_artifacts(self, engine_config: EngineConfig) -> None:
        """run_node registers input artifacts for provenance."""
        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
            from transformation_portal.runtime.process_executor import ProcessResult

            mock_executor = MagicMock()
            mock_executor.run.return_value = ProcessResult(
                outputs={"out": "c" * 64},
                manifest={},
            )
            mock_exec_cls.return_value = mock_executor

            engine = ExecutionEngine(engine_config)

            input_sha = "d" * 64
            engine.run_node(
                MockProcessNode,
                inputs={"image_sha": input_sha},
                node_id="provenance_node",
                input_artifact_shas=[input_sha],
            )

            # Input should have been auto-registered
            assert input_sha in engine._input_artifacts


class TestExecutionEngineGPU:
    """Tests for ExecutionEngine with GPU pool."""

    def test_engine_with_explicit_gpu_devices(self, tmp_path: Path) -> None:
        """Engine creates GPU pool with specified devices."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            gpu_devices=[0, 1],
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor"), patch(
            "transformation_portal.runtime.engine.GPUPool"
        ) as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool.total_devices = 2
            mock_pool_cls.return_value = mock_pool

            engine = ExecutionEngine(config)

            assert engine.gpu_pool is not None
            mock_pool_cls.assert_called_once_with(devices=[0, 1])

    def test_run_node_acquires_gpu(self, tmp_path: Path) -> None:
        """run_node acquires GPU when use_gpu=True."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            gpu_devices=[0],
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls, patch(
            "transformation_portal.runtime.engine.GPUPool"
        ) as mock_pool_cls:
            from transformation_portal.runtime.process_executor import ProcessResult
            from transformation_portal.runtime.gpu_pool import GPULease

            # Mock executor
            mock_executor = MagicMock()
            mock_executor.run.return_value = ProcessResult(
                outputs={"result": "value"},
                manifest={},
            )
            mock_exec_cls.return_value = mock_executor

            # Mock GPU pool
            mock_pool = MagicMock()
            mock_pool.total_devices = 1
            mock_lease = GPULease(device_id=0, lease_id=1, acquired_at=0.0)
            mock_pool.acquire.return_value = mock_lease
            mock_pool_cls.return_value = mock_pool

            engine = ExecutionEngine(config)

            engine.run_node(
                MockProcessNode,
                inputs={},
                node_id="gpu_node",
                use_gpu=True,
            )

            mock_pool.acquire.assert_called_once()
            mock_pool.release.assert_called_once_with(mock_lease)


class TestExecutionEngineLineage:
    """Tests for lineage and provenance in ExecutionEngine."""

    @pytest.fixture
    def provenance_engine(self, tmp_path: Path):
        """Create engine with provenance enabled."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            enable_provenance=True,
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor"):
            return ExecutionEngine(config)

    def test_get_lineage(self, provenance_engine: ExecutionEngine) -> None:
        """get_lineage returns node ancestry."""
        engine = provenance_engine

        # Register an input
        input_hash = engine.register_input(sha="e" * 64, artifact_type="image")

        # Lineage should include the artifact
        lineage = engine.get_lineage(input_hash)

        assert len(lineage) == 1
        assert lineage[0].hash == input_hash

    def test_get_lineage_no_dag(self, tmp_path: Path) -> None:
        """get_lineage returns empty list when provenance disabled."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
            enable_provenance=False,
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor"):
            engine = ExecutionEngine(config)

            lineage = engine.get_lineage("f" * 64)
            assert lineage == []

    def test_export_dag(self, provenance_engine: ExecutionEngine, tmp_path: Path) -> None:
        """export_dag writes provenance to JSON file."""
        engine = provenance_engine

        # Add some data
        engine.register_input(sha="g" * 64, artifact_type="config")

        output_path = tmp_path / "provenance.json"
        engine.export_dag(output_path)

        assert output_path.exists()

        import json

        data = json.loads(output_path.read_text())
        assert "nodes" in data
        assert "version" in data

    def test_verify_integrity(self, provenance_engine: ExecutionEngine) -> None:
        """verify_integrity returns empty list for valid DAG."""
        engine = provenance_engine

        engine.register_input(sha="h" * 64)

        errors = engine.verify_integrity()
        assert errors == []


class TestExecutionEnginePipeline:
    """Tests for pipeline execution in ExecutionEngine."""

    def test_run_pipeline(self, tmp_path: Path) -> None:
        """run_pipeline executes sequence of nodes."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
            from transformation_portal.runtime.process_executor import ProcessResult

            mock_executor = MagicMock()
            mock_executor.run.return_value = ProcessResult(
                outputs={"result": "value"},
                manifest={},
            )
            mock_exec_cls.return_value = mock_executor

            engine = ExecutionEngine(config)

            nodes = [
                ("step_1", MockProcessNode, {}, {"input": "data1"}),
                ("step_2", MockProcessNode, {}, {"input": "data2"}),
                ("step_3", MockProcessNode, {}, {"input": "data3"}),
            ]

            results = engine.run_pipeline(nodes)

            assert len(results) == 3
            assert all(h is not None for h, o in results)
            assert mock_executor.run.call_count == 3


class TestExecutionEngineSummary:
    """Tests for summary and statistics in ExecutionEngine."""

    def test_get_summary(self, tmp_path: Path) -> None:
        """get_summary returns execution statistics."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
            from transformation_portal.runtime.process_executor import ProcessResult

            mock_executor = MagicMock()
            mock_executor.run.return_value = ProcessResult(outputs={}, manifest={})
            mock_exec_cls.return_value = mock_executor

            engine = ExecutionEngine(config)

            # Run some nodes
            engine.run_node(MockProcessNode, inputs={}, node_id="sum_1")
            engine.run_node(MockProcessNode, inputs={}, node_id="sum_2")

            summary = engine.get_summary()

            assert summary["total_executions"] == 2
            assert summary["successful"] == 2
            assert summary["failed"] == 0
            assert "total_duration_seconds" in summary
            assert "cas_root" in summary
            assert summary["dag_nodes"] > 0  # Provenance nodes created

    def test_get_summary_with_failures(self, tmp_path: Path) -> None:
        """get_summary correctly counts failures."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
            from transformation_portal.runtime.process_executor import ProcessResult

            mock_executor = MagicMock()
            # Alternate success and failure
            mock_executor.run.side_effect = [
                ProcessResult(outputs={"r": "v"}, manifest={}),
                ProcessResult(outputs={}, error="Failed"),
            ]
            mock_exec_cls.return_value = mock_executor

            engine = ExecutionEngine(config)

            engine.run_node(MockProcessNode, inputs={}, node_id="ok")

            try:
                engine.run_node(MockFailingNode, inputs={}, node_id="fail")
            except ExecutionEngineError:
                pass

            summary = engine.get_summary()

            assert summary["total_executions"] == 2
            assert summary["successful"] == 1
            assert summary["failed"] == 1

    def test_executions_property(self, tmp_path: Path) -> None:
        """executions property returns copy of records."""
        config = EngineConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
        )

        with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
            from transformation_portal.runtime.process_executor import ProcessResult

            mock_executor = MagicMock()
            mock_executor.run.return_value = ProcessResult(outputs={}, manifest={})
            mock_exec_cls.return_value = mock_executor

            engine = ExecutionEngine(config)

            engine.run_node(MockProcessNode, inputs={}, node_id="exec_1")

            executions = engine.executions

            # Should be a copy
            assert executions is not engine._executions
            assert len(executions) == 1
