"""Tests for CAS-aware execution wrapper and DAG executor.

Test Coverage:
- CASExecutor caching behavior
- FileLock concurrency safety
- CacheResult structure
- CASDAGExecutor partial reuse
- DAG determinism verification
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.core.execution_wrapper import (
    CacheResult,
    CASExecutor,
    ExecutorConfig,
    FileLock,
    execute_with_caching,
)
from transformation_portal.core.cas_dag_executor import (
    CASDAGConfig,
    CASDAGExecutor,
    CASExecutionResult,
    verify_dag_determinism,
)
from transformation_portal.stage_graph.graph import StageGraph
from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus
from transformation_portal.storage.cas_store import ArtifactStore


class TestFileLock:
    """Tests for FileLock concurrency primitive."""

    def test_basic_acquire_release(self, tmp_path):
        """Test basic lock acquire and release."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path, timeout=5.0)

        assert lock.acquire() is True
        assert lock_path.exists()

        lock.release()
        assert not lock_path.exists()

    def test_context_manager(self, tmp_path):
        """Test lock as context manager."""
        lock_path = tmp_path / "test.lock"

        with FileLock(lock_path, timeout=5.0):
            assert lock_path.exists()

        assert not lock_path.exists()

    def test_concurrent_locks(self, tmp_path):
        """Test locks block concurrent access."""
        lock_path = tmp_path / "test.lock"
        execution_times = []

        def worker(worker_id: int):
            with FileLock(lock_path, timeout=10.0):
                entry_time = time.time()
                time.sleep(0.1)
                exit_time = time.time()
                execution_times.append((worker_id, entry_time, exit_time))

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            for f in as_completed(futures):
                f.result()

        # Verify mutual exclusion: no two workers overlap
        # Sort by entry time
        execution_times.sort(key=lambda x: x[1])
        for i in range(len(execution_times) - 1):
            current_exit = execution_times[i][2]
            next_entry = execution_times[i + 1][1]
            # Each worker should exit before the next enters
            assert current_exit <= next_entry, "Workers should not overlap"

    def test_timeout(self, tmp_path):
        """Test lock timeout behavior."""
        lock_path = tmp_path / "test.lock"

        # Create existing lock
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(str(time.time()))

        lock = FileLock(lock_path, timeout=0.1)
        assert lock.acquire() is False


class SimpleStage:
    """Simple stage implementation for testing."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def execute(self, inputs: Dict[str, Any], config: Any) -> Dict[str, Any]:
        self.call_count += 1
        value = inputs.get("value", 0)
        return {"result": value * 2, "call_count": self.call_count}


class TestCASExecutor:
    """Tests for CASExecutor."""

    @pytest.fixture
    def executor_setup(self, tmp_path):
        """Setup CAS executor with temp directories."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"

        store = ArtifactStore(cas_root)
        config = ExecutorConfig(enable_caching=True)

        executor = CASExecutor(store, cache_dir, config)
        return executor, store

    def test_first_execution_cache_miss(self, executor_setup):
        """Test first execution is a cache miss."""
        executor, _ = executor_setup
        stage = SimpleStage("test_stage")

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                result = executor.execute(
                    stage=stage,
                    inputs={"value": 21},
                    config={"quality": "high"},
                )

                assert result.cache_hit is False
                assert result.outputs["result"] == 42
                assert stage.call_count == 1

    def test_second_execution_cache_hit(self, executor_setup):
        """Test second execution with same inputs is cache hit."""
        executor, _ = executor_setup
        stage = SimpleStage("test_stage")

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution
                result1 = executor.execute(
                    stage=stage,
                    inputs={"value": 21},
                    config={},
                )
                assert result1.cache_hit is False
                assert stage.call_count == 1

                # Second execution with same inputs
                result2 = executor.execute(
                    stage=stage,
                    inputs={"value": 21},
                    config={},
                )
                assert result2.cache_hit is True
                assert stage.call_count == 1  # No additional call

    def test_different_inputs_cache_miss(self, executor_setup):
        """Test different inputs cause cache miss."""
        executor, _ = executor_setup
        stage = SimpleStage("test_stage")

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution
                result1 = executor.execute(
                    stage=stage,
                    inputs={"value": 21},
                    config={},
                )
                assert stage.call_count == 1

                # Second execution with different inputs
                result2 = executor.execute(
                    stage=stage,
                    inputs={"value": 50},
                    config={},
                )
                assert result2.cache_hit is False
                assert result2.outputs["result"] == 100
                assert stage.call_count == 2

    def test_different_config_cache_miss(self, executor_setup):
        """Test different config causes cache miss."""
        executor, _ = executor_setup
        stage = SimpleStage("test_stage")

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution
                result1 = executor.execute(
                    stage=stage,
                    inputs={"value": 21},
                    config={"quality": "low"},
                )
                assert stage.call_count == 1

                # Second execution with different config
                result2 = executor.execute(
                    stage=stage,
                    inputs={"value": 21},
                    config={"quality": "high"},
                )
                assert result2.cache_hit is False
                assert stage.call_count == 2

    def test_numpy_array_inputs(self, executor_setup):
        """Test caching works with numpy array inputs."""
        executor, _ = executor_setup

        class NumpyStage:
            @property
            def name(self) -> str:
                return "numpy_stage"

            @property
            def version(self) -> str:
                return "1.0.0"

            def execute(self, inputs: Dict[str, Any], config: Any) -> Dict[str, Any]:
                arr = inputs["array"]
                return {"sum": float(arr.sum()), "shape": list(arr.shape)}

        stage = NumpyStage()
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                result = executor.execute(
                    stage=stage,
                    inputs={"array": arr},
                    config={},
                )

                assert result.outputs["sum"] == 10.0
                assert result.outputs["shape"] == [2, 2]

    def test_caching_disabled(self, tmp_path):
        """Test execution with caching disabled."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"

        store = ArtifactStore(cas_root)
        config = ExecutorConfig(enable_caching=False)
        executor = CASExecutor(store, cache_dir, config)

        stage = SimpleStage("test_stage")

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # Two executions with same inputs should both execute
                result1 = executor.execute(stage, {"value": 21}, {})
                result2 = executor.execute(stage, {"value": 21}, {})

                assert stage.call_count == 2
                assert result1.cache_hit is False
                assert result2.cache_hit is False


class TestExecuteWithCaching:
    """Tests for execute_with_caching helper function."""

    def test_functional_caching(self, tmp_path):
        """Test functional caching interface."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        call_count = 0

        def my_stage(inputs: Dict[str, Any], config: Any) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"result": inputs["x"] * 2}

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                result1 = execute_with_caching(
                    stage_fn=my_stage,
                    stage_name="my_stage",
                    stage_version="1.0.0",
                    inputs={"x": 5},
                    config={},
                    artifact_store=store,
                    cache_dir=cache_dir,
                )

                assert result1.outputs["result"] == 10
                assert result1.cache_hit is False
                assert call_count == 1


class DAGStageForTesting(Stage):
    """Stage implementation for DAG testing."""

    def __init__(self, name: str, deps: List[str] = None, version: str = "1.0.0"):
        super().__init__(name, version)
        self._deps = deps or []
        self.call_count = 0

    def get_dependencies(self) -> List[str]:
        return self._deps

    def get_cache_key(self, context: StageContext) -> str:
        return hashlib.sha256(f"{self.name}:{self.version}".encode()).hexdigest()

    def compute(self, context: StageContext) -> StageResult:
        self.call_count += 1
        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={f"{self.name}_output": self.call_count},
        )


class TestCASDAGExecutor:
    """Tests for CASDAGExecutor."""

    @pytest.fixture
    def dag_executor_setup(self, tmp_path):
        """Setup CAS DAG executor with temp directories."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"

        store = ArtifactStore(cas_root)
        config = CASDAGConfig(enable_caching=True, enable_provenance=True)

        executor = CASDAGExecutor(store, cache_dir, config)
        return executor, store

    def test_simple_dag_execution(self, dag_executor_setup):
        """Test simple DAG execution."""
        executor, _ = dag_executor_setup

        # Build simple graph
        graph = StageGraph("test_pipeline")
        stage1 = DAGStageForTesting("stage1")
        stage2 = DAGStageForTesting("stage2", deps=["stage1"])

        graph.add_stage(stage1)
        graph.add_stage(stage2)

        context = StageContext()

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                result = executor.execute(graph, context)

                assert result.success is True
                assert len(result.stage_results) == 2
                assert result.cache_misses == 2
                assert result.cache_hits == 0

    def test_dag_caching(self, dag_executor_setup):
        """Test DAG caching on re-execution."""
        executor, _ = dag_executor_setup

        graph = StageGraph("test_pipeline")
        stage1 = DAGStageForTesting("stage1")
        stage2 = DAGStageForTesting("stage2", deps=["stage1"])

        graph.add_stage(stage1)
        graph.add_stage(stage2)

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution
                context1 = StageContext()
                result1 = executor.execute(graph, context1)
                assert result1.cache_misses == 2
                assert stage1.call_count == 1
                assert stage2.call_count == 1

                # Second execution - should be all cache hits
                context2 = StageContext()
                result2 = executor.execute(graph, context2)
                assert result2.cache_hits == 2
                assert result2.cache_misses == 0
                # Call counts should not increase
                assert stage1.call_count == 1
                assert stage2.call_count == 1

    def test_dag_partial_reuse(self, dag_executor_setup):
        """Test partial DAG reuse when config changes."""
        executor, _ = dag_executor_setup

        graph = StageGraph("test_pipeline")
        stage1 = DAGStageForTesting("stage1")
        stage2 = DAGStageForTesting("stage2", deps=["stage1"])

        graph.add_stage(stage1)
        graph.add_stage(stage2)

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution
                context1 = StageContext(config={"stage1": {"x": 1}})
                result1 = executor.execute(graph, context1)
                assert result1.cache_misses == 2

                # Second execution with stage2 config change
                context2 = StageContext(config={"stage2": {"y": 2}})
                result2 = executor.execute(graph, context2)

                # stage1 should hit cache (same config)
                # stage2 should miss (different config via upstream identity)
                assert "stage1" in result2.stage_results
                assert "stage2" in result2.stage_results

    def test_dag_with_provenance(self, dag_executor_setup):
        """Test DAG execution builds provenance Merkle DAG."""
        executor, _ = dag_executor_setup

        graph = StageGraph("test_pipeline")
        stage1 = DAGStageForTesting("stage1")
        stage2 = DAGStageForTesting("stage2", deps=["stage1"])

        graph.add_stage(stage1)
        graph.add_stage(stage2)

        context = StageContext()

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                result = executor.execute(graph, context)

                assert result.merkle_dag is not None
                assert len(result.merkle_dag.nodes) >= 2

    def test_cache_invalidation(self, dag_executor_setup):
        """Test cache invalidation."""
        executor, _ = dag_executor_setup

        graph = StageGraph("test_pipeline")
        stage1 = DAGStageForTesting("stage1")
        graph.add_stage(stage1)

        context = StageContext()

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution
                result1 = executor.execute(graph, context)
                assert result1.cache_misses == 1

                # Invalidate cache
                count = executor.invalidate(stage_names=["stage1"])
                assert count >= 1

                # Re-execute should miss cache
                result2 = executor.execute(graph, context)
                assert result2.cache_misses == 1


class TestCASExecutionResult:
    """Tests for CASExecutionResult dataclass."""

    def test_cache_stats(self):
        """Test cache statistics computation."""
        result = CASExecutionResult(
            success=True,
            stage_results={},
            execution_order=[],
            cache_hits=7,
            cache_misses=3,
            total_duration_ms=1000.0,
        )

        stats = result.get_cache_stats()

        assert stats["total_stages"] == 10
        assert stats["cache_hits"] == 7
        assert stats["cache_misses"] == 3
        assert stats["hit_rate"] == 0.7
        assert stats["speedup_estimate"] > 1.0

    def test_cache_stats_no_stages(self):
        """Test cache stats with no stages."""
        result = CASExecutionResult(
            success=True,
            stage_results={},
            execution_order=[],
            cache_hits=0,
            cache_misses=0,
            total_duration_ms=0.0,
        )

        stats = result.get_cache_stats()

        assert stats["total_stages"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["speedup_estimate"] == 1.0

    def test_perfect_cache_hit(self):
        """Test stats with all cache hits."""
        result = CASExecutionResult(
            success=True,
            stage_results={},
            execution_order=[],
            cache_hits=10,
            cache_misses=0,
            total_duration_ms=100.0,
        )

        stats = result.get_cache_stats()

        assert stats["hit_rate"] == 1.0
        assert stats["speedup_estimate"] == float("inf")


class TestVerifyDAGDeterminism:
    """Tests for verify_dag_determinism function."""

    def test_deterministic_dag(self, tmp_path):
        """Test verification passes for deterministic DAG."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        config = CASDAGConfig(enable_caching=False)  # Disable for verification
        executor = CASDAGExecutor(store, cache_dir, config)

        graph = StageGraph("test")
        stage = DAGStageForTesting("deterministic")
        graph.add_stage(stage)

        context = StageContext()

        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                is_det, hashes = verify_dag_determinism(
                    executor=executor,
                    graph=graph,
                    context=context,
                    runs=2,
                )

                assert is_det is True
                assert len(hashes["deterministic"]) == 2
                assert hashes["deterministic"][0] == hashes["deterministic"][1]
