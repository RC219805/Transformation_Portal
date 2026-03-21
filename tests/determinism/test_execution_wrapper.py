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

from transformation_portal.core.cas_dag_executor import (
    CASDAGConfig,
    CASDAGExecutor,
    CASExecutionResult,
    verify_dag_determinism,
)
from transformation_portal.core.execution_wrapper import (
    CacheResult,
    CASExecutor,
    ExecutorConfig,
    FileLock,
    execute_with_caching,
)
from transformation_portal.stage_graph.graph import StageGraph
from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus
from transformation_portal.storage.cas_store import ArtifactStore

pytestmark = pytest.mark.unit


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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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

    def test_dag_nested_numpy_serialization(self, dag_executor_setup):
        """Test DAG executor handles nested numpy arrays in artifacts.

        This is a regression test for the blocking issue where DAG path
        only serialized top-level numpy arrays but not nested ones.
        Now both executors use the shared recursive serialization helpers.
        """
        executor, _ = dag_executor_setup

        class NestedNumpyStage(Stage):
            """Stage that returns nested numpy arrays in artifacts."""

            def __init__(self):
                super().__init__("nested_numpy", "1.0.0")
                self.call_count = 0

            def get_dependencies(self) -> List[str]:
                return []

            def get_cache_key(self, context: StageContext) -> str:
                return hashlib.sha256(f"{self.name}:{self.version}".encode()).hexdigest()

            def compute(self, context: StageContext) -> StageResult:
                self.call_count += 1
                return StageResult(
                    stage_name=self.name,
                    stage_version=self.version,
                    status=StageStatus.COMPLETED,
                    artifacts={
                        "features": {
                            "depth": np.ones((3, 3), dtype=np.float32),
                            "normals": np.zeros((3, 3, 3), dtype=np.float32),
                        },
                        "array_list": [
                            np.array([1, 2, 3], dtype=np.int32),
                            np.array([4, 5, 6], dtype=np.int32),
                        ],
                        "scalar": 42,
                    },
                )

        graph = StageGraph("test_pipeline")
        stage = NestedNumpyStage()
        graph.add_stage(stage)

        context = StageContext()

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution - cache miss
                result1 = executor.execute(graph, context)
                assert result1.success is True
                assert result1.cache_misses == 1
                assert stage.call_count == 1

                # Verify nested structures were correctly stored and loaded
                stage_result = result1.stage_results["nested_numpy"]
                artifacts = stage_result.artifacts

                # Check nested dict with numpy arrays
                assert "features" in artifacts
                assert isinstance(artifacts["features"]["depth"], np.ndarray)
                assert artifacts["features"]["depth"].shape == (3, 3)
                assert artifacts["features"]["depth"].dtype == np.float32
                assert np.all(artifacts["features"]["depth"] == 1.0)

                assert isinstance(artifacts["features"]["normals"], np.ndarray)
                assert artifacts["features"]["normals"].shape == (3, 3, 3)

                # Check list with numpy arrays
                assert "array_list" in artifacts
                assert isinstance(artifacts["array_list"], list)
                assert len(artifacts["array_list"]) == 2
                assert isinstance(artifacts["array_list"][0], np.ndarray)
                assert np.array_equal(artifacts["array_list"][0], np.array([1, 2, 3], dtype=np.int32))

                # Check scalar preserved
                assert artifacts["scalar"] == 42

                # Second execution - cache hit
                result2 = executor.execute(graph, context)
                assert result2.success is True
                assert result2.cache_hits == 1
                assert stage.call_count == 1  # No additional execution

                # Verify cache hit still has correct nested structures
                stage_result2 = result2.stage_results["nested_numpy"]
                artifacts2 = stage_result2.artifacts

                assert isinstance(artifacts2["features"]["depth"], np.ndarray)
                assert np.all(artifacts2["features"]["depth"] == 1.0)
                assert isinstance(artifacts2["array_list"][0], np.ndarray)
                assert np.array_equal(artifacts2["array_list"][0], np.array([1, 2, 3], dtype=np.int32))


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

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
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


class TestCacheHitCompatibilityRegression:
    """Regression tests for cache-hit compatibility validation.

    These tests verify the fix for PR #1226 blocking issue #1:
    is_compatible() was called with wrong kwarg (allow_cpu_fallback vs allow_cross_platform).
    """

    def test_cache_hit_compatibility_check(self, tmp_path):
        """Test that cache hit with metadata runs compatibility check without error."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        # Executor with cross-platform disabled (strict mode)
        config = ExecutorConfig(allow_cross_platform=False)
        executor = CASExecutor(store, cache_dir, config)

        class CompatStage:
            @property
            def name(self) -> str:
                return "compat_test"

            @property
            def version(self) -> str:
                return "1.0.0"

            def __init__(self):
                self.call_count = 0

            def execute(self, inputs: Dict[str, Any], config: Any) -> Dict[str, Any]:
                self.call_count += 1
                return {"value": inputs.get("x", 0) * 2}

        stage = CompatStage()

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution - cache miss
                result1 = executor.execute(
                    stage=stage,
                    inputs={"x": 5},
                    config={},
                )
                assert result1.cache_hit is False
                assert result1.outputs["value"] == 10
                assert stage.call_count == 1

                # Second execution - cache hit with compatibility check
                # This should NOT raise TypeError about allow_cpu_fallback
                result2 = executor.execute(
                    stage=stage,
                    inputs={"x": 5},
                    config={},
                )
                assert result2.cache_hit is True
                assert result2.outputs["value"] == 10
                assert stage.call_count == 1  # No additional call


class TestNumpyOutputRegression:
    """Regression tests for numpy array output handling.

    These tests verify the fix for PR #1226 blocking issue #3:
    jcs_dumpb(outputs) was called on raw outputs containing numpy arrays,
    which would fail serialization. Now we serialize first, then hash.
    """

    def test_numpy_output_cache_miss_and_hit(self, tmp_path):
        """Test stage returning numpy array can cache miss and hit."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        executor = CASExecutor(store, cache_dir)

        class NumpyOutputStage:
            @property
            def name(self) -> str:
                return "numpy_output_stage"

            @property
            def version(self) -> str:
                return "1.0.0"

            def __init__(self):
                self.call_count = 0

            def execute(self, inputs: Dict[str, Any], config: Any) -> Dict[str, Any]:
                self.call_count += 1
                # Return numpy array in outputs - this is the regression case
                return {
                    "depth_map": np.zeros((10, 10), dtype=np.float32),
                    "scalar": 42,
                }

        stage = NumpyOutputStage()

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # First execution - cache miss (should not fail on numpy serialization)
                result1 = executor.execute(
                    stage=stage,
                    inputs={},
                    config={},
                )
                assert result1.cache_hit is False
                assert isinstance(result1.outputs["depth_map"], np.ndarray)
                assert result1.outputs["scalar"] == 42
                assert stage.call_count == 1

                # Second execution - cache hit
                result2 = executor.execute(
                    stage=stage,
                    inputs={},
                    config={},
                )
                assert result2.cache_hit is True
                assert isinstance(result2.outputs["depth_map"], np.ndarray)
                assert result2.outputs["scalar"] == 42
                assert stage.call_count == 1  # No additional call

    def test_complex_numpy_output(self, tmp_path):
        """Test stage with nested numpy arrays in output dict."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        executor = CASExecutor(store, cache_dir)

        class ComplexOutputStage:
            @property
            def name(self) -> str:
                return "complex_stage"

            @property
            def version(self) -> str:
                return "1.0.0"

            def execute(self, inputs: Dict[str, Any], config: Any) -> Dict[str, Any]:
                return {
                    "features": {
                        "depth": np.ones((5, 5), dtype=np.float32),
                        "normals": np.zeros((5, 5, 3), dtype=np.float32),
                    },
                    "metadata": {"count": 25},
                }

        stage = ComplexOutputStage()

        with patch("transformation_portal.core.execution_identity.compute_code_hash") as mock_code:
            with patch("transformation_portal.core.execution_identity.get_env_fingerprint") as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                # Execute - should handle nested numpy arrays
                result = executor.execute(
                    stage=stage,
                    inputs={},
                    config={},
                )
                assert result.cache_hit is False
                assert isinstance(result.outputs["features"]["depth"], np.ndarray)
                assert result.outputs["metadata"]["count"] == 25


class TestCILockfileEnforcement:
    """Tests for CI lockfile enforcement at executor level.

    These tests verify the fix for PR #1226 blocking issue #2:
    compute_cas_id() is called without lockfile_path/lockfile_hash in executors.
    """

    def test_executor_resolves_lockfile(self, tmp_path):
        """Test that executor resolves lockfile path on init."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        executor = CASExecutor(store, cache_dir)

        # Executor should have resolved lockfile path (or None if not found)
        # The key is that it's set, not that it fails
        assert hasattr(executor, "_lockfile_path")

    def test_executor_uses_explicit_lockfile_path(self, tmp_path):
        """Test that executor uses explicit lockfile_path from config."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        # Create a test lockfile
        lockfile = tmp_path / "test-lockfile.txt"
        lockfile.write_text("test-package==1.0.0\n")

        config = ExecutorConfig(lockfile_path=str(lockfile))
        executor = CASExecutor(store, cache_dir, config)

        assert executor._lockfile_path == str(lockfile)

    def test_dag_executor_threads_lockfile(self, tmp_path):
        """Test that DAG executor threads lockfile to stage executor."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        # Create a test lockfile
        lockfile = tmp_path / "dag-lockfile.txt"
        lockfile.write_text("torch==2.2.2\n")

        config = CASDAGConfig(lockfile_path=str(lockfile))
        executor = CASDAGExecutor(store, cache_dir, config)

        # Both executor and internal stage executor should have same path
        assert executor._lockfile_path == str(lockfile)
        assert executor._stage_executor._lockfile_path == str(lockfile)


class TestNumpyInputIdentityRegression:
    """Regression tests for NumPy input identity including shape/dtype.

    These tests verify the fix for the blocking issue:
    _compute_input_ids() was only using tobytes(), creating false cache hits
    between arrays with same bytes but different dtype/shape.
    """

    def test_different_dtype_same_bytes_different_identity(self, tmp_path):
        """Test arrays with same bytes but different dtype get different identities."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        executor = CASExecutor(store, cache_dir)

        # Two arrays with overlapping byte representation but different semantics
        arr_uint32 = np.array([1], dtype=np.uint32)  # 4 bytes: 01 00 00 00
        arr_uint8 = np.array([1, 0, 0, 0], dtype=np.uint8)  # 4 bytes: 01 00 00 00

        # Compute input IDs
        ids1 = executor._compute_input_ids({"arr": arr_uint32})
        ids2 = executor._compute_input_ids({"arr": arr_uint8})

        # They MUST be different despite same raw bytes
        assert ids1[0] != ids2[0], "Arrays with same bytes but different dtype should have different input IDs"

    def test_different_shape_same_bytes_different_identity(self, tmp_path):
        """Test arrays with same bytes but different shape get different identities."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        executor = CASExecutor(store, cache_dir)

        # Two arrays with same bytes but different shapes
        arr_1d = np.array([1, 2, 3, 4], dtype=np.float32)  # shape (4,)
        arr_2d = np.array([[1, 2], [3, 4]], dtype=np.float32)  # shape (2, 2)

        # Verify they have same raw bytes
        assert arr_1d.tobytes() == arr_2d.tobytes()

        # Compute input IDs
        ids1 = executor._compute_input_ids({"arr": arr_1d})
        ids2 = executor._compute_input_ids({"arr": arr_2d})

        # They MUST be different
        assert ids1[0] != ids2[0], "Arrays with same bytes but different shape should have different input IDs"

    def test_identical_arrays_same_identity(self, tmp_path):
        """Test identical arrays get the same identity."""
        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        executor = CASExecutor(store, cache_dir)

        arr1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        arr2 = np.array([[1, 2], [3, 4]], dtype=np.float32)

        ids1 = executor._compute_input_ids({"arr": arr1})
        ids2 = executor._compute_input_ids({"arr": arr2})

        assert ids1[0] == ids2[0], "Identical arrays should have same input ID"


class TestDAGExecutorNumpyIdentityRegression:
    """Regression tests for DAG executor NumPy artifact identity.

    These tests verify that CASDAGExecutor._compute_stage_identity() correctly
    includes dtype and shape when hashing NumPy artifacts, preventing false
    cache hits in DAG execution paths.
    """

    def test_dag_executor_numpy_identity_includes_dtype(self):
        """Test DAG executor uses dtype+shape+data for NumPy artifact identity."""
        from transformation_portal.core.execution_wrapper import _compute_numpy_array_id

        # Two arrays with same raw bytes but different semantics
        arr_uint32 = np.array([1], dtype=np.uint32)  # 4 bytes: 01 00 00 00
        arr_uint8 = np.array([1, 0, 0, 0], dtype=np.uint8)  # 4 bytes: 01 00 00 00

        # Verify they have same raw bytes
        assert arr_uint32.tobytes() == arr_uint8.tobytes()

        # Compute IDs using the shared helper (used by DAG executor)
        id1 = _compute_numpy_array_id(arr_uint32)
        id2 = _compute_numpy_array_id(arr_uint8)

        # They MUST be different despite same raw bytes
        assert id1 != id2, "NumPy arrays with same bytes but different dtype must have different IDs"

    def test_dag_executor_numpy_identity_includes_shape(self):
        """Test DAG executor uses shape in NumPy artifact identity."""
        from transformation_portal.core.execution_wrapper import _compute_numpy_array_id

        # Two arrays with same bytes but different shapes
        arr_1d = np.array([1, 2, 3, 4], dtype=np.float32)  # shape (4,)
        arr_2d = np.array([[1, 2], [3, 4]], dtype=np.float32)  # shape (2, 2)

        # Verify they have same raw bytes
        assert arr_1d.tobytes() == arr_2d.tobytes()

        # Compute IDs using the shared helper
        id1 = _compute_numpy_array_id(arr_1d)
        id2 = _compute_numpy_array_id(arr_2d)

        # They MUST be different despite same raw bytes
        assert id1 != id2, "NumPy arrays with same bytes but different shape must have different IDs"

    def test_single_stage_and_dag_use_same_numpy_identity(self, tmp_path):
        """Test single-stage and DAG executors use identical NumPy identity logic."""
        from transformation_portal.core.execution_wrapper import _compute_numpy_array_id

        cas_root = tmp_path / "cas"
        cache_dir = tmp_path / "cache"
        store = ArtifactStore(cas_root)

        # Create single-stage executor
        single_executor = CASExecutor(store, cache_dir)

        # Test array
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)

        # Get ID from single-stage executor's _compute_input_ids
        single_stage_id = single_executor._compute_input_ids({"arr": arr})[0]

        # Get ID from shared helper (used by DAG executor)
        dag_helper_id = _compute_numpy_array_id(arr)

        # They MUST be identical - same helper is used
        assert single_stage_id == dag_helper_id, "Single-stage and DAG executors must use same NumPy identity logic"


class TestAtomicCacheWrites:
    """Tests for atomic cache write operations.

    These tests verify that cache writes use atomic semantics.
    """

    def test_atomic_write_json_creates_file(self, tmp_path):
        """Test that _atomic_write_json creates a valid JSON file."""
        from transformation_portal.core.execution_wrapper import _atomic_write_json

        test_path = tmp_path / "subdir" / "test.json"
        data = {"key": "value", "number": 42}

        _atomic_write_json(test_path, data)

        assert test_path.exists()
        loaded = json.loads(test_path.read_text())
        assert loaded == data

    def test_atomic_write_json_sorted_keys(self, tmp_path):
        """Test that _atomic_write_json produces sorted keys."""
        from transformation_portal.core.execution_wrapper import _atomic_write_json

        test_path = tmp_path / "test.json"
        data = {"z_key": 1, "a_key": 2, "m_key": 3}

        _atomic_write_json(test_path, data)

        content = test_path.read_text()
        # Keys should appear in sorted order
        assert content.index("a_key") < content.index("m_key") < content.index("z_key")
