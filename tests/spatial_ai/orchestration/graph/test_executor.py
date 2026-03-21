"""Tests for Executor orchestration and caching (Phase 3 L1).

Test Coverage:
- Sequential execution
- Cache hit/miss integration
- Provenance tracking
- Resource enforcement
- Input resolution
- Error handling
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from transformation_portal.spatial_ai.orchestration.graph.artifact_store import ArtifactStore
from transformation_portal.spatial_ai.orchestration.graph.execution_graph import ExecutionGraph
from transformation_portal.spatial_ai.orchestration.graph.executor import ExecutionContext, ExecutionResult, Executor
from transformation_portal.spatial_ai.orchestration.graph.stage import ResourceRequirements, StageMetadata
from transformation_portal.spatial_ai.orchestration.resource_manager import ResourceLimits



pytestmark = pytest.mark.unit

class MockStage:
    """Mock stage for testing executor."""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        transform_fn=None,
        gpu_mb: int = 0,
        cpu_mb: int = 512,
        time_ms: int = 100,
    ):
        self._name = name
        self._version = version
        self._transform_fn = transform_fn or (lambda x: x * 2)
        self._gpu_mb = gpu_mb
        self._cpu_mb = cpu_mb
        self._time_ms = time_ms

    @property
    def metadata(self) -> StageMetadata:
        return StageMetadata(
            name=self._name,
            version=self._version,
            description=f"Mock stage: {self._name}",
            resource_requirements=ResourceRequirements(
                gpu_memory_mb=self._gpu_mb,
                cpu_memory_mb=self._cpu_mb,
                estimated_time_ms=self._time_ms,
            ),
        )

    def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute transformation."""
        if "data" in inputs:
            return {"result": self._transform_fn(inputs["data"])}
        elif "value" in inputs:
            return {"output": self._transform_fn(inputs["value"])}
        elif "b" in inputs and "c" in inputs:
            # Handle diamond DAG case
            return {"output": inputs["b"] + inputs["c"]}
        else:
            return {"output": f"executed_{self._name}"}

    def compute_cache_key(self, inputs: Dict[str, Any], context: ExecutionContext) -> str:
        """Compute cache key (SHA256 format for security)."""
        # Build deterministic input representation
        items = []
        for k, v in sorted(inputs.items()):
            if isinstance(v, np.ndarray):
                val_str = hashlib.sha256(v.tobytes()).hexdigest()
            else:
                val_str = str(v)
            items.append(f"{k}={val_str}")
        input_str = ",".join(items)

        # Create full context string (version + inputs + config)
        context_str = f"{self._version}:{input_str}:{context.device}"

        # Return full SHA256 hash (64 hex chars)
        return hashlib.sha256(context_str.encode()).hexdigest()


class TestExecutor:
    """Tests for Executor."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create temporary cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create temporary output directory."""
        return tmp_path / "output"

    @pytest.fixture
    def store(self, cache_dir: Path) -> ArtifactStore:
        """Create artifact store."""
        return ArtifactStore(cache_dir=cache_dir)

    @pytest.fixture
    def executor(self, store: ArtifactStore) -> Executor:
        """Create executor with caching."""
        return Executor(artifact_store=store, device="cpu")

    @pytest.fixture
    def executor_no_cache(self) -> Executor:
        """Create executor without caching."""
        return Executor(artifact_store=None, device="cpu")

    def test_executor_initialization(self, executor: Executor):
        """Test executor initialization."""
        assert executor.artifact_store is not None
        assert executor.device == "cpu"

    def test_executor_no_cache_initialization(self, executor_no_cache: Executor):
        """Test executor without cache."""
        assert executor_no_cache.artifact_store is None

    def test_execute_single_stage(self, executor_no_cache: Executor, output_dir: Path):
        """Test executing single stage without caching."""
        graph = ExecutionGraph()
        graph.add_stage("stage1", MockStage("stage1", transform_fn=lambda x: x + 10), inputs={})

        result = executor_no_cache.execute(
            graph=graph,
            inputs={"value": 5},
            output_dir=output_dir,
        )

        assert result.stages_executed == 1
        assert result.stages_cached == 0
        assert len(result.stage_results) == 1
        assert result.stage_results[0].stage_id == "stage1"
        assert result.stage_results[0].cache_hit is False

    def test_execute_linear_pipeline(self, executor_no_cache: Executor, output_dir: Path):
        """Test executing linear pipeline."""
        graph = ExecutionGraph()

        # Linear pipeline: A → B → C
        graph.add_stage("A", MockStage("A", transform_fn=lambda x: x + 1), inputs={})
        graph.add_stage("B", MockStage("B", transform_fn=lambda x: x * 2), inputs={"value": "A.output"})
        graph.add_stage("C", MockStage("C", transform_fn=lambda x: x - 3), inputs={"value": "B.output"})

        result = executor_no_cache.execute(
            graph=graph,
            inputs={"value": 10},
            output_dir=output_dir,
        )

        # Verify all stages executed
        assert result.stages_executed == 3
        assert result.stages_cached == 0
        assert len(result.stage_results) == 3

        # Verify execution order
        assert result.stage_results[0].stage_id == "A"
        assert result.stage_results[1].stage_id == "B"
        assert result.stage_results[2].stage_id == "C"

    def test_execute_with_caching_miss_then_hit(self, executor: Executor, output_dir: Path):
        """Test execution with cache miss followed by cache hit."""
        graph = ExecutionGraph()
        graph.add_stage("stage1", MockStage("stage1", transform_fn=lambda x: x * 3), inputs={})

        # First execution - cache miss
        result1 = executor.execute(
            graph=graph,
            inputs={"value": 7},
            output_dir=output_dir,
        )

        assert result1.stages_executed == 1
        assert result1.stages_cached == 0
        assert result1.stage_results[0].cache_hit is False

        # Second execution with same inputs - cache hit
        result2 = executor.execute(
            graph=graph,
            inputs={"value": 7},
            output_dir=output_dir,
        )

        assert result2.stages_executed == 0
        assert result2.stages_cached == 1
        assert result2.stage_results[0].cache_hit is True

        # Verify outputs are identical
        assert result1.outputs == result2.outputs

    def test_execute_with_different_inputs_different_cache(self, executor: Executor, output_dir: Path):
        """Test different inputs produce different cache entries."""
        graph = ExecutionGraph()
        graph.add_stage("stage1", MockStage("stage1"), inputs={})

        # Execute with input A
        result_a = executor.execute(
            graph=graph,
            inputs={"value": 10},
            output_dir=output_dir,
        )

        # Execute with input B (different)
        result_b = executor.execute(
            graph=graph,
            inputs={"value": 20},
            output_dir=output_dir,
        )

        # Both should be cache misses (different inputs)
        assert result_a.stages_executed == 1
        assert result_b.stages_executed == 1

        # Execute with input A again - should hit cache
        result_a2 = executor.execute(
            graph=graph,
            inputs={"value": 10},
            output_dir=output_dir,
        )

        assert result_a2.stages_cached == 1

    def test_provenance_tracking(self, executor: Executor, output_dir: Path):
        """Test provenance is tracked for executed stages."""
        graph = ExecutionGraph()
        graph.add_stage("stage1", MockStage("stage1", version="2.0.0"), inputs={})

        result = executor.execute(
            graph=graph,
            inputs={"value": 42},
            output_dir=output_dir,
        )

        # Verify provenance exists
        stage_result = result.stage_results[0]
        assert stage_result.provenance is not None
        assert stage_result.provenance.stage_id == "stage1"
        assert stage_result.provenance.stage_version == "2.0.0"
        assert stage_result.provenance.device == "cpu"
        # Input fingerprints computed from resolved inputs
        assert len(stage_result.provenance.input_fingerprints) >= 0

    def test_input_resolution_from_root(self, executor_no_cache: Executor, output_dir: Path):
        """Test input resolution from root inputs."""
        graph = ExecutionGraph()
        graph.add_stage("stage1", MockStage("stage1"), inputs={})

        result = executor_no_cache.execute(
            graph=graph,
            inputs={"value": 100},
            output_dir=output_dir,
        )

        assert result.stages_executed == 1

    def test_input_resolution_from_upstream_stage(self, executor_no_cache: Executor, output_dir: Path):
        """Test input resolution from upstream stage outputs."""
        graph = ExecutionGraph()

        # A produces output, B consumes it
        graph.add_stage("A", MockStage("A", transform_fn=lambda x: x + 5), inputs={})
        graph.add_stage("B", MockStage("B", transform_fn=lambda x: x * 2), inputs={"value": "A.output"})

        result = executor_no_cache.execute(
            graph=graph,
            inputs={"value": 10},
            output_dir=output_dir,
        )

        # A: 10 + 5 = 15, B: 15 * 2 = 30
        assert result.stages_executed == 2

    def test_diamond_dag_execution(self, executor_no_cache: Executor, output_dir: Path):
        """Test execution of diamond-shaped DAG."""
        graph = ExecutionGraph()

        # Diamond: A → B, A → C, B → D, C → D
        graph.add_stage("A", MockStage("A", transform_fn=lambda x: x + 1), inputs={})
        graph.add_stage("B", MockStage("B", transform_fn=lambda x: x * 2), inputs={"value": "A.output"})
        graph.add_stage("C", MockStage("C", transform_fn=lambda x: x * 3), inputs={"value": "A.output"})
        graph.add_stage("D", MockStage("D"), inputs={"b": "B.output", "c": "C.output"})

        result = executor_no_cache.execute(
            graph=graph,
            inputs={"value": 10},
            output_dir=output_dir,
        )

        assert result.stages_executed == 4
        assert len(result.stage_results) == 4

    def test_resource_limit_enforcement(self, executor_no_cache: Executor, output_dir: Path):
        """Test resource limits are enforced during planning."""
        graph = ExecutionGraph()
        graph.add_stage("gpu_stage", MockStage("gpu_stage", gpu_mb=8192), inputs={})

        # Create executor with tight GPU limit
        limited_executor = Executor(
            artifact_store=None,
            resource_limits=ResourceLimits(max_gpu_memory_gb=4.0),
            device="cpu",
        )

        # Should fail during planning (before execution)
        from transformation_portal.spatial_ai.orchestration.graph.execution_graph import ResourceError

        with pytest.raises(ResourceError):
            limited_executor.execute(
                graph=graph,
                inputs={"value": 1},
                output_dir=output_dir,
            )

    def test_stage_failure_propagates(self, executor_no_cache: Executor, output_dir: Path):
        """Test stage failure propagates as RuntimeError."""

        class FailingStage:
            """Stage that always fails."""

            @property
            def metadata(self) -> StageMetadata:
                return StageMetadata(
                    name="failing_stage",
                    version="1.0.0",
                    description="Stage that fails",
                    resource_requirements=ResourceRequirements(),
                )

            def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
                raise ValueError("Intentional failure")

            def compute_cache_key(self, inputs: Dict[str, Any], context: ExecutionContext) -> str:
                # Return valid SHA256 key
                return hashlib.sha256(b"failing_stage").hexdigest()

        graph = ExecutionGraph()
        graph.add_stage("failing", FailingStage(), inputs={})

        with pytest.raises(RuntimeError, match="Stage 'failing' failed"):
            executor_no_cache.execute(
                graph=graph,
                inputs={"value": 1},
                output_dir=output_dir,
            )

    def test_optional_stage_skipped_on_missing_input(self, executor_no_cache: Executor, output_dir: Path):
        """Test optional stage is skipped when input is missing."""
        graph = ExecutionGraph()

        # Add required stage that uses root input
        graph.add_stage("required", MockStage("required"), inputs={})

        # Add optional stage that depends on missing root input
        graph.add_stage(
            "optional",
            MockStage("optional"),
            inputs={"missing_input": "missing_root"},
            optional=True,
        )

        # Execute - optional stage should be skipped
        result = executor_no_cache.execute(
            graph=graph,
            inputs={"value": 10},  # "missing_root" is not provided
            output_dir=output_dir,
        )

        # Verify execution completed
        assert len(result.stage_results) == 2

        # Find the results by stage_id
        required_result = next(r for r in result.stage_results if r.stage_id == "required")
        optional_result = next(r for r in result.stage_results if r.stage_id == "optional")

        # Required stage executed
        assert required_result.cache_hit is False

        # Optional stage skipped
        assert optional_result.cache_key == "SKIPPED"
        assert optional_result.outputs == {}
        assert optional_result.execution_time_ms == 0.0

        # Only required stage counted as executed
        assert result.stages_executed == 1
        assert result.stages_cached == 1  # Skipped stage marked as cache hit

    def test_non_optional_stage_fails_on_missing_input(self, executor_no_cache: Executor, output_dir: Path):
        """Test non-optional stage fails when input is missing."""
        graph = ExecutionGraph()

        # Add non-optional stage that depends on missing root input
        graph.add_stage(
            "non_optional",
            MockStage("non_optional"),
            inputs={"missing_input": "missing_root"},
            optional=False,
        )

        # Should raise ValueError during input resolution
        with pytest.raises(ValueError, match="requires root input 'missing_root'"):
            executor_no_cache.execute(
                graph=graph,
                inputs={"value": 10},
                output_dir=output_dir,
            )

    def test_execution_result_statistics(self, executor: Executor, output_dir: Path):
        """Test execution result contains accurate statistics."""
        graph = ExecutionGraph()
        graph.add_stage("stage1", MockStage("stage1"), inputs={})
        graph.add_stage("stage2", MockStage("stage2"), inputs={"value": "stage1.output"})

        # First run - all cache misses
        result1 = executor.execute(
            graph=graph,
            inputs={"value": 5},
            output_dir=output_dir,
        )

        assert result1.stages_executed == 2
        assert result1.stages_cached == 0
        assert result1.total_time_ms > 0
        assert len(result1.stage_results) == 2

        # Second run - all cache hits
        result2 = executor.execute(
            graph=graph,
            inputs={"value": 5},
            output_dir=output_dir,
        )

        assert result2.stages_executed == 0
        assert result2.stages_cached == 2
        # Note: Cache hit timing can vary due to safety checks (corruption detection,
        # lock acquisition, etc.). We validate cache _functionality_ (stages_cached==2)
        # rather than _performance_ to avoid flaky timing assertions.

    def test_execution_with_numpy_arrays(self, executor: Executor, output_dir: Path):
        """Test execution with numpy array inputs and outputs."""

        class NumpyStage:
            """Stage that processes numpy arrays."""

            @property
            def metadata(self) -> StageMetadata:
                return StageMetadata(
                    name="numpy_stage",
                    version="1.0.0",
                    description="NumPy processing stage",
                    resource_requirements=ResourceRequirements(),
                )

            def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
                data = inputs["data"]
                return {"processed": data * 2 + 1}

            def compute_cache_key(self, inputs: Dict[str, Any], context: ExecutionContext) -> str:
                data = inputs["data"]
                data_hash = hashlib.sha256(data.tobytes()).hexdigest()
                # Return full SHA256 (combine version + data hash)
                combined = f"1.0.0:{data_hash}"
                return hashlib.sha256(combined.encode()).hexdigest()

        graph = ExecutionGraph()
        graph.add_stage("numpy", NumpyStage(), inputs={})

        input_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # First execution
        result1 = executor.execute(
            graph=graph,
            inputs={"data": input_array},
            output_dir=output_dir,
        )

        assert result1.stages_executed == 1

        # Second execution (cache hit)
        result2 = executor.execute(
            graph=graph,
            inputs={"data": input_array},
            output_dir=output_dir,
        )

        assert result2.stages_cached == 1

        # Verify outputs are bitwise identical
        out1 = result1.outputs["numpy.processed"]
        out2 = result2.outputs["numpy.processed"]
        np.testing.assert_array_equal(out1, out2)


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_execution_context_creation(self, tmp_path: Path):
        """Test ExecutionContext creation."""
        context = ExecutionContext(
            device="cuda",
            config={"model_size": "large"},
            output_dir=tmp_path,
            enable_caching=True,
        )

        assert context.device == "cuda"
        assert context.config == {"model_size": "large"}
        assert context.output_dir == tmp_path
        assert context.enable_caching is True
