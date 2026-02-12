"""Tests for ExecutionGraph DAG construction and validation (Phase 3 L1).

Test Coverage:
- DAG construction (add_stage, get_stage)
- Topological sort
- Cycle detection
- Dependency validation
- Resource planning
- Resource limit enforcement
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import pytest

from transformation_portal.spatial_ai.orchestration.graph.execution_graph import (
    ExecutionGraph,
    ExecutionPlan,
    GraphError,
    ResourceError,
    StageNode,
)
from transformation_portal.spatial_ai.orchestration.graph.stage import CheckpointPolicy, ResourceRequirements, StageMetadata
from transformation_portal.spatial_ai.orchestration.resource_manager import ResourceLimits


class MockStage:
    """Mock stage for testing."""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        gpu_mb: int = 0,
        cpu_mb: int = 512,
        time_ms: int = 1000,
        checkpoint_policy: CheckpointPolicy = CheckpointPolicy.AUTO,
    ):
        self._name = name
        self._version = version
        self._gpu_mb = gpu_mb
        self._cpu_mb = cpu_mb
        self._time_ms = time_ms
        self._checkpoint_policy = checkpoint_policy

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
            checkpoint_policy=self._checkpoint_policy,
        )

    def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        return {"output": f"result_{self._name}"}

    def compute_cache_key(self, inputs: Dict[str, Any], context: Any) -> str:
        return f"{self._version}:mock_key_{self._name}"


class TestExecutionGraph:
    """Tests for ExecutionGraph."""

    def test_empty_graph(self):
        """Test empty graph."""
        graph = ExecutionGraph()
        assert len(graph.get_all_stages()) == 0

        # Planning empty graph should work
        plan = graph.plan()
        assert len(plan.stages) == 0
        assert plan.total_gpu_memory_mb == 0
        assert plan.total_cpu_memory_mb == 0
        assert plan.estimated_time_ms == 0

    def test_add_single_stage(self):
        """Test adding single stage."""
        graph = ExecutionGraph()
        stage = MockStage("test_stage")

        graph.add_stage("stage1", stage, inputs={})

        assert len(graph.get_all_stages()) == 1
        node = graph.get_stage("stage1")
        assert node is not None
        assert node.stage_id == "stage1"
        assert node.stage == stage
        assert node.inputs == {}

    def test_add_duplicate_stage_id(self):
        """Test adding stage with duplicate ID raises error."""
        graph = ExecutionGraph()
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")

        graph.add_stage("duplicate", stage1, inputs={})

        with pytest.raises(ValueError, match="Stage 'duplicate' already exists"):
            graph.add_stage("duplicate", stage2, inputs={})

    def test_add_stage_with_dependencies(self):
        """Test adding stage with dependencies."""
        graph = ExecutionGraph()

        ingest = MockStage("ingest")
        segment = MockStage("segment")

        graph.add_stage("ingest", ingest, inputs={})
        graph.add_stage("segment", segment, inputs={"linear_rgb": "ingest.linear_rgb"})

        assert len(graph.get_all_stages()) == 2
        segment_node = graph.get_stage("segment")
        assert segment_node.inputs == {"linear_rgb": "ingest.linear_rgb"}

    def test_simple_topological_sort(self):
        """Test topological sort on simple linear pipeline."""
        graph = ExecutionGraph()

        graph.add_stage("stage1", MockStage("stage1"), inputs={})
        graph.add_stage("stage2", MockStage("stage2"), inputs={"input": "stage1.output"})
        graph.add_stage("stage3", MockStage("stage3"), inputs={"input": "stage2.output"})

        plan = graph.plan()

        # Verify order
        stage_ids = [node.stage_id for node in plan.stages]
        assert stage_ids == ["stage1", "stage2", "stage3"]

    def test_topological_sort_diamond_dag(self):
        """Test topological sort on diamond-shaped DAG."""
        graph = ExecutionGraph()

        # Diamond: A → B, A → C, B → D, C → D
        graph.add_stage("A", MockStage("A"), inputs={})
        graph.add_stage("B", MockStage("B"), inputs={"input": "A.output"})
        graph.add_stage("C", MockStage("C"), inputs={"input": "A.output"})
        graph.add_stage("D", MockStage("D"), inputs={"b": "B.output", "c": "C.output"})

        plan = graph.plan()

        # Verify order (A must be first, D must be last)
        stage_ids = [node.stage_id for node in plan.stages]
        assert stage_ids[0] == "A"
        assert stage_ids[-1] == "D"
        assert "B" in stage_ids and "C" in stage_ids

    def test_cycle_detection_self_loop(self):
        """Test cycle detection for self-loop."""
        graph = ExecutionGraph()

        # This would create a self-loop: A → A
        graph.add_stage("A", MockStage("A"), inputs={"input": "A.output"})

        with pytest.raises(GraphError, match="Graph contains cycles"):
            graph.plan()

    def test_cycle_detection_two_node_cycle(self):
        """Test cycle detection for two-node cycle."""
        graph = ExecutionGraph()

        # This creates a cycle: A → B → A
        graph.add_stage("A", MockStage("A"), inputs={"input": "B.output"})
        graph.add_stage("B", MockStage("B"), inputs={"input": "A.output"})

        with pytest.raises(GraphError, match="Graph contains cycles"):
            graph.plan()

    def test_cycle_detection_three_node_cycle(self):
        """Test cycle detection for three-node cycle."""
        graph = ExecutionGraph()

        # Cycle: A → B → C → A
        graph.add_stage("A", MockStage("A"), inputs={"input": "C.output"})
        graph.add_stage("B", MockStage("B"), inputs={"input": "A.output"})
        graph.add_stage("C", MockStage("C"), inputs={"input": "B.output"})

        with pytest.raises(GraphError, match="Graph contains cycles"):
            graph.plan()

    def test_missing_dependency(self):
        """Test validation detects missing dependencies."""
        graph = ExecutionGraph()

        # Stage references non-existent upstream stage
        graph.add_stage("stage1", MockStage("stage1"), inputs={"input": "nonexistent.output"})

        with pytest.raises(GraphError, match="depends on non-existent stage"):
            graph.plan()

    def test_invalid_input_reference_format(self):
        """Test validation detects invalid input reference format."""
        graph = ExecutionGraph()

        # Add stage that provides output
        graph.add_stage("upstream", MockStage("upstream"), inputs={})

        # Test invalid dotted ref (stage with empty output name)
        graph.add_stage("downstream", MockStage("downstream"), inputs={"input": "upstream."})

        with pytest.raises(GraphError, match="invalid input reference"):
            graph.plan()

    def test_invalid_input_reference_empty_stage(self):
        """Test validation detects empty stage in dotted reference."""
        graph = ExecutionGraph()

        # Test invalid dotted ref (empty stage name)
        graph.add_stage("stage1", MockStage("stage1"), inputs={"input": ".output"})

        with pytest.raises(GraphError, match="invalid input reference"):
            graph.plan()

    def test_root_input_reference(self):
        """Test that root inputs (dotless refs) are allowed."""
        graph = ExecutionGraph()

        # Root input (dotless ref) should be allowed in validation
        # (actual validation happens at runtime in executor)
        graph.add_stage("stage1", MockStage("stage1"), inputs={"input": "root_input"})

        # Should not raise during graph validation
        plan = graph.plan()
        assert len(plan.stages) == 1

    def test_resource_aggregation_sequential(self):
        """Test resource aggregation for sequential pipeline."""
        graph = ExecutionGraph()

        # Sequential pipeline with known resource requirements
        graph.add_stage("stage1", MockStage("stage1", gpu_mb=1024, cpu_mb=512, time_ms=1000), inputs={})
        graph.add_stage(
            "stage2",
            MockStage("stage2", gpu_mb=2048, cpu_mb=1024, time_ms=2000),
            inputs={"input": "stage1.output"},
        )
        graph.add_stage(
            "stage3",
            MockStage("stage3", gpu_mb=512, cpu_mb=256, time_ms=500),
            inputs={"input": "stage2.output"},
        )

        plan = graph.plan()

        # GPU memory is peak (max across stages)
        assert plan.total_gpu_memory_mb == 2048

        # CPU memory is sum (stages accumulate state)
        assert plan.total_cpu_memory_mb == 512 + 1024 + 256

        # Time is sum (sequential execution)
        assert plan.estimated_time_ms == 1000 + 2000 + 500

    def test_resource_limit_enforcement_gpu(self):
        """Test resource limit enforcement for GPU memory."""
        graph = ExecutionGraph()

        # Add stage that requires 4GB GPU
        graph.add_stage("gpu_stage", MockStage("gpu_stage", gpu_mb=4096), inputs={})

        # Set limit to 2GB
        limits = ResourceLimits(max_gpu_memory_gb=2.0)

        with pytest.raises(ResourceError, match="requires 4096MB GPU memory"):
            graph.plan(resource_limits=limits)

    def test_resource_limit_enforcement_cpu(self):
        """Test resource limit enforcement for CPU memory."""
        graph = ExecutionGraph()

        # Add stages that sum to > 8GB CPU
        graph.add_stage("stage1", MockStage("stage1", cpu_mb=4096), inputs={})
        graph.add_stage("stage2", MockStage("stage2", cpu_mb=4096), inputs={"input": "stage1.output"})
        graph.add_stage("stage3", MockStage("stage3", cpu_mb=2048), inputs={"input": "stage2.output"})

        # Set limit to 8GB (sum is 10GB)
        limits = ResourceLimits(max_cpu_memory_gb=8.0)

        with pytest.raises(ResourceError, match="requires.*CPU memory"):
            graph.plan(resource_limits=limits)

    def test_checkpoint_policy_collection(self):
        """Test collection of checkpointed stages."""
        graph = ExecutionGraph()

        graph.add_stage(
            "stage1",
            MockStage("stage1", checkpoint_policy=CheckpointPolicy.ALWAYS),
            inputs={},
        )
        graph.add_stage(
            "stage2",
            MockStage("stage2", checkpoint_policy=CheckpointPolicy.NEVER),
            inputs={"input": "stage1.output"},
        )
        graph.add_stage(
            "stage3",
            MockStage("stage3", checkpoint_policy=CheckpointPolicy.ALWAYS),
            inputs={"input": "stage2.output"},
        )

        plan = graph.plan()

        # Only stages with ALWAYS policy should be in checkpoints
        assert "stage1" in plan.checkpoints
        assert "stage2" not in plan.checkpoints
        assert "stage3" in plan.checkpoints

    def test_optional_stages(self):
        """Test optional stage handling."""
        graph = ExecutionGraph()

        graph.add_stage("required", MockStage("required"), inputs={})
        graph.add_stage(
            "optional",
            MockStage("optional"),
            inputs={"input": "required.output"},
            optional=True,
        )

        # Optional stages should still be in plan
        plan = graph.plan()
        stage_ids = [node.stage_id for node in plan.stages]
        assert "required" in stage_ids
        assert "optional" in stage_ids

        # Verify optional flag is preserved
        optional_node = graph.get_stage("optional")
        assert optional_node.optional is True

    def test_complex_dag(self):
        """Test complex DAG with multiple branches and joins."""
        graph = ExecutionGraph()

        # Complex DAG:
        #     A
        #    / \
        #   B   C
        #  / \ / \
        # D   E   F
        #  \ / \ /
        #   G   H
        #    \ /
        #     I

        graph.add_stage("A", MockStage("A"), inputs={})
        graph.add_stage("B", MockStage("B"), inputs={"a": "A.output"})
        graph.add_stage("C", MockStage("C"), inputs={"a": "A.output"})
        graph.add_stage("D", MockStage("D"), inputs={"b": "B.output"})
        graph.add_stage("E", MockStage("E"), inputs={"b": "B.output", "c": "C.output"})
        graph.add_stage("F", MockStage("F"), inputs={"c": "C.output"})
        graph.add_stage("G", MockStage("G"), inputs={"d": "D.output", "e": "E.output"})
        graph.add_stage("H", MockStage("H"), inputs={"e": "E.output", "f": "F.output"})
        graph.add_stage("I", MockStage("I"), inputs={"g": "G.output", "h": "H.output"})

        plan = graph.plan()

        # Verify all stages are present
        stage_ids = [node.stage_id for node in plan.stages]
        assert len(stage_ids) == 9
        assert all(sid in stage_ids for sid in "ABCDEFGHI")

        # Verify topological order
        stage_indices = {sid: i for i, sid in enumerate(stage_ids)}
        assert stage_indices["A"] < stage_indices["B"]
        assert stage_indices["A"] < stage_indices["C"]
        assert stage_indices["B"] < stage_indices["D"]
        assert stage_indices["B"] < stage_indices["E"]
        assert stage_indices["C"] < stage_indices["E"]
        assert stage_indices["C"] < stage_indices["F"]
        assert stage_indices["D"] < stage_indices["G"]
        assert stage_indices["E"] < stage_indices["G"]
        assert stage_indices["E"] < stage_indices["H"]
        assert stage_indices["F"] < stage_indices["H"]
        assert stage_indices["G"] < stage_indices["I"]
        assert stage_indices["H"] < stage_indices["I"]

    def test_topological_sort_multiple_inputs_same_stage(self):
        """Stage with multiple inputs from same upstream stage is handled correctly (CORRECTNESS).

        This tests the fix for the in-degree bug where a stage with multiple
        inputs from the same upstream stage would have its in-degree incorrectly
        incremented multiple times, causing incorrect topological ordering.
        """
        graph = ExecutionGraph()

        # StageA produces two outputs
        stageA = MockStage("stageA")
        graph.add_stage("stageA", stageA, inputs={})

        # StageB consumes both outputs from stageA
        stageB = MockStage("stageB")
        graph.add_stage(
            "stageB",
            stageB,
            inputs={
                "in1": "stageA.out1",
                "in2": "stageA.out2",
            },
        )

        # Should correctly order: stageA before stageB (not deadlock)
        plan = graph.plan()
        assert len(plan.stages) == 2
        assert plan.stages[0].stage_id == "stageA"
        assert plan.stages[1].stage_id == "stageB"

    def test_topological_sort_complex_multiple_inputs(self):
        """Complex graph with multiple stages having multiple inputs from same upstream."""
        graph = ExecutionGraph()

        # StageA produces multiple outputs
        stageA = MockStage("stageA")
        graph.add_stage("stageA", stageA, inputs={})

        # StageB consumes three outputs from stageA
        stageB = MockStage("stageB")
        graph.add_stage(
            "stageB",
            stageB,
            inputs={
                "in1": "stageA.out1",
                "in2": "stageA.out2",
                "in3": "stageA.out3",
            },
        )

        # StageC consumes from both A and B
        stageC = MockStage("stageC")
        graph.add_stage(
            "stageC",
            stageC,
            inputs={
                "from_a1": "stageA.out1",
                "from_a2": "stageA.out2",
                "from_b": "stageB.out",
            },
        )

        # Should correctly order: A, B, C
        plan = graph.plan()
        stage_ids = [node.stage_id for node in plan.stages]
        assert stage_ids == ["stageA", "stageB", "stageC"]


class TestStageNode:
    """Tests for StageNode dataclass."""

    def test_stage_node_creation(self):
        """Test StageNode creation."""
        stage = MockStage("test")
        node = StageNode(
            stage_id="test_node",
            stage=stage,
            inputs={"input": "upstream.output"},
            optional=False,
        )

        assert node.stage_id == "test_node"
        assert node.stage == stage
        assert node.inputs == {"input": "upstream.output"}
        assert node.optional is False

    def test_stage_node_optional_default(self):
        """Test StageNode optional defaults to False."""
        stage = MockStage("test")
        node = StageNode(
            stage_id="test_node",
            stage=stage,
            inputs={},
        )

        assert node.optional is False


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_execution_plan_creation(self):
        """Test ExecutionPlan creation."""
        stages = [
            StageNode("stage1", MockStage("stage1"), {}),
            StageNode("stage2", MockStage("stage2"), {"input": "stage1.output"}),
        ]

        plan = ExecutionPlan(
            stages=stages,
            total_gpu_memory_mb=2048,
            total_cpu_memory_mb=1024,
            estimated_time_ms=5000,
            checkpoints=["stage1"],
        )

        assert len(plan.stages) == 2
        assert plan.total_gpu_memory_mb == 2048
        assert plan.total_cpu_memory_mb == 1024
        assert plan.estimated_time_ms == 5000
        assert plan.checkpoints == ["stage1"]
