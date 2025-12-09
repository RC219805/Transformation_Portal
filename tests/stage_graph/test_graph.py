"""
Tests for stage graph execution and dependency management.
"""

import pytest
import time
from pathlib import Path
import tempfile

from src.transformation_portal.stage_graph.stage import (
    Stage,
    StageContext,
    StageResult,
    StageStatus,
)
from src.transformation_portal.stage_graph.graph import (
    StageGraph,
    GraphBuilder,
    GraphExecution,
)


class InputStage(Stage):
    """Stage with no dependencies."""
    
    def compute(self, context: StageContext) -> StageResult:
        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={"value": 10},
        )
    
    def get_cache_key(self, context: StageContext) -> str:
        return f"{self.name}_{self.version}"


class ProcessingStage(Stage):
    """Stage that depends on input."""
    
    def __init__(self, name: str, depends_on: list, multiplier: float = 2.0):
        super().__init__(name=name)
        self.depends_on_list = depends_on
        self.multiplier = multiplier
    
    def get_dependencies(self) -> list:
        return self.depends_on_list
    
    def compute(self, context: StageContext) -> StageResult:
        value = context.get_artifact("value", 0)
        new_value = value * self.multiplier
        
        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={"value": new_value},
        )
    
    def get_cache_key(self, context: StageContext) -> str:
        value = context.get_artifact("value", 0)
        return f"{self.name}_{self.version}_{value}"


class SlowStage(Stage):
    """Stage that takes time to execute."""
    
    def __init__(self, name: str, delay_ms: float = 100):
        super().__init__(name=name)
        self.delay_ms = delay_ms
    
    def compute(self, context: StageContext) -> StageResult:
        time.sleep(self.delay_ms / 1000.0)
        
        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={"done": True},
        )
    
    def get_cache_key(self, context: StageContext) -> str:
        return f"{self.name}_{self.version}"


def test_graph_add_stage():
    """Test adding stages to graph."""
    graph = StageGraph("test")
    
    stage = InputStage("input")
    graph.add_stage(stage)
    
    assert "input" in graph.stages
    assert graph.get_stage("input") == stage


def test_graph_dependency_validation():
    """Test graph validates dependencies exist."""
    graph = StageGraph("test")
    
    # Should fail - depends on non-existent stage
    processing = ProcessingStage("process", depends_on=["missing"])
    
    with pytest.raises(ValueError, match="not in graph"):
        graph.add_stage(processing)


def test_graph_execution_order():
    """Test topological sort for execution order."""
    graph = StageGraph("test")
    
    # Add stages in random order
    stage_c = ProcessingStage("c", depends_on=["b"])
    stage_a = InputStage("a")
    stage_b = ProcessingStage("b", depends_on=["a"])
    
    graph.add_stage(stage_a)
    graph.add_stage(stage_b)
    graph.add_stage(stage_c)
    
    order = graph.get_execution_order()
    
    # Must respect dependencies
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_graph_sequential_execution():
    """Test sequential graph execution."""
    graph = StageGraph("test")
    
    input_stage = InputStage("input")
    process_stage = ProcessingStage("process", depends_on=["input"], multiplier=3.0)
    
    graph.add_stage(input_stage)
    graph.add_stage(process_stage)
    
    context = StageContext(cache_enabled=False)
    execution = graph.execute(context, parallel=False)
    
    assert execution.success
    assert len(execution.stage_results) == 2
    assert execution.get_result("input").get_artifact("value") == 10
    assert execution.get_result("process").get_artifact("value") == 30


def test_graph_parallel_execution():
    """Test parallel graph execution."""
    graph = StageGraph("test")
    
    # Two independent slow stages
    stage_a = SlowStage("slow_a", delay_ms=100)
    stage_b = SlowStage("slow_b", delay_ms=100)
    
    graph.add_stage(stage_a)
    graph.add_stage(stage_b)
    
    context = StageContext(cache_enabled=False)
    
    start = time.time()
    execution = graph.execute(context, parallel=True, max_workers=2)
    duration = time.time() - start
    
    assert execution.success
    assert len(execution.stage_results) == 2
    
    # Should be faster than sequential (< 150ms vs 200ms)
    assert duration < 0.15


def test_graph_cache_stats():
    """Test graph cache statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        graph = StageGraph("test")
        
        input_stage = InputStage("input")
        graph.add_stage(input_stage)
        
        context = StageContext(
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        
        # First run - cache miss
        execution1 = graph.execute(context)
        assert execution1.cache_miss_count == 1
        assert execution1.cache_hit_count == 0
        
        # Second run - cache hit
        execution2 = graph.execute(context)
        assert execution2.cache_hit_count == 1
        assert execution2.cache_miss_count == 0
        
        # Check stats
        stats = execution2.get_cache_stats()
        assert stats["hit_rate"] == 1.0
        assert stats["cache_hits"] == 1


def test_graph_error_propagation():
    """Test error propagation in graph."""
    
    class FailingStage(Stage):
        def compute(self, context: StageContext) -> StageResult:
            raise ValueError("Test failure")
        
        def get_cache_key(self, context: StageContext) -> str:
            return "failing"
    
    graph = StageGraph("test")
    
    good_stage = InputStage("good")
    bad_stage = FailingStage("bad")
    
    graph.add_stage(good_stage)
    graph.add_stage(bad_stage)
    
    context = StageContext(cache_enabled=False)
    execution = graph.execute(context, parallel=False)
    
    assert not execution.success
    assert execution.error is not None
    assert "bad" in execution.error


def test_graph_builder():
    """Test fluent graph builder."""
    graph = (
        GraphBuilder("test")
        .add(InputStage("input"))
        .add(ProcessingStage("process", depends_on=["input"]))
        .build()
    )
    
    assert len(graph.stages) == 2
    assert "input" in graph.stages
    assert "process" in graph.stages


def test_graph_artifact_propagation():
    """Test artifacts propagate through graph."""
    graph = StageGraph("test")
    
    input_stage = InputStage("input")
    process_stage = ProcessingStage("process", depends_on=["input"], multiplier=2.0)
    
    graph.add_stage(input_stage)
    graph.add_stage(process_stage)
    
    context = StageContext(cache_enabled=False)
    execution = graph.execute(context, parallel=False)
    
    # Artifacts should be in context
    assert context.get_artifact("value") == 20  # 10 * 2


def test_graph_execution_metadata():
    """Test execution metadata tracking."""
    graph = StageGraph("pipeline")
    
    input_stage = InputStage("input")
    graph.add_stage(input_stage)
    
    context = StageContext(cache_enabled=False)
    execution = graph.execute(context, run_id="test-123")
    
    assert execution.run_id == "test-123"
    assert execution.graph_name == "pipeline"
    assert execution.total_duration_ms > 0
    assert len(execution.execution_order) == 1
    assert execution.execution_order[0] == "input"


def test_graph_cycle_detection():
    """Test graph detects cycles."""
    # Manually create cycle (bypassing add_stage validation)
    graph = StageGraph("test")
    
    stage_a = ProcessingStage("a", depends_on=["b"])
    stage_b = ProcessingStage("b", depends_on=["a"])
    
    # Add stages manually
    graph.stages["a"] = stage_a
    graph.stages["b"] = stage_b
    graph._dependency_graph["a"] = {"b"}
    graph._dependency_graph["b"] = {"a"}
    
    with pytest.raises(ValueError, match="cycles"):
        graph.get_execution_order()


def test_graph_complex_dag():
    """Test complex DAG execution."""
    graph = StageGraph("test")
    
    # Diamond pattern: input -> (a, b) -> output
    input_stage = InputStage("input")
    stage_a = ProcessingStage("a", depends_on=["input"], multiplier=2.0)
    stage_b = ProcessingStage("b", depends_on=["input"], multiplier=3.0)
    
    graph.add_stage(input_stage)
    graph.add_stage(stage_a)
    graph.add_stage(stage_b)
    
    context = StageContext(cache_enabled=False)
    execution = graph.execute(context, parallel=True)
    
    assert execution.success
    assert len(execution.stage_results) == 3
    
    # Input executed before a and b
    order = execution.execution_order
    assert order.index("input") < order.index("a")
    assert order.index("input") < order.index("b")
