"""
Stage graph execution and dependency management.

Manages execution order, parallel execution, and dependency tracking
for a collection of stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .stage import Stage, StageContext, StageResult

logger = logging.getLogger(__name__)


@dataclass
class GraphExecution:
    """
    Record of a complete graph execution.

    Tracks all stage results and overall metrics.
    """
    # Execution identification
    run_id: str
    graph_name: str

    # Stage results
    stage_results: Dict[str, StageResult] = field(default_factory=dict)

    # Execution order
    execution_order: List[str] = field(default_factory=list)

    # Metrics
    total_duration_ms: float = 0.0
    cache_hit_count: int = 0
    cache_miss_count: int = 0

    # Status
    success: bool = True
    error: Optional[str] = None

    def get_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for a specific stage."""
        return self.stage_results.get(stage_name)

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self.cache_hit_count + self.cache_miss_count
        hit_rate = float(self.cache_hit_count) / float(total) if total > 0 else 0.0

        # Calculate speedup: if hit_rate is 1.0, avoid division by zero
        if hit_rate >= 1.0:
            speedup_estimate = float('inf')  # Perfect caching
        elif hit_rate > 0:
            speedup_estimate = 1.0 / (1.0 - hit_rate)
        else:
            speedup_estimate = 1.0

        return {
            "total_stages": len(self.stage_results),
            "cache_hits": self.cache_hit_count,
            "cache_misses": self.cache_miss_count,
            "hit_rate": hit_rate,
            "speedup_estimate": speedup_estimate,
        }


class StageGraph:
    """
    Manages a directed acyclic graph (DAG) of processing stages.

    Features:
    - Automatic dependency resolution
    - Parallel execution where possible
    - Cache-aware optimization
    - Error propagation
    """

    def __init__(self, name: str = "pipeline"):
        """
        Initialize stage graph.

        Args:
            name: Graph name for identification
        """
        self.name = name
        self.stages: Dict[str, Stage] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

    def add_stage(self, stage: Stage):
        """
        Add stage to graph.

        Args:
            stage: Stage instance
        """
        if stage.name in self.stages:
            raise ValueError(f"Stage '{stage.name}' already exists in graph")

        self.stages[stage.name] = stage
        self._dependency_graph[stage.name] = set(stage.get_dependencies())

        # Validate dependencies exist
        for dep in self._dependency_graph[stage.name]:
            if dep not in self.stages:
                raise ValueError(
                    f"Stage '{stage.name}' depends on '{dep}', "
                    f"but '{dep}' is not in graph"
                )

    def get_stage(self, name: str) -> Optional[Stage]:
        """Get stage by name."""
        return self.stages.get(name)

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order.

        Returns:
            List of stage names in execution order

        Raises:
            ValueError: If graph has cycles
        """
        # Kahn's algorithm for topological sort
        in_degree = {name: 0 for name in self.stages}

        # Calculate in-degrees
        for name in self.stages:
            for dep in self._dependency_graph[name]:
                in_degree[name] += 1

        # Queue of stages with no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Process stages with no remaining dependencies
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for dependent stages
            for name, deps in self._dependency_graph.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        # Check for cycles
        if len(result) != len(self.stages):
            raise ValueError("Graph contains cycles")

        return result

    def execute(
        self,
        context: StageContext,
        run_id: Optional[str] = None,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> GraphExecution:
        """
        Execute all stages in dependency order.

        Args:
            context: Execution context
            run_id: Unique run identifier
            parallel: Enable parallel execution of independent stages
            max_workers: Maximum parallel workers

        Returns:
            Graph execution record
        """
        import uuid

        if run_id is None:
            run_id = str(uuid.uuid4())

        start_time = time.time()

        execution = GraphExecution(
            run_id=run_id,
            graph_name=self.name,
        )

        context.run_id = run_id

        try:
            execution_order = self.get_execution_order()

            if parallel:
                self._execute_parallel(
                    execution_order, context, execution, max_workers
                )
            else:
                self._execute_sequential(execution_order, context, execution)

        except Exception as e:
            self.logger.error(f"Graph execution failed: {e}")
            execution.success = False
            execution.error = str(e)

        execution.total_duration_ms = (time.time() - start_time) * 1000

        return execution

    def _execute_sequential(
        self,
        execution_order: List[str],
        context: StageContext,
        execution: GraphExecution,
    ):
        """Execute stages sequentially."""
        for stage_name in execution_order:
            stage = self.stages[stage_name]

            # Execute stage
            result = stage.execute(context)

            # Record result
            execution.stage_results[stage_name] = result
            execution.execution_order.append(stage_name)

            if result.cache_hit:
                execution.cache_hit_count += 1
            else:
                execution.cache_miss_count += 1

            # Propagate artifacts to context
            for name, value in result.artifacts.items():
                context.set_artifact(name, value)

            # Stop on failure
            if not result.is_success():
                self.logger.error(f"Stage {stage_name} failed: {result.error}")
                execution.success = False
                execution.error = f"Stage {stage_name} failed: {result.error}"
                break

    def _execute_parallel(
        self,
        execution_order: List[str],
        context: StageContext,
        execution: GraphExecution,
        max_workers: int,
    ):
        """
        Execute stages in parallel where possible.

        Respects dependencies while maximizing parallelism.
        """
        completed: Set[str] = set()
        pending = set(execution_order)

        while pending:
            # Find stages ready to execute (all dependencies satisfied)
            ready = []
            for stage_name in pending:
                deps = self._dependency_graph[stage_name]
                if deps.issubset(completed):
                    ready.append(stage_name)

            if not ready:
                # No progress possible - likely cycle or error
                execution.success = False
                execution.error = "Unable to resolve dependencies"
                break

            # Execute ready stages in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                for stage_name in ready:
                    stage = self.stages[stage_name]
                    future = executor.submit(stage.execute, context)
                    futures[future] = stage_name

                # Collect results
                for future in as_completed(futures):
                    stage_name = futures[future]
                    result = future.result()

                    # Record result
                    execution.stage_results[stage_name] = result
                    execution.execution_order.append(stage_name)

                    if result.cache_hit:
                        execution.cache_hit_count += 1
                    else:
                        execution.cache_miss_count += 1

                    # Propagate artifacts
                    for name, value in result.artifacts.items():
                        context.set_artifact(name, value)

                    # Check success
                    if result.is_success():
                        completed.add(stage_name)
                        pending.remove(stage_name)
                    else:
                        self.logger.error(f"Stage {stage_name} failed: {result.error}")
                        execution.success = False
                        execution.error = f"Stage {stage_name} failed: {result.error}"
                        return  # Stop on failure


class GraphBuilder:
    """
    Fluent builder for constructing stage graphs.

    Simplifies graph construction with method chaining.
    """

    def __init__(self, name: str = "pipeline"):
        """
        Initialize builder.

        Args:
            name: Graph name
        """
        self.graph = StageGraph(name)

    def add(self, stage: Stage) -> GraphBuilder:
        """
        Add stage to graph.

        Args:
            stage: Stage instance

        Returns:
            Self for chaining
        """
        self.graph.add_stage(stage)
        return self

    def build(self) -> StageGraph:
        """
        Build and return the graph.

        Returns:
            Constructed stage graph
        """
        return self.graph
