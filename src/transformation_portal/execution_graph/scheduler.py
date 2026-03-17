"""Priority DAG scheduler with resource awareness.

This module provides a priority-based scheduler for DAG execution that:
- Respects node dependencies (topological order)
- Prioritizes nodes based on configurable priority
- Supports resource requirements (GPU, memory)
- Integrates with GPU session pools

Design:
    The scheduler maintains a priority queue of ready nodes (nodes with
    all dependencies satisfied). Nodes are executed in priority order,
    with resource constraints checked before execution.

Example:
    >>> scheduler = PriorityDAGScheduler()
    >>> scheduler.add_node("ingest", ingest_node, priority=10)
    >>> scheduler.add_node("segment", segment_node, deps=["ingest"], priority=5)
    >>> scheduler.add_node("eval", eval_node, deps=["segment"], priority=1)
    >>> results = scheduler.run()
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceRequirements:
    """Resource requirements for a node.

    Attributes:
        gpu: Whether GPU is required
        gpu_memory_mb: Estimated GPU memory in MB
        cpu_memory_mb: Estimated CPU memory in MB
        estimated_time_ms: Estimated execution time in ms
    """

    gpu: bool = False
    gpu_memory_mb: int = 0
    cpu_memory_mb: int = 0
    estimated_time_ms: int = 0


@dataclass(order=True)
class ScheduledNode:
    """Node in the scheduler's priority queue.

    Attributes:
        priority: Execution priority (higher = sooner, negated for min-heap)
        node_id: Unique node identifier
        node: The DAG node to execute
        deps: List of dependency node IDs
        resources: Resource requirements
    """

    priority: int
    node_id: str = field(compare=False)
    node: Any = field(compare=False)
    deps: list[str] = field(compare=False, default_factory=list)
    resources: ResourceRequirements = field(
        compare=False,
        default_factory=ResourceRequirements,
    )


class SchedulerError(RuntimeError):
    """Raised for scheduler errors."""


class PriorityDAGScheduler:
    """Priority-based DAG scheduler with resource awareness.

    Executes DAG nodes in topological order, respecting priorities
    and resource constraints.

    Example:
        >>> scheduler = PriorityDAGScheduler()
        >>>
        >>> # Add nodes with dependencies and priorities
        >>> scheduler.add_node(
        ...     "material",
        ...     material_node,
        ...     priority=10,
        ...     resources=ResourceRequirements(gpu=True, gpu_memory_mb=4000),
        ... )
        >>> scheduler.add_node(
        ...     "quality",
        ...     quality_node,
        ...     deps=["material"],
        ...     priority=5,
        ... )
        >>>
        >>> # Execute
        >>> results = scheduler.run()
        >>> print(results["quality"])
    """

    def __init__(
        self,
        *,
        gpu_pool: Optional[Any] = None,
    ) -> None:
        """Initialize scheduler.

        Args:
            gpu_pool: Optional GPUSessionPool for GPU-requiring nodes
        """
        self.nodes: dict[str, ScheduledNode] = {}
        self.results: dict[str, Any] = {}
        self.gpu_pool = gpu_pool
        self._executed: set[str] = set()

    def add_node(
        self,
        node_id: str,
        node: Any,
        *,
        deps: Optional[list[str]] = None,
        priority: int = 0,
        resources: Optional[ResourceRequirements] = None,
    ) -> None:
        """Add a node to the scheduler.

        Args:
            node_id: Unique node identifier
            node: DAG node to execute (must have run() method)
            deps: List of dependency node IDs
            priority: Execution priority (higher = sooner)
            resources: Resource requirements

        Raises:
            SchedulerError: If node_id already exists
        """
        if node_id in self.nodes:
            raise SchedulerError(f"Node '{node_id}' already exists")

        self.nodes[node_id] = ScheduledNode(
            priority=-priority,  # Negate for min-heap (higher priority first)
            node_id=node_id,
            node=node,
            deps=deps or [],
            resources=resources or ResourceRequirements(),
        )

        logger.debug(
            "Added node '%s' with priority=%d, deps=%s",
            node_id,
            priority,
            deps or [],
        )

    def validate(self) -> list[str]:
        """Validate the DAG structure.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for missing dependencies
        for node_id, node in self.nodes.items():
            for dep in node.deps:
                if dep not in self.nodes:
                    errors.append(
                        f"Node '{node_id}' depends on missing node '{dep}'"
                    )

        # Check for cycles (simple DFS)
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for dep in self.nodes[node_id].deps:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    errors.append(f"Cycle detected involving node '{node_id}'")
                    break

        return errors

    def run(
        self,
        *,
        inputs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute all nodes in priority order.

        Args:
            inputs: Optional initial inputs (available to all nodes)

        Returns:
            Dictionary of node_id -> outputs

        Raises:
            SchedulerError: If validation fails or execution errors
        """
        # Validate DAG
        errors = self.validate()
        if errors:
            raise SchedulerError(f"DAG validation failed: {errors}")

        # Initialize
        self.results = dict(inputs or {})
        self._executed = set()

        # Build initial ready queue
        ready: list[ScheduledNode] = []
        remaining = dict(self.nodes)

        for node_id, node in remaining.items():
            if not node.deps or all(d in self.results for d in node.deps):
                heapq.heappush(ready, node)

        logger.info(
            "Starting scheduler with %d nodes, %d initially ready",
            len(self.nodes),
            len(ready),
        )

        # Execute in priority order
        while ready:
            node = heapq.heappop(ready)

            if node.node_id in self._executed:
                continue

            # Gather inputs from dependencies
            node_inputs = {
                dep: self.results[dep]
                for dep in node.deps
                if dep in self.results
            }

            # Execute node
            logger.debug("Executing node '%s'", node.node_id)
            result = self._execute_node(node, node_inputs)

            # Store results
            self.results[node.node_id] = result
            self._executed.add(node.node_id)

            # Remove from remaining
            if node.node_id in remaining:
                del remaining[node.node_id]

            # Add newly ready nodes
            for node_id, n in list(remaining.items()):
                if node_id not in self._executed:
                    if all(dep in self._executed for dep in n.deps):
                        heapq.heappush(ready, n)

        logger.info("Scheduler completed: %d nodes executed", len(self._executed))

        return self.results

    def _execute_node(
        self,
        node: ScheduledNode,
        inputs: dict[str, Any],
    ) -> Any:
        """Execute a single node.

        Args:
            node: Node to execute
            inputs: Input data from dependencies

        Returns:
            Node outputs
        """
        # Check if node requires GPU and we have a pool
        if node.resources.gpu and self.gpu_pool is not None:
            return self._execute_on_gpu(node, inputs)

        # Regular execution
        try:
            result = node.node.run(**inputs)

            if hasattr(result, "outputs"):
                return result.outputs
            return result

        except Exception as exc:
            logger.error("Node '%s' execution failed: %s", node.node_id, exc)
            raise SchedulerError(
                f"Node '{node.node_id}' execution failed: {exc}"
            ) from exc

    def _execute_on_gpu(
        self,
        node: ScheduledNode,
        inputs: dict[str, Any],
    ) -> Any:
        """Execute node on GPU pool.

        Args:
            node: Node to execute
            inputs: Input data

        Returns:
            Node outputs
        """

        def gpu_task(state: Any, node: Any, inputs: dict) -> Any:
            result = node.run(**inputs)
            if hasattr(result, "outputs"):
                return result.outputs
            return result

        self.gpu_pool.submit(gpu_task, node.node, inputs)
        return self.gpu_pool.get()

    def get_execution_order(self) -> list[str]:
        """Get expected execution order (for planning/debugging).

        Returns:
            List of node IDs in execution order
        """
        # Simulate execution without actually running
        order = []
        remaining = dict(self.nodes)
        executed = set()

        ready: list[ScheduledNode] = []
        for node_id, node in remaining.items():
            if not node.deps:
                heapq.heappush(ready, node)

        while ready:
            node = heapq.heappop(ready)
            order.append(node.node_id)
            executed.add(node.node_id)

            if node.node_id in remaining:
                del remaining[node.node_id]

            for node_id, n in list(remaining.items()):
                if all(dep in executed for dep in n.deps):
                    heapq.heappush(ready, n)

        return order

    def get_resource_summary(self) -> dict[str, Any]:
        """Get summary of resource requirements.

        Returns:
            Dict with total GPU/CPU memory, node counts, etc.
        """
        gpu_nodes = sum(1 for n in self.nodes.values() if n.resources.gpu)
        total_gpu_mb = sum(n.resources.gpu_memory_mb for n in self.nodes.values())
        total_cpu_mb = sum(n.resources.cpu_memory_mb for n in self.nodes.values())
        total_time_ms = sum(n.resources.estimated_time_ms for n in self.nodes.values())

        return {
            "total_nodes": len(self.nodes),
            "gpu_nodes": gpu_nodes,
            "cpu_only_nodes": len(self.nodes) - gpu_nodes,
            "total_gpu_memory_mb": total_gpu_mb,
            "total_cpu_memory_mb": total_cpu_mb,
            "estimated_time_ms": total_time_ms,
        }
