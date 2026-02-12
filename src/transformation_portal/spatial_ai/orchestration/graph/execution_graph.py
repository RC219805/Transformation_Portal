"""Execution graph for DAG-based pipeline orchestration (Phase 3 L1).

Provides explicit directed acyclic graph (DAG) modeling for spatial AI pipelines.

Design Principles (ADR-029):
- Explicit dependencies (no implicit sequencing)
- Topological execution order (respects data flow)
- Resource validation (fail-fast before execution)
- Introspectable (can query stages, dependencies, resource totals)
- Cycle detection (prevent infinite loops)

Key Features:
1. DAG Construction: Add stages with explicit input mappings
2. Topological Sort: Compute valid execution order (Kahn's algorithm)
3. Resource Planning: Aggregate resource requirements across stages
4. Validation: Detect cycles, missing dependencies, resource violations

Example:
    >>> graph = ExecutionGraph()
    >>> graph.add_stage("ingest", IngestStage(), inputs={})
    >>> graph.add_stage(
    ...     "segment",
    ...     SAM2Stage(),
    ...     inputs={"linear_rgb": "ingest.linear_rgb"},
    ... )
    >>>
    >>> # Validate and plan
    >>> plan = graph.plan(resource_limits=ResourceLimits(max_gpu_memory_gb=4))
    >>> print(f"Execution order: {[s.stage_id for s in plan.stages]}")
    >>> print(f"Total GPU memory: {plan.total_gpu_memory_mb}MB")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .stage import Stage, StageMetadata


class GraphError(Exception):
    """Raised when graph structure is invalid (cycles, missing deps)."""

    pass


class ResourceError(Exception):
    """Raised when resource requirements exceed limits."""

    pass


@dataclass
class StageNode:
    """Node in execution graph.

    Attributes:
        stage_id: Unique identifier for this stage instance.
        stage: Stage implementation.
        inputs: Input mapping (input_name → "source_stage.output_name").
        optional: If True, skip stage if inputs unavailable.

    Design notes:
    - stage_id is unique per graph (allows multiple instances of same stage).
    - inputs are explicit references to upstream outputs.
    - optional stages gracefully degrade (skip if inputs missing).
    """

    stage_id: str
    stage: Stage
    inputs: Dict[str, str]  # input_name → "stage_id.output_name"
    optional: bool = False


@dataclass
class ExecutionPlan:
    """Execution plan generated from graph validation.

    Attributes:
        stages: Stages in topologically sorted order.
        total_gpu_memory_mb: Peak GPU memory across all stages.
        total_cpu_memory_mb: Total CPU memory across all stages.
        estimated_time_ms: Total estimated execution time.
        checkpoints: Stage IDs to checkpoint (cache outputs).

    Design notes:
    - Stages are sorted to respect dependencies (no stage before its inputs).
    - GPU memory is peak (not sum) since stages execute sequentially in L1.
    - CPU memory is sum (stages may accumulate in-memory state).
    - Time is sum (sequential execution in L1).
    """

    stages: List[StageNode]
    total_gpu_memory_mb: int
    total_cpu_memory_mb: int
    estimated_time_ms: int
    checkpoints: List[str] = field(default_factory=list)


class ExecutionGraph:
    """Directed acyclic graph (DAG) of pipeline stages.

    Key features:
    - Explicit dependencies (data lineage via input mappings)
    - Topological execution order (valid scheduling)
    - Resource aggregation (memory budgets, time estimates)
    - Cycle detection (fail-fast on circular dependencies)
    - Introspectable (query stages, dependencies, resources)

    Design (ADR-029):
    - Domain-specific (spatial AI orchestration, not generic workflows)
    - Fail-fast validation (detect errors before execution)
    - Immutable after planning (no dynamic graph modification)

    Example:
        >>> graph = ExecutionGraph()
        >>>
        >>> # Add stages
        >>> graph.add_stage("ingest", IngestStage(), inputs={})
        >>> graph.add_stage(
        ...     "segment",
        ...     SAM2Stage(),
        ...     inputs={"linear_rgb": "ingest.linear_rgb"},
        ... )
        >>> graph.add_stage(
        ...     "materials",
        ...     MaterialStage(),
        ...     inputs={
        ...         "masks": "segment.masks",
        ...         "linear_rgb": "ingest.linear_rgb",
        ...     },
        ... )
        >>>
        >>> # Validate and plan
        >>> from transformation_portal.spatial_ai.orchestration.resource_manager import ResourceLimits
        >>> limits = ResourceLimits(max_gpu_memory_gb=4)
        >>> plan = graph.plan(resource_limits=limits)
        >>>
        >>> # Inspect plan
        >>> for node in plan.stages:
        ...     print(f"{node.stage_id}: {node.stage.metadata.name}")
        ingest: linear_ingest
        segment: sam2_segmentation
        materials: pbr_materials
    """

    def __init__(self):
        """Initialize empty execution graph."""
        self._stages: Dict[str, StageNode] = {}
        self._execution_order: Optional[List[str]] = None

    def add_stage(
        self,
        stage_id: str,
        stage: Stage,
        inputs: Dict[str, str],
        optional: bool = False,
    ) -> None:
        """Add stage to graph.

        Args:
            stage_id: Unique identifier for this stage instance.
            stage: Stage implementation (must implement Stage protocol).
            inputs: Input mapping (input_name → "source_stage.output_name").
                    Empty dict for root stages (no dependencies).
            optional: If True, skip stage if inputs unavailable.

        Raises:
            ValueError: If stage_id already exists.

        Design notes:
        - Does NOT validate graph on add (defer to plan() for fail-fast).
        - Input format: {"input_name": "source_stage.output_name"}
        - Source stages must exist when plan() is called.

        Example:
            >>> graph.add_stage("ingest", IngestStage(), inputs={})
            >>> graph.add_stage(
            ...     "segment",
            ...     SAM2Stage(),
            ...     inputs={"linear_rgb": "ingest.linear_rgb"},
            ... )
        """
        if stage_id in self._stages:
            raise ValueError(f"Stage '{stage_id}' already exists in graph")

        node = StageNode(
            stage_id=stage_id,
            stage=stage,
            inputs=inputs,
            optional=optional,
        )

        self._stages[stage_id] = node
        self._execution_order = None  # Invalidate cached order

    def plan(
        self,
        resource_limits: Optional[Any] = None,  # ResourceLimits from resource_manager
    ) -> ExecutionPlan:
        """Generate execution plan with validation.

        Validates graph structure and computes execution order:
        1. Topological sort (detect cycles, compute valid order)
        2. Dependency validation (check all inputs exist)
        3. Resource aggregation (compute memory/time totals)
        4. Resource validation (check against limits)

        Args:
            resource_limits: Optional resource constraints (ResourceLimits).

        Returns:
            ExecutionPlan with topologically sorted stages and resource totals.

        Raises:
            GraphError: If graph has cycles or missing dependencies.
            ResourceError: If resource limits exceeded.

        Example:
            >>> plan = graph.plan()
            >>> print(f"Stages: {len(plan.stages)}")
            >>> print(f"GPU memory: {plan.total_gpu_memory_mb}MB")
            >>> print(f"Estimated time: {plan.estimated_time_ms}ms")
        """
        # Topological sort (validates graph structure)
        execution_order = self._topological_sort()

        # Validate dependencies
        self._validate_dependencies()

        # Compute resource totals
        total_gpu_mb = 0
        total_cpu_mb = 0
        total_time_ms = 0
        checkpoints = []

        sorted_stages = []
        for stage_id in execution_order:
            node = self._stages[stage_id]
            reqs = node.stage.metadata.resource_requirements

            # GPU memory is peak (sequential execution in L1)
            total_gpu_mb = max(total_gpu_mb, reqs.gpu_memory_mb)

            # CPU memory is sum (stages may accumulate state)
            total_cpu_mb += reqs.cpu_memory_mb

            # Time is sum (sequential execution in L1)
            total_time_ms += reqs.estimated_time_ms

            # Collect checkpoint stages
            from .stage import CheckpointPolicy

            if node.stage.metadata.checkpoint_policy == CheckpointPolicy.ALWAYS:
                checkpoints.append(stage_id)

            sorted_stages.append(node)

        # Validate resource limits (if provided)
        if resource_limits:
            if hasattr(resource_limits, "max_gpu_memory_gb") and resource_limits.max_gpu_memory_gb:
                limit_mb = resource_limits.max_gpu_memory_gb * 1024
                if total_gpu_mb > limit_mb:
                    raise ResourceError(f"Graph requires {total_gpu_mb}MB GPU memory, " f"limit is {limit_mb}MB")

            if hasattr(resource_limits, "max_cpu_memory_gb") and resource_limits.max_cpu_memory_gb:
                limit_mb = resource_limits.max_cpu_memory_gb * 1024
                if total_cpu_mb > limit_mb:
                    raise ResourceError(f"Graph requires {total_cpu_mb}MB CPU memory, " f"limit is {limit_mb}MB")

        return ExecutionPlan(
            stages=sorted_stages,
            total_gpu_memory_mb=total_gpu_mb,
            total_cpu_memory_mb=total_cpu_mb,
            estimated_time_ms=total_time_ms,
            checkpoints=checkpoints,
        )

    def get_stage(self, stage_id: str) -> Optional[StageNode]:
        """Get stage node by ID.

        Args:
            stage_id: Stage identifier.

        Returns:
            StageNode if found, None otherwise.
        """
        return self._stages.get(stage_id)

    def get_all_stages(self) -> Dict[str, StageNode]:
        """Get all stages in graph.

        Returns:
            Dictionary of stage_id → StageNode.
        """
        return dict(self._stages)

    def _topological_sort(self) -> List[str]:
        """Compute topological execution order using Kahn's algorithm.

        Returns:
            List of stage IDs in valid execution order.

        Raises:
            GraphError: If graph contains cycles.

        Algorithm:
        1. Compute in-degree for each node (number of dependencies)
        2. Start with nodes having in-degree 0 (no dependencies)
        3. Remove nodes and update in-degrees until graph is empty
        4. If graph not empty, there's a cycle

        Design notes:
        - Kahn's algorithm is O(V + E) where V = stages, E = dependencies.
        - Detects cycles by checking if all nodes were processed.
        - Deterministic (stable sort order for nodes with same in-degree).
        """
        if not self._stages:
            return []

        # Compute in-degrees
        in_degree: Dict[str, int] = {sid: 0 for sid in self._stages}

        for node in self._stages.values():
            for source_ref in node.inputs.values():
                # Parse "stage_id.output_name" → "stage_id"
                source_stage = source_ref.split(".")[0]
                if source_stage in self._stages:
                    in_degree[node.stage_id] += 1

        # Queue of stages with no dependencies
        queue = [sid for sid, deg in sorted(in_degree.items()) if deg == 0]
        result = []

        while queue:
            # Process stage with no dependencies
            stage_id = queue.pop(0)
            result.append(stage_id)

            # Update in-degrees for downstream stages
            for other_id, other_node in self._stages.items():
                for source_ref in other_node.inputs.values():
                    source_stage = source_ref.split(".")[0]
                    if source_stage == stage_id:
                        in_degree[other_id] -= 1
                        if in_degree[other_id] == 0:
                            # Maintain deterministic order
                            queue.append(other_id)
                            queue.sort()

        # Check for cycles
        if len(result) != len(self._stages):
            unprocessed = set(self._stages.keys()) - set(result)
            raise GraphError(f"Graph contains cycles involving stages: {unprocessed}")

        return result

    def _validate_dependencies(self) -> None:
        """Validate all input dependencies exist.

        Raises:
            GraphError: If any stage references non-existent upstream stage.

        Design notes:
        - Called during plan() to fail-fast on configuration errors.
        - Checks that all "stage_id.output_name" references are valid.
        """
        for node in self._stages.values():
            for input_name, source_ref in node.inputs.items():
                # Parse "stage_id.output_name"
                parts = source_ref.split(".")
                if len(parts) < 2:
                    raise GraphError(
                        f"Stage '{node.stage_id}' has invalid input reference: "
                        f"'{source_ref}' (expected 'stage_id.output_name')"
                    )

                source_stage = parts[0]
                if source_stage not in self._stages:
                    raise GraphError(f"Stage '{node.stage_id}' depends on non-existent stage: " f"'{source_stage}'")
