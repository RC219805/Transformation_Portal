"""Executor for orchestrating execution graph with caching (Phase 3 L1).

Provides runtime orchestration for DAG-based pipelines with automatic caching and provenance.

Design Principles (ADR-029):
- Sequential execution (L1, respect topological order)
- Automatic caching (transparent to stages)
- Automatic provenance (attach metadata to every artifact)
- Resource enforcement (validate plan upfront, fail-fast on OOM)
- Introspectable results (stages executed, cached, time breakdown)

Execution Flow:
1. Validate graph (topological sort, resource checks)
2. For each stage in order:
   a. Compute cache key
   b. Check cache (load if hit, execute if miss)
   c. Record provenance
   d. Store result (if caching enabled)
3. Return execution result with statistics

Example:
    >>> executor = Executor(
    ...     artifact_store=ArtifactStore(cache_dir=".cache/spatial_ai"),
    ...     resource_limits=ResourceLimits(max_gpu_memory_gb=4),
    ... )
    >>>
    >>> result = executor.execute(
    ...     graph=graph,
    ...     inputs={"input_path": "scene.tiff"},
    ...     output_dir=Path("output/"),
    ... )
    >>>
    >>> print(f"Executed: {result.stages_executed}")
    >>> print(f"Cached: {result.stages_cached}")
    >>> print(f"Total time: {result.total_time_ms}ms")
"""

from __future__ import annotations

import hashlib
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .artifact_store import ArtifactStore, ProvenanceMetadata
from .execution_graph import ExecutionGraph, ExecutionPlan
from .stage import CheckpointPolicy, Stage

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Execution context passed to stages during execution.

    Attributes:
        device: Execution device ("cuda", "cpu", "mps").
        config: Stage-specific configuration overrides.
        output_dir: Output directory for artifacts.
        enable_caching: Whether caching is enabled.

    Design notes:
    - Immutable during execution (no state mutation).
    - Passed to stage.execute() and stage.compute_cache_key().
    - Device is resolved at executor initialization.
    """

    device: str
    config: Dict[str, Any]
    output_dir: Path
    enable_caching: bool = True


@dataclass
class StageExecutionResult:
    """Result from executing a single stage.

    Attributes:
        stage_id: Stage identifier.
        outputs: Stage outputs (key-value pairs).
        cache_hit: True if loaded from cache, False if executed.
        execution_time_ms: Execution time in milliseconds (0 if cache hit).
        cache_key: Content-addressed cache key.
        provenance: Provenance metadata.
    """

    stage_id: str
    outputs: Dict[str, Any]
    cache_hit: bool
    execution_time_ms: float
    cache_key: str
    provenance: Optional[ProvenanceMetadata] = None


@dataclass
class ExecutionResult:
    """Result from graph execution.

    Attributes:
        outputs: Final outputs (key-value pairs).
        stages_executed: Number of stages executed (cache misses).
        stages_cached: Number of stages loaded from cache (cache hits).
        total_time_ms: Total execution time in milliseconds.
        stage_results: Per-stage execution results.
        plan: Execution plan used.

    Design notes:
    - Provides introspection into execution performance.
    - stage_results enables detailed profiling and debugging.
    """

    outputs: Dict[str, Any]
    stages_executed: int
    stages_cached: int
    total_time_ms: float
    stage_results: List[StageExecutionResult] = field(default_factory=list)
    plan: Optional[ExecutionPlan] = None


class Executor:
    """Executor for orchestrating execution graphs with caching.

    Key features:
    - Sequential execution (L1, respects topological order)
    - Automatic caching (content-addressed, transparent to stages)
    - Automatic provenance (full lineage tracking)
    - Resource enforcement (fail-fast on resource limits)
    - Introspectable results (detailed execution statistics)

    Design (ADR-029):
    - Stages are pure functions (no side effects)
    - Caching is deterministic (same inputs → same cache key → same outputs)
    - Provenance is complete (inputs, models, config, timestamps)
    - Execution is fail-fast (validate before execution, not during)

    Example:
        >>> executor = Executor(artifact_store=store, resource_limits=limits)
        >>>
        >>> # Execute graph
        >>> result = executor.execute(
        ...     graph=graph,
        ...     inputs={"input_path": "scene.tiff"},
        ...     output_dir=Path("output/"),
        ... )
        >>>
        >>> # Inspect results
        >>> print(f"Total stages: {len(result.stage_results)}")
        >>> print(f"Cache hits: {result.stages_cached}")
        >>> print(f"Cache misses: {result.stages_executed}")
        >>> print(f"Total time: {result.total_time_ms}ms")
        >>>
        >>> # Per-stage breakdown
        >>> for stage_result in result.stage_results:
        ...     print(f"{stage_result.stage_id}: {stage_result.execution_time_ms}ms " f"(cached: {stage_result.cache_hit})")
    """

    def __init__(
        self,
        artifact_store: Optional[ArtifactStore] = None,
        resource_limits: Optional[Any] = None,  # ResourceLimits from resource_manager
        device: Optional[str] = None,
    ):
        """Initialize executor.

        Args:
            artifact_store: Optional artifact store for caching.
            resource_limits: Optional resource constraints.
            device: Execution device ("cuda", "cpu", "mps"). Auto-detected if None.

        Design notes:
        - If artifact_store is None, caching is disabled.
        - Device is auto-detected if not specified.
        - Resource limits are validated during plan().
        """
        self.artifact_store = artifact_store
        self.resource_limits = resource_limits
        self.device = device or self._detect_device()

    def execute(
        self,
        graph: ExecutionGraph,
        inputs: Dict[str, Any],
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute graph with caching and provenance tracking.

        Execution flow:
        1. Validate graph (plan with resource checks)
        2. Initialize execution context
        3. For each stage in topological order:
           a. Resolve stage inputs from upstream outputs
           b. Compute cache key
           c. Check cache (load if hit, execute if miss)
           d. Store result with provenance (if caching enabled)
        4. Return execution result with statistics

        Args:
            graph: Execution graph to execute.
            inputs: Root inputs (external data, file paths, etc.).
            output_dir: Output directory for artifacts.
            config: Optional global configuration overrides.

        Returns:
            ExecutionResult with outputs and execution statistics.

        Raises:
            GraphError: If graph is invalid (cycles, missing deps).
            ResourceError: If resource limits exceeded.
            RuntimeError: If stage execution fails.

        Example:
            >>> result = executor.execute(
            ...     graph=graph,
            ...     inputs={"input_path": "scene.tiff"},
            ...     output_dir=Path("output/"),
            ...     config={"device": "cuda"},
            ... )
        """
        # Validate graph and compute plan
        plan = graph.plan(resource_limits=self.resource_limits)

        # Initialize execution context
        context = ExecutionContext(
            device=config.get("device", self.device) if config else self.device,
            config=config or {},
            output_dir=Path(output_dir),
            enable_caching=self.artifact_store is not None,
        )

        # Execution state
        stage_outputs: Dict[str, Dict[str, Any]] = {}  # stage_id → outputs
        stage_results: List[StageExecutionResult] = []
        stages_executed = 0
        stages_cached = 0
        total_start = time.time()

        # Execute stages in topological order
        for node in plan.stages:
            logger.info(f"Executing stage: {node.stage_id}")

            # Resolve stage inputs
            try:
                stage_inputs = self._resolve_inputs(node, inputs, stage_outputs)
            except ValueError as e:
                if node.optional:
                    logger.warning(f"Skipping optional stage '{node.stage_id}': {e}")
                    stage_results.append(
                        StageExecutionResult(
                            stage_id=node.stage_id,
                            outputs={},
                            cache_hit=True,
                            execution_time_ms=0.0,
                            cache_key="SKIPPED",
                            provenance=None,
                        )
                    )
                    stage_outputs[node.stage_id] = {}
                    stages_cached += 1  # Count skipped as cached
                    continue
                raise

            # Compute cache key
            cache_key = node.stage.compute_cache_key(stage_inputs, context)

            # Check cache
            cache_hit = False
            execution_time_ms = 0.0
            outputs = None
            provenance = None

            if self.artifact_store and self.artifact_store.exists(cache_key):
                # Cache hit
                try:
                    outputs = self.artifact_store.load(cache_key)
                    provenance = self.artifact_store.load_provenance(cache_key)
                    cache_hit = True
                    stages_cached += 1
                    logger.info(f"Cache hit for {node.stage_id}: {cache_key}")
                except Exception as e:
                    logger.warning(f"Cache load failed for {node.stage_id}: {e}, executing stage")
                    cache_hit = False

            if not cache_hit:
                # Cache miss - execute stage
                stage_start = time.time()
                try:
                    outputs = node.stage.execute(stage_inputs, context)
                    execution_time_ms = (time.time() - stage_start) * 1000
                    stages_executed += 1
                    logger.info(f"Executed {node.stage_id} in {execution_time_ms:.1f}ms")

                    # Store in cache (if enabled AND stage allows caching)
                    # Honor stage's checkpoint_policy: NEVER means skip caching
                    checkpoint_policy = node.stage.metadata.checkpoint_policy
                    should_cache = self.artifact_store is not None and checkpoint_policy != CheckpointPolicy.NEVER

                    if should_cache:
                        provenance = self._create_provenance(
                            cache_key=cache_key,
                            stage_id=node.stage_id,
                            stage=node.stage,
                            inputs=stage_inputs,
                            context=context,
                        )
                        self.artifact_store.store(cache_key, outputs, provenance)
                    else:
                        logger.debug(
                            f"Skipping cache storage for {node.stage_id}: "
                            f"checkpoint_policy={checkpoint_policy.value}"
                        )

                except Exception as e:
                    logger.error(f"Stage {node.stage_id} failed: {e}")
                    raise RuntimeError(f"Stage '{node.stage_id}' failed: {e}") from e

            # Record result
            stage_result = StageExecutionResult(
                stage_id=node.stage_id,
                outputs=outputs,
                cache_hit=cache_hit,
                execution_time_ms=execution_time_ms,
                cache_key=cache_key,
                provenance=provenance,
            )
            stage_results.append(stage_result)

            # Store outputs for downstream stages
            stage_outputs[node.stage_id] = outputs

        # Compute total time
        total_time_ms = (time.time() - total_start) * 1000

        # Collect final outputs (from last stage or all stages)
        # For now, return all stage outputs
        final_outputs = {}
        for stage_id, outputs in stage_outputs.items():
            for key, value in outputs.items():
                final_outputs[f"{stage_id}.{key}"] = value

        return ExecutionResult(
            outputs=final_outputs,
            stages_executed=stages_executed,
            stages_cached=stages_cached,
            total_time_ms=total_time_ms,
            stage_results=stage_results,
            plan=plan,
        )

    def _resolve_inputs(
        self,
        node: Any,  # StageNode
        root_inputs: Dict[str, Any],
        stage_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve stage inputs from upstream outputs.

        Args:
            node: StageNode being executed.
            root_inputs: Root inputs (external data).
            stage_outputs: Outputs from previously executed stages.

        Returns:
            Resolved inputs for stage.

        Raises:
            ValueError: If input reference is invalid.

        Design notes:
        - Input format: {"input_name": "stage_id.output_name"}
        - Root inputs have no "." prefix (e.g., "input_path")
        - Stage outputs have "stage_id.output_name" format
        - If node.inputs is empty, pass through all root inputs
        """
        resolved = {}

        # If no input mappings, pass through all root inputs
        if not node.inputs:
            return dict(root_inputs)

        for input_name, source_ref in node.inputs.items():
            # Parse "stage_id.output_name"
            parts = source_ref.split(".")
            if len(parts) == 1:
                # Root input (no stage prefix)
                if source_ref not in root_inputs:
                    raise ValueError(f"Stage '{node.stage_id}' requires root input '{source_ref}', " "not found")
                resolved[input_name] = root_inputs[source_ref]
            else:
                # Stage output
                source_stage = parts[0]
                output_name = ".".join(parts[1:])

                if source_stage not in stage_outputs:
                    raise ValueError(f"Stage '{node.stage_id}' depends on '{source_stage}', " "which has not executed")

                stage_output = stage_outputs[source_stage]
                if output_name not in stage_output:
                    raise ValueError(
                        f"Stage '{node.stage_id}' requires '{source_ref}', "
                        f"but '{source_stage}' did not produce '{output_name}'"
                    )

                resolved[input_name] = stage_output[output_name]

        return resolved

    def _create_provenance(
        self,
        cache_key: str,
        stage_id: str,
        stage: Stage,
        inputs: Dict[str, Any],
        context: ExecutionContext,
    ) -> ProvenanceMetadata:
        """Create provenance metadata for artifact.

        Args:
            cache_key: Cache key.
            stage_id: Stage identifier.
            stage: Stage instance.
            inputs: Stage inputs.
            context: Execution context.

        Returns:
            ProvenanceMetadata.
        """
        # Compute input fingerprints
        input_fingerprints = {}
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                fingerprint = hashlib.sha256(value.tobytes()).hexdigest()[:16]
            elif isinstance(value, (str, int, float, bool)):
                fingerprint = hashlib.sha256(str(value).encode()).hexdigest()[:16]
            else:
                # Complex types - use string repr
                fingerprint = hashlib.sha256(repr(value).encode()).hexdigest()[:16]
            input_fingerprints[key] = fingerprint

        # Get versions
        numpy_version = np.__version__

        # Note: torch_version is obtained from context.config if provided by L2+ stages.
        # L1 (Tier 1 core) has no ML dependencies, so torch is not imported here.
        # Stages that use torch should include version in their config if provenance tracking is needed.

        # ADR-032: Include platform matrix and env fingerprint for reproducibility
        env_fingerprint = None
        platform_dict = None
        try:
            from ....core.platform_matrix import CURRENT_PLATFORM, get_env_fingerprint

            env_fingerprint = get_env_fingerprint()
            if CURRENT_PLATFORM is not None:
                platform_dict = CURRENT_PLATFORM.to_dict()
        except ImportError:
            # Platform matrix module not available - graceful degradation
            pass

        return ProvenanceMetadata(
            cache_key=cache_key,
            stage_id=stage_id,
            stage_version=stage.metadata.version,
            input_fingerprints=input_fingerprints,
            config_snapshot=dict(context.config),
            timestamp=datetime.now(timezone.utc).isoformat(),
            hostname=platform.node(),
            python_version=sys.version,
            numpy_version=numpy_version,
            torch_version=context.config.get("torch_version"),
            device=context.device,
            model_repo_id=context.config.get("repo_id"),
            model_revision=context.config.get("revision"),
            env_fingerprint=env_fingerprint,
            platform=platform_dict,
        )

    def _detect_device(self) -> str:
        """Detect available execution device.

        Returns:
            Device string ("cuda", "mps", or "cpu").

        Design notes:
        - Prefers GPU if available (CUDA > MPS > CPU).
        - Logs detected device for transparency.
        """
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

        logger.info(f"Detected execution device: {device}")
        return device
