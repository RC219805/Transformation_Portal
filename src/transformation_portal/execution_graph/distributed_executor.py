"""Distributed DAG executor with Ray and local backends.

This module provides a unified interface for executing DAGs either:
- Locally (in-process or via GPU session pool)
- Distributed (via Ray cluster)

Design:
    The executor wraps a PriorityDAGScheduler and executes nodes using
    the selected backend. Same DAG definition works with both backends,
    enabling seamless scaling from local development to cluster execution.

Example:
    >>> scheduler = PriorityDAGScheduler()
    >>> scheduler.add_node("ingest", ingest_node)
    >>> scheduler.add_node("process", process_node, deps=["ingest"])
    >>>
    >>> # Local execution
    >>> executor = DistributedDAGExecutor(scheduler, backend="local")
    >>> results = executor.run()
    >>>
    >>> # Ray execution (requires Ray cluster)
    >>> executor = DistributedDAGExecutor(scheduler, backend="ray")
    >>> results = executor.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Optional Ray import
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False


class DistributedExecutorError(RuntimeError):
    """Raised for distributed executor errors."""


@dataclass
class ExecutionConfig:
    """Configuration for distributed execution.

    Attributes:
        backend: Execution backend ("local" or "ray")
        ray_address: Ray cluster address (default: auto-detect)
        num_gpus_per_task: GPUs per task for Ray (0 = CPU only)
        max_retries: Maximum retries for failed tasks
        timeout_per_node: Timeout per node in seconds
    """

    backend: str = "local"
    ray_address: Optional[str] = None
    num_gpus_per_task: float = 0
    max_retries: int = 0
    timeout_per_node: Optional[float] = None


class DistributedDAGExecutor:
    """Distributed DAG executor with pluggable backends.

    Supports local execution and Ray-based distributed execution
    with the same DAG definition.

    Example:
        >>> from transformation_portal.execution_graph.scheduler import PriorityDAGScheduler
        >>>
        >>> scheduler = PriorityDAGScheduler()
        >>> scheduler.add_node("preprocess", preprocess_node)
        >>> scheduler.add_node("inference", inference_node, deps=["preprocess"])
        >>>
        >>> # Local execution
        >>> executor = DistributedDAGExecutor(scheduler)
        >>> results = executor.run()
        >>>
        >>> # Ray execution
        >>> config = ExecutionConfig(backend="ray", num_gpus_per_task=1)
        >>> executor = DistributedDAGExecutor(scheduler, config=config)
        >>> results = executor.run()
    """

    def __init__(
        self,
        scheduler: "PriorityDAGScheduler",  # type: ignore
        config: Optional[ExecutionConfig] = None,
    ) -> None:
        """Initialize distributed executor.

        Args:
            scheduler: PriorityDAGScheduler with nodes added
            config: Execution configuration

        Raises:
            DistributedExecutorError: If Ray backend requested but not available
        """
        self.scheduler = scheduler
        self.config = config or ExecutionConfig()

        if self.config.backend == "ray":
            if not RAY_AVAILABLE:
                raise DistributedExecutorError(
                    "Ray is not installed. Install with: pip install ray"
                )
            self._init_ray()

    def _init_ray(self) -> None:
        """Initialize Ray runtime."""
        if not ray.is_initialized():
            init_kwargs = {"ignore_reinit_error": True}
            if self.config.ray_address:
                init_kwargs["address"] = self.config.ray_address
            ray.init(**init_kwargs)
            logger.info("Ray initialized: %s", ray.cluster_resources())

    def run(
        self,
        *,
        inputs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute the DAG using configured backend.

        Args:
            inputs: Optional initial inputs for the DAG

        Returns:
            Dictionary of node_id -> outputs

        Raises:
            DistributedExecutorError: If execution fails
        """
        logger.info(
            "Executing DAG with backend='%s', nodes=%d",
            self.config.backend,
            len(self.scheduler.nodes),
        )

        if self.config.backend == "local":
            return self._run_local(inputs)
        elif self.config.backend == "ray":
            return self._run_ray(inputs)
        else:
            raise DistributedExecutorError(
                f"Unknown backend: {self.config.backend}"
            )

    def _run_local(
        self,
        inputs: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute DAG locally using scheduler."""
        return self.scheduler.run(inputs=inputs)

    def _run_ray(
        self,
        inputs: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute DAG using Ray.

        Uses Ray's remote execution for parallel node execution
        while respecting dependencies.
        """
        results: dict[str, Any] = dict(inputs or {})
        futures: dict[str, Any] = {}

        # Create Ray remote function based on GPU requirements
        if self.config.num_gpus_per_task > 0:
            @ray.remote(num_gpus=self.config.num_gpus_per_task)
            def execute_node(node: Any, node_inputs: dict) -> Any:
                result = node.run(**node_inputs)
                if hasattr(result, "outputs"):
                    return result.outputs
                return result
        else:
            @ray.remote
            def execute_node(node: Any, node_inputs: dict) -> Any:
                result = node.run(**node_inputs)
                if hasattr(result, "outputs"):
                    return result.outputs
                return result

        # Get execution order
        execution_order = self.scheduler.get_execution_order()

        # Submit tasks in topological order
        for node_id in execution_order:
            scheduled_node = self.scheduler.nodes[node_id]
            deps = scheduled_node.deps

            # Gather dependency results
            if deps:
                # Wait for dependencies
                dep_results = {}
                for dep in deps:
                    if dep in futures:
                        dep_results[dep] = ray.get(futures[dep])
                    elif dep in results:
                        dep_results[dep] = results[dep]

                futures[node_id] = execute_node.remote(
                    scheduled_node.node,
                    dep_results,
                )
            else:
                futures[node_id] = execute_node.remote(
                    scheduled_node.node,
                    {},
                )

        # Collect results
        for node_id, future in futures.items():
            try:
                results[node_id] = ray.get(
                    future,
                    timeout=self.config.timeout_per_node,
                )
            except Exception as exc:
                logger.error("Node '%s' failed: %s", node_id, exc)
                raise DistributedExecutorError(
                    f"Node '{node_id}' execution failed: {exc}"
                ) from exc

        return results

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        if self.config.backend == "ray" and RAY_AVAILABLE and ray.is_initialized():
            # Don't shutdown Ray by default as it may be shared
            pass


def create_executor(
    scheduler: "PriorityDAGScheduler",
    *,
    backend: str = "local",
    **kwargs: Any,
) -> DistributedDAGExecutor:
    """Factory function to create a distributed executor.

    Args:
        scheduler: DAG scheduler
        backend: Execution backend ("local" or "ray")
        **kwargs: Additional config options

    Returns:
        Configured DistributedDAGExecutor
    """
    config = ExecutionConfig(backend=backend, **kwargs)
    return DistributedDAGExecutor(scheduler, config=config)
