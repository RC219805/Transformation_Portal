"""Sandboxed DAG node executor.

This module provides the execution engine for running DAG nodes
within isolated sandboxes. It integrates:
- Sandbox for CAS-only IO
- GPU semaphore for GPU isolation
- Execution result tracking

Design:
    Executor → Sandbox → FSGuard → CAS
                 ↓
           GPUSemaphore

Example:
    >>> executor = SandboxExecutor(gpu_semaphore=GPUSemaphore(num_devices=1))
    >>>
    >>> with executor.create_sandbox("llava_001", config, cas) as sandbox:
    ...     result = executor.run_node(
    ...         node=llava_node,
    ...         sandbox=sandbox,
    ...         use_gpu=True,
    ...         image_sha_list=["abc123...", "def456..."],
    ...     )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

from transformation_portal.core.security.fs_guard import FSGuard, get_fs_guard
from transformation_portal.runtime.gpu_semaphore import GPUSemaphore, GPUSlot
from transformation_portal.runtime.sandbox import (
    Sandbox,
    SandboxConfig,
    SandboxError,
    SandboxMetrics,
)
from transformation_portal.storage.cas_store import ArtifactStore

logger = logging.getLogger(__name__)


T = TypeVar("T")


class DAGNodeProtocol(Protocol):
    """Protocol for DAG nodes that can run in a sandbox."""

    def run(self, *, sandbox: Sandbox, **inputs: Any) -> Dict[str, Any]:
        """Execute the node within a sandbox.

        Args:
            sandbox: Execution sandbox for IO
            **inputs: Node-specific inputs

        Returns:
            Dictionary of outputs (typically CAS SHA references)
        """
        ...


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a sandboxed node execution.

    Attributes:
        node_id: ID of the executed node
        outputs: Node outputs (typically CAS SHA references)
        metrics: Execution metrics
        gpu_slot: GPU slot used (if any)
        error: Error message if execution failed
        success: Whether execution succeeded
    """

    node_id: str
    outputs: Dict[str, Any]
    metrics: SandboxMetrics
    gpu_slot: Optional[GPUSlot] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether execution succeeded."""
        return self.error is None


@dataclass
class ExecutorConfig:
    """Configuration for the sandbox executor.

    Attributes:
        workspace_root: Root directory for sandboxes
        cas_root: Root directory for CAS
        max_concurrent_gpu: Maximum concurrent GPU jobs
        cleanup_sandboxes: Whether to cleanup sandboxes after execution
    """

    workspace_root: Path
    cas_root: Path
    max_concurrent_gpu: int = 1
    cleanup_sandboxes: bool = True


class SandboxExecutor:
    """Executor for running DAG nodes in isolated sandboxes.

    Provides:
    - Sandbox creation and management
    - GPU semaphore integration
    - Execution tracking and metrics

    Example:
        >>> executor = SandboxExecutor(
        ...     config=ExecutorConfig(
        ...         workspace_root=Path("/tmp/sandboxes"),
        ...         cas_root=Path("/data/cas"),
        ...     ),
        ... )
        >>>
        >>> result = executor.execute(
        ...     node_id="process_image_001",
        ...     node=ProcessImageNode(backend),
        ...     use_gpu=True,
        ...     image_sha="abc123...",
        ... )
        >>>
        >>> print(f"Output SHA: {result.outputs['result_sha']}")
    """

    def __init__(
        self,
        config: ExecutorConfig,
        fs: Optional[FSGuard] = None,
        cas: Optional[ArtifactStore] = None,
        gpu_semaphore: Optional[GPUSemaphore] = None,
    ) -> None:
        """Initialize the sandbox executor.

        Args:
            config: Executor configuration
            fs: FSGuard instance (uses global if not provided)
            cas: CAS store (creates one if not provided)
            gpu_semaphore: GPU semaphore (creates one if not provided and GPUs available)
        """
        self.config = config
        self.fs = fs or get_fs_guard()
        self.cas = cas or ArtifactStore(config.cas_root)

        # Initialize GPU semaphore if not provided
        if gpu_semaphore is None and config.max_concurrent_gpu > 0:
            try:
                self.gpu_semaphore = GPUSemaphore(num_devices=config.max_concurrent_gpu)
            except Exception as e:
                logger.warning("Failed to initialize GPU semaphore: %s", e)
                self.gpu_semaphore = None
        else:
            self.gpu_semaphore = gpu_semaphore

        # Create workspace root
        config.workspace_root.mkdir(parents=True, exist_ok=True)

        # Execution tracking
        self._execution_count = 0
        self._results: List[ExecutionResult] = []

        logger.info(
            "SandboxExecutor initialized: workspace=%s, cas=%s, gpu=%s",
            config.workspace_root,
            config.cas_root,
            "enabled" if self.gpu_semaphore else "disabled",
        )

    def create_sandbox(
        self,
        node_id: str,
        *,
        enable_gpu: bool = True,
        cleanup_on_exit: Optional[bool] = None,
    ) -> Sandbox:
        """Create a new sandbox for a node.

        Args:
            node_id: Unique identifier for the node
            enable_gpu: Whether GPU access is allowed
            cleanup_on_exit: Override for cleanup behavior

        Returns:
            Configured Sandbox instance
        """
        sandbox_config = SandboxConfig(
            workspace_root=self.config.workspace_root,
            cas_root=self.config.cas_root,
            enable_gpu=enable_gpu,
            cleanup_on_exit=(cleanup_on_exit if cleanup_on_exit is not None else self.config.cleanup_sandboxes),
        )

        return Sandbox(
            node_id=node_id,
            config=sandbox_config,
            fs=self.fs,
            cas=self.cas,
        )

    def run_node(
        self,
        node: DAGNodeProtocol,
        sandbox: Sandbox,
        *,
        use_gpu: bool = False,
        timeout: Optional[float] = None,
        **inputs: Any,
    ) -> ExecutionResult:
        """Run a node within an existing sandbox.

        Args:
            node: DAG node to execute
            sandbox: Sandbox to use for execution
            use_gpu: Whether to acquire GPU before execution
            timeout: GPU acquisition timeout in seconds
            **inputs: Inputs to pass to node.run()

        Returns:
            ExecutionResult with outputs and metrics
        """
        self._execution_count += 1
        gpu_slot: Optional[GPUSlot] = None
        error: Optional[str] = None
        outputs: Dict[str, Any] = {}

        sandbox.start()

        try:
            if use_gpu and self.gpu_semaphore:
                # Acquire GPU with semaphore
                with self.gpu_semaphore.acquire(timeout=timeout) as slot:
                    gpu_slot = slot
                    logger.debug(
                        "Node %s acquired GPU: device_id=%d",
                        sandbox.node_id,
                        slot.device_id,
                    )
                    outputs = node.run(sandbox=sandbox, **inputs)
            else:
                outputs = node.run(sandbox=sandbox, **inputs)

        except Exception as e:
            error = str(e)
            logger.error(
                "Node %s execution failed: %s",
                sandbox.node_id,
                e,
            )

        finally:
            sandbox.finish()

        result = ExecutionResult(
            node_id=sandbox.node_id,
            outputs=outputs,
            metrics=sandbox.metrics,
            gpu_slot=gpu_slot,
            error=error,
        )

        self._results.append(result)
        return result

    def execute(
        self,
        node_id: str,
        node: DAGNodeProtocol,
        *,
        use_gpu: bool = False,
        timeout: Optional[float] = None,
        **inputs: Any,
    ) -> ExecutionResult:
        """Create sandbox and execute node in one call.

        Convenience method that creates a sandbox, runs the node,
        and cleans up based on config.

        Args:
            node_id: Unique identifier for this execution
            node: DAG node to execute
            use_gpu: Whether to acquire GPU
            timeout: GPU acquisition timeout
            **inputs: Inputs to pass to node.run()

        Returns:
            ExecutionResult with outputs and metrics
        """
        sandbox = self.create_sandbox(node_id, enable_gpu=use_gpu)

        return self.run_node(
            node=node,
            sandbox=sandbox,
            use_gpu=use_gpu,
            timeout=timeout,
            **inputs,
        )

    def execute_batch(
        self,
        nodes: List[tuple[str, DAGNodeProtocol, Dict[str, Any]]],
        *,
        use_gpu: bool = False,
    ) -> List[ExecutionResult]:
        """Execute multiple nodes sequentially.

        Args:
            nodes: List of (node_id, node, inputs) tuples
            use_gpu: Whether to use GPU for all nodes

        Returns:
            List of ExecutionResults
        """
        results = []
        for node_id, node, inputs in nodes:
            result = self.execute(
                node_id=node_id,
                node=node,
                use_gpu=use_gpu,
                **inputs,
            )
            results.append(result)
        return results

    @property
    def execution_count(self) -> int:
        """Total number of executions."""
        return self._execution_count

    @property
    def results(self) -> List[ExecutionResult]:
        """All execution results."""
        return list(self._results)

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary.

        Returns:
            Dictionary with execution statistics
        """
        successful = sum(1 for r in self._results if r.success)
        failed = len(self._results) - successful

        total_duration = sum(r.metrics.duration_seconds or 0 for r in self._results)

        return {
            "total_executions": self._execution_count,
            "successful": successful,
            "failed": failed,
            "total_duration_seconds": total_duration,
            "gpu_enabled": self.gpu_semaphore is not None,
        }
