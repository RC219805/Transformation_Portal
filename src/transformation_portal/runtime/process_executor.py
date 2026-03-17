"""Process-isolated execution engine.

This module provides spawn-safe execution of DAG nodes in isolated
child processes. This ensures:
- Memory isolation between nodes
- Fault containment (node crash doesn't kill parent)
- Clean GPU state per execution
- Reproducible execution environment

Design:
    Controller (parent)
       ↓
    ProcessExecutor (spawn)
       ↓
    Sandbox (per node)
       ↓
    FSGuard + CAS

Example:
    >>> executor = ProcessExecutor()
    >>>
    >>> result = executor.run(ProcessTask(
    ...     node_cls=MyNode,
    ...     node_kwargs={"model": "large"},
    ...     inputs={"image_sha": "abc123..."},
    ...     sandbox_config={
    ...         "node_id": "process_001",
    ...         "workspace_root": Path("/tmp/ws"),
    ...         "cas_root": Path("/data/cas"),
    ...     },
    ... ))
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class ProcessTask:
    """Task to execute in a child process.

    Attributes:
        node_cls: Class of the DAG node to instantiate
        node_kwargs: Keyword arguments for node constructor
        inputs: Inputs to pass to node.run()
        sandbox_config: Sandbox configuration dictionary
    """

    node_cls: Type
    node_kwargs: Dict[str, Any]
    inputs: Dict[str, Any]
    sandbox_config: Dict[str, Any]


@dataclass
class ProcessResult:
    """Result from a process execution.

    Attributes:
        outputs: Node outputs (typically CAS SHA references)
        manifest: Sandbox execution manifest
        error: Error message if execution failed
        traceback: Full traceback if error occurred
    """

    outputs: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    traceback: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether execution succeeded."""
        return self.error is None


def _worker(task: ProcessTask, result_queue: mp.Queue) -> None:
    """Worker function executed in child process.

    This function runs in an isolated process with fresh imports.
    It reconstructs the sandbox and executes the node.

    Args:
        task: ProcessTask with execution parameters
        result_queue: Queue to send result back to parent
    """
    try:
        # Set GPU device if specified
        if "gpu_id" in task.sandbox_config:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(task.sandbox_config["gpu_id"])

        # Import inside child process for clean state
        from transformation_portal.core.security.fs_guard import FSGuard
        from transformation_portal.runtime.sandbox import Sandbox, SandboxConfig
        from transformation_portal.storage.cas_store import ArtifactStore

        # Reconstruct components in child process
        fs = FSGuard()

        cas_root = Path(task.sandbox_config["cas_root"])
        cas = ArtifactStore(cas_root)

        workspace_root = Path(task.sandbox_config["workspace_root"])
        node_id = task.sandbox_config["node_id"]

        sandbox_config = SandboxConfig(
            workspace_root=workspace_root,
            cas_root=cas_root,
            enable_gpu="gpu_id" in task.sandbox_config,
            cleanup_on_exit=task.sandbox_config.get("cleanup_on_exit", False),
        )

        # Create sandbox
        sandbox = Sandbox(
            node_id=node_id,
            config=sandbox_config,
            fs=fs,
            cas=cas,
        )

        # Instantiate and run node
        node = task.node_cls(**task.node_kwargs)

        with sandbox:
            outputs = node.run(sandbox=sandbox, **task.inputs)

        # Get manifest for provenance
        manifest = sandbox.get_manifest()

        result_queue.put(
            ProcessResult(
                outputs=outputs or {},
                manifest=manifest,
            )
        )

    except Exception as e:
        result_queue.put(
            ProcessResult(
                outputs={},
                error=str(e),
                traceback=traceback.format_exc(),
            )
        )


class ProcessExecutorError(RuntimeError):
    """Raised for process executor errors."""


class ProcessExecutor:
    """Spawn-safe execution engine for DAG nodes.

    Executes nodes in isolated child processes using multiprocessing.spawn.
    This ensures clean GPU state, memory isolation, and fault containment.

    Example:
        >>> executor = ProcessExecutor()
        >>>
        >>> result = executor.run(ProcessTask(
        ...     node_cls=ProcessImageNode,
        ...     node_kwargs={},
        ...     inputs={"image_sha": "abc123..."},
        ...     sandbox_config={
        ...         "node_id": "proc_001",
        ...         "workspace_root": Path("/tmp"),
        ...         "cas_root": Path("/cas"),
        ...     },
        ... ))
        >>>
        >>> if result.success:
        ...     print(f"Outputs: {result.outputs}")
    """

    def __init__(
        self,
        *,
        timeout: Optional[float] = None,
        start_method: str = "spawn",
    ) -> None:
        """Initialize process executor.

        Args:
            timeout: Default timeout in seconds for process execution
            start_method: Multiprocessing start method (spawn, fork, forkserver)
        """
        self.timeout = timeout
        self.ctx = mp.get_context(start_method)
        self._execution_count = 0

        logger.info(
            "ProcessExecutor initialized: start_method=%s, timeout=%s",
            start_method,
            timeout,
        )

    def run(
        self,
        task: ProcessTask,
        *,
        timeout: Optional[float] = None,
    ) -> ProcessResult:
        """Execute a task in an isolated child process.

        Args:
            task: ProcessTask with execution parameters
            timeout: Override default timeout

        Returns:
            ProcessResult with outputs or error
        """
        effective_timeout = timeout or self.timeout
        self._execution_count += 1

        # Create queue for result
        result_queue = self.ctx.Queue()

        # Create and start child process
        proc = self.ctx.Process(
            target=_worker,
            args=(task, result_queue),
            name=f"sandbox-{task.sandbox_config.get('node_id', 'unknown')}",
        )

        logger.debug(
            "Starting process for node: %s",
            task.sandbox_config.get("node_id"),
        )

        proc.start()

        try:
            # Wait for process with timeout
            proc.join(timeout=effective_timeout)

            if proc.is_alive():
                # Timeout - terminate process
                logger.warning(
                    "Process timeout, terminating: %s",
                    task.sandbox_config.get("node_id"),
                )
                proc.terminate()
                proc.join(timeout=5)

                if proc.is_alive():
                    proc.kill()
                    proc.join()

                return ProcessResult(
                    error=f"Process timeout after {effective_timeout}s",
                )

            # Get result from queue
            if not result_queue.empty():
                return result_queue.get_nowait()

            # Process exited but no result
            if proc.exitcode != 0:
                return ProcessResult(
                    error=f"Process exited with code {proc.exitcode}",
                )

            return ProcessResult(error="Process completed without result")

        finally:
            # Ensure cleanup
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)

    def run_batch(
        self,
        tasks: list[ProcessTask],
        *,
        timeout: Optional[float] = None,
    ) -> list[ProcessResult]:
        """Execute multiple tasks sequentially.

        Args:
            tasks: List of ProcessTasks
            timeout: Timeout per task

        Returns:
            List of ProcessResults
        """
        return [self.run(task, timeout=timeout) for task in tasks]

    @property
    def execution_count(self) -> int:
        """Total number of executions."""
        return self._execution_count
