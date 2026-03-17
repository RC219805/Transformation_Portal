"""Cluster Scheduler with priority queues and retries.

This module provides a scheduler for executing DAG nodes with:
- Priority-based task ordering
- Automatic retries on failure
- GPU-aware dispatch
- Resource tracking
- Local or remote execution support

Example:
    >>> scheduler = ClusterScheduler(engine)
    >>>
    >>> # Submit tasks with priority
    >>> scheduler.submit(ProcessNode, inputs={"sha": "abc..."}, priority=1)
    >>> scheduler.submit(AnalyzeNode, inputs={"sha": "def..."}, priority=10)
    >>>
    >>> # Run all tasks (high priority first)
    >>> results = scheduler.run_all()
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a scheduled task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass(order=True)
class ScheduledTask:
    """Task in the scheduler queue.

    Lower priority values are executed first.

    Attributes:
        priority: Task priority (lower = higher priority)
        created_at: Timestamp when task was created
        task_id: Unique task identifier
        node_cls: DAG node class to execute
        inputs: Inputs for node execution
        node_id: Node identifier for sandbox
        node_kwargs: Additional kwargs for node constructor
        retries: Current retry count
        max_retries: Maximum retry attempts
        use_gpu: Whether task requires GPU
        timeout: Execution timeout in seconds
    """

    priority: int
    created_at: float = field(compare=False)
    task_id: str = field(compare=False)
    node_cls: Type = field(compare=False)
    inputs: Dict[str, Any] = field(compare=False)
    node_id: str = field(compare=False)
    node_kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=2, compare=False)
    use_gpu: bool = field(default=False, compare=False)
    timeout: Optional[float] = field(default=None, compare=False)


@dataclass
class TaskResult:
    """Result of a scheduled task execution.

    Attributes:
        task_id: Task identifier
        merkle_hash: Merkle hash of the execution (if successful)
        outputs: Execution outputs
        status: Final task status
        retries_used: Number of retries attempted
        duration_seconds: Total execution time
        error: Error message if failed
    """

    task_id: str
    merkle_hash: Optional[str]
    outputs: Dict[str, Any]
    status: TaskStatus
    retries_used: int
    duration_seconds: float
    error: Optional[str] = None


class SchedulerError(RuntimeError):
    """Raised for scheduler errors."""


class ClusterScheduler:
    """Priority-based cluster scheduler with retries.

    Manages task execution across the execution engine with:
    - Priority queues (lower value = higher priority)
    - Automatic retries on transient failures
    - GPU-aware task dispatch
    - Execution history tracking

    Example:
        >>> scheduler = ClusterScheduler(engine)
        >>>
        >>> # Submit high-priority task
        >>> scheduler.submit(
        ...     CriticalNode,
        ...     inputs={"data": "abc..."},
        ...     node_id="critical_001",
        ...     priority=1,
        ...     use_gpu=True,
        ... )
        >>>
        >>> # Submit normal priority task
        >>> scheduler.submit(
        ...     NormalNode,
        ...     inputs={"data": "def..."},
        ...     node_id="normal_001",
        ...     priority=10,
        ... )
        >>>
        >>> # Execute all (high priority first)
        >>> results = scheduler.run_all()
    """

    def __init__(
        self,
        engine: "ExecutionEngine",
        *,
        max_concurrent: int = 1,
        default_priority: int = 10,
        default_max_retries: int = 2,
    ) -> None:
        """Initialize scheduler.

        Args:
            engine: Execution engine for running tasks
            max_concurrent: Maximum concurrent tasks (1 = sequential)
            default_priority: Default task priority
            default_max_retries: Default max retries per task
        """
        self.engine = engine
        self.max_concurrent = max_concurrent
        self.default_priority = default_priority
        self.default_max_retries = default_max_retries

        self._queue: List[ScheduledTask] = []
        self._lock = threading.Lock()
        self._task_counter = 0
        self._results: List[TaskResult] = []
        self._history: List[str] = []  # Merkle hashes

        logger.info(
            "ClusterScheduler initialized: max_concurrent=%d",
            max_concurrent,
        )

    def submit(
        self,
        node_cls: Type,
        *,
        inputs: Dict[str, Any],
        node_id: str,
        priority: Optional[int] = None,
        node_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        use_gpu: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        """Submit a task to the scheduler.

        Args:
            node_cls: DAG node class to execute
            inputs: Inputs for node execution
            node_id: Node identifier for sandbox
            priority: Task priority (lower = higher priority)
            node_kwargs: Additional kwargs for node constructor
            max_retries: Maximum retry attempts
            use_gpu: Whether task requires GPU
            timeout: Execution timeout

        Returns:
            Task ID
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter:06d}"

            task = ScheduledTask(
                priority=priority or self.default_priority,
                created_at=time.time(),
                task_id=task_id,
                node_cls=node_cls,
                inputs=inputs,
                node_id=node_id,
                node_kwargs=node_kwargs or {},
                max_retries=max_retries or self.default_max_retries,
                use_gpu=use_gpu,
                timeout=timeout,
            )

            heapq.heappush(self._queue, task)

            logger.debug(
                "Submitted task %s: node=%s, priority=%d",
                task_id,
                node_cls.__name__,
                task.priority,
            )

            return task_id

    def run_next(self) -> Optional[TaskResult]:
        """Execute the next task in the queue.

        Returns:
            TaskResult if a task was executed, None if queue empty
        """
        with self._lock:
            if not self._queue:
                return None
            task = heapq.heappop(self._queue)

        start_time = time.time()

        try:
            logger.info(
                "Executing task %s: node=%s, attempt=%d/%d",
                task.task_id,
                task.node_cls.__name__,
                task.retries + 1,
                task.max_retries + 1,
            )

            merkle_hash, outputs = self.engine.run_node(
                task.node_cls,
                inputs=task.inputs,
                node_id=task.node_id,
                node_kwargs=task.node_kwargs,
                use_gpu=task.use_gpu,
                timeout=task.timeout,
            )

            duration = time.time() - start_time

            result = TaskResult(
                task_id=task.task_id,
                merkle_hash=merkle_hash,
                outputs=outputs,
                status=TaskStatus.COMPLETED,
                retries_used=task.retries,
                duration_seconds=duration,
            )

            with self._lock:
                self._results.append(result)
                if merkle_hash:
                    self._history.append(merkle_hash)

            logger.info(
                "Task %s completed: merkle=%s, duration=%.2fs",
                task.task_id,
                merkle_hash[:8] if merkle_hash else "N/A",
                duration,
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            if task.retries < task.max_retries:
                # Retry
                task.retries += 1
                with self._lock:
                    heapq.heappush(self._queue, task)

                logger.warning(
                    "Task %s failed, scheduling retry %d/%d: %s",
                    task.task_id,
                    task.retries,
                    task.max_retries,
                    e,
                )

                return TaskResult(
                    task_id=task.task_id,
                    merkle_hash=None,
                    outputs={},
                    status=TaskStatus.RETRYING,
                    retries_used=task.retries,
                    duration_seconds=duration,
                    error=str(e),
                )

            else:
                # Final failure
                result = TaskResult(
                    task_id=task.task_id,
                    merkle_hash=None,
                    outputs={},
                    status=TaskStatus.FAILED,
                    retries_used=task.retries,
                    duration_seconds=duration,
                    error=str(e),
                )

                with self._lock:
                    self._results.append(result)

                logger.error(
                    "Task %s failed after %d retries: %s",
                    task.task_id,
                    task.retries,
                    e,
                )

                return result

    def run_all(
        self,
        *,
        stop_on_failure: bool = False,
    ) -> List[TaskResult]:
        """Execute all tasks in the queue.

        Args:
            stop_on_failure: If True, stop on first permanent failure

        Returns:
            List of TaskResults
        """
        results = []

        while True:
            with self._lock:
                if not self._queue:
                    break

            result = self.run_next()
            if result:
                results.append(result)

                if stop_on_failure and result.status == TaskStatus.FAILED:
                    logger.warning("Stopping scheduler due to task failure")
                    break

        return results

    @property
    def queue_size(self) -> int:
        """Number of pending tasks."""
        with self._lock:
            return len(self._queue)

    @property
    def history(self) -> List[str]:
        """List of completed task Merkle hashes."""
        with self._lock:
            return list(self._history)

    @property
    def results(self) -> List[TaskResult]:
        """All task results."""
        with self._lock:
            return list(self._results)

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler metrics
        """
        with self._lock:
            completed = sum(1 for r in self._results if r.status == TaskStatus.COMPLETED)
            failed = sum(1 for r in self._results if r.status == TaskStatus.FAILED)
            total_duration = sum(r.duration_seconds for r in self._results)

            return {
                "tasks_submitted": self._task_counter,
                "tasks_pending": len(self._queue),
                "tasks_completed": completed,
                "tasks_failed": failed,
                "total_duration_seconds": total_duration,
                "merkle_hashes": len(self._history),
            }


# Import for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformation_portal.runtime.engine import ExecutionEngine
