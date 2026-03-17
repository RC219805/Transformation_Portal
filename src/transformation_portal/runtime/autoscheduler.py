"""Cluster Autoscheduler with resource and data locality awareness.

This module provides intelligent task scheduling that considers:
- GPU availability
- Worker load balancing
- Data locality (CAS presence)
- Automatic scaling hooks

Example:
    >>> scheduler = ClusterAutoscheduler()
    >>>
    >>> # Register workers
    >>> scheduler.register_worker("worker1", WorkerState(
    ...     host="192.168.1.10",
    ...     port=5000,
    ...     gpus=2,
    ...     cas_root="/data/cas",
    ... ))
    >>>
    >>> # Dispatch task to best worker
    >>> outputs = scheduler.dispatch(
    ...     task,
    ...     requires_gpu=True,
    ...     input_shas=["abc123...", "def456..."],
    ... )
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status of a worker node."""

    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    DRAINING = "draining"


@dataclass
class WorkerState:
    """State of a worker node.

    Attributes:
        host: Worker hostname or IP
        port: Worker port
        gpus: Number of GPU devices
        memory_gb: Available memory in GB
        cas_root: Path to CAS storage on worker
        load: Current number of active tasks
        status: Worker status
        last_heartbeat: Timestamp of last heartbeat
        capabilities: Set of supported capabilities
        cas_index: Set of CAS SHA256 hashes available locally
    """

    host: str
    port: int
    gpus: int = 0
    memory_gb: float = 0
    cas_root: Optional[str] = None
    load: int = 0
    status: WorkerStatus = WorkerStatus.ONLINE
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: Set[str] = field(default_factory=set)
    cas_index: Set[str] = field(default_factory=set)

    @property
    def address(self) -> str:
        """Get worker address string."""
        return f"{self.host}:{self.port}"

    def is_available(self) -> bool:
        """Check if worker is available for tasks."""
        return self.status == WorkerStatus.ONLINE


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision.

    Attributes:
        worker_id: Selected worker ID
        worker: Selected worker state
        score: Scheduling score (lower is better)
        reason: Human-readable reason for selection
    """

    worker_id: str
    worker: WorkerState
    score: float
    reason: str


class SchedulerError(RuntimeError):
    """Raised for scheduler errors."""


class ClusterAutoscheduler:
    """Resource and data-locality aware cluster scheduler.

    Schedules tasks to workers based on:
    - GPU availability (required vs available)
    - Current worker load
    - Data locality (input SHAs present in worker CAS)
    - Worker capabilities

    Example:
        >>> scheduler = ClusterAutoscheduler()
        >>>
        >>> # Register workers
        >>> scheduler.register_worker("gpu1", WorkerState(
        ...     host="gpu-server-1",
        ...     port=5000,
        ...     gpus=4,
        ...     cas_root="/data/cas",
        ... ))
        >>>
        >>> # Dispatch with locality hints
        >>> outputs = scheduler.dispatch(
        ...     task,
        ...     requires_gpu=True,
        ...     input_shas=["abc...", "def..."],
        ... )
    """

    def __init__(
        self,
        *,
        heartbeat_timeout: float = 60.0,
        load_penalty: float = 5.0,
        locality_bonus: float = 2.0,
        gpu_bonus: float = 10.0,
    ) -> None:
        """Initialize autoscheduler.

        Args:
            heartbeat_timeout: Seconds before worker considered offline
            load_penalty: Score penalty per active task
            locality_bonus: Score bonus for local data
            gpu_bonus: Score bonus for GPU availability
        """
        self._workers: Dict[str, WorkerState] = {}
        self._lock = threading.RLock()
        self._heartbeat_timeout = heartbeat_timeout
        self._load_penalty = load_penalty
        self._locality_bonus = locality_bonus
        self._gpu_bonus = gpu_bonus

        # Scaling hooks
        self._scale_up_callback: Optional[Callable[[], None]] = None
        self._scale_down_callback: Optional[Callable[[], None]] = None

        # Statistics
        self._dispatch_count = 0
        self._locality_hits = 0

        logger.info("ClusterAutoscheduler initialized")

    def register_worker(
        self,
        worker_id: str,
        state: WorkerState,
    ) -> None:
        """Register a worker node.

        Args:
            worker_id: Unique worker identifier
            state: Worker state
        """
        with self._lock:
            self._workers[worker_id] = state
            logger.info(
                "Registered worker: %s (%s, gpus=%d)",
                worker_id,
                state.address,
                state.gpus,
            )

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker node.

        Args:
            worker_id: Worker identifier to remove
        """
        with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info("Unregistered worker: %s", worker_id)

    def heartbeat(self, worker_id: str) -> None:
        """Update worker heartbeat timestamp.

        Args:
            worker_id: Worker identifier
        """
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = time.time()
                self._workers[worker_id].status = WorkerStatus.ONLINE

    def update_cas_index(
        self,
        worker_id: str,
        sha_hashes: Set[str],
    ) -> None:
        """Update worker's CAS index.

        Args:
            worker_id: Worker identifier
            sha_hashes: Set of SHA256 hashes available locally
        """
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].cas_index = sha_hashes
                logger.debug(
                    "Updated CAS index for %s: %d hashes",
                    worker_id,
                    len(sha_hashes),
                )

    def _check_heartbeats(self) -> None:
        """Mark workers as offline if heartbeat expired."""
        now = time.time()
        with self._lock:
            for worker_id, state in self._workers.items():
                if now - state.last_heartbeat > self._heartbeat_timeout:
                    if state.status == WorkerStatus.ONLINE:
                        state.status = WorkerStatus.OFFLINE
                        logger.warning(
                            "Worker %s marked offline (no heartbeat)",
                            worker_id,
                        )

    def _score_worker(
        self,
        worker: WorkerState,
        *,
        requires_gpu: bool,
        required_memory_gb: float,
        input_shas: Optional[List[str]],
        required_capabilities: Optional[Set[str]],
    ) -> Tuple[float, str]:
        """Score a worker for task assignment.

        Lower score is better.

        Args:
            worker: Worker to score
            requires_gpu: Whether task needs GPU
            required_memory_gb: Required memory
            input_shas: Input CAS hashes for locality
            required_capabilities: Required capabilities

        Returns:
            Tuple of (score, reason)
        """
        # Check basic availability
        if not worker.is_available():
            return float("inf"), "offline"

        score = 0.0
        reasons = []

        # GPU constraint
        if requires_gpu:
            if worker.gpus <= 0:
                return float("inf"), "no GPU"
            score -= self._gpu_bonus
            reasons.append(f"gpu={worker.gpus}")

        # Memory constraint
        if required_memory_gb > 0:
            if worker.memory_gb < required_memory_gb:
                return float("inf"), "insufficient memory"

        # Capabilities check
        if required_capabilities:
            missing = required_capabilities - worker.capabilities
            if missing:
                return float("inf"), f"missing capabilities: {missing}"

        # Load penalty
        score += worker.load * self._load_penalty
        if worker.load > 0:
            reasons.append(f"load={worker.load}")

        # Data locality bonus
        if input_shas and worker.cas_index:
            local_count = sum(1 for sha in input_shas if sha in worker.cas_index)
            if local_count > 0:
                locality_score = (local_count / len(input_shas)) * self._locality_bonus
                score -= locality_score * 10
                reasons.append(f"local_data={local_count}/{len(input_shas)}")

        reason = ", ".join(reasons) if reasons else "default"
        return score, reason

    def select_worker(
        self,
        *,
        requires_gpu: bool = False,
        required_memory_gb: float = 0,
        input_shas: Optional[List[str]] = None,
        required_capabilities: Optional[Set[str]] = None,
    ) -> SchedulingDecision:
        """Select the best worker for a task.

        Args:
            requires_gpu: Whether task needs GPU
            required_memory_gb: Required memory in GB
            input_shas: Input CAS hashes for locality hints
            required_capabilities: Required capabilities

        Returns:
            SchedulingDecision with selected worker

        Raises:
            SchedulerError: If no suitable worker available
        """
        self._check_heartbeats()

        with self._lock:
            candidates = []

            for worker_id, worker in self._workers.items():
                score, reason = self._score_worker(
                    worker,
                    requires_gpu=requires_gpu,
                    required_memory_gb=required_memory_gb,
                    input_shas=input_shas,
                    required_capabilities=required_capabilities,
                )

                if score != float("inf"):
                    candidates.append((worker_id, worker, score, reason))

            if not candidates:
                # Trigger scale-up if configured
                if self._scale_up_callback:
                    logger.info("No workers available, triggering scale-up")
                    self._scale_up_callback()

                raise SchedulerError("No suitable workers available")

            # Sort by score (lower is better)
            candidates.sort(key=lambda x: x[2])
            worker_id, worker, score, reason = candidates[0]

            # Track locality hits
            if input_shas and worker.cas_index:
                local_count = sum(1 for sha in input_shas if sha in worker.cas_index)
                if local_count > 0:
                    self._locality_hits += 1

            return SchedulingDecision(
                worker_id=worker_id,
                worker=worker,
                score=score,
                reason=reason,
            )

    def dispatch(
        self,
        task: "ProcessTask",
        *,
        requires_gpu: bool = False,
        required_memory_gb: float = 0,
        input_shas: Optional[List[str]] = None,
        required_capabilities: Optional[Set[str]] = None,
        send_fn: Optional[Callable[[str, int, Dict], Dict]] = None,
    ) -> Dict[str, Any]:
        """Dispatch a task to the best available worker.

        Args:
            task: Task to dispatch
            requires_gpu: Whether task needs GPU
            required_memory_gb: Required memory
            input_shas: Input CAS hashes for locality
            required_capabilities: Required capabilities
            send_fn: Function to send task (default: distributed_client.send_task)

        Returns:
            Task outputs

        Raises:
            SchedulerError: If dispatch fails
        """
        # Select worker
        decision = self.select_worker(
            requires_gpu=requires_gpu,
            required_memory_gb=required_memory_gb,
            input_shas=input_shas,
            required_capabilities=required_capabilities,
        )

        worker = decision.worker
        self._dispatch_count += 1

        # Update load
        with self._lock:
            worker.load += 1

        logger.info(
            "Dispatching to %s (score=%.2f, reason=%s)",
            decision.worker_id,
            decision.score,
            decision.reason,
        )

        try:
            # Send task
            if send_fn is None:
                from transformation_portal.runtime.distributed_client import send_task

                send_fn = send_task

            result = send_fn(worker.host, worker.port, task.__dict__)

            if result.get("error"):
                raise SchedulerError(f"Task failed: {result['error']}")

            return result.get("outputs", {})

        finally:
            # Update load
            with self._lock:
                worker.load = max(0, worker.load - 1)

    def set_scale_callbacks(
        self,
        *,
        scale_up: Optional[Callable[[], None]] = None,
        scale_down: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set autoscaling callbacks.

        Args:
            scale_up: Called when more workers needed
            scale_down: Called when workers can be removed
        """
        self._scale_up_callback = scale_up
        self._scale_down_callback = scale_down

    def scale_check(self) -> None:
        """Check if scaling is needed.

        Calls scale_up if all workers busy, scale_down if many idle.
        """
        with self._lock:
            available = [w for w in self._workers.values() if w.is_available()]

            if not available and self._scale_up_callback:
                logger.info("All workers busy, triggering scale-up")
                self._scale_up_callback()
                return

            idle = sum(1 for w in available if w.load == 0)
            total = len(available)

            if total > 1 and idle > total // 2 and self._scale_down_callback:
                logger.info("Many workers idle, triggering scale-down")
                self._scale_down_callback()

    @property
    def workers(self) -> Dict[str, WorkerState]:
        """Get all workers (copy)."""
        with self._lock:
            return dict(self._workers)

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler metrics
        """
        with self._lock:
            online = sum(1 for w in self._workers.values() if w.status == WorkerStatus.ONLINE)
            total_gpus = sum(w.gpus for w in self._workers.values())
            total_load = sum(w.load for w in self._workers.values())

            return {
                "workers_total": len(self._workers),
                "workers_online": online,
                "total_gpus": total_gpus,
                "total_load": total_load,
                "dispatch_count": self._dispatch_count,
                "locality_hits": self._locality_hits,
            }


# Import for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformation_portal.runtime.process_executor import ProcessTask
