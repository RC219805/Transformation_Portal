"""Queue Protocols for orchestrator job admission and worker pickup.

Phase 2 introduces ``QueueBroker`` so the orchestrator can decouple
admission (FastAPI handler returns immediately after enqueue) from
execution (a worker process picks up the job and runs it). Layer 2.A
ships the Protocol + an in-process ``MemoryQueueBroker``; later
layers add a Redis backend (2.B), wire ``app.py`` to enqueue (2.C),
and add ``worker_lost`` semantics + retry classification (2.D).

The broker stores only ``job_id`` plus a small dispatch payload (the
argv list and api_version). Authoritative job state lives in the
``JobRepository`` introduced in Phase 1; the worker fetches the
``JobRecord`` from the repository when it starts a leased job.

Lease/heartbeat contract:

- ``acquire_lease(worker_id, *, lease_seconds)`` removes the next
  ready job from the queue and grants the calling worker exclusive
  access for ``lease_seconds``. Returns ``None`` when the queue is
  empty; the caller should poll-with-backoff in that case.
- ``extend_lease(worker_id, job_id, *, lease_seconds)`` is the
  worker's heartbeat. The broker rejects the call if the worker no
  longer holds the lease (e.g. it was reclaimed after an expiry).
- ``release_lease(worker_id, job_id)`` is called by the worker once
  the job has reached a terminal state. Idempotent; ignores leases
  that don't exist.
- ``reclaim_expired_leases(*, now)`` is the broker's housekeeping:
  any lease whose deadline has passed is reclaimed and the job is
  re-queued for another worker. Returns the list of reclaimed job
  ids so callers (or tests) can log / metric it.
- ``cancel(job_id)`` removes a still-queued job, or marks an
  in-progress job for cancellation so the next ``extend_lease`` from
  the worker returns ``LeaseStatus.cancelled`` and the worker can
  surface the cancellation cleanly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QueueBrokerError(RuntimeError):
    """Base class for queue-broker failures (lease conflicts, IO, etc.)."""


class LeaseNotHeldError(QueueBrokerError):
    """Raised when ``extend_lease`` / ``release_lease`` finds no matching lease.

    Surfaces the case where a worker's lease was reclaimed (likely
    after a heartbeat-expiry sweep) and another worker may now hold
    the job. The losing worker must stop processing.
    """

    def __init__(self, worker_id: str, job_id: str) -> None:
        super().__init__(f"worker {worker_id!r} does not hold lease on job {job_id!r}")
        self.worker_id = worker_id
        self.job_id = job_id


class LeaseStatus(str, Enum):
    """Result codes for ``extend_lease``.

    ``active`` — lease was extended by the requested amount.
    ``cancelled`` — the orchestrator has called ``cancel(job_id)``
    while the worker was running. The worker should stop the
    underlying subprocess and call ``release_lease``.
    """

    active = "active"
    cancelled = "cancelled"


@dataclass
class JobEnqueueRequest:
    """Payload the orchestrator hands the broker on enqueue.

    The broker stores this verbatim and returns it on ``acquire_lease``
    so the worker has everything it needs to dispatch the job
    without fetching the repository for the dispatch argv. Job
    metadata (state, progress, logs, artifacts) still lives in the
    ``JobRepository``.
    """

    job_id: str
    argv: List[str]
    api_version: str = "v1"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobLease:
    """What the broker hands back to the worker that successfully acquires.

    ``deadline`` is the absolute monotonic-time-equivalent (epoch
    seconds) at which the lease expires unless the worker calls
    ``extend_lease``. The worker should heartbeat well before the
    deadline (typical: every ``lease_seconds // 3``).
    """

    job_id: str
    worker_id: str
    deadline: float
    request: JobEnqueueRequest


class QueueBroker(ABC):
    """Async queue-broker contract.

    Implementations:
    - ``MemoryQueueBroker`` — in-process; useful for tests, local
      single-instance dev, and as the default until Phase 2.B Redis.
    - ``RedisQueueBroker`` — Phase 2.B; SETNX-style lease + sorted-set
      queue; survives orchestrator/worker restarts.
    """

    @abstractmethod
    async def enqueue(self, request: JobEnqueueRequest) -> None:
        """Admit a job for execution. Idempotent on ``job_id``.

        A second ``enqueue`` for the same ``job_id`` while the first
        is still pending or running raises ``QueueBrokerError`` so
        admission collisions surface loudly rather than producing a
        duplicate execution.
        """

    @abstractmethod
    async def acquire_lease(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> Optional[JobLease]:
        """Pull the next ready job and grant ``worker_id`` exclusive access.

        Returns ``None`` when the queue is empty. Blocking semantics
        are intentionally NOT part of the contract; backends that
        support a long-poll mode expose it as a separate method.
        """

    @abstractmethod
    async def extend_lease(
        self,
        worker_id: str,
        job_id: str,
        *,
        lease_seconds: float,
    ) -> LeaseStatus:
        """Heartbeat: extend the lease deadline by ``lease_seconds``.

        Raises ``LeaseNotHeldError`` if the worker has been reclaimed
        (its lease expired and another worker may now hold the job).
        Returns ``LeaseStatus.cancelled`` if the orchestrator has
        called ``cancel(job_id)`` while the worker was running.
        """

    @abstractmethod
    async def release_lease(self, worker_id: str, job_id: str) -> None:
        """Worker is done with the job (success, failure, or cancelled).

        Idempotent: releasing a lease the worker doesn't hold is a
        no-op (e.g. when the worker calls release after a previous
        ``extend_lease`` already raised ``LeaseNotHeldError``).
        """

    @abstractmethod
    async def reclaim_expired_leases(self, *, now: float) -> List[str]:
        """Broker housekeeping: re-queue any job whose lease has expired.

        Returns the list of reclaimed job ids so callers can
        ``await repo.update(jid, ...)`` them as ``worker_lost`` in
        Phase 2.D. Memory backend uses real-time clock; Postgres /
        Redis backends accept ``now`` so tests can pin time.
        """

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """Mark a job as cancellation-requested.

        Returns ``True`` if the job was still in the broker (queued
        or leased), ``False`` if it had already been released. A
        leased job is not pulled away from its worker; instead, the
        next ``extend_lease`` returns ``LeaseStatus.cancelled`` and
        the worker handles graceful shutdown of the subprocess.
        """

    @abstractmethod
    async def queued_job_ids(self) -> List[str]:
        """Snapshot the FIFO queue (for tests and operator inspection)."""

    @abstractmethod
    async def leased_job_ids(self) -> List[str]:
        """Snapshot the in-flight job ids (for tests and operator inspection)."""

    @abstractmethod
    async def reset(self) -> None:
        """Test-only: clear all queue + lease state."""

    async def close(self) -> None:
        """Optional shutdown hook for backends that hold connections."""
        return None


__all__ = [
    "JobEnqueueRequest",
    "JobLease",
    "LeaseNotHeldError",
    "LeaseStatus",
    "QueueBroker",
    "QueueBrokerError",
]
