"""In-process ``QueueBroker`` for tests, single-instance dev, and as the default.

State lives in process memory; a process restart loses any queued or
leased jobs. The Phase 1.C restart sweeper (``sweep_orphaned_jobs``)
will mark those orphaned-from-the-broker jobs as ``failed`` /
``worker_lost`` once Phase 2.D wires the runtime registry to read
broker state.

Lease deadlines use ``time.monotonic()`` so they survive system clock
adjustments. The broker accepts a ``now`` parameter on
``reclaim_expired_leases`` for test determinism; in production the
sweeper uses ``time.monotonic()``.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set

from transformation_portal.orchestrator.queue.base import (
    JobEnqueueRequest,
    JobLease,
    LeaseNotHeldError,
    LeaseStatus,
    QueueBroker,
    QueueBrokerError,
)


@dataclass
class _Lease:
    worker_id: str
    deadline: float
    request: JobEnqueueRequest
    cancellation_requested: bool = False


class MemoryQueueBroker(QueueBroker):
    """Single-process broker backed by an asyncio-safe FIFO and lease table."""

    def __init__(self) -> None:
        self._queue: Deque[JobEnqueueRequest] = deque()
        # job_id -> _Lease for jobs currently leased to a worker.
        self._leases: Dict[str, _Lease] = {}
        # All job_ids the broker is responsible for (queued + leased).
        # Lets ``enqueue`` reject collisions cheaply and ``queued_job_ids``
        # / ``leased_job_ids`` stay consistent without scanning.
        self._tracked: Set[str] = set()
        self._lock = asyncio.Lock()

    async def enqueue(self, request: JobEnqueueRequest) -> None:
        async with self._lock:
            if request.job_id in self._tracked:
                raise QueueBrokerError(f"job {request.job_id!r} already pending in the queue or leased")
            self._queue.append(request)
            self._tracked.add(request.job_id)

    async def acquire_lease(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> Optional[JobLease]:
        if lease_seconds <= 0:
            raise QueueBrokerError("lease_seconds must be positive")
        async with self._lock:
            if not self._queue:
                return None
            request = self._queue.popleft()
            deadline = time.monotonic() + lease_seconds
            self._leases[request.job_id] = _Lease(
                worker_id=worker_id,
                deadline=deadline,
                request=request,
            )
            return JobLease(
                job_id=request.job_id,
                worker_id=worker_id,
                deadline=deadline,
                request=request,
            )

    async def extend_lease(
        self,
        worker_id: str,
        job_id: str,
        *,
        lease_seconds: float,
    ) -> LeaseStatus:
        if lease_seconds <= 0:
            raise QueueBrokerError("lease_seconds must be positive")
        async with self._lock:
            lease = self._leases.get(job_id)
            if lease is None or lease.worker_id != worker_id:
                raise LeaseNotHeldError(worker_id=worker_id, job_id=job_id)
            if lease.cancellation_requested:
                return LeaseStatus.cancelled
            lease.deadline = time.monotonic() + lease_seconds
            return LeaseStatus.active

    async def release_lease(self, worker_id: str, job_id: str) -> None:
        async with self._lock:
            lease = self._leases.get(job_id)
            if lease is None or lease.worker_id != worker_id:
                # Idempotent: nothing to release.
                return
            self._leases.pop(job_id, None)
            self._tracked.discard(job_id)

    async def reclaim_expired_leases(self, *, now: float) -> List[str]:
        async with self._lock:
            expired = [jid for jid, lease in self._leases.items() if lease.deadline <= now]
            for jid in expired:
                lease = self._leases.pop(jid)
                # Re-queue the dispatch payload at the head so the
                # reclaimed job is the next one a worker picks up.
                self._queue.appendleft(lease.request)
                # job_id stays in self._tracked (it's still in flight,
                # just back in the queue).
            return expired

    async def cancel(self, job_id: str) -> bool:
        async with self._lock:
            if job_id not in self._tracked:
                return False
            lease = self._leases.get(job_id)
            if lease is not None:
                # In-flight: surface via ``LeaseStatus.cancelled`` on
                # the next ``extend_lease``; the worker handles the
                # graceful shutdown of its subprocess.
                lease.cancellation_requested = True
                return True
            # Still queued: drop the entry immediately so
            # ``queued_job_ids`` reflects reality and a fresh
            # ``enqueue`` of the same id is allowed without first
            # waiting for a worker to ``acquire_lease``.
            self._queue = deque(request for request in self._queue if request.job_id != job_id)
            self._tracked.discard(job_id)
            return True

    async def queued_job_ids(self) -> List[str]:
        async with self._lock:
            return [request.job_id for request in self._queue]

    async def leased_job_ids(self) -> List[str]:
        async with self._lock:
            return list(self._leases.keys())

    async def reset(self) -> None:
        async with self._lock:
            self._queue.clear()
            self._leases.clear()
            self._tracked.clear()


__all__ = ["MemoryQueueBroker"]
