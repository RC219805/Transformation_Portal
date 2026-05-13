"""In-memory implementations of ``JobRepository`` and ``JobEventStore``.

Behavior-identical to the legacy ``app.py:JOBS`` dict. State lives in
process memory; restart clears it. Per-job ``asyncio.Lock`` guards write
ordering so concurrent ``update`` calls cannot interleave on the same id.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

from transformation_portal.orchestrator.storage.base import UPDATABLE_FIELDS as _UPDATABLE_FIELDS
from transformation_portal.orchestrator.storage.base import (
    JobEvent,
    JobEventStore,
    JobNotFoundError,
    JobRecord,
    JobRepository,
    RepositoryError,
)

_ACTIVE_STATES = {"queued", "running"}


class MemoryJobRepository(JobRepository):
    """Thread-safe (within one asyncio loop) in-process job repository."""

    def __init__(self) -> None:
        self._records: Dict[str, JobRecord] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, job_id: str) -> asyncio.Lock:
        lock = self._locks.get(job_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[job_id] = lock
        return lock

    async def create(self, record: JobRecord) -> None:
        if record.id in self._records:
            raise RepositoryError(f"job id already exists: {record.id}")
        self._records[record.id] = record.copy()

    async def get(self, job_id: str) -> Optional[JobRecord]:
        existing = self._records.get(job_id)
        return None if existing is None else existing.copy()

    async def list(self, *, limit: Optional[int] = None) -> Tuple[List[JobRecord], int]:
        ordered = sorted(
            self._records.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )
        total = len(ordered)
        if limit is not None:
            ordered = ordered[:limit]
        return [r.copy() for r in ordered], total

    async def update(self, job_id: str, **fields: Any) -> JobRecord:
        async with self._lock_for(job_id):
            existing = self._records.get(job_id)
            if existing is None:
                raise JobNotFoundError(job_id)
            # Cross-backend contract: id/created_at are immutable;
            # artifact_lookup is owned by ``set_artifacts`` (the Postgres
            # backend persists it through a separate table). Forbidding
            # it here keeps both backends behavior-identical at the
            # ``update`` boundary.
            unknown = set(fields) - _UPDATABLE_FIELDS
            if unknown:
                raise RepositoryError(f"update received unknown fields: {sorted(unknown)}")
            for key, value in fields.items():
                setattr(existing, key, value)
            return existing.copy()

    async def append_log(self, job_id: str, line: str, *, tail_limit: int) -> None:
        if tail_limit <= 0:
            raise RepositoryError("tail_limit must be positive")
        async with self._lock_for(job_id):
            existing = self._records.get(job_id)
            if existing is None:
                raise JobNotFoundError(job_id)
            existing.logs_tail.append(line)
            if len(existing.logs_tail) > tail_limit:
                existing.logs_tail = existing.logs_tail[-tail_limit:]

    async def set_artifacts(
        self,
        job_id: str,
        artifacts: Dict[str, Any],
        artifact_lookup: Dict[str, Path],
    ) -> None:
        async with self._lock_for(job_id):
            existing = self._records.get(job_id)
            if existing is None:
                raise JobNotFoundError(job_id)
            existing.artifacts = dict(artifacts)
            existing.artifact_lookup = dict(artifact_lookup)

    async def delete(self, job_id: str) -> None:
        self._records.pop(job_id, None)
        self._locks.pop(job_id, None)

    async def cleanup_expired(self, now: float, retention_seconds: float) -> List[str]:
        expired = [
            jid
            for jid, rec in self._records.items()
            if rec.finished_at is not None and now - rec.finished_at >= retention_seconds
        ]
        for jid in expired:
            self._records.pop(jid, None)
            self._locks.pop(jid, None)
        return expired

    async def count_active(self) -> int:
        return sum(1 for rec in self._records.values() if rec.state in _ACTIVE_STATES)

    async def sweep_orphaned(
        self,
        *,
        live_job_ids: Optional[List[str]] = None,
        reason_code: str = "worker_lost_on_restart",
        now: Optional[float] = None,
    ) -> List[str]:
        import time

        stamp = time.time() if now is None else now
        live = set(live_job_ids or ())
        swept: List[str] = []
        for jid, rec in self._records.items():
            if rec.state not in _ACTIVE_STATES:
                continue
            if jid in live:
                continue
            rec.state = "worker_lost"
            rec.finished_at = stamp
            rec.done_published_at = stamp
            rec.last_event_at = stamp
            rec.error = {
                "code": reason_code,
                "message": "Process did not survive backend restart.",
                "retriable": True,
            }
            swept.append(jid)
        return swept

    async def reset(self) -> None:
        self._records.clear()
        self._locks.clear()


class MemoryJobEventStore(JobEventStore):
    """In-process event history, bounded per job to keep memory predictable."""

    DEFAULT_PER_JOB_CAP = 4096

    def __init__(self, *, per_job_cap: int = DEFAULT_PER_JOB_CAP) -> None:
        if per_job_cap <= 0:
            raise RepositoryError(
                f"per_job_cap must be positive; got {per_job_cap!r}. A non-positive "
                "cap silently drops events while incrementing seq counters, which "
                "would break SSE replay."
            )
        self._per_job_cap = per_job_cap
        self._events: Dict[str, Deque[JobEvent]] = defaultdict(lambda: deque(maxlen=per_job_cap))
        self._next_seq: Dict[str, int] = defaultdict(lambda: 1)

    async def append(
        self,
        job_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        created_at: float,
    ) -> JobEvent:
        seq = self._next_seq[job_id]
        self._next_seq[job_id] = seq + 1
        event = JobEvent(
            job_id=job_id,
            seq=seq,
            event_type=event_type,
            payload=dict(payload),
            created_at=created_at,
        )
        self._events[job_id].append(event)
        return event

    async def events_since(
        self,
        job_id: str,
        *,
        after_seq: Optional[int] = None,
    ) -> AsyncIterator[JobEvent]:
        threshold = -1 if after_seq is None else after_seq
        for event in list(self._events.get(job_id, ())):
            if event.seq > threshold:
                yield event

    async def reset(self) -> None:
        self._events.clear()
        self._next_seq.clear()
