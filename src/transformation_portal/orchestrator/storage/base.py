"""Storage Protocols for orchestrator job state and event history.

Phase 1 introduces `JobRepository` and `JobEventStore` so the orchestrator can
swap an in-memory backend for a durable Postgres backend without changing wire
semantics. Runtime-only handles (`asyncio.subprocess.Process`, SSE subscriber
queues) live outside this surface — see `runtime_handles.py`.

The persistent surface mirrors the field set of the legacy `app.py:Job`
dataclass minus `proc` / `terminate_task`, which never survive a restart and
must not be persisted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple


class RepositoryError(RuntimeError):
    """Base class for repository-layer failures (concurrency, IO, etc.)."""


class JobNotFoundError(RepositoryError):
    """Raised when a job lookup misses for a job id the caller expected to find."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"job not found: {job_id}")
        self.job_id = job_id


@dataclass
class JobRecord:
    """Persistent projection of an orchestrator job.

    Fields are intentionally identical (names and types) to the persistent
    subset of the legacy ``app.py:Job`` dataclass so that the existing
    ``_serialize_job`` projection can keep producing the unchanged wire shape.

    Runtime handles (``proc``, ``terminate_task``) are excluded; they live in
    ``runtime_handles.RuntimeHandles`` and are never persisted.
    """

    id: str
    created_at: float
    state: str = "queued"  # queued|running|succeeded|partial|failed|canceled
    progress: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    done_published_at: Optional[float] = None
    last_event_at: Optional[float] = None
    exit_code: Optional[int] = None
    cancel_requested: bool = False
    request: Dict[str, Any] = field(default_factory=dict)
    effective_request: Dict[str, Any] = field(default_factory=dict)
    logs_tail: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    artifact_lookup: Dict[str, Path] = field(default_factory=dict)
    run_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    def copy(self) -> "JobRecord":
        """Return a fully-isolated copy.

        Container fields with mutable nested values (``request``,
        ``effective_request``, ``artifacts``, ``run_summary``, ``error``)
        are deep-copied so callers mutating nested structures
        (e.g. ``record.request["args"]["foo"] = ...``) cannot reach back
        into the repository's stored state.

        ``logs_tail`` is shallow-copied via ``list(...)``: its entries
        are ``str``, which is immutable.

        ``artifact_lookup`` is shallow-copied via ``dict(...)``: its
        values are ``pathlib.Path``, whose user-visible attributes are
        immutable.
        """
        return replace(
            self,
            request=deepcopy(self.request),
            effective_request=deepcopy(self.effective_request),
            logs_tail=list(self.logs_tail),
            artifacts=deepcopy(self.artifacts),
            artifact_lookup=dict(self.artifact_lookup),
            run_summary=deepcopy(self.run_summary),
            error=None if self.error is None else deepcopy(self.error),
        )


@dataclass
class JobEvent:
    """Single SSE-event entry for replay across restarts."""

    job_id: str
    seq: int
    event_type: str
    payload: Dict[str, Any]
    created_at: float


class JobRepository(ABC):
    """Async repository for the persistent slice of orchestrator job state.

    Implementations:
    - ``MemoryJobRepository`` — in-process dict; behavior-identical to the
      legacy ``JOBS`` dict.
    - ``PostgresJobRepository`` — SQLAlchemy-async + asyncpg backend
      introduced in Phase 1 Layer 1.B.

    All mutators are coroutines so a single call shape works for both
    backends. Memory implementations may simply return without awaiting any
    I/O.
    """

    @abstractmethod
    async def create(self, record: JobRecord) -> None:
        """Insert a new job. Raises ``RepositoryError`` if id collides."""

    @abstractmethod
    async def get(self, job_id: str) -> Optional[JobRecord]:
        """Return a copy of the record, or ``None`` if absent."""

    @abstractmethod
    async def list(
        self,
        *,
        limit: Optional[int] = None,
    ) -> Tuple[List[JobRecord], int]:
        """Return ``(records_sorted_by_created_at_desc, total_count)``."""

    @abstractmethod
    async def update(self, job_id: str, **fields: Any) -> JobRecord:
        """Patch the named fields on the job and return the refreshed record.

        Unknown field names raise ``RepositoryError``. Missing job raises
        ``JobNotFoundError``.
        """

    @abstractmethod
    async def append_log(self, job_id: str, line: str, *, tail_limit: int) -> None:
        """Append a log line and trim ``logs_tail`` to ``tail_limit``."""

    @abstractmethod
    async def set_artifacts(
        self,
        job_id: str,
        artifacts: Dict[str, Any],
        artifact_lookup: Dict[str, Path],
    ) -> None:
        """Replace the artifact index and lookup map atomically.

        ``artifact_lookup`` maps relative artifact paths to absolute
        ``pathlib.Path`` instances, identical to the legacy
        ``app.py:Job.artifact_lookup`` semantic. Backends that need
        string serialization (e.g. Postgres) handle the conversion at
        the persistence boundary.
        """

    @abstractmethod
    async def delete(self, job_id: str) -> None:
        """Delete a job; missing ids are a no-op."""

    @abstractmethod
    async def cleanup_expired(self, now: float, retention_seconds: float) -> List[str]:
        """Delete finished jobs older than retention.

        Returns the list of removed ids so callers can clean adjacent
        runtime registries (SSE subscribers, event subscribers).
        """

    @abstractmethod
    async def count_active(self) -> int:
        """Return the number of jobs whose state is ``queued`` or ``running``."""

    @abstractmethod
    async def sweep_orphaned(
        self,
        *,
        live_job_ids: Optional[List[str]] = None,
        reason_code: str = "worker_lost_on_restart",
        now: Optional[float] = None,
    ) -> List[str]:
        """Mark any ``queued``/``running`` job not in ``live_job_ids`` as failed.

        Used by the Phase 1.C restart sweeper. Sets ``state=failed``,
        ``finished_at=now``, ``done_published_at=now``, and an ``error`` payload
        carrying ``reason_code``. Returns the list of swept ids.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Test-only: clear all state. Production callers must not invoke."""

    async def close(self) -> None:
        """Optional shutdown hook for backends that hold connections."""
        return None


class JobEventStore(ABC):
    """Async event-history surface for SSE replay across restarts."""

    @abstractmethod
    async def append(
        self,
        job_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        created_at: float,
    ) -> JobEvent:
        """Persist one event and return the assigned ``seq``."""

    @abstractmethod
    async def events_since(
        self,
        job_id: str,
        *,
        after_seq: Optional[int] = None,
    ) -> AsyncIterator[JobEvent]:
        """Yield events for ``job_id`` ordered by ``seq``.

        ``after_seq`` is exclusive. ``None`` yields from the beginning.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Test-only: clear all stored events."""

    async def close(self) -> None:
        """Optional shutdown hook for backends that hold connections."""
        return None
