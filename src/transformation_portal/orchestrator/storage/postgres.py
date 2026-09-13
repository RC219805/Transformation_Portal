"""Postgres-backed ``JobRepository`` and ``JobEventStore``.

Phase 1.B - durable orchestrator state. The repository projects the
persistent slice of orchestrator jobs into the ORM models defined in
``transformation_portal.orchestrator.models`` and back to the
``JobRecord`` dataclass that the rest of the orchestrator already knows
about.

Concurrency: per-job ``update`` and log append calls serialize on the
row lock, while remaining JSON index update paths use optimistic
concurrency and retry on conflict.

Memory ``logs_tail`` parity: this layer keeps the legacy "bounded
in-memory tail" semantic - ``append_log`` reads the current tail,
appends, trims, and writes back atomically inside a transaction.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple, cast

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from transformation_portal.orchestrator.models import (
    Base,
    DispatchAttemptModel,
    JobArtifactModel,
    JobEventModel,
    JobEventSequenceModel,
    JobModel,
    OperationalAuditEventModel,
)
from transformation_portal.orchestrator.storage.base import UPDATABLE_FIELDS as _UPDATABLE_FIELDS
from transformation_portal.orchestrator.storage.base import (
    JobEvent,
    JobEventStore,
    JobNotFoundError,
    JobRecord,
    JobRepository,
    OperationalAuditRecord,
    OperationalAuditStore,
    RepositoryError,
)

logger = logging.getLogger(__name__)

_ACTIVE_STATES = {"queued", "running"}
_OPTIMISTIC_LOCK_RETRIES = 3
DEFAULT_EVENT_RETENTION_PER_JOB = 4096


class _SharedEngine:
    """Process-wide cache of one ``AsyncEngine`` per ``database_url``.

    Both ``PostgresJobRepository`` and ``PostgresJobEventStore`` share
    the engine so the orchestrator opens a single connection pool.
    """

    _engines: Dict[str, AsyncEngine] = {}
    _sessions: Dict[str, async_sessionmaker[AsyncSession]] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get(cls, database_url: str) -> Tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
        async with cls._lock:
            engine = cls._engines.get(database_url)
            if engine is None:
                engine = create_async_engine(
                    database_url,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    connect_args={
                        "timeout": 5.0,
                        "command_timeout": 5.0,
                        "server_settings": {"statement_timeout": "5000", "lock_timeout": "5000"},
                    },
                    future=True,
                )
                cls._engines[database_url] = engine
                cls._sessions[database_url] = async_sessionmaker(
                    engine,
                    expire_on_commit=False,
                    class_=AsyncSession,
                )
            return engine, cls._sessions[database_url]

    @classmethod
    async def dispose(cls, database_url: str) -> None:
        async with cls._lock:
            engine = cls._engines.pop(database_url, None)
            cls._sessions.pop(database_url, None)
        if engine is not None:
            await engine.dispose()


def _record_from_model(model: JobModel) -> JobRecord:
    """Project an ORM ``JobModel`` to the public ``JobRecord``."""
    artifact_lookup = {artifact.path: Path(artifact.absolute_path) for artifact in model.artifact_index}
    return JobRecord(
        id=model.id,
        created_at=model.created_at,
        state=model.state,
        progress=model.progress,
        started_at=model.started_at,
        finished_at=model.finished_at,
        done_published_at=model.done_published_at,
        last_event_at=model.last_event_at,
        exit_code=model.exit_code,
        cancel_requested=model.cancel_requested,
        request=deepcopy(model.request or {}),
        effective_request=deepcopy(model.effective_request or {}),
        logs_tail=list(model.logs_tail or []),
        artifacts=deepcopy(model.artifacts or {}),
        artifact_lookup=artifact_lookup,
        run_summary=deepcopy(model.run_summary or {}),
        error=None if model.error is None else deepcopy(model.error),
    )


class PostgresJobRepository(JobRepository):
    """Durable ``JobRepository`` backed by SQLAlchemy 2.x async + asyncpg."""

    def __init__(self, *, database_url: str) -> None:
        if not database_url:
            raise RepositoryError("database_url must be a non-empty string")
        self._database_url = database_url
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def _ensure_engine(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._engine, self._session_factory = await _SharedEngine.get(self._database_url)
        return self._session_factory

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        factory = await self._ensure_engine()
        async with factory() as session:
            yield session

    async def create(self, record: JobRecord) -> None:
        record_snapshot = record.copy()
        async with self._session() as session:
            session.add(
                JobModel(
                    id=record_snapshot.id,
                    created_at=record_snapshot.created_at,
                    state=record_snapshot.state,
                    progress=record_snapshot.progress,
                    started_at=record_snapshot.started_at,
                    finished_at=record_snapshot.finished_at,
                    done_published_at=record_snapshot.done_published_at,
                    last_event_at=record_snapshot.last_event_at,
                    exit_code=record_snapshot.exit_code,
                    cancel_requested=record_snapshot.cancel_requested,
                    request=record_snapshot.request,
                    effective_request=record_snapshot.effective_request,
                    logs_tail=record_snapshot.logs_tail,
                    artifacts=record_snapshot.artifacts,
                    run_summary=record_snapshot.run_summary,
                    error=record_snapshot.error,
                    version=1,
                )
            )
            for path_str, absolute in record_snapshot.artifact_lookup.items():
                session.add(
                    JobArtifactModel(
                        job_id=record_snapshot.id,
                        path=path_str,
                        absolute_path=str(absolute),
                    )
                )
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise RepositoryError(f"job id already exists: {record_snapshot.id}") from exc

    async def get(self, job_id: str) -> Optional[JobRecord]:
        async with self._session() as session:
            model = await session.get(JobModel, job_id)
            if model is None:
                return None
            # Refresh artifact_index via relationship.
            await session.refresh(model, ["artifact_index"])
            return _record_from_model(model)

    async def list(self, *, limit: Optional[int] = None) -> Tuple[List[JobRecord], int]:
        async with self._session() as session:
            total_q = select(func.count()).select_from(JobModel)
            total = (await session.execute(total_q)).scalar_one()

            stmt = select(JobModel).order_by(JobModel.created_at.desc())
            if limit is not None:
                stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            models = result.scalars().all()
            records = [_record_from_model(m) for m in models]
            return records, int(total)

    async def update(self, job_id: str, **fields: Any) -> JobRecord:
        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            raise RepositoryError(f"update received unknown fields: {sorted(unknown)}")
        if not fields:
            existing = await self.get(job_id)
            if existing is None:
                raise JobNotFoundError(job_id)
            return existing
        payload: Dict[str, Any] = {key: deepcopy(value) for key, value in fields.items()}

        async with self._session() as session:
            stmt = select(JobModel).where(JobModel.id == job_id).with_for_update()
            model = (await session.execute(stmt)).scalar_one_or_none()
            if model is None:
                raise JobNotFoundError(job_id)
            # Normalize JSON-stored containers so SQLAlchemy detects the
            # mutation (asyncpg JSONB column).
            for key in (
                "request",
                "effective_request",
                "artifacts",
                "run_summary",
            ):
                if key in payload and payload[key] is not None:
                    payload[key] = deepcopy(payload[key])
            if "logs_tail" in payload and payload["logs_tail"] is not None:
                payload["logs_tail"] = list(payload["logs_tail"])
            if "error" in payload and payload["error"] is not None:
                payload["error"] = deepcopy(payload["error"])

            for key, value in payload.items():
                setattr(model, key, value)
            model.version += 1
            await session.commit()
            await session.refresh(model, ["artifact_index"])
            return _record_from_model(model)

    async def append_log(self, job_id: str, line: str, *, tail_limit: int) -> None:
        if tail_limit <= 0:
            raise RepositoryError("tail_limit must be positive")
        await self.append_logs(job_id, [line], tail_limit=tail_limit)

    async def append_logs(self, job_id: str, lines: Iterable[str], *, tail_limit: int) -> None:
        if tail_limit <= 0:
            raise RepositoryError("tail_limit must be positive")
        batch = list(lines)
        if not batch:
            return
        async with self._session() as session:
            stmt = select(JobModel).where(JobModel.id == job_id).with_for_update()
            model = (await session.execute(stmt)).scalar_one_or_none()
            if model is None:
                raise JobNotFoundError(job_id)
            tail = list(model.logs_tail or [])
            tail.extend(batch)
            if len(tail) > tail_limit:
                tail = tail[-tail_limit:]
            model.logs_tail = tail
            model.version += 1
            await session.commit()

    async def set_artifacts(
        self,
        job_id: str,
        artifacts: Dict[str, Any],
        artifact_lookup: Dict[str, Path],
    ) -> None:
        artifacts_snapshot = deepcopy(artifacts)
        artifact_lookup_snapshot = dict(artifact_lookup)
        for attempt in range(_OPTIMISTIC_LOCK_RETRIES):
            async with self._session() as session:
                model = await session.get(JobModel, job_id)
                if model is None:
                    raise JobNotFoundError(job_id)
                current_version = model.version
                stmt = (
                    update(JobModel)
                    .where(
                        JobModel.id == job_id,
                        JobModel.version == current_version,
                    )
                    .values(artifacts=artifacts_snapshot, version=current_version + 1)
                )
                result = cast(CursorResult[Any], await session.execute(stmt))
                if result.rowcount == 0:
                    await session.rollback()
                    if attempt + 1 < _OPTIMISTIC_LOCK_RETRIES:
                        await asyncio.sleep(0.01 * (attempt + 1))
                        continue
                    raise RepositoryError(f"optimistic-lock conflict on job {job_id} set_artifacts")
                await session.execute(delete(JobArtifactModel).where(JobArtifactModel.job_id == job_id))
                for path_str, absolute in artifact_lookup_snapshot.items():
                    session.add(
                        JobArtifactModel(
                            job_id=job_id,
                            path=path_str,
                            absolute_path=str(absolute),
                        )
                    )
                await session.commit()
                return

    async def delete(self, job_id: str) -> None:
        async with self._session() as session:
            await session.execute(select(JobModel.id).where(JobModel.id == job_id).with_for_update())
            active_dispatch = (
                select(DispatchAttemptModel.job_id)
                .where(DispatchAttemptModel.job_id == job_id, DispatchAttemptModel.state.in_(list(_ACTIVE_STATES)))
                .exists()
            )
            if bool((await session.execute(select(active_dispatch))).scalar_one()):
                raise RepositoryError("cannot delete job with an active durable dispatch")
            await session.execute(delete(JobEventSequenceModel).where(JobEventSequenceModel.job_id == job_id))
            # job_events.job_id has no SQL FK (see models.py), so cascade
            # event deletion explicitly here. job_artifacts still cascades
            # via its FK on jobs.id.
            await session.execute(delete(JobEventModel).where(JobEventModel.job_id == job_id))
            await session.execute(delete(JobModel).where(JobModel.id == job_id))
            await session.commit()

    async def cleanup_expired(self, now: float, retention_seconds: float) -> List[str]:
        async with self._session() as session:
            cutoff = now - retention_seconds
            ids_stmt = (
                select(JobModel.id)
                .where(
                    JobModel.finished_at.is_not(None),
                    JobModel.finished_at <= cutoff,
                    ~select(DispatchAttemptModel.job_id)
                    .where(DispatchAttemptModel.job_id == JobModel.id, DispatchAttemptModel.state.in_(list(_ACTIVE_STATES)))
                    .exists(),
                )
                .order_by(JobModel.id)
                .with_for_update()
            )
            expired_ids = list((await session.execute(ids_stmt)).scalars().all())
            if expired_ids:
                # Lock counters before event rows, matching terminal publication.
                await session.execute(delete(JobEventSequenceModel).where(JobEventSequenceModel.job_id.in_(expired_ids)))
                # Manual cascade for job_events (no SQL FK; see models.py).
                await session.execute(delete(JobEventModel).where(JobEventModel.job_id.in_(expired_ids)))
                await session.execute(delete(JobModel).where(JobModel.id.in_(expired_ids)))
                await session.commit()
            return expired_ids

    async def count_active(self) -> int:
        async with self._session() as session:
            stmt = select(func.count()).select_from(JobModel).where(JobModel.state.in_(list(_ACTIVE_STATES)))
            return int((await session.execute(stmt)).scalar_one())

    async def sweep_orphaned(
        self,
        *,
        live_job_ids: Optional[List[str]] = None,
        reason_code: str = "worker_lost_on_restart",
        now: Optional[float] = None,
    ) -> List[str]:
        import time as _time

        stamp = _time.time() if now is None else now
        live = list(live_job_ids or [])
        async with self._session() as session:
            stmt = select(JobModel.id).where(
                JobModel.state.in_(list(_ACTIVE_STATES)),
                ~select(DispatchAttemptModel.job_id).where(DispatchAttemptModel.job_id == JobModel.id).exists(),
            )
            if live:
                stmt = stmt.where(~JobModel.id.in_(live))
            orphan_ids = list((await session.execute(stmt)).scalars().all())
            if not orphan_ids:
                return []
            error_payload = {
                "code": reason_code,
                "message": "Process did not survive backend restart.",
                "retriable": True,
            }
            await session.execute(
                update(JobModel)
                .where(JobModel.id.in_(orphan_ids))
                .values(
                    state="worker_lost",
                    finished_at=stamp,
                    done_published_at=stamp,
                    last_event_at=stamp,
                    error=error_payload,
                    version=JobModel.version + 1,
                )
            )
            await session.commit()
            return orphan_ids

    async def reset(self) -> None:
        """Test-only: drop and recreate all orchestrator tables.

        Routes through ``_ensure_engine()`` so ``self._engine`` /
        ``self._session_factory`` are populated even if no other API was
        called first. That makes ``close()`` reliably dispose the shared
        engine even when a test exercises only ``reset()``.
        """
        await self._ensure_engine()
        assert self._engine is not None  # populated by _ensure_engine
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        # Dispose the shared engine for our database_url even if
        # _ensure_engine was never called on this instance: a sibling
        # PostgresJobEventStore may have constructed the engine via
        # _SharedEngine.get(database_url), and we own the lifecycle.
        await _SharedEngine.dispose(self._database_url)
        self._engine = None
        self._session_factory = None


def event_retention_per_job() -> int:
    """Resolve the positive replay bound; zero must never mean unbounded."""
    raw = os.getenv("TP_ORCHESTRATOR_EVENT_RETENTION_PER_JOB", str(DEFAULT_EVENT_RETENTION_PER_JOB)).strip()
    if not raw.isascii() or not raw.isdecimal() or int(raw) <= 0:
        raise RepositoryError("TP_ORCHESTRATOR_EVENT_RETENTION_PER_JOB must be a positive integer")
    return int(raw)


async def append_event_in_session(
    session: AsyncSession,
    job_id: str,
    event_type: str,
    payload: Dict[str, Any],
    created_at: float,
    *,
    per_job_cap: Optional[int] = None,
) -> JobEvent:
    """Allocate, insert and prune under one counter lock in caller transaction.

    UPSERT locks the counter row across independent processes, including the
    first append for an orphan job id. The legacy MAX seed supports preexisting
    events; subsequent sequence allocation never depends on retained history.
    Callers that lock jobs must take that lock before invoking this helper.
    """
    cap = event_retention_per_job() if per_job_cap is None else per_job_cap
    if type(cap) is not int or cap <= 0:
        raise RepositoryError("per_job_cap must be a positive integer")
    payload_snapshot = deepcopy(payload)
    legacy_next = select(func.coalesce(func.max(JobEventModel.seq), 0) + 1).where(JobEventModel.job_id == job_id)
    counter_stmt = (
        insert(JobEventSequenceModel)
        .values(job_id=job_id, last_seq=legacy_next.scalar_subquery())
        .on_conflict_do_update(
            index_elements=[JobEventSequenceModel.job_id],
            set_={"last_seq": JobEventSequenceModel.last_seq + 1},
        )
        .returning(JobEventSequenceModel.last_seq)
    )
    seq = int((await session.execute(counter_stmt)).scalar_one())
    session.add(
        JobEventModel(job_id=job_id, seq=seq, event_type=event_type, payload=deepcopy(payload_snapshot), created_at=created_at)
    )
    # Flush before pruning even when the session's autoflush is disabled.
    await session.flush()
    if seq > cap:
        await session.execute(delete(JobEventModel).where(JobEventModel.job_id == job_id, JobEventModel.seq <= seq - cap))
    return JobEvent(job_id=job_id, seq=seq, event_type=event_type, payload=payload_snapshot, created_at=created_at)


class PostgresJobEventStore(JobEventStore):
    """Durable ``JobEventStore`` backed by the same engine."""

    supports_live_replay = True

    def __init__(self, *, database_url: str, per_job_cap: Optional[int] = None) -> None:
        if not database_url:
            raise RepositoryError("database_url must be a non-empty string")
        self._per_job_cap = event_retention_per_job() if per_job_cap is None else per_job_cap
        if type(self._per_job_cap) is not int or self._per_job_cap <= 0:
            raise RepositoryError("per_job_cap must be a positive integer")
        self._database_url = database_url
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def _ensure_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            _, self._session_factory = await _SharedEngine.get(self._database_url)
        return self._session_factory

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        factory = await self._ensure_session_factory()
        async with factory() as session:
            yield session

    async def append(
        self,
        job_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        created_at: float,
    ) -> JobEvent:
        payload_snapshot = deepcopy(payload)
        async with self._session() as session:
            async with session.begin():
                return await append_event_in_session(
                    session, job_id, event_type, payload_snapshot, created_at, per_job_cap=self._per_job_cap
                )

    async def replay_batch(self, job_id: str, *, after_seq: int) -> List[JobEvent]:
        """Release the DB session before yielding to a potentially slow client."""
        async with self._session() as session:
            stmt = (
                select(JobEventModel)
                .where(JobEventModel.job_id == job_id, JobEventModel.seq > after_seq)
                .order_by(JobEventModel.seq.asc())
                .limit(self._per_job_cap)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                JobEvent(
                    job_id=row.job_id,
                    seq=row.seq,
                    event_type=row.event_type,
                    payload=deepcopy(row.payload),
                    created_at=row.created_at,
                )
                for row in rows
            ]

    async def events_since(
        self,
        job_id: str,
        *,
        after_seq: Optional[int] = None,
    ) -> AsyncIterator[JobEvent]:
        """Stream events incrementally so long histories don't spike memory.

        Uses ``session.stream_scalars`` so rows are produced as they
        arrive from Postgres rather than materialized via ``.all()``.
        The session stays open for the duration of the iteration; the
        caller must fully consume the iterator (or let it go out of
        scope) before opening another session against the same engine.
        """
        async with self._session() as session:
            stmt = select(JobEventModel).where(JobEventModel.job_id == job_id).order_by(JobEventModel.seq.asc())
            if after_seq is not None:
                stmt = stmt.where(JobEventModel.seq > after_seq)
            stream = await session.stream_scalars(stmt)
            async for row in stream:
                yield JobEvent(
                    job_id=row.job_id,
                    seq=row.seq,
                    event_type=row.event_type,
                    payload=deepcopy(row.payload or {}),
                    created_at=row.created_at,
                )

    async def reset(self) -> None:
        async with self._session() as session:
            await session.execute(delete(JobEventSequenceModel))
            await session.execute(delete(JobEventModel))
            await session.commit()

    async def close(self) -> None:
        # The shared engine is disposed by ``PostgresJobRepository.close``;
        # event store sessions live or die with that engine.
        self._session_factory = None


class PostgresOperationalAuditStore(OperationalAuditStore):
    """Append-only operational audit log backed by the orchestrator database."""

    def __init__(self, *, database_url: str) -> None:
        if not database_url:
            raise RepositoryError("database_url must be a non-empty string")
        self._database_url = database_url
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def _ensure_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            _, self._session_factory = await _SharedEngine.get(self._database_url)
        return self._session_factory

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        factory = await self._ensure_session_factory()
        async with factory() as session:
            yield session

    async def append(self, record: OperationalAuditRecord) -> None:
        snapshot = record.copy()
        async with self._session() as session:
            session.add(
                OperationalAuditEventModel(
                    created_at=snapshot.created_at,
                    action=snapshot.action,
                    decision=snapshot.decision,
                    tenant_id=snapshot.tenant_id,
                    job_id=snapshot.job_id,
                    actor=snapshot.actor,
                    request_context=snapshot.request_context,
                    details=snapshot.details,
                )
            )
            await session.commit()

    async def close(self) -> None:
        # The shared engine is disposed by ``PostgresJobRepository.close``.
        self._session_factory = None


__all__ = ["PostgresJobEventStore", "PostgresJobRepository", "PostgresOperationalAuditStore"]
