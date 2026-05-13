"""Postgres-backed ``JobRepository`` and ``JobEventStore``.

Phase 1.B - durable orchestrator state. The repository projects the
persistent slice of orchestrator jobs into the ORM models defined in
``transformation_portal.orchestrator.models`` and back to the
``JobRecord`` dataclass that the rest of the orchestrator already knows
about.

Concurrency: every ``update`` and ``set_artifacts`` call increments the
row's ``version`` column under an optimistic-concurrency guard and
retries on conflict.

Memory ``logs_tail`` parity: this layer keeps the legacy "bounded
in-memory tail" semantic - ``append_log`` reads the current tail,
appends, trims, and writes back atomically inside a transaction.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from transformation_portal.orchestrator.models import (
    Base,
    JobArtifactModel,
    JobEventModel,
    JobModel,
)
from transformation_portal.orchestrator.storage.base import (
    JobEvent,
    JobEventStore,
    JobNotFoundError,
    JobRecord,
    JobRepository,
    RepositoryError,
)

logger = logging.getLogger(__name__)

_ACTIVE_STATES = {"queued", "running"}
_OPTIMISTIC_LOCK_RETRIES = 3

# Fields callers may patch via ``update``. ``id``/``created_at`` are immutable;
# ``artifact_lookup`` is set via ``set_artifacts``; ``version`` is repo-managed.
_UPDATABLE_FIELDS = {
    "state",
    "progress",
    "started_at",
    "finished_at",
    "done_published_at",
    "last_event_at",
    "exit_code",
    "cancel_requested",
    "request",
    "effective_request",
    "logs_tail",
    "artifacts",
    "run_summary",
    "error",
}


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
        request=dict(model.request or {}),
        effective_request=dict(model.effective_request or {}),
        logs_tail=list(model.logs_tail or []),
        artifacts=dict(model.artifacts or {}),
        artifact_lookup=artifact_lookup,
        run_summary=dict(model.run_summary or {}),
        error=None if model.error is None else dict(model.error),
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
        async with self._session() as session:
            session.add(
                JobModel(
                    id=record.id,
                    created_at=record.created_at,
                    state=record.state,
                    progress=record.progress,
                    started_at=record.started_at,
                    finished_at=record.finished_at,
                    done_published_at=record.done_published_at,
                    last_event_at=record.last_event_at,
                    exit_code=record.exit_code,
                    cancel_requested=record.cancel_requested,
                    request=dict(record.request),
                    effective_request=dict(record.effective_request),
                    logs_tail=list(record.logs_tail),
                    artifacts=dict(record.artifacts),
                    run_summary=dict(record.run_summary),
                    error=None if record.error is None else dict(record.error),
                    version=1,
                )
            )
            for path_str, absolute in record.artifact_lookup.items():
                session.add(
                    JobArtifactModel(
                        job_id=record.id,
                        path=path_str,
                        absolute_path=str(absolute),
                    )
                )
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise RepositoryError(f"job id already exists: {record.id}") from exc

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

        for attempt in range(_OPTIMISTIC_LOCK_RETRIES):
            async with self._session() as session:
                model = await session.get(JobModel, job_id)
                if model is None:
                    raise JobNotFoundError(job_id)
                current_version = model.version
                payload: Dict[str, Any] = dict(fields)
                # Normalize JSON-stored containers so SQLAlchemy detects the
                # mutation (asyncpg JSONB column).
                for key in (
                    "request",
                    "effective_request",
                    "artifacts",
                    "run_summary",
                ):
                    if key in payload and payload[key] is not None:
                        payload[key] = dict(payload[key])
                if "logs_tail" in payload and payload["logs_tail"] is not None:
                    payload["logs_tail"] = list(payload["logs_tail"])
                if "error" in payload and payload["error"] is not None:
                    payload["error"] = dict(payload["error"])

                stmt = (
                    update(JobModel)
                    .where(
                        JobModel.id == job_id,
                        JobModel.version == current_version,
                    )
                    .values(version=current_version + 1, **payload)
                )
                result = await session.execute(stmt)
                if result.rowcount == 0:
                    await session.rollback()
                    if attempt + 1 < _OPTIMISTIC_LOCK_RETRIES:
                        await asyncio.sleep(0.01 * (attempt + 1))
                        continue
                    raise RepositoryError(
                        f"optimistic-lock conflict on job {job_id} after " f"{_OPTIMISTIC_LOCK_RETRIES} retries"
                    )
                await session.commit()
                refreshed = await session.get(JobModel, job_id)
                assert refreshed is not None  # just updated successfully
                await session.refresh(refreshed, ["artifact_index"])
                return _record_from_model(refreshed)

        # Should be unreachable; included for type-checkers.
        raise RepositoryError(f"failed to update job {job_id}")

    async def append_log(self, job_id: str, line: str, *, tail_limit: int) -> None:
        if tail_limit <= 0:
            raise RepositoryError("tail_limit must be positive")
        for attempt in range(_OPTIMISTIC_LOCK_RETRIES):
            async with self._session() as session:
                model = await session.get(JobModel, job_id)
                if model is None:
                    raise JobNotFoundError(job_id)
                current_version = model.version
                tail = list(model.logs_tail or [])
                tail.append(line)
                if len(tail) > tail_limit:
                    tail = tail[-tail_limit:]
                stmt = (
                    update(JobModel)
                    .where(
                        JobModel.id == job_id,
                        JobModel.version == current_version,
                    )
                    .values(logs_tail=tail, version=current_version + 1)
                )
                result = await session.execute(stmt)
                if result.rowcount == 0:
                    await session.rollback()
                    if attempt + 1 < _OPTIMISTIC_LOCK_RETRIES:
                        await asyncio.sleep(0.01 * (attempt + 1))
                        continue
                    raise RepositoryError(f"optimistic-lock conflict on job {job_id} log append")
                await session.commit()
                return

    async def set_artifacts(
        self,
        job_id: str,
        artifacts: Dict[str, Any],
        artifact_lookup: Dict[str, Path],
    ) -> None:
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
                    .values(artifacts=dict(artifacts), version=current_version + 1)
                )
                result = await session.execute(stmt)
                if result.rowcount == 0:
                    await session.rollback()
                    if attempt + 1 < _OPTIMISTIC_LOCK_RETRIES:
                        await asyncio.sleep(0.01 * (attempt + 1))
                        continue
                    raise RepositoryError(f"optimistic-lock conflict on job {job_id} set_artifacts")
                await session.execute(delete(JobArtifactModel).where(JobArtifactModel.job_id == job_id))
                for path_str, absolute in artifact_lookup.items():
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
            await session.execute(delete(JobModel).where(JobModel.id == job_id))
            await session.commit()

    async def cleanup_expired(self, now: float, retention_seconds: float) -> List[str]:
        async with self._session() as session:
            cutoff = now - retention_seconds
            ids_stmt = select(JobModel.id).where(
                JobModel.finished_at.is_not(None),
                JobModel.finished_at <= cutoff,
            )
            expired_ids = list((await session.execute(ids_stmt)).scalars().all())
            if expired_ids:
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
            stmt = select(JobModel.id).where(JobModel.state.in_(list(_ACTIVE_STATES)))
            if live:
                stmt = stmt.where(~JobModel.id.in_(live))
            orphan_ids = list((await session.execute(stmt)).scalars().all())
            if not orphan_ids:
                return []
            error_payload = {
                "code": reason_code,
                "message": "Process did not survive backend restart.",
            }
            await session.execute(
                update(JobModel)
                .where(JobModel.id.in_(orphan_ids))
                .values(
                    state="failed",
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
        """Test-only: drop and recreate all orchestrator tables."""
        engine = (await _SharedEngine.get(self._database_url))[0]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        if self._engine is not None:
            await _SharedEngine.dispose(self._database_url)
            self._engine = None
            self._session_factory = None


class PostgresJobEventStore(JobEventStore):
    """Durable ``JobEventStore`` backed by the same engine."""

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

    async def append(
        self,
        job_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        created_at: float,
    ) -> JobEvent:
        async with self._session() as session:
            current_max_stmt = select(func.coalesce(func.max(JobEventModel.seq), 0)).where(JobEventModel.job_id == job_id)
            current_max = int((await session.execute(current_max_stmt)).scalar_one())
            seq = current_max + 1
            session.add(
                JobEventModel(
                    job_id=job_id,
                    seq=seq,
                    event_type=event_type,
                    payload=dict(payload),
                    created_at=created_at,
                )
            )
            await session.commit()
            return JobEvent(
                job_id=job_id,
                seq=seq,
                event_type=event_type,
                payload=dict(payload),
                created_at=created_at,
            )

    async def events_since(
        self,
        job_id: str,
        *,
        after_seq: Optional[int] = None,
    ) -> AsyncIterator[JobEvent]:
        async with self._session() as session:
            stmt = select(JobEventModel).where(JobEventModel.job_id == job_id).order_by(JobEventModel.seq.asc())
            if after_seq is not None:
                stmt = stmt.where(JobEventModel.seq > after_seq)
            result = await session.execute(stmt)
            for row in result.scalars().all():
                yield JobEvent(
                    job_id=row.job_id,
                    seq=row.seq,
                    event_type=row.event_type,
                    payload=dict(row.payload or {}),
                    created_at=row.created_at,
                )

    async def reset(self) -> None:
        async with self._session() as session:
            await session.execute(delete(JobEventModel))
            await session.commit()

    async def close(self) -> None:
        # The shared engine is disposed by ``PostgresJobRepository.close``;
        # event store sessions live or die with that engine.
        self._session_factory = None


__all__ = ["PostgresJobEventStore", "PostgresJobRepository"]
