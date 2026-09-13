"""Postgres admission, once-only dispatch claims, and publication authority.

This extends the existing operational audit boundary. Mutable job/event
projections are written in the same transaction as their canonical record
and outbox entry. Redis delivery and staged objects are never authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from transformation_portal.core.execution_plan import EXECUTION_COMPLETE, parse_execution_plan_json
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator
from transformation_portal.orchestrator.models import (
    AdmissionCapacityModel,
    CommittedGenerationModel,
    DispatchAttemptModel,
    DispatchPlanModel,
    JobModel,
    OperationalOutboxModel,
    OperationalRecordModel,
)
from transformation_portal.orchestrator.storage.base import JobRecord, RepositoryError
from transformation_portal.orchestrator.storage.postgres import PostgresOperationalAuditStore, append_event_in_session

GENERATION_CLEANUP_RECHECK_SECONDS = 3600.0

TERMINAL_STATES = frozenset({"succeeded", "partial", "failed", "canceled", "worker_lost"})


class AdmissionRejected(RepositoryError):
    def __init__(self, scope: str) -> None:
        self.scope = scope
        super().__init__(f"active job capacity exhausted: {scope}")


class DispatchAuthorityLost(RepositoryError):
    """A dispatch is claimed, expired, tombstoned, or does not match authority."""


class PostgresOperationalRecordStore(PostgresOperationalAuditStore):
    """One authoritative boundary sharing the existing Postgres engine."""

    @staticmethod
    async def _now(session: AsyncSession) -> float:
        now = await session.scalar(select(func.extract("epoch", func.clock_timestamp())))
        if now is None:
            raise RepositoryError("Postgres clock is unavailable")
        return float(now)

    @staticmethod
    async def _allow_projection(session: AsyncSession) -> None:
        await session.execute(text("SELECT set_config('tp.operational_write', 'on', true)"))

    @staticmethod
    async def _capacities(
        session: AsyncSession, tenant_id: str, limits: Optional[tuple[int, int]] = None
    ) -> list[AdmissionCapacityModel]:
        rows = []
        for index, scope in enumerate(("global", f"tenant:{tenant_id}")):
            if limits is not None:
                limit = limits[index]
                if type(limit) is not int or limit <= 0:
                    raise RepositoryError("distributed admission requires positive global and tenant limits")
                await session.execute(
                    insert(AdmissionCapacityModel).values(scope=scope, limit=limit, active=0).on_conflict_do_nothing()
                )
            row = await session.scalar(
                select(AdmissionCapacityModel).where(AdmissionCapacityModel.scope == scope).with_for_update()
            )
            if row is None:
                raise RepositoryError("missing authoritative capacity row")
            if limits is not None and row.limit != limits[index]:
                raise RepositoryError(f"conflicting configured admission limit for {scope}")
            rows.append(row)
        return rows

    @staticmethod
    def _record(session: AsyncSession, attempt: DispatchAttemptModel, kind: str, now: float, payload: dict[str, Any]) -> None:
        raw = canonicalize_json({"schema": "tp.operational.record.v1", "kind": kind, **payload})
        if len(raw) > 1_048_576:
            raise RepositoryError("operational record exceeds byte limit")
        session.add(
            OperationalRecordModel(
                job_id=attempt.job_id,
                tenant_id=attempt.tenant_id,
                kind=kind,
                created_at=now,
                canonical_bytes=raw,
                digest=hashlib.sha256(raw).hexdigest(),
            )
        )

    @staticmethod
    def _locator(attempt: DispatchAttemptModel) -> DispatchLocator:
        return DispatchLocator(
            job_id=attempt.job_id,
            tenant_id=attempt.tenant_id,
            attempt_id=attempt.attempt_id,
            dispatch_id=attempt.dispatch_id,
            plan_digest=attempt.plan_digest,
            api_version=attempt.api_version,
        )

    async def admit(
        self,
        record: JobRecord,
        plan_bytes: bytes,
        *,
        tenant_id: str,
        output_root: str,
        requested_output_root: str,
        global_limit: int,
        tenant_limit: int,
    ) -> DispatchLocator:
        plan = parse_execution_plan_json(plan_bytes)
        if plan.to_canonical_json().encode("utf-8") != plan_bytes or plan.configuration_completeness != EXECUTION_COMPLETE:
            raise RepositoryError("admission requires exact canonical execution-complete plan bytes")
        if record.state != "queued" or record.effective_request.get("tenant_id") != tenant_id:
            raise RepositoryError("admission requires a queued tenant-bound job")
        for root in (output_root, requested_output_root):
            if not Path(root).is_absolute() or "\x00" in root:
                raise RepositoryError("dispatch output roots must be authorized absolute paths")
        digest = hashlib.sha256(plan_bytes).hexdigest()
        locator = DispatchLocator(
            job_id=record.id,
            attempt_id=uuid.uuid4().hex,
            dispatch_id=uuid.uuid4().hex,
            tenant_id=tenant_id,
            plan_digest=digest,
        )
        async with self._session() as session:
            async with session.begin():
                capacities = await self._capacities(session, tenant_id, (global_limit, tenant_limit))
                for capacity in capacities:
                    if capacity.active >= capacity.limit:
                        raise AdmissionRejected(capacity.scope)
                now = await self._now(session)
                await session.execute(
                    insert(DispatchPlanModel).values(digest=digest, canonical_bytes=plan_bytes).on_conflict_do_nothing()
                )
                existing = await session.get(DispatchPlanModel, digest)
                if existing is None or existing.canonical_bytes != plan_bytes:
                    raise RepositoryError("canonical plan digest collision")
                model = JobModel(
                    **{
                        key: deepcopy(getattr(record, key))
                        for key in (
                            "id",
                            "created_at",
                            "state",
                            "progress",
                            "request",
                            "effective_request",
                            "logs_tail",
                            "artifacts",
                            "run_summary",
                        )
                    }
                )
                session.add(model)
                attempt = DispatchAttemptModel(
                    **{key: value for key, value in locator.to_payload().items() if key != "schema"},
                    output_root=output_root,
                    requested_output_root=requested_output_root,
                    state="queued",
                    admitted_at=now,
                    lease_epoch=0,
                )
                session.add(attempt)
                for capacity in capacities:
                    capacity.active += 1
                self._record(
                    session,
                    attempt,
                    "admitted",
                    now,
                    {
                        "locator": locator.to_payload(),
                        "output_root": output_root,
                        "requested_output_root": requested_output_root,
                    },
                )
                session.add(
                    OperationalOutboxModel(job_id=record.id, kind="dispatch", payload=locator.to_payload(), created_at=now)
                )
        return locator

    async def get_locator(self, job_id: str) -> Optional[DispatchLocator]:
        async with self._session() as session:
            attempt = await session.get(DispatchAttemptModel, job_id)
            return None if attempt is None else self._locator(attempt)

    @staticmethod
    def _matches(attempt: DispatchAttemptModel, locator: DispatchLocator) -> bool:
        return PostgresOperationalRecordStore._locator(attempt) == locator

    async def fetch_plan(self, locator: DispatchLocator) -> bytes:
        async with self._session() as session:
            attempt = await session.get(DispatchAttemptModel, locator.job_id)
            if attempt is None or not self._matches(attempt, locator):
                raise DispatchAuthorityLost("dispatch locator does not match immutable authority")
            model = await session.get(DispatchPlanModel, locator.plan_digest)
            if model is None or hashlib.sha256(model.canonical_bytes).hexdigest() != locator.plan_digest:
                raise DispatchAuthorityLost("canonical dispatch plan is unavailable or corrupted")
            raw = bytes(model.canonical_bytes)
        plan = parse_execution_plan_json(raw)
        if plan.to_canonical_json().encode("utf-8") != raw or plan.configuration_completeness != EXECUTION_COMPLETE:
            raise DispatchAuthorityLost("stored plan is not canonical execution authority")
        return raw

    async def claim_dispatch(self, locator: DispatchLocator, holder: str, *, lease_seconds: float) -> DispatchFence:
        self._validate_lease(holder, lease_seconds)
        async with self._session() as session:
            async with session.begin():
                attempt = await session.scalar(
                    select(DispatchAttemptModel).where(DispatchAttemptModel.job_id == locator.job_id).with_for_update()
                )
                if (
                    attempt is None
                    or not self._matches(attempt, locator)
                    or attempt.state != "queued"
                    or attempt.lease_epoch != 0
                ):
                    raise DispatchAuthorityLost("dispatch is unknown, already claimed, or tombstoned")
                now = await self._now(session)
                # Epoch comes from a database sequence; no process-local value can grant authority.
                epoch = int(await session.scalar(text("SELECT nextval('dispatch_lease_epoch_seq')")))
                attempt.state, attempt.holder, attempt.lease_epoch = "running", holder, epoch
                attempt.lease_valid_until = now + lease_seconds
                job = await session.scalar(select(JobModel).where(JobModel.id == locator.job_id).with_for_update())
                if job is None or job.state != "queued":
                    raise DispatchAuthorityLost("job projection is not admitted")
                await self._allow_projection(session)
                job.state, job.started_at = "running", now
                self._record(
                    session,
                    attempt,
                    "claimed",
                    now,
                    {
                        "dispatch_id": locator.dispatch_id,
                        "holder": holder,
                        "lease_epoch": epoch,
                        "lease_valid_until": attempt.lease_valid_until,
                    },
                )
                return DispatchFence(
                    locator, holder, epoch, attempt.lease_valid_until, attempt.output_root, attempt.requested_output_root
                )

    @staticmethod
    def _validate_lease(holder: str, lease_seconds: float) -> None:
        if (
            not isinstance(holder, str)
            or not holder
            or len(holder) > 128
            or not math.isfinite(lease_seconds)
            or lease_seconds <= 0
        ):
            raise RepositoryError("invalid holder or lease duration")

    async def _locked_fence(self, session: AsyncSession, fence: DispatchFence) -> tuple[DispatchAttemptModel, float]:
        attempt = await session.scalar(
            select(DispatchAttemptModel).where(DispatchAttemptModel.job_id == fence.locator.job_id).with_for_update()
        )
        now = await self._now(session)
        if (
            attempt is None
            or not self._matches(attempt, fence.locator)
            or attempt.state != "running"
            or attempt.holder != fence.holder
            or attempt.lease_epoch != fence.lease_epoch
            or attempt.lease_valid_until is None
            or now >= attempt.lease_valid_until
        ):
            raise DispatchAuthorityLost("dispatch publication authority is expired or revoked")
        return attempt, now

    async def renew_dispatch(self, fence: DispatchFence, *, lease_seconds: float) -> None:
        """Called only after a confirmed successful broker heartbeat."""
        self._validate_lease(fence.holder, lease_seconds)
        async with self._session() as session:
            async with session.begin():
                attempt, now = await self._locked_fence(session, fence)
                attempt.lease_valid_until = now + lease_seconds

    async def _terminal(
        self,
        session: AsyncSession,
        attempt: DispatchAttemptModel,
        capacities: list[AdmissionCapacityModel],
        *,
        state: str,
        now: float,
        exit_code: Optional[int],
        error: Optional[dict[str, Any]],
        artifacts: Optional[dict[str, Any]] = None,
        run_summary: Optional[dict[str, Any]] = None,
        requires_live_fence: bool = False,
    ) -> dict[str, Any]:
        if state not in TERMINAL_STATES:
            raise RepositoryError("invalid terminal state")
        job = await session.scalar(select(JobModel).where(JobModel.id == attempt.job_id).with_for_update())
        if job is None:
            raise RepositoryError("missing admitted job projection")
        now = await self._now(session)
        if requires_live_fence and (attempt.lease_valid_until is None or now >= attempt.lease_valid_until):
            raise DispatchAuthorityLost("lease expired while waiting for terminal projection lock")
        await self._allow_projection(session)
        attempt.state, attempt.finished_at, attempt.lease_valid_until = state, now, None
        for capacity in capacities:
            if capacity.active <= 0:
                raise RepositoryError("admission capacity underflow")
            capacity.active -= 1
        job.state, job.finished_at, job.done_published_at, job.last_event_at = state, now, now, now
        job.exit_code, job.error = exit_code, deepcopy(error)
        job.cancel_requested = state == "canceled"
        if state == "succeeded":
            job.progress = 100
        if artifacts is not None:
            job.artifacts = deepcopy(artifacts)
        if run_summary is not None:
            job.run_summary = deepcopy(run_summary)
        payload = {
            "id": job.id,
            "state": state,
            "exit_code": exit_code,
            "error": deepcopy(error),
            "artifacts": deepcopy(job.artifacts),
            "run_summary": deepcopy(job.run_summary) or None,
        }
        self._record(
            session,
            attempt,
            "terminal",
            now,
            {
                "state": state,
                "dispatch_id": attempt.dispatch_id,
                "lease_epoch": attempt.lease_epoch,
                "generation_id": attempt.generation_id,
            },
        )
        event = await append_event_in_session(session, job.id, "done", payload, now)
        seq = event.seq
        session.add(
            OperationalOutboxModel(
                job_id=job.id, kind="event", payload={"event_type": "done", "seq": seq, "payload": payload}, created_at=now
            )
        )
        return payload

    async def finish_dispatch(
        self, fence: DispatchFence, *, state: str, exit_code: Optional[int] = None, error: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        async with self._session() as session:
            async with session.begin():
                capacities = await self._capacities(session, fence.locator.tenant_id)
                attempt, now = await self._locked_fence(session, fence)
                return await self._terminal(
                    session,
                    attempt,
                    capacities,
                    state=state,
                    now=now,
                    exit_code=exit_code,
                    error=error,
                    requires_live_fence=True,
                )

    async def cancel_dispatch(self, job_id: str, tenant_id: str) -> bool:
        async with self._session() as session:
            async with session.begin():
                capacities = await self._capacities(session, tenant_id)
                attempt = await session.scalar(
                    select(DispatchAttemptModel).where(DispatchAttemptModel.job_id == job_id).with_for_update()
                )
                if attempt is None or attempt.tenant_id != tenant_id:
                    raise DispatchAuthorityLost("unknown tenant dispatch")
                if attempt.state in TERMINAL_STATES:
                    return False
                await self._terminal(
                    session, attempt, capacities, state="canceled", now=await self._now(session), exit_code=None, error=None
                )
                return True

    async def expire_dispatches(self, *, limit: int = 100) -> list[str]:
        """DB-clock expiry is independent of whether the Redis sweep survived."""
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(DispatchAttemptModel.job_id, DispatchAttemptModel.tenant_id)
                    .where(
                        DispatchAttemptModel.state == "running",
                        DispatchAttemptModel.lease_valid_until <= func.extract("epoch", func.clock_timestamp()),
                    )
                    .order_by(DispatchAttemptModel.lease_valid_until)
                    .limit(min(max(limit, 1), 1000))
                )
            ).all()
        expired = []
        for job_id, tenant_id in rows:
            async with self._session() as session:
                async with session.begin():
                    capacities = await self._capacities(session, tenant_id)
                    attempt = await session.scalar(
                        select(DispatchAttemptModel).where(DispatchAttemptModel.job_id == job_id).with_for_update()
                    )
                    now = await self._now(session)
                    if (
                        attempt is None
                        or attempt.state != "running"
                        or attempt.lease_valid_until is None
                        or attempt.lease_valid_until > now
                    ):
                        continue
                    await self._terminal(
                        session,
                        attempt,
                        capacities,
                        state="worker_lost",
                        now=now,
                        exit_code=None,
                        error={
                            "code": "worker_lost_via_lease_reclaim",
                            "message": "Worker lease expired before completion.",
                            "retriable": True,
                        },
                    )
                    expired.append(job_id)
        return expired

    async def commit_generation(
        self,
        fence: DispatchFence,
        *,
        generation_id: str,
        manifest_bytes: bytes,
        state: str,
        exit_code: Optional[int],
        artifacts: dict[str, Any],
        run_summary: dict[str, Any],
        error: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        from transformation_portal.orchestrator.artifact_store.generation import validate_manifest

        validate_manifest(manifest_bytes, fence=fence, generation_id=generation_id)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        async with self._session() as session:
            async with session.begin():
                capacities = await self._capacities(session, fence.locator.tenant_id)
                attempt, now = await self._locked_fence(session, fence)
                session.add(
                    CommittedGenerationModel(
                        generation_id=generation_id,
                        job_id=attempt.job_id,
                        tenant_id=attempt.tenant_id,
                        manifest_digest=digest,
                        manifest_bytes=manifest_bytes,
                        created_at=now,
                    )
                )
                attempt.generation_id, attempt.manifest_digest = generation_id, digest
                self._record(
                    session,
                    attempt,
                    "generation_committed",
                    now,
                    {"generation_id": generation_id, "manifest_digest": digest, "lease_epoch": fence.lease_epoch},
                )
                return await self._terminal(
                    session,
                    attempt,
                    capacities,
                    state=state,
                    now=now,
                    exit_code=exit_code,
                    error=error,
                    artifacts=artifacts,
                    run_summary=run_summary,
                    requires_live_fence=True,
                )

    async def committed_manifest(self, job_id: str, tenant_id: str) -> Optional[dict[str, Any]]:
        async with self._session() as session:
            row = (
                await session.execute(
                    select(CommittedGenerationModel.manifest_bytes, CommittedGenerationModel.manifest_digest)
                    .join(DispatchAttemptModel, DispatchAttemptModel.generation_id == CommittedGenerationModel.generation_id)
                    .where(DispatchAttemptModel.job_id == job_id, DispatchAttemptModel.tenant_id == tenant_id)
                )
            ).one_or_none()
            if row is None:
                return None
            if hashlib.sha256(row.manifest_bytes).hexdigest() != row.manifest_digest:
                raise RepositoryError("committed generation manifest integrity failure")
            return json.loads(row.manifest_bytes)

    async def pending_outbox(self, *, limit: int = 100) -> list[tuple[int, str, str, dict[str, Any]]]:
        async with self._session() as session:
            rows = (
                (
                    await session.execute(
                        select(OperationalOutboxModel)
                        .where(OperationalOutboxModel.delivered_at.is_(None))
                        .order_by(OperationalOutboxModel.id)
                        .limit(min(max(limit, 1), 1000))
                    )
                )
                .scalars()
                .all()
            )
            return [(row.id, row.job_id, row.kind, deepcopy(row.payload)) for row in rows]

    async def acknowledge_outbox(self, outbox_id: int) -> None:
        async with self._session() as session:
            async with session.begin():
                # Delivery intents are expendable after acknowledgement. Their
                # canonical record and committed event remain the evidence.
                await session.execute(delete(OperationalOutboxModel).where(OperationalOutboxModel.id == outbox_id))

    async def queued_dispatches(self, *, limit: int = 100) -> list[DispatchLocator]:
        async with self._session() as session:
            rows = (
                (
                    await session.execute(
                        select(DispatchAttemptModel)
                        .where(DispatchAttemptModel.state == "queued")
                        .order_by(DispatchAttemptModel.admitted_at)
                        .limit(min(max(limit, 1), 1000))
                    )
                )
                .scalars()
                .all()
            )
            return [self._locator(row) for row in rows]

    async def revoke_generation(self, job_id: str, tenant_id: str, *, reason: str) -> None:
        """Revoke reader visibility before retention or explicit physical deletion."""
        async with self._session() as session:
            async with session.begin():
                attempt = await session.scalar(
                    select(DispatchAttemptModel).where(DispatchAttemptModel.job_id == job_id).with_for_update()
                )
                if attempt is None or attempt.tenant_id != tenant_id or attempt.state not in TERMINAL_STATES:
                    raise DispatchAuthorityLost("only terminal tenant generations may be revoked")
                job = await session.scalar(select(JobModel).where(JobModel.id == job_id).with_for_update())
                if job is None:
                    raise RepositoryError("missing generation projection")
                if attempt.generation_id is None:
                    return
                now = await self._now(session)
                await self._allow_projection(session)
                previous = attempt.generation_id
                attempt.generation_id, attempt.manifest_digest, attempt.cleaned_at = None, None, None
                attempt.cleanup_checked_at = None
                artifacts = deepcopy(job.artifacts)
                lifecycle = dict(artifacts.get("lifecycle") or {})
                lifecycle.update(deleted_at=now, deletion_status="pending", deletion_reason=reason)
                artifacts["lifecycle"] = lifecycle
                job.artifacts = artifacts
                self._record(session, attempt, "generation_revoked", now, {"generation_id": previous, "reason": reason})

    async def record_generation_deletion(
        self, job_id: str, tenant_id: str, *, count: Optional[int], reason: str, backend: str
    ) -> dict[str, Any]:
        async with self._session() as session:
            async with session.begin():
                attempt = await session.scalar(
                    select(DispatchAttemptModel).where(DispatchAttemptModel.job_id == job_id).with_for_update()
                )
                if (
                    attempt is None
                    or attempt.tenant_id != tenant_id
                    or attempt.state not in TERMINAL_STATES
                    or attempt.generation_id is not None
                ):
                    raise DispatchAuthorityLost("generation visibility must be revoked before deletion")
                job = await session.scalar(select(JobModel).where(JobModel.id == job_id).with_for_update())
                if job is None:
                    raise RepositoryError("missing generation projection")
                await self._allow_projection(session)
                now = await self._now(session)
                artifacts = deepcopy(job.artifacts)
                lifecycle = dict(artifacts.get("lifecycle") or {})
                lifecycle.update(
                    deleted_at=now,
                    deletion_reason=reason,
                    artifact_store_backend=backend,
                    deletion_status="deleted" if count is not None else "failed",
                )
                if count is not None:
                    lifecycle.update(deleted_count=count, store_deleted_count=count, legacy_deleted_count=0)
                    lifecycle.pop("deletion_error", None)
                else:
                    lifecycle["deletion_error"] = "artifact_deletion_failed"
                artifacts["lifecycle"] = lifecycle
                job.artifacts = artifacts
                event_type = "artifact_deleted" if count is not None else "artifact_deletion_failed"
                payload = {"id": job_id, "reason": reason, "deleted_count": count, "artifact_store_backend": backend}
                self._record(session, attempt, event_type, now, payload)
                event = await append_event_in_session(session, job_id, event_type, payload, now)
                session.add(
                    OperationalOutboxModel(
                        job_id=job_id,
                        kind="event",
                        payload={"event_type": event_type, "seq": event.seq, "payload": payload},
                        created_at=now,
                    )
                )
                return artifacts

    async def generation_cleanup_state(self, job_id: str) -> Optional[dict[str, Any]]:
        async with self._session() as session:
            attempt = await session.get(DispatchAttemptModel, job_id)
            if attempt is None or attempt.state not in TERMINAL_STATES:
                return None
            return {
                "generation_id": attempt.generation_id,
                "output_root": attempt.output_root,
                "requested_output_root": attempt.requested_output_root,
            }

    async def generation_cleanup_candidates(self, *, limit: int = 100) -> list[str]:
        """Reserve a bounded, fair batch of due checks, including old tombstones.

        The cursor advances before filesystem I/O so one persistent failure
        cannot starve other jobs. Successful-cleanup evidence remains separate.
        """
        async with self._session() as session:
            async with session.begin():
                now = await self._now(session)
                attempts = list(
                    (
                        await session.scalars(
                            select(DispatchAttemptModel)
                            .where(
                                text("state IN ('succeeded', 'partial', 'failed', 'canceled', 'worker_lost')"),
                                or_(
                                    DispatchAttemptModel.cleanup_checked_at.is_(None),
                                    DispatchAttemptModel.cleanup_checked_at <= now - GENERATION_CLEANUP_RECHECK_SECONDS,
                                ),
                            )
                            .order_by(DispatchAttemptModel.cleanup_checked_at.asc().nulls_first(), DispatchAttemptModel.job_id)
                            .limit(min(max(limit, 1), 1000))
                            .with_for_update(skip_locked=True)
                        )
                    ).all()
                )
                await self._allow_projection(session)
                for attempt in attempts:
                    attempt.cleanup_checked_at = now
                return [attempt.job_id for attempt in attempts]

    async def mark_generation_cleaned(self, job_id: str, *, expected_generation_id: Optional[str]) -> None:
        async with self._session() as session:
            async with session.begin():
                attempt = await session.scalar(
                    select(DispatchAttemptModel).where(DispatchAttemptModel.job_id == job_id).with_for_update()
                )
                if attempt is None or attempt.state not in TERMINAL_STATES or attempt.generation_id != expected_generation_id:
                    return
                await self._allow_projection(session)
                now = await self._now(session)
                first_cleanup = attempt.cleaned_at is None
                attempt.cleaned_at = now
                attempt.cleanup_checked_at = now
                if first_cleanup:
                    self._record(session, attempt, "staging_cleaned", now, {"generation_id": attempt.generation_id})

    async def validate_cutover(self) -> None:
        """Legacy live jobs must be drained before distributed admission starts."""
        async with self._session() as session:
            legacy = await session.scalar(
                select(func.count())
                .select_from(JobModel)
                .where(
                    JobModel.state.in_({"queued", "running"}),
                    ~select(DispatchAttemptModel.job_id).where(DispatchAttemptModel.job_id == JobModel.id).exists(),
                )
            )
            if legacy:
                raise RepositoryError("legacy active jobs must be drained before locator dispatch cutover")
