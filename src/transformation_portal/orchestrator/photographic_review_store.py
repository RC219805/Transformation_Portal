"""Atomic review projections and append-only, hash-bound mutation records.

Postgres shares the orchestrator engine and must be migrated through 0008.
The memory implementation has the same lifetime as memory-backed jobs. Neither
implementation changes dispatch plans, committed artifacts, or their authority.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from copy import deepcopy
from typing import Any, Callable

from transformation_portal.ingest.canonical_json import canonicalize_json

ReviewState = dict[str, Any]
ReviewChange = Callable[[ReviewState], ReviewState]


class ReviewConflict(ValueError):
    """The caller acted on a stale review projection."""


def empty_state() -> ReviewState:
    return {"state_version": 0, "decisions": [], "groups": [], "revisions": [], "deliveries": []}


def _projection_version(current: ReviewState) -> int:
    version = current.get("state_version")
    if type(version) is not int or version < 0:
        raise ValueError("Review projection version is invalid")
    return version


def _mutation(
    current: ReviewState, expected_version: int, change: ReviewChange, event: dict[str, Any]
) -> tuple[ReviewState, bytes]:
    if type(expected_version) is not int or expected_version != _projection_version(current):
        raise ReviewConflict("Review changed; reload before continuing")
    updated = change(deepcopy(current))
    updated["state_version"] = expected_version + 1
    if set(event) & {"schema", "version", "projection_sha256"}:
        raise ValueError("Review event cannot replace authority fields")
    projection = canonicalize_json(updated)
    raw = canonicalize_json(
        {
            **event,
            "schema": "tp.photographic_review.event.v1",
            "version": expected_version + 1,
            "projection_sha256": hashlib.sha256(projection).hexdigest(),
        }
    )
    if len(raw) > 128 * 1024 or len(projection) > 2 * 1024 * 1024:
        raise ValueError("Review history or projection exceeds its bounded size")
    return updated, raw


class MemoryReviewStore:
    def __init__(self) -> None:
        self.states: dict[tuple[str, str], ReviewState] = {}
        self.events: dict[tuple[str, str], list[bytes]] = {}
        self.owners: dict[str, str] = {}
        self.lock = asyncio.Lock()

    async def get(self, job_id: str, tenant_id: str) -> ReviewState:
        if self.owners.get(job_id, tenant_id) != tenant_id:
            raise ValueError("Review tenant mismatch")
        return deepcopy(self.states.get((job_id, tenant_id), empty_state()))

    async def forget(self, job_id: str, tenant_id: str) -> None:
        """Drop memory-only projections after the owning job's retention/deletion."""
        async with self.lock:
            if self.owners.get(job_id, tenant_id) != tenant_id:
                raise ValueError("Review tenant mismatch")
            self.states.pop((job_id, tenant_id), None)
            self.events.pop((job_id, tenant_id), None)
            self.owners.pop(job_id, None)

    async def mutate(
        self, job_id: str, tenant_id: str, expected_version: int, change: ReviewChange, event: dict[str, Any]
    ) -> ReviewState:
        event = deepcopy(event)
        async with self.lock:
            if self.owners.get(job_id, tenant_id) != tenant_id:
                raise ValueError("Review tenant mismatch")
            key = job_id, tenant_id
            updated, raw = _mutation(self.states.get(key, empty_state()), expected_version, change, event)
            self.owners[job_id] = tenant_id
            self.states[key] = deepcopy(updated)
            self.events.setdefault(key, []).append(raw)
            return deepcopy(updated)


class PostgresReviewStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    async def get(self, job_id: str, tenant_id: str) -> ReviewState:
        from transformation_portal.orchestrator.models import PhotographicReviewStateModel
        from transformation_portal.orchestrator.storage.postgres import _SharedEngine

        _, factory = await _SharedEngine.get(self.database_url)
        async with factory() as session:
            row = await session.get(PhotographicReviewStateModel, job_id)
            if row is None:
                return empty_state()
            if row.tenant_id != tenant_id:
                raise ValueError("Review tenant mismatch")
            if row.version != _projection_version(row.payload):
                raise ValueError("Review projection version mismatch")
            return deepcopy(row.payload)

    async def mutate(
        self, job_id: str, tenant_id: str, expected_version: int, change: ReviewChange, event: dict[str, Any]
    ) -> ReviewState:
        event = deepcopy(event)
        from sqlalchemy import select
        from sqlalchemy.dialects.postgresql import insert

        from transformation_portal.orchestrator.models import PhotographicReviewEventModel, PhotographicReviewStateModel
        from transformation_portal.orchestrator.storage.postgres import _SharedEngine

        _, factory = await _SharedEngine.get(self.database_url)
        async with factory() as session, session.begin():
            await session.execute(
                insert(PhotographicReviewStateModel)
                .values(job_id=job_id, tenant_id=tenant_id, version=0, payload=empty_state())
                .on_conflict_do_nothing()
            )
            row = await session.scalar(
                select(PhotographicReviewStateModel).where(PhotographicReviewStateModel.job_id == job_id).with_for_update()
            )
            if row is None or row.tenant_id != tenant_id:
                raise ValueError("Review tenant mismatch")
            if row.version != _projection_version(row.payload):
                raise ValueError("Review projection version mismatch")
            updated, raw = _mutation(row.payload, expected_version, change, event)
            row.payload, row.version = updated, updated["state_version"]
            session.add(
                PhotographicReviewEventModel(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    version=row.version,
                    canonical_bytes=raw,
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
            return deepcopy(updated)


_memory = MemoryReviewStore()


def get_review_store() -> MemoryReviewStore | PostgresReviewStore:
    backend = os.environ.get("TP_ORCHESTRATOR_STATE_BACKEND", "memory").strip().lower() or "memory"
    if backend == "memory":
        return _memory
    if backend != "postgres" or not os.environ.get("TP_DATABASE_URL", "").strip():
        raise RuntimeError("Photographic review requires the configured orchestrator state backend")
    return PostgresReviewStore(os.environ["TP_DATABASE_URL"].strip())
