"""Durable upload correlation; filesystem locks retain staging ownership.

This ledger promises API-process restart recovery on an intact filesystem.
It does not establish storage synchronization or host/power-loss durability.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.models import UploadJobReferenceModel, UploadReceiptModel
from transformation_portal.orchestrator.storage.base import JobRecord, RepositoryError
from transformation_portal.orchestrator.storage.postgres import PostgresOperationalAuditStore


class UploadReceiptConflict(RepositoryError):
    """An upload request ID has already been reserved and cannot be replayed."""


class UploadOwnershipLost(RepositoryError):
    """A completion or recovery no longer owns the ledger generation."""


def validate_upload_request_id(value: str) -> str:
    """Require one canonical UUID, bounded before any filesystem work."""
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Upload request ID must be a canonical UUID") from error
    if str(parsed) != value:
        raise ValueError("Upload request ID must be a canonical UUID")
    return value


def upload_owner(tenant_id: str, actor: Mapping[str, str]) -> str:
    values = [actor.get(name) for name in ("accessEmail", "username")]
    if not tenant_id or any(not isinstance(value, str) or not value.strip() or len(value) > 320 for value in values):
        raise ValueError("Upload receipts require a verified tenant and composite owner")
    return hashlib.sha256(
        canonicalize_json(
            {
                "schema": "tp.upload_staging.owner.v1",
                "tenant_id": tenant_id,
                "access_email": str(values[0]).strip().lower(),
                "username": str(values[1]).strip().lower(),
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class UploadReservation:
    batch_id: str
    tenant_id: str
    owner_sha256: str
    request_id: str
    generation: str
    batch_root: str
    state: str
    created_at: float
    expires_at: float
    receipt_sha256: str | None = None
    response_data: dict[str, Any] | None = None
    cleanup_checked_at: float | None = None

    @classmethod
    def from_model(cls, row: UploadReceiptModel) -> UploadReservation:
        return cls(**{name: getattr(row, name) for name in cls.__dataclass_fields__})


class PostgresUploadReceiptStore(PostgresOperationalAuditStore):
    """Shares the governed Postgres engine; no memory durability fallback."""

    async def reserve(self, reservation: UploadReservation) -> None:
        async with self._session() as session:
            async with session.begin():
                result = await session.scalar(
                    insert(UploadReceiptModel)
                    .values(**{name: getattr(reservation, name) for name in reservation.__dataclass_fields__})
                    .on_conflict_do_nothing()
                    .returning(UploadReceiptModel.batch_id)
                )
                if result is None:
                    raise UploadReceiptConflict("upload request ID is already reserved")

    async def get(self, tenant_id: str, owner_sha256: str, request_id: str) -> UploadReservation | None:
        async with self._session() as session:
            row = await session.scalar(
                select(UploadReceiptModel).where(
                    UploadReceiptModel.tenant_id == tenant_id,
                    UploadReceiptModel.owner_sha256 == owner_sha256,
                    UploadReceiptModel.request_id == request_id,
                )
            )
            return UploadReservation.from_model(row) if row is not None else None

    async def complete(self, expected: UploadReservation, receipt_sha256: str, data: dict[str, Any]) -> None:
        async with self._session() as session:
            async with session.begin():
                row = await session.scalar(
                    select(UploadReceiptModel).where(UploadReceiptModel.batch_id == expected.batch_id).with_for_update()
                )
                if row is None or row.generation != expected.generation or row.state not in {"staging", "ready"}:
                    raise UploadOwnershipLost("staged upload ownership was revoked")
                if row.state == "ready" and (row.receipt_sha256 != receipt_sha256 or row.response_data != data):
                    raise UploadOwnershipLost("sealed upload differs from its recorded completion")
                row.state, row.receipt_sha256, row.response_data = "ready", receipt_sha256, data

    async def cleanup_candidates(self, now: float, limit: int = 100) -> list[UploadReservation]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Upload cleanup selection requires a limit from 1 to 100")
        # Mark checks before trying filesystem locks so a large cohort of live
        # or failed batches cannot starve later expired uploads indefinitely.
        async with self._session() as session:
            async with session.begin():
                rows = (
                    await session.scalars(
                        select(UploadReceiptModel)
                        .where(
                            UploadReceiptModel.expires_at <= now,
                            UploadReceiptModel.state != "expired",
                            ~select(UploadJobReferenceModel.batch_id)
                            .where(UploadJobReferenceModel.batch_id == UploadReceiptModel.batch_id)
                            .exists(),
                        )
                        .order_by(
                            UploadReceiptModel.cleanup_checked_at.asc().nulls_first(),
                            UploadReceiptModel.expires_at,
                            UploadReceiptModel.batch_id,
                        )
                        .limit(limit)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
                for row in rows:
                    row.cleanup_checked_at = now
                return [UploadReservation.from_model(row) for row in rows]

    async def claim_deletion(self, expected: UploadReservation, now: float, retained: set[str]) -> UploadReservation | None:
        """Caller holds the stable filesystem lock before entering this transaction."""
        async with self._session() as session:
            async with session.begin():
                row = await session.scalar(
                    select(UploadReceiptModel).where(UploadReceiptModel.batch_id == expected.batch_id).with_for_update()
                )
                if row is None or row.generation != expected.generation or row.state == "expired" or row.expires_at > now:
                    return None
                if str(Path(row.batch_root, "input").resolve()) in retained:
                    return None
                reference = await session.scalar(
                    select(UploadJobReferenceModel.job_id).where(UploadJobReferenceModel.batch_id == row.batch_id).limit(1)
                )
                if reference is not None:
                    return None
                if row.state != "deleting":
                    row.state, row.generation = "deleting", uuid.uuid4().hex
                return UploadReservation.from_model(row)

    async def finish_deletion(self, expected: UploadReservation) -> None:
        async with self._session() as session:
            async with session.begin():
                row = await session.scalar(
                    select(UploadReceiptModel).where(UploadReceiptModel.batch_id == expected.batch_id).with_for_update()
                )
                if row is None or row.generation != expected.generation or row.state != "deleting":
                    raise UploadOwnershipLost("upload deletion generation changed")
                row.state, row.response_data = "expired", None


async def bind_upload_references(session: AsyncSession, record: JobRecord, tenant_id: str, now: float | None = None) -> None:
    """Called only for NEW jobs inside admission, including keyless requests.

    A replay returns before this helper. Row locks serialize new references
    against expiry; foreign keys retain references for the job's lifetime.
    """
    args = record.effective_request.get("args", {})
    if not isinstance(args, dict):
        return
    raw = args.get("input_dir") or args.get("archive_root") or args.get("inputDir")
    if not isinstance(raw, str):
        return
    root = Path(os.getenv("TP_PORTAL_UPLOAD_ROOT", str(Path(tempfile.gettempdir()) / "transformation-portal/uploads")))
    try:
        parts = Path(raw).resolve().relative_to(root.resolve()).parts
    except ValueError:
        return
    if len(parts) < 2 or not parts[0].startswith("upload_") or parts[1] != "input":
        return
    row = await session.scalar(select(UploadReceiptModel).where(UploadReceiptModel.batch_id == parts[0]).with_for_update())
    if row is None:  # Pre-ledger staged batches retain their existing contracts.
        return
    if now is None:
        clock = await session.scalar(select(func.extract("epoch", func.clock_timestamp())))
        if clock is None:
            raise UploadOwnershipLost("upload admission clock is unavailable")
        now = float(clock)
    if row.tenant_id != tenant_id or row.state != "ready" or row.expires_at <= now:
        raise UploadOwnershipLost("staged input is expired, unavailable, or outside this tenant")
    session.add(UploadJobReferenceModel(batch_id=row.batch_id, job_id=record.id))
