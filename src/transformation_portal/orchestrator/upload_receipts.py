"""Durable upload correlation; filesystem locks retain staging ownership.

This ledger promises API-process restart recovery on an intact filesystem.
It does not establish storage synchronization or host/power-loss durability.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import and_, func, literal, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from transformation_portal.core.archive_execution_plan import ARCHIVE_OPERATIONS
from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.models import UploadJobReferenceModel, UploadReceiptModel
from transformation_portal.orchestrator.storage.base import JobRecord, RepositoryError
from transformation_portal.orchestrator.storage.postgres import PostgresOperationalAuditStore
from transformation_portal.portal import path_security
from transformation_portal.vlm_captioning.fastvlm_runtime import FASTVLM_CHECKPOINT_DIRS

_ARCHIVE_DEFAULT_COMMANDS = {
    "archive-gate-a": "fixity-scan",
    "archive-gate-b": "bag-build",
    "archive-gate-c": "mets-export",
}
_INPUT_PATH_ALIASES = {
    "input_dir": ("input_dir", "inputDir"),
    "archive_root": ("archive_root", "archiveRoot"),
    "companions_manifest": ("companions_manifest",),
    "materials_manifest": ("materials_manifest",),
    "sam2_checkpoint_path": ("sam2_checkpoint_path", "sam2CheckpointPath"),
    "cameras_sidecar_path": ("cameras_sidecar_path", "camerasSidecarPath"),
    "fastvlm_python_executable": ("fastvlm_python_executable", "fastvlmPythonExecutable"),
    "fastvlm_mlx_vlm_dir": ("fastvlm_mlx_vlm_dir", "fastvlmMlxVlmDir"),
    "vlm_captioning_model": ("vlm_captioning_model", "vlmCaptioningModel"),
}


class UploadReceiptConflict(RepositoryError):
    """An upload request ID has already been reserved and cannot be replayed."""


class UploadOwnershipLost(RepositoryError):
    """A completion or recovery no longer owns the ledger generation."""


class UploadReceiptExpired(UploadOwnershipLost):
    """A current locked receipt crossed its retained upload lifetime."""


async def _upload_now(session: AsyncSession) -> float:
    """Sample authoritative time after acquiring every relevant row lock."""
    clock = await session.scalar(select(func.extract("epoch", func.clock_timestamp())))
    if clock is None:
        raise UploadOwnershipLost("upload clock is unavailable")
    return float(clock)


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
                    .on_conflict_do_nothing(index_elements=["tenant_id", "owner_sha256", "request_id"])
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
                if row.expires_at <= await _upload_now(session):
                    raise UploadReceiptExpired("staged upload lifetime expired")
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
                batch_root = Path(row.batch_root).resolve()
                if any(path.is_relative_to(batch_root) or batch_root.is_relative_to(path) for path in map(Path, retained)):
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


async def bind_upload_references(
    session: AsyncSession,
    record: JobRecord,
    tenant_id: str,
    *,
    plan_bytes: bytes | None = None,
    execution_bindings: bytes | None = None,
    read_paths: tuple[str, ...] = (),
    upload_read_base: Path | None = None,
) -> None:
    """Called only for NEW jobs inside admission, including keyless requests.

    A replay returns before this helper. Row locks serialize new references
    against expiry; foreign keys retain references for the job's lifetime.
    Tracked batches require tenant scope even for trusted/direct callers.
    """
    paths = set(upload_source_paths(record.request, record.effective_request, upload_read_base=upload_read_base))
    paths.update(_resolved_source_paths(list(read_paths), upload_read_base=upload_read_base))
    if plan_bytes is not None:
        paths.update(_admitted_source_paths(plan_bytes, execution_bindings, upload_read_base=upload_read_base))
    if not paths:
        return
    overlaps = []
    for path in sorted(paths):
        value = str(path)
        prefix = value.rstrip("/") + "/"
        # Ledger locators are server-owned absolute paths. Compare complete
        # segments literally, including '%' and '_', without a CWD/env anchor.
        # Expired tombstones cannot block an ancestor forever after deletion,
        # but an explicit read inside one must still fail admission.
        overlaps.extend(
            (
                UploadReceiptModel.batch_root == value,
                func.left(literal(value), func.length(UploadReceiptModel.batch_root) + 1)
                == UploadReceiptModel.batch_root + "/",
                and_(
                    UploadReceiptModel.state != "expired",
                    func.left(UploadReceiptModel.batch_root, len(prefix)) == prefix,
                ),
            )
        )
    # Every admission takes the same lock order even when a request's input
    # and archive selectors refer to different batches or use opposite aliases.
    rows = (
        await session.scalars(
            select(UploadReceiptModel)
            .options(
                load_only(
                    UploadReceiptModel.batch_id,
                    UploadReceiptModel.tenant_id,
                    UploadReceiptModel.state,
                    UploadReceiptModel.expires_at,
                    raiseload=True,
                )
            )
            .where(or_(*overlaps))
            .order_by(UploadReceiptModel.batch_id)
            .with_for_update()
        )
    ).all()
    if not rows:
        return
    now = await _upload_now(session)
    for row in rows:
        if row.tenant_id != tenant_id or row.state != "ready" or row.expires_at <= now:
            raise UploadOwnershipLost("staged input is expired, unavailable, or outside this tenant")
        session.add(UploadJobReferenceModel(batch_id=row.batch_id, job_id=record.id))


def upload_source_paths(*requests: dict[str, Any], upload_read_base: Path | None = None) -> tuple[Path, ...]:
    """Collect read selectors using only an explicit server-owned relative base.

    Absolute reads need no base. Missing or failed relative resolution cannot
    silently authorize an unpinned job or discard retained cleanup protection.
    Command output fields never retain a batch.
    """
    values: list[Any] = []
    for request in requests:
        args = request.get("args") if isinstance(request, dict) else None
        if not isinstance(args, dict):
            continue
        pipeline = request.get("pipeline")
        names: tuple[str, ...]
        if isinstance(pipeline, str) and pipeline in {"lux-depth", "lux-depth-v5", "lux-depth-v6"}:
            names = ("input_dir", "companions_manifest", "materials_manifest")
        elif pipeline == "lux-depth-v3":
            names = tuple(
                name
                for name in _INPUT_PATH_ALIASES
                if name not in {"archive_root", "companions_manifest", "materials_manifest"}
            )
        elif isinstance(pipeline, str) and pipeline in _ARCHIVE_DEFAULT_COMMANDS:
            names = ("input_dir",)
        else:
            names = ("input_dir", "archive_root")
        aliases = {name: _INPUT_PATH_ALIASES[name] for name in names}
        default = _ARCHIVE_DEFAULT_COMMANDS.get(pipeline) if isinstance(pipeline, str) else None
        if default is not None:
            command = args.get("archive_command", args.get("archiveCommand", default)) or default
            operation = ARCHIVE_OPERATIONS.get(command) if isinstance(command, str) else None
            if operation is not None and operation.pipeline == pipeline:
                for name in operation.files + operation.directories:
                    parts = name.split("_")
                    aliases[name] = (name, parts[0] + "".join(part.title() for part in parts[1:]))
        for field, names in aliases.items():
            for name in names:
                value = args.get(name)
                if field != "vlm_captioning_model" or not _caption_role(value):
                    values.append(value)
    return _resolved_source_paths(values, upload_read_base=upload_read_base)


def _caption_role(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in FASTVLM_CHECKPOINT_DIRS


def _resolved_source_paths(values: list[Any], *, upload_read_base: Path | None = None) -> tuple[Path, ...]:
    values = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if not values:
        return ()
    base = None
    if upload_read_base is not None:
        try:
            base = Path(upload_read_base)
            if not base.is_absolute():
                raise ValueError("upload read base must be absolute")
            base = base.resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise UploadOwnershipLost("upload reads require an absolute server-owned base") from error
    paths = set()
    for raw in values:
        if not Path(raw).is_absolute() and base is None:
            raise UploadOwnershipLost("relative upload reads require an explicit server-owned base")
        try:
            resolved = path_security._resolve_untrusted_request_path(raw, repo_root=base)
            # Preserve loop/error detection after shared resolution has produced
            # an absolute path; this never consults a process or module CWD.
            paths.add(resolved.resolve())
        except (OSError, RuntimeError, ValueError) as error:
            raise UploadOwnershipLost("upload read path resolution is unavailable") from error
    return tuple(sorted(paths))


def _admitted_source_paths(
    plan_bytes: bytes, execution_bindings: bytes | None, *, upload_read_base: Path | None = None
) -> tuple[Path, ...]:
    """Read only the already-validated immutable plan and physical bindings."""
    plan = decode_bounded_json_object(plan_bytes)
    values = []
    selection = plan.get("input_selection")
    if isinstance(selection, dict):
        values.append(selection.get("root"))
    for node in plan.get("nodes", []):
        configuration = node.get("configuration", {})
        if plan.get("planned_backend") == "archive":
            operation = ARCHIVE_OPERATIONS[configuration["operation"]]
            values.extend(configuration["parameters"].get(name) for name in operation.files + operation.directories)
        else:
            values.extend(
                configuration.get(name)
                for name in (
                    "raw_python_executable",
                    "da3_python_executable",
                    "depth_pro_python_executable",
                    "depth_pro_checkpoint_path",
                    "sam2_checkpoint_path",
                    "sam_vit_h_checkpoint_path",
                    "cameras_sidecar_path",
                )
            )
            captioning = configuration.get("captioning")
            if isinstance(captioning, dict) and captioning.get("enabled"):
                values.extend(
                    captioning.get(name) for name in ("model_path", "review_model_path", "python_executable", "mlx_vlm_dir")
                )
                if not _caption_role(captioning.get("selector")):
                    values.append(captioning.get("selector"))
    if execution_bindings is not None:
        bindings = decode_bounded_json_object(execution_bindings)
        if bindings.get("schema") == "tp.job.lux_depth.bindings.v1":
            bindings = bindings["photography_bindings"]
        values.extend(
            bindings.get(name) for name in ("input_root", "companion_root", "materials_root", "runtime_python", "raw_python")
        )
    return _resolved_source_paths(values, upload_read_base=upload_read_base)
