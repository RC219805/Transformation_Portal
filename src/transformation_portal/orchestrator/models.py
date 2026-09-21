"""SQLAlchemy ORM models for the durable orchestrator state backend.

Phase 1.B - mirrors the persistent slice of ``app.py:Job`` as Postgres
tables. The wire shape produced by ``app.py:_serialize_job`` is unchanged;
these models are an internal persistence representation that
``PostgresJobRepository`` projects to / from ``JobRecord``.

Schema notes:
- ``jobs``: one row per orchestrator job. ``version`` is incremented on
  every update for optimistic concurrency.
- ``job_events``: bounded retained replay with per-job monotonic ``seq``.
- ``job_artifacts``: keyed by ``(job_id, artifact_path)`` to match the
  legacy ``Job.artifact_lookup`` semantic.
- ``operational_audit_events``: append-only pilot control-plane audit log.

All complex fields (``request``, ``effective_request``, ``run_summary``,
``error``, the artifact item dict) use JSONB so they can be queried in
later phases. ``logs_tail`` is stored as JSONB rather than a separate
``job_logs`` table because Phase 1 keeps the legacy "bounded in-memory
tail" semantic; a full log table is a Phase 2/6 concern.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    PrimaryKeyConstraint,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for orchestrator ORM models."""


class JobModel(Base):
    """Persistent slice of an orchestrator job."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    finished_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    done_published_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_event_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    request: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    effective_request: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    logs_tail: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    artifacts: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    run_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    artifact_index: Mapped[list["JobArtifactModel"]] = relationship(
        "JobArtifactModel",
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    # NOTE: ``job_events`` is intentionally *not* a relationship on this
    # model. The event store contract (see ``JobEventStore.append``) allows
    # appending events for arbitrary job_ids - including ids that have no
    # corresponding ``jobs`` row - and the memory backend honors that. A
    # foreign-key relationship here would FK-violate at the SQL level and
    # break behavior parity. ``PostgresJobRepository.delete`` and
    # ``.cleanup_expired`` cascade event deletion at the application layer
    # instead.


class JobArtifactModel(Base):
    """One row per artifact-lookup entry; ``path`` matches the legacy key."""

    __tablename__ = "job_artifacts"
    __table_args__ = (
        PrimaryKeyConstraint("job_id", "path", name="pk_job_artifacts"),
        Index("ix_job_artifacts_job_id", "job_id"),
    )

    job_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    absolute_path: Mapped[str] = mapped_column(Text, nullable=False)

    job: Mapped[JobModel] = relationship("JobModel", back_populates="artifact_index")


class JobEventModel(Base):
    """Retained SSE event history with per-job monotonic seq.

    ``job_id`` deliberately has no SQL-level foreign key to ``jobs.id``
    because the ``JobEventStore`` contract allows appending events for
    arbitrary job ids (the memory backend never enforces a parent).
    Cascade-on-job-delete is handled in ``PostgresJobRepository.delete``
    and ``.cleanup_expired``.
    """

    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_job_id_seq", "job_id", "seq", unique=True),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)


class OperationalAuditEventModel(Base):
    """Append-only pilot control-plane audit event.

    ``job_id`` deliberately has no SQL-level foreign key so job retention and
    artifact deletion cannot erase the operational audit history.
    """

    __tablename__ = "operational_audit_events"
    __table_args__ = (
        Index("ix_operational_audit_events_created_at", "created_at"),
        Index("ix_operational_audit_events_tenant_created", "tenant_id", "created_at"),
        Index("ix_operational_audit_events_job_id", "job_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    actor: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    request_context: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


__all__ = ["Base", "JobArtifactModel", "JobEventModel", "JobModel", "OperationalAuditEventModel"]


class AdmissionCapacityModel(Base):
    """Locked capacity counters; global is always locked before tenant."""

    __tablename__ = "admission_capacity"
    scope: Mapped[str] = mapped_column(String(160), primary_key=True)
    limit: Mapped[int] = mapped_column(Integer, nullable=False)
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    __table_args__ = (CheckConstraint('"limit" > 0 AND active >= 0 AND active <= "limit"'),)


class DispatchPlanModel(Base):
    """Immutable canonical bytes addressed by their exact SHA-256 digest."""

    __tablename__ = "dispatch_plans"
    digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    __table_args__ = (CheckConstraint("octet_length(canonical_bytes) <= 1048576"),)


class DispatchAttemptModel(Base):
    """One admitted attempt and once-only dispatch; retained as a tombstone."""

    __tablename__ = "dispatch_attempts"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    dispatch_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    plan_digest: Mapped[str] = mapped_column(ForeignKey("dispatch_plans.digest"), nullable=False)
    execution_bindings: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    execution_bindings_digest: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    api_version: Mapped[str] = mapped_column(String(16), nullable=False)
    output_root: Mapped[str] = mapped_column(Text, nullable=False)
    requested_output_root: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    holder: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    lease_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    lease_valid_until: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    generation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    manifest_digest: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cleaned_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cleanup_checked_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    admitted_at: Mapped[float] = mapped_column(Float, nullable=False)
    finished_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    __table_args__ = (
        CheckConstraint("lease_epoch >= 0"),
        CheckConstraint(
            "(execution_bindings IS NULL AND execution_bindings_digest IS NULL) OR "
            "(execution_bindings IS NOT NULL AND execution_bindings_digest IS NOT NULL AND "
            "octet_length(execution_bindings) BETWEEN 1 AND 65536 AND "
            "execution_bindings_digest ~ '^[0-9a-f]{64}$')",
            name="ck_dispatch_execution_bindings",
        ),
        Index(
            "ix_dispatch_terminal_cleanup_due",
            cleanup_checked_at.asc().nulls_first(),
            job_id,
            postgresql_where=state.in_(("succeeded", "partial", "failed", "canceled", "worker_lost")),
        ),
    )


class OperationalRecordModel(Base):
    """Append-only canonical evidence, distinct from mutable API projections."""

    __tablename__ = "operational_records"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    canonical_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    digest: Mapped[str] = mapped_column(String(64), nullable=False)


class OperationalOutboxModel(Base):
    """Transactionally committed dispatch and event delivery intents."""

    __tablename__ = "operational_outbox"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    delivered_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)


class CommittedGenerationModel(Base):
    """Immutable manifests; visibility is solely the dispatch pointer."""

    __tablename__ = "committed_generations"
    generation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)


class JobEventSequenceModel(Base):
    """Per-job counter survives replay pruning; orphan event ids remain supported."""

    __tablename__ = "job_event_sequences"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    __table_args__ = (CheckConstraint("last_seq >= 0", name="ck_job_event_sequence_nonnegative"),)
