"""SQLAlchemy ORM models for the durable orchestrator state backend.

Phase 1.B - mirrors the persistent slice of ``app.py:Job`` as Postgres
tables. The wire shape produced by ``app.py:_serialize_job`` is unchanged;
these models are an internal persistence representation that
``PostgresJobRepository`` projects to / from ``JobRecord``.

Schema notes:
- ``jobs``: one row per orchestrator job. ``version`` is incremented on
  every update for optimistic concurrency.
- ``job_events``: append-only, per-job monotonic ``seq`` for SSE replay.
- ``job_artifacts``: keyed by ``(job_id, artifact_path)`` to match the
  legacy ``Job.artifact_lookup`` semantic.

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
    Float,
    ForeignKey,
    Index,
    Integer,
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
    events: Mapped[list["JobEventModel"]] = relationship(
        "JobEventModel",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobEventModel.seq",
        lazy="select",
    )


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
    """Append-only SSE event history with per-job monotonic seq."""

    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_job_id_seq", "job_id", "seq", unique=True),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)

    job: Mapped[JobModel] = relationship("JobModel", back_populates="events")


__all__ = ["Base", "JobArtifactModel", "JobEventModel", "JobModel"]
