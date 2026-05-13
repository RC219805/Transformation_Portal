"""Typed request/response models for the orchestrator's job lifecycle routes.

Eight routes get ``response_model=`` annotations in PR C of the Phase 1.2
sequence (the v1 routes plus their v2 mirrors):

- ``POST /v[12]/jobs``                  → ``ApiEnvelope[JobBriefData]``
  (``tp.orchestrator.job.v1``)
- ``POST /v[12]/jobs/{id}/cancel``      → ``ApiEnvelope[JobBriefData]``
  (``tp.orchestrator.job.v1``)
- ``GET  /v[12]/jobs``                  → ``ApiEnvelope[JobsListData]``
  (``tp.orchestrator.jobs.v1``)
- ``GET  /v[12]/jobs/{id}``             → ``ApiEnvelope[JobStatusData]``
  (``tp.orchestrator.job_status.v1``)

Every route handler returns ``JSONResponse(_api_envelope(...))`` directly, so
``response_model`` is **OpenAPI-only** — no runtime serialization is applied
by FastAPI. The wire shape produced by ``app.py:_serialize_job`` (line 6530)
is therefore unchanged by this PR; the models exist to type the OpenAPI
schema and provide a stable surface for future PRs that want to consume the
job envelopes from typed callers.

``JobCreateRequest`` is defined here for type-discipline / future use but is
**not yet wired** as the handler parameter. Wiring it would shift FastAPI's
default 422 response to the orchestrator's 400 envelope (handled by the
existing ``RequestValidationError`` exception handler at app.py:7879), but
would also drop specific error-reason codes — currently ``_create_job``
returns ``reason="unsupported_pipeline"`` etc., and Pydantic-level
validation collapses everything to ``reason="request_validation_failed"``.
That trade-off deserves its own focused decision/PR.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from transformation_portal.api.v1.envelopes import ApiEnvelope
from transformation_portal.api.v1.errors import ErrorObject

JobState = Literal[
    "queued",
    "running",
    "succeeded",
    "partial",
    "failed",
    "canceled",
    "worker_lost",
]
"""The closed set of values for ``job.state``.

Sourced from the ``state: str = "queued"`` comment in ``app.py:Job`` and the
``"queued|running|succeeded|partial|failed|canceled|worker_lost"`` enumeration
there (``worker_lost`` added in Phase 2.D — the worker died holding the
lease, the job payload is intact and the error envelope carries
``retriable=True``). No intermediate states are emitted on the wire —
cancellation transitions
directly from ``running`` to ``canceled`` (see ``_request_cancel``).
"""


class JobBriefData(BaseModel):
    """Payload for ``tp.orchestrator.job.v1``.

    Used by two routes:

    - ``POST /v[12]/jobs`` — create. Returns ``{id, state, events_url}``.
    - ``POST /v[12]/jobs/{id}/cancel`` — cancel. Returns ``{id, state}``
      (no ``events_url``).

    ``events_url`` is therefore Optional in this shared model.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    state: JobState
    events_url: str | None = None


class JobStatusData(BaseModel):
    """Payload for ``tp.orchestrator.job_status.v1`` and entries in
    ``tp.orchestrator.jobs.v1``.

    Mirrors ``app.py:_serialize_job`` output (line 6530) field-for-field.
    Several fields are Optional because they're only set after a transition:
    ``started_at`` (when the runner starts), ``finished_at`` /
    ``exit_code`` (after the runner exits), ``logs_tail`` (only when the
    handler is called with ``include_logs=True`` — list endpoints set
    ``include_logs=False``).

    ``extra="allow"`` is intentional: ``_serialize_job`` may grow new fields
    in subsequent PRs, and external callers should keep working without a
    model bump.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    pipeline: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    state: JobState
    progress: int
    exit_code: int | None = None
    events_url: str
    artifacts: dict[str, Any] = Field(default_factory=dict)
    error: ErrorObject | None = None
    run_summary: dict[str, Any] | None = None
    last_event_at: float | None = None
    logs_tail: list[str] | None = None


class JobsListData(BaseModel):
    """Payload for ``tp.orchestrator.jobs.v1``.

    Mirrors the dict constructed at ``app.py:_list_jobs`` (line 8534):
    ``{"jobs": [_serialize_job(...), ...], "total": N, "returned": M}``.

    The list endpoint always passes ``include_logs=False`` to ``_serialize_job``,
    so ``JobStatusData.logs_tail`` will always be ``None`` for entries in
    this list. The model accepts both — there's no need for a separate
    ``JobBriefStatusData`` variant.
    """

    model_config = ConfigDict(extra="forbid")

    jobs: list[JobStatusData]
    total: int
    returned: int


class JobCreateRequest(BaseModel):
    """Typed request body for ``POST /v[12]/jobs``.

    NOT YET wired as the handler parameter — see module docstring for why.
    Defined here so future PRs can adopt it once the error-reason-code
    trade-off is accepted.

    ``args`` is the pipeline-specific dispatch payload; its shape varies by
    pipeline and is validated downstream by ``_build_config_preview`` and the
    pipeline-specific dispatch preflights. Keeping it as ``dict[str, Any]``
    rather than a discriminated union is intentional — see app.py for the
    actual per-pipeline validation logic.
    """

    model_config = ConfigDict(extra="allow")

    pipeline: str
    args: dict[str, Any] = Field(default_factory=dict)


JobEnvelope = ApiEnvelope[JobBriefData]
"""Convenience alias for the typed envelope wrapping ``JobBriefData``.

Schema field stays under model-level ``SchemaName`` validation (any of the 12
known schemas); the route handler is responsible for setting the right
schema string. Same pattern as ``ReadinessEnvelope``."""

JobStatusEnvelope = ApiEnvelope[JobStatusData]
"""Convenience alias for the typed envelope wrapping ``JobStatusData``."""

JobsListEnvelope = ApiEnvelope[JobsListData]
"""Convenience alias for the typed envelope wrapping ``JobsListData``."""

__all__ = [
    "JobBriefData",
    "JobCreateRequest",
    "JobEnvelope",
    "JobsListData",
    "JobsListEnvelope",
    "JobState",
    "JobStatusData",
    "JobStatusEnvelope",
]
