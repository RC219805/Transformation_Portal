"""Typed response models for the orchestrator's health/readiness routes.

Three routes are wired in PR B of the Phase 1.2 sequence:

- ``GET /healthz``  — minimal liveness probe; raw JSON, NOT enveloped.
- ``GET /ready``    — verbose-toggleable readiness; raw JSON, NOT enveloped.
- ``GET /v1/readiness`` — pipeline readiness; enveloped under
  ``tp.orchestrator.readiness.v1`` via ``ApiEnvelope[ReadinessData]``.

The first two intentionally bypass the orchestrator envelope because external
load balancers and Kubernetes probes expect a raw ``{"ok": ...}`` shape (see
``tests/test_app_orchestrator_contract_http.py::test_ready_keeps_non_enveloped_shape``).
The contract test asserts neither ``schema`` nor ``success`` keys are present.

Internal nested fields (``cli``, ``jobs``, ``security``, per-pipeline data) are
modeled as ``dict[str, Any]`` rather than fully-typed nested classes because
their internals churn (especially ``security`` which exposes new feature flags
as they're rolled out) and the typing-vs-maintenance trade-off favors keeping
those dicts open.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from transformation_portal.api.v1.envelopes import ApiEnvelope


class HealthzResponse(BaseModel):
    """Response shape for ``GET /healthz`` (raw, not enveloped).

    Mirrors ``app.py``'s explicit ``JSONResponse({"ok": True, "time": _now()})``.
    Closed schema: any extra keys would surprise the load balancers / probes
    that consume this endpoint, so ``extra="forbid"`` is correct.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    time: float


class ReadyResponse(BaseModel):
    """Response shape for ``GET /ready`` (raw, not enveloped).

    Always-present fields: ``ok``, ``time``, ``version``, ``artifact_store``.
    Verbose-only fields (when ``TP_READY_VERBOSE=true``): ``cli``, ``jobs``,
    ``security``. Each nested block is churning internal-state telemetry; we
    model them as ``dict[str, Any]`` rather than nested classes so adding a
    new feature flag to the security dict doesn't require a model bump.

    ``extra="allow"`` lets unrecognised top-level fields pass through unchanged
    instead of failing FastAPI response validation. (The handler returns
    ``Dict[str, Any]`` directly — unlike ``/healthz`` which returns
    ``JSONResponse`` — so response_model IS enforced at runtime here.)
    """

    model_config = ConfigDict(extra="allow")

    ok: bool
    time: float
    version: str
    artifact_store: dict[str, Any] | None = None
    cli: dict[str, Any] | None = None
    jobs: dict[str, Any] | None = None
    security: dict[str, Any] | None = None


class ReadinessServer(BaseModel):
    """The ``server`` block inside a readiness envelope's ``data`` payload."""

    model_config = ConfigDict(extra="forbid")

    time: float
    version: str
    auth_mode: str
    backend_live: bool


class ReadinessData(BaseModel):
    """Payload of the readiness envelope (``ApiEnvelope[ReadinessData]``).

    ``pipelines`` is a mapping from pipeline name (e.g. ``"lux-depth-v3"``,
    ``"archive-gate-a"``) to a per-pipeline readiness dict. Per-pipeline
    contents come from ``app.py:_evaluate_pipeline_readiness`` and vary by
    pipeline; modeling each variant would couple this file to deep
    pipeline-specific logic, so we keep it ``dict[str, Any]``.
    """

    model_config = ConfigDict(extra="forbid")

    server: ReadinessServer
    pipelines: dict[str, dict[str, Any]]


ReadinessEnvelope = ApiEnvelope[ReadinessData]
"""Convenience alias for the typed envelope wrapping ``ReadinessData``.

Use this as the ``response_model`` on the ``/v1/readiness`` route so FastAPI's
OpenAPI generation picks up both the envelope shape and the readiness payload.
"""

__all__ = [
    "HealthzResponse",
    "ReadinessData",
    "ReadinessEnvelope",
    "ReadinessServer",
    "ReadyResponse",
]
