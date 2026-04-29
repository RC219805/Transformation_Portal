"""Typed request/response models for the orchestrator's telemetry routes.

Two routes get ``response_model=`` annotations in PR E of the Phase 1.2
sequence:

- ``POST /v1/portal/events``  → ``ApiEnvelope[PortalEventData]``
  (``tp.orchestrator.portal_event.v1``)
- ``POST /v1/portal/rum``     → ``ApiEnvelope[PortalRumIngestData]``
  (``tp.orchestrator.portal_rum_ingest.v1``)

Every route handler returns ``JSONResponse(_api_envelope(...))`` directly,
so ``response_model`` is **OpenAPI-only** — no runtime serialisation by
FastAPI. Wire shapes are unchanged by this PR; the models exist to type
the OpenAPI schema and provide a stable surface for typed callers.

The ``event`` field on both payloads is a sanitised record dict whose
internal keys churn as new event types and RUM metrics are added. Keeping
it as ``dict[str, Any]`` rather than a typed sub-class is intentional —
the record structure is validated upstream by ``_record_portal_event`` and
``_record_portal_rum``; fully typing it here would chain this module to
those internals and force a model bump on every new event or metric kind.

Request bodies are accepted as ``Dict[str, Any]`` by the existing handlers
and are **not yet wired** as Pydantic parameters here — same reasoning as
``JobCreateRequest`` (Phase 1.2 PR C). Wiring them would shift FastAPI's
422 to the orchestrator's 400 envelope but would also drop specific
error-reason codes produced by the validation helpers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from transformation_portal.api.v1.envelopes import ApiEnvelope


class PortalEventData(BaseModel):
    """Payload for ``tp.orchestrator.portal_event.v1``.

    Mirrors the ``data`` dict returned by ``portal_events`` in ``app.py``:
    ``{"accepted": True, "event": record}``.

    ``event`` holds the sanitised record written by ``_record_portal_event``
    (schema, timestamp, event_type, pipeline, surface, field, metadata,
    reasons). ``extra="allow"`` lets the model absorb new top-level keys
    without a bump if the handler adds them in a later PR.
    """

    model_config = ConfigDict(extra="allow")

    accepted: bool
    event: dict[str, Any] | None = None


class PortalRumIngestData(BaseModel):
    """Payload for ``tp.orchestrator.portal_rum_ingest.v1``.

    Three shapes are possible:

    - RUM disabled: ``{"accepted": False, "disabled": True}``
    - Invalid payload (→ error envelope, not this model)
    - Accepted: ``{"accepted": True, "event": record}``

    ``disabled`` is ``None`` on the accepted path; ``event`` is ``None``
    on the disabled path. Both are Optional so a single model covers all
    non-error shapes. ``extra="allow"`` accommodates future additions.
    """

    model_config = ConfigDict(extra="allow")

    accepted: bool
    disabled: bool | None = None
    event: dict[str, Any] | None = None


PortalEventEnvelope = ApiEnvelope[PortalEventData]
"""Convenience alias for the typed envelope wrapping ``PortalEventData``."""

PortalRumIngestEnvelope = ApiEnvelope[PortalRumIngestData]
"""Convenience alias for the typed envelope wrapping ``PortalRumIngestData``."""

__all__ = [
    "PortalEventData",
    "PortalEventEnvelope",
    "PortalRumIngestData",
    "PortalRumIngestEnvelope",
]
