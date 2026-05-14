"""Typed error envelope payload for orchestrator v1.

Mirrors the wire shape of ``app.py:_error_obj``:

    {"code": code, "message": message, "details": details or {}}

Note that ``details`` is never ``None`` on the wire — the helper coerces a
missing or ``None`` argument to ``{}`` via ``details or {}``. ``ErrorObject``
reproduces that behavior by defaulting ``details`` to an empty dict AND by
coercing an explicit ``None`` to ``{}`` via a ``mode="before"`` validator,
so callers that pass an ``Optional[dict]`` (e.g. routes that reuse a helper
returning ``None`` when there are no structured details) don't trip a
ValidationError that would turn an intended 4xx into a 500.

Phase 2.D — additive ``retriable`` field. Callers that opt in
(``retriable=True`` for broker-level ``worker_lost_*`` payloads;
``retriable=False`` for executor-level ``RUNNER_*`` payloads) get the
key on the wire:

    {"code": code, "message": message, "details": details or {}, "retriable": bool}

Callers that omit the argument (most HTTP-level 4xx envelopes) get the
pre-Phase-2.D shape with no ``retriable`` key. The omission is enforced
by ``ErrorObject._drop_unset_retriable``, a wrap ``@model_serializer``
that pops ``retriable`` when ``None`` — Pydantic v2's default JSON
output would otherwise emit ``retriable: null`` for the ``None`` default
and break wire compat with ``_error_obj``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer

ErrorCode = Literal[
    # Top-level envelope error codes emitted by orchestrator HTTP surfaces.
    # Most are HTTP-status-derived (see HTTP_STATUS_ERROR_CODES in app.py),
    # while some — such as AUTH_CONFIGURATION_ERROR — are emitted directly
    # by middleware on a 503 response when the auth env-var contract is
    # incomplete.
    "AUTH_CONFIGURATION_ERROR",
    "ARTIFACT_DELETED",
    "ARTIFACT_STORE_UNAVAILABLE",
    "CONFLICT",
    "FORBIDDEN",
    "HTTP_ERROR",
    "INTERNAL_ERROR",
    "INVALID_ARGUMENT",
    "METHOD_NOT_ALLOWED",
    "NOT_FOUND",
    "QUEUE_UNAVAILABLE",
    "RATE_LIMITED",
    "REQUEST_TOO_LARGE",
    "SERVICE_UNAVAILABLE",
    "UNAUTHORIZED",
    # Job-runner failure codes (set on job.error by app.py's runner exception
    # handlers; surface inside the data payload of a job_status envelope, not
    # the outer envelope's error field)
    "RUNNER_ERROR",
    "RUNNER_EXIT_NONZERO",
    "RUNNER_NOT_FOUND",
    "RUNNER_PARTIAL_FAILURE",
    # Broker-level failure codes (Phase 2.D). Distinct from the executor
    # RUNNER_* codes so callers and operator tooling can branch on the
    # ``retriable=True`` flag.
    "worker_lost_on_restart",
    "worker_lost_via_lease_reclaim",
]
"""Closed set of error codes emitted by orchestrator routes.

Verified against ``app.py`` by extracting every ``code=...`` and
``_error_obj("CODE", ...)`` literal plus every value in
``HTTP_STATUS_ERROR_CODES``.
"""


class ErrorObject(BaseModel):
    """The ``error`` field inside an error envelope."""

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    # Phase 2.D — optional retry classification. ``True`` for broker-level
    # failures whose underlying work is intact (``worker_lost_*``); ``False``
    # for executor-level failures (``RUNNER_EXIT_NONZERO`` / ``RUNNER_ERROR``
    # / ``RUNNER_NOT_FOUND``); ``None`` (omitted on the wire by the wrap
    # ``@model_serializer`` below) for HTTP-level error envelopes that have
    # not been opted into the classification. Operator tooling and future
    # auto-retry policy branch on this field.
    retriable: bool | None = None

    @field_validator("details", mode="before")
    @classmethod
    def _coerce_none_details_to_empty(cls, value: Any) -> Any:
        """Match ``app.py:_error_obj``'s ``details or {}`` coercion.

        Callers that pass an Optional[dict] (e.g. forwarding a helper return
        value that's None when there are no structured details) shouldn't
        hit a ValidationError. Pre-validate by collapsing ``None`` to ``{}``.
        """
        if value is None:
            return {}
        return value

    @model_serializer(mode="wrap")
    def _drop_unset_retriable(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit ``retriable`` from the wire when not opted in.

        Phase 2.D additive contract: callers that set ``retriable`` to
        ``True``/``False`` (executor failure paths, worker_lost emit)
        embed the field on the wire; callers that omit the argument
        (HTTP-level error envelopes) get the pre-Phase-2.D byte-identical
        shape with no ``retriable`` key. Implemented as a wrap-serializer
        because Pydantic v2's default JSON output still emits
        ``retriable: null`` for the ``None`` default, which would break
        wire compat with ``app.py:_error_obj``.
        """
        result = handler(self)
        if self.retriable is None:
            result.pop("retriable", None)
        return result
