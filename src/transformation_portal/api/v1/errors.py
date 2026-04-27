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
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ErrorCode = Literal[
    # HTTP-status-derived codes (see HTTP_STATUS_ERROR_CODES in app.py)
    "AUTH_CONFIGURATION_ERROR",
    "FORBIDDEN",
    "HTTP_ERROR",
    "INTERNAL_ERROR",
    "INVALID_ARGUMENT",
    "METHOD_NOT_ALLOWED",
    "NOT_FOUND",
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
