"""Typed error envelope payload for orchestrator v1.

Mirrors the wire shape of ``app.py:_error_obj``:

    {"code": code, "message": message, "details": details or {}}

Note that ``details`` is never ``None`` on the wire — the helper coerces a
missing or ``None`` argument to ``{}``. ``ErrorObject`` reproduces that
behavior by defaulting ``details`` to an empty dict, not ``None``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ErrorCode = Literal[
    "AUTH_CONFIGURATION_ERROR",
    "FORBIDDEN",
    "HTTP_ERROR",
    "INTERNAL_ERROR",
    "INVALID_ARGUMENT",
    "NOT_FOUND",
    "RATE_LIMITED",
    "REQUEST_TOO_LARGE",
    "SERVICE_UNAVAILABLE",
    "UNAUTHORIZED",
]
"""Closed set of error codes emitted by orchestrator routes."""


class ErrorObject(BaseModel):
    """The ``error`` field inside an error envelope."""

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
