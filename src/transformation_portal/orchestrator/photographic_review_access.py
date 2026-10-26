"""Injected authenticated IO ports for photographic review.

Only the app adapter may construct source references. Browser requests select
their public artifact_path and SHA, never physical paths or storage keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Request
from starlette.responses import Response


class ReviewAccessError(ValueError):
    def __init__(self, message: str, *, status: int = 409, code: str = "ARTIFACT_CHANGED"):
        super().__init__(message)
        self.status, self.code = status, code


class ReviewInputError(ValueError):
    """A deliberately public validation diagnostic from the review boundary."""


@dataclass(frozen=True)
class ReviewContext:
    job_id: str
    tenant_id: str
    actor: dict[str, str]
    references: list[dict[str, Any]]
    materialize: Callable[[dict[str, Any], Path, int], Awaitable[dict[str, Any]]]
    publish: Callable[[str, Path, str], Awaitable[None]]
    discard: Callable[[list[str]], Awaitable[None]] | None = None
    revalidate: Callable[[], Awaitable[None]] | None = None


@dataclass(frozen=True)
class ReviewRuntime:
    authorize: Callable[[Request, str], Awaitable[ReviewContext | Response]]
    error_response: Callable[..., Response]
    authorize_history: Callable[[Request, str], Awaitable[ReviewContext | Response]] | None = None
