from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class HardeningError(RuntimeError):
    """Base exception for hardening-layer failures."""


@dataclass(frozen=True)
class InputValidationError(HardeningError):
    message: str
    path: Optional[str] = None
    details: Optional[dict[str, Any]] = None

    def __str__(self) -> str:
        p = f" path={self.path!r}" if self.path else ""
        return f"{self.message}{p}"


@dataclass(frozen=True)
class PolicyViolationError(HardeningError):
    message: str
    rule: Optional[str] = None
    details: Optional[dict[str, Any]] = None

    def __str__(self) -> str:
        r = f" rule={self.rule!r}" if self.rule else ""
        return f"{self.message}{r}"
