"""Typed ingest-domain errors and deterministic aggregation helpers.

This module provides a domain-level error taxonomy that separates:
- domain semantics (typed error classes + priority)
- transport semantics (stable integer-compatible exit codes)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Optional


class IngestExitCode(IntEnum):
    """Stable ingest contract exit codes.

    Expansion policy:
    - Numeric values are wire-level contract and must not be renumbered or reused.
    - Add new codes with strictly increasing integer values.
    """

    SUCCESS = 0
    SCHEMA_VALIDATION_FAILED = 1
    BIT_DEPTH_VIOLATION = 2
    GAMMA_VIOLATION = 3
    SCHEMA_DRIFT = 4
    OTHER_FAILURE = 5


PRIORITY_SCHEMA_VALIDATION_FAILURE = 10
PRIORITY_BIT_DEPTH_VIOLATION = 20
PRIORITY_GAMMA_VIOLATION = 30
PRIORITY_SCHEMA_DRIFT = 40
PRIORITY_OTHER_FAILURE = 0
PRIORITY_SUCCESS = -1


@dataclass(eq=False, frozen=True)
class IngestError(Exception):
    """Base typed ingest error with explicit severity priority."""

    message: str
    exit_code: IngestExitCode
    priority: int

    def __post_init__(self) -> None:
        # Keep BaseException.args populated for pickling/tooling compatibility.
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"exit_code={int(self.exit_code)}, "
            f"priority={self.priority}, "
            f"message={self.message!r})"
        )


class SchemaValidationFailure(IngestError):
    """General schema validation failure."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.SCHEMA_VALIDATION_FAILED,
            priority=PRIORITY_SCHEMA_VALIDATION_FAILURE,
        )


class BitDepthViolation(IngestError):
    """8-bit conversion / bit-depth violation."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.BIT_DEPTH_VIOLATION,
            priority=PRIORITY_BIT_DEPTH_VIOLATION,
        )


class GammaViolation(IngestError):
    """Gamma/non-linear violation."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.GAMMA_VIOLATION,
            priority=PRIORITY_GAMMA_VIOLATION,
        )


class SchemaDriftFailure(IngestError):
    """Schema drift / unexpected field violation."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.SCHEMA_DRIFT,
            priority=PRIORITY_SCHEMA_DRIFT,
        )


class OtherIngestFailure(IngestError):
    """Fallback ingest failure."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.OTHER_FAILURE,
            priority=PRIORITY_OTHER_FAILURE,
        )


_PRIORITY_BY_EXIT_CODE = {
    IngestExitCode.SUCCESS: PRIORITY_SUCCESS,
    IngestExitCode.SCHEMA_DRIFT: PRIORITY_SCHEMA_DRIFT,
    IngestExitCode.GAMMA_VIOLATION: PRIORITY_GAMMA_VIOLATION,
    IngestExitCode.BIT_DEPTH_VIOLATION: PRIORITY_BIT_DEPTH_VIOLATION,
    IngestExitCode.SCHEMA_VALIDATION_FAILED: PRIORITY_SCHEMA_VALIDATION_FAILURE,
    IngestExitCode.OTHER_FAILURE: PRIORITY_OTHER_FAILURE,
}


def aggregate_errors(errors: Iterable[IngestError]) -> Optional[IngestError]:
    """Return highest-priority ingest error, or None if empty."""
    materialized = list(errors)
    if not materialized:
        return None
    return max(materialized, key=lambda error: error.priority)


def aggregate_exit_code(errors: Iterable[IngestError]) -> IngestExitCode:
    """Return dominant exit code from typed ingest errors."""
    dominant_error = aggregate_errors(errors)
    if dominant_error is None:
        return IngestExitCode.SUCCESS
    return dominant_error.exit_code


def exit_code_priority(exit_code: IngestExitCode) -> int:
    """Return priority for a transport-layer exit code."""
    return _PRIORITY_BY_EXIT_CODE.get(exit_code, PRIORITY_OTHER_FAILURE)
