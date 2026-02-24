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
    """Stable ingest contract exit codes."""

    SUCCESS = 0
    SCHEMA_VALIDATION_FAILED = 1
    BIT_DEPTH_VIOLATION = 2
    GAMMA_VIOLATION = 3
    SCHEMA_DRIFT = 4
    OTHER_FAILURE = 5


@dataclass(eq=True, frozen=True)
class IngestError(Exception):
    """Base typed ingest error with explicit severity priority."""

    message: str
    exit_code: IngestExitCode
    priority: int

    def __str__(self) -> str:
        return self.message


class SchemaValidationFailure(IngestError):
    """General schema validation failure."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.SCHEMA_VALIDATION_FAILED,
            priority=10,
        )


class BitDepthViolation(IngestError):
    """8-bit conversion / bit-depth violation."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.BIT_DEPTH_VIOLATION,
            priority=20,
        )


class GammaViolation(IngestError):
    """Gamma/non-linear violation."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.GAMMA_VIOLATION,
            priority=30,
        )


class SchemaDriftFailure(IngestError):
    """Schema drift / unexpected field violation."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.SCHEMA_DRIFT,
            priority=40,
        )


class OtherIngestFailure(IngestError):
    """Fallback ingest failure."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            exit_code=IngestExitCode.OTHER_FAILURE,
            priority=0,
        )


_PRIORITY_BY_EXIT_CODE = {
    IngestExitCode.SCHEMA_DRIFT: SchemaDriftFailure("schema drift").priority,
    IngestExitCode.GAMMA_VIOLATION: GammaViolation("gamma violation").priority,
    IngestExitCode.BIT_DEPTH_VIOLATION: BitDepthViolation("bit-depth violation").priority,
    IngestExitCode.SCHEMA_VALIDATION_FAILED: SchemaValidationFailure("schema validation failed").priority,
    IngestExitCode.OTHER_FAILURE: OtherIngestFailure("other failure").priority,
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
    return _PRIORITY_BY_EXIT_CODE.get(exit_code, OtherIngestFailure("other failure").priority)
