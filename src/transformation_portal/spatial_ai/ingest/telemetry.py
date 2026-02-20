"""Structured telemetry protocol for ingest boundary instrumentation.

This module provides a lightweight Protocol for emitting structured events
at ingest decision points — validation failures, matrix fallbacks, and
postprocess guards.

Usage:
    class MyTelemetry:
        def emit(self, event: str, **fields: object) -> None:
            print(f"Event: {event}, Fields: {fields}")

    decoder = LinearDecoder(telemetry=MyTelemetry())

Zero overhead when unused (default NullTelemetry is a no-op).
"""

from __future__ import annotations

from typing import Protocol


class IngestTelemetry(Protocol):
    """Protocol for ingest boundary telemetry.

    Implementations can log, emit metrics, or forward events to observability systems.
    """

    def emit(self, event: str, **fields: object) -> None:
        """Emit a structured event with optional fields.

        Args:
            event: Event name (e.g., "ingest.validation_failed").
            **fields: Arbitrary key-value metadata (e.g., field="camera_whitebalance", reason="non_numeric").
        """
        ...


class NullTelemetry:
    """No-op telemetry implementation (default — zero overhead)."""

    def emit(self, event: str, **fields: object) -> None:
        """No-op emit — does nothing."""
        return
