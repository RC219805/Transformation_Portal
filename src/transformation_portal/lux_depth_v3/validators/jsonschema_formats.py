"""Shared JSON Schema format helpers for run-card validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def build_jsonschema_format_checker() -> Any:
    """Build a jsonschema FormatChecker with deterministic date-time validation."""
    try:
        from jsonschema import FormatChecker
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema dependency is required for run card schema validation") from exc

    checker = FormatChecker()

    @checker.checks("date-time")
    def _validate_datetime(value: object) -> bool:
        if not isinstance(value, str):
            return True
        candidate = value.strip()
        if not candidate:
            return False
        normalized = f"{candidate[:-1]}+00:00" if candidate.endswith("Z") else candidate
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return False
        return parsed.tzinfo is not None

    return checker
