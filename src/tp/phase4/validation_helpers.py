"""Shared validation helpers for Phase 4 deterministic provenance tooling.

This module consolidates common validation patterns used across Phase 4C/4D/4E/4F:
- SHA256 hex digest validation
- Relative path uniqueness and ordering checks
- Record indexing by relative_path
- Contract version enforcement
"""

from __future__ import annotations

import re
from typing import Any, Callable, TypeVar

# Compile once, reuse everywhere
SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")

# Type variable for exception classes
E = TypeVar("E", bound=Exception)


def is_valid_sha256_hex(value: Any) -> bool:
    """Check if value is a valid lowercase 64-character SHA256 hex digest."""
    return isinstance(value, str) and SHA256_HEX_RE.fullmatch(value) is not None


def ensure_sha256_hex(value: Any, *, label: str, error_cls: type[E]) -> str:
    """Validate and return a SHA256 hex digest, raising error_cls if invalid.

    Args:
        value: The value to validate.
        label: Human-readable label for error messages.
        error_cls: Exception class to raise on validation failure.

    Returns:
        The validated SHA256 hex string.

    Raises:
        error_cls: If validation fails.
    """
    if not is_valid_sha256_hex(value):
        raise error_cls(f"{label} must be a lowercase 64-character sha256 hex digest")
    return value


def require_unique_relative_paths(
    records: list[dict[str, Any]],
    *,
    label: str,
    error_cls: type[E],
) -> None:
    """Ensure all records have unique relative_path values.

    Args:
        records: List of record dictionaries with relative_path keys.
        label: Human-readable label for error messages (e.g., "capture metadata").
        error_cls: Exception class to raise on validation failure.

    Raises:
        error_cls: If a record is missing relative_path or has a duplicate.
    """
    seen: set[str] = set()
    for index, record in enumerate(records):
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise error_cls(f"{label} record[{index}] missing relative_path")
        if relative_path in seen:
            raise error_cls(f"{label} duplicate relative_path: {relative_path}")
        seen.add(relative_path)


def require_sorted_relative_paths(
    records: list[dict[str, Any]],
    *,
    label: str,
    error_cls: type[E],
) -> None:
    """Ensure records are sorted by relative_path in ascending order.

    Args:
        records: List of record dictionaries with relative_path keys.
        label: Human-readable label for error messages.
        error_cls: Exception class to raise on validation failure.

    Raises:
        error_cls: If a record is missing relative_path or records are not
            sorted by relative_path.
    """
    relative_paths: list[str] = []
    for index, record in enumerate(records):
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise error_cls(f"{label} record[{index}] missing relative_path")
        relative_paths.append(relative_path)

    if relative_paths != sorted(relative_paths):
        raise error_cls(f"{label} must be sorted by relative_path")


def build_path_index(
    records: list[dict[str, Any]],
    *,
    label: str,
    error_cls: type[E],
) -> dict[str, dict[str, Any]]:
    """Build a lookup index from relative_path to record.

    Args:
        records: List of record dictionaries with relative_path keys.
        label: Human-readable label for error messages.
        error_cls: Exception class to raise on validation failure.

    Returns:
        Dictionary mapping relative_path strings to their corresponding records.

    Raises:
        error_cls: If a record is missing relative_path.
    """
    index: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise error_cls(f"{label} record[{position}] missing relative_path")
        index[relative_path] = record
    return index


def require_contract_version(
    value: Any,
    *,
    expected: str,
    label: str,
    error_cls: type[E],
) -> str:
    """Validate that a contract version matches the expected value.

    Args:
        value: The contract version value to check.
        expected: The expected contract version string.
        label: Human-readable label for error messages.
        error_cls: Exception class to raise on validation failure.

    Returns:
        The validated contract version string.

    Raises:
        error_cls: If the version does not match.
    """
    if value != expected:
        raise error_cls(f"{label} mismatch: expected {expected}, got {value!r}")
    return value


def validate_records_with_schema(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    *,
    error_cls: type[E],
    label: str,
    record_label_fn: Callable[[int, dict[str, Any]], str] | None = None,
) -> None:
    """Validate a list of records against a JSON schema.

    Args:
        records: List of record dictionaries to validate.
        schema: JSON Schema dictionary for validation.
        error_cls: Exception class to raise on validation failure.
        label: Human-readable label for the schema (e.g., "metadata").
        record_label_fn: Optional function to generate record-specific labels.
            Called with (index, record) and should return a string.

    Raises:
        error_cls: If schema validation fails for any record.
    """
    from .schema_validation import build_draft202012_validator

    validator = build_draft202012_validator(schema, error_cls=error_cls, label=label)
    for index, record in enumerate(records):
        try:
            errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        except (TypeError, ValueError) as exc:
            record_label = record_label_fn(index, record) if record_label_fn else f"record[{index}]"
            raise error_cls(
                f"{record_label} schema validation failed due to validator runtime error ({type(exc).__name__})"
            ) from exc
        if errors:
            first = errors[0]
            path = ".".join(str(part) for part in first.path) or "<root>"
            record_label = record_label_fn(index, record) if record_label_fn else f"record[{index}]"
            raise error_cls(f"{record_label} schema validation failed at {path}: {first.message}")


def validate_payload_with_schema(
    payload: dict[str, Any],
    schema: dict[str, Any],
    *,
    error_cls: type[E],
    label: str,
) -> None:
    """Validate a single payload object against a JSON schema.

    Args:
        payload: Payload dictionary to validate.
        schema: JSON Schema dictionary for validation.
        error_cls: Exception class to raise on validation failure.
        label: Human-readable label for error messages.

    Raises:
        error_cls: If schema validation fails.
    """
    from .schema_validation import build_draft202012_validator

    validator = build_draft202012_validator(schema, error_cls=error_cls, label=label)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<root>"
        raise error_cls(f"{label} schema validation failed at {path}: {first.message}")


def string_or_none(value: Any) -> str | None:
    """Return value as string if it is a string, otherwise None.

    This is used for optional field extraction where we want to preserve
    string values but convert non-strings to None.
    """
    if isinstance(value, str):
        return value
    return None
