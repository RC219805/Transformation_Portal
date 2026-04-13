"""Shared run-card versioning and path normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from transformation_portal.schemas.run_card import (
    RUN_CARD_SCHEMA_URIS,
    SUPPORTED_RUN_CARD_VERSIONS,
    get_run_card_schema_uri,
    normalize_run_card_version,
)


class RunCardPathValidationError(ValueError):
    """Raised when a run-card relative path is empty or unsafe."""


def infer_run_card_version(payload: Mapping[str, Any]) -> str:
    """Infer the run-card version with legacy compatibility."""
    explicit_version = payload.get("run_card_version")
    if explicit_version is not None:
        return normalize_run_card_version(explicit_version)
    return "v2" if "artifact_tree" in payload else "v1"


def with_inferred_run_card_version(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow payload copy with run_card_version materialized."""
    version = infer_run_card_version(payload)
    hydrated = dict(payload)
    hydrated.setdefault("run_card_version", version)
    return hydrated


def get_run_card_schema_uri_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical schema URI for the given payload."""
    return get_run_card_schema_uri(infer_run_card_version(payload))


def normalize_run_card_relative_path(relative_path: Any) -> str:
    """Normalize a run-card relative path and reject unsafe forms."""
    raw = str(relative_path or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw or "\\" in raw:
        raise RunCardPathValidationError(f"artifact relative_path is invalid: {relative_path!r}")

    candidate = PurePosixPath(raw)
    if candidate.is_absolute():
        raise RunCardPathValidationError(f"artifact relative_path must not be absolute: {relative_path}")

    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise RunCardPathValidationError(f"artifact relative_path is invalid: {relative_path!r}")
    if any(part == ".." for part in candidate.parts):
        raise RunCardPathValidationError(f"artifact relative_path must not contain traversal segments: {relative_path}")
    return normalized


__all__ = [
    "RUN_CARD_SCHEMA_URIS",
    "SUPPORTED_RUN_CARD_VERSIONS",
    "RunCardPathValidationError",
    "get_run_card_schema_uri",
    "get_run_card_schema_uri_for_payload",
    "infer_run_card_version",
    "normalize_run_card_relative_path",
    "normalize_run_card_version",
    "with_inferred_run_card_version",
]
