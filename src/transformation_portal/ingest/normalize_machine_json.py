"""Canonical normalization helpers for ingest JSON."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, MutableMapping
from typing import Any

from .canonical_json import canonicalize_json

DEFAULT_NORMALIZATION_PROFILE = "ingest_v1"

_SUPPORTED_PROFILES = frozenset({DEFAULT_NORMALIZATION_PROFILE})
_VOLATILE_PROVENANCE_FIELDS = frozenset(
    {
        "run_id",
        "timestamps",
        "host",
        "toolchain",
        "git_commit",
    }
)
_VOLATILE_MACHINE_DATA_FIELDS = frozenset(
    {
        "elapsed_seconds",
        "exiftool_version",
        "pydantic_version",
        "git_version",
        "rawpy_version",
        "libraw_version",
    }
)


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize payload with deterministic rules."""
    return canonicalize_json(payload)


def _validate_profile(profile: str) -> None:
    if profile not in _SUPPORTED_PROFILES:
        supported = ", ".join(sorted(_SUPPORTED_PROFILES))
        raise ValueError(f"Unsupported normalization profile: " f"{profile}. Supported: {supported}")


def _looks_like_provenance_payload(payload: Mapping[str, Any]) -> bool:
    return {
        "file_integrity",
        "exif",
        "pipeline_config",
    }.issubset(payload.keys())


def _strip_provenance_volatile_fields(
    payload: MutableMapping[str, Any],
) -> None:
    for field in _VOLATILE_PROVENANCE_FIELDS:
        payload.pop(field, None)


def _normalize_machine_data_block(data: MutableMapping[str, Any]) -> None:
    for key in _VOLATILE_MACHINE_DATA_FIELDS:
        data.pop(key, None)

    items = data.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, MutableMapping):
                item.pop("elapsed_seconds", None)

    if _looks_like_provenance_payload(data):
        _strip_provenance_volatile_fields(data)


def _normalize_in_place(payload: Any) -> None:
    if isinstance(payload, MutableMapping):
        if _looks_like_provenance_payload(payload):
            _strip_provenance_volatile_fields(payload)

        data = payload.get("data")
        if isinstance(data, MutableMapping):
            _normalize_machine_data_block(data)

        for value in payload.values():
            _normalize_in_place(value)
        return

    if isinstance(payload, list):
        for item in payload:
            _normalize_in_place(item)


def normalize_machine_payload(
    payload: Mapping[str, Any],
    *,
    profile: str = DEFAULT_NORMALIZATION_PROFILE,
) -> dict[str, Any]:
    """Return normalized payload for deterministic comparisons."""
    _validate_profile(profile)
    normalized = copy.deepcopy(dict(payload))
    _normalize_in_place(normalized)
    return normalized


def normalize_machine_json_bytes(
    raw_json: str | bytes,
    *,
    profile: str = DEFAULT_NORMALIZATION_PROFILE,
) -> bytes:
    """Normalize JSON from raw bytes/string into canonical bytes."""
    _validate_profile(profile)
    if isinstance(raw_json, bytes):
        parsed = json.loads(raw_json.decode("utf-8"))
    else:
        parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("Input JSON must be an object")
    normalized = normalize_machine_payload(parsed, profile=profile)
    return canonical_json_bytes(normalized)


def normalized_payload_sha256(
    payload: Mapping[str, Any],
    *,
    profile: str = DEFAULT_NORMALIZATION_PROFILE,
) -> str:
    """Compute SHA256 hex digest over normalized canonical JSON bytes."""
    normalized = normalize_machine_payload(payload, profile=profile)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()
