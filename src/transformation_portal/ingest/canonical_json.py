"""Canonical JSON serialization helpers for ingest evidence workflows.

This module defines a single canonicalization profile for evidence hashing.
The profile is intentionally explicit and separate from machine-mode rendering.
"""

from __future__ import annotations

import json
from typing import Any

TP_CANONICAL_JSON_PROFILE = "tp.canonical.json.v1"
_CANONICAL_JSON_KWARGS: dict[str, Any] = {
    "sort_keys": True,
    "ensure_ascii": False,
    "separators": (",", ":"),
    "allow_nan": False,
}


def canonicalize_json(payload: Any) -> bytes:
    """Serialize payload deterministically under ``tp.canonical.json.v1``."""
    return json.dumps(payload, **_CANONICAL_JSON_KWARGS).encode("utf-8")
