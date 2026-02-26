"""Canonical JSON serialization helpers for ingest deterministic artifacts.

This profile is used by evidence hashing and normalization outputs. It is not
the machine-mode wire serializer: machine-mode rendering intentionally uses
``ensure_ascii=True`` while this profile keeps unicode code points
(``ensure_ascii=False``) under ``tp.canonical.json.v1``.
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
