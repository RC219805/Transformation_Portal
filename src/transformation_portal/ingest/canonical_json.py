"""Canonical JSON serialization helpers for ingest deterministic artifacts.

This profile is used by evidence hashing and normalization outputs. It is not
the machine-mode wire serializer: machine-mode rendering intentionally uses
``ensure_ascii=True`` while this profile keeps unicode code points
(``ensure_ascii=False``) under ``tp.canonical.json.v1``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, TextIO

TP_CANONICAL_JSON_PROFILE = "tp.canonical.json.v1"
_CANONICAL_JSON_KWARGS: dict[str, Any] = {
    "sort_keys": True,
    "ensure_ascii": False,
    "separators": (",", ":"),
    "allow_nan": False,
}


def _set_sort_key(value: Any) -> str:
    """Build a deterministic key for sorting set/frozenset values."""
    try:
        return json.dumps(value, **_CANONICAL_JSON_KWARGS)
    except (TypeError, ValueError):
        return f"{type(value).__name__}:{value!r}"


def _numpy_module_for(payload: Any) -> Any:
    """Return NumPy only when normalization actually sees a NumPy value."""

    numpy_module = sys.modules.get("numpy")
    if numpy_module is not None:
        return numpy_module
    if type(payload).__module__.partition(".")[0] != "numpy":
        return None
    try:
        return import_module("numpy")
    except Exception:  # pragma: no cover - optional import failure
        return None


def to_jsonable(payload: Any) -> Any:
    """Recursively normalize payload to JSON-safe primitive/container types."""
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload

    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")

    if isinstance(payload, Path):
        return str(payload)

    if isinstance(payload, Enum):
        return to_jsonable(payload.value)

    np = _numpy_module_for(payload)
    if np is not None:
        if isinstance(payload, np.ndarray):
            return [to_jsonable(item) for item in payload.tolist()]
        if isinstance(payload, np.generic):
            return to_jsonable(payload.item())

    if is_dataclass(payload) and not isinstance(payload, type):
        return to_jsonable(asdict(payload))

    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        return to_jsonable(payload.to_dict())

    if isinstance(payload, Mapping):
        return {str(key): to_jsonable(value) for key, value in payload.items()}

    if isinstance(payload, (list, tuple)):
        return [to_jsonable(item) for item in payload]

    if isinstance(payload, (set, frozenset)):
        normalized = [to_jsonable(item) for item in payload]
        return sorted(normalized, key=_set_sort_key)

    if hasattr(payload, "__dict__"):
        return {str(key): to_jsonable(value) for key, value in vars(payload).items() if not str(key).startswith("_")}

    payload_repr = repr(payload)
    if len(payload_repr) > 256:
        payload_repr = payload_repr[:253] + "..."
    raise TypeError("Unsupported type for canonical JSON " "serialization: " f"{type(payload).__name__} ({payload_repr})")


def dumps_json(payload: Any, **kwargs: Any) -> str:
    """Serialize payload after JSON-safe normalization."""
    return json.dumps(to_jsonable(payload), **kwargs)


def dump_json(payload: Any, fp: TextIO, **kwargs: Any) -> None:
    """Write payload as JSON after JSON-safe normalization."""
    json.dump(to_jsonable(payload), fp, **kwargs)


def canonicalize_json(payload: Any) -> bytes:
    """Serialize payload deterministically under ``tp.canonical.json.v1``."""
    return dumps_json(payload, **_CANONICAL_JSON_KWARGS).encode("utf-8")
