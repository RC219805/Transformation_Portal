"""Shared artifact payload helpers for Spatial AI orchestration."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def _sha256_array(array: np.ndarray) -> str:
    """Return a deterministic SHA-256 for a numpy array payload."""
    contiguous = array if array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous.view(np.uint8).ravel())).hexdigest()


def _sanitize_json_value(value: Any) -> Any:
    """Convert values into canonical-JSON-safe primitives."""
    if is_dataclass(value):
        return _sanitize_json_value(asdict(value))

    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(inner) for key, inner in value.items()}

    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(inner) for inner in value]

    if isinstance(value, np.ndarray):
        return _sanitize_json_value(value.tolist())

    if isinstance(value, np.generic):
        return _sanitize_json_value(value.item())

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


__all__ = [
    "_sanitize_json_value",
    "_sha256_array",
]
