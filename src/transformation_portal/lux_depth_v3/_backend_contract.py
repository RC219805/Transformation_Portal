"""Internal helpers for Lux Depth V3 backend identifiers.

This module keeps backend naming and alias handling consistent across CLI,
config normalization, manifests, run cards, and backend registry calls.
It is intentionally internal and not part of the public package surface.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

CANONICAL_BACKEND_IDS = (
    "da3",
    "depth_pro",
    "synthetic",
    "da2",
    "depthcrafter",
    "ensemble",
)

LEGACY_BACKEND_ALIASES = {
    "depth_anything_v3": "da3",
    "depth-anything-v3": "da3",
}

BACKEND_ALIASES = {
    **LEGACY_BACKEND_ALIASES,
    "depth-pro": "depth_pro",
}

VALID_BACKEND_IDS = frozenset(CANONICAL_BACKEND_IDS)


def _coerce_backend_value(value: object) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def backend_alias_warning(alias: str, canonical: str) -> str:
    """Return the standard legacy backend alias warning text."""
    return f"Legacy backend alias '{alias}' is deprecated; use '{canonical}' instead."


def normalize_backend_id(
    value: object,
    *,
    warn: bool = False,
    warning_context: str = "backend",
) -> Optional[str]:
    """Normalize a backend identifier to its canonical ID.

    Unknown backend identifiers are lowercased and trimmed but otherwise
    preserved so callers can still surface a precise validation error.
    """
    normalized = _coerce_backend_value(value)
    if normalized is None:
        return None

    canonical = BACKEND_ALIASES.get(normalized, normalized)
    if warn and normalized in LEGACY_BACKEND_ALIASES:
        warnings.warn(
            f"{warning_context}: {backend_alias_warning(normalized, canonical)}",
            FutureWarning,
            stacklevel=3,
        )
    return canonical


def normalize_backend_sequence(
    values: Iterable[object],
    *,
    warn: bool = False,
    warning_context: str = "backend sequence",
) -> tuple[str, ...]:
    """Normalize a sequence of backend identifiers and remove duplicates."""
    normalized_values: list[str] = []
    for value in values:
        canonical = normalize_backend_id(
            value,
            warn=warn,
            warning_context=warning_context,
        )
        if canonical and canonical not in normalized_values:
            normalized_values.append(canonical)
    return tuple(normalized_values)


def normalize_backend_provenance(value: object) -> Optional[str]:
    """Normalize backend IDs used in manifests and run cards."""
    return normalize_backend_id(value, warn=False)


def is_legacy_backend_alias(value: object) -> bool:
    """Return True when a value uses a deprecated backend alias."""
    normalized = _coerce_backend_value(value)
    return bool(normalized and normalized in LEGACY_BACKEND_ALIASES)
