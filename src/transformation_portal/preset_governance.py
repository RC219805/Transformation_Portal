"""Lightweight shared preset-governance helpers.

This module centralizes placeholder detection and explicit preset-family
markers so health scanning, compliance validation, and schema checks do not
drift over time.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

MATERIALS_PBR_PRESET_FAMILY = "materials_pbr"
MATERIAL_PRESET_FAMILIES = frozenset({MATERIALS_PBR_PRESET_FAMILY})
PLACEHOLDER_MARKERS = ("NEEDS_VERIFICATION", "PLACEHOLDER", "PENDING", "TODO", "TBD", "UPDATE_WHEN")
_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"NEEDS_VERIFICATION", re.IGNORECASE),
    re.compile(r"PLACEHOLDER", re.IGNORECASE),
    re.compile(r"PENDING_VERIFICATION", re.IGNORECASE),
    re.compile(r"TODO_REPLACE", re.IGNORECASE),
    re.compile(r"^0{20,}$"),
)


def normalize_preset_family(value: Any) -> str | None:
    """Normalize a preset-family marker to a stable lowercase identifier."""
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    return normalized or None


def is_material_preset_family(value: Any) -> bool:
    """Return True when a marker declares the materials preset family."""
    normalized = normalize_preset_family(value)
    return normalized in MATERIAL_PRESET_FAMILIES


def is_placeholder_string(value: Any, *, treat_empty_as_placeholder: bool = True) -> bool:
    """Return True when a value is an unresolved placeholder string."""
    if not isinstance(value, str):
        return False

    normalized = value.strip()
    if not normalized:
        return treat_empty_as_placeholder

    upper = normalized.upper()
    if any(marker in upper for marker in PLACEHOLDER_MARKERS):
        return True

    return any(pattern.search(normalized) for pattern in _PLACEHOLDER_PATTERNS)


def iter_placeholder_string_paths(
    node: Any,
    prefix: str = "",
    *,
    treat_empty_as_placeholder: bool = True,
) -> Iterator[tuple[str, str]]:
    """Yield placeholder-bearing YAML paths and values from nested payloads."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_placeholder_string_paths(
                value,
                path,
                treat_empty_as_placeholder=treat_empty_as_placeholder,
            )
        return

    if isinstance(node, list):
        for index, value in enumerate(node):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from iter_placeholder_string_paths(
                value,
                path,
                treat_empty_as_placeholder=treat_empty_as_placeholder,
            )
        return

    if is_placeholder_string(node, treat_empty_as_placeholder=treat_empty_as_placeholder):
        yield prefix, str(node)
