"""Bounded, filesystem-independent validation of sharded checkpoint indexes.

This validates the exact index bytes supplied by an owning loader. It does not
freeze a mutable model directory or authorize Accelerate's vulnerable loaders.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_WEIGHT_MAP_ENTRIES = 250_000


def checkpoint_relative_path(value: str) -> str:
    """Reject ambiguous or escaping shard names on either supported path syntax."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Checkpoint index has an unsafe weight-map path")
    return value


def parse_checkpoint_weight_map(
    raw: bytes,
    *,
    maximum_bytes: int = MAX_INDEX_BYTES,
    maximum_entries: int = MAX_WEIGHT_MAP_ENTRIES,
) -> dict[str, str]:
    """Parse bounded JSON once, rejecting duplicate keys and unsafe shard paths."""
    if not raw or len(raw) > maximum_bytes:
        raise ValueError("Checkpoint index is empty or oversized")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, value in pairs:
            if key in values:
                raise ValueError("Checkpoint index repeats a key")
            values[key] = value
        return values

    def reject_constant(value: str) -> None:
        raise ValueError(f"Checkpoint index contains a non-finite number: {value}")

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Checkpoint index is not valid bounded JSON") from exc
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index has no non-empty weight_map")
    if len(weight_map) > maximum_entries:
        raise ValueError("Checkpoint index has too many weight-map entries")
    for value in weight_map.values():
        checkpoint_relative_path(value)
    return weight_map
