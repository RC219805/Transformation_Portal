#!/usr/bin/env python3
"""Shared helpers for archive governance tools."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from transformation_portal.determinism.jcs import dumps as jcs_dumps

ARCHIVE_MACHINE_SCHEMA_VERSION = "tp.archive.machine.v1"
CANONICAL_PROFILE_V1 = "canonical_v1"
CANONICAL_PROFILE_JCS = "jcs"
CANONICAL_PROFILES = (CANONICAL_PROFILE_V1, CANONICAL_PROFILE_JCS)


@dataclass(frozen=True)
class CommandError:
    """Machine-readable command error payload."""

    type: str
    message: str
    exit_code: dict[str, Any]
    priority: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "message": self.message,
            "exit_code": dict(self.exit_code),
            "priority": int(self.priority),
        }


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically (temp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically with UTF-8 and LF semantics."""
    atomic_write_bytes(path, text.encode("utf-8"))


def _ensure_finite_numbers(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float encountered at {path}")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _ensure_finite_numbers(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _ensure_finite_numbers(nested, path=f"{path}[{index}]")


def deterministic_json_dumps(
    payload: Any,
    *,
    pretty: bool,
    canonical_profile: str = CANONICAL_PROFILE_V1,
) -> str:
    """Serialize payload with deterministic canonical profiles."""
    _ensure_finite_numbers(payload)

    if canonical_profile == CANONICAL_PROFILE_JCS:
        text = jcs_dumps(payload)
        if pretty:
            parsed = json.loads(text)
            return json.dumps(
                parsed,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        return text

    if canonical_profile != CANONICAL_PROFILE_V1:
        supported = ", ".join(CANONICAL_PROFILES)
        raise ValueError(f"Unsupported canonical profile {canonical_profile!r}. Supported: {supported}")

    kwargs: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    if pretty:
        kwargs["indent"] = 2
        return json.dumps(payload, **kwargs)
    kwargs["separators"] = (",", ":")
    return json.dumps(payload, **kwargs)


def json_line(payload: Mapping[str, Any], *, canonical_profile: str = CANONICAL_PROFILE_V1) -> str:
    """Return deterministic JSONL line with trailing newline."""
    return deterministic_json_dumps(payload, pretty=False, canonical_profile=canonical_profile) + "\n"


def make_typed_error(
    *,
    type_name: str,
    message: str,
    exit_code: int,
    exit_name: str = "OTHER_FAILURE",
    priority: int = 10,
) -> dict[str, Any]:
    """Create typed machine error payload."""
    return CommandError(
        type=type_name,
        message=message,
        exit_code={"name": exit_name, "value": int(exit_code)},
        priority=priority,
    ).to_dict()


def make_machine_envelope(
    *,
    command: str,
    exit_code: int,
    data: Mapping[str, Any],
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create archive machine-mode envelope."""
    return {
        "schema": ARCHIVE_MACHINE_SCHEMA_VERSION,
        "command": command,
        "success": int(exit_code) == 0,
        "exit_code": int(exit_code),
        "data": dict(data),
        "error": None if error is None else dict(error),
    }


def emit_machine_payload(
    *,
    envelope: Mapping[str, Any],
    pretty: bool,
    json_output: Path | None,
    canonical_profile: str,
) -> None:
    """Emit deterministic machine payload to stdout or file."""
    text = deterministic_json_dumps(
        dict(envelope),
        pretty=pretty,
        canonical_profile=canonical_profile,
    )

    if json_output is not None:
        atomic_write_text(json_output, text)
        return

    print(text)
