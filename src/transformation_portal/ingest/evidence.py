"""Evidence artifact builders for projected machine-mode envelopes."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from .canonical_json import TP_CANONICAL_JSON_PROFILE, canonicalize_json

EVIDENCE_SCHEMA_VERSION = "tp.meta.evidence.v1"
MACHINE_SCHEMA_VERSION = "tp.meta.machine.v1"
DEFAULT_PROJECTION_PROFILE = "tp.projection.machine_to_evidence.v1"
_ALLOWED_MACHINE_COMMANDS = frozenset({"check-system", "extract", "validate", "extract-batch", "summarize"})
DEFAULT_PROJECTION_PROFILE_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "profiles" / f"{DEFAULT_PROJECTION_PROFILE}.json"
)
_DEFAULT_PROJECTION_PROFILE_PAYLOAD: dict[str, Any] = {
    "schema": DEFAULT_PROJECTION_PROFILE,
    "source_schema": MACHINE_SCHEMA_VERSION,
    "drop_paths": [
        "/data/elapsed_seconds",
        "/data/items/*/elapsed_seconds",
        "/data/exiftool_version",
        "/data/pydantic_version",
        "/data/git_version",
        "/data/rawpy_version",
        "/data/libraw_version",
    ],
}


def canonical_evidence_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize an evidence payload with the canonical JSON profile."""
    return canonicalize_json(dict(payload))


def _decode_pointer_token(token: str) -> str:
    # RFC 6901 requires "~0" -> "~" before "~1" -> "/" to avoid double-unescaping.
    return token.replace("~0", "~").replace("~1", "/")


def _split_json_pointer(pointer: str) -> tuple[str, ...]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"drop path must be an absolute JSON pointer: {pointer!r}")
    return tuple(_decode_pointer_token(token) for token in pointer.split("/")[1:])


def _drop_path_in_place(node: Any, tokens: tuple[str, ...], index: int = 0) -> None:
    if index >= len(tokens):
        return

    token = tokens[index]
    is_last = index == len(tokens) - 1

    if token == "*":
        if isinstance(node, MutableMapping):
            if is_last:
                node.clear()
                return
            for value in node.values():
                _drop_path_in_place(value, tokens, index + 1)
            return
        if isinstance(node, list):
            if is_last:
                node.clear()
                return
            for item in node:
                _drop_path_in_place(item, tokens, index + 1)
        return

    if isinstance(node, MutableMapping):
        if token not in node:
            return
        if is_last:
            node.pop(token, None)
            return
        _drop_path_in_place(node[token], tokens, index + 1)
        return

    if isinstance(node, list):
        if not token.isdigit():
            return
        item_index = int(token)
        if item_index < 0 or item_index >= len(node):
            return
        if is_last:
            node.pop(item_index)
            return
        _drop_path_in_place(node[item_index], tokens, index + 1)


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field} must be a 64-character sha256 digest")
    lowered = value.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        raise ValueError(f"{field} must be lowercase hex")
    return lowered


def load_projection_profile(profile_path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate a machine-to-evidence projection profile."""
    path = profile_path or DEFAULT_PROJECTION_PROFILE_PATH
    if path.exists():
        try:
            raw_profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to load projection profile {path}: {exc}") from exc
    elif profile_path is None:
        raw_profile = copy.deepcopy(_DEFAULT_PROJECTION_PROFILE_PAYLOAD)
    else:
        raise ValueError(f"projection profile path not found: {path}")

    if not isinstance(raw_profile, dict):
        raise ValueError("projection profile must be a JSON object")
    if raw_profile.get("schema") != DEFAULT_PROJECTION_PROFILE:
        raise ValueError(f"projection profile schema must be {DEFAULT_PROJECTION_PROFILE}")
    if raw_profile.get("source_schema") != MACHINE_SCHEMA_VERSION:
        raise ValueError(f"projection profile source_schema must be {MACHINE_SCHEMA_VERSION}")

    drop_paths = raw_profile.get("drop_paths")
    if not isinstance(drop_paths, list) or not drop_paths:
        raise ValueError("projection profile drop_paths must be a non-empty list")

    for drop_path in drop_paths:
        _split_json_pointer(drop_path)

    return raw_profile


def project_machine_envelope(
    machine_payload: Mapping[str, Any],
    *,
    projection_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a versioned projection profile to a machine-mode envelope."""
    if machine_payload.get("schema") != MACHINE_SCHEMA_VERSION:
        found = machine_payload.get("schema")
        raise ValueError(f"machine payload schema must be {MACHINE_SCHEMA_VERSION}, got {found!r}")

    profile = copy.deepcopy(dict(projection_profile or load_projection_profile()))
    drop_paths = profile.get("drop_paths")
    if not isinstance(drop_paths, list):
        raise ValueError("projection profile drop_paths must be a list")

    projected = copy.deepcopy(dict(machine_payload))
    for pointer in drop_paths:
        _drop_path_in_place(projected, _split_json_pointer(pointer))
    return projected


def _extract_file_sha256(projected_envelope: Mapping[str, Any]) -> str | None:
    file_integrity = projected_envelope.get("file_integrity")
    if isinstance(file_integrity, Mapping) and "sha256" in file_integrity:
        try:
            return _validate_sha256(file_integrity.get("sha256"), field="file_integrity.sha256")
        except ValueError:
            return None

    data = projected_envelope.get("data")
    if isinstance(data, Mapping):
        data_file_integrity = data.get("file_integrity")
        if isinstance(data_file_integrity, Mapping) and "sha256" in data_file_integrity:
            try:
                return _validate_sha256(data_file_integrity.get("sha256"), field="data.file_integrity.sha256")
            except ValueError:
                return None
    return None


def _validate_machine_contract_surface(machine_payload: Mapping[str, Any]) -> tuple[str, bool, int]:
    command = machine_payload.get("command")
    if not isinstance(command, str) or command not in _ALLOWED_MACHINE_COMMANDS:
        allowed = ", ".join(sorted(_ALLOWED_MACHINE_COMMANDS))
        raise ValueError(f"machine payload command must be one of: {allowed}")

    success = machine_payload.get("success")
    if type(success) is not bool:
        raise ValueError("machine payload success must be a boolean")

    exit_code = machine_payload.get("exit_code")
    if type(exit_code) is not int or not (0 <= exit_code <= 255):
        raise ValueError("machine payload exit_code must be an integer in [0,255]")
    if success and exit_code != 0:
        raise ValueError("machine payload exit_code must be 0 when success is true")
    if not success and exit_code == 0:
        raise ValueError("machine payload exit_code must be non-zero when success is false")

    data = machine_payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("machine payload data must be an object")

    error = machine_payload.get("error")
    if error is not None and not isinstance(error, Mapping):
        raise ValueError("machine payload error must be an object or null")

    return command, success, exit_code


def build_evidence_payload(
    machine_payload: Mapping[str, Any],
    *,
    projection_profile: Mapping[str, Any] | None = None,
    signature: Mapping[str, Any] | None = None,
    timestamp: Mapping[str, Any] | None = None,
    bundle_root_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a ``tp.meta.evidence.v1`` object from a machine envelope."""
    profile = copy.deepcopy(dict(projection_profile or load_projection_profile()))
    projected_envelope = project_machine_envelope(machine_payload, projection_profile=profile)
    command, success, exit_code = _validate_machine_contract_surface(projected_envelope)
    evidence_sha256 = hashlib.sha256(canonicalize_json(projected_envelope)).hexdigest()

    if bundle_root_sha256 is not None:
        bundle_root_sha256 = _validate_sha256(bundle_root_sha256, field="bundle_root_sha256")

    return {
        "schema": EVIDENCE_SCHEMA_VERSION,
        "source_schema": MACHINE_SCHEMA_VERSION,
        "command": command,
        "success": success,
        "exit_code": exit_code,
        "envelope_projection_profile": profile["schema"],
        "canonicalization": TP_CANONICAL_JSON_PROFILE,
        "evidence_sha256": evidence_sha256,
        "file_sha256": _extract_file_sha256(projected_envelope),
        "bundle_root_sha256": bundle_root_sha256,
        "signature": dict(signature) if signature is not None else None,
        "timestamp": dict(timestamp) if timestamp is not None else None,
        "projected_envelope": projected_envelope,
    }
