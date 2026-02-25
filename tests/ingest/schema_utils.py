"""JSON Schema validation helpers for machine-mode contract tests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "machine_mode" / "tp.meta.machine.v1"
SCHEMA_ENTRYPOINT = "machine_mode.schema.json"
VOLATILE_VERSION_FIELDS = {
    "exiftool_version",
    "pydantic_version",
    "git_version",
    "rawpy_version",
    "libraw_version",
}


@lru_cache(maxsize=1)
def _load_schema_bundle() -> tuple[dict[str, Any], Registry]:
    if not SCHEMA_DIR.exists():
        raise FileNotFoundError(f"Machine-mode schema directory not found: {SCHEMA_DIR}")

    schemas_by_name: dict[str, dict[str, Any]] = {}
    registry = Registry()
    for schema_path in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            raise AssertionError(f"Schema file is missing $id: {schema_path}")
        schemas_by_name[schema_path.name] = schema
        registry = registry.with_resource(schema_id, Resource.from_contents(schema))

    if SCHEMA_ENTRYPOINT not in schemas_by_name:
        raise AssertionError(f"Schema entrypoint missing: {SCHEMA_ENTRYPOINT}")

    return schemas_by_name[SCHEMA_ENTRYPOINT], registry


def _format_json_path(path: Iterable[Any]) -> str:
    rendered = "$"
    for segment in path:
        if isinstance(segment, int):
            rendered += f"[{segment}]"
        else:
            rendered += f"[{segment!r}]"
    return rendered


def validate_machine_payload(payload: dict[str, Any]) -> None:
    """Validate a machine-mode payload against the canonical JSON Schema."""
    schema, registry = _load_schema_bundle()
    validator = Draft202012Validator(schema, registry=registry)
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (len(error.path), tuple(str(part) for part in error.path), error.message),
    )
    if not errors:
        return

    lines = ["Machine payload failed JSON Schema validation:"]
    for error in errors:
        location = _format_json_path(error.path)
        lines.append(f"- {location}: {error.message}")
    raise AssertionError("\n".join(lines))


def normalize_machine_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize known-volatile machine payload fields for stability checks."""
    normalized = json.loads(json.dumps(payload))
    data = normalized.get("data")
    if not isinstance(data, dict):
        return normalized

    if "elapsed_seconds" in data:
        data["elapsed_seconds"] = 0.0

    items = data.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "elapsed_seconds" in item:
                item["elapsed_seconds"] = 0.0

    for key in VOLATILE_VERSION_FIELDS:
        if key in data and data[key] is not None:
            data[key] = "__VOLATILE__"

    return normalized
