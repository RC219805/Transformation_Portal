"""Schema helpers for tp.archive.machine.v1 tests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    from jsonschema import Draft202012Validator  # type: ignore
    from referencing import Registry, Resource  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    Draft202012Validator = None  # type: ignore[assignment]
    Registry = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "machine_mode" / "tp.archive.machine.v1"
SCHEMA_ENTRYPOINT = "machine_mode.schema.json"


@lru_cache(maxsize=1)
def _load_schema_bundle() -> tuple[dict[str, Any], Registry]:
    if Draft202012Validator is None or Registry is None or Resource is None:
        raise RuntimeError("jsonschema + referencing are required for full archive contract validation")

    schemas_by_name: dict[str, dict[str, Any]] = {}
    registry = Registry()

    for schema_path in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_id = schema.get("$id")
        if isinstance(schema_id, str) and schema_id:
            registry = registry.with_resource(schema_id, Resource.from_contents(schema))
        schemas_by_name[schema_path.name] = schema

    if SCHEMA_ENTRYPOINT not in schemas_by_name:
        raise AssertionError(f"Missing schema entrypoint: {SCHEMA_ENTRYPOINT}")

    return schemas_by_name[SCHEMA_ENTRYPOINT], registry


def _format_path(path: Iterable[Any]) -> str:
    rendered = "$"
    for segment in path:
        if isinstance(segment, int):
            rendered += f"[{segment}]"
        else:
            rendered += f"[{segment!r}]"
    return rendered


def validate_archive_machine_payload(payload: dict[str, Any]) -> None:
    if Draft202012Validator is None:
        required = {"schema", "command", "success", "exit_code", "data", "error"}
        missing = sorted(required.difference(payload))
        if missing:
            raise AssertionError(f"Archive machine payload missing required keys: {', '.join(missing)}")
        if payload.get("schema") != "tp.archive.machine.v1":
            raise AssertionError("Archive machine payload schema must be tp.archive.machine.v1")
        if not isinstance(payload.get("data"), dict):
            raise AssertionError("Archive machine payload data field must be an object")
        return

    schema, registry = _load_schema_bundle()
    validator = Draft202012Validator(schema, registry=registry)
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (len(error.path), tuple(str(part) for part in error.path), error.message),
    )
    if not errors:
        return

    lines = ["Archive machine payload failed schema validation:"]
    for error in errors:
        lines.append(f"- {_format_path(error.path)}: {error.message}")
    raise AssertionError("\n".join(lines))
