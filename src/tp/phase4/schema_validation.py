"""Shared JSON Schema validation helpers for Phase 4 contract tooling."""

from __future__ import annotations

from typing import Any


def build_draft202012_validator(schema: dict[str, Any], *, error_cls: type[Exception], label: str) -> Any:
    """Build a Draft 2020-12 validator and normalize schema-loading failures."""
    try:
        import jsonschema
    except ImportError as exc:
        raise error_cls("jsonschema dependency is required for schema validation") from exc

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise error_cls(f"invalid {label} schema: {exc.message}") from exc

    return jsonschema.Draft202012Validator(schema)
