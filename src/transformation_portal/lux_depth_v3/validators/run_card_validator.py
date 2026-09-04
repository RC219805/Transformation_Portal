"""Run card validation for lux_depth_v3 pipeline.

Extracted from orchestrator.py as part of ADR-043 decomposition.

This module provides:
- JSON Schema validation for run cards (Draft2020-12)
- Backend resolution semantics validation
- Deterministic error reporting with path context

The run card is the primary audit artifact for pipeline executions,
containing provenance, timing, backend selection, and artifact manifests.

Usage:
    from transformation_portal.lux_depth_v3.validators import (
        RunCardValidator,
        validate_run_card_payload,
        validate_run_card_backend_semantics,
    )

    validator = RunCardValidator()
    result = validator.validate(payload)  # Returns ValidationResult
    if not result:
        raise RunCardValidationError(result.errors)

    # Or use validate_or_raise for throwing behavior:
    validator.validate_or_raise(payload)  # Raises RunCardValidationError on failure

    # Or use standalone functions for specific validation phases:
    validate_run_card_payload(payload, schema_path)
    validate_run_card_backend_semantics(payload)
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformation_portal.lux_depth_v3.artifact_tree import (
    MAX_ARTIFACT_TREE_LEAVES,
    MAX_ARTIFACT_TREE_PROOF_DEPTH,
)
from transformation_portal.lux_depth_v3.run_card_contract import (
    infer_run_card_version,
    with_inferred_run_card_version,
)
from transformation_portal.schemas.run_card import get_run_card_schema_path, load_run_card_schema

from .jsonschema_formats import build_jsonschema_format_checker
from .run_card_backend_semantics import collect_run_card_backend_semantic_errors

JSONSCHEMA_REQUIREMENT = "jsonschema>=4.21.0,<5"
JSONSCHEMA_INSTALL_HINT = (
    "jsonschema dependency is required for run card schema validation "
    f"({JSONSCHEMA_REQUIREMENT}); install the core runtime with "
    "`make install-core` or install dependencies from requirements/base.in"
)
_MAX_RUN_CARD_GENERIC_COLLECTION_ITEMS = 4_096
_MAX_RUN_CARD_MAPPING_ITEMS = 4_096
_MAX_RUN_CARD_NESTING_DEPTH = 64
_MAX_RUN_CARD_INTEGER_BITS = 4_096
_V2_ARTIFACT_TREE_LIMIT = MAX_ARTIFACT_TREE_LEAVES
_V2_ARTIFACT_TREE_PROOF_DEPTH_LIMIT = MAX_ARTIFACT_TREE_PROOF_DEPTH
_MAX_RUN_CARD_SCHEMA_BYTES = 4 * 1024 * 1024
_MAX_VALIDATION_ERROR_CHARS = 1_024
_MAX_VALIDATION_PATH_CHARS = 256
_MAX_VALIDATION_DETAIL_CHARS = 640


class RunCardValidationError(RuntimeError):
    """Raised when run card validation fails.

    Attributes:
        code: Error classification code (schema_error, semantics_error, etc.)
        errors: List of individual validation error messages
        details: Additional context for debugging
    """

    def __init__(
        self,
        message: str,
        code: str = "validation_error",
        errors: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.errors = errors or []
        self.details = details or {}
        super().__init__(message)


def _default_schema_path(version: str = "v1") -> Path:
    """Return the actual installed package path for a run-card schema."""
    normalized_version = infer_run_card_version({"run_card_version": version})
    return get_run_card_schema_path(normalized_version)


@lru_cache(maxsize=4)
def _load_schema_bytes_from_path(schema_path_str: str) -> bytes:
    """Cache immutable bytes for an explicit schema path override."""
    schema_path = Path(schema_path_str)
    with open(schema_path, "rb") as schema_file:
        payload = schema_file.read(_MAX_RUN_CARD_SCHEMA_BYTES + 1)
    if len(payload) > _MAX_RUN_CARD_SCHEMA_BYTES:
        raise RuntimeError(
            "Run card schema exceeds the bounded byte limit of " f"{_MAX_RUN_CARD_SCHEMA_BYTES} bytes: {schema_path}"
        )
    return payload


def _load_schema_from_path(schema_path_str: str) -> Dict[str, Any]:
    """Load a fresh JSON schema from an explicit path override."""
    return json.loads(_load_schema_bytes_from_path(schema_path_str))


def _truncate_validation_text(value: Any, *, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_schema_error(error: Any) -> str:
    path = ".".join(_truncate_validation_text(part, limit=64) for part in error.path) or "<root>"
    path = _truncate_validation_text(path, limit=_MAX_VALIDATION_PATH_CHARS)
    validator_name = _truncate_validation_text(error.validator or "unknown", limit=64)
    detail = _truncate_validation_text(error.message, limit=_MAX_VALIDATION_DETAIL_CHARS)
    return _truncate_validation_text(
        f"Schema validation failed for run card at {path} [{validator_name}]: {detail}",
        limit=_MAX_VALIDATION_ERROR_CHARS,
    )


def _format_collection_path(path: tuple[str | int, ...]) -> str:
    if not path:
        return "<root>"
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += ("." if rendered else "") + part
    return rendered


def _collection_limit(path: tuple[str | int, ...]) -> int:
    if path == ("artifact_index",):
        return MAX_ARTIFACT_TREE_LEAVES
    if path in (("artifact_tree", "artifacts"), ("artifact_tree", "proofs")):
        return _V2_ARTIFACT_TREE_LIMIT
    if (
        len(path) == 4
        and path[0] == "artifact_tree"
        and path[1] == "proofs"
        and isinstance(path[2], int)
        and path[3] == "path"
    ):
        return _V2_ARTIFACT_TREE_PROOF_DEPTH_LIMIT
    return _MAX_RUN_CARD_GENERIC_COLLECTION_ITEMS


def _validate_bounded_collections(payload: Dict[str, Any], *, schema_version: str) -> None:
    """Bound every traversed container before invoking jsonschema.

    The published v1 schema remains unchanged for compatibility. These are
    operational resource limits applied to both schema versions so malformed
    in-memory payloads cannot induce an unbounded schema walk.
    """

    def visit(value: Any, *, path: tuple[str | int, ...], depth: int) -> None:
        if depth > _MAX_RUN_CARD_NESTING_DEPTH:
            raise RuntimeError(
                "Run card validation failed: JSON nesting exceeds the bounded limit of "
                f"{_MAX_RUN_CARD_NESTING_DEPTH} at {_format_collection_path(path)}"
            )
        if type(value) is int and value.bit_length() > _MAX_RUN_CARD_INTEGER_BITS:
            raise RuntimeError(
                "Run card validation failed: integer exceeds the bounded bit-length limit of "
                f"{_MAX_RUN_CARD_INTEGER_BITS} at {_format_collection_path(path)}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("Run card validation failed: non-finite number at " f"{_format_collection_path(path)}")
        if isinstance(value, dict):
            if len(value) > _MAX_RUN_CARD_MAPPING_ITEMS:
                raise RuntimeError(
                    f"Run card validation failed: {_format_collection_path(path)} mapping exceeds the bounded "
                    f"limit of {_MAX_RUN_CARD_MAPPING_ITEMS}"
                )
            for key, child in value.items():
                if not isinstance(key, str):
                    raise RuntimeError(
                        "Run card validation failed: mapping keys must be strings at " f"{_format_collection_path(path)}"
                    )
                visit(child, path=(*path, key), depth=depth + 1)
            return
        if not isinstance(value, list):
            return
        maximum = _collection_limit(path)
        if len(value) > maximum:
            raise RuntimeError(
                f"Run card validation failed: {_format_collection_path(path)} exceeds the bounded limit of {maximum}"
            )
        for index, child in enumerate(value):
            visit(child, path=(*path, index), depth=depth + 1)

    visit(payload, path=(), depth=0)

    if schema_version != "v2":
        return
    artifact_tree = payload.get("artifact_tree")
    if not isinstance(artifact_tree, dict):
        return
    leaf_count = artifact_tree.get("leaf_count")
    if type(leaf_count) is int and leaf_count > _V2_ARTIFACT_TREE_LIMIT:
        raise RuntimeError(
            "Run card validation failed: artifact_tree.leaf_count " f"exceeds the bounded limit of {_V2_ARTIFACT_TREE_LIMIT}"
        )


def _build_validator(schema: Dict[str, Any]) -> Any:
    """Build a Draft 2020-12 validator with format assertions enabled."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            JSONSCHEMA_INSTALL_HINT,
        ) from exc

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise RuntimeError(
            _truncate_validation_text(
                f"Run card schema is invalid: {exc.message}",
                limit=_MAX_VALIDATION_ERROR_CHARS,
            ),
        ) from exc
    return jsonschema.Draft202012Validator(
        schema,
        format_checker=build_jsonschema_format_checker(),
    )


@lru_cache(maxsize=4)
def _load_validator(schema_path_str: Optional[str], schema_version: Optional[str]) -> Any:
    """Load a cached validator from either a schema path or packaged version."""
    if schema_path_str is not None:
        schema = _load_schema_from_path(schema_path_str)
    else:
        schema = load_run_card_schema(schema_version or "v1")
    return _build_validator(schema)


def validate_run_card_payload(
    payload: Dict[str, Any],
    schema_path: Optional[Path] = None,
    *,
    schema_version: Optional[str] = None,
) -> None:
    """Validate run card payload against the appropriate run-card schema.

    When ``schema_path`` is omitted, validation uses packaged runtime schemas and
    infers the version from ``run_card_version`` or legacy v1/v2 structure.

    Args:
        payload: Run card dictionary to validate
        schema_path: Optional explicit JSON schema override
        schema_version: Optional explicit schema version when no path override is provided

    Raises:
        RuntimeError: If validation fails, reporting the first bounded error
    """
    payload_for_validation = with_inferred_run_card_version(payload)
    resolved_schema_version = (
        infer_run_card_version(payload_for_validation)
        if schema_version is None
        else infer_run_card_version({"run_card_version": schema_version})
    )
    _validate_bounded_collections(payload_for_validation, schema_version=resolved_schema_version)
    validator = _load_validator(str(schema_path) if schema_path is not None else None, resolved_schema_version)
    first_error = next(validator.iter_errors(payload_for_validation), None)
    if first_error is not None:
        raise RuntimeError(_format_schema_error(first_error))


def validate_run_card_backend_semantics(payload: Dict[str, Any]) -> None:
    """Validate backend resolution semantics for run-card transparency.

    This validation ensures consistency between:
    - backend_selection.resolved
    - backend_summary.final_backends_used
    - backend_summary.primary_backend
    - Wrapper semantics (logical_backend vs resolved_engine)

    The semantics rules are:
    1. If success_count > 0, final_backends_used must be non-empty
    2. final_backends_used[0] must match primary_backend
    3. resolved must match final_backends_used[0]
    4. For wrapper backends: logical_backend != resolved_engine
    5. Wrapper semantics require fallback_images == 0

    Args:
        payload: Run card dictionary to validate

    Raises:
        RuntimeError: If backend semantics are inconsistent
    """
    errors = collect_run_card_backend_semantic_errors(payload)
    if errors:
        details = "; ".join(_truncate_validation_text(error, limit=256) for error in errors[:8])
        if len(errors) > 8:
            details += f"; ... {len(errors) - 8} additional errors"
        raise RuntimeError(
            _truncate_validation_text(
                "Run card backend semantics validation failed: " + details,
                limit=_MAX_VALIDATION_ERROR_CHARS,
            )
        )


class RunCardValidator:
    """Unified run card validation interface.

    Provides a single entry point for all run card validation phases:
    1. Schema validation (JSON Schema Draft2020-12)
    2. Backend semantics validation (consistency rules)

    This class is the primary interface for run card validation per ADR-043.

    Example:
        validator = RunCardValidator()
        result = validator.validate(run_card_payload)
        if not result.is_valid:
            print(result.errors)

        # Or with exception on failure:
        validator.validate_or_raise(run_card_payload)
    """

    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize validator with optional custom schema path.

        Args:
            schema_path: Path to JSON schema file. Defaults to repository schema.
        """
        self._schema_path = schema_path

    @property
    def schema_path(self) -> Path:
        """Return the schema path used for validation."""
        return self._schema_path or _default_schema_path()

    def validate_payload(self, payload: Dict[str, Any]) -> None:
        """Validate payload against JSON schema.

        Args:
            payload: Run card dictionary to validate

        Raises:
            RuntimeError: If schema validation fails
        """
        validate_run_card_payload(payload, self._schema_path)

    def validate_backend_semantics(self, payload: Dict[str, Any]) -> None:
        """Validate backend resolution semantics.

        Args:
            payload: Run card dictionary to validate

        Raises:
            RuntimeError: If semantics validation fails
        """
        validate_run_card_backend_semantics(payload)

    def validate(self, payload: Dict[str, Any]) -> "ValidationResult":
        """Validate run card payload (non-throwing).

        Performs both schema and semantics validation, collecting all errors.

        Args:
            payload: Run card dictionary to validate

        Returns:
            ValidationResult with is_valid flag and collected errors
        """
        errors: List[str] = []

        try:
            self.validate_payload(payload)
        except RuntimeError as exc:
            errors.append(str(exc))

        try:
            self.validate_backend_semantics(payload)
        except RuntimeError as exc:
            errors.append(str(exc))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def validate_or_raise(self, payload: Dict[str, Any]) -> None:
        """Validate run card payload (throwing).

        Performs both schema and semantics validation, raising on first error.

        Args:
            payload: Run card dictionary to validate

        Raises:
            RunCardValidationError: If any validation fails
        """
        result = self.validate(payload)
        if not result.is_valid:
            raise RunCardValidationError(
                "Run card validation failed",
                code="validation_error",
                errors=result.errors,
            )


class ValidationResult:
    """Result of run card validation.

    Attributes:
        is_valid: True if all validation passed
        errors: List of error messages (empty if valid)
    """

    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []

    def __bool__(self) -> bool:
        """Allow truthiness check for validation result."""
        return self.is_valid

    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self.is_valid}, errors={self.errors})"
