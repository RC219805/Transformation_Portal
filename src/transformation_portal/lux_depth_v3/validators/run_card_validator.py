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
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformation_portal.lux_depth_v3.run_card_contract import (
    infer_run_card_version,
    with_inferred_run_card_version,
)
from transformation_portal.schemas.run_card import load_run_card_schema

from .jsonschema_formats import build_jsonschema_format_checker
from .run_card_backend_semantics import collect_run_card_backend_semantic_errors


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
    """Return the published documentation path for a run-card schema.

    Runtime validation loads packaged schema resources. This helper is kept for
    legacy imports, documentation sync tests, and CLI help text.
    """
    normalized_version = infer_run_card_version({"run_card_version": version})
    return Path(__file__).resolve().parents[4] / "docs" / "schemas" / "run_card" / f"run_card.{normalized_version}.schema.json"


@lru_cache(maxsize=4)
def _load_schema_from_path(schema_path_str: str) -> Dict[str, Any]:
    """Load a JSON schema from an explicit path override."""
    schema_path = Path(schema_path_str)
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _build_validator(schema: Dict[str, Any]) -> Any:
    """Build a Draft 2020-12 validator with format assertions enabled."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "jsonschema dependency is required for run card schema validation",
        ) from exc

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise RuntimeError(
            f"Run card schema is invalid: {exc.message}",
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
        RuntimeError: If validation fails, with concatenated error messages
    """
    payload_for_validation = with_inferred_run_card_version(payload)
    resolved_schema_version = infer_run_card_version(payload_for_validation) if schema_version is None else schema_version
    validator = _load_validator(str(schema_path) if schema_path is not None else None, resolved_schema_version)
    errors = sorted(
        validator.iter_errors(payload_for_validation),
        key=lambda error: list(error.path),
    )
    if not errors:
        return

    formatted = []
    for error in errors:
        path = ".".join(str(p) for p in error.path) or "<root>"
        formatted.append(f"{path}: {error.message}")
    raise RuntimeError(
        "Run card schema validation failed: " + "; ".join(formatted),
    )


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
        raise RuntimeError("Run card backend semantics validation failed: " + "; ".join(errors))


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
