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
    """Return repository-local run card schema path.

    The schema is located at:
    <repo_root>/docs/schemas/run_card/run_card.<version>.schema.json

    This path is resolved relative to this module's location in the
    installed package structure.
    """
    normalized_version = str(version or "v1").strip().lower()
    if normalized_version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported run card schema version: {version!r}")
    return Path(__file__).resolve().parents[4] / "docs" / "schemas" / "run_card" / f"run_card.{normalized_version}.schema.json"


@lru_cache(maxsize=1)
def _load_schema(schema_path_str: str) -> Dict[str, Any]:
    """Load run card JSON schema once per process.

    Args:
        schema_path_str: String path for hashability (LRU cache key)

    Returns:
        Parsed JSON schema dictionary

    Raises:
        FileNotFoundError: If schema file does not exist
        json.JSONDecodeError: If schema is not valid JSON
    """
    schema_path = Path(schema_path_str)
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


@lru_cache(maxsize=1)
def _load_validator(schema_path_str: str) -> Any:
    """Build cached Draft202012 validator for run card schema.

    Uses the jsonschema library for JSON Schema Draft2020-12 validation.
    The validator instance is cached per schema path.

    Args:
        schema_path_str: String path for hashability (LRU cache key)

    Returns:
        jsonschema.Draft202012Validator instance

    Raises:
        RuntimeError: If jsonschema is not installed or schema is invalid
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "jsonschema dependency is required for run card schema validation",
        ) from exc

    schema = _load_schema(schema_path_str)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise RuntimeError(
            f"Run card schema is invalid: {exc.message}",
        ) from exc
    return jsonschema.Draft202012Validator(schema)


def validate_run_card_payload(
    payload: Dict[str, Any],
    schema_path: Optional[Path] = None,
) -> None:
    """Validate run card payload against run_card.v1 schema.

    Validates the payload structure and types against the JSON Schema.
    On validation failure, raises RuntimeError with formatted error messages.

    Args:
        payload: Run card dictionary to validate
        schema_path: Path to JSON schema file. Defaults to repository schema.

    Raises:
        RuntimeError: If validation fails, with concatenated error messages
        FileNotFoundError: If schema file does not exist
    """
    if schema_path is None:
        schema_path = _default_schema_path()

    validator = _load_validator(str(schema_path))
    errors = sorted(
        validator.iter_errors(payload),
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
    backend_selection = payload.get("backend_selection")
    backend_summary = payload.get("backend_summary")
    if not isinstance(backend_selection, dict) or not isinstance(backend_summary, dict):
        return

    final_backends_used = backend_summary.get("final_backends_used")
    if not isinstance(final_backends_used, list):
        return

    success_count = payload.get("success_count")
    if not isinstance(success_count, int):
        success_count = 0

    if not final_backends_used:
        if success_count > 0:
            raise RuntimeError(
                "Run card backend semantics validation failed: "
                "backend_summary.final_backends_used must be "
                "non-empty when success_count > 0."
            )
        return

    primary_backend = final_backends_used[0]
    if not isinstance(primary_backend, str) or not primary_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_summary.final_backends_used[0] must be a non-empty string."
        )

    summary_primary = backend_summary.get("primary_backend")
    if summary_primary != primary_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_summary.primary_backend must equal "
            "backend_summary.final_backends_used[0]."
        )

    resolved = backend_selection.get("resolved")
    if not isinstance(resolved, str) or not resolved:
        raise RuntimeError(
            "Run card backend semantics validation failed: " "backend_selection.resolved must be a non-empty string."
        )

    if resolved != primary_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.resolved must match "
            "backend_summary.final_backends_used[0]."
        )

    requested_backend = backend_selection.get("requested") or backend_summary.get("requested_backend")
    fallback_images = backend_summary.get("fallback_images")
    if (
        requested_backend == "depth_pro"
        and isinstance(fallback_images, int)
        and success_count > 0
        and fallback_images == success_count
        and primary_backend != requested_backend
    ):
        raise RuntimeError(
            "Run card backend request fulfillment validation failed: "
            "requested backend 'depth_pro' was not honored; "
            f"all successful images used fallback backend '{primary_backend}'."
        )

    # Wrapper semantics validation
    logical_backend = backend_selection.get("logical_backend")
    resolved_engine = backend_selection.get("resolved_engine")
    wrapper_declared = logical_backend is not None or resolved_engine is not None
    if not wrapper_declared:
        return

    if not isinstance(logical_backend, str) or not logical_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.logical_backend must be a non-empty string "
            "when wrapper semantics are declared."
        )
    if not isinstance(resolved_engine, str) or not resolved_engine:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.resolved_engine must be a non-empty string "
            "when wrapper semantics are declared."
        )
    if logical_backend == resolved_engine:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.logical_backend and "
            "backend_selection.resolved_engine must differ."
        )
    if resolved_engine != primary_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.resolved_engine must match "
            "backend_summary.final_backends_used[0]."
        )

    if isinstance(fallback_images, int) and fallback_images != 0:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "wrapper semantics are only valid when "
            "backend_summary.fallback_images == 0."
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
        self._schema_path = schema_path or _default_schema_path()

    @property
    def schema_path(self) -> Path:
        """Return the schema path used for validation."""
        return self._schema_path

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
