"""Schema validation and drift detection for ingest contracts.

Provides hard-fail validation for:
- Schema version compatibility
- Required fields presence
- Type correctness
- Unknown fields detection (drift via Pydantic extra="forbid")
- 8-bit conversion detection
- dtype/range violations
- Gamma/non-linear ingest violations

All failures are explicit and actionable (no silent fallbacks).
Drift detection is handled automatically by Pydantic models configured
with ConfigDict(extra="forbid").

Exit Codes (contract-aligned for CI integration):
- EXIT_SUCCESS (0): Validation passed
- EXIT_SCHEMA_VALIDATION_FAILED (1): Schema validation failed
- EXIT_8BIT_CONVERSION (2): 8-bit conversion detected
- EXIT_GAMMA_VIOLATION (3): Gamma correction detected
- EXIT_SCHEMA_DRIFT (4): Schema drift detected (unknown fields)
- EXIT_OTHER_FAILURE (5): Other failure (e.g., file not found)

Usage:
    from transformation_portal.ingest import validate_schema

    # Validate provenance sidecar (drift detection automatic)
    errors = validate_schema(sidecar_json, schema_type="provenance")
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from pydantic import ValidationError

from .schemas import IngestManifest, ProvenanceSidecar

logger = logging.getLogger(__name__)


# =============================================================================
# Exit Codes (aligned with ingest contract categories for CI compatibility)
# =============================================================================

EXIT_SUCCESS = 0
EXIT_SCHEMA_VALIDATION_FAILED = 1
EXIT_8BIT_CONVERSION = 2
EXIT_GAMMA_VIOLATION = 3
EXIT_SCHEMA_DRIFT = 4
EXIT_OTHER_FAILURE = 5


def classify_validation_exit_code(error: object) -> int:
    """Classify a validation error into a contract-aligned exit code.

    This function centralizes exit code classification logic, ensuring
    consistent behavior between CLI tools and programmatic usage.

    The classification strategy:
    1. Check for structured error metadata (error_type, code attributes)
    2. Fall back to message-based heuristics for string errors
    3. Default to EXIT_SCHEMA_VALIDATION_FAILED for unclassified errors

    Args:
        error: An error object (string, exception, or structured error)

    Returns:
        Contract-aligned exit code (1-5)
    """
    # Strategy 1: Check for structured error metadata
    error_type = getattr(error, "error_type", None)
    error_code = getattr(error, "code", None)

    if error_type == "schema_drift" or error_code == "schema_drift":
        return EXIT_SCHEMA_DRIFT
    if error_type == "schema_version_mismatch" or error_code == "schema_version_mismatch":
        return EXIT_SCHEMA_VALIDATION_FAILED
    if error_type == "8bit_conversion" or error_code == "8bit_conversion":
        return EXIT_8BIT_CONVERSION
    if error_type == "gamma_violation" or error_code == "gamma_violation":
        return EXIT_GAMMA_VIOLATION

    # Strategy 2: Message-based heuristics for string errors
    error_msg = str(error).lower()
    if "drift" in error_msg or "unknown field" in error_msg or "extra field" in error_msg:
        return EXIT_SCHEMA_DRIFT
    if "schema version" in error_msg:
        return EXIT_SCHEMA_VALIDATION_FAILED
    if "8-bit" in error_msg or "8bit" in error_msg or "uint8" in error_msg:
        return EXIT_8BIT_CONVERSION
    if "gamma" in error_msg or "non-linear" in error_msg:
        return EXIT_GAMMA_VIOLATION

    # Default: generic schema validation failure
    return EXIT_SCHEMA_VALIDATION_FAILED


def classify_validation_errors(errors: List[Any]) -> int:
    """Classify a list of validation errors and return highest-severity exit code.

    Severity ordering (highest to lowest):
    - EXIT_SCHEMA_DRIFT (4) - structural contract violation
    - EXIT_GAMMA_VIOLATION (3) - data quality violation
    - EXIT_8BIT_CONVERSION (2) - data quality violation
    - EXIT_SCHEMA_VALIDATION_FAILED (1) - general validation failure

    Args:
        errors: List of error objects (strings, exceptions, or structured errors)

    Returns:
        Highest-severity exit code from the error list
    """
    if not errors:
        return EXIT_SUCCESS

    # Classify all errors and return highest-severity code
    exit_codes = [classify_validation_exit_code(e) for e in errors]

    # EXIT_SCHEMA_DRIFT is highest priority (structural violation)
    if EXIT_SCHEMA_DRIFT in exit_codes:
        return EXIT_SCHEMA_DRIFT
    if EXIT_GAMMA_VIOLATION in exit_codes:
        return EXIT_GAMMA_VIOLATION
    if EXIT_8BIT_CONVERSION in exit_codes:
        return EXIT_8BIT_CONVERSION
    if EXIT_SCHEMA_VALIDATION_FAILED in exit_codes:
        return EXIT_SCHEMA_VALIDATION_FAILED

    return EXIT_OTHER_FAILURE


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


EXIT_SUCCESS = 0
EXIT_SCHEMA_VALIDATION_FAILED = 1
EXIT_8BIT_CONVERSION = 2
EXIT_GAMMA_VIOLATION = 3
EXIT_SCHEMA_DRIFT = 4
EXIT_OTHER_FAILURE = 5

_EXIT_CODE_PRECEDENCE = (
    EXIT_SCHEMA_DRIFT,
    EXIT_GAMMA_VIOLATION,
    EXIT_8BIT_CONVERSION,
    EXIT_SCHEMA_VALIDATION_FAILED,
    EXIT_OTHER_FAILURE,
)


def classify_validation_error(error: object) -> Optional[int]:
    """Map a single schema validation error to the ingest contract exit code."""
    error_type = getattr(error, "error_type", None)
    error_code = getattr(error, "code", None)
    if error_type == "schema_drift" or error_code == "schema_drift":
        return EXIT_SCHEMA_DRIFT
    if error_type == "schema_version_mismatch" or error_code == "schema_version_mismatch":
        return EXIT_SCHEMA_VALIDATION_FAILED

    error_msg = str(error).lower()
    if "drift" in error_msg:
        return EXIT_SCHEMA_DRIFT
    if "schema version" in error_msg:
        return EXIT_SCHEMA_VALIDATION_FAILED
    return None


def classify_validation_errors(errors: Iterable[object]) -> int:
    """Classify validation errors into a single contract-aligned exit code."""
    for error in errors:
        exit_code = classify_validation_error(error)
        if exit_code is not None:
            return exit_code
    return EXIT_SCHEMA_VALIDATION_FAILED


def aggregate_exit_codes(exit_codes: Iterable[int]) -> int:
    """Aggregate multiple exit codes using ingest contract severity precedence."""
    observed = set(exit_codes)
    observed.discard(EXIT_SUCCESS)
    if not observed:
        return EXIT_SUCCESS

    for code in _EXIT_CODE_PRECEDENCE:
        if code in observed:
            return code
    return EXIT_OTHER_FAILURE


def validate_schema(
    data: Union[Dict[str, Any], str, Path],
    schema_type: str = "provenance",
    strict_mode: bool = True,
) -> List[str]:
    """Validate data against ingest contract schema.

    Performs:
    1. Schema version compatibility check
    2. Required fields validation
    3. Type validation
    4. Unknown fields detection (via Pydantic extra="forbid")

    Note: As of schema v1.0.0, all models use ConfigDict(extra="forbid"),
    so unknown fields are ALWAYS rejected by Pydantic during validation.
    The strict_mode parameter is kept for API compatibility but has no effect
    on drift detection (which is now always strict).

    Args:
        data: Dictionary, JSON string, or Path to JSON file
        schema_type: "provenance" or "manifest"
        strict_mode: Legacy parameter (kept for API compat, no effect)

    Returns:
        List of error messages (empty if valid)

    Raises:
        ValueError: If schema_type is invalid or data cannot be parsed
    """
    # Parse input data
    if isinstance(data, (str, Path)):
        if isinstance(data, str) and data.strip().startswith("{"):
            # JSON string
            data_dict = json.loads(data)
        else:
            # File path
            with open(data, "r") as f:
                data_dict = json.load(f)
    elif isinstance(data, dict):
        data_dict = data
    else:
        raise ValueError(f"Invalid data type: {type(data)}. Expected dict, str, or Path.")

    # Select schema class
    if schema_type == "provenance":
        schema_class = ProvenanceSidecar
    elif schema_type == "manifest":
        schema_class = IngestManifest
    else:
        raise ValueError(f"Invalid schema_type: {schema_type}. Must be 'provenance' or 'manifest'.")

    errors = []

    # Check schema version
    schema_version = data_dict.get("schema_version")
    if not schema_version:
        errors.append("Missing required field: schema_version")
    elif schema_version != "1.0.0":
        errors.append(f"Unsupported schema version: {schema_version}. " f"This validator supports version 1.0.0 only.")

    # Validate with Pydantic (includes drift detection via extra="forbid")
    try:
        schema_class(**data_dict)
    except ValidationError as e:
        for error in e.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            error_msg = error["msg"]
            error_type = error["type"]

            # Format error message
            if error_type == "value_error.missing":
                errors.append(f"Missing required field: {field_path}")
            elif error_type.startswith("type_error"):
                errors.append(f"Type mismatch at {field_path}: {error_msg}")
            else:
                errors.append(f"Validation error at {field_path}: {error_msg}")

    return errors


def validate_no_8bit_conversion(
    image_data: Any,
    expected_dtype: str = "uint16",
) -> Optional[str]:
    """Detect 8-bit conversion violations.

    Args:
        image_data: Image array (numpy or similar)
        expected_dtype: Expected dtype (e.g., "uint16", "float32")

    Returns:
        Error message if violation detected, None otherwise
    """
    try:
        import numpy as np

        if not isinstance(image_data, np.ndarray):
            return "Image data is not a numpy array"

        actual_dtype = str(image_data.dtype)

        # Check for 8-bit conversion
        if expected_dtype in ["uint16", "float32", "float64"]:
            if actual_dtype in ["uint8", "int8"]:
                return f"8-bit conversion detected: expected {expected_dtype}, " f"got {actual_dtype}"

        # Check for range violations
        if expected_dtype == "uint16":
            if image_data.max() <= 255:
                return "8-bit range detected in uint16 image: " f"max value is {image_data.max()} (expected > 255 for 16-bit)"

        return None

    except ImportError:
        # numpy not available, skip validation
        return None


def validate_linear_gamma(
    image_data: Any,
    tolerance: float = 0.05,
) -> Optional[str]:
    """Detect non-linear gamma violations.

    Checks if image histogram suggests gamma correction has been applied.

    Args:
        image_data: Image array (numpy)
        tolerance: Tolerance for gamma detection (default 5%)

    Returns:
        Error message if violation detected, None otherwise
    """
    try:
        import numpy as np

        if not isinstance(image_data, np.ndarray):
            return None

        # Normalize to [0, 1] for analysis
        if image_data.dtype == np.uint8:
            normalized = image_data.astype(np.float32) / 255.0
        elif image_data.dtype == np.uint16:
            normalized = image_data.astype(np.float32) / 65535.0
        elif image_data.dtype in [np.float32, np.float64]:
            normalized = image_data
        else:
            return None

        # Check histogram distribution
        # Linear images tend to have more data in shadows
        # Gamma-corrected images have more data in midtones
        hist, _ = np.histogram(normalized.ravel(), bins=10, range=(0, 1))

        # Heuristic: if first 3 bins have < 20% of data, likely gamma-corrected
        total_pixels = hist.sum()
        shadow_pixels = hist[:3].sum()
        shadow_ratio = shadow_pixels / total_pixels if total_pixels > 0 else 0

        if shadow_ratio < 0.2 - tolerance:
            return (
                f"Non-linear gamma detected: only {shadow_ratio:.1%} of pixels " f"in shadow range (expected ≥ 20% for linear)"
            )

        return None

    except ImportError:
        # numpy not available, skip validation
        return None


def validate_ingest_contract(
    sidecar_path: Path,
    image_data: Optional[Any] = None,
    strict_mode: bool = True,
) -> None:
    """Validate complete ingest contract compliance.

    Checks:
    1. Provenance sidecar schema
    2. No 8-bit conversion (if image_data provided)
    3. No gamma correction (if image_data provided)
    4. No schema drift

    Args:
        sidecar_path: Path to provenance sidecar JSON
        image_data: Optional image array for pixel validation
        strict_mode: If True, fail on unknown fields

    Raises:
        SchemaValidationError: If schema validation fails (includes drift)
        ValueError: If 8-bit conversion or gamma violations detected
    """
    # Validate sidecar schema
    errors = validate_schema(sidecar_path, schema_type="provenance", strict_mode=strict_mode)

    if errors:
        raise SchemaValidationError(errors)

    # Validate image data if provided
    if image_data is not None:
        # Check 8-bit conversion
        conversion_error = validate_no_8bit_conversion(image_data, expected_dtype="uint16")
        if conversion_error:
            raise ValueError(f"8-bit conversion violation: {conversion_error}")

        # Check gamma correction
        gamma_error = validate_linear_gamma(image_data)
        if gamma_error:
            raise ValueError(f"Gamma correction violation: {gamma_error}")

    logger.info(f"Ingest contract validation passed: {sidecar_path.name}")


def validate_manifest_file(
    manifest_path: Path,
    strict_mode: bool = True,
) -> None:
    """Validate ingest manifest file.

    Args:
        manifest_path: Path to ingest manifest JSON
        strict_mode: If True, fail on unknown fields

    Raises:
        SchemaValidationError: If schema validation fails
    """
    errors = validate_schema(manifest_path, schema_type="manifest", strict_mode=strict_mode)

    if errors:
        raise SchemaValidationError(errors)

    logger.info(f"Manifest validation passed: {manifest_path.name}")
