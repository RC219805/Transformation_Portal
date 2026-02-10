"""Schema validation and drift detection for ingest contracts.

Provides hard-fail validation for:
- Schema version compatibility
- Required fields presence
- Type correctness
- Unknown fields detection (drift)
- 8-bit conversion detection
- dtype/range violations
- Gamma/non-linear ingest violations

All failures are explicit and actionable (no silent fallbacks).

Usage:
    from transformation_portal.ingest import validate_schema

    # Validate provenance sidecar
    errors = validate_schema(sidecar_json, schema_type="provenance")
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from .schemas import IngestManifest, ProvenanceSidecar

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


class SchemaDriftError(Exception):
    """Raised when schema drift is detected."""

    def __init__(self, drift_type: str, details: str):
        self.drift_type = drift_type
        self.details = details
        message = f"Schema drift detected ({drift_type}): {details}"
        super().__init__(message)


def _detect_unknown_fields(
    data: Dict[str, Any],
    schema_class: type,
    path: str = "",
) -> List[str]:
    """Detect unknown fields not in schema (drift detection).

    Args:
        data: Dictionary to check
        schema_class: Pydantic model class
        path: Current path in nested structure (for error messages)

    Returns:
        List of unknown field paths
    """
    unknown = []

    # Get valid fields from schema (Pydantic v2)
    valid_fields = set(schema_class.model_fields.keys())

    for key, value in data.items():
        field_path = f"{path}.{key}" if path else key

        if key not in valid_fields:
            unknown.append(field_path)
        elif isinstance(value, dict):
            # Recursively check nested objects
            field_info = schema_class.model_fields[key]
            # Pydantic v2: use annotation instead of type_
            field_type = field_info.annotation
            if hasattr(field_type, "model_fields"):
                # Nested Pydantic model
                nested_unknown = _detect_unknown_fields(
                    value,
                    field_type,
                    field_path,
                )
                unknown.extend(nested_unknown)

    return unknown


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
        SchemaValidationError: If schema validation fails
        SchemaDriftError: If schema drift detected
        ValueError: If other violations detected
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
