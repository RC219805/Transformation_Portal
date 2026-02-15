"""Validation guardrails for Spatial AI linear ingest pipeline.

These validators enforce hard constraints on training data quality:
- Bit depth requirements (no lossy 8-bit collapse)
- Range validation (no NaN/Inf, non-negative)
- Dtype enforcement (float32 only)
- Schema version compatibility

All validators fail fast with clear, actionable error messages.

Architecture: ADR-023 (Isolation), Issue #890 (Phase I)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .exceptions import BitDepthViolationError, LinearityViolationError, RangeViolationError, SchemaVersionError

logger = logging.getLogger(__name__)


# Supported schema versions (semantic versioning)
SUPPORTED_SCHEMA_VERSIONS = ["1.0.0", "1.0.1"]  # Can add compatible versions here
CURRENT_SCHEMA_VERSION = "1.0.0"


def validate_bit_depth(
    input_path: Path,
    array: np.ndarray,
    min_bits: int = 16,
    strict: bool = False,
) -> None:
    """Validate input bit depth is sufficient for linear ingest.

    Args:
        input_path: Path to input file (for error context).
        array: NumPy array from decoded input.
        min_bits: Minimum required bit depth.
        strict: If True, raises error on violation. If False, logs warning.

    Raises:
        BitDepthViolationError: If strict=True and bit depth < min_bits.

    Example:
        >>> img = np.array(Image.open("test.png"))  # uint8
        >>> validate_bit_depth(Path("test.png"), img, strict=True)
        BitDepthViolationError: test.png is uint8 (8-bit), but requires ≥16-bit
    """
    dtype_str = str(array.dtype)
    inferred_bits = _infer_bit_depth_from_dtype(dtype_str)

    if inferred_bits < min_bits:
        if strict:
            raise BitDepthViolationError(
                input_path=input_path,
                detected_dtype=dtype_str,
                min_required_bits=min_bits,
            )
        else:
            logger.warning(
                f"Bit depth below recommended minimum: {input_path.name} is {dtype_str} "
                f"({inferred_bits}-bit), recommended ≥{min_bits}-bit. "
                f"Set strict_ingest=True to enforce this as a hard requirement."
            )


def validate_dtype(
    array: np.ndarray,
    expected_dtype: np.dtype = np.dtype(np.float32),
    field_name: str = "output",
) -> None:
    """Validate array dtype matches expected type.

    Args:
        array: NumPy array to validate.
        expected_dtype: Expected dtype (default: float32).
        field_name: Name of field for error messages.

    Raises:
        LinearityViolationError: If dtype doesn't match expected.

    Example:
        >>> arr = np.zeros((10, 10, 3), dtype=np.float16)
        >>> validate_dtype(arr, np.dtype(np.float32))
        LinearityViolationError: output dtype must be float32, got float16
    """
    if array.dtype != expected_dtype:
        raise LinearityViolationError(
            field=f"{field_name}.dtype",
            expected=str(expected_dtype),
            actual=str(array.dtype),
        )


def validate_gamma(
    gamma: float,
    expected: float = 1.0,
    tolerance: float = 1e-6,
) -> None:
    """Validate gamma value is linear (1.0).

    Args:
        gamma: Gamma value to validate.
        expected: Expected gamma value (default: 1.0 for linear).
        tolerance: Floating point comparison tolerance.

    Raises:
        LinearityViolationError: If gamma is not within tolerance of expected.

    Example:
        >>> validate_gamma(2.2)
        LinearityViolationError: gamma must be 1.0, got 2.2
    """
    if abs(gamma - expected) > tolerance:
        raise LinearityViolationError(
            field="gamma",
            expected=expected,
            actual=gamma,
        )


def validate_range(
    array: np.ndarray,
    allow_negative: bool = False,
    allow_above_one: bool = True,
    check_nan: bool = True,
    check_inf: bool = True,
) -> None:
    """Validate array values are within acceptable range.

    For linear light training data:
    - NaN: FORBIDDEN (indicates corruption/failure)
    - Inf: FORBIDDEN (indicates overflow/failure)
    - Negative: FORBIDDEN (non-physical for light intensity)
    - >1.0: ALLOWED (HDR preservation required)

    Args:
        array: NumPy array to validate.
        allow_negative: If False, raises error on negative values.
        allow_above_one: If False, raises error on values >1.0 (not recommended).
        check_nan: If True, check for NaN values.
        check_inf: If True, check for infinite values.

    Raises:
        RangeViolationError: If values violate range constraints.

    Example:
        >>> arr = np.array([[[0.5, 1.2, np.nan]]])  # HDR with NaN
        >>> validate_range(arr)
        RangeViolationError: NaN values detected
    """
    min_val = float(np.min(array))
    max_val = float(np.max(array))
    has_nan = False
    has_inf = False

    if check_nan:
        has_nan = bool(np.isnan(array).any())

    if check_inf:
        has_inf = bool(np.isinf(array).any())

    # Check for violations
    violations = []

    if has_nan:
        violations.append("NaN")
    if has_inf:
        violations.append("Inf")
    if not allow_negative and min_val < 0:
        violations.append("negative")
    if not allow_above_one and max_val > 1.0:
        violations.append(">1.0")

    if violations:
        raise RangeViolationError(
            min_value=min_val,
            max_value=max_val,
            has_nan=has_nan,
            has_inf=has_inf,
        )


def validate_shape(
    array: np.ndarray,
    expected_ndim: int = 3,
    expected_channels: int = 3,
) -> None:
    """Validate array shape is correct for RGB images.

    Args:
        array: NumPy array to validate.
        expected_ndim: Expected number of dimensions (default: 3 for H×W×C).
        expected_channels: Expected number of channels (default: 3 for RGB).

    Raises:
        ValueError: If shape is incorrect.

    Example:
        >>> arr = np.zeros((100, 100, 4))  # RGBA
        >>> validate_shape(arr, expected_channels=3)
        ValueError: Expected 3 channels, got 4
    """
    if array.ndim != expected_ndim:
        raise ValueError(f"Expected {expected_ndim}D array, got {array.ndim}D array with shape {array.shape}")

    if expected_ndim == 3 and array.shape[2] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {array.shape[2]} (shape: {array.shape})")


def validate_schema_version(
    manifest: Dict[str, Any],
    manifest_path: Path,
    supported_versions: list[str] = None,
) -> None:
    """Validate manifest schema version is supported.

    Args:
        manifest: Manifest dictionary.
        manifest_path: Path to manifest file (for error context).
        supported_versions: List of supported versions (default: SUPPORTED_SCHEMA_VERSIONS).

    Raises:
        SchemaVersionError: If version is not supported.
        KeyError: If 'schema_version' field is missing.

    Example:
        >>> manifest = {"schema_version": "2.0.0", "data": []}
        >>> validate_schema_version(manifest, Path("manifest.json"))
        SchemaVersionError: manifest.json has version '2.0.0', but only [...] are supported
    """
    if supported_versions is None:
        supported_versions = SUPPORTED_SCHEMA_VERSIONS

    # Check for version field
    if "schema_version" not in manifest:
        raise KeyError(
            f"Manifest {manifest_path.name} is missing required 'schema_version' field. "
            f'Add: {{"schema_version": "{CURRENT_SCHEMA_VERSION}", ...}}'
        )

    found_version = manifest["schema_version"]

    # Validate version
    if found_version not in supported_versions:
        raise SchemaVersionError(
            manifest_path=manifest_path,
            found_version=found_version,
            supported_versions=supported_versions,
        )


def validate_provenance_completeness(
    provenance: Dict[str, Any],
    required_fields: list[str] = None,
) -> None:
    """Validate provenance metadata contains all required fields.

    Args:
        provenance: Provenance dictionary.
        required_fields: List of required top-level fields.
            Default: ["input", "decode", "output"]

    Raises:
        ValueError: If required fields are missing.

    Example:
        >>> prov = {"input": {...}, "decode": {...}}  # Missing "output"
        >>> validate_provenance_completeness(prov)
        ValueError: Missing required provenance fields: ['output']
    """
    if required_fields is None:
        required_fields = ["input", "decode", "output"]

    missing = [field for field in required_fields if field not in provenance]

    if missing:
        raise ValueError(f"Missing required provenance fields: {missing}. " f"Provenance must include: {required_fields}")


def validate_linear_output(
    array: np.ndarray,
    gamma: float,
    input_path: Path,
    strict_bit_depth: bool = False,
) -> None:
    """Comprehensive validation of linear ingest output.

    Combines all validation checks for linear training data:
    - Gamma = 1.0 (linear light)
    - Dtype = float32 (precision)
    - No NaN/Inf (valid data)
    - Non-negative (physical light)
    - Proper shape (H×W×3)

    Args:
        array: Output array to validate.
        gamma: Gamma value used for decode.
        input_path: Input file path (for error context).
        strict_bit_depth: If True, enforce bit depth validation.

    Raises:
        LinearityViolationError: If linearity constraints violated.
        RangeViolationError: If range constraints violated.
        ValueError: If shape constraints violated.

    Example:
        >>> img = decode_to_linear("test.tiff")
        >>> validate_linear_output(img, gamma=1.0, input_path=Path("test.tiff"))
        # Passes if all constraints satisfied
    """
    # Gamma validation
    validate_gamma(gamma, expected=1.0)

    # Dtype validation
    validate_dtype(array, expected_dtype=np.dtype(np.float32))

    # Range validation
    validate_range(
        array,
        allow_negative=False,
        allow_above_one=True,  # HDR preservation
        check_nan=True,
        check_inf=True,
    )

    # Shape validation
    validate_shape(array, expected_ndim=3, expected_channels=3)

    # Optional bit depth validation (if original array available)
    # This is handled at decode time, not at validation time


# Helper functions


def _infer_bit_depth_from_dtype(dtype_str: str) -> int:
    """Infer bit depth from NumPy dtype string.

    Args:
        dtype_str: NumPy dtype as string (e.g., "uint8", "float32").

    Returns:
        Bit depth (8, 16, 32, 64), or 0 if unknown.
    """
    dtype_str = dtype_str.lower()

    if "8" in dtype_str:
        return 8
    elif "16" in dtype_str:
        return 16
    elif "32" in dtype_str:
        return 32
    elif "64" in dtype_str:
        return 64
    else:
        return 0  # Unknown


def get_current_schema_version() -> str:
    """Get current manifest schema version.

    Returns:
        Current schema version string (semantic versioning).
    """
    return CURRENT_SCHEMA_VERSION


def is_schema_version_supported(version: str) -> bool:
    """Check if schema version is supported.

    Args:
        version: Schema version string to check.

    Returns:
        True if version is supported, False otherwise.
    """
    return version in SUPPORTED_SCHEMA_VERSIONS
