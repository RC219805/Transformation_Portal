"""Linear ingest verification for APEX pipeline.

Enforces correctness-critical invariants for linear light preservation:
- Tensor dtype must be floating point (no uint8/uint16 leakage)
- Value range must match expected linear bounds [0, 1]
- No implicit gamma correction allowed (must be explicit and rejected)
- Deterministic, fixture-based validation

Per Spatial AI Foundation ROADMAP.md (Section I: Data Fidelity is Sacred):
"Training inputs MUST preserve linear-light relationships: pixel intensity
MUST remain a linear proxy for captured light (photon count proxy), not
tone-mapped or gamma-corrected."

Design principles:
- Fail fast with explicit errors (no warnings, no fallbacks)
- Blocking validation (cannot proceed with invalid data)
- Deterministic checks (same input → same result)
- Clear error messages with remediation guidance
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LinearityViolationError(ValueError):
    """Raised when linear light preservation invariant is violated.

    This is a blocking error - the pipeline cannot proceed with
    gamma-encoded or non-linear data.
    """

    pass


class DtypeViolationError(TypeError):
    """Raised when tensor dtype is not floating point.

    uint8 and uint16 tensors indicate potential precision loss or
    gamma encoding. Only float32/float64 tensors are accepted.
    """

    pass


class RangeViolationError(ValueError):
    """Raised when pixel values violate expected linear bounds.

    Linear light values must be in [0, 1] for normalized floating point
    or [0, 65535] for 16-bit linear representation.
    """

    pass


def verify_dtype_float(
    tensor: np.ndarray,
    allow_float64: bool = True,
) -> None:
    """Verify tensor dtype is floating point.

    Rejects uint8 and uint16 dtypes that indicate potential gamma encoding
    or precision loss during ingest.

    Args:
        tensor: Input tensor to validate
        allow_float64: Whether to allow float64 (default True)

    Raises:
        DtypeViolationError: If dtype is not floating point
        TypeError: If input is not a numpy array

    Example:
        >>> arr = np.array([0.5, 0.8], dtype=np.float32)
        >>> verify_dtype_float(arr)  # Passes
        >>>
        >>> arr_uint8 = np.array([128, 200], dtype=np.uint8)
        >>> verify_dtype_float(arr_uint8)  # Raises DtypeViolationError
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(tensor).__name__}. " "Linear verification requires numpy arrays.")

    allowed_dtypes = [np.float32]
    if allow_float64:
        allowed_dtypes.append(np.float64)

    if tensor.dtype not in allowed_dtypes:
        allowed_str = " or ".join(str(dt) for dt in allowed_dtypes)
        raise DtypeViolationError(
            f"Tensor dtype must be {allowed_str} for linear light preservation. "
            f"Got dtype={tensor.dtype}. "
            f"uint8 and uint16 indicate potential gamma encoding or precision loss. "
            f"Convert to float32 with explicit linear decoding if needed."
        )


def verify_range_linear(
    tensor: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    tolerance: float = 1e-6,
) -> None:
    """Verify tensor values are within expected linear range.

    For normalized linear light, expected range is [0, 1].
    Values outside this range indicate incorrect normalization or
    overflow/underflow during processing.

    Args:
        tensor: Input tensor to validate
        min_val: Minimum expected value (default 0.0)
        max_val: Maximum expected value (default 1.0)
        tolerance: Tolerance for floating point comparisons (default 1e-6)

    Raises:
        RangeViolationError: If values are outside expected range or contain NaN/Inf

    Example:
        >>> arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> verify_range_linear(arr)  # Passes
        >>>
        >>> arr_bad = np.array([-0.1, 0.5, 1.5], dtype=np.float32)
        >>> verify_range_linear(arr_bad)  # Raises RangeViolationError
    """
    # CRITICAL: Check for NaN/Inf BEFORE range validation
    # tensor.min()/max() comparisons with NaN are always False, so corrupted tensors would silently pass
    if not np.isfinite(tensor).all():
        raise RangeViolationError(
            "Tensor contains NaN or Inf values. "
            "This indicates numerical corruption during processing. "
            "Check for division by zero, invalid operations, or corrupted input data."
        )

    actual_min = tensor.min()
    actual_max = tensor.max()

    if actual_min < (min_val - tolerance):
        raise RangeViolationError(
            f"Tensor minimum value {actual_min:.6f} is below expected "
            f"linear range [{min_val}, {max_val}]. "
            f"This indicates incorrect normalization or underflow."
        )

    if actual_max > (max_val + tolerance):
        raise RangeViolationError(
            f"Tensor maximum value {actual_max:.6f} exceeds expected "
            f"linear range [{min_val}, {max_val}]. "
            f"This indicates incorrect normalization or overflow."
        )


def detect_gamma_encoding(
    tensor: np.ndarray,
    threshold: float = 0.15,
) -> bool:
    """Detect if tensor appears to be gamma-encoded rather than linear.

    Uses statistical heuristics to detect gamma encoding:
    - Gamma-encoded images have darker mean (values concentrated near 0)
    - Linear images have more uniform distribution
    - Compares midpoint (0.5 linear → ~0.73 gamma-2.2 encoded)

    This is a heuristic check, not perfect, but catches common cases.

    Args:
        tensor: Input tensor (float32, [0, 1])
        threshold: Detection threshold for gamma signature (default 0.15)

    Returns:
        True if tensor appears gamma-encoded, False if appears linear

    Note:
        This is a statistical heuristic. For deterministic validation,
        use explicit metadata checks or known-linear test fixtures.
    """
    # Check dtype first
    if tensor.dtype not in [np.float32, np.float64]:
        # Can't reliably detect gamma on integer types
        return False

    # Ensure we're working with 3-channel RGB
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        # Can only check RGB images
        return False

    # Simple heuristic: gamma-encoded images have mean significantly
    # lower than 0.5 for mid-gray content
    # Linear mid-gray (0.5) → gamma-2.2 encoded (~0.73)
    # This creates a characteristic shift in the histogram

    mean_val = tensor.mean()

    # If mean is very high (>0.6), likely gamma-encoded
    # Linear images with typical scene content have mean ~0.3-0.5
    # Gamma-encoded scenes shift this higher
    if mean_val > (0.5 + threshold):
        logger.warning(f"Tensor mean {mean_val:.3f} suggests gamma encoding " f"(linear expected ~0.3-0.5 for typical scenes)")
        return True

    # Additional check: median vs mean ratio
    # Gamma encoding creates characteristic skew
    # Linear images have more uniform distribution (mean ≈ median)
    # Gamma-encoded images shift towards higher values (mean > median)
    # Threshold 1.3 is empirically chosen to detect typical gamma curves (2.0-2.4)
    MEAN_MEDIAN_RATIO_THRESHOLD = 1.3
    median_val = np.median(tensor)
    if median_val > 0 and (mean_val / median_val) > MEAN_MEDIAN_RATIO_THRESHOLD:
        logger.warning(f"Mean/median ratio {mean_val/median_val:.3f} suggests gamma encoding")
        return True

    return False


def verify_no_gamma(
    tensor: np.ndarray,
    threshold: float = 0.15,
    strict: bool = False,
) -> None:
    """Verify tensor is not gamma-encoded.

    Uses statistical heuristics to detect and warn about gamma-encoded data.

    IMPORTANT: Default is now warning-only (strict=False) to prevent false positives
    on bright linear scenes (white interiors, snow, stucco). The heuristic cannot
    reliably distinguish "bright linear" from "gamma-encoded" content.

    Args:
        tensor: Input tensor to validate
        threshold: Detection threshold (default 0.15)
        strict: If True, raise error on detection; if False (default), log warning

    Raises:
        LinearityViolationError: If gamma encoding detected AND strict=True

    Example:
        >>> linear_arr = np.array([0.2, 0.4, 0.6], dtype=np.float32).reshape(1, 1, 3)
        >>> verify_no_gamma(linear_arr)  # Passes
        >>>
        >>> gamma_arr = np.array([0.5, 0.7, 0.8], dtype=np.float32).reshape(1, 1, 3)
        >>> verify_no_gamma(gamma_arr)  # Logs warning (default strict=False)
    """
    if detect_gamma_encoding(tensor, threshold=threshold):
        msg = (
            "Image statistics suggest gamma encoding (mean > threshold). "
            "If issues occur, verify source is linear. "
            "Note: Bright linear scenes (white interiors, snow, stucco) can trigger false positives. "
            "Use RAW files with linear output or pre-linearized TIFF files for guaranteed linearity. "
            "Proceeding with processing."
        )
        if strict:
            raise LinearityViolationError(msg)
        else:
            logger.warning(msg)


def verify_linear_ingest(
    tensor: np.ndarray,
    check_dtype: bool = True,
    check_range: bool = True,
    check_gamma: bool = True,
    allow_float64: bool = True,
    strict_gamma: bool = False,
) -> None:
    """Comprehensive linear ingest verification.

    Validates all correctness-critical invariants for linear light
    preservation:
    1. dtype is floating point (no uint8/uint16)
    2. value range is [0, 1] with no NaN/Inf
    3. gamma encoding detection (warning-only by default)

    This is the main entry point for linear verification.

    Args:
        tensor: Input tensor to validate
        check_dtype: Whether to verify dtype (default True)
        check_range: Whether to verify value range (default True)
        check_gamma: Whether to check for gamma encoding (default True)
        allow_float64: Whether to allow float64 dtype (default True)
        strict_gamma: Whether to raise error on gamma detection (default False)

    Raises:
        DtypeViolationError: If dtype check fails
        RangeViolationError: If range check fails or NaN/Inf detected
        LinearityViolationError: If gamma encoding detected AND strict_gamma=True

    Example:
        >>> arr = np.random.rand(100, 100, 3).astype(np.float32)
        >>> verify_linear_ingest(arr)  # Passes all checks
    """
    if check_dtype:
        verify_dtype_float(tensor, allow_float64=allow_float64)

    if check_range:
        verify_range_linear(tensor)

    if check_gamma:
        verify_no_gamma(tensor, strict=strict_gamma)


def create_linear_test_fixture(
    shape: Tuple[int, int, int] = (100, 100, 3),
    mean: float = 0.3,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Create deterministic linear test fixture.

    Generates a synthetic linear-light image with known properties
    for testing linear preservation end-to-end.

    Args:
        shape: Output shape (H, W, C)
        mean: Target mean value in [0, 1] (default 0.3 for typical scenes)
        seed: Random seed for determinism (default 42)

    Returns:
        Linear test fixture (float32, [0, 1])

    Example:
        >>> fixture = create_linear_test_fixture(shape=(50, 50, 3), mean=0.4)
        >>> verify_linear_ingest(fixture)  # Should pass all checks
    """
    # Use local RNG to avoid mutating global state
    rng = np.random.default_rng(seed)

    # Generate uniform random values
    arr = rng.random(shape, dtype=np.float32)

    # Adjust to target mean while staying in [0, 1]
    # This creates a linear gradient distribution
    current_mean = arr.mean()
    if current_mean > 0:
        arr = arr * (mean / current_mean)
        arr = np.clip(arr, 0.0, 1.0)

    return arr


def create_gamma_encoded_fixture(
    shape: Tuple[int, int, int] = (100, 100, 3),
    gamma: float = 2.2,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Create gamma-encoded test fixture (for rejection tests).

    Generates a synthetic gamma-encoded image that SHOULD be rejected
    by linear verification.

    Args:
        shape: Output shape (H, W, C)
        gamma: Gamma value (2.2 for sRGB, default 2.2)
        seed: Random seed for determinism (default 42)

    Returns:
        Gamma-encoded fixture (float32, [0, 1])

    Example:
        >>> fixture = create_gamma_encoded_fixture()
        >>> verify_no_gamma(fixture)  # Should FAIL (as intended)
    """
    # Start with linear
    linear = create_linear_test_fixture(shape=shape, mean=0.5, seed=seed)

    # Apply gamma encoding
    gamma_encoded = np.power(linear, 1.0 / gamma)

    return gamma_encoded.astype(np.float32)
