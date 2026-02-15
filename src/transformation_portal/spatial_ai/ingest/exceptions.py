"""Custom exceptions for Spatial AI linear ingest pipeline.

These exceptions provide clear, actionable error messages with remediation guidance.
All exceptions include context about what went wrong and how to fix it.

Architecture: ADR-023 (Isolation), Issue #890 (Phase I)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class LinearIngestError(Exception):
    """Base exception for all linear ingest errors.

    All custom exceptions in this module inherit from this base class,
    allowing callers to catch any ingest-related error with a single handler.
    """

    pass


class BitDepthViolationError(LinearIngestError):
    """Raised when input bit depth is insufficient for linear ingest.

    This error is raised when strict_ingest=True and the input is 8-bit,
    which would result in lossy quantization unsuitable for training data.

    Attributes:
        input_path: Path to the offending input file.
        detected_dtype: NumPy dtype detected (e.g., "uint8").
        min_required_bits: Minimum bit depth required (typically 16).
    """

    def __init__(
        self,
        input_path: Path,
        detected_dtype: str,
        min_required_bits: int = 16,
        message: Optional[str] = None,
    ):
        """Initialize bit depth violation error.

        Args:
            input_path: Path to input file.
            detected_dtype: Detected NumPy dtype.
            min_required_bits: Minimum required bit depth.
            message: Optional custom message (auto-generated if None).
        """
        self.input_path = input_path
        self.detected_dtype = detected_dtype
        self.min_required_bits = min_required_bits

        if message is None:
            message = (
                f"Bit depth violation: {input_path.name} is {detected_dtype} "
                f"({self._infer_bit_depth(detected_dtype)}-bit), "
                f"but linear ingest requires ≥{min_required_bits}-bit inputs.\n\n"
                f"Remediation:\n"
                f"  1. Use 16-bit TIFF/PNG or 32-bit EXR inputs for training data\n"
                f"  2. Convert RAW files to 16-bit TIFF with linear gamma\n"
                f"  3. Set strict_ingest=False to allow lossy 8-bit normalization (NOT recommended)\n\n"
                f"Context: 8-bit quantization destroys shadow/highlight detail needed for accurate training."
            )

        super().__init__(message)

    @staticmethod
    def _infer_bit_depth(dtype_str: str) -> int:
        """Infer bit depth from dtype string."""
        if "uint8" in dtype_str or "int8" in dtype_str:
            return 8
        elif "uint16" in dtype_str or "int16" in dtype_str:
            return 16
        elif "float32" in dtype_str or "uint32" in dtype_str:
            return 32
        elif "float64" in dtype_str or "uint64" in dtype_str:
            return 64
        else:
            return 0  # Unknown


class SchemaVersionError(LinearIngestError):
    """Raised when manifest schema version is incompatible.

    This error is raised when attempting to load a manifest with an unsupported
    schema version, preventing silent data corruption from schema drift.

    Attributes:
        manifest_path: Path to the manifest file.
        found_version: Version string found in manifest.
        supported_versions: List of supported version strings.
    """

    def __init__(
        self,
        manifest_path: Path,
        found_version: str,
        supported_versions: list[str],
        message: Optional[str] = None,
    ):
        """Initialize schema version error.

        Args:
            manifest_path: Path to manifest file.
            found_version: Version found in manifest.
            supported_versions: List of supported versions.
            message: Optional custom message.
        """
        self.manifest_path = manifest_path
        self.found_version = found_version
        self.supported_versions = supported_versions

        if message is None:
            message = (
                f"Schema version incompatibility: {manifest_path.name} has version '{found_version}', "
                f"but only versions {supported_versions} are supported.\n\n"
                f"Remediation:\n"
                f"  1. Regenerate manifest with current ingest pipeline version\n"
                f"  2. Upgrade ingest pipeline to support '{found_version}'\n"
                f"  3. Migrate manifest to supported version (if migration path exists)\n\n"
                f"Context: Schema version mismatches can cause silent data corruption. "
                f"This error prevents loading incompatible manifests."
            )

        super().__init__(message)


class LinearityViolationError(LinearIngestError):
    """Raised when output data violates linearity constraints.

    This error is raised when:
    - Gamma is not 1.0 (non-linear light)
    - Output dtype is not float32
    - Values are clipped/clamped unexpectedly

    Attributes:
        field: Name of the field that violated constraints.
        expected: Expected value.
        actual: Actual value.
    """

    def __init__(
        self,
        field: str,
        expected: Any,
        actual: Any,
        message: Optional[str] = None,
    ):
        """Initialize linearity violation error.

        Args:
            field: Name of field (e.g., "gamma", "dtype").
            expected: Expected value.
            actual: Actual value.
            message: Optional custom message.
        """
        self.field = field
        self.expected = expected
        self.actual = actual

        if message is None:
            message = (
                f"Linearity violation: {field} must be {expected}, got {actual}.\n\n"
                f"Remediation:\n"
                f"  - For training data, gamma must be 1.0 (linear light)\n"
                f"  - Output dtype must be float32 for precision\n"
                f"  - Do not use tone-mapped or gamma-corrected images\n\n"
                f"Context: Non-linear light relationships break physics-based learning. "
                f"This is a non-negotiable contract for spatial AI training data."
            )

        super().__init__(message)


class RangeViolationError(LinearIngestError):
    """Raised when output values are outside expected range.

    This error is raised when float32 output contains:
    - NaN values (invalid)
    - Infinite values (invalid)
    - Negative values (invalid for linear light)

    Note: Values >1.0 are ALLOWED and expected (HDR preservation).

    Attributes:
        min_value: Minimum value found.
        max_value: Maximum value found.
        has_nan: Whether NaN values were found.
        has_inf: Whether infinite values were found.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        has_nan: bool = False,
        has_inf: bool = False,
        message: Optional[str] = None,
    ):
        """Initialize range violation error.

        Args:
            min_value: Minimum value found.
            max_value: Maximum value found.
            has_nan: Whether NaN detected.
            has_inf: Whether Inf detected.
            message: Optional custom message.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.has_nan = has_nan
        self.has_inf = has_inf

        if message is None:
            issues = []
            if has_nan:
                issues.append("NaN values detected")
            if has_inf:
                issues.append("Infinite values detected")
            if min_value < 0:
                issues.append(f"Negative values detected (min={min_value:.6f})")

            message = (
                f"Range violation: {', '.join(issues)}.\n\n"
                f"Range: [{min_value:.6f}, {max_value:.6f}]\n"
                f"Expected: [0.0, ∞) (negative/NaN/Inf not allowed, but HDR >1.0 is allowed)\n\n"
                f"Remediation:\n"
                f"  - Check input image for corruption\n"
                f"  - Verify decode pipeline is not introducing NaN/Inf\n"
                f"  - Ensure proper normalization (e.g., uint16 / 65535.0)\n\n"
                f"Context: Invalid values indicate decode failure or corrupted input."
            )

        super().__init__(message)


class ProvenanceError(LinearIngestError):
    """Raised when provenance capture or validation fails.

    This error is raised when:
    - EXIF extraction fails
    - Required metadata fields are missing
    - Provenance file cannot be written

    Attributes:
        source: Source of the error (e.g., "EXIF extraction", "file write").
        detail: Detailed error message.
    """

    def __init__(
        self,
        source: str,
        detail: str,
        message: Optional[str] = None,
    ):
        """Initialize provenance error.

        Args:
            source: Source of error.
            detail: Detailed message.
            message: Optional custom message.
        """
        self.source = source
        self.detail = detail

        if message is None:
            message = (
                f"Provenance error in {source}: {detail}\n\n"
                f"Remediation:\n"
                f"  - Verify input file has valid EXIF metadata\n"
                f"  - Check write permissions for output directory\n"
                f"  - Ensure sufficient disk space for provenance JSON\n\n"
                f"Context: Full provenance tracking is required for training data reproducibility."
            )

        super().__init__(message)


class ManifestError(LinearIngestError):
    """Raised when manifest creation or validation fails.

    This error is raised when:
    - Manifest schema validation fails
    - Required manifest fields are missing
    - Manifest file cannot be written/read

    Attributes:
        manifest_path: Path to manifest file (if applicable).
        detail: Detailed error message.
    """

    def __init__(
        self,
        detail: str,
        manifest_path: Optional[Path] = None,
        message: Optional[str] = None,
    ):
        """Initialize manifest error.

        Args:
            detail: Detailed error message.
            manifest_path: Optional manifest file path.
            message: Optional custom message.
        """
        self.manifest_path = manifest_path
        self.detail = detail

        if message is None:
            path_str = f" ({manifest_path})" if manifest_path else ""
            message = (
                f"Manifest error{path_str}: {detail}\n\n"
                f"Remediation:\n"
                f"  - Verify manifest conforms to schema version\n"
                f"  - Check all required fields are present\n"
                f"  - Ensure manifest file is valid JSON\n\n"
                f"Context: Manifest validation ensures dataset integrity and reproducibility."
            )

        super().__init__(message)


class UnsupportedFormatError(LinearIngestError):
    """Raised when input file format is not supported.

    This error is raised when attempting to decode a file with an
    unsupported extension or format.

    Attributes:
        input_path: Path to input file.
        detected_format: Detected format (if any).
        supported_formats: List of supported formats.
    """

    def __init__(
        self,
        input_path: Path,
        detected_format: Optional[str] = None,
        supported_formats: Optional[list[str]] = None,
        message: Optional[str] = None,
    ):
        """Initialize unsupported format error.

        Args:
            input_path: Path to input file.
            detected_format: Detected format.
            supported_formats: List of supported formats.
            message: Optional custom message.
        """
        self.input_path = input_path
        self.detected_format = detected_format
        self.supported_formats = supported_formats or [
            "TIFF (16-bit/32-bit)",
            "PNG (16-bit)",
            "EXR (32-bit float)",
            "RAW (CR2, NEF, ARW, DNG)",
        ]

        if message is None:
            format_str = f" (detected: {detected_format})" if detected_format else ""
            message = (
                f"Unsupported format: {input_path.name}{format_str}\n\n"
                f"Supported formats:\n" + "\n".join(f"  - {fmt}" for fmt in self.supported_formats) + "\n\n"
                f"Remediation:\n"
                f"  - Convert to supported format (recommend 16-bit TIFF)\n"
                f"  - Use RAW files directly (Phase II support)\n\n"
                f"Context: Only formats that preserve linear light are supported."
            )

        super().__init__(message)
