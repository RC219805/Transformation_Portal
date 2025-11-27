"""Input validation for image processing pipelines.

Provides comprehensive validation for:
- Image files (format, dimensions, color space, corruption)
- File paths and permissions
- Pipeline configurations
- Context data

Example:
    >>> from transformation_portal.utils.input_validation import ImageValidator
    >>> validator = ImageValidator()
    >>> result = validator.validate("render.jpg")
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(f"Error: {error}")
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from .error_handling import ProcessingError

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    message: str
    severity: ValidationSeverity
    suggestion: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        result = f"[{self.severity.value.upper()}] {self.code}: {self.message}"
        if self.suggestion:
            result += f" ({self.suggestion})"
        return result


@dataclass
class ValidationResult:
    """Result of validation with all issues found."""
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False

    def add_error(
        self,
        code: str,
        message: str,
        suggestion: Optional[str] = None,
        **details
    ) -> None:
        """Add an error issue."""
        self.add_issue(ValidationIssue(
            code=code,
            message=message,
            severity=ValidationSeverity.ERROR,
            suggestion=suggestion,
            details=details,
        ))

    def add_warning(
        self,
        code: str,
        message: str,
        suggestion: Optional[str] = None,
        **details
    ) -> None:
        """Add a warning issue."""
        self.add_issue(ValidationIssue(
            code=code,
            message=message,
            severity=ValidationSeverity.WARNING,
            suggestion=suggestion,
            details=details,
        ))

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if not self.is_valid:
            error_messages = [str(e) for e in self.errors]
            raise ImageValidationError(
                f"Validation failed with {len(self.errors)} error(s):\n"
                + "\n".join(error_messages)
            )


class ImageValidationError(ProcessingError):
    """Exception raised when image validation fails."""
    pass


class ImageValidator:
    """Comprehensive image validator for processing pipelines.

    Validates:
    - File existence and readability
    - Image format (JPEG, PNG, TIFF, WebP, etc.)
    - Dimensions (min/max constraints)
    - Color space (RGB, RGBA, grayscale)
    - File size limits
    - Image corruption detection

    Example:
        >>> validator = ImageValidator(
        ...     min_width=512,
        ...     max_width=8192,
        ...     allowed_formats={'JPEG', 'PNG', 'TIFF'},
        ...     max_file_size_mb=100
        ... )
        >>> result = validator.validate("large_render.tiff")
        >>> if result.is_valid:
        ...     process_image("large_render.tiff")
    """

    # Common image formats for architectural rendering
    DEFAULT_FORMATS = {'JPEG', 'PNG', 'TIFF', 'WEBP', 'BMP', 'EXR', 'HDR'}

    # Supported color modes
    SUPPORTED_MODES = {'RGB', 'RGBA', 'L', 'LA', 'P', 'I', 'F'}

    def __init__(
        self,
        min_width: int = 64,
        max_width: int = 16384,
        min_height: int = 64,
        max_height: int = 16384,
        allowed_formats: Optional[Set[str]] = None,
        allowed_modes: Optional[Set[str]] = None,
        max_file_size_mb: float = 500,
        require_rgb: bool = False,
        check_corruption: bool = True,
    ):
        """Initialize validator with constraints.

        Args:
            min_width: Minimum image width in pixels
            max_width: Maximum image width in pixels
            min_height: Minimum image height in pixels
            max_height: Maximum image height in pixels
            allowed_formats: Set of allowed format names (e.g., {'JPEG', 'PNG'})
            allowed_modes: Set of allowed color modes (e.g., {'RGB', 'RGBA'})
            max_file_size_mb: Maximum file size in megabytes
            require_rgb: If True, require RGB or RGBA mode
            check_corruption: If True, attempt to load full image to detect corruption
        """
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.allowed_formats = allowed_formats or self.DEFAULT_FORMATS
        self.allowed_modes = allowed_modes or self.SUPPORTED_MODES
        self.max_file_size_mb = max_file_size_mb
        self.require_rgb = require_rgb
        self.check_corruption = check_corruption

    def validate(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        context: Optional[str] = None,
    ) -> ValidationResult:
        """Validate an image input.

        Args:
            image_input: Path to image, PIL Image, or numpy array
            context: Optional context string for error messages

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult()
        context_prefix = f"[{context}] " if context else ""

        # Handle different input types
        if isinstance(image_input, (str, Path)):
            self._validate_file(Path(image_input), result, context_prefix)
        elif isinstance(image_input, Image.Image):
            self._validate_pil_image(image_input, result, context_prefix)
        elif isinstance(image_input, np.ndarray):
            self._validate_numpy_array(image_input, result, context_prefix)
        else:
            result.add_error(
                "INVALID_INPUT_TYPE",
                f"{context_prefix}Expected file path, PIL Image, or numpy array, "
                f"got {type(image_input).__name__}",
            )

        return result

    def _validate_file(
        self,
        path: Path,
        result: ValidationResult,
        prefix: str = "",
    ) -> Optional[Image.Image]:
        """Validate an image file."""
        # Check file exists
        if not path.exists():
            result.add_error(
                "FILE_NOT_FOUND",
                f"{prefix}File not found: {path}",
                suggestion="Check the file path and ensure the file exists",
            )
            return None

        # Check file is readable
        if not os.access(path, os.R_OK):
            result.add_error(
                "FILE_NOT_READABLE",
                f"{prefix}File is not readable: {path}",
                suggestion="Check file permissions",
            )
            return None

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            result.add_error(
                "FILE_TOO_LARGE",
                f"{prefix}File size ({file_size_mb:.1f} MB) exceeds "
                f"maximum ({self.max_file_size_mb} MB)",
                suggestion="Reduce image resolution or use compression",
                file_size_mb=file_size_mb,
            )
            return None

        # Warn for very large files
        if file_size_mb > self.max_file_size_mb * 0.8:
            result.add_warning(
                "FILE_SIZE_WARNING",
                f"{prefix}File is large ({file_size_mb:.1f} MB), "
                "processing may be slow",
            )

        # Try to open the image
        try:
            image = Image.open(path)
        except UnidentifiedImageError:
            result.add_error(
                "UNRECOGNIZED_FORMAT",
                f"{prefix}Cannot identify image format: {path}",
                suggestion="Ensure the file is a valid image",
            )
            return None
        except Exception as e:
            result.add_error(
                "FILE_READ_ERROR",
                f"{prefix}Failed to open image: {e}",
            )
            return None

        # Check format
        if image.format and image.format.upper() not in self.allowed_formats:
            result.add_error(
                "UNSUPPORTED_FORMAT",
                f"{prefix}Format '{image.format}' is not supported",
                suggestion=f"Use one of: {', '.join(sorted(self.allowed_formats))}",
            )

        # Validate the PIL image
        self._validate_pil_image(image, result, prefix)

        # Check for corruption by loading full image
        if self.check_corruption and result.is_valid:
            try:
                image.load()
            except Exception as e:
                result.add_error(
                    "IMAGE_CORRUPTED",
                    f"{prefix}Image appears to be corrupted: {e}",
                    suggestion="Re-export the image from the source application",
                )
                return None

        return image

    def _validate_pil_image(
        self,
        image: Image.Image,
        result: ValidationResult,
        prefix: str = "",
    ) -> None:
        """Validate a PIL Image object."""
        width, height = image.size

        # Check dimensions
        if width < self.min_width:
            result.add_error(
                "WIDTH_TOO_SMALL",
                f"{prefix}Image width ({width}px) is below minimum ({self.min_width}px)",
                width=width,
                min_width=self.min_width,
            )

        if width > self.max_width:
            result.add_error(
                "WIDTH_TOO_LARGE",
                f"{prefix}Image width ({width}px) exceeds maximum ({self.max_width}px)",
                suggestion="Resize the image before processing",
                width=width,
                max_width=self.max_width,
            )

        if height < self.min_height:
            result.add_error(
                "HEIGHT_TOO_SMALL",
                f"{prefix}Image height ({height}px) is below minimum ({self.min_height}px)",
                height=height,
                min_height=self.min_height,
            )

        if height > self.max_height:
            result.add_error(
                "HEIGHT_TOO_LARGE",
                f"{prefix}Image height ({height}px) exceeds maximum ({self.max_height}px)",
                suggestion="Resize the image before processing",
                height=height,
                max_height=self.max_height,
            )

        # Check color mode
        if image.mode not in self.allowed_modes:
            result.add_error(
                "UNSUPPORTED_COLOR_MODE",
                f"{prefix}Color mode '{image.mode}' is not supported",
                suggestion=f"Convert to one of: {', '.join(sorted(self.allowed_modes))}",
            )

        if self.require_rgb and image.mode not in ('RGB', 'RGBA'):
            result.add_error(
                "RGB_REQUIRED",
                f"{prefix}RGB/RGBA mode required, got '{image.mode}'",
                suggestion="Convert image to RGB mode",
            )

        # Warn about unusual aspect ratios
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            result.add_warning(
                "UNUSUAL_ASPECT_RATIO",
                f"{prefix}Unusual aspect ratio ({aspect_ratio:.2f}:1)",
                suggestion="Verify this is intentional",
            )

        # Warn about very high resolution
        megapixels = (width * height) / 1_000_000
        if megapixels > 50:
            result.add_warning(
                "VERY_HIGH_RESOLUTION",
                f"{prefix}Very high resolution ({megapixels:.1f} MP), "
                "processing may be slow",
            )

    def _validate_numpy_array(
        self,
        array: np.ndarray,
        result: ValidationResult,
        prefix: str = "",
    ) -> None:
        """Validate a numpy array image."""
        # Check dimensions
        if array.ndim < 2 or array.ndim > 3:
            result.add_error(
                "INVALID_ARRAY_DIMENSIONS",
                f"{prefix}Expected 2D or 3D array, got {array.ndim}D",
            )
            return

        height, width = array.shape[:2]

        # Check dimension constraints
        if width < self.min_width:
            result.add_error(
                "WIDTH_TOO_SMALL",
                f"{prefix}Array width ({width}) is below minimum ({self.min_width})",
            )

        if width > self.max_width:
            result.add_error(
                "WIDTH_TOO_LARGE",
                f"{prefix}Array width ({width}) exceeds maximum ({self.max_width})",
            )

        if height < self.min_height:
            result.add_error(
                "HEIGHT_TOO_SMALL",
                f"{prefix}Array height ({height}) is below minimum ({self.min_height})",
            )

        if height > self.max_height:
            result.add_error(
                "HEIGHT_TOO_LARGE",
                f"{prefix}Array height ({height}) exceeds maximum ({self.max_height})",
            )

        # Check channels for 3D arrays
        if array.ndim == 3:
            channels = array.shape[2]
            if channels not in (1, 3, 4):
                result.add_error(
                    "INVALID_CHANNEL_COUNT",
                    f"{prefix}Expected 1, 3, or 4 channels, got {channels}",
                )

            if self.require_rgb and channels == 1:
                result.add_error(
                    "RGB_REQUIRED",
                    f"{prefix}RGB image required but got grayscale (1 channel)",
                )
        elif self.require_rgb:
            result.add_error(
                "RGB_REQUIRED",
                f"{prefix}RGB image required but got 2D (grayscale) array",
            )

        # Check data type and range
        if array.dtype in (np.float32, np.float64):
            # Float images should be in [0, 1] range
            arr_min, arr_max = np.nanmin(array), np.nanmax(array)
            if arr_min < -0.1 or arr_max > 1.1:
                result.add_warning(
                    "FLOAT_RANGE_WARNING",
                    f"{prefix}Float array values outside [0, 1] range "
                    f"(min={arr_min:.2f}, max={arr_max:.2f})",
                    suggestion="Normalize values to [0, 1] range",
                )
        elif array.dtype not in (np.uint8, np.uint16):
            result.add_warning(
                "UNUSUAL_DTYPE",
                f"{prefix}Unusual dtype '{array.dtype}', expected uint8/uint16/float32",
            )

        # Check for NaN or Inf values
        if np.issubdtype(array.dtype, np.floating):
            if np.any(np.isnan(array)):
                result.add_error(
                    "CONTAINS_NAN",
                    f"{prefix}Array contains NaN values",
                )
            if np.any(np.isinf(array)):
                result.add_error(
                    "CONTAINS_INF",
                    f"{prefix}Array contains infinite values",
                )


class BatchValidator:
    """Validator for batch processing with progress tracking."""

    def __init__(self, image_validator: Optional[ImageValidator] = None):
        """Initialize batch validator.

        Args:
            image_validator: ImageValidator instance to use
        """
        self.image_validator = image_validator or ImageValidator()
        self.results: Dict[str, ValidationResult] = {}

    def validate_batch(
        self,
        paths: List[Union[str, Path]],
        stop_on_first_error: bool = False,
    ) -> Dict[str, ValidationResult]:
        """Validate a batch of images.

        Args:
            paths: List of image paths
            stop_on_first_error: Stop validation on first error

        Returns:
            Dictionary mapping paths to validation results
        """
        self.results = {}

        for path in paths:
            path_str = str(path)
            result = self.image_validator.validate(path)
            self.results[path_str] = result

            if stop_on_first_error and not result.is_valid:
                logger.warning(f"Stopping batch validation due to error in {path}")
                break

        return self.results

    @property
    def all_valid(self) -> bool:
        """Check if all validated images are valid."""
        return all(r.is_valid for r in self.results.values())

    @property
    def valid_paths(self) -> List[str]:
        """Get list of valid image paths."""
        return [p for p, r in self.results.items() if r.is_valid]

    @property
    def invalid_paths(self) -> List[str]:
        """Get list of invalid image paths."""
        return [p for p, r in self.results.items() if not r.is_valid]

    def summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        total = len(self.results)
        valid = len(self.valid_paths)
        invalid = len(self.invalid_paths)

        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "success_rate": valid / total if total > 0 else 0,
            "invalid_paths": self.invalid_paths,
        }


# Convenience functions

def validate_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    **kwargs
) -> ValidationResult:
    """Validate an image with default settings.

    Args:
        image_input: Image path, PIL Image, or numpy array
        **kwargs: Additional arguments for ImageValidator

    Returns:
        ValidationResult
    """
    validator = ImageValidator(**kwargs)
    return validator.validate(image_input)


def validate_image_strict(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    min_size: int = 512,
    max_size: int = 8192,
) -> ValidationResult:
    """Validate an image with strict architectural rendering settings.

    Args:
        image_input: Image input
        min_size: Minimum dimension (width or height)
        max_size: Maximum dimension (width or height)

    Returns:
        ValidationResult
    """
    validator = ImageValidator(
        min_width=min_size,
        min_height=min_size,
        max_width=max_size,
        max_height=max_size,
        require_rgb=True,
        allowed_formats={'JPEG', 'PNG', 'TIFF'},
        check_corruption=True,
    )
    return validator.validate(image_input)


def require_valid_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    **kwargs
) -> None:
    """Validate an image and raise exception if invalid.

    Args:
        image_input: Image input
        **kwargs: Additional arguments for ImageValidator

    Raises:
        ImageValidationError: If validation fails
    """
    result = validate_image(image_input, **kwargs)
    result.raise_if_invalid()


def is_valid_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    **kwargs
) -> bool:
    """Quick check if an image is valid.

    Args:
        image_input: Image input
        **kwargs: Additional arguments for ImageValidator

    Returns:
        True if image is valid
    """
    return validate_image(image_input, **kwargs).is_valid
