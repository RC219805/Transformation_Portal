"""Dimension validation for image processing pipelines.

Lightweight utility module with zero ML dependencies.
Safe to import in any context (tests, CLI, lightweight scripts).
"""

from __future__ import annotations

from typing import Optional, Tuple

# Stable Diffusion 1.5 dimension requirements
SD_DIMENSION_MULTIPLE = 64
MIN_SD_DIMENSION = 512
MAX_RECOMMENDED_PIXELS = 1024 * 1024  # 1MP recommended maximum


def validate_sd_dimensions(
    width: int, height: int, auto_correct: bool = True, warn_callback: Optional[callable] = None
) -> Tuple[int, int]:
    """Validate and optionally auto-correct dimensions for Stable Diffusion 1.5 compatibility.

    SD 1.5 requires dimensions that are multiples of 64 for proper feature map alignment.
    This prevents cryptic tensor dimension mismatch errors during processing.

    Args:
        width: Desired image width in pixels
        height: Desired image height in pixels
        auto_correct: If True, auto-corrects to nearest valid dimensions.
                      If False, raises an error for invalid dimensions.
        warn_callback: Optional callback(message: str) for warnings.
                       If None, warnings are silent.

    Returns:
        Tuple of (validated_width, validated_height)

    Raises:
        ValueError: If dimensions are invalid and auto_correct is False

    Examples:
        >>> validate_sd_dimensions(1024, 768)
        (1024, 768)
        >>> validate_sd_dimensions(1024, 770, auto_correct=True)
        (1024, 768)
        >>> validate_sd_dimensions(511, 511, auto_correct=False)
        Traceback (most recent call last):
        ...
        ValueError: Dimensions 511×511 are invalid for Stable Diffusion 1.5: ...
    """
    original_width, original_height = width, height

    # Check if dimensions are multiples of SD_DIMENSION_MULTIPLE or below minimum
    needs_correction = (
        width % SD_DIMENSION_MULTIPLE != 0
        or height % SD_DIMENSION_MULTIPLE != 0
        or width < MIN_SD_DIMENSION
        or height < MIN_SD_DIMENSION
    )

    if needs_correction:
        if auto_correct:
            # Round down to nearest multiple of SD_DIMENSION_MULTIPLE
            corrected_width = (width // SD_DIMENSION_MULTIPLE) * SD_DIMENSION_MULTIPLE
            corrected_height = (height // SD_DIMENSION_MULTIPLE) * SD_DIMENSION_MULTIPLE

            # Ensure minimum dimensions
            corrected_width = max(MIN_SD_DIMENSION, corrected_width)
            corrected_height = max(MIN_SD_DIMENSION, corrected_height)

            if warn_callback:
                warn_callback(
                    f"⚠ Corrected dimensions from {original_width}×{original_height} "
                    f"to {corrected_width}×{corrected_height} (SD 1.5 compatible)"
                )

            return corrected_width, corrected_height

        # Build error message
        errors = []
        if width % SD_DIMENSION_MULTIPLE != 0 or height % SD_DIMENSION_MULTIPLE != 0:
            errors.append(f"must be multiples of {SD_DIMENSION_MULTIPLE}")
        if width < MIN_SD_DIMENSION or height < MIN_SD_DIMENSION:
            errors.append(f"must be at least {MIN_SD_DIMENSION}")

        raise ValueError(
            f"Dimensions {width}×{height} are invalid for Stable Diffusion 1.5: "
            f"{' and '.join(errors)}. Use auto_correct=True to fix automatically."
        )

    # Warn about very large dimensions
    if width * height > MAX_RECOMMENDED_PIXELS and warn_callback:
        warn_callback(
            f"⚠ Dimensions {width}×{height} exceed recommended maximum "
            f"({MAX_RECOMMENDED_PIXELS} pixels). Processing may be slow or fail."
        )

    return width, height
