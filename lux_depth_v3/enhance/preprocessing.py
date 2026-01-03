"""EXIF orientation preprocessing for V3+V2 pipeline alignment.

This module ensures that both DA3 (PIL-based) and V2 (OpenCV-based) pipelines
see the same pixel data by pre-normalizing EXIF orientation before any processing.

The Problem:
- PIL (DA3): Automatically applies EXIF orientation when loading images
- OpenCV (V2): Ignores EXIF orientation, reads raw pixel data
- Result: Depth and image are misaligned if EXIF orientation present

The Solution:
- Pre-normalize EXIF orientation once using PIL's ImageOps.exif_transpose()
- Strip EXIF orientation tag to prevent double-application
- Feed normalized file to both DA3 and V2 pipelines
"""

from __future__ import annotations

from pathlib import Path
import logging
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def normalize_exif_orientation(input_path: Path, output_path: Path) -> bool:
    """Apply EXIF orientation and write normalized file.

    This function reads an image, applies any EXIF orientation transformation,
    removes the EXIF orientation tag, and writes a normalized version. This ensures
    that both PIL-based and OpenCV-based tools see the same pixel arrangement.

    Args:
        input_path: Original image with potential EXIF orientation
        output_path: Path to write normalized image (EXIF orientation applied, tag removed)

    Returns:
        True if normalization was applied (EXIF orientation tag existed), False otherwise

    Side effects:
        - Writes normalized image to output_path
        - Strips EXIF orientation tag (0x0112) to prevent double-application
        - Creates parent directories if needed

    EXIF Orientation Values:
        1: Normal (no rotation)
        2: Flip horizontal
        3: Rotate 180°
        4: Flip vertical
        5: Transpose (flip along top-left to bottom-right diagonal)
        6: Rotate 90° CW
        7: Transverse (flip along top-right to bottom-left diagonal)
        8: Rotate 90° CCW

    Examples:
        >>> # Image with orientation 6 (90° CW rotation)
        >>> normalize_exif_orientation("portrait.jpg", "normalized.png")
        True  # Orientation was applied
        >>>
        >>> # Image without EXIF orientation
        >>> normalize_exif_orientation("landscape.jpg", "normalized.png")
        False  # No orientation tag found
    """
    try:
        img = Image.open(input_path)

        # Check if EXIF orientation exists
        has_exif_orientation = False
        if hasattr(img, "getexif"):
            exif = img.getexif()
            if exif and 0x0112 in exif:  # 0x0112 = Orientation tag
                has_exif_orientation = True
                original_orientation = exif[0x0112]
                logger.debug(f"Found EXIF orientation {original_orientation} in {input_path}")

        # Apply orientation transformation
        # ImageOps.exif_transpose() handles all 8 orientations correctly
        img_normalized = ImageOps.exif_transpose(img)

        # Strip EXIF orientation tag (prevent double-application)
        if has_exif_orientation and hasattr(img_normalized, "getexif"):
            exif_new = img_normalized.getexif()
            if exif_new and 0x0112 in exif_new:
                del exif_new[0x0112]
                logger.debug(f"Stripped EXIF orientation tag from {output_path}")

        # Write normalized image
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as PNG to avoid lossy compression
        # This preserves quality for depth estimation
        img_normalized.save(output_path, format="PNG")

        if has_exif_orientation:
            logger.info(f"Normalized EXIF orientation: {input_path} → {output_path}")
        else:
            logger.debug(f"No EXIF orientation found, passthrough: {input_path} → {output_path}")

        return has_exif_orientation

    except Exception as e:
        logger.warning(f"EXIF normalization failed for {input_path}: {e}")
        # Fallback: copy original file
        # This ensures pipeline doesn't fail on bad EXIF data
        import shutil

        try:
            shutil.copy2(input_path, output_path)
            logger.info(f"Fallback: copied original file without EXIF normalization")
            return False
        except Exception as copy_error:
            logger.error(f"Failed to copy file as fallback: {copy_error}")
            raise


def get_exif_orientation(path: Path) -> int:
    """Get EXIF orientation value from image.

    Args:
        path: Path to image file

    Returns:
        EXIF orientation value (1-8), or 1 if no orientation tag found

    Notes:
        - Returns 1 (normal) if image has no EXIF data
        - Returns 1 (normal) if orientation tag is missing
        - Does not modify the image
    """
    try:
        img = Image.open(path)
        if hasattr(img, "getexif"):
            exif = img.getexif()
            if exif and 0x0112 in exif:
                return exif[0x0112]
    except Exception as e:
        logger.debug(f"Could not read EXIF orientation from {path}: {e}")

    return 1  # Default: normal orientation


def has_exif_orientation(path: Path) -> bool:
    """Check if image has EXIF orientation tag.

    Args:
        path: Path to image file

    Returns:
        True if EXIF orientation tag exists, False otherwise
    """
    try:
        img = Image.open(path)
        if hasattr(img, "getexif"):
            exif = img.getexif()
            if exif and 0x0112 in exif:
                return True
    except Exception as e:
        logger.debug(f"Could not check EXIF orientation for {path}: {e}")

    return False
