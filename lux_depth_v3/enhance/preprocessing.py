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
import numpy as np

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
        True if normalization succeeded (always, unless error fallback occurs)
        False only on error fallback (file copied without normalization)

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
        True  # Normalization succeeded
        >>>
        >>> # Image without EXIF orientation
        >>> normalize_exif_orientation("landscape.jpg", "normalized.png")
        True  # Normalization succeeded (no-op transformation)
    """
    try:
        # For TIFF files, try tifffile first (handles 32-bit floating point TIFFs)
        if input_path.suffix.lower() in [".tif", ".tiff"]:
            try:
                import tifffile

                # Read with tifffile (handles 32-bit float TIFFs that Pillow can't)
                img_array = tifffile.imread(input_path)

                # Convert to 8-bit for preprocessing
                # Normalize to [0, 1] range and convert to uint8
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    # Clip to [0, 1] range for HDR images
                    img_array = np.clip(img_array, 0, 1)
                    img_array = (img_array * 255).astype(np.uint8)
                elif img_array.dtype == np.uint16:
                    # Convert 16-bit to 8-bit
                    img_array = (img_array / 257).astype(np.uint8)

                # Convert to PIL Image
                if len(img_array.shape) == 2:
                    # Grayscale
                    img = Image.fromarray(img_array, mode="L")
                else:
                    # RGB
                    img = Image.fromarray(img_array, mode="RGB")

                logger.info(f"Loaded 32-bit TIFF with tifffile: {input_path}")
                has_exif_orientation = False  # tifffile doesn't preserve EXIF

            except Exception as tiff_error:
                logger.debug(f"tifffile failed, trying PIL: {tiff_error}")
                # Fall back to PIL
                img = Image.open(input_path)

                # Check if EXIF orientation exists
                has_exif_orientation = False
                if hasattr(img, "getexif"):
                    exif = img.getexif()
                    if exif and 0x0112 in exif:  # 0x0112 = Orientation tag
                        has_exif_orientation = True
                        original_orientation = exif[0x0112]
                        logger.debug(f"Found EXIF orientation {original_orientation} in {input_path}")
        else:
            # Non-TIFF files: use PIL directly
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

        # Always return True because we always create a normalized file
        # This ensures the manifest correctly shows exif_normalized=true
        # even when no EXIF tag was present (file is still normalized via ImageOps.exif_transpose)
        return True

    except Exception as e:
        logger.warning(f"EXIF normalization failed for {input_path}: {e}")
        # Fallback: copy original file
        # This ensures pipeline doesn't fail on bad EXIF data
        import shutil

        try:
            shutil.copy2(input_path, output_path)
            logger.info("Fallback: copied original file without EXIF normalization")
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


def validate_depth_image_alignment(
    image_path: Path,
    depth_path: Path,
) -> None:
    """Validate that depth and image have matching dimensions.

    This preflight check catches EXIF orientation mismatches before V2 runs,
    providing early, deterministic failure instead of cryptic V2 crashes.

    Args:
        image_path: Path to normalized input image
        depth_path: Path to depth PNG

    Raises:
        ValueError: If shapes don't match (EXIF mismatch, invalid depth, etc.)

    Example errors caught:
        - Image: (6000, 3600), Depth: (3600, 6000) → EXIF orientation mismatch
        - Depth has 3 channels instead of 1 → Invalid depth PNG
        - Depth is not uint16 → Quantization error
    """
    try:
        # Load normalized image
        img = Image.open(image_path)
        img_shape = (img.height, img.width)  # (H, W)

        # Load depth PNG
        depth = np.array(Image.open(depth_path))

        # Validate depth format
        if depth.ndim != 2:
            raise ValueError(
                f"Depth must be single-channel (H, W), got shape {depth.shape}. "
                f"This likely means the depth PNG has RGB channels instead of grayscale."
            )

        if depth.dtype != np.uint16:
            raise ValueError(
                f"Depth must be uint16, got {depth.dtype}. "
                f"This indicates a quantization error in depth generation."
            )

        depth_shape = depth.shape  # (H, W)

        # Check shape match
        if img_shape != depth_shape:
            raise ValueError(
                f"Image/depth shape mismatch (likely EXIF orientation issue):\n"
                f"  Image (normalized): {img_shape} (H, W)\n"
                f"  Depth:              {depth_shape} (H, W)\n"
                f"\n"
                f"This usually means:\n"
                f"  1. EXIF normalization was not applied to the input image, OR\n"
                f"  2. Depth was generated from a different version of the input, OR\n"
                f"  3. The normalized file path was not used consistently.\n"
                f"\n"
                f"Expected behavior: Both should match because depth is generated\n"
                f"from the same normalized file that V2 will process."
            )

        logger.debug(
            f"Preflight validation passed: "
            f"image {img_shape} matches depth {depth_shape}, "
            f"dtype {depth.dtype}, "
            f"range [{depth.min()}, {depth.max()}]"
        )

    except FileNotFoundError as e:
        raise ValueError(f"Preflight validation failed: {e}") from e
    except Exception as e:
        raise ValueError(f"Preflight validation failed: {e}") from e
