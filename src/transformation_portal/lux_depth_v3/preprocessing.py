"""Preprocessing utilities for image normalization.

Provides EXIF normalization and depth/image alignment validation.
"""
from __future__ import annotations
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try importing dependencies with graceful fallback
try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, install with: pip install Pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available, install with: pip install opencv-python")


def normalize_exif_orientation(input_path: Path, output_path: Path):
    """Normalize EXIF orientation by rotating image to upright position.

    Args:
        input_path: Input image path
        output_path: Output image path (normalized)
    """
    if not PIL_AVAILABLE:
        # If PIL not available, just copy the file
        logger.warning(
            "PIL not available - skipping EXIF normalization, copying file instead"
        )
        import shutil
        shutil.copy(input_path, output_path)
        return

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Open image
        image = Image.open(input_path)

        # Get EXIF orientation tag if present
        try:
            exif = image.getexif()
            if exif is not None:
                # Find orientation tag
                orientation_key = None
                for tag_id, tag_name in ExifTags.TAGS.items():
                    if tag_name == 'Orientation':
                        orientation_key = tag_id
                        break

                if orientation_key and orientation_key in exif:
                    orientation = exif[orientation_key]

                    # Apply orientation transformations
                    # See EXIF orientation spec: https://www.impulseadventure.com/photo/exif-orientation.html
                    if orientation == 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 4:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 7:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)

                    logger.debug(f"Normalized EXIF orientation {orientation} for {input_path}")
        except (AttributeError, KeyError, IndexError):
            # No EXIF data or orientation tag - that's fine
            pass

        # Save normalized image
        # Remove EXIF data to prevent re-application
        image.save(output_path, exif=b'')

        logger.debug(f"Saved normalized image to {output_path}")

    except Exception as e:
        logger.error(f"Failed to normalize EXIF orientation: {e}")
        # Fallback: copy original file
        import shutil
        shutil.copy(input_path, output_path)


def validate_depth_image_alignment(
    image_path: Path,
    depth_path: Path
) -> bool:
    """Validate that depth map and image have matching dimensions.

    Args:
        image_path: Path to image
        depth_path: Path to depth map

    Returns:
        True if dimensions match, False otherwise
    """
    image_path = Path(image_path)
    depth_path = Path(depth_path)

    # Check files exist
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    try:
        # Load image dimensions
        if PIL_AVAILABLE:
            with Image.open(image_path) as img:
                image_shape = (img.height, img.width)
        elif CV2_AVAILABLE:
            img = cv2.imread(str(image_path))
            if img is None:
                raise IOError(f"Failed to read image: {image_path}")
            image_shape = img.shape[:2]
        else:
            raise ImportError(
                "Either Pillow or opencv-python is required. "
                "Install with: pip install Pillow or pip install opencv-python"
            )

        # Load depth map dimensions
        if CV2_AVAILABLE:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise IOError(f"Failed to read depth map: {depth_path}")
            depth_shape = depth.shape[:2]
        elif PIL_AVAILABLE:
            with Image.open(depth_path) as depth:
                depth_shape = (depth.height, depth.width)
        else:
            raise ImportError(
                "Either opencv-python or Pillow is required. "
                "Install with: pip install opencv-python or pip install Pillow"
            )

        # Check dimensions match
        if image_shape != depth_shape:
            logger.error(
                f"Dimension mismatch: image {image_shape} != depth {depth_shape}"
            )
            return False

        logger.debug(f"Validation passed: {image_path} and {depth_path} have matching dimensions {image_shape}")
        return True

    except Exception as e:
        logger.error(f"Failed to validate alignment: {e}")
        raise
