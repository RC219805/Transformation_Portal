"""Preprocessing utilities for image normalization.

Standardizes image validation and normalization for V3 depth inference.

Design:
- PIL + NumPy only (no torch/transformers)
- Validates image integrity and format
- Normalizes to float32 RGB [0, 1]
- Enforces multiple-of-14 dimensions for Depth Anything V3
- Preserves original dimensions for post-processing
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

# Depth Anything V3 requires dimensions to be multiples of 14
DIMENSION_MULTIPLE = 14
MIN_DIMENSION = 14


def validate_image_format(image_path: Union[str, Path]) -> Path:
    """Validate image format and integrity.

    Checks:
    - File exists
    - Extension is supported
    - File can be opened by PIL
    - Image data is not corrupt

    Args:
        image_path: Path to image file

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format unsupported or image corrupt
    """
    image_path = Path(image_path)

    # Check existence
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check extension
    ext = image_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Verify image integrity
    try:
        # Open and verify (checks file structure)
        img = Image.open(image_path)
        img.verify()

        # verify() invalidates the image object, reopen to test pixel load
        img = Image.open(image_path)
        img.load()  # Force load pixel data

    except Exception as e:
        raise ValueError(
            f"Image file corrupt or invalid: {image_path}"
        ) from e

    return image_path


def preprocess_image(
    image: Union[np.ndarray, Path, str],
    target_size: Optional[int] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess image for depth inference.

    Converts to RGB float32 [0, 1] and ensures dimensions are multiples of 14.

    Processing steps:
    1. Load/validate image
    2. Convert to RGB float32 [0, 1]
    3. Optionally resize to target_size (long edge)
    4. Enforce multiple-of-14 dimensions
    5. Return processed image + original shape

    Args:
        image: Input as numpy array, Path, or str
        target_size: Optional target size for long edge (maintains aspect)

    Returns:
        Tuple of:
            - Processed image (float32, HxWx3, [0, 1])
            - Original shape (H, W) before any resizing

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If image invalid or unsupported format
    """
    # Load image
    if isinstance(image, (str, Path)):
        image_path = validate_image_format(image)
        pil_img = Image.open(image_path)
    elif isinstance(image, np.ndarray):
        # Convert numpy array to PIL for consistent processing
        if image.ndim == 2:
            # Grayscale
            pil_img = Image.fromarray(image, mode='L')
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB
            if image.dtype == np.uint8:
                pil_img = Image.fromarray(image, mode='RGB')
            elif image.dtype == np.float32 or image.dtype == np.float64:
                # Convert float [0, 1] to uint8 for PIL
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                pil_img = Image.fromarray(image_uint8, mode='RGB')
            else:
                raise ValueError(f"Unsupported array dtype: {image.dtype}")
        elif image.ndim == 3 and image.shape[2] == 4:
            # RGBA - drop alpha channel
            if image.dtype == np.uint8:
                pil_img = Image.fromarray(image, mode='RGBA')
            else:
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                pil_img = Image.fromarray(image_uint8, mode='RGBA')
        else:
            raise ValueError(
                f"Unsupported array shape: {image.shape}. "
                f"Expected (H, W), (H, W, 3), or (H, W, 4)"
            )
    else:
        raise TypeError(
            f"Image must be np.ndarray, Path, or str. Got: {type(image)}"
        )

    # Save original dimensions
    original_h, original_w = pil_img.size[1], pil_img.size[0]  # PIL uses (W, H)

    # Convert to RGB (handles grayscale, RGBA)
    pil_img = pil_img.convert('RGB')

    # Optional resize to target size (long edge)
    if target_size is not None:
        pil_img = _resize_keep_aspect(pil_img, target_size)

    # Convert to numpy float32 [0, 1]
    img_array = np.array(pil_img, dtype=np.float32) / 255.0

    # Enforce multiple-of-14 dimensions
    img_array = _enforce_dimension_multiple(img_array, DIMENSION_MULTIPLE)

    return img_array, (original_h, original_w)


def _resize_keep_aspect(pil_img: Image.Image, target_size: int) -> Image.Image:
    """Resize image to target size (long edge), maintaining aspect ratio.

    Args:
        pil_img: PIL Image
        target_size: Target size for long edge

    Returns:
        Resized PIL Image
    """
    w, h = pil_img.size

    # Determine scaling factor (long edge to target_size)
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    # Use LANCZOS for quality
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _enforce_dimension_multiple(
    img_array: np.ndarray,
    multiple: int
) -> np.ndarray:
    """Enforce that image dimensions are multiples of a given value.

    Rounds down to nearest multiple, ensuring minimum of `multiple`.

    Args:
        img_array: Image array (H, W, C) float32
        multiple: Required multiple (e.g., 14 for Depth Anything V3)

    Returns:
        Resized image array with compliant dimensions
    """
    h, w = img_array.shape[:2]

    # Round down to nearest multiple, clamp to minimum
    new_h = max(multiple, (h // multiple) * multiple)
    new_w = max(multiple, (w // multiple) * multiple)

    # Only resize if dimensions changed
    if (new_h, new_w) != (h, w):
        # Convert back to PIL for quality resize
        img_uint8 = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='RGB')
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert back to float32
        img_array = np.array(pil_img, dtype=np.float32) / 255.0

        logger.debug(
            f"Enforced dimension multiple: ({h}, {w}) → ({new_h}, {new_w})"
        )

    return img_array


# Keep legacy stubs for backward compatibility (not yet used by V3 orchestrator)

def normalize_exif_orientation(input_path: Path, output_path: Path):
    """Normalize EXIF orientation by rotating image to upright position.

    STUB: Not implemented. Not required for V3 depth inference.

    Args:
        input_path: Input image path
        output_path: Output image path (normalized)

    Raises:
        NotImplementedError: This is a stub implementation
    """
    raise NotImplementedError(
        "normalize_exif_orientation() is a stub - full implementation pending. "
        "This module was created to enable package imports."
    )


def validate_depth_image_alignment(
    image_path: Path,
    depth_path: Path
) -> bool:
    """Validate that depth map and image have matching dimensions.

    STUB: Not implemented. Not required for V3 depth inference.

    Args:
        image_path: Path to image
        depth_path: Path to depth map

    Returns:
        True if dimensions match, False otherwise

    Raises:
        NotImplementedError: This is a stub implementation
    """
    raise NotImplementedError(
        "validate_depth_image_alignment() is a stub - full implementation pending. "
        "This module was created to enable package imports."
    )
