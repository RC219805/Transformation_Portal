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
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from .raw_loader import RAW_EXTENSIONS, is_raw_file, load_raw_as_pil

logger = logging.getLogger(__name__)

# Supported image formats (standard + RAW)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"} | RAW_EXTENSIONS

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
        raise ValueError(f"Unsupported image format: {ext}. " f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    # Verify image integrity
    try:
        # RAW files require different validation (PIL can't open them)
        if is_raw_file(image_path):
            # For RAW, just check file is readable
            # Full validation happens during load_raw_as_pil()
            with open(image_path, "rb") as f:
                # Read first few bytes to ensure file is readable
                header = f.read(16)
                if len(header) < 16:
                    raise ValueError("RAW file too small or empty")
        else:
            # Standard image validation for non-RAW
            # Open and verify (checks file structure)
            with Image.open(image_path) as img:
                img.verify()

            # verify() invalidates the image object, reopen to test pixel load
            with Image.open(image_path) as img:
                img.load()  # Force load pixel data

    except Exception as e:
        raise ValueError(f"Image file corrupt or invalid: {image_path}") from e

    return image_path


def preprocess_image(
    image: Union[np.ndarray, Path, str], target_size: Optional[int] = None
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

        # PIL-first fallback pattern: try PIL, fallback to RAW only if needed
        try:
            pil_img = Image.open(image_path).convert("RGB")
        except (Image.UnidentifiedImageError, OSError, ValueError):
            # PIL failed - try RAW only if extension suggests RAW and rawpy available
            if is_raw_file(image_path):
                logger.debug(f"PIL failed, loading as RAW: {image_path.name}")
                pil_img = load_raw_as_pil(image_path, use_camera_wb=True, half_size=False)
            else:
                # Re-raise original error if not a RAW file
                raise
    elif isinstance(image, np.ndarray):
        # Convert numpy array to PIL for consistent processing
        if image.ndim == 2:
            # Grayscale
            pil_img = Image.fromarray(image, mode="L")
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB
            if image.dtype == np.uint8:
                pil_img = Image.fromarray(image, mode="RGB")
            elif image.dtype == np.float32 or image.dtype == np.float64:
                # Convert float [0, 1] to uint8 for PIL
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                pil_img = Image.fromarray(image_uint8, mode="RGB")
            else:
                raise ValueError(f"Unsupported array dtype: {image.dtype}")
        elif image.ndim == 3 and image.shape[2] == 4:
            # RGBA - drop alpha channel
            if image.dtype == np.uint8:
                pil_img = Image.fromarray(image, mode="RGBA")
            else:
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                pil_img = Image.fromarray(image_uint8, mode="RGBA")
        else:
            raise ValueError(f"Unsupported array shape: {image.shape}. " f"Expected (H, W), (H, W, 3), or (H, W, 4)")
    else:
        raise TypeError(f"Image must be np.ndarray, Path, or str. Got: {type(image)}")

    # Save original dimensions
    original_h, original_w = pil_img.size[1], pil_img.size[0]  # PIL uses (W, H)

    # Convert to RGB (handles grayscale, RGBA)
    pil_img = pil_img.convert("RGB")

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


def _enforce_dimension_multiple(img_array: np.ndarray, multiple: int) -> np.ndarray:
    """Enforce that image dimensions are multiples of a given value.

    Uses center crop + pad to preserve pixel fidelity rather than resampling.
    This avoids subtle quality degradation from resizing the entire image.

    Args:
        img_array: Image array (H, W, C) float32
        multiple: Required multiple (e.g., 14 for Depth Anything V3)

    Returns:
        Image array with compliant dimensions (may be cropped/padded)
    """
    h, w = img_array.shape[:2]

    # Compute target dimensions (nearest multiple)
    # Round down to nearest multiple, then ensure minimum
    new_h = max(multiple, (h // multiple) * multiple)
    new_w = max(multiple, (w // multiple) * multiple)

    # Only adjust if dimensions changed
    if (new_h, new_w) != (h, w):
        # Strategy: handle crop and pad independently per dimension
        # This supports mixed scenarios (e.g., 15x10 → 14x14: crop width, pad height)

        # Handle height dimension
        if new_h < h:
            # Crop height (center crop)
            crop_top = (h - new_h) // 2
            img_array = img_array[crop_top : crop_top + new_h, :]
        elif new_h > h:
            # Pad height (symmetric padding)
            pad_h = new_h - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            img_array = np.pad(img_array, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="edge")

        # Handle width dimension
        if new_w < w:
            # Crop width (center crop)
            crop_left = (w - new_w) // 2
            img_array = img_array[:, crop_left : crop_left + new_w]
        elif new_w > w:
            # Pad width (symmetric padding)
            pad_w = new_w - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            img_array = np.pad(img_array, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge")

        logger.debug(f"Enforced dimension multiple: ({h}, {w}) → ({new_h}, {new_w})")

    return img_array


def preprocess_image_linear(
    image: Union[np.ndarray, Path, str],
    target_size: Optional[int] = None,
    verify_linearity: bool = True,
    apex_strict_formats: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess image with linear light preservation for APEX pipeline.

    This is the APEX-compliant preprocessing path that enforces:
    - Deterministic format boundary (RAW + TIFF only by default)
    - Linear light preservation (no gamma encoding)
    - Floating point dtype (no uint8/uint16 leakage)
    - Value range validation [0, 1]
    - 16-bit precision preservation for RAW/TIFF

    Per Spatial AI Foundation ROADMAP (Section I: Data Fidelity is Sacred):
    "Training inputs MUST preserve linear-light relationships."

    Processing steps:
    1. Validate format boundary (RAW + TIFF only unless apex_strict_formats=False)
    2. Load/validate image (RAW → linear uint16, TIFF → preserve bit depth)
    3. Convert to linear float32 [0, 1] preserving precision
    4. Validate linearity (dtype, range, gamma detection)
    5. Optionally resize to target_size (long edge)
    6. Enforce multiple-of-14 dimensions
    7. Return validated linear image + original shape

    Args:
        image: Input as numpy array, Path, or str
               For RAW: must be RAW file (linear output enforced)
               For TIFF: must be linear (gamma-encoded rejected)
               For JPEG/PNG: rejected by default (set apex_strict_formats=False to allow)
        target_size: Optional target size for long edge (maintains aspect)
        verify_linearity: Whether to run linear verification checks (default True)
        apex_strict_formats: Whether to enforce RAW + TIFF only (default True).
                            Set to False only if you understand the data-fidelity tradeoffs.

    Returns:
        Tuple of:
            - Processed image (float32, HxWx3, [0, 1], linear light)
            - Original shape (H, W) before any resizing

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If image invalid, gamma-encoded, unsupported format, or violates format boundary
        DtypeViolationError: If dtype is uint8/uint16 after conversion
        RangeViolationError: If values outside [0, 1]
        LinearityViolationError: If gamma encoding detected

    Example:
        >>> # Load linear RAW file (default behavior)
        >>> img, orig_shape = preprocess_image_linear("photo.CR2")
        >>> # img is float32 [0, 1] linear light, verified

        >>> # Load 16-bit linear TIFF (default behavior)
        >>> img, orig_shape = preprocess_image_linear("render.tif")
        >>> # img is float32 [0, 1] linear light, preserving TIFF precision

        >>> # JPEG rejected by default (format boundary enforcement)
        >>> img, orig_shape = preprocess_image_linear("photo.jpg")
        >>> # → ValueError: APEX linear ingest only supports RAW + TIFF inputs

        >>> # Explicit escape hatch (discouraged)
        >>> img, orig_shape = preprocess_image_linear("photo.jpg", apex_strict_formats=False)
        >>> # Allowed but may violate linear-light preservation
    """
    from .linear_verify import verify_linear_ingest
    from .raw_loader import load_raw_as_rgb

    # Load image preserving bit depth and linearity
    if isinstance(image, (str, Path)):
        image_path = validate_image_format(image)

        # APEX ingest boundary: deterministic by format
        # Training ingest must be scene-referred by construction.
        # RAW files are inherently linear (sensor data).
        # TIFF files preserve bit depth and can carry linear data.
        # JPEG/PNG are display-referred and typically gamma-encoded.
        if apex_strict_formats and (not is_raw_file(image_path)) and (image_path.suffix.lower() not in {".tif", ".tiff"}):
            raise ValueError(
                f"APEX linear ingest only supports RAW + TIFF inputs. "
                f"Got: {image_path.suffix.lower()}.\n"
                "Reason: JPEG/PNG are typically gamma-encoded (display-referred), "
                "violating linear-light preservation requirements.\n"
                "Use preprocess_image() for JPEG/PNG, or explicitly set "
                "apex_strict_formats=False if you understand the data-fidelity tradeoffs."
            )

        # Handle RAW files with linear output
        if is_raw_file(image_path):
            logger.debug(f"Loading RAW file with linear output: {image_path.name}")
            # load_raw_as_rgb with output_linear=True (default) gives uint16 linear
            rgb_array = load_raw_as_rgb(
                image_path,
                use_camera_wb=True,
                half_size=False,
                output_bps=16,  # 16-bit for precision
                output_linear=True,  # Linear RGB (enforced)
            )
            original_h, original_w = rgb_array.shape[:2]

            # Convert uint16 [0, 65535] → float32 [0, 1] preserving linearity
            img_array = rgb_array.astype(np.float32) / 65535.0

        else:
            # Handle standard formats (TIFF, PNG, JPEG)
            # Use tifffile for TIFF to preserve 16-bit
            if image_path.suffix.lower() in {".tif", ".tiff"}:
                try:
                    import tifffile
                except ImportError as e:
                    raise ImportError(
                        "tifffile is required for linear TIFF processing in APEX pipeline. "
                        "Install with: pip install tifffile\n"
                        "Or: pip install -e '.[tiff]'"
                    ) from e

                # Load TIFF preserving bit depth
                tiff_array = tifffile.imread(str(image_path))
                original_h, original_w = tiff_array.shape[0], tiff_array.shape[1]  # NumPy: (H, W, C)

                # Normalize based on dtype
                if tiff_array.dtype == np.uint8:
                    img_array = tiff_array.astype(np.float32) / 255.0
                elif tiff_array.dtype == np.uint16:
                    img_array = tiff_array.astype(np.float32) / 65535.0
                elif tiff_array.dtype in [np.float32, np.float64]:
                    img_array = tiff_array.astype(np.float32)
                    # Assume already in [0, 1] if float
                else:
                    raise ValueError(f"Unsupported TIFF dtype: {tiff_array.dtype}")

                # Ensure 3 channels
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] == 4:
                    # Drop alpha
                    img_array = img_array[:, :, :3]
            else:
                # Use PIL for other formats (JPEG, PNG, etc.)
                pil_img = Image.open(image_path).convert("RGB")
                original_w, original_h = pil_img.size  # PIL returns (W, H)
                img_array = np.array(pil_img, dtype=np.float32) / 255.0

    elif isinstance(image, np.ndarray):
        # Handle numpy array input
        original_h, original_w = image.shape[0], image.shape[1]  # NumPy: (H, W, C)

        if image.dtype == np.uint8:
            img_array = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_array = image.astype(np.float32) / 65535.0
        elif image.dtype in [np.float32, np.float64]:
            img_array = image.astype(np.float32)
        else:
            raise ValueError(f"Unsupported array dtype: {image.dtype}")

        # Ensure 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        elif img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Unsupported array shape: {image.shape}")
    else:
        raise TypeError(f"Image must be np.ndarray, Path, or str. Got: {type(image)}")

    # Verify linearity before any further processing
    if verify_linearity:
        try:
            verify_linear_ingest(img_array)
            logger.debug("Linear ingest verification passed")
        except Exception as e:
            raise ValueError(
                f"Linear ingest verification failed: {e}\n"
                f"Image: {image if isinstance(image, (str, Path)) else 'numpy array'}\n"
                f"This indicates the input is gamma-encoded, has incorrect dtype, or invalid range."
            ) from e

    # Optional resize to target size (long edge)
    if target_size is not None:
        h, w = img_array.shape[:2]
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))

        # Resize using high-quality interpolation
        # Use cv2 to avoid lossy float32→uint8→float32 conversion
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is required for resizing in linear space for APEX pipeline. "
                "Install with: pip install opencv-python\n"
                "Or: pip install -e '.[cv2]'"
            ) from e

        # cv2.resize preserves dtype (float32 stays float32)
        # Use INTER_LANCZOS4 for high-quality resampling
        img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Clip to [0, 1] after resize (interpolation can introduce small out-of-range values)
        img_array = np.clip(img_array, 0.0, 1.0)

    # Enforce multiple-of-14 dimensions
    img_array = _enforce_dimension_multiple(img_array, DIMENSION_MULTIPLE)

    # Final verification after all processing
    if verify_linearity:
        verify_linear_ingest(img_array)

    return img_array, (original_h, original_w)


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


def validate_depth_image_alignment(image_path: Path, depth_path: Path) -> bool:
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
