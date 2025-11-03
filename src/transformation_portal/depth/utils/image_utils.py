"""
Image utility functions for depth pipeline.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(
    path: Union[str, Path],
    color_space: str = "RGB",
    dtype: str = "float32",
    normalize: bool = True,
) -> np.ndarray:
    """
    Load image from file.

    Args:
        path: Image file path
        color_space: Color space ('RGB', 'BGR', 'GRAY')
        dtype: Output dtype ('float32', 'float64', 'uint8')
        normalize: Normalize to [0, 1] for float dtypes

    Returns:
        Image as numpy array
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Load with PIL for better format support
    try:
        img = Image.open(path)

        # Convert color space
        if color_space == "RGB":
            img = img.convert("RGB")
        elif color_space == "GRAY":
            img = img.convert("L")

        # Convert to numpy
        image = np.array(img)

    except Exception as e:
        logger.warning(f"PIL failed ({e}), trying OpenCV")
        # Fallback to OpenCV
        if color_space == "GRAY":
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if color_space == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert dtype
    if dtype in ["float32", "float64"]:
        image = image.astype(dtype)
        if normalize and image.max() > 1.0:
            image = image / 255.0
    elif dtype == "uint8":
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

    logger.debug(f"Loaded image: {path.name} {image.shape} {image.dtype}")

    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
    color_space: str = "RGB",
) -> Path:
    """
    Save image to file.

    Args:
        image: Image array
        path: Output path
        quality: JPEG quality (1-100)
        color_space: Input color space ('RGB', 'BGR')

    Returns:
        Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Convert color space for OpenCV if needed
    if color_space == "RGB" and image.ndim == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    # Determine format
    ext = path.suffix.lower()

    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(
            str(path),
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
    elif ext == '.png':
        cv2.imwrite(
            str(path),
            image_bgr,
            [cv2.IMWRITE_PNG_COMPRESSION, 9]
        )
    elif ext in ['.tif', '.tiff']:
        cv2.imwrite(str(path), image_bgr)
    else:
        # Fallback to PIL
        if color_space == "RGB":
            img = Image.fromarray(image)
        else:
            img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        img.save(path, quality=quality)

    logger.debug(f"Saved image: {path}")

    return path


def resize_image(
    image: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
    max_size: Optional[int] = None,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize image.

    Args:
        image: Input image
        size: Target (height, width). Overrides scale and max_size.
        scale: Scale factor. Overrides max_size.
        max_size: Maximum dimension. Only used if size and scale are None.
        interpolation: Interpolation method ('bilinear', 'bicubic', 'lanczos', 'nearest')

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    # Determine target size
    if size is not None:
        target_h, target_w = size
    elif scale is not None:
        target_h = int(h * scale)
        target_w = int(w * scale)
    elif max_size is not None:
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            target_h = int(h * scale)
            target_w = int(w * scale)
        else:
            return image
    else:
        return image

    # Interpolation method
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
    }
    interp_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)

    # Resize
    resized = cv2.resize(
        image,
        (target_w, target_h),
        interpolation=interp_flag
    )

    return resized


def compute_image_hash(image: np.ndarray, method: str = "md5") -> str:
    """
    Compute hash of image content.

    Args:
        image: Input image
        method: Hash method ('md5', 'sha256')

    Returns:
        Hash string (hex)
    """
    # Normalize to uint8 for consistent hashing
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image_bytes = (image * 255).astype(np.uint8).tobytes()
        else:
            image_bytes = image.astype(np.uint8).tobytes()
    else:
        image_bytes = image.tobytes()

    # Compute hash
    if method == "md5":
        hash_obj = hashlib.md5(image_bytes)
    elif method == "sha256":
        hash_obj = hashlib.sha256(image_bytes)
    else:
        raise ValueError(f"Unknown hash method: {method}")

    return hash_obj.hexdigest()


def pad_to_multiple(
    image: np.ndarray,
    multiple: int = 32,
    mode: str = "reflect",
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image to multiple of given size (required for some models).

    Args:
        image: Input image
        multiple: Pad to multiple of this value
        mode: Padding mode ('reflect', 'constant', 'edge')

    Returns:
        Padded image and padding amounts (top, bottom, left, right)
    """
    h, w = image.shape[:2]

    # Compute padding
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad
    if image.ndim == 2:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=mode
        )
    else:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=mode
        )

    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad_image(
    image: np.ndarray,
    padding: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Remove padding added by pad_to_multiple.

    Args:
        image: Padded image
        padding: Padding amounts (top, bottom, left, right)

    Returns:
        Unpadded image
    """
    top, bottom, left, right = padding

    if bottom == 0:
        bottom = None
    else:
        bottom = -bottom

    if right == 0:
        right = None
    else:
        right = -right

    if image.ndim == 2:
        return image[top:bottom, left:right]
    else:
        return image[top:bottom, left:right, :]


def apply_gamma_correction(
    image: np.ndarray,
    gamma: float = 2.2,
) -> np.ndarray:
    """
    Apply gamma correction to image.

    Args:
        image: Input image [0, 1]
        gamma: Gamma value (>1 = darken, <1 = brighten)

    Returns:
        Gamma-corrected image
    """
    return np.power(np.clip(image, 0, 1), 1.0 / gamma)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB to sRGB.

    Args:
        image: Linear RGB image [0, 1]

    Returns:
        sRGB image [0, 1]
    """
    # sRGB transfer function
    linear_mask = image <= 0.0031308
    srgb = np.where(
        linear_mask,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    )

    return srgb


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """
    Convert sRGB to linear RGB.

    Args:
        image: sRGB image [0, 1]

    Returns:
        Linear RGB image [0, 1]
    """
    # Inverse sRGB transfer function
    srgb_mask = image <= 0.04045
    linear = np.where(
        srgb_mask,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4)
    )

    return linear
