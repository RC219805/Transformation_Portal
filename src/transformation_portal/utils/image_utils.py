"""
Common image I/O utilities.

This module consolidates image loading and conversion functions that were
previously duplicated across multiple files (lux_render_pipeline.py,
depth_tools.py, depth_pipeline/utils/image_utils.py).
"""
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load path into an RGB Image instance.

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB mode
    """
    img = Image.open(path).convert("RGB")
    return img


def save_image(img: Image.Image, path: Union[str, Path]) -> None:
    """Persist img to path, creating parent directories when missing.

    Args:
        img: PIL Image to save
        path: Destination path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def pil_to_np(img: Image.Image, to_float: bool = True) -> np.ndarray:
    """Convert a PIL image to a NumPy array optionally scaled to [0, 1].

    Args:
        img: PIL Image to convert
        to_float: If True, scale to [0, 1] range as float32

    Returns:
        NumPy array representation of the image
    """
    arr = np.array(img)
    if to_float:
        arr = arr.astype(np.float32) / 255.0
    return arr


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a float array in [0, 1] back to an 8-bit Image.

    Args:
        arr: NumPy array in [0, 1] range

    Returns:
        PIL Image with uint8 values
    """
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def load_image_rgb(path: Union[str, Path]) -> np.ndarray:
    """Load path into a float32 RGB NumPy array normalized to [0, 1].

    This is a convenience function combining load_image and pil_to_np.

    Args:
        path: Path to image file

    Returns:
        Float32 NumPy array in [0, 1] range with shape (H, W, 3)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = load_image(path)
    return pil_to_np(img, to_float=True)
