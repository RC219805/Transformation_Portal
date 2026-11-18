"""
Common image I/O utilities.

This module consolidates image loading and conversion functions that were
previously duplicated across multiple files (lux_render_pipeline.py,
depth_tools.py, depth_pipeline/utils/image_utils.py).
"""

import functools
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from PIL import Image

# Adobe Cube LUT Specification 1.0: LUT_3D_SIZE must lie in [2, 256].
CUBE_LUT_3D_MIN_SIZE = 2
CUBE_LUT_3D_MAX_SIZE = 256


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load path into an RGB Image instance.

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB mode
    """
    img = Image.open(path).convert("RGB")
    return img


def extract_output_metadata(img: Image.Image) -> Dict[str, Any]:
    """Collect the ICC profile, DPI and EXIF of a loaded image for re-attachment on save.

    The returned mapping holds only keys that are present and usable, named after the
    Pillow save keywords (``icc_profile``, ``dpi``, ``exif``). The ICC profile is kept
    only when it describes an RGB colour space: pipeline outputs are RGB, so a CMYK or
    greyscale profile carried over from a ``convert("RGB")`` source would mis-describe them.

    Args:
        img: Image as loaded (``convert`` keeps ``info``, so a converted image works too)

    Returns:
        Dict suitable for ``Image.save(path, **metadata)``
    """
    metadata: Dict[str, Any] = {}
    icc = img.info.get("icc_profile")
    if isinstance(icc, bytes) and len(icc) >= 20 and icc[16:20] == b"RGB ":
        metadata["icc_profile"] = icc
    dpi = img.info.get("dpi")
    if (
        isinstance(dpi, tuple)
        and len(dpi) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in dpi)
    ):
        metadata["dpi"] = (dpi[0], dpi[1])
    exif_bytes = img.info.get("exif")
    if isinstance(exif_bytes, bytes) and exif_bytes:
        try:
            exif = Image.Exif()
            exif.load(exif_bytes)
        except Exception:  # noqa: BLE001 - malformed EXIF is dropped, not fatal
            exif = None
        if exif is not None and len(exif) > 0:
            metadata["exif"] = exif
    return metadata


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


def _parse_cube_lut(path: Path) -> np.ndarray:
    """Parse a 3D ``.cube`` file into an array indexed ``lut[r, g, b]``.

    ``.cube`` data rows list the red index fastest, then green, then blue, so a
    plain ``reshape(N, N, N, 3)`` is indexed ``[b, g, r]``; the transpose puts
    the axes in channel order.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is not a valid 3D ``.cube`` LUT.
    """
    lut_size = 0
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                lut_size = int(line.split()[-1])
                continue
            if line.startswith(("TITLE", "DOMAIN_", "LUT_")):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    rows.append([float(part) for part in parts])
                except ValueError:
                    continue

    if not CUBE_LUT_3D_MIN_SIZE <= lut_size <= CUBE_LUT_3D_MAX_SIZE:
        raise ValueError(f"LUT_3D_SIZE must be between {CUBE_LUT_3D_MIN_SIZE} and {CUBE_LUT_3D_MAX_SIZE}, got {lut_size}")
    if len(rows) != lut_size**3:
        raise ValueError(f"Invalid LUT data: size={lut_size}, expected {lut_size**3} entries, got {len(rows)}")

    table = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(table).all():
        raise ValueError("LUT contains non-finite entries")
    return np.ascontiguousarray(table.reshape(lut_size, lut_size, lut_size, 3).transpose(2, 1, 0, 3))


@functools.lru_cache(maxsize=4)
def _load_cube_lut_cached(path_str: str, mtime_ns: int, size_bytes: int) -> np.ndarray:
    del mtime_ns, size_bytes  # cache-key components only
    lut = _parse_cube_lut(Path(path_str))
    lut.setflags(write=False)
    return lut


def load_cube_lut(path: Union[str, Path]) -> np.ndarray:
    """Load a 3D ``.cube`` LUT as a read-only float32 array indexed ``lut[r, g, b]``.

    Parsed tables are cached per resolved path, modification time and size, so a
    batch applying the same LUT to many images parses it once.

    Raises:
        OSError: The file is missing or unreadable.
        ValueError: The file is not a valid 3D ``.cube`` LUT.
    """
    resolved = Path(path).resolve()
    stat_result = resolved.stat()
    return _load_cube_lut_cached(str(resolved), stat_result.st_mtime_ns, stat_result.st_size)


def apply_cube_lut(image: np.ndarray, lut: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply a 3D LUT indexed ``lut[r, g, b]`` with trilinear interpolation.

    Args:
        image: Float image in [0, 1] with shape (..., 3); values outside are clipped.
        lut: Table of shape (N, N, N, 3) as returned by :func:`load_cube_lut`.
        strength: Blend factor between the input (0.0) and the LUT output (1.0).

    Returns:
        Float32 image clipped to [0, 1] with the same shape as ``image``.
    """
    if lut.ndim != 4 or lut.shape[3] != 3 or not lut.shape[0] == lut.shape[1] == lut.shape[2]:
        raise ValueError(f"LUT must have shape (N, N, N, 3), got {lut.shape}")
    lut_size = lut.shape[0]
    if lut_size < CUBE_LUT_3D_MIN_SIZE:
        raise ValueError(f"LUT size must be at least {CUBE_LUT_3D_MIN_SIZE}, got {lut_size}")
    if image.ndim < 1 or image.shape[-1] != 3:
        raise ValueError(f"Image must have 3 colour channels in the last axis, got shape {image.shape}")

    array = np.clip(image, 0, 1).astype(np.float32)
    indices = array * np.float32(lut_size - 1)
    # Clamp the lower cell corner to N - 2 so that 1.0 lands exactly on the last grid node (frac == 1).
    lower_f = np.minimum(np.floor(indices), np.float32(lut_size - 2))
    frac = indices - lower_f
    lower = lower_f.astype(np.int32)
    upper = lower + 1

    corners = ((lower, 1.0 - frac), (upper, frac))
    graded = np.zeros_like(array)
    for r_idx, r_weight in corners:
        for g_idx, g_weight in corners:
            for b_idx, b_weight in corners:
                weight = r_weight[..., 0:1] * g_weight[..., 1:2] * b_weight[..., 2:3]
                graded += lut[r_idx[..., 0], g_idx[..., 1], b_idx[..., 2]] * weight

    result = array * (1.0 - strength) + graded * strength
    return np.clip(result, 0, 1).astype(np.float32)
