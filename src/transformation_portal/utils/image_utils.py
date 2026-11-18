"""
Common image I/O utilities.

This module consolidates image loading and conversion functions that were
previously duplicated across multiple files (lux_render_pipeline.py,
depth_tools.py, depth_pipeline/utils/image_utils.py).
"""

import functools
import hashlib
import logging
import threading
import unicodedata
from pathlib import Path
from typing import Any, ContextManager, Dict, Tuple, Union

import numpy as np
from PIL import Image

from transformation_portal.lux_depth_v3.io_atomic import atomic_temp_file

log = logging.getLogger(__name__)

# Adobe Cube LUT Specification 1.0: LUT_3D_SIZE must lie in [2, 256].
CUBE_LUT_3D_MIN_SIZE = 2
CUBE_LUT_3D_MAX_SIZE = 256

# Largest EXIF payload a JPEG APP1 segment holds (Pillow refuses to save more).
_JPEG_MAX_EXIF_BYTES = 65533

# Pillow modes holding 16-bit unsigned greyscale (16-bit PNG, TIFF and JPEG 2000).
_GREY16_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N"})
_GREY16_MAX = 65535


def _is_grey16(img: Image.Image) -> bool:
    """Return True when ``img`` carries 16-bit greyscale with a 0..65535 full scale.

    Besides the ``I;16*`` modes this covers 16-bit PGM/PPM files, which Pillow
    decodes as mode ``I`` with values rescaled to 0..65535. Other ``I`` images
    (32-bit integer TIFFs) have no agreed full scale and are left to Pillow.
    """
    return img.mode in _GREY16_MODES or (img.mode == "I" and img.format == "PPM")


def _grey16_to_8bit(value: np.ndarray) -> np.ndarray:
    """Map 16-bit greyscale values onto 0..255 with rounding (``round(v / 257)``)."""
    clipped = np.clip(np.asarray(value), 0, _GREY16_MAX).astype(np.uint32)
    return ((clipped + 128) // 257).astype(np.uint8)


def convert_to_rgb(img: Image.Image) -> Image.Image:
    """Convert ``img`` to 8-bit RGB, scaling 16-bit greyscale instead of clipping it.

    ``Image.convert("RGB")`` clips 16-bit greyscale values at 255, so a 16-bit
    greyscale PNG/TIFF/PGM comes back almost pure white, and it cannot convert
    big-endian ``I;16B`` TIFFs at all. Such images are mapped from their 0..65535
    full scale to 0..255; ``info`` (ICC profile, DPI, EXIF) is kept and an integer
    ``transparency`` key is scaled the same way. Every other mode uses Pillow's
    conversion unchanged.

    Args:
        img: PIL Image in any mode

    Returns:
        PIL Image in RGB mode
    """
    if not _is_grey16(img):
        return img.convert("RGB")

    rgb = Image.fromarray(_grey16_to_8bit(np.asarray(img))).convert("RGB")
    info = dict(img.info)
    transparency = info.pop("transparency", None)
    if isinstance(transparency, int):
        level = int(_grey16_to_8bit(np.asarray(transparency)))
        info["transparency"] = (level, level, level)
    rgb.info.update(info)
    return rgb


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load path into an RGB Image instance.

    16-bit greyscale inputs are scaled to 8 bits (see :func:`convert_to_rgb`).

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB mode
    """
    img = convert_to_rgb(Image.open(path))
    return img


def extract_output_metadata(img: Image.Image) -> Dict[str, Any]:
    """Collect the ICC profile, DPI and EXIF of a loaded image for re-attachment on save.

    The returned mapping holds only keys that are present and usable, named after the
    Pillow save keywords (``icc_profile``, ``dpi``, ``exif``). The ICC profile is kept
    only when it describes an RGB colour space: pipeline outputs are RGB, so a CMYK or
    greyscale profile carried over from a ``convert("RGB")`` source would mis-describe them.
    EXIF larger than a JPEG APP1 segment holds is dropped, since saving it as JPEG raises.

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
            exif_size = len(exif.tobytes())
        except Exception:  # noqa: BLE001 - malformed EXIF is dropped, not fatal
            exif = None
            exif_size = 0
        if exif is not None and exif_size > _JPEG_MAX_EXIF_BYTES:
            # A JPEG APP1 segment cannot hold it and Pillow raises on save, which would
            # fail an image that processed fine; drop the EXIF rather than the output.
            log.warning("Source EXIF is %d bytes, over the JPEG limit; it is not carried to outputs", exif_size)
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
    with atomic_output_path(path) as temp_path:
        img.save(temp_path)


def atomic_output_path(path: Union[str, Path]) -> ContextManager[Path]:
    """Yield a temporary sibling of ``path`` that is moved onto it with ``os.replace``.

    The temporary name keeps the destination suffix, so path-based writers (Pillow,
    tifffile) infer the same format. A failed write leaves the destination untouched
    and removes the temporary file; parent directories are created when missing.
    """
    destination = Path(path)
    return atomic_temp_file(destination, suffix=destination.suffix, prefix=f".{destination.stem}.")


def _stem_collision_key(stem: str) -> str:
    """Case- and Unicode-normalised key: output names that differ only so collide on macOS."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", stem).casefold())


class OutputStemAllocator:
    """Assign each distinct batch input an output stem no other input of the batch uses.

    Output names are derived from ``input_path.stem``, so ``kitchen.png`` and
    ``kitchen.tif`` (or ``a/kitchen.png`` and ``b/kitchen.png``) would write the same
    deliverables and silently overwrite each other. The first input to claim a stem keeps
    it unchanged; a later, different input whose stem collides (ignoring case) gets
    ``<stem>_<ext>``, then ``<stem>_<ext>_<path digest>``, so only names that would collide
    change. Claiming the same input again returns the same stem. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stem_by_input: Dict[str, str] = {}
        self._input_by_key: Dict[str, str] = {}

    def claim(self, input_path: Union[str, Path]) -> str:
        """Return the output stem reserved for ``input_path`` in this batch."""
        source = Path(input_path)
        identity = str(source.resolve(strict=False))
        with self._lock:
            existing = self._stem_by_input.get(identity)
            if existing is not None:
                return existing
            stem = source.stem
            ext = source.suffix.lstrip(".").lower()
            digest = hashlib.sha256(identity.encode("utf-8", "surrogateescape")).hexdigest()[:8]
            candidates = [stem]
            if ext:
                candidates.append(f"{stem}_{ext}")
            candidates.append(f"{candidates[-1]}_{digest}")
            counter = 2
            while True:
                for candidate in candidates:
                    key = _stem_collision_key(candidate)
                    if key not in self._input_by_key:
                        self._input_by_key[key] = identity
                        self._stem_by_input[identity] = candidate
                        if candidate != stem:
                            log.warning(
                                "Output name %r is already used by another input of this batch; "
                                "writing outputs of %s as %r",
                                stem,
                                source,
                                candidate,
                            )
                        return candidate
                candidates = [f"{stem}_{ext or 'input'}_{digest}_{counter}"]
                counter += 1


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


def fit_size_within(size: Tuple[int, int], bounds: Tuple[int, int]) -> Tuple[int, int]:
    """Scale ``size`` to fit ``bounds`` keeping the aspect ratio.

    The binding edge lands exactly on its bound and the other edge is rounded
    to the nearest pixel (at least 1, at most its bound). Computing both edges
    as ``int(edge * ratio)`` instead truncates, so float error leaves the
    binding edge one pixel short of the target for many sizes.

    Args:
        size: Source ``(width, height)`` in pixels (both positive)
        bounds: Target ``(max_width, max_height)`` in pixels (both positive)

    Returns:
        New ``(width, height)``
    """
    width, height = int(size[0]), int(size[1])
    max_w, max_h = int(bounds[0]), int(bounds[1])
    if width <= 0 or height <= 0 or max_w <= 0 or max_h <= 0:
        raise ValueError(f"fit_size_within needs positive sizes, got size={size!r} bounds={bounds!r}")
    # Compare max_w/width <= max_h/height exactly in integers.
    if max_w * height <= max_h * width:
        return max_w, min(max_h, max(1, round(height * max_w / width)))
    return min(max_w, max(1, round(width * max_h / height))), max_h


def quantize_unit_float(arr: np.ndarray, dtype: type = np.uint8) -> np.ndarray:
    """Quantise a float array in [0, 1] to ``dtype`` (uint8 or uint16) by rounding.

    ``(arr * 255).astype(np.uint8)`` truncates, which biases every conversion
    half a level dark; repeated across processing stages the bias accumulates.
    Values outside [0, 1] are clipped first so they cannot wrap around.
    """
    target = np.dtype(dtype)
    if target not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError(f"quantize_unit_float supports uint8 and uint16, got {target}")
    max_value = float(np.iinfo(target).max)
    return np.rint(np.clip(arr, 0.0, 1.0) * max_value).astype(target)


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
