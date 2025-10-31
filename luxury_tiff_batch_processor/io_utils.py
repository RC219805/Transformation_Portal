"""I/O primitives and capability detection for high-fidelity TIFF processing.

This module handles image loading, saving, and format conversion with support for
16-bit precision, HDR workflows, and comprehensive metadata preservation. It provides
capability detection for optional dependencies and falls back gracefully when needed.

Key Components
--------------

ProcessingCapabilities
    Detects available I/O capabilities (16-bit support, HDR, compression formats).

FloatDynamicRange
    Tracks normalization parameters for float image data to preserve dynamic range.

ImageToFloatResult
    Container for image conversion results with metadata and alpha channel info.

ProcessingContext
    Context manager for atomic file operations with staged writes.

Functions
---------

image_to_float
    Convert PIL Image to normalized float32 RGB array.

float_to_dtype_array
    Convert float array back to target dtype with proper scaling.

save_image
    Save image with comprehensive TIFF metadata and compression support.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:  # Optional high-fidelity TIFF writer
    import tifffile  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    tifffile = None

try:  # Optional codec pack used by tifffile for certain compressions
    import imagecodecs  # type: ignore  # pylint: disable=import-error
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    imagecodecs = None

LOGGER = logging.getLogger("luxury_tiff_batch_processor")


class LuxuryGradeException(RuntimeError):
    """Raised when the processing environment cannot meet luxury standards."""


class ProcessingCapabilities:
    """Detects available image processing capabilities based on installed dependencies.

    Attributes:
        bit_depth: Maximum supported bit depth (8 or 16).
        hdr_capable: Whether HDR workflows are supported.
    """

    _SENTINEL = object()

    def __init__(self, tifffile_module: Any | None | object = _SENTINEL) -> None:
        """Initialize capability detection.

        Args:
            tifffile_module: Optional tifffile module for testing, defaults to global import.
        """
        if tifffile_module is self._SENTINEL:
            self._tifffile = tifffile
        else:
            self._tifffile = tifffile_module
        self.bit_depth = 16 if self._supports_16_bit_output() else 8
        self.hdr_capable = self._detect_hdr_support()

    def _supports_16_bit_output(self) -> bool:
        """Return ``True`` when a writer capable of 16-bit output is available."""

        return bool(getattr(self._tifffile, "imwrite", None))

    def _detect_hdr_support(self) -> bool:
        """Best-effort check for HDR support given optional dependencies."""

        if not self._supports_16_bit_output():
            return False

        supports_hdr = getattr(self._tifffile, "supports_hdr", True)
        try:
            return bool(supports_hdr)
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            return False

    def assert_luxury_grade(self) -> None:
        """Validate that the environment meets luxury-grade requirements.

        Raises:
            LuxuryGradeException: If 16-bit support or HDR capability is missing.
        """
        if self.bit_depth < 16:
            raise LuxuryGradeException(
                "Material Response requires 16-bit precision. "
                "Install tifffile to unlock quantum color depth."
            )
        if not self.hdr_capable:
            raise LuxuryGradeException(
                "Luxury-grade processing requires HDR capability. "
                "Install tifffile or ensure your environment supports HDR."
            )


@dataclasses.dataclass
class ProcessingContext:
    """Context manager for atomic file writes using staged temporary files.

    Writes to a temporary file in the same directory as the destination, then
    atomically moves it to the final location on success. Cleans up temporary
    files on failure.

    Attributes:
        destination: Final output file path.
        suffix: Temporary file suffix (default: ".tmp").
    """

    destination: Path
    suffix: str = ".tmp"

    def __post_init__(self) -> None:
        self._staged_path: Optional[Path] = None

    def _temp_path(self) -> Path:
        unique = uuid.uuid4().hex
        name = f".{self.destination.name}{self.suffix}-{unique}"
        return self.destination.parent / name

    def __enter__(self) -> Path:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._staged_path = self._temp_path()
        return self._staged_path

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._staged_path is None:
            return False

        staged = self._staged_path
        self._staged_path = None

        if exc_type is None:
            try:
                os.replace(staged, self.destination)
            except Exception:
                with contextlib.suppress(Exception):
                    staged.unlink()
                raise
        else:
            with contextlib.suppress(FileNotFoundError):
                staged.unlink()
        return False


@dataclasses.dataclass(frozen=True)
class FloatDynamicRange:
    """Tracks normalization parameters for floating-point image data.

    Preserves per-channel offset and scale to enable reversible normalization
    of floating-point images that may exceed the [0, 1] range.

    Attributes:
        offset: Per-channel minimum values.
        scale: Per-channel range (max - min).
        scale_recip: Reciprocal of scale for efficient normalization.
    """

    offset: np.ndarray
    scale: np.ndarray
    scale_recip: np.ndarray

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Optional["FloatDynamicRange"]:
        """Analyze array to create normalization descriptor.

        Args:
            arr: Input array to analyze.

        Returns:
            FloatDynamicRange instance, or None if array is empty or has no finite values.
        """
        if arr.size == 0:
            return None

        if arr.ndim == 2:
            working = arr[:, :, None]
        else:
            working = arr

        channels = working.shape[-1]
        flattened = working.reshape(-1, channels)

        offsets = np.zeros(channels, dtype=np.float32)
        scales = np.ones(channels, dtype=np.float32)
        saw_finite = False

        for idx in range(channels):
            channel = flattened[:, idx]
            finite = channel[np.isfinite(channel)]
            if finite.size == 0:
                offsets[idx] = 0.0
                scales[idx] = 1.0
                continue

            saw_finite = True
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
            offsets[idx] = np.float32(min_val)
            diff = float(max_val - min_val)
            if diff <= 0.0 or not math.isfinite(diff):
                scales[idx] = 1.0
            else:
                scales[idx] = np.float32(diff)

        if not saw_finite:
            return None

        scale_recip = np.where(scales == 0.0, 1.0, 1.0 / scales).astype(np.float32)
        return cls(offset=offsets, scale=scales.astype(np.float32), scale_recip=scale_recip)

    @staticmethod
    def _prepare(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
        working = np.asarray(arr, dtype=np.float32)
        squeezed = False
        if working.ndim == 2:
            working = working[:, :, None]
            squeezed = True
        return working, squeezed

    def normalise(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array values to [0, 1] range using stored parameters.

        Args:
            arr: Input array to normalize.

        Returns:
            Normalized array with values in [0, 1] range.
        """
        working, squeezed = self._prepare(arr)
        offset = self.offset.reshape((1, 1, -1))
        scale = self.scale_recip.reshape((1, 1, -1))
        normalised = (working - offset) * scale
        if squeezed:
            return normalised[:, :, 0]
        return normalised

    def denormalise(self, arr: np.ndarray) -> np.ndarray:
        """Restore array to original dynamic range using stored parameters.

        Args:
            arr: Normalized array in [0, 1] range.

        Returns:
            Array restored to original scale and offset.
        """
        working, squeezed = self._prepare(arr)
        offset = self.offset.reshape((1, 1, -1))
        scale = self.scale.reshape((1, 1, -1))
        restored = working * scale + offset
        if squeezed:
            return restored[:, :, 0]
        return restored


@dataclasses.dataclass(frozen=True)
class ImageToFloatResult:
    """Results from image_to_float with metadata for reversible conversion.

    Attributes:
        array: RGB float32 array in [0, 1] range.
        dtype: Original image dtype for round-trip conversion.
        alpha: Optional alpha channel as float32 array.
        base_channels: Number of color channels in original image.
        float_normalisation: Dynamic range info for floating-point sources.
    """

    array: np.ndarray
    dtype: np.dtype
    alpha: Optional[np.ndarray]
    base_channels: int
    float_normalisation: Optional[FloatDynamicRange] = None

    def __iter__(self):
        yield self.array
        yield self.dtype
        yield self.alpha
        yield self.base_channels


def image_to_float(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    image: Image.Image,
    return_format: Literal["tuple3", "tuple4", "object"] = "object",
) -> Union[
    Tuple[np.ndarray, np.dtype, Optional[np.ndarray]],
    Tuple[np.ndarray, np.dtype, Optional[np.ndarray], int],
    ImageToFloatResult,
]:
    """Convert a PIL image to an RGB float32 array in the 0-1 range."""

    supported_modes = {
        "RGB",
        "RGBA",
        "I",
        "I;16",
        "I;16L",
        "I;16B",
        "I;16S",
        "F",
        "L",
        "LA",
    }
    if image.mode not in supported_modes:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

    arr = np.array(image)
    alpha_channel: Optional[np.ndarray] = None

    if arr.ndim == 2:
        base_channels = 1
        color_data = arr[:, :, None]
    else:
        if arr.shape[2] == 4:
            alpha_channel = arr[:, :, 3]
            base_channels = 3
            color_data = arr[:, :, :3]
        elif arr.shape[2] == 2:
            alpha_channel = arr[:, :, 1]
            base_channels = 1
            color_data = arr[:, :, :1]
        else:
            base_channels = arr.shape[2]
            color_data = arr[:, :, :base_channels]

    color_float = color_data.astype(np.float32, copy=False)
    alpha_float = alpha_channel.astype(np.float32, copy=False) if alpha_channel is not None else None

    float_norm: Optional[FloatDynamicRange] = None
    if np.issubdtype(color_data.dtype, np.integer):
        dtype_info = np.iinfo(color_data.dtype)
        scale = dtype_info.max - dtype_info.min
        if scale == 0:
            normalised = np.zeros_like(color_float, dtype=np.float32)
            if alpha_float is not None:
                alpha_float = np.zeros_like(alpha_float, dtype=np.float32)
        else:
            offset = float(dtype_info.min)
            inv_scale = 1.0 / float(scale)
            normalised = (color_float - offset) * inv_scale
            if alpha_float is not None:
                alpha_float = (alpha_float - offset) * inv_scale
        normalised = np.clip(normalised, 0.0, 1.0)
        if alpha_float is not None:
            alpha_float = np.clip(alpha_float, 0.0, 1.0)
    else:
        float_norm = FloatDynamicRange.from_array(color_float)
        if float_norm is not None:
            normalised = float_norm.normalise(color_float)
        else:
            normalised = color_float
        normalised = np.clip(normalised, 0.0, 1.0)
        if alpha_float is not None:
            alpha_float = np.clip(alpha_float, 0.0, 1.0)

    if base_channels == 1:
        working = np.repeat(normalised, 3, axis=2)
    else:
        working = normalised

    result = ImageToFloatResult(
        array=np.ascontiguousarray(working, dtype=np.float32),
        dtype=np.dtype(color_data.dtype),
        alpha=None if alpha_float is None else np.ascontiguousarray(alpha_float, dtype=np.float32),
        base_channels=base_channels,
        float_normalisation=float_norm,
    )

    if return_format == "tuple3":
        return result.array, result.dtype, result.alpha
    if return_format == "tuple4":
        return result.array, result.dtype, result.alpha, result.base_channels
    allowed_formats = ("object", "tuple3", "tuple4")
    if return_format not in allowed_formats:
        raise ValueError(
            f"Unsupported return_format: {return_format}. Allowed values are: {allowed_formats}"
        )
    return result


def float_to_dtype_array(  # pylint: disable=too-many-branches
    arr: np.ndarray,
    dtype: np.dtype,
    alpha: Optional[np.ndarray],
    base_channels: Optional[int] = None,
    *,
    float_normalisation: Optional[FloatDynamicRange] = None,
) -> np.ndarray:
    """Convert float array to specified dtype with optional alpha channel."""
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        working = arr[:, :, None]
    else:
        working = arr

    if base_channels is None:
        base_channels = working.shape[2]
    color = working[:, :, :base_channels]

    np_dtype = np.dtype(dtype)
    dtype_info = np.iinfo(np_dtype) if np.issubdtype(np_dtype, np.integer) else None
    if dtype_info:
        scale = float(dtype_info.max - dtype_info.min)
        if scale == 0:
            color_int = np.full_like(color, dtype_info.min, dtype=np_dtype)
        else:
            color_int = np.round(color * scale + dtype_info.min).astype(np_dtype)
    else:
        if float_normalisation is not None and np.issubdtype(np_dtype, np.floating):
            color = float_normalisation.denormalise(color)
        color_int = color.astype(np_dtype, copy=False)

    channels: list[np.ndarray] = [color_int]

    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
        if dtype_info:
            scale = float(dtype_info.max - dtype_info.min)
            if scale == 0:
                alpha_int = np.full_like(alpha, dtype_info.min, dtype=np_dtype)
            else:
                alpha_int = np.round(alpha * scale + dtype_info.min).astype(np_dtype)
        else:
            alpha_int = alpha.astype(np_dtype, copy=False)
        channels.append(alpha_int[:, :, None])

    result = np.concatenate(channels, axis=2) if len(channels) > 1 else color_int
    if result.shape[2] == 1:
        result = result[:, :, 0]
    return np.ascontiguousarray(result)


def compression_for_tifffile(compression: str) -> Optional[str]:
    """Map Pillow compression names to tifffile-compatible format names.

    Args:
        compression: Compression identifier (e.g., "tiff_lzw", "deflate").

    Returns:
        Tifffile compression name, or None for uncompressed output.
    """
    comp = compression.lower()
    mapping = {
        "tiff_lzw": "lzw",
        "lzw": "lzw",
        "tiff_adobe_deflate": "deflate",
        "adobe_deflate": "deflate",
        "deflate": "deflate",
        "tiff_zip": "deflate",
        "zip": "deflate",
        "tiff_jpeg": "jpeg",
        "jpeg": "jpeg",
        "tiff_none": None,
        "none": None,
        "raw": None,
    }
    return mapping.get(comp, comp)


def sanitize_tiff_metadata(raw_metadata: Optional[Any]) -> Optional[Dict[int, Any]]:
    """Remove TIFF tags that conflict with image dimensions and encoding.

    Args:
        raw_metadata: Raw TIFF metadata dictionary.

    Returns:
        Sanitized metadata dictionary with forbidden tags removed, or None.
    """
    if raw_metadata is None:
        return None
    safe: Dict[int, Any] = {}
    forbidden_tags = {256, 257, 273, 279, 322, 323, 324, 325}
    try:
        for tag in raw_metadata:
            if tag in forbidden_tags:
                continue
            safe[tag] = raw_metadata[tag]
    except (TypeError, ValueError, KeyError):  # pragma: no cover - metadata best effort
        LOGGER.debug("Unable to sanitise TIFF metadata", exc_info=True)
        return None
    return safe or None


def _metadata_to_extratag(tag: int, value: Any) -> Optional[tuple[int, str, int, Any, bool]]:
    """Convert a metadata entry to a tifffile extratag tuple when possible."""

    try:
        tag_int = int(tag)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        LOGGER.debug("Skipping non-integer TIFF tag %r", tag)
        return None

    if isinstance(value, str):
        return (tag_int, "s", 0, value, False)

    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        return (tag_int, "B", len(data), data, False)

    if isinstance(value, (tuple, list)):
        items = list(value)
        if not items:
            return None
        if all(isinstance(item, int) for item in items):
            max_val = max(items)
            dtype_code = "H" if 0 <= max_val <= 0xFFFF else "I"
            return (tag_int, dtype_code, len(items), items, False)
        if all(isinstance(item, float) for item in items):
            return (tag_int, "d", len(items), items, False)
        LOGGER.debug("Unsupported iterable metadata type for tag %r", tag)
        return None

    if isinstance(value, int):
        dtype_code = "H" if 0 <= value <= 0xFFFF else "I"
        return (tag_int, dtype_code, 1, value, False)

    if isinstance(value, float):
        return (tag_int, "d", 1, value, False)

    LOGGER.debug("Unsupported TIFF metadata value type for tag %r: %s", tag, type(value).__name__)
    return None


def save_image(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
                 # pylint: disable=too-many-branches,too-many-statements
    destination: Path,
    arr_int: np.ndarray,
    dtype: np.dtype,
    metadata,
    icc_profile: Optional[bytes],
    compression: str,
) -> None:
    """Save image array to disk with comprehensive metadata support."""
    destination_fs = os.fspath(destination)
    metadata = sanitize_tiff_metadata(metadata)
    np_dtype = np.dtype(dtype)
    dtype_info = np.iinfo(np_dtype) if np.issubdtype(np_dtype, np.integer) else None

    writer_compression = compression_for_tifffile(compression)
    lzw_requires_codec = writer_compression in {"lzw", "jpeg"} and imagecodecs is None

    use_tifffile = tifffile is not None and not lzw_requires_codec and (
        (dtype_info is not None and dtype_info.bits >= 16)
        or np.issubdtype(np_dtype, np.floating)
    )

    array_to_write = arr_int
    extrasample_needed = False
    if array_to_write.ndim == 2:
        photometric = "minisblack"
    else:
        channels = array_to_write.shape[2]
        if channels == 2:
            photometric = "minisblack"
            extrasample_needed = True
        elif channels >= 3:
            photometric = "rgb"
            if channels > 3:
                extrasample_needed = True
        else:
            photometric = "minisblack"

    if use_tifffile:
        tiff_kwargs = {
            "photometric": photometric,
            "compression": writer_compression,
            "metadata": None,
        }
        if extrasample_needed or (
            array_to_write.ndim == 3 and array_to_write.shape[2] > 3
        ):
            tiff_kwargs["extrasamples"] = "unassoc"
        extratags = []
        if icc_profile:
            extratags.append((34675, "B", len(icc_profile), icc_profile, False))
        if metadata:
            for tag, value in metadata.items():
                extratag = _metadata_to_extratag(tag, value)
                if extratag is not None:
                    extratags.append(extratag)
                else:  # pragma: no cover - best-effort fallback
                    LOGGER.debug("Skipping TIFF metadata tag %r", tag)
        if extratags:
            tiff_kwargs["extratags"] = extratags
        tifffile.imwrite(destination_fs, array_to_write, **tiff_kwargs)
        return

    if dtype_info and dtype_info.bits == 16:
        LOGGER.warning(
            "Falling back to Pillow for 16-bit save; output will be 8-bit. Install 'tifffile' for full 16-bit support."
        )
        scale = dtype_info.max / 255.0 if dtype_info.max else 1.0
        clipped = np.clip(array_to_write, 0, dtype_info.max).astype(np.float32)
        converted = np.clip(np.round(clipped / scale), 0, 255).astype(np.uint8)
        if converted.ndim == 3 and converted.shape[2] > 3:
            rgb8 = converted[..., :3]
            alpha8 = converted[..., 3]
            converted = np.concatenate([rgb8, alpha8[:, :, None]], axis=2)
        array_to_write = converted
    elif dtype_info and dtype_info.bits < 16 and array_to_write.ndim == 3 and array_to_write.shape[2] > 3:
        array_to_write = array_to_write.astype(np.uint8)

    if array_to_write.ndim == 3 and array_to_write.shape[2] == 4:
        mode = "RGBA"
    elif array_to_write.ndim == 3 and array_to_write.shape[2] == 2:
        mode = "LA"
    elif array_to_write.ndim == 3 and array_to_write.shape[2] == 3:
        mode = "RGB"
    else:
        mode = "L"

    image = Image.fromarray(array_to_write, mode=mode)
    save_kwargs = {"compression": compression}
    if metadata is not None:
        normalised_metadata = {}
        for key, value in metadata.items():
            if isinstance(key, str):
                try:
                    normalised_key = int(key)
                except ValueError:
                    normalised_key = key
            else:
                normalised_key = key
            normalised_metadata[normalised_key] = value
        save_kwargs["tiffinfo"] = normalised_metadata
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    image.save(destination_fs, format="TIFF", **save_kwargs)


__all__ = [
    "FloatDynamicRange",
    "ImageToFloatResult",
    "LuxuryGradeException",
    "ProcessingCapabilities",
    "ProcessingContext",
    "float_to_dtype_array",
    "image_to_float",
    "save_image",
]
