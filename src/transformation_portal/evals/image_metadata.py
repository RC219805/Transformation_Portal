"""Observable image metadata helpers for APEX eval evidence."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np


def normalize_image_format(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().lstrip(".")
    if not normalized:
        return None
    if normalized == "jpg":
        return "jpeg"
    if normalized == "tif":
        return "tiff"
    return normalized


def bit_depth_from_dtype(dtype: Any) -> int | None:
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        return None
    if np_dtype.kind == "b":
        return 1
    if np_dtype.kind in {"u", "i", "f"}:
        return int(np_dtype.itemsize * 8)
    return None


def bit_depth_from_pil_mode(mode: str | None) -> int | None:
    if mode in {"1"}:
        return 1
    if mode in {"L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
        return 8
    if mode in {"I;16", "I;16B", "I;16L"}:
        return 16
    if mode in {"I", "F"}:
        return 32
    return None


def _tiff_bit_depth(path: Path) -> int | None:
    try:
        tifffile = importlib.import_module("tifffile")
        with tifffile.TiffFile(path) as tiff:
            if not tiff.pages:
                return None
            page = tiff.pages[0]
            dtype_bits = bit_depth_from_dtype(getattr(page, "dtype", None))
            if dtype_bits is not None:
                return dtype_bits
            bits_per_sample = getattr(page, "bitspersample", None)
            if isinstance(bits_per_sample, (tuple, list)):
                return int(max(bits_per_sample)) if bits_per_sample else None
            if bits_per_sample is not None:
                return int(bits_per_sample)
    except (ImportError, OSError, ValueError, TypeError, AttributeError, IndexError):
        return None
    return None


def inspect_reference_image(path: Path | None) -> dict[str, Any]:
    """Inspect observable metadata without treating manifest assertions as truth."""
    metadata: dict[str, Any] = {
        "detected_reference_format": None,
        "detected_reference_bit_depth": None,
        "detected_reference_dimensions": None,
        "detected_reference_mode": None,
        "detected_reference_channel_count": None,
        "detected_reference_icc_profile_name": None,
        "detected_reference_icc_profile_sha256": None,
        "observable_reference_metadata_status": "missing_reference_asset",
        "observable_reference_metadata_error": None,
    }
    if path is None or not path.is_file():
        return metadata

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            detected_format = normalize_image_format(image.format or path.suffix)
            detected_bit_depth = bit_depth_from_pil_mode(image.mode)
            width, height = image.size
            mode = image.mode
            bands = image.getbands()
            icc_profile = image.info.get("icc_profile")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        metadata["observable_reference_metadata_status"] = "unreadable_reference_image"
        metadata["observable_reference_metadata_error"] = str(exc)
        return metadata

    if detected_format == "tiff":
        tiff_bit_depth = _tiff_bit_depth(path)
        detected_bit_depth = tiff_bit_depth if tiff_bit_depth is not None else detected_bit_depth
    if detected_format == "jpeg" and detected_bit_depth is None:
        detected_bit_depth = 8

    icc_sha = hashlib.sha256(icc_profile).hexdigest() if isinstance(icc_profile, bytes) and icc_profile else None
    metadata.update(
        {
            "detected_reference_format": detected_format,
            "detected_reference_bit_depth": detected_bit_depth,
            "detected_reference_dimensions": [int(width), int(height)],
            "detected_reference_mode": mode,
            "detected_reference_channel_count": len(bands),
            "detected_reference_icc_profile_sha256": icc_sha,
            "observable_reference_metadata_status": (
                "ok"
                if detected_format is not None and detected_bit_depth is not None
                else "missing_observable_reference_metadata"
            ),
        }
    )
    return metadata
