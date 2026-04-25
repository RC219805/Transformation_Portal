"""Deterministic APEX image I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from transformation_portal.evals.image_metadata import bit_depth_from_dtype, inspect_reference_image

ARTIFACT_REFERENCE_16BIT = "reference_16bit"
ARTIFACT_MODEL_INPUT = "model_input"
ARTIFACT_WORKING_16 = "working_16"
ARTIFACT_CANDIDATE_OUTPUT = "candidate_output"
ARTIFACT_DELIVERY_8BIT = "delivery_8bit"
ARTIFACT_DIFF_HEATMAP = "diff_heatmap"
ARTIFACT_MASK_OVERLAY = "mask_overlay"

ARTIFACT_ROLES = frozenset(
    {
        ARTIFACT_REFERENCE_16BIT,
        ARTIFACT_MODEL_INPUT,
        ARTIFACT_WORKING_16,
        ARTIFACT_CANDIDATE_OUTPUT,
        ARTIFACT_DELIVERY_8BIT,
        ARTIFACT_DIFF_HEATMAP,
        ARTIFACT_MASK_OVERLAY,
    }
)


def load_16bit_tiff(path: Path) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load a TIFF/TIF expected to preserve at least 16-bit precision."""
    metadata = inspect_reference_image(path)
    if metadata["observable_reference_metadata_status"] != "ok":
        return None, {"status": "invalid_input", "reason": metadata["observable_reference_metadata_status"], **metadata}
    if metadata["detected_reference_format"] != "tiff":
        return None, {"status": "invalid_input", "reason": "non_tiff_reference", **metadata}
    bit_depth = metadata["detected_reference_bit_depth"]
    if bit_depth is None or bit_depth < 16:
        return None, {"status": "unsupported_bit_depth", "reason": "reference_bit_depth_below_16", **metadata}

    with Image.open(path) as image:
        arr = np.asarray(image)
    arr_bit_depth = bit_depth_from_dtype(arr.dtype)
    if arr_bit_depth is None or arr_bit_depth < 16:
        return None, {"status": "unsupported_bit_depth", "reason": "loaded_bit_depth_below_16", **metadata}
    return arr, {"status": "ok", "reason": None, "artifact_role": ARTIFACT_REFERENCE_16BIT, **metadata}


def derive_model_input_metadata(
    reference_metadata: dict[str, Any],
    *,
    downsampled_for_inference: bool,
    input_dimensions: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    """Describe a normalized model input without changing canonical reference semantics."""
    reference_dimensions = reference_metadata.get("detected_reference_dimensions")
    return {
        "artifact_role": ARTIFACT_MODEL_INPUT,
        "derived_from_role": ARTIFACT_REFERENCE_16BIT,
        "input_bit_depth": 8 if downsampled_for_inference else reference_metadata.get("detected_reference_bit_depth"),
        "input_dimensions": list(input_dimensions) if input_dimensions is not None else reference_dimensions,
        "reference_dimensions": reference_dimensions,
        "downsampled_for_inference": bool(downsampled_for_inference),
    }


def write_16bit_master(array: np.ndarray, path: Path) -> dict[str, Any]:
    """Write a 16-bit master image and return artifact metadata."""
    arr = np.asarray(array)
    bit_depth = bit_depth_from_dtype(arr.dtype)
    if bit_depth is None or bit_depth < 16:
        raise ValueError("16-bit master output requires an integer or float array with at least 16 bits")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="TIFF")
    return {
        "artifact_role": ARTIFACT_WORKING_16,
        "path": str(path),
        "bit_depth": bit_depth,
        "dimensions": [int(arr.shape[1]), int(arr.shape[0])] if arr.ndim >= 2 else None,
    }


def write_delivery_srgb8(array: np.ndarray, path: Path) -> dict[str, Any]:
    """Write an 8-bit sRGB delivery derivative from normalized or integer data."""
    arr = np.asarray(array)
    if arr.dtype.kind == "f":
        arr8 = np.clip(arr, 0.0, 1.0)
        arr8 = np.rint(arr8 * 255.0).astype(np.uint8)
    else:
        max_value = float(np.iinfo(arr.dtype).max) if arr.dtype.kind in {"u", "i"} else 255.0
        arr8 = np.rint(np.clip(arr.astype(np.float32) / max_value, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr8).save(path, format="JPEG")
    return {
        "artifact_role": ARTIFACT_DELIVERY_8BIT,
        "path": str(path),
        "bit_depth": 8,
        "dimensions": [int(arr8.shape[1]), int(arr8.shape[0])] if arr8.ndim >= 2 else None,
        "color_space": "srgb",
    }
