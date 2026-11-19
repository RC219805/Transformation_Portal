"""Shared ndarray conventions for depth backends.

Every depth backend accepts ``np.ndarray`` images alongside PIL images.  The
pixel-range interpretation of those arrays lives here so that identical input
is converted identically, whichever backend receives it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["parse_worker_input_size", "to_rgb_uint8", "validate_worker_depth_map"]


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image array's samples to uint8 pixel values.

    The channel layout is left unchanged; only the sample type and range are
    normalized:

    * ``uint8`` is returned unchanged (the same object).
    * ``uint16`` (any byte order) maps its full storage range with
      ``rint(x / 257)``; a direct cast would wrap modulo 256.
    * Other integer types are clipped to ``[0, 255]``.
    * Finite floats whose maximum is at most ``1.0`` are normalized pixels,
      clipped to ``[0, 1]`` and scaled by 255; other finite floats are
      ``[0, 255]`` pixels and are clipped to that range.  Both truncate, as
      the backends always have for in-range values.

    Raises:
        ValueError: for non-finite float samples and for boolean, complex or
            non-numeric arrays, which have no pixel interpretation.
    """
    dtype = image.dtype
    if dtype == np.uint8:
        return image
    if dtype.kind == "u" and dtype.itemsize == 2:
        return np.rint(image.astype(np.float32) / 257.0).astype(np.uint8)
    if dtype.kind in "iu":
        return np.clip(image, 0, 255).astype(np.uint8)
    if dtype.kind == "f":
        if image.size == 0:
            return image.astype(np.uint8)
        if not np.isfinite(image).all():
            raise ValueError("Image array contains non-finite samples; it has no pixel interpretation")
        if float(image.max()) <= 1.0:
            return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        return np.clip(image, 0.0, 255.0).astype(np.uint8)
    raise ValueError(f"Unsupported image array dtype {dtype}; expected uint8, uint16, another integer type or float")


def validate_worker_depth_map(depth_map: Any, *, label: str) -> np.ndarray:
    """Return a subprocess worker's depth map as float32, or raise ``ValueError``.

    A worker that exits 0 is only trusted once its depth is a non-empty 2-D
    real-valued array whose samples are all finite.
    """
    if not isinstance(depth_map, np.ndarray):
        raise ValueError(f"{label} depth output is not an array")
    if depth_map.dtype.kind not in "fiu":
        raise ValueError(f"{label} depth output has non-real dtype {depth_map.dtype}")
    if depth_map.ndim != 2 or depth_map.size == 0:
        raise ValueError(f"{label} depth output must be a non-empty HxW map, got shape {depth_map.shape}")
    depth = depth_map.astype(np.float32)
    if not np.isfinite(depth).all():
        raise ValueError(f"{label} depth output contains non-finite values")
    return depth


def parse_worker_input_size(value: Any, *, fallback: tuple[int, int]) -> tuple[int, int]:
    """Parse the ``input_size`` a worker echoes; an absent echo keeps ``fallback``.

    Raises:
        ValueError: when a two-element echo does not hold two positive integers.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return fallback
    try:
        height, width = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"worker input_size echo {value!r} is not two integers") from exc
    if height <= 0 or width <= 0:
        raise ValueError(f"worker input_size echo {value!r} is not positive")
    return height, width
