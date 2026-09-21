"""Bounded, plan-declared sRGB browser derivatives of the final photographic master."""

from __future__ import annotations

import hashlib
import io
from typing import Any

import numpy as np
from PIL import Image

from transformation_portal.core.execution_plan_v4 import BROWSER_PREVIEW_RECIPE
from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc

MAX_PREVIEW_EDGE = 1600
MAX_PREVIEW_BYTES = MAX_PREVIEW_EDGE**2 * 4 + 1024 * 1024


def preview_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Retain aspect ratio without upscaling, with deterministic integer rounding."""
    height, width = shape
    edge = max(height, width)
    if edge <= MAX_PREVIEW_EDGE:
        return height, width
    return max(1, height * MAX_PREVIEW_EDGE // edge), max(1, width * MAX_PREVIEW_EDGE // edge)


def preview_samples(master: ImageMaster) -> np.ndarray:
    """Area-average linear light, premultiplying alpha to prevent transparent halos.

    Process one float32 source plane at a time. All RGB encoding and interleaved
    preview allocations are bounded by the 1600-pixel derivative, not the source.
    """
    height, width = preview_shape(master.shape)

    def resize(values: np.ndarray) -> np.ndarray:
        if values.shape == (height, width):
            return values
        with Image.fromarray(values) as image:
            with image.resize((width, height), resample=Image.Resampling.BOX) as resized:
                return np.asarray(resized, dtype=np.float32)

    alpha = None if master.alpha is None else resize(master.alpha)
    linear = np.empty((height, width, 3), dtype=np.float32)
    for channel in range(3):
        values = master.pixels[..., channel]
        if master.alpha is not None:
            values = values * master.alpha
        reduced = resize(values)
        if alpha is not None:
            np.divide(reduced, alpha, out=linear[..., channel], where=alpha > 0)
            linear[..., channel][alpha <= 0] = 0
        else:
            linear[..., channel] = reduced
    samples = np.rint(np.clip(linear_to_srgb(linear), 0, 1) * 255).astype(np.uint8)
    if alpha is not None:
        samples = np.concatenate([samples, np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)[..., None]], axis=2)
    return samples


def preview_descriptor(master: ImageMaster, *, relative_path: str) -> dict[str, Any]:
    return {
        "schema": "tp.image.browser_preview.v1",
        "path": relative_path,
        "recipe": BROWSER_PREVIEW_RECIPE,
        "shape": list(preview_shape(master.shape)),
        "max_edge": MAX_PREVIEW_EDGE,
        "color_space": "srgb",
        "output_bit_depth": 8,
        "alpha_preserved": master.alpha is not None,
        "master_content_hash": master.content_hash(),
        "output_icc_sha256": hashlib.sha256(output_srgb_icc()).hexdigest(),
        "classification": "photographic_browser_preview",
    }


def encode_preview(master: ImageMaster, *, relative_path: str) -> tuple[bytes, dict[str, Any]]:
    """Encode only the display derivative; preserve the float master and 16-bit TIFF."""
    buffer = io.BytesIO()
    with Image.fromarray(preview_samples(master)) as image:
        image.save(buffer, format="PNG", icc_profile=output_srgb_icc(), optimize=False, compress_level=6)
    raw = buffer.getvalue()
    if len(raw) > MAX_PREVIEW_BYTES:
        raise ValueError("V5 browser preview exceeds its encoded byte bound")
    return raw, preview_descriptor(master, relative_path=relative_path)
