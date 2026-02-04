"""Pixel operations registry for Materials V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass(frozen=True)
class PixelOpDefinition:
    """Definition for a pixel operation."""

    name: str
    op: Callable[[np.ndarray, np.ndarray, dict], np.ndarray]
    implemented: bool
    description: str


def _normalize_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    if image.dtype == np.uint8:
        scale = 255.0
    else:
        scale = 1.0
    return image.astype(np.float32) / scale, scale


def _apply_mask_blend(image: np.ndarray, mask: np.ndarray, modified: np.ndarray) -> np.ndarray:
    mask_3 = np.clip(mask, 0.0, 1.0)[..., None]
    return image * (1.0 - mask_3) + modified * mask_3


def brightness_boost(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Increase brightness within the mask."""
    strength = float(params.get("strength", 0.08))
    normalized = params.get("normalized")
    if normalized is None:
        normalized, scale = _normalize_image(image)
    else:
        scale = float(params.get("scale", 1.0))
    boosted = np.clip(normalized + strength, 0.0, 1.0)
    blended = _apply_mask_blend(normalized, mask, boosted)
    return (blended * scale).astype(image.dtype)


def edge_contrast(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Apply a mild contrast boost within the mask."""
    strength = float(params.get("strength", 0.1))
    normalized = params.get("normalized")
    if normalized is None:
        normalized, scale = _normalize_image(image)
    else:
        scale = float(params.get("scale", 1.0))
    contrast = np.clip((normalized - 0.5) * (1.0 + strength) + 0.5, 0.0, 1.0)
    blended = _apply_mask_blend(normalized, mask, contrast)
    return (blended * scale).astype(image.dtype)


OP_REGISTRY: Dict[str, Dict[str, PixelOpDefinition]] = {
    "glass": {
        "brightness_boost": PixelOpDefinition(
            name="brightness_boost",
            op=brightness_boost,
            implemented=True,
            description="Boost brightness for glass highlights.",
        ),
        "edge_contrast": PixelOpDefinition(
            name="edge_contrast",
            op=edge_contrast,
            implemented=True,
            description="Increase contrast around glass edges.",
        ),
    },
    "stone": {
        "microcontrast": PixelOpDefinition(
            name="microcontrast",
            op=edge_contrast,
            implemented=False,
            description="Placeholder microcontrast enhancement.",
        ),
    },
}
