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


def stone_microcontrast(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Apply subtle texture enhancement for stone materials.

    Enhances mid-frequency detail typical of stone surfaces (grain, veining, texture)
    without over-sharpening or introducing artifacts.

    Args:
        image: Input image (uint8 or float32)
        mask: Material mask (0-1 float)
        params: Operation parameters
            - strength: Enhancement strength (default: 0.12)
            - normalized: Pre-normalized image (optional)
            - scale: Normalization scale (optional)

    Returns:
        Enhanced image with same dtype as input
    """
    strength = float(params.get("strength", 0.12))
    normalized = params.get("normalized")
    if normalized is None:
        normalized, scale = _normalize_image(image)
    else:
        scale = float(params.get("scale", 1.0))

    # Apply subtle contrast enhancement
    # Stone textures benefit from mid-range contrast boost
    enhanced = np.clip((normalized - 0.5) * (1.0 + strength) + 0.5, 0.0, 1.0)

    # Blend using mask
    blended = _apply_mask_blend(normalized, mask, enhanced)

    return (blended * scale).astype(image.dtype)


def water_reflection_enhance(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Enhance water surface reflections.

    Increases brightness slightly and boosts contrast to enhance reflectivity
    typical of water surfaces (pools, lakes, ocean). Uses subtle enhancement
    to avoid over-processing water surfaces.

    Args:
        image: Input image (uint8 or float32)
        mask: Water material mask (0-1 float)
        params: Operation parameters
            - strength: Enhancement strength (default: 0.10 for subtle enhancement)
            - normalized: Pre-normalized image (optional)
            - scale: Normalization scale (optional)

    Returns:
        Enhanced image with same dtype as input
    """
    strength = float(params.get("strength", 0.10))
    normalized = params.get("normalized")
    if normalized is None:
        normalized, scale = _normalize_image(image)
    else:
        scale = float(params.get("scale", 1.0))

    # Combine brightness boost and contrast for reflection enhancement
    # Water needs subtle enhancement to maintain natural appearance
    # First apply brightness boost
    brightened = np.clip(normalized + strength * 0.5, 0.0, 1.0)
    # Then apply contrast boost
    enhanced = np.clip((brightened - 0.5) * (1.0 + strength) + 0.5, 0.0, 1.0)

    # Blend using mask
    blended = _apply_mask_blend(normalized, mask, enhanced)

    return (blended * scale).astype(image.dtype)


def foliage_vibrance_boost(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Boost foliage vibrance (green enhancement).

    Selectively enhances green channel within foliage regions to make
    vegetation appear more vibrant without oversaturation. Preserves
    natural color balance while enhancing green tones.

    Args:
        image: Input image (uint8 or float32)
        mask: Foliage material mask (0-1 float)
        params: Operation parameters
            - strength: Enhancement strength (default: 0.08)
            - normalized: Pre-normalized image (optional)
            - scale: Normalization scale (optional)

    Returns:
        Enhanced image with same dtype as input
    """
    strength = float(params.get("strength", 0.08))
    normalized = params.get("normalized")
    if normalized is None:
        normalized, scale = _normalize_image(image)
    else:
        scale = float(params.get("scale", 1.0))

    # Enhance green channel selectively
    # Create a copy to avoid modifying original
    enhanced = normalized.copy()

    # Boost green channel (index 1 in RGB)
    if normalized.ndim == 3 and normalized.shape[2] >= 3:
        # Apply selective green boost
        enhanced[..., 1] = np.clip(normalized[..., 1] * (1.0 + strength), 0.0, 1.0)

    # Blend using mask
    blended = _apply_mask_blend(normalized, mask, enhanced)

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
            op=stone_microcontrast,
            implemented=True,
            description="Subtle texture enhancement for stone surfaces.",
        ),
    },
    "water": {
        "reflection_enhance": PixelOpDefinition(
            name="reflection_enhance",
            op=water_reflection_enhance,
            implemented=True,
            description="Enhance reflections and clarity for water surfaces.",
        ),
    },
    "foliage": {
        "vibrance_boost": PixelOpDefinition(
            name="vibrance_boost",
            op=foliage_vibrance_boost,
            implemented=True,
            description="Boost green channel vibrance for foliage.",
        ),
    },
}
