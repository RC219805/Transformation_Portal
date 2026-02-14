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
    """Normalize image to [0, 1] float32 range.

    NOTE (A4): This function is DEPRECATED for use in pixel ops.
    All pixel operations now receive pre-normalized [0,1] float32 input
    via the executor. Use params["normalized"] instead.

    This function is retained for backward compatibility with external code
    that may still reference it.

    Args:
        image: Input image (uint8, uint16, or float)

    Returns:
        Tuple of (normalized_image, scale_factor)
    """
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        scale = 1.0
    return image.astype(np.float32) / scale, scale


def _apply_mask_blend(image: np.ndarray, mask: np.ndarray, modified: np.ndarray) -> np.ndarray:
    mask_3 = np.clip(mask, 0.0, 1.0)[..., None]
    return image * (1.0 - mask_3) + modified * mask_3


def brightness_boost(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Increase brightness within the mask.

    CONTRACT (A4): This operation receives pre-normalized [0,1] float32 input
    from the executor. The params["normalized"] contains the working image.

    Args:
        image: Input image (pre-normalized to [0,1] by executor)
        mask: Material mask (0-1 float)
        params: Operation parameters
            - strength: Brightness boost amount (default: 0.08)
            - normalized: Pre-normalized image (provided by executor)
            - scale: Normalization scale (legacy, always 1.0)

    Returns:
        Enhanced image in [0,1] float32 range
    """
    strength = float(params.get("strength", 0.08))
    normalized = params.get("normalized")
    if normalized is None:
        # Fallback for backward compatibility
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


# =============================================================================
# Sky Operations (Phase B)
# =============================================================================


def sky_dehaze(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Reduce atmospheric haze in sky regions.

    Increases contrast and saturation slightly to reduce gray veil caused by
    atmospheric haze. Creates clearer, more vibrant sky appearance.

    Args:
        image: Input image (pre-normalized to [0,1] by executor)
        mask: Sky material mask (0-1 float)
        params: Operation parameters
            - strength: Dehaze strength (default: 0.12)
            - normalized: Pre-normalized image (provided by executor)
            - scale: Normalization scale (legacy, always 1.0)

    Returns:
        Enhanced image with same dtype as input
    """
    strength = float(params.get("strength", 0.12))
    normalized = params.get("normalized")
    if normalized is None:
        normalized, scale = _normalize_image(image)
    else:
        scale = float(params.get("scale", 1.0))

    # Contrast boost to reduce haze veil
    contrast = np.clip((normalized - 0.5) * (1.0 + strength * 0.5) + 0.5, 0.0, 1.0)

    # Subtle saturation boost (if RGB)
    if normalized.ndim == 3 and normalized.shape[2] >= 3:
        # Compute luminance (grayscale)
        gray = np.mean(contrast, axis=2, keepdims=True)
        # Increase color separation from gray (saturation)
        enhanced = contrast + (contrast - gray) * strength * 0.3
        enhanced = np.clip(enhanced, 0.0, 1.0)
    else:
        enhanced = contrast

    # Blend using mask
    blended = _apply_mask_blend(normalized, mask, enhanced)

    return (blended * scale).astype(image.dtype)


def sky_gradient_smooth(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Smooth sky gradients to reduce color banding.

    Applies subtle smoothing to reduce visible banding in sky gradients
    (common in sunset/sunrise scenes). Uses gentle blending toward mean color.

    Args:
        image: Input image (pre-normalized to [0,1] by executor)
        mask: Sky material mask (0-1 float)
        params: Operation parameters
            - strength: Smoothing strength (default: 0.10)
            - normalized: Pre-normalized image (provided by executor)
            - scale: Normalization scale (legacy, always 1.0)

    Returns:
        Enhanced image with same dtype as input
    """
    strength = float(params.get("strength", 0.10))
    normalized = params.get("normalized")
    if normalized is None:
        normalized, scale = _normalize_image(image)
    else:
        scale = float(params.get("scale", 1.0))

    # Compute mean color within masked region
    # Simple approach: blend toward mean (future: use bilateral/guided filter)
    masked_pixels = mask > 0.5
    if np.any(masked_pixels):
        if normalized.ndim == 3:
            # RGB: compute mean per channel
            mean_color = np.mean(normalized[masked_pixels], axis=0, keepdims=False)
            # Blend toward mean using NumPy broadcasting (no intermediate allocation)
            smoothed = normalized * (1.0 - strength) + mean_color * strength
        else:
            # Grayscale
            mean_color = np.mean(normalized[masked_pixels])
            smoothed = normalized * (1.0 - strength) + mean_color * strength

        smoothed = np.clip(smoothed, 0.0, 1.0)
    else:
        smoothed = normalized

    # Blend using mask
    blended = _apply_mask_blend(normalized, mask, smoothed)

    return (blended * scale).astype(image.dtype)


def sky_temperature_shift(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """Shift sky color temperature (warmer/cooler).

    Adjusts color temperature to simulate different times of day:
    - Positive strength: warmer (sunset/sunrise, golden hour)
    - Negative strength: cooler (midday, blue hour)

    Args:
        image: Input image (pre-normalized to [0,1] by executor)
        mask: Sky material mask (0-1 float)
        params: Operation parameters
            - strength: Temperature shift (default: 0.05, range: -0.1 to +0.1)
            - normalized: Pre-normalized image (provided by executor)
            - scale: Normalization scale (legacy, always 1.0)

    Returns:
        Enhanced image with same dtype as input
    """
    strength = float(params.get("strength", 0.05))
    normalized = params.get("normalized")
    if normalized is None:
        normalized, scale = _normalize_image(image)
    else:
        scale = float(params.get("scale", 1.0))

    # Apply color temperature shift (RGB only)
    if normalized.ndim == 3 and normalized.shape[2] >= 3:
        shifted = normalized.copy()
        # Warm: boost red, reduce blue
        # Cool: reduce red, boost blue
        shifted[..., 0] = np.clip(shifted[..., 0] + strength, 0.0, 1.0)  # R
        shifted[..., 2] = np.clip(shifted[..., 2] - strength, 0.0, 1.0)  # B
    else:
        # Grayscale: no temperature shift possible
        shifted = normalized

    # Blend using mask
    blended = _apply_mask_blend(normalized, mask, shifted)

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
    "sky": {
        "dehaze": PixelOpDefinition(
            name="dehaze",
            op=sky_dehaze,
            implemented=True,
            description="Reduce atmospheric haze in sky.",
        ),
        "gradient_smooth": PixelOpDefinition(
            name="gradient_smooth",
            op=sky_gradient_smooth,
            implemented=True,
            description="Smooth sky gradients to reduce banding.",
        ),
        "temperature_shift": PixelOpDefinition(
            name="temperature_shift",
            op=sky_temperature_shift,
            implemented=True,
            description="Shift sky color temperature (warm/cool).",
        ),
    },
}
