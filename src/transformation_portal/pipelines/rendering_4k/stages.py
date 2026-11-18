#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure image stage functions for the 4K rendering pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageFilter

from ...utils.image_utils import apply_cube_lut, load_cube_lut
from .types import (
    ColorGradingConfig,
    MaterialResponseConfig,
    ToneMappingConfig,
    ToneMappingMethod,
    UpscalingConfig,
)

try:
    from scipy.ndimage import gaussian_filter, uniform_filter

    HAS_SCIPY_STAGE_FILTERS = True
except ImportError:
    HAS_SCIPY_STAGE_FILTERS = False
    gaussian_filter = None
    uniform_filter = None


logger = logging.getLogger("transformation_portal.pipelines.rendering_4k_pipeline")


def apply_tone_mapping(
    image: np.ndarray,
    config: ToneMappingConfig,
) -> np.ndarray:
    """
    Apply HDR tone mapping to image.

    Args:
        image: HDR image as float32 array
        config: Tone mapping configuration

    Returns:
        Tone-mapped image in [0, 1] range
    """
    if not config.enabled:
        return np.clip(image, 0, 1)

    # Apply exposure adjustment first
    if config.exposure != 0:
        image = image * (2.0**config.exposure)

    # Select tone mapping operator
    if config.method == ToneMappingMethod.REINHARD:
        # Simple Reinhard global operator
        mapped = image / (1.0 + image)

    elif config.method == ToneMappingMethod.FILMIC:
        # Hable/Uncharted 2 filmic curve
        mapped = _filmic_hable(image, config.white_point)

    elif config.method == ToneMappingMethod.ACES:
        # ACES approximation
        mapped = _aces_approximation(image)

    else:  # AGX (default)
        # AgX-inspired sigmoid curve
        mapped = _agx_sigmoid(image)

    # Apply contrast adjustment
    if config.contrast != 1.0:
        mean = np.mean(mapped)
        mapped = (mapped - mean) * config.contrast + mean

    return np.clip(mapped, 0, 1).astype(np.float32)


def _filmic_hable(x: np.ndarray, white_point: float = 11.2) -> np.ndarray:
    """Hable/Uncharted 2 filmic tone mapping curve."""

    def hable_curve(v: np.ndarray) -> np.ndarray:
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

    curr = hable_curve(x)
    white = hable_curve(np.array([white_point]))
    return curr / white


def _aces_approximation(x: np.ndarray) -> np.ndarray:
    """Simple ACES approximation (Krzysztof Narkowicz)."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _agx_sigmoid(x: np.ndarray) -> np.ndarray:
    """AgX-inspired sigmoid tone mapping."""
    # Apply log-space compression
    x = np.maximum(x, 1e-10)
    log_x = np.log2(x + 0.001)

    # Sigmoid in log space
    sigmoid = 1.0 / (1.0 + np.exp(-log_x * 0.5))

    # Scale to output range
    return np.clip(sigmoid, 0, 1)


def apply_material_response(
    image: np.ndarray,
    depth_map: Optional[np.ndarray],
    config: MaterialResponseConfig,
) -> np.ndarray:
    """
    Apply Material Response Technology enhancement.

    Enhances surface textures and material properties using depth information.

    Args:
        image: Input image as float32 array [0, 1]
        depth_map: Depth map (optional, improves results)
        config: Material Response configuration

    Returns:
        Enhanced image
    """
    if not config.enabled:
        return image

    enhanced = image.copy()

    # Texture enhancement via high-frequency boost
    if config.texture_boost > 0:
        if HAS_SCIPY_STAGE_FILTERS and gaussian_filter is not None:
            blurred = gaussian_filter(enhanced, sigma=(1.2, 1.2, 0))
        else:
            # Fallback: use PIL-based blur
            blurred = _simple_gaussian_blur(enhanced, sigma=1.2)
        detail = enhanced - blurred
        enhanced = np.clip(enhanced + config.texture_boost * detail, 0, 1)

    # Micro-contrast enhancement
    if config.micro_contrast > 0:
        enhanced = _apply_local_contrast(enhanced, config.micro_contrast)

    # Apply strength blending
    enhanced = image * (1 - config.strength) + enhanced * config.strength

    return np.clip(enhanced, 0, 1).astype(np.float32)


def _simple_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Simple Gaussian blur using PIL as fallback."""
    # Convert to PIL, blur, convert back
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred).astype(np.float32) / 255.0


def _apply_local_contrast(image: np.ndarray, strength: float) -> np.ndarray:
    """Apply local contrast enhancement (CLAHE-like)."""
    if HAS_SCIPY_STAGE_FILTERS and uniform_filter is not None:
        # Local mean using scipy
        local_mean = uniform_filter(image, size=(32, 32, 1))
    else:
        # Fallback: use simple box blur
        local_mean = _simple_box_blur(image, size=32)

    # Local contrast enhancement
    enhanced = image + strength * (image - local_mean)

    return np.clip(enhanced, 0, 1)


def _simple_box_blur(image: np.ndarray, size: int) -> np.ndarray:
    """Simple box blur as fallback for uniform_filter."""
    # Handle each channel
    h, w, c = image.shape
    result = np.zeros_like(image)
    for ch in range(c):
        img_uint8 = (np.clip(image[..., ch], 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="L")
        blurred = pil_img.filter(ImageFilter.BoxBlur(size // 2))
        result[..., ch] = np.array(blurred).astype(np.float32) / 255.0
    return result


def apply_color_grading(
    image: np.ndarray,
    config: ColorGradingConfig,
) -> np.ndarray:
    """
    Apply color grading adjustments including LUT stacks.

    Supports:
    - Temperature shift (RGB multipliers)
    - Saturation and vibrance adjustments
    - LUT (Look-Up Table) application with configurable strengths

    Args:
        image: Input image as float32 array [0, 1]
        config: Color grading configuration

    Returns:
        Color-graded image
    """
    if not config.enabled:
        return image

    graded = image.copy()

    # Apply LUTs first (before other adjustments)
    if config.lut_paths and config.lut_strengths:
        for lut_path, strength in zip(config.lut_paths, config.lut_strengths):
            if strength > 0:
                lut_result = _apply_lut(graded, lut_path, strength)
                if lut_result is not None:
                    graded = lut_result
                    logger.debug(f"Applied LUT: {Path(lut_path).name} @ {strength:.0%}")
                else:
                    logger.warning(f"Failed to apply LUT: {Path(lut_path).name} (strength={strength:.0%})")

    # Apply temperature shift (RGB multipliers)
    r_mult, g_mult, b_mult = config.temperature_shift
    graded[..., 0] *= r_mult
    graded[..., 1] *= g_mult
    graded[..., 2] *= b_mult

    # Apply saturation adjustment
    if config.saturation != 1.0:
        # Convert to HSV-like representation
        lum = 0.2126 * graded[..., 0] + 0.7152 * graded[..., 1] + 0.0722 * graded[..., 2]
        graded = lum[..., np.newaxis] + config.saturation * (graded - lum[..., np.newaxis])

    # Apply vibrance (saturation that targets less saturated colors)
    if config.vibrance != 1.0:
        graded = _apply_vibrance(graded, config.vibrance)

    return np.clip(graded, 0, 1).astype(np.float32)


def _load_cube_lut(lut_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load a .cube LUT file.

    Args:
        lut_path: Path to .cube LUT file

    Returns:
        Read-only 3D LUT array (size, size, size, 3) indexed ``lut[r, g, b]``,
        or None if loading fails
    """
    lut_path = Path(lut_path)
    if not lut_path.exists():
        logger.warning(f"LUT file not found: {lut_path}")
        return None

    try:
        return load_cube_lut(lut_path)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load LUT {lut_path}: {e}")
        return None


def _apply_lut(
    image: np.ndarray,
    lut_path: Union[str, Path],
    strength: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Apply a .cube LUT to an image using trilinear interpolation.

    Args:
        image: Input image as float32 array [0, 1] with shape (H, W, 3)
        lut_path: Path to .cube LUT file
        strength: LUT application strength (0.0-1.0)

    Returns:
        LUT-processed image, or None if LUT could not be applied
    """
    lut = _load_cube_lut(lut_path)
    if lut is None:
        return None

    return apply_cube_lut(image, lut, strength)


def _apply_vibrance(image: np.ndarray, vibrance: float) -> np.ndarray:
    """Apply vibrance (smart saturation targeting less saturated colors)."""
    # Compute current saturation
    max_rgb = np.max(image, axis=2, keepdims=True)
    min_rgb = np.min(image, axis=2, keepdims=True)
    sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)

    # Low saturation areas get more boost
    boost = 1.0 + (vibrance - 1.0) * (1.0 - sat)

    # Apply saturation boost
    lum = 0.2126 * image[..., 0:1] + 0.7152 * image[..., 1:2] + 0.0722 * image[..., 2:3]
    boosted = lum + boost * (image - lum)

    return np.clip(boosted, 0, 1)


def apply_upscaling(
    image: Image.Image,
    config: UpscalingConfig,
) -> Image.Image:
    """
    Upscale image to target resolution.

    Args:
        image: PIL Image to upscale
        config: Upscaling configuration

    Returns:
        Upscaled PIL Image
    """
    if not config.enabled:
        return image

    current_w, current_h = image.size
    target_w, target_h = config.target_resolution

    # Check if upscaling is needed
    if current_w >= target_w and current_h >= target_h:
        logger.info(f"Image already at or above target resolution ({current_w}x{current_h})")
        return image

    # Calculate scale to fit within target while maintaining aspect ratio
    scale_w = target_w / current_w
    scale_h = target_h / current_h
    scale = min(scale_w, scale_h)
    if scale <= 1.0:
        logger.info(
            f"Image not upscaled because fitting target resolution would require downscaling "
            f"({current_w}x{current_h} target {target_w}x{target_h})"
        )
        return image

    new_w = int(current_w * scale)
    new_h = int(current_h * scale)

    # Apply upscaling method
    if config.method == "esrgan":
        # Real-ESRGAN upscaling (requires optional dependencies)
        logger.warning("ESRGAN upscaling requires optional ML dependencies. Using Lanczos fallback.")
        upscaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        # Default Lanczos
        upscaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Optional sharpening after upscale
    if config.preserve_sharpness:
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=50, threshold=0))

    logger.info(f"Upscaled from {current_w}x{current_h} to {new_w}x{new_h}")

    return upscaled


def estimate_depth_simple(image: np.ndarray) -> np.ndarray:
    """
    Simple depth estimation using luminance gradient.

    This is a lightweight fallback when Depth Anything V2 is not available.
    For production use, the full depth model should be used.

    Args:
        image: RGB image as float32 array [0, 1]

    Returns:
        Depth map as float32 array [0, 1]
    """
    # Compute luminance
    lum = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]

    # Simple depth proxy using luminance + spatial gradient
    if HAS_SCIPY_STAGE_FILTERS and gaussian_filter is not None:
        # Blur for depth approximation (distant objects blur more)
        blurred = gaussian_filter(lum, sigma=15)
    else:
        # Fallback: use PIL-based blur
        blurred = _simple_gaussian_blur_2d(lum, sigma=15)

    # Vertical gradient (sky typically brighter at top)
    h, w = lum.shape
    y_gradient = np.linspace(0, 1, h)[:, np.newaxis]
    y_gradient = np.tile(y_gradient, (1, w))

    # Combine luminance inversion with spatial cues
    depth = 0.5 * (1 - blurred) + 0.5 * y_gradient

    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    return depth.astype(np.float32)


def _simple_gaussian_blur_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    """Simple 2D Gaussian blur using PIL as fallback."""
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred).astype(np.float32) / 255.0


__all__ = [
    "apply_color_grading",
    "apply_material_response",
    "apply_tone_mapping",
    "apply_upscaling",
    "estimate_depth_simple",
]
