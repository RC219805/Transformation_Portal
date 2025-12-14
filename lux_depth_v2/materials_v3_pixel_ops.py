"""Materials V3 Pixel Operations - Apply Response Plan to Image.

PR-4B: Glass Response Application Only

This module takes the response plan from PR-4A and applies pixel-level
enhancements to glass specifically. Future PRs will add other materials.

Key Principles:
- Conservative edge handling (avoid halos)
- Separate core vs edge treatment
- Reversible operations for A/B testing
- Full auditability via stats
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .logging_utils import setup_logging

log = setup_logging(__name__)


@dataclass
class GlassResponseConfig:
    """Configuration for glass-specific material response."""
    
    # Core enhancement
    core_contrast: float = 1.12  # Boost contrast in core regions
    core_clarity: float = 0.08   # Subtle clarity/sharpness
    core_saturation: float = 0.95  # Slight desaturation (realistic glass)
    
    # Edge enhancement
    edge_contrast: float = 1.05  # Very conservative on edges
    edge_clarity: float = 0.03   # Minimal sharpness (avoid halos)
    edge_saturation: float = 0.92  # More desaturation at edges
    
    # Highlight preservation
    preserve_highlights: bool = True
    highlight_threshold: float = 0.85  # Pixels above this are clamped
    
    # Reflection enhancement
    enhance_reflections: bool = True
    reflection_threshold: float = 0.65  # Detect specular regions
    reflection_boost: float = 1.08  # Subtle boost
    
    # Safety guards
    max_delta: float = 0.15  # Maximum pixel change (prevent artifacts)
    blend_edge_width_px: int = 3  # Blend zone at core/edge boundary


def extract_core_edge_masks(
    mask: np.ndarray,
    edge_width_px: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract core, edge, and blend zone masks from a class mask.
    
    Args:
        mask: HxW float32 mask in [0,1]
        edge_width_px: Width of edge band in pixels
        
    Returns:
        core_mask: HxW bool (core region, excluding edges)
        edge_mask: HxW bool (edge band region)
        blend_mask: HxW bool (transition zone for smooth blending)
    """
    from scipy.ndimage import binary_erosion, binary_dilation
    
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    
    # Threshold to binary
    binary = mask > 0.5
    
    # Core = eroded by edge width
    struct = np.ones((edge_width_px * 2 + 1, edge_width_px * 2 + 1), dtype=bool)
    core = binary_erosion(binary, structure=struct)
    
    # Edge = binary minus core
    edge = binary & ~core
    
    # Blend zone = slightly smaller erosion
    blend_struct = np.ones((edge_width_px, edge_width_px), dtype=bool)
    blend_core = binary_erosion(binary, structure=blend_struct)
    blend = blend_core & ~core
    
    return core, edge, blend


def apply_local_contrast(
    rgb: np.ndarray,
    strength: float = 1.10,
    preserve_highlights: bool = True,
    highlight_threshold: float = 0.85,
) -> np.ndarray:
    """Apply local contrast enhancement.
    
    Args:
        rgb: HxWx3 float32 in [0,1]
        strength: Contrast multiplier (1.0 = no change)
        preserve_highlights: Clamp bright pixels to avoid blowing out
        highlight_threshold: Threshold for highlight preservation
        
    Returns:
        Enhanced HxWx3 float32 in [0,1]
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}")
    
    # Convert to LAB for perceptual contrast
    # Simple approximation: use luminance channel
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    
    # Local contrast = (pixel - local_mean) * strength + local_mean
    from scipy.ndimage import gaussian_filter
    local_mean = gaussian_filter(luma, sigma=5.0)
    
    contrast_luma = (luma - local_mean) * strength + local_mean
    
    # Preserve relative chrominance
    scale = np.where(luma > 0, contrast_luma / (luma + 1e-6), 1.0)
    result = rgb * scale[..., None]
    
    # Preserve highlights
    if preserve_highlights:
        bright = luma > highlight_threshold
        result[bright] = rgb[bright]  # Keep original
    
    return np.clip(result, 0.0, 1.0)


def apply_clarity(
    rgb: np.ndarray,
    strength: float = 0.05,
) -> np.ndarray:
    """Apply clarity (local high-frequency boost) similar to Lightroom.
    
    Args:
        rgb: HxWx3 float32 in [0,1]
        strength: Clarity strength (typical 0.02-0.10)
        
    Returns:
        Enhanced HxWx3 float32 in [0,1]
    """
    from scipy.ndimage import gaussian_filter
    
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    
    # High-pass filter
    low = gaussian_filter(luma, sigma=10.0)
    high = luma - low
    
    # Boost high frequencies
    clarity_luma = luma + high * strength
    
    # Apply to RGB proportionally
    scale = np.where(luma > 0, clarity_luma / (luma + 1e-6), 1.0)
    result = rgb * scale[..., None]
    
    return np.clip(result, 0.0, 1.0)


def apply_saturation(
    rgb: np.ndarray,
    scale: float = 0.95,
) -> np.ndarray:
    """Apply saturation adjustment.
    
    Args:
        rgb: HxWx3 float32 in [0,1]
        scale: Saturation multiplier (1.0 = no change, <1.0 = desaturate)
        
    Returns:
        HxWx3 float32 in [0,1]
    """
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    result = luma[..., None] + (rgb - luma[..., None]) * scale
    return np.clip(result, 0.0, 1.0)


def apply_glass_response(
    rgb01: np.ndarray,
    glass_mask: np.ndarray,
    cfg: GlassResponseConfig,
    response_plan: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Apply glass-specific material response to image.
    
    Args:
        rgb01: HxWx3 float32 input image in [0,1]
        glass_mask: HxW float32 glass confidence mask in [0,1]
        cfg: Glass response configuration
        response_plan: Optional per-class response plan from PR-4A
        
    Returns:
        result_rgb: HxWx3 float32 enhanced image in [0,1]
        stats: Dict with per-region stats for auditability
    """
    if rgb01.shape[:2] != glass_mask.shape:
        raise ValueError(
            f"Image shape {rgb01.shape[:2]} != mask shape {glass_mask.shape}"
        )
    
    H, W, _ = rgb01.shape
    result = rgb01.copy()
    
    # Extract core/edge/blend zones
    core_mask, edge_mask, blend_mask = extract_core_edge_masks(
        glass_mask,
        edge_width_px=5,
    )
    
    core_px = int(core_mask.sum())
    edge_px = int(edge_mask.sum())
    blend_px = int(blend_mask.sum())
    
    log.debug(
        f"Glass zones: core={core_px}px, edge={edge_px}px, blend={blend_px}px"
    )
    
    # Apply core enhancement
    if core_px > 0:
        core_rgb = rgb01[core_mask]
        enhanced_core = core_rgb.copy()
        
        # Contrast
        enhanced_core = apply_local_contrast(
            enhanced_core.reshape(-1, 1, 3),
            strength=cfg.core_contrast,
            preserve_highlights=cfg.preserve_highlights,
            highlight_threshold=cfg.highlight_threshold,
        ).reshape(-1, 3)
        
        # Clarity
        enhanced_core = apply_clarity(
            enhanced_core.reshape(-1, 1, 3),
            strength=cfg.core_clarity,
        ).reshape(-1, 3)
        
        # Saturation
        enhanced_core = apply_saturation(
            enhanced_core,
            scale=cfg.core_saturation,
        )
        
        # Safety clamp
        delta = np.abs(enhanced_core - core_rgb).max(axis=1)
        excessive = delta > cfg.max_delta
        if excessive.any():
            log.warning(
                f"{excessive.sum()} core pixels exceeded max_delta; clamping"
            )
            enhanced_core[excessive] = core_rgb[excessive]
        
        result[core_mask] = enhanced_core
    
    # Apply edge enhancement (more conservative)
    if edge_px > 0:
        edge_rgb = rgb01[edge_mask]
        enhanced_edge = edge_rgb.copy()
        
        # Contrast (gentler)
        enhanced_edge = apply_local_contrast(
            enhanced_edge.reshape(-1, 1, 3),
            strength=cfg.edge_contrast,
            preserve_highlights=cfg.preserve_highlights,
            highlight_threshold=cfg.highlight_threshold,
        ).reshape(-1, 3)
        
        # Clarity (minimal)
        enhanced_edge = apply_clarity(
            enhanced_edge.reshape(-1, 1, 3),
            strength=cfg.edge_clarity,
        ).reshape(-1, 3)
        
        # Saturation (more desaturation)
        enhanced_edge = apply_saturation(
            enhanced_edge,
            scale=cfg.edge_saturation,
        )
        
        # Safety clamp
        delta = np.abs(enhanced_edge - edge_rgb).max(axis=1)
        excessive = delta > cfg.max_delta
        if excessive.any():
            log.warning(
                f"{excessive.sum()} edge pixels exceeded max_delta; clamping"
            )
            enhanced_edge[excessive] = edge_rgb[excessive]
        
        result[edge_mask] = enhanced_edge
    
    # Smooth blend in transition zone
    if blend_px > 0:
        from scipy.ndimage import gaussian_filter
        
        # Create blend weight map (0 at core, 1 at edge boundary)
        blend_weight = np.zeros_like(glass_mask)
        blend_weight[edge_mask] = 1.0
        blend_weight = gaussian_filter(blend_weight, sigma=cfg.blend_edge_width_px)
        
        # Blend original and enhanced in blend zone
        alpha = blend_weight[blend_mask, None]
        result[blend_mask] = (
            alpha * rgb01[blend_mask] + (1 - alpha) * result[blend_mask]
        )
    
    # Compute stats
    delta_rgb = result - rgb01
    delta_magnitude = np.linalg.norm(delta_rgb, axis=-1)
    
    stats = {
        "core_pixels": core_px,
        "edge_pixels": edge_px,
        "blend_pixels": blend_px,
        "total_glass_pixels": int((glass_mask > 0.5).sum()),
        "mean_delta_core": float(delta_magnitude[core_mask].mean()) if core_px > 0 else 0.0,
        "mean_delta_edge": float(delta_magnitude[edge_mask].mean()) if edge_px > 0 else 0.0,
        "max_delta": float(delta_magnitude.max()),
        "pixels_clamped": int((delta_magnitude > cfg.max_delta).sum()),
    }
    
    return result, stats
