"""Materials V3 Pixel Operations - Stone Material Response.

PR-4D: Stone Response Application

This module applies conservative pixel-level enhancements to stone materials
(granite, marble, limestone, etc.). Stone requires more conservative treatment
than glass due to high-contrast veining patterns.

Key Principles:
- Very conservative delta clamp (0.08 vs glass 0.12)
- Mild edge processing to avoid halos on veining
- Core/edge split for localized control
- Full auditability via stats
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .logging_utils import setup_logging

log = setup_logging(__name__)


@dataclass
class StoneResponseConfig:
    """Configuration for stone-specific material response."""
    
    # Core enhancement (subtle local contrast)
    core_local_contrast: float = 1.04  # Very conservative boost
    core_clarity: float = 1.02         # Minimal clarity
    core_saturation: float = 1.00      # No saturation change
    
    # Edge enhancement (very mild)
    edge_local_contrast: float = 1.02  # Extremely conservative
    edge_clarity: float = 1.01         # Barely perceptible
    edge_saturation: float = 1.00      # No saturation change
    
    # Safety guards
    max_delta: float = 0.08            # Tight clamp (vs glass 0.12)
    halo_p95_threshold: float = 0.06   # p95 boundary delta guard
    min_coverage_px: int = 50_000      # Avoid tiny/degenerate application
    edge_width_px: int = 3             # Core/edge split width


def apply_stone_local_contrast(
    rgb: np.ndarray,
    strength: float = 1.04,
) -> np.ndarray:
    """Apply very conservative local contrast for stone.
    
    Args:
        rgb: HxWx3 float32 in [0,1]
        strength: Contrast multiplier (1.0 = no change, use values close to 1.0)
        
    Returns:
        Enhanced HxWx3 float32 in [0,1]
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}")
    
    from scipy.ndimage import gaussian_filter
    
    # Compute luminance
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    
    # Local contrast with conservative kernel
    local_mean = gaussian_filter(luma, sigma=7.0)
    contrast_luma = (luma - local_mean) * strength + local_mean
    
    # Scale RGB channels proportionally with safety
    epsilon = 1e-3
    scale = contrast_luma / (luma + epsilon)
    scale = np.clip(scale, 0.5, 2.0)  # Conservative scale range
    scale = np.where(luma > epsilon, scale, 1.0)
    
    result = rgb * scale[..., None]
    return np.clip(result, 0.0, 1.0)


def apply_stone_clarity(
    rgb: np.ndarray,
    strength: float = 1.02,
) -> np.ndarray:
    """Apply very subtle clarity boost for stone.
    
    Args:
        rgb: HxWx3 float32 in [0,1]
        strength: Clarity multiplier (1.0 = no change, use values 1.01-1.03)
        
    Returns:
        Enhanced HxWx3 float32 in [0,1]
    """
    from scipy.ndimage import gaussian_filter, laplace
    
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    
    # Gentle high-pass filter
    low = gaussian_filter(luma, sigma=12.0)
    high = luma - low
    
    # Very subtle boost (strength close to 1.0)
    clarity_factor = (strength - 1.0) * 0.5  # Scale down for subtlety
    clarity_luma = luma + high * clarity_factor
    
    # Apply to RGB
    epsilon = 1e-6
    scale = np.where(luma > 0, clarity_luma / (luma + epsilon), 1.0)
    scale = np.clip(scale, 0.8, 1.2)
    result = rgb * scale[..., None]
    
    return np.clip(result, 0.0, 1.0)


def apply_stone_saturation(
    rgb: np.ndarray,
    scale: float = 1.00,
) -> np.ndarray:
    """Apply saturation adjustment (typically neutral for stone).
    
    Args:
        rgb: HxWx3 float32 in [0,1]
        scale: Saturation multiplier (1.0 = no change)
        
    Returns:
        HxWx3 float32 in [0,1]
    """
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    result = luma[..., None] + (rgb - luma[..., None]) * scale
    return np.clip(result, 0.0, 1.0)


def apply_stone_response(
    rgb01: np.ndarray,
    stone_mask: np.ndarray,
    cfg: StoneResponseConfig,
    response_plan: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Apply stone-specific material response to image.
    
    Args:
        rgb01: HxWx3 float32 input image in [0,1]
        stone_mask: HxW float32 stone confidence mask in [0,1]
        cfg: Stone response configuration
        response_plan: Optional per-class response plan from PR-4A
        
    Returns:
        result_rgb: HxWx3 float32 enhanced image in [0,1]
        stats: Dict with per-region stats for auditability
    """
    if rgb01.shape[:2] != stone_mask.shape:
        raise ValueError(
            f"Image shape {rgb01.shape[:2]} != mask shape {stone_mask.shape}"
        )
    
    H, W, _ = rgb01.shape
    total_px = int((stone_mask > 0.5).sum())
    
    # Check minimum coverage
    if total_px < cfg.min_coverage_px:
        log.info(
            f"Stone coverage {total_px}px < min {cfg.min_coverage_px}px, skipping"
        )
        return rgb01, {
            'applied': False,
            'reason': 'below_min_coverage',
            'coverage_px': total_px,
        }
    
    result = rgb01.copy()
    
    # Extract core/edge masks using binary erosion
    from scipy.ndimage import binary_erosion
    
    binary_mask = stone_mask > 0.5
    struct_size = cfg.edge_width_px * 2 + 1
    struct = np.ones((struct_size, struct_size), dtype=bool)
    core_mask = binary_erosion(binary_mask, structure=struct)
    edge_mask = binary_mask & ~core_mask
    
    core_px = int(core_mask.sum())
    edge_px = int(edge_mask.sum())
    
    log.debug(
        f"Stone zones: core={core_px}px, edge={edge_px}px, total={total_px}px"
    )
    
    # Apply core enhancement
    core_delta_max = 0.0
    if core_px > 0:
        core_rgb = result[core_mask].copy()
        enhanced_core = core_rgb.copy()
        
        # Local contrast
        enhanced_core = apply_stone_local_contrast(
            enhanced_core.reshape(-1, 1, 3),
            strength=cfg.core_local_contrast,
        ).reshape(-1, 3)
        
        # Clarity
        enhanced_core = apply_stone_clarity(
            enhanced_core.reshape(-1, 1, 3),
            strength=cfg.core_clarity,
        ).reshape(-1, 3)
        
        # Saturation (typically neutral)
        enhanced_core = apply_stone_saturation(
            enhanced_core,
            scale=cfg.core_saturation,
        )
        
        # Delta clamp for safety
        delta = np.abs(enhanced_core - core_rgb)
        per_pixel_max_delta = delta.max(axis=1)
        core_delta_max = float(per_pixel_max_delta.max())
        
        excessive = per_pixel_max_delta > cfg.max_delta
        clamp_count = int(excessive.sum())
        if clamp_count > 0:
            log.warning(
                f"{clamp_count} core pixels exceeded max_delta={cfg.max_delta:.3f}; clamping"
            )
            # Clamp by scaling back the delta
            for i in np.where(excessive)[0]:
                scale_back = cfg.max_delta / per_pixel_max_delta[i]
                enhanced_core[i] = core_rgb[i] + (enhanced_core[i] - core_rgb[i]) * scale_back
        
        result[core_mask] = enhanced_core
    
    # Apply edge enhancement (very conservative)
    edge_delta_max = 0.0
    edge_delta_p95 = 0.0
    edge_clamp_count = 0
    if edge_px > 0:
        edge_rgb = result[edge_mask].copy()
        enhanced_edge = edge_rgb.copy()
        
        # Local contrast (very mild)
        enhanced_edge = apply_stone_local_contrast(
            enhanced_edge.reshape(-1, 1, 3),
            strength=cfg.edge_local_contrast,
        ).reshape(-1, 3)
        
        # Clarity (barely perceptible)
        enhanced_edge = apply_stone_clarity(
            enhanced_edge.reshape(-1, 1, 3),
            strength=cfg.edge_clarity,
        ).reshape(-1, 3)
        
        # Saturation
        enhanced_edge = apply_stone_saturation(
            enhanced_edge,
            scale=cfg.edge_saturation,
        )
        
        # Delta clamp
        delta = np.abs(enhanced_edge - edge_rgb)
        per_pixel_max_delta = delta.max(axis=1)
        edge_delta_max = float(per_pixel_max_delta.max())
        edge_delta_p95 = float(np.percentile(per_pixel_max_delta, 95))
        
        excessive = per_pixel_max_delta > cfg.max_delta
        edge_clamp_count = int(excessive.sum())
        if edge_clamp_count > 0:
            log.warning(
                f"{edge_clamp_count} edge pixels exceeded max_delta={cfg.max_delta:.3f}; clamping"
            )
            for i in np.where(excessive)[0]:
                scale_back = cfg.max_delta / per_pixel_max_delta[i]
                enhanced_edge[i] = edge_rgb[i] + (enhanced_edge[i] - edge_rgb[i]) * scale_back
        
        result[edge_mask] = enhanced_edge
    
    # Compute halo risk metric
    halo_risk = 'NONE'
    if edge_delta_p95 > cfg.halo_p95_threshold:
        halo_risk = 'HIGH'
        log.warning(
            f"Halo risk: edge p95 delta {edge_delta_p95:.4f} > threshold {cfg.halo_p95_threshold:.4f}"
        )
    elif edge_delta_p95 > cfg.halo_p95_threshold * 0.7:
        halo_risk = 'MEDIUM'
    
    # Compute mean delta over stone region
    stone_region = stone_mask > 0.5
    if stone_region.sum() > 0:
        delta_image = np.abs(result - rgb01)
        mean_delta = float(delta_image[stone_region].mean())
    else:
        mean_delta = 0.0
    
    stats = {
        'applied': True,
        'coverage_px': total_px,
        'core_px': core_px,
        'edge_px': edge_px,
        'core_delta_max': core_delta_max,
        'edge_delta_max': edge_delta_max,
        'edge_delta_p95': edge_delta_p95,
        'mean_delta': mean_delta,
        'halo_risk': halo_risk,
        'clamp_count': clamp_count if core_px > 0 else 0,
        'edge_clamp_count': edge_clamp_count,
    }
    
    log.info(
        f"Stone response applied: {total_px}px, "
        f"mean_delta={mean_delta:.4f}, "
        f"halo_risk={halo_risk}"
    )
    
    return result, stats
