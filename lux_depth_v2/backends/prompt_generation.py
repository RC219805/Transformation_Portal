# lux_depth_v2/backends/prompt_generation.py
"""
Intelligent prompt generation for EfficientSAM refinement.

This module generates point prompts from SegFormer confidence masks,
using mask-driven sampling to improve alignment and refinement quality.

Key improvements over naive box-center prompts:
- Samples from high-confidence regions (not just geometric center)
- Enforces spatial distribution (farthest-point sampling)
- Conservative negative points near boundaries only
- Handles edge cases (tiny masks, low confidence, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import logging

import numpy as np
from scipy.ndimage import distance_transform_edt

log = logging.getLogger(__name__)


@dataclass
class PromptGenerationConfig:
    """Configuration for mask-driven prompt generation."""
    
    # Foreground point sampling
    num_fg_points: int = 4
    fg_confidence_threshold: float = 0.60
    fg_top_percentile: float = 10.0  # sample from top 10% of confident pixels
    
    # Negative point sampling
    num_bg_points: int = 2
    bg_boundary_band: int = 10  # pixels from mask edge
    
    # Skip guards
    min_mask_pixels: int = 500
    max_roi_side: int = 4096
    
    # Spatial distribution
    enforce_spacing: bool = True
    min_spacing_pixels: int = 50


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int,
    initial_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Sample N points from a point cloud using farthest-point sampling.
    
    Parameters
    ----------
    points : np.ndarray
        Mx2 array of (y, x) coordinates
    n_samples : int
        Number of points to sample
    initial_idx : Optional[int]
        Index of first point to select; if None, chooses randomly
    
    Returns
    -------
    np.ndarray
        n_samples x 2 array of selected points
    """
    if len(points) <= n_samples:
        return points
    
    n_total = len(points)
    selected_indices = []
    
    # Initialize with random or specified point
    if initial_idx is None:
        initial_idx = np.random.randint(0, n_total)
    selected_indices.append(initial_idx)
    
    # Track minimum distances to selected set
    min_dists = np.full(n_total, np.inf, dtype=np.float32)
    
    for _ in range(n_samples - 1):
        last_selected = points[selected_indices[-1]]
        
        # Update minimum distances
        dists_to_last = np.linalg.norm(points - last_selected, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)
        
        # Select point with maximum minimum distance
        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)
    
    return points[selected_indices]


def generate_prompts_from_mask(
    base_mask: np.ndarray,
    cfg: PromptGenerationConfig,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate point prompts from a SegFormer confidence mask.
    
    Parameters
    ----------
    base_mask : np.ndarray
        HxW float32 confidence map in [0,1]
    cfg : PromptGenerationConfig
        Configuration for prompt generation
    
    Returns
    -------
    fg_points : np.ndarray
        Nx2 array of (y, x) foreground points in pixel coordinates
    bg_points : np.ndarray
        Mx2 array of (y, x) background points in pixel coordinates
    stats : dict
        Metadata about prompt generation
    """
    if base_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {base_mask.shape}")
    
    H, W = base_mask.shape
    stats = {
        "fg_points_generated": 0,
        "bg_points_generated": 0,
        "skip_reason": None,
    }
    
    # Skip guard: mask too small
    confident_pixels = (base_mask > cfg.fg_confidence_threshold).sum()
    if confident_pixels < cfg.min_mask_pixels:
        stats["skip_reason"] = f"mask_too_small ({confident_pixels} < {cfg.min_mask_pixels})"
        return np.zeros((0, 2)), np.zeros((0, 2)), stats
    
    # Sample foreground points from high-confidence region
    percentile_thresh = np.percentile(
        base_mask[base_mask > cfg.fg_confidence_threshold],
        100 - cfg.fg_top_percentile,
    )
    high_conf_mask = base_mask >= percentile_thresh
    
    high_conf_coords = np.column_stack(np.where(high_conf_mask))  # Nx2 (y, x)
    
    if len(high_conf_coords) == 0:
        stats["skip_reason"] = "no_high_confidence_pixels"
        return np.zeros((0, 2)), np.zeros((0, 2)), stats
    
    # Apply farthest-point sampling for spatial distribution
    if cfg.enforce_spacing and len(high_conf_coords) > cfg.num_fg_points:
        fg_points = farthest_point_sampling(high_conf_coords, cfg.num_fg_points)
    else:
        # Random sampling if fewer candidates than needed
        indices = np.random.choice(
            len(high_conf_coords),
            size=min(cfg.num_fg_points, len(high_conf_coords)),
            replace=False,
        )
        fg_points = high_conf_coords[indices]
    
    stats["fg_points_generated"] = len(fg_points)
    
    # Generate background points near boundary (conservative)
    binary_mask = base_mask > cfg.fg_confidence_threshold
    
    # Distance transform to find boundary band
    dist_transform = distance_transform_edt(~binary_mask)
    boundary_band = (dist_transform > 0) & (dist_transform <= cfg.bg_boundary_band)
    
    bg_coords = np.column_stack(np.where(boundary_band))
    
    bg_points = np.zeros((0, 2))
    if len(bg_coords) > 0 and cfg.num_bg_points > 0:
        if cfg.enforce_spacing and len(bg_coords) > cfg.num_bg_points:
            bg_points = farthest_point_sampling(bg_coords, cfg.num_bg_points)
        else:
            indices = np.random.choice(
                len(bg_coords),
                size=min(cfg.num_bg_points, len(bg_coords)),
                replace=False,
            )
            bg_points = bg_coords[indices]
    
    stats["bg_points_generated"] = len(bg_points)
    
    return fg_points, bg_points, stats


def compute_roi_from_mask(
    base_mask: np.ndarray,
    padding: int = 50,
    max_side: int = 4096,
) -> Tuple[Optional[Tuple[int, int, int, int]], dict]:
    """
    Compute a padded ROI bounding box from a mask.
    
    Parameters
    ----------
    base_mask : np.ndarray
        HxW confidence map
    padding : int
        Pixels to pad around mask bbox
    max_side : int
        Maximum allowed ROI side length
    
    Returns
    -------
    roi : Optional[Tuple[int, int, int, int]]
        (y0, x0, y1, x1) in pixel coordinates, or None if invalid
    stats : dict
        Metadata
    """
    H, W = base_mask.shape
    stats = {"skip_reason": None}
    
    # Find bounding box of confident pixels
    binary_mask = base_mask > 0.5
    coords = np.column_stack(np.where(binary_mask))
    
    if len(coords) == 0:
        stats["skip_reason"] = "empty_mask"
        return None, stats
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Apply padding
    y0 = max(0, y_min - padding)
    x0 = max(0, x_min - padding)
    y1 = min(H, y_max + padding)
    x1 = min(W, x_max + padding)
    
    roi_h = y1 - y0
    roi_w = x1 - x0
    
    # Skip guard: ROI too large
    if max(roi_h, roi_w) > max_side:
        stats["skip_reason"] = f"roi_too_large ({roi_h}x{roi_w} > {max_side})"
        return None, stats
    
    return (y0, x0, y1, x1), stats
