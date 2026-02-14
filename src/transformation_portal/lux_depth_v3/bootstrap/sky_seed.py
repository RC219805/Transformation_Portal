"""Sky bootstrap heuristic for Materials V3.

This module provides heuristic-based sky detection to seed SAM2 segmentation.
Sky is an amorphous "stuff" material that benefits from spatial and color priors
rather than pure object detection.

Strategy:
1. Top-of-frame spatial prior (sky typically in upper 40-60%)
2. Low gradient magnitude (smooth regions without texture)
3. Brightness threshold (sky is generally bright)
4. Color characteristics (optional blue/cyan bias for clear sky)

The heuristic produces:
- Coarse binary mask for initial region identification
- Confidence score for reliability assessment
- Bounding box for region localization
- Positive/negative prompt points for SAM2 refinement
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def detect_sky_seed(image: np.ndarray, config: Any) -> Dict[str, Any]:
    """Detect sky region using heuristics.

    Strategy:
    1. Top-of-frame prior (sky typically in upper 40-60%)
    2. Low gradient magnitude (smooth regions)
    3. Brightness threshold (sky typically bright)
    4. Color characteristics (blue/cyan bias in RGB)

    Args:
        image: RGB image (H,W,3) in [0,1] float32 or uint8/uint16
        config: Configuration object with sky_* attributes

    Returns:
        Dict with:
        - coarse_mask: (H,W) binary mask [0,1] float32
        - confidence: float [0,1]
        - bbox: (x0,y0,x1,y1) or None
        - points_positive: List[(x,y)] points inside sky
        - points_negative: List[(x,y)] points outside sky

    Raises:
        ValueError: If image has invalid shape or dtype
    """
    # Input validation
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")

    # Normalize to [0,1] float32
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        img = image.astype(np.float32) / 65535.0
    elif image.dtype in (np.float32, np.float64):
        img = image.astype(np.float32)
        # Assume already normalized, but clip to be safe
        img = np.clip(img, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")

    H, W = img.shape[:2]

    # Get configuration parameters with safe defaults
    top_region_fraction = getattr(config, "sky_top_region_fraction", 0.5)
    gradient_threshold = getattr(config, "sky_gradient_threshold", 0.05)
    brightness_threshold = getattr(config, "sky_brightness_threshold", 0.4)

    # 1. Top-of-frame prior
    # Sky is typically in the upper portion of the image
    top_region_height = int(H * top_region_fraction)
    top_mask = np.zeros((H, W), dtype=np.float32)
    top_mask[:top_region_height, :] = 1.0

    # 2. Low gradient magnitude (smooth sky)
    # Sky has low texture and smooth gradients
    gray = np.mean(img, axis=2) if img.ndim == 3 else img
    grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[0:1]))
    grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, 0:1]))
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    smooth_mask = (grad_mag < gradient_threshold).astype(np.float32)

    # 3. Brightness threshold
    # Sky is generally brighter than foreground objects
    brightness = np.mean(img, axis=2) if img.ndim == 3 else img
    bright_mask = (brightness > brightness_threshold).astype(np.float32)

    # 4. Color characteristics (optional, simple version)
    # For now, we rely on the above three heuristics
    # Future: Add blue/cyan channel ratio for clear sky detection

    # Combine heuristics with AND logic
    # All three conditions must be met for a pixel to be considered sky
    combined = top_mask * smooth_mask * bright_mask
    coarse_mask = (combined > 0.5).astype(np.float32)

    # Compute confidence based on coverage
    # Sky should occupy a reasonable portion but not the entire image
    sky_pixels = float(np.sum(coarse_mask > 0.5))
    total_pixels = float(H * W)
    coverage_ratio = sky_pixels / total_pixels if total_pixels > 0 else 0.0

    # Confidence increases with coverage up to ~50%, then decreases
    # (full image coverage suggests poor detection)
    if coverage_ratio < 0.05:
        confidence = coverage_ratio / 0.05  # Low coverage = low confidence
    elif coverage_ratio <= 0.5:
        confidence = 1.0  # Good coverage = high confidence
    else:
        # Over-coverage suggests failure (e.g., overcast uniformly bright image)
        confidence = max(0.0, 1.0 - (coverage_ratio - 0.5) * 2.0)

    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Generate bbox if mask non-empty
    bbox = _compute_bbox(coarse_mask) if confidence > 0.1 else None

    # Generate prompt points for SAM2 refinement
    points_positive = _sample_points_inside(coarse_mask, num_points=5)
    points_negative = _sample_points_outside(coarse_mask, H, W, num_points=5)

    return {
        "coarse_mask": coarse_mask,
        "confidence": confidence,
        "bbox": bbox,
        "points_positive": points_positive,
        "points_negative": points_negative,
    }


def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    """Compute bounding box from binary mask.

    Args:
        mask: Binary mask (H, W) with values in {0, 1}

    Returns:
        Bounding box (x0, y0, x1, y1) or None if mask is empty
    """
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _sample_points_inside(mask: np.ndarray, num_points: int = 5) -> List[Tuple[int, int]]:
    """Sample random points inside the mask.

    Args:
        mask: Binary mask (H, W) with values in {0, 1}
        num_points: Number of points to sample

    Returns:
        List of (x, y) coordinates inside the mask
    """
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return []

    # Sample without replacement if possible
    num_to_sample = min(num_points, len(ys))
    indices = np.random.choice(len(ys), size=num_to_sample, replace=False)

    return [(int(xs[i]), int(ys[i])) for i in indices]


def _sample_points_outside(mask: np.ndarray, H: int, W: int, num_points: int = 5) -> List[Tuple[int, int]]:
    """Sample random points outside the mask.

    Args:
        mask: Binary mask (H, W) with values in {0, 1}
        H: Image height
        W: Image width
        num_points: Number of points to sample

    Returns:
        List of (x, y) coordinates outside the mask
    """
    ys, xs = np.where(mask <= 0.5)
    if ys.size == 0:
        return []

    # Sample without replacement if possible
    num_to_sample = min(num_points, len(ys))
    indices = np.random.choice(len(ys), size=num_to_sample, replace=False)

    return [(int(xs[i]), int(ys[i])) for i in indices]
