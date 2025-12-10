"""
Autotune helpers for adaptive export configuration.

Phase 2 Slice 3 Integration: Image statistics computation for autotune_export_config.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ImageStats:
    """
    Image statistics for autotune decisions.
    
    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        megapixels: Image size in megapixels
        scene_complexity: Optional scene complexity score (0.0-1.0)
                         0.0 = simple (sky/gradients), 1.0 = complex (interiors/textures)
    """
    width: int
    height: int
    megapixels: float
    scene_complexity: Optional[float] = None


def compute_image_stats(input_path: Path, rgb_array: Optional[np.ndarray] = None) -> ImageStats:
    """
    Compute image statistics for autotune decisions.
    
    Can use either file path OR already-loaded array for efficiency.
    
    Args:
        input_path: Path to input image (used if rgb_array is None)
        rgb_array: Optional pre-loaded RGB array (H, W, 3) in [0, 1]
                  If provided, avoids redundant image loading
    
    Returns:
        ImageStats with dimensions and optional complexity score
    
    Example:
        >>> # From file path only
        >>> stats = compute_image_stats(Path("image.jpg"))
        >>> print(f"{stats.megapixels:.1f} MP")
        
        >>> # From already-loaded array (avoids redundant I/O)
        >>> import numpy as np
        >>> rgb = np.random.rand(4000, 6000, 3)
        >>> stats = compute_image_stats(Path("dummy.jpg"), rgb_array=rgb)
        >>> print(f"Complexity: {stats.scene_complexity:.3f}")
    """
    # Get dimensions
    if rgb_array is not None:
        H, W = rgb_array.shape[:2]
    else:
        from PIL import Image
        with Image.open(input_path) as im:
            W, H = im.size
    
    megapixels = (W * H) / 1_000_000
    
    # Compute scene complexity (gradient-based heuristic)
    complexity = None
    if rgb_array is not None:
        complexity = _estimate_scene_complexity(rgb_array)
    
    return ImageStats(
        width=W,
        height=H,
        megapixels=megapixels,
        scene_complexity=complexity
    )


def _estimate_scene_complexity(rgb_array: np.ndarray) -> float:
    """
    Estimate scene complexity from RGB array.
    
    Uses gradient magnitude as proxy for texture density:
    - Low complexity: Sky, water, gradients (homogeneous regions)
    - High complexity: Interiors, textures, fine details
    
    Args:
        rgb_array: RGB array (H, W, 3) in [0, 1]
    
    Returns:
        Complexity score in [0, 1], where:
        - 0.0-0.3: Simple (aerial-like, benefits from tiled_atomic)
        - 0.3-0.6: Medium complexity
        - 0.6-1.0: Complex (interiors, disable optimizations)
    """
    # Convert to grayscale for gradient computation
    if rgb_array.ndim == 3:
        gray = np.mean(rgb_array, axis=2, dtype=np.float32)
    else:
        gray = rgb_array.astype(np.float32)
    
    # Compute gradients (Sobel-like)
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_x = np.abs(np.diff(gray, axis=1))
    
    # Average gradient magnitude
    grad_mag = float(np.mean(grad_y) + np.mean(grad_x))
    
    # Normalize to [0, 1] range
    # Empirical calibration from benchmark scenes:
    # - Aerial (sky/water): ~0.02-0.04 → complexity ~0.2-0.3
    # - GreatRoom (interior): ~0.06-0.08 → complexity ~0.5-0.6
    # - Pool (complex textures): ~0.10-0.15 → complexity ~0.7-0.9
    GRADIENT_SCALE = 0.15  # Gradients above this are considered "highly complex"
    complexity = min(1.0, grad_mag / GRADIENT_SCALE)
    
    return complexity
