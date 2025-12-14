"""
Image complexity scoring for auto-preset tier selection.

Provides fast, deterministic complexity metrics without heavy dependencies.
Used to decide whether to recommend APEX vs Max quality tiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ComplexityScore:
    """Image complexity metrics."""
    
    gradient_energy: float  # Sobel gradient magnitude (normalized)
    edge_density: float     # Proportion of edge pixels
    megapixels: float       # Image size in megapixels
    complexity_class: str   # "low" | "medium" | "high"
    
    @property
    def is_high_complexity(self) -> bool:
        """Check if image is high complexity (benefits from APEX)."""
        return self.complexity_class == "high"
    
    @property
    def is_medium_complexity(self) -> bool:
        """Check if image is medium complexity."""
        return self.complexity_class == "medium"
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "gradient_energy": float(self.gradient_energy),
            "edge_density": float(self.edge_density),
            "megapixels": float(self.megapixels),
            "complexity_class": self.complexity_class,
        }


def compute_complexity(
    image: np.ndarray,
    *,
    downsample_size: int = 512,
    gradient_threshold: float = 0.15,
    edge_density_threshold: float = 0.20,
    megapixel_threshold: float = 20.0,
) -> ComplexityScore:
    """
    Compute image complexity score.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as HxWx3 uint8 or float32.
    downsample_size : int
        Max size on longest side for gradient computation (default: 512).
    gradient_threshold : float
        Threshold for high gradient energy classification (default: 0.15).
    edge_density_threshold : float
        Threshold for high edge density classification (default: 0.20).
    megapixel_threshold : float
        Threshold for large image classification (default: 20.0 MP).
        
    Returns
    -------
    ComplexityScore
        Complexity metrics and classification.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")
    
    H, W, _ = image.shape
    megapixels = (H * W) / 1e6
    
    # Downsample for fast gradient computation
    img_small = _downsample_for_analysis(image, max_size=downsample_size)
    
    # Compute gradient energy
    gradient_energy = _compute_gradient_energy(img_small)
    
    # Compute edge density
    edge_density = _compute_edge_density(img_small, threshold=0.1)
    
    # Classify complexity
    complexity_class = _classify_complexity(
        gradient_energy=gradient_energy,
        edge_density=edge_density,
        megapixels=megapixels,
        grad_thresh=gradient_threshold,
        edge_thresh=edge_density_threshold,
        mp_thresh=megapixel_threshold,
    )
    
    return ComplexityScore(
        gradient_energy=gradient_energy,
        edge_density=edge_density,
        megapixels=megapixels,
        complexity_class=complexity_class,
    )


def _downsample_for_analysis(
    image: np.ndarray,
    max_size: int,
) -> np.ndarray:
    """
    Downsample image to max_size on longest side for fast analysis.
    
    Returns HxWx3 float32 in [0,1].
    """
    H, W, _ = image.shape
    
    if max(H, W) <= max_size:
        # Already small enough
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32, copy=False)
    
    # Compute new dimensions
    if H > W:
        new_h = max_size
        new_w = int(W * (max_size / H))
    else:
        new_w = max_size
        new_h = int(H * (max_size / W))
    
    # Use PIL for clean resize
    if image.dtype == np.uint8:
        pil_img = Image.fromarray(image)
    else:
        img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8)
    
    pil_small = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    img_small = np.array(pil_small, dtype=np.float32) / 255.0
    
    return img_small


def _compute_gradient_energy(image: np.ndarray) -> float:
    """
    Compute normalized gradient energy using Sobel.
    
    Parameters
    ----------
    image : np.ndarray
        HxWx3 float32 in [0,1].
        
    Returns
    -------
    float
        Gradient energy in [0,1] (typically 0.05-0.25 for natural images).
    """
    # Convert to grayscale
    gray = np.mean(image, axis=2).astype(np.float32)
    
    # Sobel gradients (no scipy dependency)
    gx = _sobel_x(gray)
    gy = _sobel_y(gray)
    
    # Gradient magnitude
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize by theoretical max (sqrt(2) for unit range)
    energy = float(np.mean(grad_mag) / np.sqrt(2.0))
    
    return np.clip(energy, 0.0, 1.0)


def _compute_edge_density(
    image: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    Compute proportion of edge pixels.
    
    Parameters
    ----------
    image : np.ndarray
        HxWx3 float32 in [0,1].
    threshold : float
        Gradient magnitude threshold for edge classification.
        
    Returns
    -------
    float
        Proportion of pixels classified as edges in [0,1].
    """
    gray = np.mean(image, axis=2).astype(np.float32)
    gx = _sobel_x(gray)
    gy = _sobel_y(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    edge_mask = grad_mag > threshold
    density = float(np.mean(edge_mask))
    
    return np.clip(density, 0.0, 1.0)


def _sobel_x(gray: np.ndarray) -> np.ndarray:
    """Sobel horizontal gradient (no scipy)."""
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    return _convolve2d_simple(gray, kernel)


def _sobel_y(gray: np.ndarray) -> np.ndarray:
    """Sobel vertical gradient (no scipy)."""
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0
    return _convolve2d_simple(gray, kernel)


def _convolve2d_simple(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Simple 2D convolution (valid mode, no padding).
    
    Good enough for gradient computation on downsampled images.
    """
    from scipy.ndimage import convolve
    # Use scipy if available (safe since we're only using it for optional complexity scoring)
    return convolve(image, kernel, mode='constant', cval=0.0)


def _classify_complexity(
    gradient_energy: float,
    edge_density: float,
    megapixels: float,
    grad_thresh: float,
    edge_thresh: float,
    mp_thresh: float,
) -> str:
    """
    Classify complexity as low/medium/high.
    
    High complexity criteria (any two trigger APEX recommendation):
      - High gradient energy (lots of detail/texture)
      - High edge density (complex boundaries)
      - Large image (>20MP by default)
    """
    high_gradient = gradient_energy >= grad_thresh
    high_edges = edge_density >= edge_thresh
    large_image = megapixels >= mp_thresh
    
    triggers = sum([high_gradient, high_edges, large_image])
    
    if triggers >= 2:
        return "high"
    elif triggers == 1:
        return "medium"
    else:
        return "low"
