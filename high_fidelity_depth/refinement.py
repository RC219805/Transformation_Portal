#!/usr/bin/env python3
"""
Depth Refinement Module
=======================

PRIORITY 5: Edge-snapping and guided refinement for depth maps.

Refinement techniques:
- Edge-gated sharpening (AND-gate RGB + depth edges)
- Guided filter for edge-preserving smoothing
- CLAHE on low-frequency component
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_edges(image: np.ndarray, threshold_low: float = 50, threshold_high: float = 150) -> np.ndarray:
    """
    Detect edges using gradient-based method for float or Canny for uint8.
    
    Args:
        image: Grayscale image (uint8 or float32)
        threshold_low: Low threshold
        threshold_high: High threshold
        
    Returns:
        Binary edge map (uint8)
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Gradient-based for float
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        valid_grads = grad_mag[grad_mag > 1e-6]
        if len(valid_grads) > 0:
            thresh_low = np.percentile(valid_grads, 60)
            thresh_high = np.percentile(valid_grads, 85)
            
            strong_edges = (grad_mag > thresh_high).astype(np.uint8) * 255
            weak_edges = ((grad_mag > thresh_low) & (grad_mag <= thresh_high)).astype(np.uint8) * 255
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            strong_dilated = cv2.dilate(strong_edges, kernel, iterations=1)
            connected_weak = cv2.bitwise_and(weak_edges, strong_dilated)
            
            edges = cv2.bitwise_or(strong_edges, connected_weak)
        else:
            edges = np.zeros(image.shape, dtype=np.uint8)
    else:
        # Standard Canny for uint8
        edges = cv2.Canny(image, threshold_low, threshold_high)
    
    return edges


def edge_snap_refinement(
    depth: np.ndarray,
    rgb: np.ndarray,
    strength: float = 0.2,
    dilation: int = 5
) -> np.ndarray:
    """
    PRIORITY 5: Edge-gated sharpening - only where RGB AND depth edges exist.
    
    This prevents oversharpening in smooth regions while enhancing
    true depth discontinuities aligned with RGB edges.
    
    Args:
        depth: Depth map (float32 [0, 1])
        rgb: RGB image (uint8 or float32)
        strength: Sharpening strength [0, 1]
        dilation: Dilation radius for edge region
        
    Returns:
        Refined depth map (float32 [0, 1])
    """
    # Convert RGB to grayscale
    if rgb.ndim == 3:
        if rgb.dtype == np.float32:
            rgb_gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        rgb_gray = (rgb * 255).astype(np.uint8) if rgb.dtype == np.float32 else rgb
    
    # Detect edges
    rgb_edges = detect_edges(rgb_gray)
    depth_edges = detect_edges(depth)
    
    # AND-gate: sharpen only where both exist
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    rgb_dilated = cv2.dilate(rgb_edges, kernel)
    depth_dilated = cv2.dilate(depth_edges, kernel)
    
    snap_mask = (rgb_dilated > 0) & (depth_dilated > 0)
    
    # Unsharp mask
    blurred = cv2.GaussianBlur(depth, (0, 0), 1.0)
    sharp = depth + (depth - blurred) * strength
    sharp = np.clip(sharp, 0.0, 1.0)
    
    # Blend only where mask is active
    result = np.where(snap_mask, sharp, depth)
    
    edge_pixels = snap_mask.sum()
    total_pixels = snap_mask.size
    logger.info(f"Edge snapping: {edge_pixels}/{total_pixels} pixels ({100*edge_pixels/total_pixels:.1f}%), strength={strength}")
    
    return result


def guided_filter_refinement(
    depth: np.ndarray,
    rgb: np.ndarray,
    radius: int = 8,
    epsilon: float = 0.01
) -> np.ndarray:
    """
    Edge-preserving smoothing using guided filter.
    
    Uses RGB as guidance to preserve edges while smoothing flat regions.
    
    Args:
        depth: Depth map (float32 [0, 1])
        rgb: RGB image (uint8 or float32)
        radius: Filter radius
        epsilon: Regularization parameter (smaller = more edge-preserving)
        
    Returns:
        Filtered depth map (float32 [0, 1])
    """
    # Convert RGB to float32 [0, 1]
    if rgb.dtype == np.uint8:
        rgb_float = rgb.astype(np.float32) / 255.0
    else:
        rgb_float = rgb
    
    # Convert RGB to grayscale for guidance
    if rgb_float.ndim == 3:
        guide = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2GRAY)
    else:
        guide = rgb_float
    
    # Guided filter (using OpenCV's approximation via bilateral filter)
    # Note: True guided filter requires cv2.ximgproc.guidedFilter (opencv-contrib)
    # Fallback to bilateral filter with RGB guidance
    try:
        import cv2.ximgproc as ximgproc
        filtered = ximgproc.guidedFilter(guide.astype(np.float32), depth.astype(np.float32), radius, epsilon)
        logger.info(f"Guided filter applied: radius={radius}, epsilon={epsilon}")
    except (ImportError, AttributeError):
        # Fallback to bilateral filter
        depth_uint8 = (depth * 255).astype(np.uint8)
        filtered_uint8 = cv2.bilateralFilter(depth_uint8, d=radius*2+1, sigmaColor=50, sigmaSpace=radius)
        filtered = filtered_uint8.astype(np.float32) / 255.0
        logger.info(f"Bilateral filter applied (guided filter unavailable): radius={radius}")
    
    return filtered


def clahe_refinement(
    depth: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8
) -> np.ndarray:
    """
    Contrast-limited adaptive histogram equalization on depth map.
    
    Applied to low-frequency component to avoid edge artifacts.
    
    Args:
        depth: Depth map (float32 [0, 1])
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile size
        
    Returns:
        Enhanced depth map (float32 [0, 1])
    """
    # Decompose into low and high frequency
    sigma = 5.0
    depth_lf = cv2.GaussianBlur(depth, (0, 0), sigma)
    depth_hf = depth - depth_lf
    
    # Apply CLAHE to low-frequency component
    depth_lf_uint8 = (np.clip(depth_lf, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    depth_lf_enhanced = clahe.apply(depth_lf_uint8).astype(np.float32) / 255.0
    
    # Recombine
    depth_enhanced = depth_lf_enhanced + depth_hf
    depth_enhanced = np.clip(depth_enhanced, 0.0, 1.0)
    
    logger.info(f"CLAHE refinement applied: clip_limit={clip_limit}, tile_size={tile_size}")
    
    return depth_enhanced


def apply_refinement(
    depth: np.ndarray,
    rgb: np.ndarray,
    edge_snap: bool = True,
    edge_snap_strength: float = 0.2,
    guided_filter: bool = False,
    guided_radius: int = 8,
    clahe: bool = False,
    clahe_clip: float = 2.0
) -> np.ndarray:
    """
    Apply depth refinement pipeline.
    
    Args:
        depth: Input depth map (float32 [0, 1])
        rgb: RGB image (uint8 or float32)
        edge_snap: Enable edge snapping
        edge_snap_strength: Edge snapping strength
        guided_filter: Enable guided filter
        guided_radius: Guided filter radius
        clahe: Enable CLAHE
        clahe_clip: CLAHE clip limit
        
    Returns:
        Refined depth map (float32 [0, 1])
    """
    result = depth.copy()
    
    if guided_filter:
        result = guided_filter_refinement(result, rgb, radius=guided_radius)
    
    if edge_snap:
        result = edge_snap_refinement(result, rgb, strength=edge_snap_strength)
    
    if clahe:
        result = clahe_refinement(result, clip_limit=clahe_clip)
    
    return result
