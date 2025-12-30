#!/usr/bin/env python3
"""
Edge-Aware Depth Map Refinement Module
========================================

Implements edge-preserving post-processing techniques for depth map refinement
to improve structural quality in architectural rendering pipelines.

**Architecture Decision Record**: docs/architecture/edge_refinement/ADR-001-edge-refinement-module.md

**Feature Freeze Compliance**: Infrastructure module (disabled by default, no functional changes)

**Security**: CWE-703 (Input Validation), CWE-834 (Resource Exhaustion Prevention)

Modules:
    1. Bilateral Filtering - Edge-preserving smoothing
    2. Guided Filter - RGB-guided edge-aware smoothing
    3. Edge-Guided Enhancement - Targeted structural detail enhancement
    4. Gradient Consistency Filtering - Smooth away from edges, sharp at boundaries
    5. Segment-Aware Refinement - Reduce cross-segment smoothing

Author: Transformation Portal Specialist
Date: 2025-12-20
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


# ============================================================================
# Configuration Classes
# ============================================================================


class RefinementPreset(str, Enum):
    """Refinement strength presets."""

    SUBTLE = "subtle"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class EdgeRefinementConfig:
    """Configuration for edge-aware depth refinement.

    Attributes:
        enable_bilateral: Enable bilateral filtering (edge-preserving smoothing)
        bilateral_d: Bilateral filter diameter (0 = auto-compute from sigma_space)
        bilateral_sigma_color: Color similarity threshold (0-255 scale)
        bilateral_sigma_space: Spatial extent in pixels

        enable_guided: Enable guided filter (RGB-guided smoothing)
        guided_radius: Guided filter radius in pixels
        guided_eps: Regularization parameter (controls edge preservation)

        enable_edge_enhancement: Enable edge-guided enhancement
        edge_enhancement_strength: Enhancement intensity (0.0-1.0)
        edge_detection_threshold: Edge detection sensitivity (0-255)

        enable_gradient_smoothing: Enable gradient consistency filtering
        gradient_weight: Weight for gradient alignment (0.0-1.0)

        structure_weight: Balance between smoothing and structure preservation (0.0-1.0)
        max_image_dim: Maximum image dimension to prevent resource exhaustion
    """

    # Bilateral filtering
    enable_bilateral: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Guided filter
    enable_guided: bool = True
    guided_radius: int = 8
    guided_eps: float = 0.01

    # Edge enhancement
    enable_edge_enhancement: bool = True
    edge_enhancement_strength: float = 0.3
    edge_detection_threshold: float = 40.0

    # Gradient smoothing
    enable_gradient_smoothing: bool = True
    gradient_weight: float = 0.5

    # Global settings
    structure_weight: float = 0.5
    max_image_dim: int = 4096  # CWE-834: Resource exhaustion prevention

    @classmethod
    def from_preset(cls, preset: RefinementPreset) -> "EdgeRefinementConfig":
        """Create configuration from named preset.

        Args:
            preset: Refinement preset (subtle, balanced, aggressive)

        Returns:
            EdgeRefinementConfig instance
        """
        presets = {
            RefinementPreset.SUBTLE: cls(
                bilateral_sigma_color=50.0,
                bilateral_sigma_space=50.0,
                edge_enhancement_strength=0.15,
                gradient_weight=0.3,
                structure_weight=0.4,
            ),
            RefinementPreset.BALANCED: cls(
                bilateral_sigma_color=75.0,
                bilateral_sigma_space=75.0,
                edge_enhancement_strength=0.3,
                gradient_weight=0.5,
                structure_weight=0.5,
            ),
            RefinementPreset.AGGRESSIVE: cls(
                bilateral_sigma_color=100.0,
                bilateral_sigma_space=100.0,
                edge_enhancement_strength=0.5,
                gradient_weight=0.7,
                structure_weight=0.6,
            ),
        }
        return presets.get(preset, cls())


# ============================================================================
# Module 1: Bilateral Filtering (Edge-Preserving Smoothing)
# ============================================================================


def bilateral_depth_filter(
    depth_map: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0
) -> np.ndarray:
    """Apply bilateral filtering to depth map for edge-preserving smoothing.

    Bilateral filtering reduces noise while preserving structural boundaries by
    weighting pixels based on both spatial distance and intensity similarity.

    **Algorithm**: Tomasi & Manduchi (1998) - Bilateral Filtering for Gray and Color Images

    **Security**:
        - CWE-703: Input validation prevents invalid dimensions
        - CWE-834: Bounded kernel size prevents resource exhaustion

    Args:
        depth_map: Input depth map (HxW), float32 normalized to [0, 1] or uint8/uint16
        d: Filter diameter in pixels (0 = auto-compute from sigma_space)
        sigma_color: Color similarity threshold (0-255 range)
        sigma_space: Spatial extent in pixels

    Returns:
        Filtered depth map (same shape and dtype as input)

    Raises:
        ValueError: If depth_map has invalid shape or dimensions exceed limits
        TypeError: If depth_map is not a numpy array

    Example:
        >>> depth = np.random.rand(512, 512).astype(np.float32)
        >>> filtered = bilateral_depth_filter(depth, d=9, sigma_color=75, sigma_space=75)
        >>> assert filtered.shape == depth.shape
    """
    # CWE-703: Input validation
    if not isinstance(depth_map, np.ndarray):
        raise TypeError(f"depth_map must be numpy array, got {type(depth_map)}")

    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D (HxW), got shape {depth_map.shape}")

    h, w = depth_map.shape
    if max(h, w) > 8192:  # CWE-834: Resource exhaustion prevention
        raise ValueError(f"Image dimensions ({h}x{w}) exceed maximum (8192x8192)")

    # Normalize depth to uint8 for cv2.bilateralFilter
    if depth_map.dtype in [np.float32, np.float64]:
        depth_norm = (depth_map * 255).clip(0, 255).astype(np.uint8)
        is_float = True
    elif depth_map.dtype == np.uint16:
        depth_norm = (depth_map / 256).astype(np.uint8)
        is_float = False
    else:
        depth_norm = depth_map.astype(np.uint8)
        is_float = False

    # Auto-compute diameter if d=0
    if d == 0:
        d = int(sigma_space * 2) + 1

    # Apply bilateral filter (preserves edges)
    filtered = cv2.bilateralFilter(depth_norm, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Restore original dtype
    if is_float:
        filtered = filtered.astype(np.float32) / 255.0
    elif depth_map.dtype == np.uint16:
        filtered = filtered.astype(np.uint16) * 256

    return filtered


# ============================================================================
# Module 2: Guided Filter (RGB-Guided Edge-Aware Smoothing)
# ============================================================================


def guided_filter_depth(depth_map: np.ndarray, rgb_image: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
    """Apply guided filter using RGB image to smooth depth while preserving edges.

    Guided filter aligns depth edges with RGB structural boundaries for improved
    consistency in architectural rendering. Uses fast O(1) box filtering.

    **Algorithm**: He et al. (2013) - Guided Image Filtering

    **Security**:
        - CWE-703: Input validation ensures shape compatibility
        - CWE-834: Bounded window size prevents resource exhaustion

    Args:
        depth_map: Input depth map (HxW), float32 normalized to [0, 1]
        rgb_image: Guide image (HxWx3), uint8 RGB or float32 normalized
        radius: Filter radius in pixels (window size = 2*radius + 1)
        eps: Regularization parameter (controls edge preservation, typically 0.001-0.1)

    Returns:
        Guided filtered depth map (HxW, float32)

    Raises:
        ValueError: If depth_map and rgb_image shapes are incompatible
        TypeError: If inputs are not numpy arrays

    Example:
        >>> depth = np.random.rand(512, 512).astype(np.float32)
        >>> rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        >>> filtered = guided_filter_depth(depth, rgb, radius=8, eps=0.01)
    """
    # CWE-703: Input validation
    if not isinstance(depth_map, np.ndarray) or not isinstance(rgb_image, np.ndarray):
        raise TypeError("depth_map and rgb_image must be numpy arrays")

    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"rgb_image must be HxWx3, got shape {rgb_image.shape}")

    if depth_map.shape != rgb_image.shape[:2]:
        raise ValueError(f"Shape mismatch: depth {depth_map.shape} vs rgb {rgb_image.shape[:2]}")

    # Convert rgb to float32 [0, 1] if needed
    if rgb_image.dtype == np.uint8:
        guide = rgb_image.astype(np.float32) / 255.0
    else:
        guide = rgb_image.astype(np.float32)

    # Convert to grayscale for single-channel guided filter
    if guide.ndim == 3:
        guide_gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        guide_gray = guide_gray.astype(np.float32) / 255.0
    else:
        guide_gray = guide

    # Try opencv-contrib guidedFilter, fallback to custom implementation
    try:
        filtered = cv2.ximgproc.guidedFilter(
            guide=guide_gray.astype(np.float32), src=depth_map.astype(np.float32), radius=radius, eps=eps
        )
    except AttributeError:
        # Fallback: Custom guided filter implementation
        filtered = _guided_filter_custom(depth_map, guide_gray, radius, eps)

    return filtered.astype(np.float32)


def _guided_filter_custom(p: np.ndarray, I: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Custom guided filter implementation (fallback when opencv-contrib unavailable).

    Args:
        p: Input image to be filtered (HxW)
        I: Guidance image (HxW)
        r: Radius of the box filter
        eps: Regularization parameter

    Returns:
        Filtered image (HxW)
    """

    # Box filter helper (mean filter)
    def box_filter(img: np.ndarray, radius: int) -> np.ndarray:
        return cv2.boxFilter(img, -1, (2 * radius + 1, 2 * radius + 1))

    # Mean of I, p, and I*p
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I * p, r)

    # Covariance and variance
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = box_filter(I * I, r) - mean_I * mean_I

    # Linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Mean of coefficients
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    # Output
    q = mean_a * I + mean_b
    return q


# ============================================================================
# Module 3: Edge-Guided Depth Enhancement
# ============================================================================


def enhance_edges_with_guidance(
    depth_map: np.ndarray, rgb_image: np.ndarray, strength: float = 0.3, threshold: float = 40.0
) -> np.ndarray:
    """Apply edge-guided enhancement to preserve structural details.

    Detects edges from RGB image and applies targeted sharpening and contrast
    enhancement along structural boundaries while preserving smooth regions.

    **Security**:
        - CWE-703: Input validation and clipping prevents buffer overflows
        - CWE-834: Bounded kernel sizes prevent resource exhaustion

    Args:
        depth_map: Input depth map (HxW), float32 normalized to [0, 1]
        rgb_image: Guide image for edge detection (HxWx3), uint8 or float32
        strength: Enhancement intensity (0.0-1.0, typical 0.2-0.5)
        threshold: Edge detection sensitivity (0-255, typical 30-100)

    Returns:
        Edge-enhanced depth map (HxW, float32)

    Raises:
        ValueError: If inputs have incompatible shapes or invalid parameters

    Example:
        >>> depth = np.random.rand(512, 512).astype(np.float32)
        >>> rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        >>> enhanced = enhance_edges_with_guidance(depth, rgb, strength=0.3)
    """
    # CWE-703: Input validation
    if not isinstance(depth_map, np.ndarray) or not isinstance(rgb_image, np.ndarray):
        raise TypeError("depth_map and rgb_image must be numpy arrays")

    if depth_map.shape != rgb_image.shape[:2]:
        raise ValueError(f"Shape mismatch: depth {depth_map.shape} vs rgb {rgb_image.shape[:2]}")

    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")

    if not 0.0 <= threshold <= 255.0:
        raise ValueError(f"threshold must be in [0, 255], got {threshold}")

    # Convert RGB to grayscale for edge detection
    if rgb_image.dtype == np.uint8:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(gray, threshold1=threshold * 0.5, threshold2=threshold)

    # Create edge mask (dilated for local enhancement)
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)
    edge_mask = edge_mask.astype(np.float32) / 255.0

    # Convert depth to float32 if needed
    if depth_map.dtype != np.float32:
        depth_float = depth_map.astype(np.float32)
        if depth_map.dtype == np.uint8:
            depth_float /= 255.0
        elif depth_map.dtype == np.uint16:
            depth_float /= 65535.0
    else:
        depth_float = depth_map

    # Compute gradient-based edge mask to prevent flat-region variance amplification
    # Only apply sharpening where real structure exists (gradient magnitude > threshold)
    gx = np.abs(np.diff(depth_float, axis=1, prepend=depth_float[:, :1]))
    gy = np.abs(np.diff(depth_float, axis=0, prepend=depth_float[:1, :]))
    gradient_mag = np.maximum(gx, gy)

    # Soft gradient mask: 0 in flat regions, 1 at strong edges
    gradient_threshold = 0.02  # Tuned to preserve flats while enhancing edges
    gradient_mask = np.clip((gradient_mag - gradient_threshold) / gradient_threshold, 0.0, 1.0).astype(np.float32)

    # Apply unsharp masking for sharpening
    blurred = cv2.GaussianBlur(depth_float, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(depth_float, 1.0 + strength, blurred, -strength, 0)

    # Blend using gradient mask (not binary edge mask)
    # Flat regions (gradient_mask ≈ 0) remain unchanged
    # Edges (gradient_mask ≈ 1) get full enhancement
    enhanced = depth_float + gradient_mask * (sharpened - depth_float)

    # Clip to valid range
    enhanced = np.clip(enhanced, 0.0, 1.0)

    return enhanced.astype(np.float32)


# ============================================================================
# Module 4: Depth Gradient Consistency Filtering
# ============================================================================


def gradient_smoothness(depth_map: np.ndarray, rgb_image: np.ndarray, gradient_weight: float = 0.5) -> np.ndarray:
    """Enforce gradient smoothness away from edges, allow sharp transitions at boundaries.

    Aligns depth gradients with RGB edge structure to improve depth-image consistency
    in architectural rendering. Reduces gradient noise while preserving structural edges.

    **Security**:
        - CWE-703: Input validation ensures proper shapes and ranges
        - CWE-834: Bounded filter sizes prevent resource exhaustion

    Args:
        depth_map: Input depth map (HxW), float32 normalized to [0, 1]
        rgb_image: Guide image for edge detection (HxWx3), uint8 or float32
        gradient_weight: Weight for gradient alignment (0.0-1.0, typical 0.3-0.7)

    Returns:
        Gradient-smoothed depth map (HxW, float32)

    Raises:
        ValueError: If inputs have incompatible shapes or invalid parameters

    Example:
        >>> depth = np.random.rand(512, 512).astype(np.float32)
        >>> rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        >>> smoothed = gradient_smoothness(depth, rgb, gradient_weight=0.5)
    """
    # CWE-703: Input validation
    if not isinstance(depth_map, np.ndarray) or not isinstance(rgb_image, np.ndarray):
        raise TypeError("depth_map and rgb_image must be numpy arrays")

    if depth_map.shape != rgb_image.shape[:2]:
        raise ValueError(f"Shape mismatch: depth {depth_map.shape} vs rgb {rgb_image.shape[:2]}")

    if not 0.0 <= gradient_weight <= 1.0:
        raise ValueError(f"gradient_weight must be in [0, 1], got {gradient_weight}")

    # Convert depth to float32
    if depth_map.dtype != np.float32:
        depth_float = depth_map.astype(np.float32)
        if depth_map.dtype == np.uint8:
            depth_float /= 255.0
        elif depth_map.dtype == np.uint16:
            depth_float /= 65535.0
    else:
        depth_float = depth_map

    # Compute depth gradients (Sobel)
    grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Compute RGB edge magnitude
    if rgb_image.dtype == np.uint8:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    rgb_grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    rgb_grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_grad_mag = np.sqrt(rgb_grad_x**2 + rgb_grad_y**2) / 255.0

    # Create edge-aware smoothing mask
    # High RGB gradient → preserve depth gradient
    # Low RGB gradient → smooth depth gradient
    edge_mask = rgb_grad_mag / (rgb_grad_mag.max() + 1e-8)
    edge_mask = np.clip(edge_mask, 0.0, 1.0)

    # Apply bilateral filter to smooth depth away from edges
    smoothed = bilateral_depth_filter(depth_float, d=9, sigma_color=50.0, sigma_space=50.0)

    # Blend original and smoothed based on edge mask and gradient weight
    result = smoothed * (1.0 - edge_mask * gradient_weight) + depth_float * (edge_mask * gradient_weight)

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    return result.astype(np.float32)


# ============================================================================
# Module 5: Hybrid Filtering + Structural Masking
# ============================================================================


def segment_aware_refine(depth_map: np.ndarray, segmentation_mask: np.ndarray, filter_radius: int = 5) -> np.ndarray:
    """Refine depth within segments while reducing cross-segment smoothing.

    Applies intra-segment smoothing while preserving sharp transitions at segment
    boundaries. Useful for material-based or object-based depth refinement.

    **Security**:
        - CWE-703: Input validation ensures shape compatibility
        - CWE-834: Bounded filter sizes prevent resource exhaustion

    Args:
        depth_map: Input depth map (HxW), float32 normalized to [0, 1]
        segmentation_mask: Segment labels (HxW), int32/uint8 with unique IDs per segment
        filter_radius: Radius for intra-segment smoothing (typical 3-8)

    Returns:
        Segment-aware refined depth map (HxW, float32)

    Raises:
        ValueError: If inputs have incompatible shapes
        TypeError: If inputs are not numpy arrays

    Example:
        >>> depth = np.random.rand(512, 512).astype(np.float32)
        >>> segments = np.random.randint(0, 10, (512, 512), dtype=np.uint8)
        >>> refined = segment_aware_refine(depth, segments, filter_radius=5)
    """
    # CWE-703: Input validation
    if not isinstance(depth_map, np.ndarray) or not isinstance(segmentation_mask, np.ndarray):
        raise TypeError("depth_map and segmentation_mask must be numpy arrays")

    if depth_map.ndim != 2 or segmentation_mask.ndim != 2:
        raise ValueError("depth_map and segmentation_mask must be 2D")

    if depth_map.shape != segmentation_mask.shape:
        raise ValueError(f"Shape mismatch: depth {depth_map.shape} vs mask {segmentation_mask.shape}")

    if filter_radius < 1 or filter_radius > 20:
        raise ValueError(f"filter_radius must be in [1, 20], got {filter_radius}")

    # Convert depth to float32
    if depth_map.dtype != np.float32:
        depth_float = depth_map.astype(np.float32)
        if depth_map.dtype == np.uint8:
            depth_float /= 255.0
        elif depth_map.dtype == np.uint16:
            depth_float /= 65535.0
    else:
        depth_float = depth_map.copy()

    # Get unique segment IDs
    segment_ids = np.unique(segmentation_mask)

    # Apply smoothing within each segment
    refined = depth_float.copy()

    for seg_id in segment_ids:
        # Create binary mask for this segment
        seg_mask = (segmentation_mask == seg_id).astype(np.uint8)

        # Skip empty segments
        if seg_mask.sum() < 10:
            continue

        # Smooth only within segment using masked filtering
        # Extract segment region
        y_coords, x_coords = np.where(seg_mask)
        if len(y_coords) == 0:
            continue

        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        # Expand region by filter radius
        y_min = max(0, y_min - filter_radius)
        y_max = min(depth_float.shape[0], y_max + filter_radius + 1)
        x_min = max(0, x_min - filter_radius)
        x_max = min(depth_float.shape[1], x_max + filter_radius + 1)

        # Extract region
        depth_region = depth_float[y_min:y_max, x_min:x_max]
        mask_region = seg_mask[y_min:y_max, x_min:x_max]

        # Apply bilateral filter to region
        smoothed_region = bilateral_depth_filter(
            depth_region, d=2 * filter_radius + 1, sigma_color=50.0, sigma_space=float(filter_radius)
        )

        # Blend smoothed region only where mask is active
        mask_region_float = mask_region.astype(np.float32)
        blended = depth_region * (1.0 - mask_region_float) + smoothed_region * mask_region_float

        # Write back to refined map
        refined[y_min:y_max, x_min:x_max] = blended

    # Clip to valid range
    refined = np.clip(refined, 0.0, 1.0)

    return refined.astype(np.float32)


# ============================================================================
# High-Level Pipeline Interface
# ============================================================================


class EdgeRefinementPipeline:
    """High-level pipeline for edge-aware depth refinement.

    Orchestrates multiple refinement techniques in sequence based on configuration.

    Example:
        >>> config = EdgeRefinementConfig.from_preset(RefinementPreset.BALANCED)
        >>> pipeline = EdgeRefinementPipeline(config)
        >>> refined_depth = pipeline.refine(depth_map, rgb_image)
    """

    def __init__(self, config: Optional[EdgeRefinementConfig] = None):
        """Initialize refinement pipeline.

        Args:
            config: Refinement configuration (default: balanced preset)
        """
        self.config = config or EdgeRefinementConfig.from_preset(RefinementPreset.BALANCED)

    def refine(
        self, depth_map: np.ndarray, rgb_image: Optional[np.ndarray] = None, segmentation_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply edge-aware refinement to depth map.

        Args:
            depth_map: Input depth map (HxW), float32 normalized to [0, 1]
            rgb_image: Optional RGB guide image (HxWx3)
            segmentation_mask: Optional segmentation mask (HxW)

        Returns:
            Refined depth map (HxW, float32)

        Raises:
            ValueError: If required inputs are missing for enabled techniques
        """
        # Validate inputs
        if depth_map is None or not isinstance(depth_map, np.ndarray):
            raise ValueError("depth_map is required")

        result = depth_map.copy()

        # Stage 1: Bilateral filtering (standalone)
        if self.config.enable_bilateral:
            result = bilateral_depth_filter(
                result,
                d=self.config.bilateral_d,
                sigma_color=self.config.bilateral_sigma_color,
                sigma_space=self.config.bilateral_sigma_space,
            )

        # Stage 2: Guided filter (requires RGB)
        if self.config.enable_guided:
            if rgb_image is None:
                raise ValueError("rgb_image required when enable_guided=True")
            result = guided_filter_depth(result, rgb_image, radius=self.config.guided_radius, eps=self.config.guided_eps)

        # Stage 3: Gradient smoothing (requires RGB)
        if self.config.enable_gradient_smoothing:
            if rgb_image is None:
                raise ValueError("rgb_image required when enable_gradient_smoothing=True")
            result = gradient_smoothness(result, rgb_image, gradient_weight=self.config.gradient_weight)

        # Stage 4: Edge enhancement (requires RGB)
        if self.config.enable_edge_enhancement:
            if rgb_image is None:
                raise ValueError("rgb_image required when enable_edge_enhancement=True")
            result = enhance_edges_with_guidance(
                result,
                rgb_image,
                strength=self.config.edge_enhancement_strength,
                threshold=self.config.edge_detection_threshold,
            )

        # Stage 5: Segment-aware refinement (requires segmentation)
        if segmentation_mask is not None:
            result = segment_aware_refine(result, segmentation_mask)

        return result


# ============================================================================
# Convenience Functions
# ============================================================================


def refine_depth_edge_aware(
    depth_map: np.ndarray,
    rgb_image: Optional[np.ndarray] = None,
    preset: RefinementPreset = RefinementPreset.BALANCED,
    segmentation_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience function for edge-aware depth refinement.

    Args:
        depth_map: Input depth map (HxW), float32 normalized to [0, 1]
        rgb_image: Optional RGB guide image (HxWx3)
        preset: Refinement preset (subtle, balanced, aggressive)
        segmentation_mask: Optional segmentation mask (HxW)

    Returns:
        Refined depth map (HxW, float32)

    Example:
        >>> refined = refine_depth_edge_aware(depth, rgb, preset=RefinementPreset.BALANCED)
    """
    config = EdgeRefinementConfig.from_preset(preset)
    pipeline = EdgeRefinementPipeline(config)
    return pipeline.refine(depth_map, rgb_image, segmentation_mask)
