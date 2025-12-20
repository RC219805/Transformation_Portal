#!/usr/bin/env python3
"""
Production-Grade Depth Refinement Pipeline
===========================================

Fixes critical implementation errors identified in validation:
1. Edge metrics bug (0.09 anomaly) - now computed on float32, not uint8
2. Guided filter actually applied (not skipped)
3. CLAHE enhancement for flat region recovery
4. Edge-snap refinement (sharpen only at RGB edges)

Reference: User feedback 2025-12-18
"These are errors worth fixing—issues suppressing real depth quality"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class DepthRefinementConfig:
    """Configuration for production-grade depth refinement."""
    
    # CLAHE enhancement (flat region recovery)
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    
    # Edge-aware filtering (priority cascade)
    use_edge_filter: bool = True
    edge_filter_radius: int = 8
    edge_filter_eps: float = 0.01
    
    # Edge-snap refinement (sharpen only at RGB edges)
    use_edge_snap: bool = True
    edge_snap_amount: float = 1.5
    edge_snap_radius: float = 1.0
    edge_snap_threshold: int = 50
    
    # Bilateral fallback parameters
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75


class ProductionDepthRefiner:
    """
    Production-grade depth refinement addressing validation failures.
    
    Fixes:
    - Edge metrics computed on float32 (not uint8 collapsed)
    - Guided filter actually applied (not skipped)
    - CLAHE for flat region detail recovery
    - Edge-snap for RGB-aligned boundaries
    """
    
    def __init__(self, config: DepthRefinementConfig):
        self.config = config
        logger.info("ProductionDepthRefiner initialized")
    
    def _apply_clahe(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE for flat region detail recovery.
        
        From diagnosis: CLAHE drove ~20-40x unique level improvement
        and ~5-6x edge gradient improvement.
        """
        if not self.config.use_clahe:
            return depth
        
        # Convert to uint16 for CLAHE (preserves 16-bit precision)
        if depth.dtype == np.float32:
            depth_uint16 = (depth * 65535).astype(np.uint16)
        else:
            depth_uint16 = depth.astype(np.uint16)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_grid, self.config.clahe_tile_grid)
        )
        
        # Apply CLAHE
        enhanced = clahe.apply(depth_uint16)
        
        # Convert back to float32 [0, 1]
        enhanced_float = enhanced.astype(np.float32) / 65535.0
        
        logger.info(f"✓ CLAHE applied: clip={self.config.clahe_clip_limit} grid={self.config.clahe_tile_grid}")
        return enhanced_float
    
    def _apply_guided_filter(
        self, 
        depth: np.ndarray, 
        guide: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply edge-aware filtering with priority cascade.
        
        Priority:
        1. cv2.ximgproc.guidedFilter (best quality)
        2. cv2.ximgproc.jointBilateralFilter (RGB-guided fallback)
        3. cv2.bilateralFilter (depth-only last resort)
        """
        if not self.config.use_edge_filter:
            return depth
        
        # Convert depth to uint8 for filtering
        if depth.dtype == np.float32:
            depth_uint8 = (depth * 255).astype(np.uint8)
        else:
            depth_uint8 = (depth / 256).astype(np.uint8) if depth.dtype == np.uint16 else depth
        
        try:
            # Priority 1: Guided filter (best)
            if guide is not None and hasattr(cv2, 'ximgproc'):
                # Convert guide to uint8 if needed
                if guide.dtype == np.float32:
                    guide_uint8 = (guide * 255).astype(np.uint8)
                else:
                    guide_uint8 = guide
                
                filtered = cv2.ximgproc.guidedFilter(
                    guide=guide_uint8,
                    src=depth_uint8,
                    radius=self.config.edge_filter_radius,
                    eps=self.config.edge_filter_eps
                )
                logger.info(f"✓ Guided filter applied: r={self.config.edge_filter_radius} eps={self.config.edge_filter_eps}")
            
            # Priority 2: Joint bilateral (fallback)
            elif guide is not None and hasattr(cv2.ximgproc, 'jointBilateralFilter'):
                if guide.dtype == np.float32:
                    guide_uint8 = (guide * 255).astype(np.uint8)
                else:
                    guide_uint8 = guide
                
                filtered = cv2.ximgproc.jointBilateralFilter(
                    joint=guide_uint8,
                    src=depth_uint8,
                    d=self.config.bilateral_d,
                    sigmaColor=self.config.bilateral_sigma_color,
                    sigmaSpace=self.config.bilateral_sigma_space
                )
                logger.info("✓ Joint bilateral filter applied (guided filter unavailable)")
            
            # Priority 3: Standard bilateral (last resort)
            else:
                filtered = cv2.bilateralFilter(
                    src=depth_uint8,
                    d=self.config.bilateral_d,
                    sigmaColor=self.config.bilateral_sigma_color,
                    sigmaSpace=self.config.bilateral_sigma_space
                )
                logger.warning("✓ Bilateral filter applied (ximgproc unavailable)")
        
        except Exception as e:
            logger.error(f"Edge filtering failed: {e}, returning original depth")
            return depth
        
        # Convert back to float32 [0, 1]
        filtered_float = filtered.astype(np.float32) / 255.0
        
        return filtered_float
    
    def _detect_rgb_edges(self, rgb: np.ndarray) -> np.ndarray:
        """Detect edges in RGB image for edge-snap refinement."""
        # Convert to grayscale
        if rgb.dtype == np.float32:
            gray = (rgb * 255).astype(np.uint8)
        else:
            gray = rgb
        
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, self.config.edge_snap_threshold, self.config.edge_snap_threshold * 3)
        
        # Dilate slightly to capture edge neighborhood
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return edges_dilated.astype(np.float32) / 255.0
    
    def _apply_edge_snap(
        self, 
        depth: np.ndarray, 
        rgb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply edge-snap refinement: sharpen only where RGB has edges.
        
        This prevents halos while achieving crisp architectural boundaries.
        """
        if not self.config.use_edge_snap or rgb is None:
            return depth
        
        # Detect RGB edges
        edge_mask = self._detect_rgb_edges(rgb)
        
        # Create unsharp mask
        blurred = cv2.GaussianBlur(depth, (0, 0), self.config.edge_snap_radius)
        sharpened = depth + self.config.edge_snap_amount * (depth - blurred)
        sharpened = np.clip(sharpened, 0, 1)
        
        # Apply sharpening only at edges
        depth_snapped = edge_mask * sharpened + (1 - edge_mask) * depth
        
        logger.info(f"✓ Edge-snap applied: amount={self.config.edge_snap_amount} at {(edge_mask > 0.5).sum()} edge pixels")
        return depth_snapped
    
    def refine(
        self,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply production-grade refinement pipeline.
        
        Stages:
        1. CLAHE (flat region recovery)
        2. Guided filter (edge-aware smoothing)
        3. Edge-snap (RGB-aligned sharpening)
        
        Args:
            depth: Depth map as float32 [0, 1] or uint16
            rgb: Optional RGB guide image
            
        Returns:
            Refined depth as float32 [0, 1]
        """
        # Normalize to float32 [0, 1]
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 65535.0
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0
        
        # Stage 1: CLAHE
        depth = self._apply_clahe(depth)
        
        # Stage 2: Guided filter
        depth = self._apply_guided_filter(depth, guide=rgb)
        
        # Stage 3: Edge-snap
        depth = self._apply_edge_snap(depth, rgb=rgb)
        
        logger.info("✓ Production refinement complete")
        return depth


def compute_robust_edge_metrics(depth: np.ndarray, rgb: Optional[np.ndarray] = None) -> dict:
    """
    Compute edge metrics correctly (float32, not uint8 collapsed).
    
    Fixes the "0.09" anomaly by computing on non-quantized depth.
    
    Args:
        depth: Depth map as float32/uint16/uint8
        rgb: Optional RGB for edge alignment metric
    
    Returns:
        {
            'gradient_mean': float,
            'gradient_p95': float,
            'gradient_p99': float,
            'gradient_max': float,
            'edge_alignment': float (if rgb provided),
            'effective_scale': str  # "0-255 equivalent"
        }
    """
    # Ensure float32 [0, 1]
    if depth.dtype == np.uint16:
        depth_norm = depth.astype(np.float32) / 65535.0
    elif depth.dtype == np.uint8:
        depth_norm = depth.astype(np.float32) / 255.0
    else:
        depth_norm = depth.astype(np.float32)
        if depth_norm.max() > 1.0:
            depth_norm = depth_norm / depth_norm.max()
    
    # Compute Sobel gradients on FLOAT (critical fix)
    sobel_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    
    # Gradient magnitude
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Scale to "0-255 equivalent" for comparison with old metrics
    gradient_mag_scaled = gradient_mag * 255.0
    
    metrics = {
        'gradient_mean': float(gradient_mag_scaled.mean()),
        'gradient_p95': float(np.percentile(gradient_mag_scaled, 95)),
        'gradient_p99': float(np.percentile(gradient_mag_scaled, 99)),
        'gradient_max': float(gradient_mag_scaled.max()),
        'effective_scale': '0-255 equivalent'
    }
    
    # Edge alignment with RGB (if provided)
    if rgb is not None:
        # RGB edges (Canny)
        if rgb.dtype == np.float32:
            gray = (rgb * 255).astype(np.uint8)
        else:
            gray = rgb
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        
        rgb_edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        
        # Depth edges (thresholded gradient)
        depth_edges = (gradient_mag > np.percentile(gradient_mag, 90)).astype(np.float32)
        
        # Correlation
        correlation = np.corrcoef(rgb_edges.ravel(), depth_edges.ravel())[0, 1]
        metrics['edge_alignment'] = float(correlation)
    
    return metrics


def compute_depth_statistics(depth: np.ndarray) -> dict:
    """
    Compute comprehensive depth statistics.
    
    Includes unique levels, flat ratio, effective bits (not just headline "65536").
    """
    # Convert to uint16 for analysis
    if depth.dtype == np.float32:
        depth_uint16 = (depth * 65535).astype(np.uint16)
    else:
        depth_uint16 = depth.astype(np.uint16)
    
    # Unique levels
    unique_levels = len(np.unique(depth_uint16))
    effective_bits = np.log2(max(unique_levels, 1))
    
    # Flat ratio (regions with very low gradient)
    gradient = np.gradient(depth_uint16.astype(np.float32))
    gradient_mag = np.sqrt(gradient[0]**2 + gradient[1]**2)
    flat_pixels = (gradient_mag < 1.0).sum()
    flat_ratio = flat_pixels / gradient_mag.size
    
    # Percentile range (robust dynamic range)
    p1 = np.percentile(depth_uint16, 1)
    p99 = np.percentile(depth_uint16, 99)
    percentile_range = p99 - p1
    
    stats = {
        'unique_levels': unique_levels,
        'unique_levels_max': 65536,
        'effective_bits': effective_bits,
        'flat_ratio': flat_ratio,
        'percentile_range': int(percentile_range),
        'min': int(depth_uint16.min()),
        'max': int(depth_uint16.max()),
        'mean': int(depth_uint16.mean()),
        'std': int(depth_uint16.std())
    }
    
    return stats


# Convenience function

def refine_depth_production(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    use_clahe: bool = True,
    use_edge_filter: bool = True,
    use_edge_snap: bool = True
) -> np.ndarray:
    """
    One-shot production refinement with sensible defaults.
    
    Args:
        depth: Depth map as float32 [0,1] or uint16
        rgb: Optional RGB guide for edge-aware processing
        use_clahe: Enable CLAHE flat region recovery
        use_edge_filter: Enable guided/bilateral filtering
        use_edge_snap: Enable RGB-aligned edge sharpening
        
    Returns:
        Refined depth as float32 [0, 1]
    """
    config = DepthRefinementConfig(
        use_clahe=use_clahe,
        use_edge_filter=use_edge_filter,
        use_edge_snap=use_edge_snap
    )
    
    refiner = ProductionDepthRefiner(config)
    return refiner.refine(depth, rgb)
