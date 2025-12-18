#!/usr/bin/env python3
"""
Edge Snapping for Depth Maps
=============================

Joint bilateral upsampling to snap depth discontinuities to RGB edges.

NOT OPTIONAL for luxury-grade DOF/masking:
"Given your current outputs (soft boundaries), edge snapping is not a 
luxury add-on. It's part of the minimum viable 'luxury-grade' result."

Reference: User feedback 2025-12-18
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EdgeSnappingConfig:
    """Configuration for edge snapping."""
    
    # Bilateral filter parameters
    sigma_spatial: float = 5.0    # Spatial smoothing (pixels)
    sigma_color: float = 0.1      # Color similarity (0-1)
    
    # Edge detection
    edge_threshold_low: int = 50   # Canny low threshold
    edge_threshold_high: int = 150 # Canny high threshold
    
    # Snapping strength
    snap_strength: float = 1.0     # 0=no snapping, 1=full snapping
    
    # Multi-scale processing
    use_multiscale: bool = False
    scales: list = None  # [0.5, 1.0, 2.0] for multi-scale
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [1.0]


class EdgeSnapper:
    """
    Snap depth edges to RGB edges using joint bilateral filtering.
    
    Key insight: Depth should have sharp discontinuities where RGB has edges,
    but smooth gradients within uniform regions.
    """
    
    def __init__(self, config: EdgeSnappingConfig):
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for edge snapping")
        
        self.config = config
        logger.info(f"EdgeSnapper: sigma_spatial={config.sigma_spatial} sigma_color={config.sigma_color}")
    
    def _detect_rgb_edges(self, rgb: np.ndarray) -> np.ndarray:
        """Detect edges in RGB image."""
        # Convert to grayscale
        if rgb.dtype == np.float32:
            gray = (rgb * 255).astype(np.uint8)
        else:
            gray = rgb
        
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(
            gray,
            self.config.edge_threshold_low,
            self.config.edge_threshold_high
        )
        
        return edges.astype(np.float32) / 255.0
    
    def _joint_bilateral_filter(
        self,
        depth: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """
        Apply joint bilateral filter: smooth depth while respecting RGB edges.
        
        This is the core edge-snapping operation.
        """
        # Ensure correct dtypes
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        
        if rgb.dtype == np.float32:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb
        
        if depth.min() < 0 or depth.max() > 1:
            # Normalize if needed
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Convert depth to uint8 for filtering
        depth_uint8 = (depth * 255).astype(np.uint8)
        
        # Apply joint bilateral filter
        # OpenCV jointBilateralFilter: filter src using guide
        # Parameters: (src, dst, d, sigmaColor, sigmaSpace)
        d = int(self.config.sigma_spatial * 2)  # Diameter
        sigma_color = self.config.sigma_color * 255  # Scale to 0-255
        sigma_space = self.config.sigma_spatial
        
        try:
            # Try ximgproc joint bilateral (better quality)
            filtered = cv2.ximgproc.jointBilateralFilter(
                rgb_uint8,
                depth_uint8,
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
        except AttributeError:
            # Fallback: standard bilateral filter (no joint guide)
            logger.warning("ximgproc not available, using standard bilateral")
            filtered = cv2.bilateralFilter(
                depth_uint8,
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
        
        # Convert back to float32 [0, 1]
        filtered_float = filtered.astype(np.float32) / 255.0
        
        return filtered_float
    
    def _snap_at_edges(
        self,
        depth: np.ndarray,
        depth_filtered: np.ndarray,
        edges: np.ndarray
    ) -> np.ndarray:
        """
        Apply snapping only at detected edges.
        
        At edges: use filtered depth (snapped to RGB)
        Away from edges: blend filtered and original
        """
        # Dilate edges slightly to capture edge neighborhood
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Blend: edges → filtered, non-edges → original
        snap_weight = edges_dilated * self.config.snap_strength
        snapped = (
            snap_weight * depth_filtered +
            (1 - snap_weight) * depth
        )
        
        return snapped
    
    def snap(
        self,
        depth: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """
        Snap depth edges to RGB edges.
        
        Args:
            depth: Depth map as float32 [0, 1]
            rgb: RGB image as uint8 or float32
            
        Returns:
            Snapped depth map as float32 [0, 1]
        """
        # Detect RGB edges
        edges = self._detect_rgb_edges(rgb)
        logger.debug(f"Detected edges: {(edges > 0).sum()} pixels")
        
        # Apply joint bilateral filtering
        depth_filtered = self._joint_bilateral_filter(depth, rgb)
        
        # Snap at edges only
        depth_snapped = self._snap_at_edges(depth, depth_filtered, edges)
        
        logger.info("✓ Edge snapping complete")
        return depth_snapped
    
    def snap_multiscale(
        self,
        depth: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """
        Multi-scale edge snapping for robustness.
        
        Process at multiple scales and combine:
        - Fine scale: capture sharp edges
        - Coarse scale: preserve global structure
        """
        if not self.config.use_multiscale:
            return self.snap(depth, rgb)
        
        h, w = depth.shape
        snapped_scales = []
        
        for scale in self.config.scales:
            if scale == 1.0:
                # Full resolution
                snapped = self.snap(depth, rgb)
                snapped_scales.append(snapped)
            else:
                # Rescale
                new_h, new_w = int(h * scale), int(w * scale)
                
                depth_scaled = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                snapped_scaled = self.snap(depth_scaled, rgb_scaled)
                
                # Resize back
                snapped = cv2.resize(snapped_scaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
                snapped_scales.append(snapped)
        
        # Combine scales (simple average)
        snapped_final = np.mean(snapped_scales, axis=0)
        
        logger.info(f"✓ Multi-scale snapping complete ({len(self.config.scales)} scales)")
        return snapped_final


def snap_depth_to_rgb(
    depth: np.ndarray,
    rgb: np.ndarray,
    sigma_spatial: float = 5.0,
    sigma_color: float = 0.1,
    snap_strength: float = 1.0
) -> np.ndarray:
    """
    Convenience function for edge snapping.
    
    Args:
        depth: Depth map as float32 [0, 1]
        rgb: RGB image as uint8 or float32
        sigma_spatial: Spatial smoothing radius (pixels)
        sigma_color: Color similarity threshold (0-1)
        snap_strength: Snapping strength (0-1)
        
    Returns:
        Snapped depth map as float32 [0, 1]
    """
    config = EdgeSnappingConfig(
        sigma_spatial=sigma_spatial,
        sigma_color=sigma_color,
        snap_strength=snap_strength
    )
    
    snapper = EdgeSnapper(config)
    return snapper.snap(depth, rgb)


# Preset configurations for different use cases

PRESETS = {
    "subtle": EdgeSnappingConfig(
        sigma_spatial=3.0,
        sigma_color=0.15,
        snap_strength=0.5,
        edge_threshold_low=30,
        edge_threshold_high=100
    ),
    "balanced": EdgeSnappingConfig(
        sigma_spatial=5.0,
        sigma_color=0.1,
        snap_strength=0.8,
        edge_threshold_low=50,
        edge_threshold_high=150
    ),
    "aggressive": EdgeSnappingConfig(
        sigma_spatial=7.0,
        sigma_color=0.05,
        snap_strength=1.0,
        edge_threshold_low=70,
        edge_threshold_high=200
    ),
    "multiscale": EdgeSnappingConfig(
        sigma_spatial=5.0,
        sigma_color=0.1,
        snap_strength=0.8,
        use_multiscale=True,
        scales=[0.5, 1.0, 2.0]
    ),
}
