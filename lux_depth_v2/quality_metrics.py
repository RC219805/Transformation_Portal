#!/usr/bin/env python3
"""
Correct Depth Quality Metrics for Luxury Rendering
===================================================

Replaces misleading "edge gradient" metrics with metrics that actually reflect
the practical requirements for DOF/masking/compositing in luxury real estate.

The Problem (from user feedback):
- Current metric: "edge gradient ≥180 vs achieved 0.09" is misleading
- Wrong proxy for "usable depth matte edges"
- Easy to compute incorrectly (wrong Sobel derivative, wrong scaling)

The Fix:
- Edge alignment score: correlation between RGB edges and depth edges
- Edge width: transition sharpness at object boundaries
- Halo/ringing detection: penalize overshoot artifacts
- Context-aware scoring for luxury interior use cases

Reference: User feedback 2025-12-17 - "Your edge 'sharpness' metric is misleading"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DepthQualityMetrics:
    """Comprehensive depth quality metrics for luxury rendering."""
    
    # Edge quality (critical for masking/DOF)
    edge_alignment_score: float  # [0, 1] - correlation with RGB edges
    edge_width_median_px: float  # Pixels - how sharp are depth transitions
    edge_overshoot_score: float  # [0, 1] - halo/ringing around edges
    
    # Spatial detail
    unique_levels_16bit: int  # Number of unique depth values
    effective_bit_depth: float  # log2(unique_levels)
    spatial_detail_score: float  # [0, 1] - variance in local windows
    
    # Range utilization
    histogram_entropy: float  # Bits - how well is dynamic range used
    percentile_99_range: float  # [0, 1] - robust range (excluding outliers)
    
    # Overall scores
    edge_quality_score: float  # [0, 100] - weighted edge metrics
    overall_quality_score: float  # [0, 100] - composite quality
    
    # Luxury-specific
    glass_boundary_quality: Optional[float] = None  # [0, 1] - glass edge handling
    reflection_coherence: Optional[float] = None  # [0, 1] - reflection consistency
    
    def __str__(self) -> str:
        return (
            f"DepthQualityMetrics(\n"
            f"  Edge Quality: {self.edge_quality_score:.1f}/100\n"
            f"  Edge Alignment: {self.edge_alignment_score:.3f}\n"
            f"  Edge Width: {self.edge_width_median_px:.1f}px\n"
            f"  Overshoot: {self.edge_overshoot_score:.3f}\n"
            f"  Unique Levels: {self.unique_levels_16bit:,}\n"
            f"  Effective Bits: {self.effective_bit_depth:.2f}\n"
            f"  Overall Quality: {self.overall_quality_score:.1f}/100\n"
            f")"
        )


class DepthQualityAnalyzer:
    """
    Analyze depth map quality for luxury rendering use cases.
    
    Focus on metrics that matter for practical use:
    - DOF/bokeh effects
    - Depth-based masking and compositing
    - Material-aware processing
    - Glass and reflection handling
    """
    
    def __init__(
        self,
        target_edge_alignment: float = 0.6,  # Minimum acceptable
        target_edge_width_px: float = 3.0,   # Maximum acceptable
        target_unique_levels: int = 10000    # Minimum for smooth gradients
    ):
        self.target_edge_alignment = target_edge_alignment
        self.target_edge_width_px = target_edge_width_px
        self.target_unique_levels = target_unique_levels
        logger.info("DepthQualityAnalyzer initialized")
    
    def compute_edge_alignment(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray
    ) -> float:
        """
        Edge alignment score: correlation between RGB edges and depth edges.
        
        This is the CORRECT metric for "does depth respect scene boundaries?"
        High score = depth edges align with image edges = good for masking.
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - skipping edge alignment")
            return 0.0
        
        # Convert inputs
        if rgb.dtype == np.float32:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb
        
        if depth.dtype == np.float32:
            depth_uint8 = (depth * 255).astype(np.uint8)
        elif depth.dtype == np.uint16:
            depth_uint8 = (depth / 256).astype(np.uint8)
        else:
            depth_uint8 = depth
        
        # RGB edges (Canny)
        rgb_gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb_uint8
        rgb_edges = cv2.Canny(rgb_gray, 50, 150).astype(np.float32) / 255.0
        
        # Depth edges (Sobel magnitude)
        sobel_x = cv2.Sobel(depth_uint8, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_uint8, cv2.CV_32F, 0, 1, ksize=3)
        depth_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        depth_edges = depth_edges / (depth_edges.max() + 1e-8)
        
        # Threshold depth edges for binary comparison
        depth_edges_binary = (depth_edges > 0.2).astype(np.float32)
        
        # Compute correlation
        correlation = np.corrcoef(rgb_edges.ravel(), depth_edges_binary.ravel())[0, 1]
        
        return max(0.0, min(1.0, correlation))
    
    def compute_edge_width(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray
    ) -> float:
        """
        Measure median width of depth transitions at RGB edges.
        
        Narrower transitions = sharper depth boundaries = better for masking.
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - skipping edge width")
            return 0.0
        
        # Find RGB edges
        if rgb.dtype == np.float32:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb
        
        rgb_gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb_uint8
        rgb_edges = cv2.Canny(rgb_gray, 50, 150)
        
        # Dilate RGB edges to create search region
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(rgb_edges, kernel, iterations=1)
        
        # Compute depth gradient magnitude in edge regions
        if depth.dtype == np.float32:
            depth_for_grad = depth
        elif depth.dtype == np.uint16:
            depth_for_grad = depth.astype(np.float32) / 65535.0
        else:
            depth_for_grad = depth.astype(np.float32) / 255.0
        
        sobel_x = cv2.Sobel(depth_for_grad, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_for_grad, cv2.CV_32F, 0, 1, ksize=3)
        depth_gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Sample gradient values at edge locations
        edge_gradients = depth_gradient[edge_region > 0]
        
        if len(edge_gradients) < 100:
            return 0.0
        
        # Estimate edge width from gradient (higher gradient = narrower edge)
        # Width ≈ depth_change / gradient
        # Assume typical depth change of 0.2 (normalized)
        edge_widths = 0.2 / (edge_gradients + 1e-8)
        edge_width_median = np.median(edge_widths)
        
        return float(edge_width_median)
    
    def compute_overshoot(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray
    ) -> float:
        """
        Detect halo/ringing artifacts (overshoot) around edges.
        
        Lower score = more overshoot = bad
        Higher score = clean edges = good
        """
        if not CV2_AVAILABLE:
            return 1.0  # Assume no overshoot if can't detect
        
        # Find RGB edges
        if rgb.dtype == np.float32:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb
        
        rgb_gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb_uint8
        rgb_edges = cv2.Canny(rgb_gray, 50, 150)
        
        # Create edge neighborhood (1-3px from edge)
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        edge_near = cv2.dilate(rgb_edges, kernel_small, iterations=1)
        edge_far = cv2.dilate(rgb_edges, kernel_large, iterations=1)
        edge_neighborhood = edge_far & ~edge_near
        
        # Compute local variance in edge neighborhoods
        if depth.dtype == np.float32:
            depth_norm = depth
        elif depth.dtype == np.uint16:
            depth_norm = depth.astype(np.float32) / 65535.0
        else:
            depth_norm = depth.astype(np.float32) / 255.0
        
        # Local variance
        mean_depth = cv2.GaussianBlur(depth_norm, (7, 7), 1.5)
        variance = (depth_norm - mean_depth) ** 2
        variance_blur = cv2.GaussianBlur(variance, (7, 7), 1.5)
        
        # Overshoot = high variance in edge neighborhoods
        edge_variance = variance_blur[edge_neighborhood > 0]
        
        if len(edge_variance) < 100:
            return 1.0
        
        # Score: low variance = low overshoot = good
        overshoot_score = 1.0 - np.clip(np.median(edge_variance) * 10, 0, 1)
        
        return float(overshoot_score)
    
    def compute_spatial_detail(self, depth: np.ndarray, window_size: int = 32) -> float:
        """
        Measure spatial detail: variance in local windows.
        
        Higher = more fine-grained depth structure
        Lower = smooth/flat (poor for luxury rendering)
        """
        if depth.dtype == np.float32:
            depth_norm = depth
        elif depth.dtype == np.uint16:
            depth_norm = depth.astype(np.float32) / 65535.0
        else:
            depth_norm = depth.astype(np.float32) / 255.0
        
        # Compute local variance using integral image (fast)
        h, w = depth_norm.shape
        num_windows_y = h // window_size
        num_windows_x = w // window_size
        
        if num_windows_y < 2 or num_windows_x < 2:
            return 0.0
        
        variances = []
        for i in range(num_windows_y):
            for j in range(num_windows_x):
                y0 = i * window_size
                y1 = y0 + window_size
                x0 = j * window_size
                x1 = x0 + window_size
                
                window = depth_norm[y0:y1, x0:x1]
                variances.append(window.var())
        
        # Average variance across windows
        detail_score = np.mean(variances)
        
        return float(detail_score)
    
    def analyze(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray,
        depth_uint16: Optional[np.ndarray] = None
    ) -> DepthQualityMetrics:
        """
        Comprehensive depth quality analysis.
        
        Args:
            rgb: RGB image as uint8 or float32
            depth: Depth map as float32 [0,1] or uint16
            depth_uint16: Optional 16-bit depth for bit depth analysis
            
        Returns:
            DepthQualityMetrics with all computed metrics
        """
        logger.info("Computing depth quality metrics...")
        
        # Normalize depth
        if depth.dtype == np.float32:
            depth_norm = depth
        elif depth.dtype == np.uint16:
            depth_norm = depth.astype(np.float32) / 65535.0
            depth_uint16 = depth if depth_uint16 is None else depth_uint16
        else:
            depth_norm = depth.astype(np.float32) / 255.0
        
        # Edge quality metrics
        edge_alignment = self.compute_edge_alignment(rgb, depth)
        edge_width = self.compute_edge_width(rgb, depth)
        edge_overshoot = self.compute_overshoot(rgb, depth)
        
        # Spatial detail
        spatial_detail = self.compute_spatial_detail(depth_norm)
        
        # Bit depth analysis
        if depth_uint16 is not None:
            unique_levels = len(np.unique(depth_uint16))
        else:
            unique_levels = len(np.unique((depth_norm * 65535).astype(np.uint16)))
        
        effective_bits = np.log2(max(unique_levels, 1))
        
        # Histogram analysis
        hist, _ = np.histogram(depth_norm, bins=256, range=(0, 1))
        hist_prob = hist / (hist.sum() + 1e-8)
        hist_entropy = -np.sum(hist_prob * np.log2(hist_prob + 1e-8))
        
        # Robust range (exclude outliers)
        p1 = np.percentile(depth_norm, 0.5)
        p99 = np.percentile(depth_norm, 99.5)
        percentile_range = p99 - p1
        
        # Compute scores
        # Edge quality score [0, 100]
        edge_quality = (
            edge_alignment * 40 +  # Most important
            (1.0 - min(edge_width / self.target_edge_width_px, 1.0)) * 30 +
            edge_overshoot * 30
        )
        
        # Overall quality score [0, 100]
        overall_quality = (
            edge_quality * 0.5 +  # Edge quality is critical
            min(unique_levels / self.target_unique_levels, 1.0) * 30 +
            spatial_detail * 100 * 0.2  # Spatial detail
        )
        
        metrics = DepthQualityMetrics(
            edge_alignment_score=edge_alignment,
            edge_width_median_px=edge_width,
            edge_overshoot_score=edge_overshoot,
            unique_levels_16bit=unique_levels,
            effective_bit_depth=effective_bits,
            spatial_detail_score=spatial_detail,
            histogram_entropy=hist_entropy,
            percentile_99_range=percentile_range,
            edge_quality_score=edge_quality,
            overall_quality_score=overall_quality
        )
        
        logger.info(f"✓ Quality analysis complete:\n{metrics}")
        return metrics
    
    def validate_for_luxury_rendering(self, metrics: DepthQualityMetrics) -> Tuple[bool, List[str]]:
        """
        Validate if depth map meets luxury rendering quality bar.
        
        Returns:
            (passes: bool, issues: List[str])
        """
        issues = []
        
        # Critical: edge alignment
        if metrics.edge_alignment_score < self.target_edge_alignment:
            issues.append(
                f"Edge alignment too low: {metrics.edge_alignment_score:.3f} < {self.target_edge_alignment:.3f} "
                f"(depth edges don't align with image boundaries)"
            )
        
        # Critical: unique levels
        if metrics.unique_levels_16bit < self.target_unique_levels:
            issues.append(
                f"Insufficient unique levels: {metrics.unique_levels_16bit:,} < {self.target_unique_levels:,} "
                f"(will cause banding in gradients)"
            )
        
        # Important: edge width
        if metrics.edge_width_median_px > self.target_edge_width_px:
            issues.append(
                f"Edges too soft: {metrics.edge_width_median_px:.1f}px > {self.target_edge_width_px:.1f}px "
                f"(masking will have halos)"
            )
        
        # Important: overshoot
        if metrics.edge_overshoot_score < 0.7:
            issues.append(
                f"Edge overshoot detected: score={metrics.edge_overshoot_score:.3f} "
                f"(ringing artifacts around boundaries)"
            )
        
        # Overall quality
        if metrics.overall_quality_score < 70:
            issues.append(
                f"Overall quality below luxury standard: {metrics.overall_quality_score:.1f}/100 < 70/100"
            )
        
        passes = len(issues) == 0
        
        if passes:
            logger.info("✅ Depth map PASSES luxury rendering quality bar")
        else:
            logger.warning(f"❌ Depth map FAILS luxury rendering quality bar ({len(issues)} issues)")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return passes, issues


def quick_quality_check(
    rgb: np.ndarray,
    depth: np.ndarray,
    depth_uint16: Optional[np.ndarray] = None
) -> DepthQualityMetrics:
    """Convenience function for quick quality check."""
    analyzer = DepthQualityAnalyzer()
    return analyzer.analyze(rgb, depth, depth_uint16)
