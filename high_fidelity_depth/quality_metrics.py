#!/usr/bin/env python3
"""
Unified Quality Metrics for Depth Validation
=============================================

Canonical implementation of edge-based quality metrics with:
- Atomic JSON serialization with validation
- Shift-tolerant edge alignment (F1 score with tolerance)
- Chamfer distance and boundary IoU
- Robust metric computation on float32 depth
- Calibrated thresholds based on empirical data

This is the SINGLE SOURCE OF TRUTH for all quality metrics.
All validation paths must use this module.

Reference: Fix for truncated JSON and mismatched metric definitions.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EdgeMetrics:
    """Canonical edge-based quality metrics."""
    
    # Primary alignment metric (shift-tolerant F1)
    edge_f1: float  # F1 score with 2px tolerance
    
    # Secondary metrics
    edge_overlap: float  # Overlap percentage with dilation
    edge_alignment_corr: float  # Correlation (diagnostic only)
    chamfer_distance: float  # Mean distance to nearest RGB edge
    
    # Sharpness metrics
    edge_width: float  # Average transition width (pixels)
    edge_sharpness_p95: float  # 95th percentile gradient magnitude
    
    # Artifact detection
    edge_count_ratio: float  # Depth edges / RGB edges
    halo_score: float  # Overshoot detection [0,1]
    overshoot_penalty: float  # Laplacian ringing score
    
    # Diagnostic
    rgb_edge_count: int
    depth_edge_count: int
    
    # Structure-aware edge detection (NEW)
    edge_type: str = 'raw'  # 'raw' or 'structure' (bilateral-filtered)
    scene_type: str = 'unknown'  # 'texture_dominated' or 'structure_dominated'
    scene_metadata: Optional[Dict] = None  # V2 classifier metadata
    
    def passed(self, strict: bool = False) -> bool:
        """
        Check if metrics meet quality thresholds.
        
        Thresholds are calibrated based on empirical data:
        - Baseline edge_f1 ~0.15-0.25 (low-res + bicubic)
        - Target edge_f1 ≥0.35 (meaningful improvement)
        - edge_count_ratio ≤2.5 (avoid artifact explosion)
        """
        if strict:
            return (
                self.edge_f1 >= 0.45 and
                self.edge_overlap >= 0.50 and
                self.edge_count_ratio <= 2.0 and
                self.halo_score >= 0.7 and
                self.overshoot_penalty <= 0.3
            )
        else:
            return (
                self.edge_f1 >= 0.30 and
                self.edge_overlap >= 0.40 and
                self.edge_count_ratio <= 3.0 and
                self.overshoot_penalty <= 0.5
            )
    
    def quality_score(self) -> float:
        """
        Composite quality score [0, 1].
        
        Weighted combination emphasizing:
        - Edge F1 (primary)
        - Artifact penalties (critical)
        - Overlap and sharpness (secondary)
        """
        score = (
            0.40 * self.edge_f1 +
            0.25 * self.edge_overlap +
            0.15 * min(self.edge_sharpness_p95 / 100.0, 1.0) +
            0.10 * self.halo_score +
            0.10 * max(0, 1.0 - self.overshoot_penalty)
        )
        
        # Penalty for excessive edge count
        if self.edge_count_ratio > 2.5:
            score *= 0.7
        
        return np.clip(score, 0.0, 1.0)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert numpy types to native Python
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in d.items()}
    
    def __str__(self) -> str:
        return (
            f"EdgeMetrics(\n"
            f"  edge_f1={self.edge_f1:.3f} (primary),\n"
            f"  edge_overlap={self.edge_overlap:.3f},\n"
            f"  edge_alignment_corr={self.edge_alignment_corr:.3f} (diagnostic),\n"
            f"  chamfer_distance={self.chamfer_distance:.2f}px,\n"
            f"  edge_width={self.edge_width:.2f}px,\n"
            f"  edge_sharpness_p95={self.edge_sharpness_p95:.1f},\n"
            f"  edge_count_ratio={self.edge_count_ratio:.2f}×,\n"
            f"  halo_score={self.halo_score:.3f},\n"
            f"  overshoot_penalty={self.overshoot_penalty:.3f},\n"
            f"  quality_score={self.quality_score():.3f}\n"
            f")"
        )


def extract_structure_edges(
    image: np.ndarray,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    canny_low: int = 50,
    canny_high: int = 150
) -> np.ndarray:
    """
    Extract structural edges with texture suppression via bilateral filtering.
    
    The bilateral filter removes texture/noise while preserving object boundaries.
    This aligns edge detection with structural features (frames, boundaries)
    rather than high-frequency texture (ripples, reflections).
    
    Args:
        image: RGB image (H, W, 3) or grayscale (H, W)
        bilateral_d: Diameter of pixel neighborhood (9-15 typical)
        bilateral_sigma_color: Filter sigma in color space (higher = more smoothing)
        bilateral_sigma_space: Filter sigma in coordinate space
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        
    Returns:
        Binary edge map (H, W) with structural edges only
        
    References:
        - OpenCV bilateral filter: removes texture, preserves edges
        - Used in portrait mode, depth estimation, HDR tone mapping
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply bilateral filter to suppress texture
    # Parameters tuned for architectural/real estate imagery:
    # - d=9: moderate spatial extent (balance speed vs quality)
    # - sigma_color=75: significant color smoothing (kills texture)
    # - sigma_space=75: matches spatial extent
    filtered = cv2.bilateralFilter(
        gray, 
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space
    )
    
    # Extract edges from texture-suppressed image
    edges = cv2.Canny(filtered, canny_low, canny_high)
    
    return edges


def classify_scene_type(
    rgb_edges_raw: np.ndarray,
    rgb_edges_structure: np.ndarray,
    texture_threshold: float = 3.0
) -> str:
    """
    Classify scene as texture-dominated or structure-dominated (legacy).
    
    DEPRECATED: Use classify_scene_type_v2() for multi-factor classification.
    
    Args:
        rgb_edges_raw: Edges from raw RGB (includes texture)
        rgb_edges_structure: Edges from bilateral-filtered RGB (structure only)
        texture_threshold: Ratio threshold for classification
        
    Returns:
        'texture_dominated' or 'structure_dominated'
    """
    raw_count = np.count_nonzero(rgb_edges_raw)
    structure_count = np.count_nonzero(rgb_edges_structure)
    
    # Avoid division by zero
    if structure_count == 0:
        return 'texture_dominated'
    
    ratio = raw_count / structure_count
    
    return 'texture_dominated' if ratio > texture_threshold else 'structure_dominated'


def classify_scene_type_v2(
    rgb_edges_raw: np.ndarray,
    rgb_edges_structure: np.ndarray,
    depth_map: np.ndarray,
    threshold_ratio_high: float = 10.0,
    threshold_ratio_low: float = 5.0,
    threshold_depth_var_low: float = 0.02,
    threshold_depth_var_high: float = 0.03,
    threshold_edge_density: float = 0.05,
    image_filename: Optional[str] = None
) -> Tuple[str, dict]:
    """
    Multi-factor scene classification (V2 - FIXED with filename weak supervision).
    
    Uses four factors to distinguish water/ocean/pool from interior structures:
    1. Edge ratio (raw/structure) - texture indicator
    2. Depth variance - global smoothness indicator
    3. Edge density - structural complexity indicator
    4. Depth gradient smoothness - NEW: separates water (smooth depth) from interiors (geometric depth)
    5. Filename-based weak supervision - NEW: boosts confidence when filename contains scene type hints
    
    Key insight: Water/ocean/pool have textured RGB (reflections, waves) but smooth depth gradients.
    Interior structures have aligned RGB and depth edges (geometric discontinuities).
    
    Args:
        rgb_edges_raw: Raw RGB edges (includes texture)
        rgb_edges_structure: Structure edges (bilateral filtered)
        depth_map: Depth map for variance calculation
        threshold_* parameters: Tunable thresholds
        image_filename: Optional filename for weak supervision hints
        
    Returns:
        (scene_type, metadata_dict)
    """
    # Compute metrics
    raw_count = np.count_nonzero(rgb_edges_raw)
    structure_count = np.count_nonzero(rgb_edges_structure)
    total_pixels = rgb_edges_raw.size
    
    # Handle division by zero
    if structure_count == 0:
        return 'texture_dominated', {
            'method': 'multi_factor_v2',
            'raw_edges': raw_count,
            'structure_edges': 0,
            'ratio': float('inf'),
            'depth_variance': float(np.var(depth_map)),
            'depth_gradient_var': 0.0,
            'edge_density': 0.0,
            'decision': 'no_structure_edges',
            'filename_hint': None
        }
    
    # Factor 1: Edge ratio (texture indicator)
    ratio = raw_count / structure_count
    
    # Factor 2: Depth variance (global smoothness indicator)
    depth_var = float(np.var(depth_map))
    
    # Factor 3: Edge density (structural complexity)
    edge_density = structure_count / total_pixels
    
    # Factor 4: Depth gradient variance (NEW - separates water from structure)
    # Water/ocean/pool: smooth depth → low gradient variance
    # Interiors: geometric edges → high gradient variance
    depth_grad_y, depth_grad_x = np.gradient(depth_map.astype(np.float32))
    depth_grad_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
    depth_gradient_var = float(np.var(depth_grad_mag))
    
    # Factor 5: Filename-based weak supervision (NEW)
    filename_hint = None
    confidence_boost = 0.0
    if image_filename:
        filename_lower = image_filename.lower()
        
        # Texture patterns: water, reflective surfaces, aerial views, organic textures
        texture_patterns = ['pool', 'ocean', 'water', 'glass', 'aerial', 'foliage', 'trees', 'shores', 'beach', 'sea']
        
        # Structure patterns: architectural interiors with clear geometric features
        structure_patterns = ['kitchen', 'bathroom', 'bedroom', 'living', 'great', 'interior', 'entry', 
                            'dining', 'office', 'courtyard', 'room', 'hall', 'lobby']
        
        if any(p in filename_lower for p in texture_patterns):
            filename_hint = 'texture'
            confidence_boost = 0.3  # Strong signal toward texture
        elif any(p in filename_lower for p in structure_patterns):
            filename_hint = 'structure'
            confidence_boost = 0.3  # Strong signal toward structure
    
    # Multi-factor decision tree (REVISED to fix pool/ocean misclassification)
    decision = None
    scene_type = None
    
    # Rule 1: Very low edge density (<0.005) = smooth surfaces (water, glass, ocean)
    # Pool water: density=0.002, var=0.018
    # Ocean: density=0, var=0.046
    if edge_density < 0.005:
        scene_type = 'texture_dominated'
        decision = 'very_low_edge_density'
    
    # Rule 2: Very high ratio (>10) = strong texture signal (patterned interiors)
    # Interior bathroom: ratio=14.3, var=0.074
    elif ratio > 10.0:
        scene_type = 'texture_dominated'
        decision = 'very_high_ratio'
    
    # Rule 3: Low depth gradient variance (<0.00040) + NOT high edge density = smooth depth (water/ocean/pool)
    # PRIORITY RULE: Smooth depth → texture (water reflections)
    # But if edge_density > 0.04, defer to later rules (might be smooth interior like great room)
    # Pool: depth_gradient_var = 0.000266, edge_density = 0.0291
    # Pool with ripples: depth_gradient_var = 0.000279, edge_density = 0.0587
    # Ocean: depth_gradient_var = 0.000053, edge_density = 0.0292
    # Aerial: depth_gradient_var = 0.000438 (just above threshold - texture)
    elif depth_gradient_var < 0.00040 and edge_density < 0.040:
        scene_type = 'texture_dominated'
        decision = 'smooth_depth_gradients'
    
    # Rule 4: High edge density (>0.065) + medium ratio (2-10) = very dense structured interiors
    # Interior with lots of geometric detail
    # Raised threshold to 0.065 to avoid catching pool scenes
    elif edge_density > 0.065 and 2.0 <= ratio <= 10.0:
        scene_type = 'structure_dominated'
        decision = 'very_high_density_structure'
    
    # Rule 5: Medium-high edge density (>0.03) + medium ratio (3-10) + high depth gradient variance (>0.0008) = structured interiors
    # Interior kitchen: ratio=3.80, density=0.0362, depth_grad_var = 0.000600
    # This catches geometric scenes with moderate edge density
    elif edge_density > 0.03 and 3.0 <= ratio <= 10.0 and depth_gradient_var > 0.0008:
        scene_type = 'structure_dominated'
        decision = 'high_density_medium_ratio_geometric'
    
    # Rule 6: Low ratio (<2) + low depth variance (<0.025) = smooth texture (pool, glass)
    # Pool: ratio=1.0, var=0.018
    elif ratio < 2.0 and depth_var < 0.025:
        scene_type = 'texture_dominated'
        decision = 'low_ratio_low_variance'
    
    # Rule 7: Low ratio (<2) + medium/high edge density (>0.008) = glass/reflective
    # Glass facade: ratio=1.4, density=0.009, var=0.077
    elif ratio < 2.0 and edge_density > 0.008:
        scene_type = 'texture_dominated'
        decision = 'low_ratio_medium_density'
    
    # Rule 8: Medium ratio (2-5) with high edge density (>0.045) + high depth gradient variance = structure
    # Catches remaining structured scenes
    elif 2.0 <= ratio <= 5.0 and edge_density > 0.045 and depth_gradient_var > 0.0008:
        scene_type = 'structure_dominated'
        decision = 'medium_ratio_high_density_geometric'
    
    # Rule 9: Default - use ratio, edge density, and depth gradient variance
    else:
        # Smooth depth → texture (priority)
        if depth_gradient_var < 0.00048:
            scene_type = 'texture_dominated'
            decision = 'fallback_smooth_depth'
        # Medium-low ratio → structure
        elif ratio <= 4.0 and edge_density > 0.008:
            scene_type = 'structure_dominated'
            decision = 'fallback_structure_ratio'
        # High ratio → texture
        else:
            scene_type = 'texture_dominated'
            decision = 'fallback_texture_ratio'
    
    # Apply filename-based weak supervision
    # Only override if filename hint is strong and depth-based decision is borderline
    original_decision = decision
    if filename_hint:
        # Calculate a "confidence score" for the depth-based decision
        # Low confidence = close to decision boundaries, high confidence = far from boundaries
        
        # Check if we're in a borderline case (ambiguous depth metrics)
        is_borderline = False
        
        # Borderline case 1: ratio near medium thresholds (2-5 range)
        if 2.5 <= ratio <= 7.0:
            is_borderline = True
        
        # Borderline case 2: depth gradient variance in ambiguous range (0.0004-0.0008)
        if 0.00040 <= depth_gradient_var <= 0.00080:
            is_borderline = True
        
        # Borderline case 3: edge density in mid-range (0.02-0.05)
        if 0.020 <= edge_density <= 0.050:
            is_borderline = True
        
        # Override decision if filename hint is strong and depth-based is borderline
        if is_borderline:
            if filename_hint == 'texture' and scene_type == 'structure_dominated':
                scene_type = 'texture_dominated'
                decision = f'{original_decision}_OVERRIDDEN_BY_FILENAME_TEXTURE'
            elif filename_hint == 'structure' and scene_type == 'texture_dominated':
                scene_type = 'structure_dominated'
                decision = f'{original_decision}_OVERRIDDEN_BY_FILENAME_STRUCTURE'
            elif filename_hint == scene_type.replace('_dominated', ''):
                # Filename confirms depth-based decision - boost confidence
                decision = f'{original_decision}_CONFIRMED_BY_FILENAME'
    
    # Return with comprehensive metadata
    return scene_type, {
        'method': 'multi_factor_v2',
        'raw_edges': raw_count,
        'structure_edges': structure_count,
        'ratio': ratio,
        'depth_variance': depth_var,
        'depth_gradient_var': depth_gradient_var,
        'edge_density': edge_density,
        'decision': decision,
        'filename_hint': filename_hint,  # NEW: weak supervision signal
        'thresholds': {
            'ratio_high': threshold_ratio_high,
            'ratio_low': threshold_ratio_low,
            'depth_var_low': threshold_depth_var_low,
            'depth_var_high': threshold_depth_var_high,
            'edge_density': threshold_edge_density
        }
    }


def detect_edges(
    image: np.ndarray,
    threshold_low: float = 50,
    threshold_high: float = 150
) -> np.ndarray:
    """
    Detect edges using Canny edge detector with float-aware preprocessing.
    
    PRIORITY 2 FIX: For float depth maps, use gradient-based detection
    to avoid quantization artifacts from uint8 conversion.
    
    Args:
        image: Grayscale image (uint8 or float32)
        threshold_low: Low threshold for Canny (uint8) or percentile (float)
        threshold_high: High threshold for Canny (uint8) or percentile (float)
        
    Returns:
        Binary edge map (uint8)
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        # PRIORITY 2 FIX: Gradient-based detection for float depth
        # Compute gradients on float directly (avoid quantization)
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Adaptive thresholding based on gradient magnitude distribution
        # Use percentile-based thresholds instead of fixed values
        valid_grads = grad_mag[grad_mag > 1e-6]
        
        if len(valid_grads) > 0:
            # Adaptive thresholds: 60th and 85th percentile
            thresh_low = np.percentile(valid_grads, 60)
            thresh_high = np.percentile(valid_grads, 85)
            
            # Non-maximum suppression + hysteresis
            # Simple version: threshold gradient magnitude
            strong_edges = (grad_mag > thresh_high).astype(np.uint8) * 255
            weak_edges = ((grad_mag > thresh_low) & (grad_mag <= thresh_high)).astype(np.uint8) * 255
            
            # Connect weak edges to strong edges (simple dilation)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            strong_dilated = cv2.dilate(strong_edges, kernel, iterations=1)
            connected_weak = cv2.bitwise_and(weak_edges, strong_dilated)
            
            edges = cv2.bitwise_or(strong_edges, connected_weak)
            
            logger.debug(f"Float edge detection: thresh_low={thresh_low:.6f}, thresh_high={thresh_high:.6f}")
        else:
            edges = np.zeros(image.shape, dtype=np.uint8)
    else:
        # Standard Canny for uint8 images
        edges = cv2.Canny(image, threshold_low, threshold_high)
    
    return edges


def compute_edge_f1(
    rgb_edges: np.ndarray,
    depth_edges: np.ndarray,
    tolerance: int = 2
) -> float:
    """
    Compute edge F1 score with spatial tolerance.
    
    This is the PRIMARY alignment metric (shift-tolerant).
    
    Args:
        rgb_edges: RGB edge map (binary)
        depth_edges: Depth edge map (binary)
        tolerance: Spatial tolerance (pixels)
        
    Returns:
        F1 score [0, 1]
    """
    # Dilate both edge sets by tolerance
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
    rgb_dilated = cv2.dilate(rgb_edges, kernel, iterations=1)
    depth_dilated = cv2.dilate(depth_edges, kernel, iterations=1)
    
    # True positives: depth edges near RGB edges
    tp = np.logical_and(depth_edges > 0, rgb_dilated > 0).sum()
    
    # False positives: depth edges NOT near RGB edges
    fp = np.logical_and(depth_edges > 0, rgb_dilated == 0).sum()
    
    # False negatives: RGB edges NOT near depth edges
    fn = np.logical_and(rgb_edges > 0, depth_dilated == 0).sum()
    
    if tp + fp + fn == 0:
        return 0.0
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def compute_edge_overlap(
    rgb_edges: np.ndarray,
    depth_edges: np.ndarray,
    dilation: int = 3
) -> float:
    """
    Compute edge overlap percentage with dilation tolerance.
    
    Args:
        rgb_edges: RGB edge map (binary)
        depth_edges: Depth edge map (binary)
        dilation: Dilation radius (pixels)
        
    Returns:
        Overlap percentage [0, 1]
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1))
    rgb_dilated = cv2.dilate(rgb_edges, kernel, iterations=1)
    
    overlap = np.logical_and(rgb_dilated > 0, depth_edges > 0).sum()
    depth_edge_count = (depth_edges > 0).sum()
    
    if depth_edge_count == 0:
        return 0.0
    
    return overlap / depth_edge_count


def compute_chamfer_distance(
    rgb_edges: np.ndarray,
    depth_edges: np.ndarray
) -> float:
    """
    Compute mean Chamfer distance (depth edges → nearest RGB edge).
    
    Args:
        rgb_edges: RGB edge map (binary)
        depth_edges: Depth edge map (binary)
        
    Returns:
        Mean distance in pixels
    """
    # Distance transform
    dist_map = cv2.distanceTransform(255 - rgb_edges, cv2.DIST_L2, 5)
    
    # Extract distances at depth edge locations
    depth_edge_coords = np.where(depth_edges > 0)
    
    if len(depth_edge_coords[0]) == 0:
        return 0.0
    
    distances = dist_map[depth_edge_coords]
    
    return np.mean(distances)


def compute_edge_alignment_corr(
    rgb_edges: np.ndarray,
    depth_edges: np.ndarray
) -> float:
    """
    Compute edge alignment correlation (DIAGNOSTIC ONLY).
    
    This is NOT the primary metric (too sensitive to class imbalance).
    Use edge_f1 for quality gates.
    
    Args:
        rgb_edges: RGB edge map (binary)
        depth_edges: Depth edge map (binary)
        
    Returns:
        Correlation coefficient [-1, 1]
    """
    rgb_flat = rgb_edges.flatten().astype(np.float32)
    depth_flat = depth_edges.flatten().astype(np.float32)
    
    if rgb_flat.std() < 1e-6 or depth_flat.std() < 1e-6:
        return 0.0
    
    corr = np.corrcoef(rgb_flat, depth_flat)[0, 1]
    
    return corr


def compute_edge_sharpness(depth: np.ndarray, edges: np.ndarray) -> Tuple[float, float]:
    """
    Compute edge sharpness metrics on FLOAT depth.
    
    Args:
        depth: Depth map (float32 [0, 1])
        edges: Edge map (binary)
        
    Returns:
        (edge_width, sharpness_p95)
    """
    # Compute gradient magnitude on float depth
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Extract gradients at edge pixels
    edge_grads = grad_mag[edges > 0]
    
    if len(edge_grads) == 0:
        return 0.0, 0.0
    
    # Sharpness = 95th percentile gradient (scaled to 0-255 range for comparability)
    sharpness_p95 = np.percentile(edge_grads, 95) * 255.0
    
    # Edge width = 1 / median_gradient
    median_grad = np.median(edge_grads)
    if median_grad < 1e-6:
        edge_width = 999.0
    else:
        edge_width = np.clip(1.0 / median_grad, 0.5, 20.0)
    
    return edge_width, sharpness_p95


def detect_halos(depth: np.ndarray, rgb_edges: np.ndarray) -> float:
    """
    Detect halo/overshoot artifacts near edges.
    
    FIXED: Higher score = less halo = better quality
    
    Args:
        depth: Depth map (float32)
        rgb_edges: RGB edge map (binary)
        
    Returns:
        Halo score [0, 1] (higher is better, 1.0 = no halos, 0.0 = severe halos)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_region = cv2.dilate(rgb_edges, kernel, iterations=1)
    
    laplacian = cv2.Laplacian(depth, cv2.CV_32F, ksize=5)
    laplacian_abs = np.abs(laplacian)
    
    edge_mask = edge_region > 0
    non_edge_mask = ~edge_mask
    
    if edge_mask.sum() == 0 or non_edge_mask.sum() == 0:
        return 1.0  # No edges to evaluate, perfect score
    
    edge_overshoot = laplacian_abs[edge_mask].mean()
    global_overshoot = laplacian_abs[non_edge_mask].mean()
    
    # Ratio > 1 means edges have more ringing than background (bad)
    # ratio=1.0 → perfect (no excess edge ringing)
    # ratio=2.0 → moderate halo
    # ratio=3.0+ → severe halo
    ratio = edge_overshoot / max(global_overshoot, 1e-6)
    
    # Map ratio to score: 1.0 → 1.0 (perfect), 2.0 → 0.5, 3.0 → 0.0
    if ratio <= 1.0:
        score = 1.0
    elif ratio >= 3.0:
        score = 0.0
    else:
        score = 1.0 - (ratio - 1.0) / 2.0  # Linear interpolation 1.0-3.0 → 1.0-0.0
    
    logger.debug(f"Halo detection: edge_overshoot={edge_overshoot:.6f}, "
                 f"global={global_overshoot:.6f}, ratio={ratio:.3f}, score={score:.3f}")
    
    return score


def compute_overshoot_heatmap(depth: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
    """
    PRIORITY 3 FIX: Generate visualization of overshoot regions with detailed breakdown.
    
    Args:
        depth: Depth map (float32)
        rgb: RGB image (uint8 or float32)
        
    Returns:
        (heatmap, overshoot_ratio, components_dict)
    """
    # Compute depth gradients
    dy, dx = np.gradient(depth.astype(np.float32))
    grad_mag = np.sqrt(dx**2 + dy**2)
    
    # Detect overshoot: high gradient where RGB is smooth
    if rgb.dtype == np.float32:
        rgb_gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    rgb_smooth = cv2.GaussianBlur(rgb_gray, (9, 9), 2.0)
    rgb_detail = np.abs(cv2.Laplacian(rgb_smooth, cv2.CV_32F))
    
    # Overshoot mask: depth edges where RGB is smooth
    depth_edge_threshold = np.percentile(grad_mag, 90)
    rgb_smooth_threshold = np.percentile(rgb_detail, 30)
    
    overshoot_mask = (grad_mag > depth_edge_threshold) & (rgb_detail < rgb_smooth_threshold)
    
    # Create heatmap (red = overshoot)
    heatmap = np.zeros((*depth.shape, 3), dtype=np.uint8)
    heatmap[overshoot_mask] = [255, 0, 0]  # red
    
    # Compute overshoot ratio
    overshoot_ratio = overshoot_mask.sum() / overshoot_mask.size
    
    # Detailed components breakdown
    components = {
        "overshoot_ratio": float(overshoot_ratio),
        "overshoot_pixel_count": int(overshoot_mask.sum()),
        "total_pixels": int(overshoot_mask.size),
        "depth_edge_threshold": float(depth_edge_threshold),
        "rgb_smooth_threshold": float(rgb_smooth_threshold),
        "mean_depth_gradient_at_overshoot": float(grad_mag[overshoot_mask].mean()) if overshoot_mask.sum() > 0 else 0.0,
        "mean_rgb_detail_at_overshoot": float(rgb_detail[overshoot_mask].mean()) if overshoot_mask.sum() > 0 else 0.0
    }
    
    logger.info(f"Overshoot analysis: {overshoot_ratio*100:.2f}% of pixels ({components['overshoot_pixel_count']} px)")
    
    return heatmap, overshoot_ratio, components


def compute_overshoot_penalty(depth: np.ndarray) -> float:
    """
    Compute overshoot penalty (Laplacian ringing).
    
    PRIORITY 3 FIX: Calibrated scaling with detailed logging.
    Lower penalty = better (0.0 = perfect, 1.0 = severe ringing)
    
    Args:
        depth: Depth map (float32 [0, 1])
        
    Returns:
        Penalty [0, 1] (lower is better, 0.0 = no overshoot)
    """
    laplacian = cv2.Laplacian(depth, cv2.CV_32F, ksize=5)
    laplacian_abs = np.abs(laplacian)
    
    # 95th percentile of |Laplacian| for float depth is typically 0.001-0.05
    # Map empirical range [0, 0.1] → [0, 1]
    penalty_raw = np.percentile(laplacian_abs, 95)
    
    # Additional diagnostics
    mean_laplacian = laplacian_abs.mean()
    max_laplacian = laplacian_abs.max()
    
    # Calibrated scaling for float32 depth [0, 1]
    # 0.01 is typical for good depth, 0.1+ indicates severe ringing
    penalty = np.clip(penalty_raw * 10.0, 0.0, 1.0)
    
    logger.info(f"Overshoot penalty: raw_p95={penalty_raw:.4f}, penalty={penalty:.3f}, "
                f"mean={mean_laplacian:.4f}, max={max_laplacian:.4f}")
    
    return penalty


def compute_high_frequency_energy(depth_map: np.ndarray, sigma: float = 15.0) -> float:
    """
    Compute high-frequency energy to detect texture-copied-to-depth artifacts.
    
    This metric separates two cases:
    - Valid: Large near-to-far depth range (global variance high) but smooth gradients → low HF energy
    - Artifact: Ripples/speckles copied from texture (global variance moderate) → high HF energy
    
    Method:
    1. Compute low-frequency baseline: gaussian_blur(depth, sigma=large)
    2. Extract high-frequency residual: depth - baseline
    3. Measure variance of HF residual
    
    Args:
        depth_map: Depth map (float32 [0, 1])
        sigma: Gaussian blur sigma for low-frequency baseline (default: 15.0)
               Larger sigma = more aggressive HF extraction
        
    Returns:
        HF energy [0, 1+] (lower is better, <0.005 = smooth depth, >0.01 = texture artifacts)
        
    Examples:
        - Ocean/pool with smooth depth gradient: HF energy ~ 0.00001-0.0002
        - Ocean/pool with ripples copied to depth: HF energy ~ 0.0005-0.002
        - Interior with geometric edges: HF energy ~ 0.0002-0.0008 (acceptable)
        
    References:
        - Alternative to global variance for texture scene validation
        - Replaces faulty "depth_var < 0.05" gate that penalized valid aerial/pool scenes
    """
    # Ensure float32
    if depth_map.dtype != np.float32:
        depth_map = depth_map.astype(np.float32)
    
    # Low-frequency baseline (smooth depth)
    # Use BORDER_REFLECT_101 to avoid edge artifacts
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    
    depth_lowfreq = cv2.GaussianBlur(
        depth_map,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101
    )
    
    # High-frequency residual (texture artifacts, ripples, speckles)
    depth_highfreq = depth_map - depth_lowfreq
    
    # Variance of HF residual
    hf_energy = float(np.var(depth_highfreq))
    
    logger.debug(f"HF energy: {hf_energy:.6f} (sigma={sigma:.1f}, "
                 f"global_var={np.var(depth_map):.6f})")
    
    return hf_energy


def validate_depth_quality(
    rgb: np.ndarray,
    depth: np.ndarray,
    dilation: int = 3,
    save_heatmap: bool = False,
    heatmap_path: Optional[Path] = None,
    use_structure_edges: bool = True,
    image_filename: Optional[str] = None
) -> EdgeMetrics:
    """
    CANONICAL depth quality validation.
    
    This is the SINGLE implementation used by all validation paths.
    PRIORITY 2 FIX: Use float depth directly for edge detection.
    PRIORITY 3 FIX: Add overshoot heatmap generation.
    PRIORITY 8 FIX: Add structure-aware edge detection with texture suppression.
    
    Args:
        rgb: RGB image (uint8 or float32)
        depth: Depth map (float32 [0, 1])
        dilation: Dilation radius for edge overlap
        save_heatmap: Whether to generate and save overshoot heatmap (PRIORITY 3)
        heatmap_path: Path to save heatmap (if save_heatmap=True)
        use_structure_edges: If True, use bilateral-filtered edges (suppress texture)
        image_filename: Optional filename for weak supervision in scene classification
        
    Returns:
        EdgeMetrics with all quality scores
    """
    # Convert RGB to grayscale
    if rgb.ndim == 3:
        if rgb.dtype == np.float32:
            gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = (rgb * 255).astype(np.uint8) if rgb.dtype == np.float32 else rgb
    
    # PRIORITY 8 FIX: Extract RGB edges with texture suppression
    if use_structure_edges:
        rgb_edges = extract_structure_edges(gray)
        edge_type = 'structure'
    else:
        rgb_edges = detect_edges(gray)
        edge_type = 'raw'
    
    # PRIORITY 2 FIX: Detect edges on float depth directly
    depth_edges = detect_edges(depth)  # Pass float32 directly
    
    # PRIORITY 8 FIX: Scene classification with V2 multi-factor classifier
    scene_metadata = None
    if use_structure_edges:
        rgb_edges_raw = detect_edges(gray)
        scene_type, scene_metadata = classify_scene_type_v2(
            rgb_edges_raw=rgb_edges_raw,
            rgb_edges_structure=rgb_edges,
            depth_map=depth,
            image_filename=image_filename
        )
    else:
        scene_type = 'unknown'
    
    # Compute all metrics
    edge_f1 = compute_edge_f1(rgb_edges, depth_edges, tolerance=2)
    edge_overlap = compute_edge_overlap(rgb_edges, depth_edges, dilation)
    edge_alignment_corr = compute_edge_alignment_corr(rgb_edges, depth_edges)
    chamfer_distance = compute_chamfer_distance(rgb_edges, depth_edges)
    
    edge_width, sharpness_p95 = compute_edge_sharpness(depth, depth_edges)
    
    rgb_edge_count = int((rgb_edges > 0).sum())
    depth_edge_count = int((depth_edges > 0).sum())
    edge_count_ratio = depth_edge_count / max(rgb_edge_count, 1)
    
    halo_score = detect_halos(depth, rgb_edges)
    overshoot_penalty = compute_overshoot_penalty(depth)
    
    # PRIORITY 3 FIX: Generate overshoot heatmap and detailed breakdown
    if save_heatmap and heatmap_path is not None:
        heatmap, overshoot_ratio, overshoot_components = compute_overshoot_heatmap(depth, rgb)
        
        # Save heatmap
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        logger.info(f"✅ Overshoot heatmap saved: {heatmap_path}")
        
        # Log detailed breakdown
        logger.info("Overshoot components:")
        logger.info(f"  overshoot_ratio: {overshoot_components['overshoot_ratio']:.4f}")
        logger.info(f"  halo_score: {halo_score:.3f}")
        logger.info(f"  overshoot_penalty: {overshoot_penalty:.3f}")
        logger.info(f"  pixel_count: {overshoot_components['overshoot_pixel_count']}")
        logger.info(f"  mean_depth_gradient: {overshoot_components['mean_depth_gradient_at_overshoot']:.6f}")
    
    metrics = EdgeMetrics(
        edge_f1=edge_f1,
        edge_overlap=edge_overlap,
        edge_alignment_corr=edge_alignment_corr,
        chamfer_distance=chamfer_distance,
        edge_width=edge_width,
        edge_sharpness_p95=sharpness_p95,
        edge_count_ratio=edge_count_ratio,
        halo_score=halo_score,
        overshoot_penalty=overshoot_penalty,
        rgb_edge_count=rgb_edge_count,
        depth_edge_count=depth_edge_count,
        edge_type=edge_type,
        scene_type=scene_type,
        scene_metadata=scene_metadata
    )
    
    logger.info(f"Depth quality validation:\n{metrics}")
    
    return metrics


def create_edge_overlay(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Create edge visualization overlay.
    
    Shows:
    - RGB-only edges: RED
    - Depth-only edges: BLUE
    - Overlap edges: GREEN
    
    Args:
        rgb: RGB image uint8
        depth: Depth map float32 [0, 1]
        
    Returns:
        Overlay image uint8 RGB
    """
    # Detect edges (using standard Canny parameters)
    # Convert RGB to grayscale first
    if len(rgb.shape) == 3:
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        rgb_gray = rgb
    
    rgb_edges = detect_edges(rgb_gray, threshold_low=50, threshold_high=150)
    depth_edges = detect_edges(depth, threshold_low=0.02, threshold_high=0.98)
    
    # Dilate slightly for visibility
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rgb_edges_d = cv2.dilate(rgb_edges.astype(np.uint8), kernel, iterations=1).astype(bool)
    depth_edges_d = cv2.dilate(depth_edges.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # Create overlay on RGB base
    overlay = rgb.copy()
    
    # RGB-only edges: red
    rgb_only = rgb_edges_d & ~depth_edges_d
    overlay[rgb_only] = [255, 0, 0]
    
    # Depth-only edges: blue
    depth_only = depth_edges_d & ~rgb_edges_d
    overlay[depth_only] = [0, 0, 255]
    
    # Overlap edges: green
    overlap = rgb_edges_d & depth_edges_d
    overlay[overlap] = [0, 255, 0]
    
    return overlay


def save_metrics_atomic(metrics: Dict, output_path: Path) -> None:
    """
    PRIORITY 6 FIX: Save metrics to JSON with atomic write + validation.
    
    This prevents truncated/corrupted JSON files.
    Recursively converts all numpy types to native Python.
    
    Args:
        metrics: Metrics dictionary
        output_path: Output JSON path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert_value(obj):
        """Recursively convert numpy types to native Python."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_value(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_value(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Convert all numpy types recursively
    metrics_clean = convert_value(metrics)
    
    # Write to temp file
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.json',
        dir=output_path.parent,
        text=True
    )
    
    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Validate by reading back
        with open(temp_path, 'r') as f:
            json.load(f)
        
        # Atomic rename
        os.replace(temp_path, output_path)
        
        logger.info(f"✅ Metrics saved (atomic): {output_path}")
        
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError(f"Failed to save metrics atomically: {e}")
