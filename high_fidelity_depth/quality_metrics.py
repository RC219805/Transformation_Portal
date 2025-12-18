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


def validate_depth_quality(
    rgb: np.ndarray,
    depth: np.ndarray,
    dilation: int = 3,
    save_heatmap: bool = False,
    heatmap_path: Optional[Path] = None
) -> EdgeMetrics:
    """
    CANONICAL depth quality validation.
    
    This is the SINGLE implementation used by all validation paths.
    PRIORITY 2 FIX: Use float depth directly for edge detection.
    PRIORITY 3 FIX: Add overshoot heatmap generation.
    
    Args:
        rgb: RGB image (uint8 or float32)
        depth: Depth map (float32 [0, 1])
        dilation: Dilation radius for edge overlap
        save_heatmap: Whether to generate and save overshoot heatmap (PRIORITY 3)
        heatmap_path: Path to save heatmap (if save_heatmap=True)
        
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
    
    # PRIORITY 2 FIX: Detect edges on float depth directly
    rgb_edges = detect_edges(gray)
    depth_edges = detect_edges(depth)  # Pass float32 directly
    
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
        depth_edge_count=depth_edge_count
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
    # Detect edges
    rgb_edges = detect_edges(rgb, mode='canny', threshold1=50, threshold2=150)
    depth_edges = detect_edges(depth, mode='sobel', low_threshold=0.02, high_threshold=0.98)
    
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
