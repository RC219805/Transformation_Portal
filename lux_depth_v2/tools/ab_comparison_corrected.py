#!/usr/bin/env python3
"""
Corrected A/B Comparison with Production Refinement
====================================================

Fixes identified implementation errors:
1. Edge metrics computed on float32 (not uint8)
2. Production refinement pipeline applied (CLAHE + guided filter + edge snap)
3. Comprehensive metrics (not just headline numbers)

Reference: User feedback 2025-12-18
"Fix the metric bugs and apply the refinement pipeline properly"
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class DepthMetrics:
    """Comprehensive depth quality metrics."""
    
    # Edge metrics (CORRECTED - computed on float32)
    edge_gradient_mean: float
    edge_gradient_p95: float
    edge_gradient_p99: float
    edge_alignment_rgb: float
    
    # Depth statistics
    unique_levels: int
    effective_bits: float
    flat_ratio: float
    percentile_range: int
    
    # Processing
    time_ms: float
    
    def to_dict(self) -> dict:
        return asdict(self)


def compute_corrected_edge_metrics(depth: np.ndarray, rgb: Optional[np.ndarray] = None) -> dict:
    """
    Compute edge metrics CORRECTLY on float32.
    
    Fixes the "0.09" anomaly caused by uint8 quantization.
    """
    # Convert to float32 [0, 1]
    if depth.dtype == np.uint16:
        depth_norm = depth.astype(np.float32) / 65535.0
    elif depth.dtype == np.uint8:
        depth_norm = depth.astype(np.float32) / 255.0
    else:
        depth_norm = depth.astype(np.float32)
        if depth_norm.max() > 1.0:
            depth_norm = depth_norm / depth_norm.max()
    
    # Sobel gradients on FLOAT (critical)
    sobel_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Scale to "0-255 equivalent" for interpretability
    gradient_scaled = gradient_mag * 255.0
    
    # Compute statistics
    metrics = {
        'gradient_mean': float(gradient_scaled.mean()),
        'gradient_p95': float(np.percentile(gradient_scaled, 95)),
        'gradient_p99': float(np.percentile(gradient_scaled, 99)),
        'gradient_max': float(gradient_scaled.max())
    }
    
    # Edge alignment with RGB (if provided)
    if rgb is not None and CV2_AVAILABLE:
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
    else:
        metrics['edge_alignment'] = 0.0
    
    return metrics


def compute_depth_statistics(depth: np.ndarray) -> dict:
    """
    Compute comprehensive depth statistics.
    
    Not just "65536 unique levels" - also flat ratio, effective bits, gradients.
    """
    # Convert to uint16
    if depth.dtype == np.float32:
        depth_uint16 = (depth * 65535).astype(np.uint16)
    else:
        depth_uint16 = depth.astype(np.uint16)
    
    # Unique levels
    unique_levels = len(np.unique(depth_uint16))
    effective_bits = np.log2(max(unique_levels, 1))
    
    # Flat ratio (low gradient regions)
    gradient = np.gradient(depth_uint16.astype(np.float32))
    gradient_mag = np.sqrt(gradient[0]**2 + gradient[1]**2)
    flat_pixels = (gradient_mag < 1.0).sum()
    flat_ratio = flat_pixels / gradient_mag.size
    
    # Percentile range
    p1 = np.percentile(depth_uint16, 1)
    p99 = np.percentile(depth_uint16, 99)
    
    return {
        'unique_levels': unique_levels,
        'effective_bits': effective_bits,
        'flat_ratio': flat_ratio,
        'percentile_range': int(p99 - p1)
    }


def run_baseline(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Baseline: HF pipeline only (will resize to 518px internally).
    
    NO refinement applied.
    """
    try:
        from transformers import pipeline
    except ImportError:
        logger.error("transformers required")
        return None, 0.0
    
    logger.info("Baseline: HF pipeline (518px internal resize, no refinement)")
    
    pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=-1
    )
    
    # Convert to PIL
    if rgb.dtype == np.float32:
        rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        rgb_pil = Image.fromarray(rgb)
    
    # Inference
    start = time.time()
    result = pipe(rgb_pil)
    elapsed_ms = (time.time() - start) * 1000
    
    # Extract and normalize
    if isinstance(result, dict):
        depth = np.array(result.get('predicted_depth', result.get('depth')))
    else:
        depth = np.array(result.depth if hasattr(result, 'depth') else result)
    
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    
    logger.info(f"✓ Baseline: {elapsed_ms:.0f}ms, shape={depth.shape}")
    return depth, elapsed_ms


def run_production_refined(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Production: HF pipeline + FULL refinement (CLAHE + guided filter + edge snap).
    
    This is the CORRECT implementation with all fixes applied.
    """
    from lux_depth_v2.depth_refinement import refine_depth_production
    
    logger.info("Production: HF pipeline + CLAHE + guided filter + edge snap")
    
    # Step 1: Get baseline depth
    depth_baseline, time_baseline = run_baseline(rgb)
    
    # Step 2: Apply production refinement
    start = time.time()
    depth_refined = refine_depth_production(
        depth_baseline,
        rgb=rgb,
        use_clahe=True,
        use_edge_filter=True,
        use_edge_snap=True
    )
    time_refinement = (time.time() - start) * 1000
    
    total_time = time_baseline + time_refinement
    
    logger.info(f"✓ Production: {total_time:.0f}ms ({time_baseline:.0f}ms inference + {time_refinement:.0f}ms refinement)")
    return depth_refined, total_time


def compare_pipelines(
    rgb: np.ndarray,
    output_dir: Path
) -> dict:
    """
    Run A/B comparison: baseline vs production-refined.
    
    Returns comprehensive metrics and saves visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("A/B Comparison: Baseline vs Production-Refined")
    logger.info("=" * 60)
    
    # Run baseline
    depth_baseline, time_baseline = run_baseline(rgb)
    
    # Run production
    depth_production, time_production = run_production_refined(rgb)
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    
    # Baseline metrics
    baseline_edge = compute_corrected_edge_metrics(depth_baseline, rgb)
    baseline_stats = compute_depth_statistics(depth_baseline)
    
    baseline_metrics = DepthMetrics(
        edge_gradient_mean=baseline_edge['gradient_mean'],
        edge_gradient_p95=baseline_edge['gradient_p95'],
        edge_gradient_p99=baseline_edge['gradient_p99'],
        edge_alignment_rgb=baseline_edge['edge_alignment'],
        unique_levels=baseline_stats['unique_levels'],
        effective_bits=baseline_stats['effective_bits'],
        flat_ratio=baseline_stats['flat_ratio'],
        percentile_range=baseline_stats['percentile_range'],
        time_ms=time_baseline
    )
    
    # Production metrics
    production_edge = compute_corrected_edge_metrics(depth_production, rgb)
    production_stats = compute_depth_statistics(depth_production)
    
    production_metrics = DepthMetrics(
        edge_gradient_mean=production_edge['gradient_mean'],
        edge_gradient_p95=production_edge['gradient_p95'],
        edge_gradient_p99=production_edge['gradient_p99'],
        edge_alignment_rgb=production_edge['edge_alignment'],
        unique_levels=production_stats['unique_levels'],
        effective_bits=production_stats['effective_bits'],
        flat_ratio=production_stats['flat_ratio'],
        percentile_range=production_stats['percentile_range'],
        time_ms=time_production
    )
    
    # Compute improvements
    improvements = {
        'edge_gradient_mean': (production_metrics.edge_gradient_mean / max(baseline_metrics.edge_gradient_mean, 0.01) - 1) * 100,
        'edge_gradient_p95': (production_metrics.edge_gradient_p95 / max(baseline_metrics.edge_gradient_p95, 0.01) - 1) * 100,
        'edge_alignment': (production_metrics.edge_alignment_rgb - baseline_metrics.edge_alignment_rgb) * 100,
        'unique_levels': (production_metrics.unique_levels / max(baseline_metrics.unique_levels, 1) - 1) * 100,
        'time_overhead': (production_metrics.time_ms / max(baseline_metrics.time_ms, 1) - 1) * 100
    }
    
    # Save visualizations
    logger.info("Saving visualizations...")
    
    # Save depths as images
    depth_baseline_vis = (depth_baseline * 255).astype(np.uint8)
    depth_production_vis = (depth_production * 255).astype(np.uint8)
    
    Image.fromarray(depth_baseline_vis).save(output_dir / "baseline_depth.png")
    Image.fromarray(depth_production_vis).save(output_dir / "production_depth.png")
    
    # Side-by-side comparison
    if CV2_AVAILABLE:
        comparison = np.hstack([depth_baseline_vis, depth_production_vis])
        Image.fromarray(comparison).save(output_dir / "comparison.png")
    
    # Save report
    report = {
        'baseline': baseline_metrics.to_dict(),
        'production': production_metrics.to_dict(),
        'improvements': improvements
    }
    
    with open(output_dir / "comparison_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nBaseline (HF only):")
    logger.info(f"  Edge gradient (mean): {baseline_metrics.edge_gradient_mean:.2f}")
    logger.info(f"  Edge gradient (p95):  {baseline_metrics.edge_gradient_p95:.2f}")
    logger.info(f"  Edge alignment:       {baseline_metrics.edge_alignment_rgb:.3f}")
    logger.info(f"  Unique levels:        {baseline_metrics.unique_levels:,}")
    logger.info(f"  Time:                 {baseline_metrics.time_ms:.0f}ms")
    
    logger.info(f"\nProduction (HF + CLAHE + guided filter + edge snap):")
    logger.info(f"  Edge gradient (mean): {production_metrics.edge_gradient_mean:.2f} ({improvements['edge_gradient_mean']:+.1f}%)")
    logger.info(f"  Edge gradient (p95):  {production_metrics.edge_gradient_p95:.2f} ({improvements['edge_gradient_p95']:+.1f}%)")
    logger.info(f"  Edge alignment:       {production_metrics.edge_alignment_rgb:.3f} ({improvements['edge_alignment']:+.1f} pp)")
    logger.info(f"  Unique levels:        {production_metrics.unique_levels:,} ({improvements['unique_levels']:+.1f}%)")
    logger.info(f"  Time:                 {production_metrics.time_ms:.0f}ms ({improvements['time_overhead']:+.1f}%)")
    
    logger.info(f"\n✓ Results saved to {output_dir}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Corrected A/B comparison")
    parser.add_argument("--input", type=Path, required=True, help="Input RGB image")
    parser.add_argument("--output", type=Path, default=Path("outputs/ab_corrected"), help="Output directory")
    
    args = parser.parse_args()
    
    # Load image
    rgb = np.array(Image.open(args.input).convert("RGB"))
    
    # Run comparison
    report = compare_pipelines(rgb, args.output)
