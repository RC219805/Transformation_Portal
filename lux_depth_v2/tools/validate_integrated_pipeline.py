#!/usr/bin/env python3
"""
Full Integrated Pipeline Test: Tiled + Global + Production Refinement
======================================================================

Tests the complete high-fidelity depth pipeline with all fixes applied:
1. Tiled inference (bypass 518px resize)
2. Global anchor fusion (prevent tiling artifacts)
3. Production refinement (CLAHE + guided filter + edge snap)

Compares against baseline to measure actual improvements.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_baseline(rgb: np.ndarray):
    """Baseline: HF pipeline (518px resize, no refinement)."""
    from transformers import pipeline
    
    logger.info("=" * 60)
    logger.info("BASELINE: HF Pipeline (518px resize)")
    logger.info("=" * 60)
    
    pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=-1
    )
    
    if rgb.dtype == np.float32:
        rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        rgb_pil = Image.fromarray(rgb)
    
    start = time.time()
    result = pipe(rgb_pil)
    elapsed = time.time() - start
    
    # Extract depth
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
    
    logger.info(f"✓ Baseline complete: {elapsed:.1f}s, shape={depth.shape}")
    return depth, elapsed


def run_full_pipeline(rgb: np.ndarray):
    """Full pipeline: Tiled + Global + Production Refinement."""
    from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig
    
    logger.info("=" * 60)
    logger.info("FULL PIPELINE: Tiled + Global + Refinement")
    logger.info("=" * 60)
    
    # Configure with ALL enhancements
    config = TiledInferenceConfig(
        tile_size=1024,
        overlap=128,
        bypass_image_processor=True,  # Bypass 518px resize
        use_global_anchor=True,       # Prevent tiling artifacts
        use_edge_snapping=False,      # Disable old edge snapping (refinement has better one)
        use_production_refinement=True,  # ← NEW: Production refinement
        refinement_use_clahe=True,
        refinement_use_edge_filter=True,
        refinement_use_edge_snap=True
    )
    
    estimator = TiledDepthEstimator(config)
    
    start = time.time()
    depth = estimator.estimate_depth(rgb)
    elapsed = time.time() - start
    
    logger.info(f"✓ Full pipeline complete: {elapsed:.1f}s, shape={depth.shape}")
    return depth, elapsed


def compute_metrics(depth: np.ndarray, rgb: np.ndarray) -> dict:
    """Compute comprehensive quality metrics."""
    from lux_depth_v2.depth_refinement import compute_robust_edge_metrics, compute_depth_statistics
    
    edge_metrics = compute_robust_edge_metrics(depth, rgb)
    depth_stats = compute_depth_statistics(depth)
    
    return {
        **edge_metrics,
        **depth_stats
    }


def compare_pipelines(rgb: np.ndarray, output_dir: Path):
    """Run both pipelines and compare results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATED PIPELINE VALIDATION")
    logger.info("=" * 60)
    
    # Run baseline
    logger.info("\n[1/2] Running baseline...")
    depth_baseline, time_baseline = run_baseline(rgb)
    metrics_baseline = compute_metrics(depth_baseline, rgb)
    
    # Run full pipeline
    logger.info("\n[2/2] Running full pipeline...")
    depth_full, time_full = run_full_pipeline(rgb)
    metrics_full = compute_metrics(depth_full, rgb)
    
    # Compute improvements
    improvements = {
        'edge_gradient_mean': (metrics_full['gradient_mean'] / max(metrics_baseline['gradient_mean'], 0.01) - 1) * 100,
        'edge_gradient_p95': (metrics_full['gradient_p95'] / max(metrics_baseline['gradient_p95'], 0.01) - 1) * 100,
        'edge_gradient_p99': (metrics_full['gradient_p99'] / max(metrics_baseline['gradient_p99'], 0.01) - 1) * 100,
        'edge_alignment': (metrics_full.get('edge_alignment', 0) - metrics_baseline.get('edge_alignment', 0)) * 100,
        'unique_levels': (metrics_full['unique_levels'] / max(metrics_baseline['unique_levels'], 1) - 1) * 100,
        'flat_ratio': (metrics_full['flat_ratio'] - metrics_baseline['flat_ratio']) * 100,
        'time_overhead': (time_full / max(time_baseline, 0.1) - 1) * 100
    }
    
    # Save visualizations
    logger.info("\nSaving outputs...")
    
    depth_baseline_vis = (depth_baseline * 255).astype(np.uint8)
    depth_full_vis = (depth_full * 255).astype(np.uint8)
    
    Image.fromarray(depth_baseline_vis).save(output_dir / "baseline_depth.png")
    Image.fromarray(depth_full_vis).save(output_dir / "full_pipeline_depth.png")
    
    # Side-by-side comparison
    import cv2
    comparison = np.hstack([depth_baseline_vis, depth_full_vis])
    Image.fromarray(comparison).save(output_dir / "comparison.png")
    
    # Save report
    report = {
        'baseline': {
            **metrics_baseline,
            'time_seconds': time_baseline
        },
        'full_pipeline': {
            **metrics_full,
            'time_seconds': time_full
        },
        'improvements': improvements,
        'pipeline_config': {
            'tile_size': 1024,
            'overlap': 128,
            'bypass_image_processor': True,
            'use_global_anchor': True,
            'use_production_refinement': True,
            'refinement_stages': ['CLAHE', 'Guided Filter (bilateral fallback)', 'Edge Snap']
        }
    }
    
    with open(output_dir / "validation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\nBaseline (HF only):")
    logger.info(f"  Edge gradient (mean): {metrics_baseline['gradient_mean']:.2f}")
    logger.info(f"  Edge gradient (p95):  {metrics_baseline['gradient_p95']:.2f}")
    logger.info(f"  Edge gradient (p99):  {metrics_baseline['gradient_p99']:.2f}")
    logger.info(f"  Edge alignment:       {metrics_baseline.get('edge_alignment', 0):.3f}")
    logger.info(f"  Unique levels:        {metrics_baseline['unique_levels']:,}")
    logger.info(f"  Effective bits:       {metrics_baseline['effective_bits']:.2f}")
    logger.info(f"  Flat ratio:           {metrics_baseline['flat_ratio']:.3f}")
    logger.info(f"  Time:                 {time_baseline:.1f}s")
    
    logger.info(f"\nFull Pipeline (Tiled + Global + Refinement):")
    logger.info(f"  Edge gradient (mean): {metrics_full['gradient_mean']:.2f} ({improvements['edge_gradient_mean']:+.1f}%)")
    logger.info(f"  Edge gradient (p95):  {metrics_full['gradient_p95']:.2f} ({improvements['edge_gradient_p95']:+.1f}%)")
    logger.info(f"  Edge gradient (p99):  {metrics_full['gradient_p99']:.2f} ({improvements['edge_gradient_p99']:+.1f}%)")
    logger.info(f"  Edge alignment:       {metrics_full.get('edge_alignment', 0):.3f} ({improvements['edge_alignment']:+.1f} pp)")
    logger.info(f"  Unique levels:        {metrics_full['unique_levels']:,} ({improvements['unique_levels']:+.1f}%)")
    logger.info(f"  Effective bits:       {metrics_full['effective_bits']:.2f}")
    logger.info(f"  Flat ratio:           {metrics_full['flat_ratio']:.3f} ({improvements['flat_ratio']:+.1f} pp)")
    logger.info(f"  Time:                 {time_full:.1f}s ({improvements['time_overhead']:+.1f}%)")
    
    # Verdict
    logger.info("\n" + "=" * 60)
    edge_improved = improvements['edge_gradient_p95'] > 50
    time_acceptable = improvements['time_overhead'] < 500
    
    if edge_improved and time_acceptable:
        verdict = "✅ VALIDATION PASSED - Ready for production"
    elif edge_improved:
        verdict = "⚠️  CONDITIONAL PASS - Quality improved but slow"
    else:
        verdict = "❌ VALIDATION FAILED - Insufficient improvement"
    
    logger.info(f"VERDICT: {verdict}")
    logger.info("=" * 60)
    
    logger.info(f"\n✓ Full report saved to {output_dir}/validation_report.json")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate integrated high-fidelity pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Input RGB image")
    parser.add_argument("--output", type=Path, default=Path("outputs/integrated_validation"), help="Output directory")
    
    args = parser.parse_args()
    
    # Load image
    logger.info(f"Loading image: {args.input}")
    rgb = np.array(Image.open(args.input).convert("RGB"))
    logger.info(f"Image loaded: {rgb.shape}")
    
    # Run comparison
    report = compare_pipelines(rgb, args.output)
    
    logger.info("\n✓ Validation complete!")
