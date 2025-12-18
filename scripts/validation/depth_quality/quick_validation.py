#!/usr/bin/env python3
"""
Quick Validation Test for Depth Pipeline Fixes
==============================================

Fast test with smaller image to validate fixes quickly.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator, DepthConfig
from high_fidelity_depth.quality_metrics import validate_depth_quality
from high_fidelity_depth.refinement import edge_snap_refinement


def main():
    """Quick validation test."""
    
    # Load and resize to small size for speed
    test_image_path = Path("/Users/rc/Transformation_Portal/input_images/750_Picacho/Pool.tif")
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        sys.exit(1)
    
    logger.info(f"Loading: {test_image_path.name}")
    rgb_pil = Image.open(test_image_path)
    
    # Resize to 1024px for quick test
    max_size = 1024
    scale = max_size / max(rgb_pil.size)
    new_size = (int(rgb_pil.width * scale), int(rgb_pil.height * scale))
    rgb_pil = rgb_pil.resize(new_size, Image.LANCZOS)
    
    rgb = np.array(rgb_pil)
    logger.info(f"Test image: {rgb.shape}")
    
    # Test 1: Baseline (single pass)
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Baseline (Single Pass)")
    logger.info("="*60)
    
    config = DepthConfig(tile_size=9999, overlap=0, reconcile_scales=False)
    estimator = HighFidelityDepthEstimator(config)
    depth_baseline = estimator.estimate_depth(rgb, use_global_anchor=False)
    
    metrics_baseline = validate_depth_quality(rgb, depth_baseline)
    logger.info(f"✓ Edge F1: {metrics_baseline.edge_f1:.3f}")
    logger.info(f"✓ Quality: {metrics_baseline.quality_score():.3f}")
    
    # Test 2: Tiling with robust reconciliation
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Tiling + Robust Reconciliation")
    logger.info("="*60)
    
    config = DepthConfig(
        tile_size=512,  # Smaller tiles for faster test
        overlap=64,
        reconcile_scales=True,
        reconcile_method="robust",
        fusion_mode="weighted"
    )
    estimator = HighFidelityDepthEstimator(config)
    depth_tiled = estimator.estimate_depth(rgb, use_global_anchor=True)
    
    metrics_tiled = validate_depth_quality(rgb, depth_tiled)
    logger.info(f"✓ Edge F1: {metrics_tiled.edge_f1:.3f} (Δ{metrics_tiled.edge_f1 - metrics_baseline.edge_f1:+.3f})")
    logger.info(f"✓ Quality: {metrics_tiled.quality_score():.3f} (Δ{metrics_tiled.quality_score() - metrics_baseline.quality_score():+.3f})")
    
    # Test 3: Edge snapping refinement
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Edge Snapping Refinement")
    logger.info("="*60)
    
    depth_refined = edge_snap_refinement(depth_tiled, rgb, strength=0.2)
    
    metrics_refined = validate_depth_quality(rgb, depth_refined)
    logger.info(f"✓ Edge F1: {metrics_refined.edge_f1:.3f} (Δ{metrics_refined.edge_f1 - metrics_tiled.edge_f1:+.3f})")
    logger.info(f"✓ Sharpness: {metrics_refined.edge_sharpness_p95:.1f} (Δ{metrics_refined.edge_sharpness_p95 - metrics_tiled.edge_sharpness_p95:+.1f})")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Baseline Edge F1:    {metrics_baseline.edge_f1:.3f}")
    logger.info(f"Tiled Edge F1:       {metrics_tiled.edge_f1:.3f}")
    logger.info(f"Refined Edge F1:     {metrics_refined.edge_f1:.3f}")
    
    # Validation criteria
    passed_all = True
    
    if metrics_baseline.edge_f1 >= 0.30:
        logger.info("✅ Baseline Edge F1 ≥ 0.30 (float edge detection working)")
    else:
        logger.error("❌ Baseline Edge F1 < 0.30")
        passed_all = False
    
    if metrics_tiled.edge_count_ratio <= 2.0:
        logger.info("✅ Edge count ratio ≤ 2.0 (no artifact explosion)")
    else:
        logger.error("❌ Edge count ratio > 2.0")
        passed_all = False
    
    if metrics_tiled.chamfer_distance < 15.0:
        logger.info("✅ Chamfer distance < 15px (good alignment)")
    else:
        logger.error("❌ Chamfer distance ≥ 15px")
        passed_all = False
    
    logger.info("="*60)
    
    if passed_all:
        logger.info("✅ ALL VALIDATION CRITERIA PASSED")
        return 0
    else:
        logger.error("❌ SOME CRITERIA FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
