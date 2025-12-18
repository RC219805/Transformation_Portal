#!/usr/bin/env python3
"""
Isolation Test Suite: Identify Which Stage Creates Artifacts
=============================================================

Runs minimal isolation tests to pinpoint the exact failure:
1. Baseline only (HF pipeline)
2. Tiling only (no refinement)
3. Global anchor only
4. Guided filter only
5. Edge snapping only
6. CLAHE only

Each test logs metrics. The first test where edge_overlap collapses
or edge_count spikes is the culprit.

Reference: User feedback 2025-12-18
"Do not tune. Isolate."
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_edge_metrics_strict(depth: np.ndarray, rgb: np.ndarray) -> Dict:
    """
    Strict edge metrics with spatial tolerance.
    
    Returns:
        - edge_count: number of edge pixels
        - edge_overlap: % of depth edges that align with RGB (3px tolerance)
        - edge_alignment: correlation of gradient magnitudes
        - boundary_energy: gradient energy at tile boundaries (if applicable)
    """
    # Normalize depth to [0, 1]
    if depth.dtype == np.uint16:
        depth_norm = depth.astype(np.float32) / 65535.0
    elif depth.dtype == np.uint8:
        depth_norm = depth.astype(np.float32) / 255.0
    else:
        depth_norm = depth.astype(np.float32)
        if depth_norm.max() > 1.0:
            depth_norm = depth_norm / depth_norm.max()
    
    # RGB to grayscale
    if rgb.dtype == np.float32:
        gray = (rgb * 255).astype(np.uint8)
    else:
        gray = rgb
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    
    # Detect edges (Canny)
    rgb_edges = cv2.Canny(gray, 50, 150)
    
    # Depth gradient magnitude
    sobel_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Depth edges (threshold at p90)
    threshold = np.percentile(depth_grad_mag, 90)
    depth_edges = (depth_grad_mag > threshold).astype(np.uint8) * 255
    
    # Edge count
    edge_count = (depth_edges > 0).sum()
    
    # Edge overlap with spatial tolerance (3px dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rgb_edges_dilated = cv2.dilate(rgb_edges, kernel, iterations=1)
    
    overlap_pixels = np.sum((rgb_edges_dilated > 0) & (depth_edges > 0))
    edge_overlap = overlap_pixels / max(edge_count, 1)
    
    # Correlation of gradient magnitudes
    rgb_grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)**2
    rgb_grad_mag = np.sqrt(rgb_grad)
    
    correlation = np.corrcoef(rgb_grad_mag.ravel(), depth_grad_mag.ravel())[0, 1]
    
    return {
        'edge_count': int(edge_count),
        'edge_overlap': float(edge_overlap),
        'edge_alignment_correlation': float(correlation),
        'edge_gradient_mean': float(depth_grad_mag.mean() * 255),
        'edge_gradient_p95': float(np.percentile(depth_grad_mag * 255, 95))
    }


def test_baseline(rgb: np.ndarray) -> np.ndarray:
    """Test 1: Baseline HF pipeline only."""
    from transformers import pipeline
    
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Baseline (HF pipeline, 518px resize)")
    logger.info("="*60)
    
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=-1)
    
    if rgb.dtype == np.float32:
        rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        rgb_pil = Image.fromarray(rgb)
    
    result = pipe(rgb_pil)
    
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
    
    logger.info(f"✓ Baseline depth: {depth.shape}")
    return depth


def test_tiling_only(rgb: np.ndarray) -> np.ndarray:
    """Test 2: Tiling with bypass, NO refinement, NO global anchor."""
    from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig
    
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Tiling only (bypass 518px, no refinement, no anchor)")
    logger.info("="*60)
    
    config = TiledInferenceConfig(
        tile_size=1024,
        overlap=128,
        bypass_image_processor=True,
        use_global_anchor=False,  # DISABLED
        use_edge_snapping=False,  # DISABLED
        use_production_refinement=False  # DISABLED
    )
    
    estimator = TiledDepthEstimator(config)
    depth = estimator.estimate_depth(rgb)
    
    logger.info(f"✓ Tiled depth: {depth.shape}")
    return depth


def test_global_anchor_only(rgb: np.ndarray) -> np.ndarray:
    """Test 3: Global anchor fusion, NO tiling detail."""
    from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig
    
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Global anchor only (low-res global, no tiles)")
    logger.info("="*60)
    
    # Run low-res global pass only
    config = TiledInferenceConfig(
        tile_size=1024,
        overlap=128,
        bypass_image_processor=True,
        use_global_anchor=True,
        use_edge_snapping=False,
        use_production_refinement=False
    )
    
    estimator = TiledDepthEstimator(config)
    
    # Manually run just global pass
    from lux_depth_v2.global_anchor import GlobalAnchorFusion
    fusion = GlobalAnchorFusion(config.global_anchor_config)
    
    rgb_global, scale = fusion._resize_for_global_pass(rgb)
    depth_global_lowres = estimator._infer_single_image(rgb_global)
    depth = fusion._upsample_global_depth(depth_global_lowres, (rgb.shape[0], rgb.shape[1]))
    
    logger.info(f"✓ Global anchor depth: {depth.shape}")
    return depth


def test_guided_filter_only(rgb: np.ndarray) -> np.ndarray:
    """Test 4: Baseline + guided/bilateral filter only."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Guided filter only (on baseline depth)")
    logger.info("="*60)
    
    # Get baseline
    depth = test_baseline(rgb)
    
    # Apply bilateral filter (guided filter fallback)
    if depth.dtype == np.float32:
        depth_uint8 = (depth * 255).astype(np.uint8)
    else:
        depth_uint8 = depth
    
    filtered = cv2.bilateralFilter(depth_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    depth_filtered = filtered.astype(np.float32) / 255.0
    
    logger.info(f"✓ Filtered depth: {depth_filtered.shape}")
    return depth_filtered


def test_edge_snap_only(rgb: np.ndarray) -> np.ndarray:
    """Test 5: Baseline + edge snapping only (AND-gated)."""
    from lux_depth_v2.depth_refinement import ProductionDepthRefiner, DepthRefinementConfig
    
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Edge snapping only (AND-gated, on baseline)")
    logger.info("="*60)
    
    # Get baseline
    depth = test_baseline(rgb)
    
    # Apply ONLY edge snapping
    config = DepthRefinementConfig(
        use_clahe=False,
        use_edge_filter=False,
        use_edge_snap=True
    )
    
    refiner = ProductionDepthRefiner(config)
    depth_snapped = refiner._apply_edge_snap(depth, rgb)
    
    logger.info(f"✓ Edge-snapped depth: {depth_snapped.shape}")
    return depth_snapped


def test_clahe_only(rgb: np.ndarray) -> np.ndarray:
    """Test 6: Baseline + CLAHE only (conservative params)."""
    from lux_depth_v2.depth_refinement import ProductionDepthRefiner, DepthRefinementConfig
    
    logger.info("\n" + "="*60)
    logger.info("TEST 6: CLAHE only (conservative, on baseline)")
    logger.info("="*60)
    
    # Get baseline
    depth = test_baseline(rgb)
    
    # Apply ONLY CLAHE
    config = DepthRefinementConfig(
        use_clahe=True,
        clahe_clip_limit=1.5,  # More conservative than 2.0
        clahe_tile_grid=16,    # Larger than 8 to reduce structure
        use_edge_filter=False,
        use_edge_snap=False
    )
    
    refiner = ProductionDepthRefiner(config)
    depth_clahe = refiner._apply_clahe(depth)
    
    logger.info(f"✓ CLAHE depth: {depth_clahe.shape}")
    return depth_clahe


def run_isolation_tests(rgb: np.ndarray, output_dir: Path):
    """Run all isolation tests and report which stage fails."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("ISOLATION TEST SUITE")
    logger.info("="*60)
    logger.info(f"Input: {rgb.shape}")
    
    tests = [
        ("baseline", test_baseline),
        ("tiling_only", test_tiling_only),
        ("global_anchor_only", test_global_anchor_only),
        ("guided_filter_only", test_guided_filter_only),
        ("edge_snap_only", test_edge_snap_only),
        ("clahe_only", test_clahe_only)
    ]
    
    results = {}
    baseline_metrics = None
    
    for test_name, test_func in tests:
        try:
            depth = test_func(rgb)
            
            # Compute strict metrics
            metrics = compute_edge_metrics_strict(depth, rgb)
            
            # Save depth
            depth_vis = (depth * 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(output_dir / f"{test_name}_depth.png")
            
            # Store results
            results[test_name] = metrics
            
            if test_name == "baseline":
                baseline_metrics = metrics
            
            # Check for failures
            if baseline_metrics:
                edge_count_ratio = metrics['edge_count'] / max(baseline_metrics['edge_count'], 1)
                overlap_drop = baseline_metrics['edge_overlap'] - metrics['edge_overlap']
                
                failure = False
                if edge_count_ratio > 10:
                    logger.error(f"  ❌ FAILURE: Edge count exploded {edge_count_ratio:.1f}x")
                    failure = True
                if overlap_drop > 0.1:
                    logger.error(f"  ❌ FAILURE: Edge overlap dropped by {overlap_drop:.2%}")
                    failure = True
                if metrics['edge_overlap'] < 0.05:
                    logger.error(f"  ❌ FAILURE: Edge overlap near-zero ({metrics['edge_overlap']:.2%})")
                    failure = True
                
                if failure:
                    logger.error(f"\n{'='*60}")
                    logger.error(f"CULPRIT IDENTIFIED: {test_name.upper()}")
                    logger.error(f"{'='*60}\n")
            
            # Log metrics
            logger.info(f"\n{test_name} metrics:")
            logger.info(f"  Edge count:       {metrics['edge_count']:,}")
            logger.info(f"  Edge overlap:     {metrics['edge_overlap']:.2%}")
            logger.info(f"  Edge correlation: {metrics['edge_alignment_correlation']:.3f}")
            logger.info(f"  Gradient p95:     {metrics['edge_gradient_p95']:.2f}")
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = {"error": str(e)}
    
    # Save report
    with open(output_dir / "isolation_report.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ISOLATION SUMMARY")
    logger.info("="*60)
    
    for test_name, metrics in results.items():
        if 'error' in metrics:
            logger.info(f"{test_name:25s} ERROR")
        else:
            edge_ratio = metrics['edge_count'] / max(baseline_metrics['edge_count'], 1) if baseline_metrics else 1.0
            logger.info(f"{test_name:25s} edges={edge_ratio:6.1f}x  overlap={metrics['edge_overlap']:5.1%}  corr={metrics['edge_alignment_correlation']:+.3f}")
    
    logger.info(f"\n✓ Report saved to {output_dir}/isolation_report.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Isolation test suite")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/isolation_tests"))
    
    args = parser.parse_args()
    
    logger.info(f"Loading: {args.input}")
    rgb = np.array(Image.open(args.input).convert("RGB"))
    
    run_isolation_tests(rgb, args.output)
