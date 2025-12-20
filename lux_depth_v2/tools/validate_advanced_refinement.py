#!/usr/bin/env python3
"""
Advanced Refinement Validation Script
======================================

Validates edge-aware refinement techniques on structure scenes
to measure pass rate improvement from 50% → 60%+.

Usage:
    python validate_advanced_refinement.py --input-dir validation_baseline/structure/ \
                                           --output-dir output/refined/ \
                                           --technique hybrid \
                                           --report

Author: Transformation Portal Specialist
Date: 2025-12-20
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import cv2

from lux_depth_v2.advanced_refinement import (
    DepthRefiner,
    AdvancedRefinementConfig,
    compute_edge_metrics,
    compute_chamfer_distance
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    # Convert BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path: Path) -> np.ndarray:
    """Load depth map (supports various formats)."""
    if path.suffix.lower() in ['.tif', '.tiff']:
        # Try to load as 16-bit TIFF
        depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    
    if depth is None:
        raise ValueError(f"Failed to load depth: {path}")
    
    return depth


def save_depth(depth: np.ndarray, path: Path) -> None:
    """Save depth map preserving precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as 16-bit TIFF if high precision
    if depth.dtype == np.uint16 or depth.max() <= 1.0:
        if depth.dtype == np.float32:
            depth_uint16 = (depth * 65535).astype(np.uint16)
        else:
            depth_uint16 = depth.astype(np.uint16)
        cv2.imwrite(str(path), depth_uint16)
    else:
        cv2.imwrite(str(path), depth)


def infer_depth_placeholder(rgb: np.ndarray) -> np.ndarray:
    """
    Placeholder depth inference (replace with actual model).
    
    For validation, assume depth maps are pre-computed.
    """
    # Simple gradient-based fake depth for testing
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    depth = cv2.GaussianBlur(gray, (15, 15), 5)
    depth = depth.astype(np.float32) / 255.0
    return depth


def validate_scene(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    refiner: DepthRefiner,
    technique: str,
    compute_gt_metrics: bool = False,
    depth_gt: np.ndarray = None
) -> Dict:
    """
    Validate refinement on a single scene.
    
    Returns:
        Dict with metrics before/after refinement
    """
    # Compute metrics before refinement
    metrics_before = compute_edge_metrics(depth_raw, rgb, "comprehensive")
    
    # Apply refinement
    start_time = time.time()
    depth_refined = refiner.refine(depth_raw, rgb, technique=technique)
    refinement_time = time.time() - start_time
    
    # Compute metrics after refinement
    metrics_after = compute_edge_metrics(depth_refined, rgb, "comprehensive")
    
    # Compute improvements
    improvements = {
        f'{k}_improvement': metrics_after[k] - metrics_before[k]
        for k in metrics_before.keys()
        if isinstance(metrics_before[k], (int, float))
    }
    
    # Ground truth metrics (if available)
    gt_metrics = {}
    if compute_gt_metrics and depth_gt is not None:
        chamfer_before = compute_chamfer_distance(depth_raw, depth_gt)
        chamfer_after = compute_chamfer_distance(depth_refined, depth_gt)
        gt_metrics = {
            'chamfer_before': chamfer_before,
            'chamfer_after': chamfer_after,
            'chamfer_improvement': chamfer_before - chamfer_after
        }
    
    return {
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'improvements': improvements,
        'gt_metrics': gt_metrics,
        'refinement_time_ms': refinement_time * 1000,
        'depth_refined': depth_refined
    }


def compute_pass_rate(
    results: List[Dict],
    metric_name: str = 'edge_f1',
    threshold: float = 0.55
) -> Tuple[float, float]:
    """
    Compute pass rate before and after refinement.
    
    Returns:
        Tuple of (pass_rate_before, pass_rate_after)
    """
    pass_count_before = sum(
        1 for r in results
        if r['metrics_before'].get(metric_name, 0) >= threshold
    )
    pass_count_after = sum(
        1 for r in results
        if r['metrics_after'].get(metric_name, 0) >= threshold
    )
    
    total = len(results)
    pass_rate_before = pass_count_before / total if total > 0 else 0
    pass_rate_after = pass_count_after / total if total > 0 else 0
    
    return pass_rate_before, pass_rate_after


def generate_report(results: List[Dict], output_path: Path) -> None:
    """Generate validation report with statistics."""
    report = {
        'summary': {},
        'scenes': []
    }
    
    # Aggregate statistics
    total_scenes = len(results)
    
    # Pass rate analysis
    for metric_name in ['edge_f1', 'edge_alignment', 'edge_precision', 'edge_recall']:
        if metric_name in results[0]['metrics_before']:
            pass_before, pass_after = compute_pass_rate(results, metric_name, threshold=0.55)
            report['summary'][f'{metric_name}_pass_rate_before'] = pass_before
            report['summary'][f'{metric_name}_pass_rate_after'] = pass_after
            report['summary'][f'{metric_name}_improvement_pct'] = (
                (pass_after - pass_before) * 100
            )
    
    # Average metrics
    for metric_name in results[0]['metrics_before'].keys():
        if isinstance(results[0]['metrics_before'][metric_name], (int, float)):
            avg_before = np.mean([r['metrics_before'][metric_name] for r in results])
            avg_after = np.mean([r['metrics_after'][metric_name] for r in results])
            report['summary'][f'{metric_name}_avg_before'] = float(avg_before)
            report['summary'][f'{metric_name}_avg_after'] = float(avg_after)
            report['summary'][f'{metric_name}_avg_improvement'] = float(avg_after - avg_before)
    
    # Processing time
    avg_time = np.mean([r['refinement_time_ms'] for r in results])
    report['summary']['avg_refinement_time_ms'] = float(avg_time)
    
    # Per-scene results
    for i, result in enumerate(results):
        scene_report = {
            'scene_id': i,
            'metrics_before': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                             for k, v in result['metrics_before'].items()},
            'metrics_after': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                            for k, v in result['metrics_after'].items()},
            'improvements': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in result['improvements'].items()},
            'refinement_time_ms': float(result['refinement_time_ms'])
        }
        
        if result['gt_metrics']:
            scene_report['gt_metrics'] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v 
                for k, v in result['gt_metrics'].items()
            }
        
        report['scenes'].append(scene_report)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {output_path}")


def print_summary(results: List[Dict]) -> None:
    """Print validation summary to console."""
    print("\n" + "="*60)
    print("ADVANCED REFINEMENT VALIDATION SUMMARY")
    print("="*60)
    
    # Pass rate
    pass_before, pass_after = compute_pass_rate(results, 'edge_f1', threshold=0.55)
    print(f"\nPass Rate (Edge F1 >= 0.55):")
    print(f"  Before: {pass_before*100:.1f}%")
    print(f"  After:  {pass_after*100:.1f}%")
    print(f"  Improvement: {(pass_after - pass_before)*100:+.1f}%")
    
    # Key metrics
    print(f"\nAverage Edge Metrics:")
    for metric in ['edge_f1', 'edge_alignment', 'edge_precision', 'edge_recall']:
        if metric in results[0]['metrics_before']:
            avg_before = np.mean([r['metrics_before'][metric] for r in results])
            avg_after = np.mean([r['metrics_after'][metric] for r in results])
            improvement = avg_after - avg_before
            print(f"  {metric:20s}: {avg_before:.3f} → {avg_after:.3f} ({improvement:+.3f})")
    
    # Processing time
    avg_time = np.mean([r['refinement_time_ms'] for r in results])
    print(f"\nAverage Refinement Time: {avg_time:.1f} ms")
    
    # Target achievement
    target_pass_rate = 0.60
    print(f"\nTarget Achievement:")
    if pass_after >= target_pass_rate:
        print(f"  ✓ Target achieved: {pass_after*100:.1f}% >= {target_pass_rate*100:.1f}%")
    else:
        gap = target_pass_rate - pass_after
        print(f"  ✗ Target not met: {pass_after*100:.1f}% < {target_pass_rate*100:.1f}%")
        print(f"    Gap: {gap*100:.1f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate advanced depth refinement")
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory with RGB images'
    )
    parser.add_argument(
        '--depth-dir',
        type=Path,
        help='Directory with pre-computed depth maps (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for refined depth maps'
    )
    parser.add_argument(
        '--technique',
        type=str,
        default='hybrid',
        choices=['bilateral', 'guided', 'edge_guided', 'gradient_consistency', 'hybrid'],
        help='Refinement technique to use'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate JSON report'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='JSON config file for AdvancedRefinementConfig'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and args.config.exists():
        with open(args.config) as f:
            config_dict = json.load(f)
        config = AdvancedRefinementConfig(**config_dict)
    else:
        config = AdvancedRefinementConfig()
    
    # Initialize refiner
    refiner = DepthRefiner(config)
    logger.info(f"Initialized refiner with technique: {args.technique}")
    
    # Find input images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(args.input_dir.glob(f'*{ext}'))
        image_paths.extend(args.input_dir.glob(f'*{ext.upper()}'))
    
    image_paths = sorted(set(image_paths))
    logger.info(f"Found {len(image_paths)} images in {args.input_dir}")
    
    if len(image_paths) == 0:
        logger.error("No images found!")
        return
    
    # Process each scene
    results = []
    for i, img_path in enumerate(image_paths):
        logger.info(f"Processing {i+1}/{len(image_paths)}: {img_path.name}")
        
        # Load RGB
        rgb = load_image(img_path)
        
        # Load or infer depth
        if args.depth_dir:
            depth_path = args.depth_dir / img_path.name.replace(img_path.suffix, '_depth.tif')
            if not depth_path.exists():
                depth_path = args.depth_dir / img_path.name
            
            if depth_path.exists():
                depth_raw = load_depth(depth_path)
            else:
                logger.warning(f"Depth not found: {depth_path}, inferring from RGB")
                depth_raw = infer_depth_placeholder(rgb)
        else:
            depth_raw = infer_depth_placeholder(rgb)
        
        # Validate scene
        result = validate_scene(rgb, depth_raw, refiner, args.technique)
        results.append(result)
        
        # Save refined depth
        output_path = args.output_dir / img_path.name.replace(img_path.suffix, '_refined.tif')
        save_depth(result['depth_refined'], output_path)
        logger.info(f"  Saved refined depth to {output_path}")
    
    # Print summary
    print_summary(results)
    
    # Generate report
    if args.report:
        report_path = args.output_dir / 'validation_report.json'
        generate_report(results, report_path)
    
    logger.info("Validation complete!")


if __name__ == '__main__':
    main()
