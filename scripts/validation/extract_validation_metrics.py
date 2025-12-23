#!/usr/bin/env python3
"""Extract quality metrics from already-generated depth maps."""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from high_fidelity_depth.quality_metrics import validate_depth_quality, save_metrics_atomic

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def load_depth_16bit(path: Path) -> np.ndarray:
    """Load 16-bit depth as float32 [0, 1]."""
    img = Image.open(path)
    depth_u16 = np.array(img)
    return depth_u16.astype(np.float32) / 65535.0


def main():
    rgb_dir = Path("data/validation_quick")
    depth_dir = Path("outputs/validation_sliver_quick_20251218_122536")
    output_dir = Path("outputs/validation_metrics_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Find all depth files
    depth_files = list(depth_dir.glob("*_depth.tiff"))
    logger.info(f"Found {len(depth_files)} depth files")
    
    for depth_path in sorted(depth_files):
        stem = depth_path.stem.replace("_depth", "")
        
        # Find corresponding RGB
        rgb_candidates = list(rgb_dir.glob(f"{stem}.*"))
        if not rgb_candidates:
            logger.warning(f"No RGB found for {stem}")
            continue
        
        rgb_path = rgb_candidates[0]
        logger.info(f"\nProcessing: {stem}")
        logger.info(f"  RGB: {rgb_path.name}")
        logger.info(f"  Depth: {depth_path.name}")
        
        try:
            # Load RGB and depth
            rgb = load_image_rgb(rgb_path)
            depth = load_depth_16bit(depth_path)
            
            logger.info(f"  RGB shape: {rgb.shape}")
            logger.info(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
            
            # Validate quality (NOW WITH CORRECT API)
            metrics_obj = validate_depth_quality(rgb, depth)
            
            # Convert to dict
            metrics = {
                'image_name': stem,
                'rgb_path': str(rgb_path),
                'depth_path': str(depth_path),
                'edge_f1': metrics_obj.edge_f1,
                'edge_overlap': metrics_obj.edge_overlap,
                'edge_alignment_corr': metrics_obj.edge_alignment_corr,
                'chamfer_distance': metrics_obj.chamfer_distance,
                'edge_width': metrics_obj.edge_width,
                'edge_sharpness_p95': metrics_obj.edge_sharpness_p95,
                'edge_count_ratio': metrics_obj.edge_count_ratio,
                'halo_score': metrics_obj.halo_score,
                'overshoot_penalty': metrics_obj.overshoot_penalty,
                'rgb_edge_count': metrics_obj.rgb_edge_count,
                'depth_edge_count': metrics_obj.depth_edge_count,
                'quality_score': metrics_obj.quality_score(),
                'passed_lenient': metrics_obj.passed(strict=False),
                'passed_strict': metrics_obj.passed(strict=True)
            }
            
            results.append(metrics)
            
            # Save per-image metrics
            metrics_path = output_dir / f"{stem}_metrics.json"
            save_metrics_atomic(metrics, metrics_path)
            
            logger.info(f"  ✅ edge_f1={metrics['edge_f1']:.3f}, chamfer={metrics['chamfer_distance']:.1f}px, quality={metrics['quality_score']:.3f}")
            logger.info(f"      lenient={'PASS' if metrics['passed_lenient'] else 'FAIL'}, strict={'PASS' if metrics['passed_strict'] else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    lenient_passed = sum(1 for r in results if r['passed_lenient'])
    strict_passed = sum(1 for r in results if r['passed_strict'])
    
    avg_edge_f1 = sum(r['edge_f1'] for r in results) / len(results) if results else 0
    avg_chamfer = sum(r['chamfer_distance'] for r in results) / len(results) if results else 0
    avg_quality = sum(r['quality_score'] for r in results) / len(results) if results else 0
    
    summary = {
        'total_images': len(results),
        'lenient_passed': lenient_passed,
        'strict_passed': strict_passed,
        'lenient_pass_rate': lenient_passed / len(results) if results else 0,
        'strict_pass_rate': strict_passed / len(results) if results else 0,
        'avg_edge_f1': avg_edge_f1,
        'avg_chamfer_distance': avg_chamfer,
        'avg_quality_score': avg_quality,
        'results': results
    }
    
    summary_path = output_dir / "summary.json"
    save_metrics_atomic(summary, summary_path)
    
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total images: {len(results)}")
    print(f"Lenient passed: {lenient_passed}/{len(results)} ({100*lenient_passed/len(results) if results else 0:.1f}%)")
    print(f"Strict passed: {strict_passed}/{len(results)} ({100*strict_passed/len(results) if results else 0:.1f}%)")
    print(f"Avg Edge F1: {avg_edge_f1:.3f}")
    print(f"Avg Chamfer: {avg_chamfer:.1f}px")
    print(f"Avg Quality: {avg_quality:.3f}")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
