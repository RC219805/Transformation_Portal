#!/usr/bin/env python3
"""
A/B Validation Script for High-Fidelity Depth Pipeline
=======================================================

Run validation on 750_Picacho images:
- Baseline (single-pass) vs High-Fidelity (tiled + reconciliation)
- Compute edge metrics for each image
- Generate comparison report

Usage:
    python ab_validation.py --input-dir /path/to/images --output-dir /path/to/output
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator, DepthConfig
from high_fidelity_depth.validation import validate_depth_quality, EdgeMetrics
from high_fidelity_depth.isolation_tests import run_isolation_tests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def save_depth(depth: np.ndarray, path: Path):
    """Save depth map as 16-bit TIFF."""
    depth_uint16 = (depth * 65535).astype(np.uint16)
    img = Image.fromarray(depth_uint16)
    img.save(path)
    logger.info(f"Saved depth map: {path}")


def run_baseline(rgb: np.ndarray) -> tuple[np.ndarray, EdgeMetrics]:
    """Run baseline low-res inference + upsampling."""
    from PIL import Image
    
    h, w = rgb.shape[:2]
    max_size = 1024
    
    # Resize to low-res
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if rgb.dtype == np.float32:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb
        
        rgb_pil = Image.fromarray(rgb_uint8)
        rgb_resized = rgb_pil.resize((new_w, new_h), Image.LANCZOS)
        rgb_baseline = np.array(rgb_resized)
        logger.info(f"Baseline: {rgb.shape[:2]} → {rgb_baseline.shape[:2]}")
    else:
        rgb_baseline = rgb
    
    config = DepthConfig(
        tile_size=9999,
        overlap=0,
        reconcile_scales=False
    )
    
    estimator = HighFidelityDepthEstimator(config)
    depth_lowres = estimator.estimate_depth(rgb_baseline, use_global_anchor=False)
    
    # Upsample to original resolution
    if max(h, w) > max_size:
        depth_pil = Image.fromarray((depth_lowres * 255).astype(np.uint8))
        depth_upsampled = depth_pil.resize((w, h), Image.BICUBIC)
        depth = np.array(depth_upsampled).astype(np.float32) / 255.0
    else:
        depth = depth_lowres
    
    metrics = validate_depth_quality(rgb, depth, dilation=3)
    
    return depth, metrics


def run_high_fidelity(rgb: np.ndarray) -> tuple[np.ndarray, EdgeMetrics]:
    """Run high-fidelity tiled inference with scale reconciliation."""
    config = DepthConfig(
        tile_size=1024,
        overlap=128,
        reconcile_scales=True,
        fusion_mode="weighted",
        validate_seams=True
    )
    
    estimator = HighFidelityDepthEstimator(config)
    depth = estimator.estimate_depth(rgb, use_global_anchor=True)
    metrics = validate_depth_quality(rgb, depth, dilation=3)
    
    return depth, metrics


def compare_metrics(baseline: EdgeMetrics, hifi: EdgeMetrics) -> Dict:
    """Compare baseline vs high-fidelity metrics."""
    return {
        "edge_alignment": {
            "baseline": baseline.edge_alignment,
            "hifi": hifi.edge_alignment,
            "delta": hifi.edge_alignment - baseline.edge_alignment,
            "improvement": (hifi.edge_alignment - baseline.edge_alignment) / max(abs(baseline.edge_alignment), 0.001)
        },
        "edge_overlap": {
            "baseline": baseline.edge_overlap,
            "hifi": hifi.edge_overlap,
            "delta": hifi.edge_overlap - baseline.edge_overlap,
            "improvement": (hifi.edge_overlap - baseline.edge_overlap) / max(baseline.edge_overlap, 0.001)
        },
        "edge_count_ratio": {
            "baseline": baseline.edge_count_ratio,
            "hifi": hifi.edge_count_ratio,
            "delta": hifi.edge_count_ratio - baseline.edge_count_ratio,
            "improvement": (baseline.edge_count_ratio - hifi.edge_count_ratio) / max(baseline.edge_count_ratio, 0.001)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="A/B validation for high-fidelity depth pipeline")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory with images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for depth maps")
    parser.add_argument("--run-isolation", action="store_true", help="Run isolation tests on first image")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.output_dir / "baseline"
    hifi_dir = args.output_dir / "high_fidelity"
    baseline_dir.mkdir(exist_ok=True)
    hifi_dir.mkdir(exist_ok=True)
    
    # Find images
    image_paths = sorted(args.input_dir.glob("*.tiff")) + sorted(args.input_dir.glob("*.tif"))
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    logger.info(f"Found {len(image_paths)} images")
    
    results = []
    
    for idx, image_path in enumerate(image_paths):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {idx+1}/{len(image_paths)}: {image_path.name}")
        logger.info(f"{'='*60}")
        
        # Load image
        rgb = load_image(image_path)
        logger.info(f"Loaded image: {rgb.shape}")
        
        # Run isolation tests on first image
        if idx == 0 and args.run_isolation:
            logger.info("\nRunning isolation tests on first image...")
            isolation_results = run_isolation_tests(rgb, args.output_dir)
            
            # Save isolation results
            isolation_file = args.output_dir / "isolation_test_results.json"
            with open(isolation_file, 'w') as f:
                json.dump({
                    name: {
                        "name": r.name,
                        "description": r.description,
                        "passed": bool(r.passed),
                        "edge_alignment": float(r.metrics.edge_alignment),
                        "edge_overlap": float(r.metrics.edge_overlap),
                        "edge_count_ratio": float(r.metrics.edge_count_ratio)
                    }
                    for name, r in isolation_results.items()
                }, f, indent=2)
            logger.info(f"Saved isolation results: {isolation_file}")
        
        # Baseline inference
        logger.info("\nRunning baseline inference...")
        baseline_depth, baseline_metrics = run_baseline(rgb)
        save_depth(baseline_depth, baseline_dir / f"{image_path.stem}_baseline.tiff")
        
        # High-fidelity inference
        logger.info("\nRunning high-fidelity inference...")
        hifi_depth, hifi_metrics = run_high_fidelity(rgb)
        save_depth(hifi_depth, hifi_dir / f"{image_path.stem}_hifi.tiff")
        
        # Compare metrics
        comparison = compare_metrics(baseline_metrics, hifi_metrics)
        
        result = {
            "image": image_path.name,
            "baseline": {
                "edge_alignment": baseline_metrics.edge_alignment,
                "edge_overlap": baseline_metrics.edge_overlap,
                "edge_width": baseline_metrics.edge_width,
                "edge_count_ratio": baseline_metrics.edge_count_ratio,
                "halo_score": baseline_metrics.halo_score
            },
            "high_fidelity": {
                "edge_alignment": hifi_metrics.edge_alignment,
                "edge_overlap": hifi_metrics.edge_overlap,
                "edge_width": hifi_metrics.edge_width,
                "edge_count_ratio": hifi_metrics.edge_count_ratio,
                "halo_score": hifi_metrics.halo_score
            },
            "comparison": comparison,
            "passed": hifi_metrics.passed(strict=False)
        }
        
        results.append(result)
        
        # Log summary
        logger.info(f"\nResults for {image_path.name}:")
        logger.info(f"  Baseline edge alignment: {baseline_metrics.edge_alignment:.3f}")
        logger.info(f"  HiFi edge alignment:     {hifi_metrics.edge_alignment:.3f} ({comparison['edge_alignment']['delta']:+.3f})")
        logger.info(f"  Baseline edge overlap:   {baseline_metrics.edge_overlap:.3f}")
        logger.info(f"  HiFi edge overlap:       {hifi_metrics.edge_overlap:.3f} ({comparison['edge_overlap']['delta']:+.3f})")
        logger.info(f"  Quality gate: {'✅ PASS' if result['passed'] else '❌ FAIL'}")
    
    # Save results
    results_file = args.output_dir / "ab_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"A/B VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Processed {len(results)} images")
    logger.info(f"Results saved to: {results_file}")
    
    # Aggregate statistics
    passed_count = sum(1 for r in results if r['passed'])
    logger.info(f"\nQuality gate: {passed_count}/{len(results)} passed ({passed_count/len(results)*100:.1f}%)")
    
    avg_alignment_baseline = np.mean([r['baseline']['edge_alignment'] for r in results])
    avg_alignment_hifi = np.mean([r['high_fidelity']['edge_alignment'] for r in results])
    avg_overlap_baseline = np.mean([r['baseline']['edge_overlap'] for r in results])
    avg_overlap_hifi = np.mean([r['high_fidelity']['edge_overlap'] for r in results])
    
    logger.info(f"\nAverage edge alignment: {avg_alignment_baseline:.3f} → {avg_alignment_hifi:.3f} ({avg_alignment_hifi - avg_alignment_baseline:+.3f})")
    logger.info(f"Average edge overlap:   {avg_overlap_baseline:.3f} → {avg_overlap_hifi:.3f} ({avg_overlap_hifi - avg_overlap_baseline:+.3f})")
    
    # Materials V3 readiness
    logger.info(f"\n{'='*60}")
    logger.info(f"MATERIALS V3 INTEGRATION READINESS")
    logger.info(f"{'='*60}")
    
    if avg_overlap_hifi >= 0.4 and avg_alignment_hifi >= 0.5:
        logger.info("✅ READY: Depth quality meets Materials V3 requirements")
        logger.info("   - Edge overlap >40%: ✅")
        logger.info("   - Edge alignment >0.5: ✅")
        logger.info("   - Depth boundaries are crisp and aligned with RGB edges")
        logger.info("   - Normal maps will be accurate for material segmentation")
    else:
        logger.warning("⚠️  NOT READY: Depth quality below Materials V3 threshold")
        if avg_overlap_hifi < 0.4:
            logger.warning(f"   - Edge overlap: {avg_overlap_hifi:.3f} < 0.40 ❌")
        if avg_alignment_hifi < 0.5:
            logger.warning(f"   - Edge alignment: {avg_alignment_hifi:.3f} < 0.50 ❌")


if __name__ == "__main__":
    main()
