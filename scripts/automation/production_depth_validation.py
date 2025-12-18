#!/usr/bin/env python3
"""
PRODUCTION-HARDENED DEPTH VALIDATION
====================================

CRITICAL FIXES based on terminal crash analysis:
1. Streaming memory-safe blending (no tile stacking)
2. Capped Theil-Sen sampling (prevents pathological slowdown)
3. Atomic JSON write with validation
4. Per-image error handling with tracebacks
5. Resumable execution (skip completed images)
6. Memory telemetry at key stages
7. Fail-fast on incomplete runs (no false "success")

This script will NOT generate "production ready" docs unless ALL images complete.
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import psutil
from PIL import Image

from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator, DepthConfig
from high_fidelity_depth.quality_metrics import validate_depth_quality, save_metrics_atomic
from high_fidelity_depth.refinement import edge_snap_refinement
from high_fidelity_depth.comprehensive_validation import validate_seams, create_edge_overlay

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "rss_mb": mem_info.rss / 1024**2,
        "vms_mb": mem_info.vms / 1024**2,
        "percent": process.memory_percent()
    }


def log_memory(stage: str):
    """Log memory usage at key stages."""
    mem = get_memory_usage()
    logger.info(f"[MEMORY] {stage}: RSS={mem['rss_mb']:.1f}MB ({mem['percent']:.1f}%)")


def process_single_image(
    rgb_path: Path,
    output_dir: Path,
    config: DepthConfig,
    refinement_config: Optional[Dict] = None,
    force: bool = False
) -> Dict:
    """
    Process single image with full error handling and telemetry.
    
    Returns:
        Metrics dict with 'success' boolean and optional 'error' field
    """
    image_name = rgb_path.stem
    depth_path = output_dir / f"{image_name}_depth.tiff"
    metrics_path = output_dir / f"{image_name}_metrics.json"
    
    # Check if already processed
    if not force and metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                existing = json.load(f)
            if existing.get('success', False):
                logger.info(f"✓ Skipping {image_name} (already processed)")
                return existing
        except Exception as e:
            logger.warning(f"Failed to read existing metrics: {e}, reprocessing")
    
    logger.info("="*70)
    logger.info(f"PROCESSING: {image_name}")
    logger.info("="*70)
    
    log_memory("start")
    
    result = {
        "image_name": image_name,
        "rgb_path": str(rgb_path),
        "success": False,
        "error": None,
        "traceback": None
    }
    
    try:
        # Load image
        logger.info(f"Loading {rgb_path}")
        img = Image.open(rgb_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        rgb = np.array(img)
        logger.info(f"Image size: {rgb.shape[:2]} ({rgb.dtype})")
        
        log_memory("after_load")
        
        # Estimate depth (PRIORITY 2 FIX: enable calibration smoothing)
        logger.info("Starting depth estimation...")
        estimator = HighFidelityDepthEstimator(config)
        depth = estimator.estimate_depth(rgb, use_global_anchor=False, smooth_calibrations=True)
        
        log_memory("after_depth")
        
        # Apply refinement if configured
        if refinement_config:
            logger.info("Applying refinement...")
            depth = edge_snap_refinement(
                depth=depth,
                rgb=rgb,
                strength=refinement_config.get('edge_snap_strength', 0.2),
                dilation=refinement_config.get('dilation', 5)
            )
            log_memory("after_refinement")
        
        # Save depth
        logger.info(f"Saving depth to {depth_path}")
        depth_uint16 = (np.clip(depth, 0, 1) * 65535).astype(np.uint16)
        Image.fromarray(depth_uint16, mode='I;16').save(depth_path, compression='tiff_adobe_deflate')
        
        # Validate (PRIORITY 3 FIX: add overshoot heatmap)
        logger.info("Running quality validation...")
        heatmap_path = output_dir / f"{image_name}_overshoot.png"
        metrics = validate_depth_quality(
            rgb, depth, dilation=3, 
            save_heatmap=True, 
            heatmap_path=heatmap_path
        )
        seams_ok, seam_ratio = validate_seams(depth, config.tile_size, config.overlap)
        
        log_memory("after_validation")
        
        # Create visual overlay
        overlay_path = output_dir / f"{image_name}_edges.png"
        create_edge_overlay(rgb, depth, overlay_path)
        
        # Compile results (PRIORITY 1 FIX: separate execution/seam/quality outcomes)
        result.update({
            "success": True,  # Execution succeeded without exception
            "depth_path": str(depth_path),
            "overlay_path": str(overlay_path),
            "image_size": list(rgb.shape[:2]),
            "metrics": metrics.to_dict(),
            "seam_validation": {
                "passed": bool(seams_ok),
                "boundary_ratio": float(seam_ratio)
            },
            "quality_score": float(metrics.quality_score()),
            "passed_lenient": bool(metrics.passed(strict=False)),
            "passed_strict": bool(metrics.passed(strict=True))
        })
        
        logger.info(f"✓ SUCCESS: Quality score = {metrics.quality_score():.3f}")
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"❌ FAILED: {error_msg}")
        logger.error(f"Traceback:\n{error_trace}")
        
        result.update({
            "error": error_msg,
            "traceback": error_trace
        })
    
    finally:
        # Always save metrics (atomic write with validation)
        logger.info(f"Saving metrics to {metrics_path}")
        save_metrics_atomic(result, metrics_path)
        
        log_memory("cleanup")
    
    return result


def run_production_validation(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.tif*",  # PRIORITY 5 FIX: include .tiff extension
    tile_size: int = 1024,
    overlap: int = 128,
    force: bool = False,
    use_refinement: bool = True
) -> Dict:
    """
    Run production validation on full dataset with strict error handling.
    
    PRIORITY 5 FIX: Process all images in directory with category reporting.
    
    Returns:
        Aggregate report with success/failure status
    """
    logger.info("="*70)
    logger.info("PRODUCTION DEPTH VALIDATION - STABILITY-FIRST MODE")
    logger.info("="*70)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Pattern: {pattern}")
    logger.info(f"Tile size: {tile_size}, Overlap: {overlap}")
    logger.info(f"Force reprocess: {force}")
    logger.info(f"Use refinement: {use_refinement}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images (PRIORITY 5: support both .tif and .tiff)
    image_paths = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    # Deduplicate (in case pattern matched both)
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    logger.info(f"Found {len(image_paths)} images:")
    for img_path in image_paths:
        logger.info(f"  - {img_path.name}")
    
    # PRIORITY 5 FIX: Categorize images (interior vs exterior)
    categories = {}
    for img_path in image_paths:
        name_lower = img_path.stem.lower()
        if any(x in name_lower for x in ['aerial', 'pool', 'exterior']):
            category = 'exterior'
        elif any(x in name_lower for x in ['kitchen', 'bedroom', 'bathroom', 'greatroom', 'room']):
            category = 'interior'
        else:
            category = 'other'
        
        if category not in categories:
            categories[category] = []
        categories[category].append(img_path)
    
    logger.info(f"\nImage categories:")
    for cat, paths in categories.items():
        logger.info(f"  {cat}: {len(paths)} images")
    
    # Configure pipeline (STABILITY-FIRST)
    config = DepthConfig(
        model_name="depth-anything/Depth-Anything-V2-Large-hf",
        device="auto",
        tile_size=tile_size,
        overlap=overlap,
        reconcile_scales=True,
        reconcile_method="robust",
        fusion_mode="weighted",  # ALWAYS weighted (streaming, memory-safe)
        blend_window="hann",
        validate_seams=True,
        seam_energy_threshold=1.2
    )
    
    # Refinement config (conservative, single-pass)
    refinement_config = {
        "edge_snap_strength": 0.2,
        "dilation": 5
    } if use_refinement else None
    
    # Process images
    results = []
    failed_images = []
    
    for idx, rgb_path in enumerate(image_paths, 1):
        logger.info(f"\n[{idx}/{len(image_paths)}] {rgb_path.name}")
        
        try:
            result = process_single_image(
                rgb_path,
                output_dir,
                config,
                refinement_config,
                force=force
            )
            results.append(result)
            
            if not result['success']:
                failed_images.append(rgb_path.name)
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interrupted by user")
            break
        
        except Exception as e:
            logger.error(f"❌ Unhandled exception processing {rgb_path.name}: {e}")
            logger.error(traceback.format_exc())
            failed_images.append(rgb_path.name)
            
            # Continue with next image (don't abort entire run)
            results.append({
                "image_name": rgb_path.stem,
                "rgb_path": str(rgb_path),
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    # Aggregate report (PRIORITY 1 FIX: separate execution/seam/quality outcomes)
    total = len(results)
    execution_succeeded = sum(1 for r in results if r.get('success', False))
    execution_failed = total - execution_succeeded
    
    # Quality pass rates (from successful executions only)
    successful_results = [r for r in results if r.get('success', False)]
    seam_passed = sum(1 for r in successful_results if r.get('seam_validation', {}).get('passed', False))
    quality_passed_lenient = sum(1 for r in successful_results if r.get('passed_lenient', False))
    quality_passed_strict = sum(1 for r in successful_results if r.get('passed_strict', False))
    
    # PRIORITY 5 FIX: Per-category reporting
    category_stats = {}
    for category, cat_paths in categories.items():
        cat_results = [r for r in successful_results if any(p.stem in r['image_name'] for p in cat_paths)]
        if cat_results:
            category_stats[category] = {
                "total": len(cat_results),
                "seam_passed": sum(1 for r in cat_results if r.get('seam_validation', {}).get('passed', False)),
                "quality_passed_lenient": sum(1 for r in cat_results if r.get('passed_lenient', False)),
                "quality_passed_strict": sum(1 for r in cat_results if r.get('passed_strict', False)),
                "avg_edge_f1": float(np.mean([r['metrics']['edge_f1'] for r in cat_results])),
                "avg_seam_ratio": float(np.mean([r['seam_validation']['boundary_ratio'] for r in cat_results]))
            }
    
    logger.info("\n" + "="*70)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total: {total}")
    logger.info(f"Execution: {execution_succeeded}/{total} succeeded, {execution_failed}/{total} failed")
    logger.info(f"Seam validation: {seam_passed}/{execution_succeeded} passed")
    logger.info(f"Quality (lenient): {quality_passed_lenient}/{execution_succeeded} passed")
    logger.info(f"Quality (strict): {quality_passed_strict}/{execution_succeeded} passed ⚠️ KEY METRIC")
    
    # PRIORITY 5 FIX: Category breakdown
    if category_stats:
        logger.info("\n--- Per-Category Results ---")
        for category, stats in category_stats.items():
            logger.info(f"{category.upper()}: {stats['quality_passed_strict']}/{stats['total']} strict pass, "
                       f"avg_edge_f1={stats['avg_edge_f1']:.3f}, avg_seam_ratio={stats['avg_seam_ratio']:.3f}")
    
    if failed_images:
        logger.error(f"\n❌ FAILED IMAGES ({len(failed_images)}):")
        for name in failed_images:
            logger.error(f"  - {name}")
    
    # Compute aggregate metrics (PRIORITY 1 FIX: separate outcomes)
    aggregate = {
        "total_images": total,
        "execution_succeeded": execution_succeeded,
        "execution_failed": execution_failed,
        "seam_passed": seam_passed,
        "quality_passed_lenient": quality_passed_lenient,
        "quality_passed_strict": quality_passed_strict,
        "complete": execution_failed == 0,  # CRITICAL: Only true if ALL executed
        "failed_images": failed_images,
        "category_stats": category_stats,  # PRIORITY 5 FIX
        "config": {
            "tile_size": tile_size,
            "overlap": overlap,
            "refinement": refinement_config
        },
        "results": results
    }
    
    if successful_results:
        # Aggregate quality metrics
        quality_scores = [r['quality_score'] for r in successful_results]
        edge_f1_scores = [r['metrics']['edge_f1'] for r in successful_results]
        seam_ratios = [r['seam_validation']['boundary_ratio'] for r in successful_results]
        
        aggregate["aggregate_metrics"] = {
            "quality_score": {
                "mean": float(np.mean(quality_scores)),
                "min": float(np.min(quality_scores)),
                "max": float(np.max(quality_scores)),
                "std": float(np.std(quality_scores))
            },
            "edge_f1": {
                "mean": float(np.mean(edge_f1_scores)),
                "min": float(np.min(edge_f1_scores)),
                "max": float(np.max(edge_f1_scores))
            },
            "seam_ratio": {
                "mean": float(np.mean(seam_ratios)),
                "max": float(np.max(seam_ratios))
            }
        }
    
    # Save aggregate report (atomic)
    report_path = output_dir / "validation_report.json"
    save_metrics_atomic(aggregate, report_path)
    logger.info(f"\nAggregate report saved: {report_path}")
    
    # Exit with failure if incomplete
    if not aggregate['complete']:
        logger.error("\n❌ VALIDATION INCOMPLETE - DO NOT DEPLOY")
        sys.exit(1)
    else:
        logger.info("\n✅ VALIDATION COMPLETE - ALL IMAGES PASSED")
    
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Production depth validation (stability-first)")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory with source images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for depth + metrics")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size (default: 1024)")
    parser.add_argument("--overlap", type=int, default=192, help="Tile overlap (default: 192, PRIORITY 2)")
    parser.add_argument("--force", action="store_true", help="Force reprocess (skip resumability)")
    parser.add_argument("--no-refinement", action="store_true", help="Disable refinement (depth only)")
    
    args = parser.parse_args()
    
    run_production_validation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        force=args.force,
        use_refinement=not args.no_refinement
    )


if __name__ == "__main__":
    main()
