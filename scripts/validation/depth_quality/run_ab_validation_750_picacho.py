#!/usr/bin/env python3
"""
A/B Validation: High-Fidelity Depth Pipeline on 750 Picacho Luxury Interiors
============================================================================

Validates improvements from:
- Baseline: HF pipeline (518px resize)
- Enhanced: Tiled inference + global anchor + edge snapping

Target metrics:
- Edge alignment: 0.1 → 0.4-0.7 (4-7x improvement)
- Edge sharpness: Sharper furniture/window boundaries
- Processing time: 2-5s → 10-30s (acceptable for quality)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def downsample_if_needed(image: Image.Image, max_dimension: int = 4000) -> Image.Image:
    """Downsample image to max dimension while preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_dimension:
        return image
    
    scale = max_dimension / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info(f"  Downsampling {w}x{h} → {new_w}x{new_h} (max_dim={max_dimension})")
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def run_single_comparison(
    image_path: Path,
    output_dir: Path,
    max_dimension: int = 4000
) -> Dict:
    """Run A/B comparison on a single image."""
    from lux_depth_v2.tools.ab_comparison import run_ab_comparison
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {image_path.name}")
    logger.info(f"{'='*60}")
    
    # Load and downsample
    logger.info(f"Loading {image_path}...")
    image = Image.open(image_path)
    original_size = image.size
    image = downsample_if_needed(image, max_dimension)
    rgb = np.array(image)
    
    logger.info(f"  Original: {original_size[0]}x{original_size[1]}")
    logger.info(f"  Processing: {rgb.shape[1]}x{rgb.shape[0]}")
    
    # Create output directory for this image
    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    try:
        result = run_ab_comparison(rgb, image_output_dir)
        
        # Convert to dict for JSON serialization
        result_dict = {
            "image_name": image_path.name,
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "processed_size": f"{rgb.shape[1]}x{rgb.shape[0]}",
            "baseline": {
                "edge_alignment": float(result.baseline_edge_alignment),
                "edge_sharpness": float(result.baseline_edge_sharpness),
                "time_ms": float(result.baseline_time_ms)
            },
            "enhanced": {
                "edge_alignment": float(result.enhanced_edge_alignment),
                "edge_sharpness": float(result.enhanced_edge_sharpness),
                "time_ms": float(result.enhanced_time_ms)
            },
            "improvements": {
                "edge_alignment_improvement_pct": float(result.edge_alignment_improvement * 100),
                "edge_sharpness_improvement_pct": float(result.edge_sharpness_improvement * 100),
                "time_overhead_factor": float(result.time_overhead_factor)
            },
            "verdict": "IMPROVEMENT" if result.edge_alignment_improvement > 0 else "NO_IMPROVEMENT"
        }
        
        logger.info(f"✓ Comparison complete: {image_path.name}")
        return result_dict
        
    except Exception as e:
        logger.error(f"✗ Failed to process {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "image_name": image_path.name,
            "error": str(e),
            "verdict": "FAILED"
        }


def main():
    """Run A/B validation on 750 Picacho luxury interiors."""
    
    # Configuration
    input_dir = Path("/Users/rc/Transformation_Portal/input_images/750_Picacho/Source_TIFFs")
    output_dir = Path("/Users/rc/Transformation_Portal/outputs/ab_validation_750_Picacho")
    
    # Select representative images
    selected_images = [
        "V2_750Picacho_GreatRoom.tiff",      # Already 4K, varied content
        "V2_750Picacho_PrimaryBedroom.tiff", # 6K, windows and furniture
        "750Picacho_Kitchen_16bit.tiff"      # 12K, complex scene (will downsample)
    ]
    
    logger.info("="*80)
    logger.info("A/B VALIDATION: 750 Picacho Luxury Interiors")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Selected images: {len(selected_images)}")
    logger.info("")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparisons
    results = []
    start_time = time.time()
    
    for image_name in selected_images:
        image_path = input_dir / image_name
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        result = run_single_comparison(image_path, output_dir, max_dimension=4000)
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Aggregate statistics
    successful_results = [r for r in results if r.get("verdict") != "FAILED"]
    
    if successful_results:
        avg_baseline_edge_align = np.mean([r["baseline"]["edge_alignment"] for r in successful_results])
        avg_enhanced_edge_align = np.mean([r["enhanced"]["edge_alignment"] for r in successful_results])
        avg_edge_align_improvement = np.mean([r["improvements"]["edge_alignment_improvement_pct"] for r in successful_results])
        
        avg_baseline_edge_sharp = np.mean([r["baseline"]["edge_sharpness"] for r in successful_results])
        avg_enhanced_edge_sharp = np.mean([r["enhanced"]["edge_sharpness"] for r in successful_results])
        avg_edge_sharp_improvement = np.mean([r["improvements"]["edge_sharpness_improvement_pct"] for r in successful_results])
        
        avg_baseline_time = np.mean([r["baseline"]["time_ms"] for r in successful_results])
        avg_enhanced_time = np.mean([r["enhanced"]["time_ms"] for r in successful_results])
        avg_time_overhead = np.mean([r["improvements"]["time_overhead_factor"] for r in successful_results])
        
        summary = {
            "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images_processed": len(successful_results),
            "total_time_seconds": total_time,
            "aggregate_metrics": {
                "baseline": {
                    "avg_edge_alignment": avg_baseline_edge_align,
                    "avg_edge_sharpness": avg_baseline_edge_sharp,
                    "avg_time_ms": avg_baseline_time
                },
                "enhanced": {
                    "avg_edge_alignment": avg_enhanced_edge_align,
                    "avg_edge_sharpness": avg_enhanced_edge_sharp,
                    "avg_time_ms": avg_enhanced_time
                },
                "improvements": {
                    "avg_edge_alignment_improvement_pct": avg_edge_align_improvement,
                    "avg_edge_sharpness_improvement_pct": avg_edge_sharp_improvement,
                    "avg_time_overhead_factor": avg_time_overhead
                }
            },
            "individual_results": results
        }
    else:
        summary = {
            "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images_processed": 0,
            "total_time_seconds": total_time,
            "error": "All comparisons failed",
            "individual_results": results
        }
    
    # Save summary
    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_path}")
    
    # Print executive summary
    if successful_results:
        print(f"\n{'='*80}")
        print("EXECUTIVE SUMMARY")
        print(f"{'='*80}")
        print(f"Images processed: {len(successful_results)}/{len(selected_images)}")
        print(f"Total processing time: {total_time:.1f}s")
        print()
        print("Average Metrics:")
        print(f"  Baseline Edge Alignment: {avg_baseline_edge_align:.3f}")
        print(f"  Enhanced Edge Alignment: {avg_enhanced_edge_align:.3f} ({avg_edge_align_improvement:+.1f}%)")
        print()
        print(f"  Baseline Edge Sharpness: {avg_baseline_edge_sharp:.1f}")
        print(f"  Enhanced Edge Sharpness: {avg_enhanced_edge_sharp:.1f} ({avg_edge_sharp_improvement:+.1f}%)")
        print()
        print(f"  Processing Time Overhead: {avg_time_overhead:.1f}x")
        print()
        
        # Verdict
        if avg_edge_align_improvement > 200:  # >200% improvement (3x)
            verdict = "✓ SIGNIFICANT IMPROVEMENT - Ready for production"
        elif avg_edge_align_improvement > 50:  # >50% improvement
            verdict = "✓ MODERATE IMPROVEMENT - Consider deployment"
        elif avg_edge_align_improvement > 0:
            verdict = "⚠ MINOR IMPROVEMENT - Further optimization needed"
        else:
            verdict = "✗ NO IMPROVEMENT - Do not deploy"
        
        print(f"Verdict: {verdict}")
        print(f"{'='*80}\n")
    else:
        print(f"\n✗ All comparisons failed. Check logs for details.\n")
    
    return summary


if __name__ == "__main__":
    summary = main()
