#!/usr/bin/env python3
"""
Phase 2 Integration Example
============================
Demonstrates integration of all Phase 2 features in a typical processing pipeline.
"""

from pathlib import Path
import numpy as np
from PIL import Image

# Phase 2 imports
from tools.material_detector import MaterialDetector, MaterialType
from tools.depth_aware_lut import (
    DepthAwareLUT, DepthAwareLUTConfig, ZoneLUTConfig, DepthZone
)
from utils.performance_profiler import PerformanceProfiler
from utils.exposure_fusion import ExposureFusion, ExposureTarget


def process_luxury_image_phase2(
    image_path: Path,
    output_dir: Path,
    lut_fg: Path,
    lut_bg: Path,
    enable_profiling: bool = True
):
    """
    Process luxury real estate image with Phase 2 features.
    
    Args:
        image_path: Input image path
        output_dir: Output directory
        lut_fg: Foreground LUT path
        lut_bg: Background LUT path
        enable_profiling: Enable performance profiling
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize profiler
    profiler = None
    if enable_profiling:
        profiler = PerformanceProfiler(session_id=f"phase2_{image_path.stem}")
    
    print(f"\n{'='*70}")
    print(f"Phase 2 Processing Pipeline")
    print(f"{'='*70}\n")
    print(f"Input: {image_path}")
    
    # Load image
    if profiler:
        with profiler.stage('image_loading', items=1):
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
    else:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
    
    print(f"✓ Loaded: {img.size[0]}x{img.size[1]}")
    
    # Step 1: Material Detection with Confidence Scores
    print(f"\n{'Material Detection':-^70}")
    
    if profiler:
        with profiler.stage('material_detection', items=1):
            detector = MaterialDetector(min_confidence=0.3)
            material_result = detector.detect(image_path)
    else:
        detector = MaterialDetector(min_confidence=0.3)
        material_result = detector.detect(image_path)
    
    print(f"Dominant Material: {material_result.dominant_material.value.upper()}")
    
    for material_type, confidence in sorted(
        material_result.materials.items(),
        key=lambda x: x[1].percentage,
        reverse=True
    )[:3]:
        print(
            f"  • {material_type.value:<10} "
            f"{confidence.percentage:>5.1f}% coverage, "
            f"{confidence.mean_confidence:.3f} confidence"
        )
    
    # Save material report
    detector.generate_report(
        material_result,
        output_dir / f"{image_path.stem}_materials.json"
    )
    
    # Generate heatmap for dominant material
    detector.generate_heatmap(
        material_result,
        material_result.dominant_material,
        output_dir / f"{image_path.stem}_material_heatmap.png"
    )
    
    # Step 2: Depth-Aware LUT Application
    print(f"\n{'Depth-Aware Color Grading':-^70}")
    
    # Create simple depth map (fallback - in production use Depth Anything V2)
    h, w = img_array.shape[:2]
    depth_map = np.linspace(0.3, 1.0, h)[:, np.newaxis]
    depth_map = np.repeat(depth_map, w, axis=1)
    
    # Configure depth-aware LUT
    if lut_fg.exists() and lut_bg.exists():
        config = DepthAwareLUTConfig(
            zone_configs={
                DepthZone.FOREGROUND: ZoneLUTConfig(
                    zone=DepthZone.FOREGROUND,
                    lut_path=lut_fg,
                    strength=0.8,
                    color_temp_shift=0
                ),
                DepthZone.BACKGROUND: ZoneLUTConfig(
                    zone=DepthZone.BACKGROUND,
                    lut_path=lut_bg,
                    strength=0.6,
                    color_temp_shift=200  # Warmer background
                )
            },
            atmospheric_strength=0.3,
            depth_falloff=2.0
        )
        
        if profiler:
            with profiler.stage('depth_aware_lut', items=1):
                lut_processor = DepthAwareLUT(config)
                graded = lut_processor.apply(img_array, depth_map)
        else:
            lut_processor = DepthAwareLUT(config)
            graded = lut_processor.apply(img_array, depth_map)
        
        print(f"✓ Applied depth-aware LUT with atmospheric perspective")
        
        # Save graded image
        graded_img = Image.fromarray((graded * 255).astype(np.uint8))
        graded_img.save(output_dir / f"{image_path.stem}_graded.png", quality=95)
    else:
        print(f"⚠ LUT files not found, skipping depth-aware grading")
        graded = img_array
    
    # Step 3: Multi-Exposure Fusion (if HDR input)
    print(f"\n{'Exposure Optimization':-^70}")
    
    # Check if input is HDR (simplified check)
    is_hdr = img_array.max() > 1.0 or image_path.suffix.lower() in ['.tif', '.tiff', '.exr']
    
    if is_hdr:
        if profiler:
            with profiler.stage('exposure_fusion', items=3):
                fusion = ExposureFusion()
                variants = fusion.generate_variants(graded * 5.0)  # Scale to HDR range
        else:
            fusion = ExposureFusion()
            variants = fusion.generate_variants(graded * 5.0)
        
        print(f"Generated {len(variants)} exposure-optimized variants:")
        
        for variant in variants:
            if variant.target in [ExposureTarget.WEB, ExposureTarget.PRINT]:
                variant_path = output_dir / f"{image_path.stem}_{variant.target.value}.png"
                variant_img = Image.fromarray((variant.image * 255).astype(np.uint8))
                variant_img.save(variant_path, quality=95)
                print(f"  • {variant.target.value:8s} (EV {variant.exposure_ev:+.1f}): {variant_path.name}")
    else:
        print(f"Standard dynamic range input, saving single output")
        output_path = output_dir / f"{image_path.stem}_processed.png"
        output_img = Image.fromarray((graded * 255).astype(np.uint8))
        output_img.save(output_path, quality=95)
    
    # Step 4: Performance Report
    if profiler:
        print(f"\n{'Performance Analysis':-^70}")
        
        report = profiler.generate_report()
        profiler.print_report(report)
        
        # Save detailed report
        profiler.save_report(report, output_dir / 'performance_report.json')
    
    print(f"\n{'='*70}")
    print(f"Processing Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    """CLI for Phase 2 integration example."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 2 Integration Example - Luxury Image Processing"
    )
    parser.add_argument('input', type=Path, help='Input image path')
    parser.add_argument('--output-dir', type=Path, default=Path('output_phase2_example'),
                       help='Output directory')
    parser.add_argument('--lut-fg', type=Path,
                       default=Path('assets/luts/film_emulation/Kodak_2383.cube'),
                       help='Foreground LUT path')
    parser.add_argument('--lut-bg', type=Path,
                       default=Path('assets/luts/film_emulation/Kodak_2393.cube'),
                       help='Background LUT path')
    parser.add_argument('--no-profiling', action='store_true',
                       help='Disable performance profiling')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    process_luxury_image_phase2(
        args.input,
        args.output_dir,
        args.lut_fg,
        args.lut_bg,
        enable_profiling=not args.no_profiling
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
