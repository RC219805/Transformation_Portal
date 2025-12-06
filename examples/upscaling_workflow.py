#!/usr/bin/env python3
"""
Advanced Upscaling Workflow Examples
=====================================

Demonstrates production-grade upscaling workflows for luxury real estate:
1. Single high-quality upscale (SwinIR)
2. Batch processing with progress tracking
3. Integration with depth pipeline
4. Color validation and quality metrics
5. Comparing multiple models
"""

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_single_upscale():
    """Example 1: Single image upscale with maximum quality."""
    from utils.upscaling_engine import UpscalingEngine, UpscalingConfig, UpscalingModel
    
    print("\n" + "="*60)
    print("Example 1: Single Image Upscale (Maximum Quality)")
    print("="*60)
    
    # Configure for best quality
    config = UpscalingConfig(
        model=UpscalingModel.SWINIR_REAL_4X,  # Best for photos
        preserve_16bit=True,
        validate_colors=True,
        device="auto"  # Auto-detect best device
    )
    
    engine = UpscalingEngine(config)
    
    # Upscale
    input_path = Path("input_images/sample_interior.tif")
    output_path = Path("output_images/sample_interior_4x.tif")
    
    if input_path.exists():
        upscaled, metrics = engine.upscale_image(input_path, output_path)
        
        print(f"\n✓ Upscale complete:")
        print(f"  Input:  {metrics.input_size[0]}x{metrics.input_size[1]}")
        print(f"  Output: {metrics.output_size[0]}x{metrics.output_size[1]}")
        print(f"  Time:   {metrics.processing_time:.2f}s")
        print(f"  Tiles:  {metrics.tiles_processed}")
        print(f"  Color deviation: {metrics.color_deviation:.4f}")
    else:
        print(f"⚠️  Input not found: {input_path}")


def example_batch_processing():
    """Example 2: Batch process directory with progress tracking."""
    from utils.upscaling_engine import UpscalingEngine, UpscalingConfig, UpscalingModel
    
    print("\n" + "="*60)
    print("Example 2: Batch Processing (20+ Images)")
    print("="*60)
    
    # Configure for batch efficiency
    config = UpscalingConfig(
        model=UpscalingModel.REALESRGAN_4X,  # Faster for large batches
        cache_model=True,  # Critical for batch performance
        preserve_16bit=True,
        validate_colors=True
    )
    
    engine = UpscalingEngine(config)
    
    # Gather input images
    input_dir = Path("input_images")
    input_paths = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.jpg"))
    
    if not input_paths:
        print(f"⚠️  No images found in {input_dir}")
        return
    
    output_dir = Path("output_images/batch_upscaled")
    
    # Progress callback
    def progress(current, total, filename):
        percent = (current / total) * 100
        print(f"  [{current}/{total}] ({percent:.1f}%) Processing: {filename}")
    
    # Batch process
    print(f"\nProcessing {len(input_paths)} images...")
    results = engine.batch_upscale(input_paths, output_dir, progress_callback=progress)
    
    # Summary statistics
    total_time = sum(m.processing_time for m in results.values())
    avg_time = total_time / len(results) if results else 0
    avg_color_dev = np.mean([m.color_deviation for m in results.values()])
    
    print(f"\n✓ Batch complete:")
    print(f"  Images:     {len(results)}/{len(input_paths)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time:   {avg_time:.2f}s per image")
    print(f"  Avg color deviation: {avg_color_dev:.4f}")
    print(f"  Throughput: {len(results) / (total_time / 3600):.1f} images/hour")


def example_depth_integration():
    """Example 3: Upscale then apply depth-aware enhancements."""
    from utils.upscaling_engine import UpscalingEngine, UpscalingConfig, UpscalingModel
    
    print("\n" + "="*60)
    print("Example 3: Depth-Integrated Upscaling")
    print("="*60)
    
    # Step 1: Upscale base image
    upscale_config = UpscalingConfig(
        model=UpscalingModel.SWINIR_REAL_4X,
        preserve_16bit=True
    )
    
    engine = UpscalingEngine(upscale_config)
    
    input_path = Path("input_images/exterior_view.tif")
    upscaled_path = Path("output_images/temp_upscaled.tif")
    
    if not input_path.exists():
        print(f"⚠️  Input not found: {input_path}")
        return
    
    print("\n1. Upscaling to 4x resolution...")
    upscaled, metrics = engine.upscale_image(input_path, upscaled_path)
    print(f"   ✓ {metrics.input_size} → {metrics.output_size} in {metrics.processing_time:.2f}s")
    
    # Step 2: Generate depth map at upscaled resolution
    try:
        from depth_pipeline.pipeline import ArchitecturalDepthPipeline
        
        print("\n2. Generating depth map at 4x resolution...")
        depth_config = Path("config/exterior_preset.yaml")
        
        if depth_config.exists():
            pipeline = ArchitecturalDepthPipeline.from_config(depth_config)
            depth_result = pipeline.process_render(upscaled_path)
            
            output_path = Path("output_images/exterior_view_ultimate.tif")
            pipeline.save_result(depth_result, output_path)
            
            print(f"   ✓ Depth-aware processing complete")
            print(f"   Final output: {output_path}")
        else:
            print(f"   ⚠️  Depth config not found: {depth_config}")
    
    except ImportError:
        print("   ⚠️  Depth pipeline not available")


def example_model_comparison():
    """Example 4: Compare multiple upscaling models."""
    from utils.upscaling_engine import UpscalingEngine, UpscalingConfig, UpscalingModel
    
    print("\n" + "="*60)
    print("Example 4: Model Comparison (Quality vs Speed)")
    print("="*60)
    
    input_path = Path("input_images/sample_room.jpg")
    
    if not input_path.exists():
        print(f"⚠️  Input not found: {input_path}")
        return
    
    models_to_test = [
        (UpscalingModel.REALESRGAN_4X, "Real-ESRGAN 4x (Fast)"),
        (UpscalingModel.REALESRGAN_GENERAL_4X, "Real-ESRGAN General (Robust)"),
        (UpscalingModel.SWINIR_REAL_4X, "SwinIR Real 4x (Highest Quality)"),
    ]
    
    results = {}
    
    for model, description in models_to_test:
        print(f"\nTesting: {description}")
        
        config = UpscalingConfig(
            model=model,
            preserve_16bit=True,
            validate_colors=True,
            cache_model=False  # Don't cache for fair comparison
        )
        
        engine = UpscalingEngine(config)
        
        output_path = Path(f"output_images/comparison_{model.value}.tif")
        upscaled, metrics = engine.upscale_image(input_path, output_path)
        
        results[description] = metrics
        
        print(f"  Time:            {metrics.processing_time:.2f}s")
        print(f"  Tiles:           {metrics.tiles_processed}")
        print(f"  Color deviation: {metrics.color_deviation:.4f}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("Comparison Summary:")
    print("="*60)
    print(f"{'Model':<35} {'Time':<8} {'Color Dev':<12} {'Speed':<10}")
    print("-" * 70)
    
    for desc, metrics in results.items():
        speed = "★★★★★" if metrics.processing_time < 10 else "★★★" if metrics.processing_time < 20 else "★"
        print(f"{desc:<35} {metrics.processing_time:>6.1f}s  {metrics.color_deviation:>10.4f}  {speed:<10}")
    
    print("\nRecommendation:")
    fastest = min(results.items(), key=lambda x: x[1].processing_time)
    best_color = min(results.items(), key=lambda x: x[1].color_deviation)
    print(f"  Fastest:      {fastest[0]}")
    print(f"  Best quality: {best_color[0]}")


def example_quality_validation():
    """Example 5: Validate output quality with detailed metrics."""
    from utils.upscaling_engine import UpscalingEngine, UpscalingConfig, UpscalingModel
    from PIL import Image
    import numpy as np
    
    print("\n" + "="*60)
    print("Example 5: Quality Validation & Metrics")
    print("="*60)
    
    input_path = Path("input_images/archival_scan.tif")
    output_path = Path("output_images/archival_scan_4x_validated.tif")
    
    if not input_path.exists():
        print(f"⚠️  Input not found: {input_path}")
        return
    
    # Load original for comparison
    original = np.array(Image.open(input_path))
    
    # Upscale with validation
    config = UpscalingConfig(
        model=UpscalingModel.SWINIR_REAL_4X,
        preserve_16bit=True,
        validate_colors=True,
        color_tolerance=0.015  # Strict tolerance for archival
    )
    
    engine = UpscalingEngine(config)
    
    print("\nProcessing with strict quality validation...")
    upscaled, metrics = engine.upscale_image(input_path, output_path)
    
    # Additional quality checks
    print("\n" + "-"*60)
    print("Quality Metrics:")
    print("-"*60)
    
    # 1. Color consistency
    color_ok = metrics.color_deviation < config.color_tolerance
    print(f"Color consistency: {'✓ PASS' if color_ok else '✗ FAIL'}")
    print(f"  Deviation: {metrics.color_deviation:.4f} (tolerance: {config.color_tolerance})")
    
    # 2. Bit depth preservation
    if output_path.exists():
        output_arr = np.array(Image.open(output_path))
        bit_depth = 16 if output_arr.dtype == np.uint16 else 8
        bit_depth_ok = bit_depth == 16 if config.preserve_16bit else True
        print(f"Bit depth:         {'✓ PASS' if bit_depth_ok else '✗ FAIL'}")
        print(f"  Output: {bit_depth}-bit (expected: {16 if config.preserve_16bit else 8}-bit)")
        
        # 3. Gradient smoothness (check for banding)
        if len(output_arr.shape) == 3:
            # Check blue channel gradients
            blue = output_arr[:, :, 2]
            gradient = np.diff(blue.astype(np.float32), axis=0)
            smoothness = np.std(gradient)
            print(f"Gradient smoothness: {smoothness:.2f} (lower is better)")
    
    # 4. Processing efficiency
    pixels_per_sec = (metrics.output_size[0] * metrics.output_size[1]) / metrics.processing_time
    mpixels_per_sec = pixels_per_sec / 1e6
    print(f"Processing speed:  {mpixels_per_sec:.1f} MP/s")
    
    print("\n" + "="*60)
    if color_ok and bit_depth_ok:
        print("✓ All quality checks passed - Archival grade achieved")
    else:
        print("⚠️  Quality issues detected - Review output")


def main():
    """Run all examples."""
    examples = [
        ("Single Upscale", example_single_upscale),
        ("Batch Processing", example_batch_processing),
        ("Depth Integration", example_depth_integration),
        ("Model Comparison", example_model_comparison),
        ("Quality Validation", example_quality_validation),
    ]
    
    print("\n" + "="*60)
    print("Advanced Upscaling Workflow Examples")
    print("="*60)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRun individual examples with:")
    print("  python examples/upscaling_workflow.py")
    print("\nOr import functions:")
    print("  from examples.upscaling_workflow import example_single_upscale")
    print("  example_single_upscale()")
    
    # Uncomment to run all examples
    # for name, func in examples:
    #     try:
    #         func()
    #     except Exception as e:
    #         logger.error(f"Example '{name}' failed: {e}")


if __name__ == "__main__":
    main()
