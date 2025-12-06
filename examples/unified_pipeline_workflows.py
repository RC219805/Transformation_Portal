#!/usr/bin/env python3
"""
Unified Pipeline Workflow Examples
===================================

Demonstrates production workflows combining upscaling, depth processing,
and luxury enhancements for various real estate scenarios.
"""

import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_single_photo_realistic():
    """Example 1: Single image with maximum quality (photo-realistic)."""
    from unified_luxury_pipeline import (
        UnifiedLuxuryPipeline,
        UnifiedPipelineConfig,
        PipelinePreset
    )
    
    print("\n" + "="*70)
    print("Example 1: Photo-Realistic Single Image Processing")
    print("="*70)
    
    # Configure for maximum quality
    config = UnifiedPipelineConfig(
        input_path=Path("input_images/sample_interior.tif"),
        output_dir=Path("output_examples/photo_realistic/"),
        preset=PipelinePreset.PHOTO_REALISTIC
    )
    
    # Process
    pipeline = UnifiedLuxuryPipeline(config)
    
    if config.input_path.exists():
        result = pipeline.process_image(config.input_path)
        print(f"\n{result.summary()}")
    else:
        print(f"⚠️  Sample image not found: {config.input_path}")


def example_2_architectural_batch():
    """Example 2: Batch process architectural renders."""
    from unified_luxury_pipeline import (
        UnifiedLuxuryPipeline,
        UnifiedPipelineConfig,
        PipelinePreset
    )
    
    print("\n" + "="*70)
    print("Example 2: Architectural Batch Processing")
    print("="*70)
    
    # Configure for architectural rendering
    config = UnifiedPipelineConfig(
        input_path=Path("input_images/"),
        output_dir=Path("output_examples/architectural_batch/"),
        preset=PipelinePreset.ARCHITECTURAL
    )
    
    pipeline = UnifiedLuxuryPipeline(config)
    
    # Gather images
    input_dir = Path("input_images")
    if input_dir.exists():
        input_paths = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.jpg"))
        
        if input_paths:
            print(f"\nProcessing {len(input_paths)} images...")
            
            # Progress callback
            def progress(current, total, filename):
                print(f"  [{current}/{total}] {filename}")
            
            results = pipeline.batch_process(input_paths, progress_callback=progress)
            
            print(f"\n✓ Processed {len(results)} images")
            print(f"  Output: {config.output_dir}")
        else:
            print("⚠️  No images found in input_images/")
    else:
        print("⚠️  input_images/ directory not found")


def example_3_signature_estate_showcase():
    """Example 3: Luxury estate showcase with full enhancement suite."""
    from unified_luxury_pipeline import (
        UnifiedLuxuryPipeline,
        UnifiedPipelineConfig,
        PipelinePreset,
        UpscalingModel
    )
    
    print("\n" + "="*70)
    print("Example 3: Signature Estate Showcase")
    print("="*70)
    
    # Configure for luxury estate marketing
    config = UnifiedPipelineConfig(
        input_path=Path("input_images/estate_exterior.tif"),
        output_dir=Path("output_examples/signature_estate/"),
        preset=PipelinePreset.SIGNATURE_ESTATE,
        
        # Override for maximum quality
        upscale_model=UpscalingModel.SWINIR_REAL_4X,
        material_strength=0.90,  # Maximum material response
        saturation_boost=1.12,   # Enhanced vibrancy
        
        # Quality validation
        validate_colors=True,
        color_tolerance=0.015,   # Strict
        
        # Save intermediate stages for review
        save_intermediate=True
    )
    
    pipeline = UnifiedLuxuryPipeline(config)
    
    if config.input_path.exists():
        result = pipeline.process_image(config.input_path)
        
        print(f"\n{'='*70}")
        print("Signature Estate Processing Complete")
        print(f"{'='*70}")
        print(f"Input:  {result.input_path.name}")
        print(f"Output: {result.output_path.name}")
        print(f"Time:   {result.processing_time:.1f}s")
        print(f"Size:   {result.final_size[0]}x{result.final_size[1]}")
        print(f"File:   {result.file_size_mb:.1f} MB")
        
        if result.upscaling_metrics:
            print(f"\nUpscaling:")
            print(f"  Model: {result.upscaling_metrics.model_name}")
            print(f"  Color Deviation: {result.upscaling_metrics.color_deviation:.4f}")
    else:
        print(f"⚠️  Sample image not found: {config.input_path}")


def example_4_fast_batch_preview():
    """Example 4: Fast batch processing for previews (speed-optimized)."""
    from unified_luxury_pipeline import (
        UnifiedLuxuryPipeline,
        UnifiedPipelineConfig,
        PipelinePreset
    )
    
    print("\n" + "="*70)
    print("Example 4: Fast Batch Preview Generation")
    print("="*70)
    
    # Configure for speed
    config = UnifiedPipelineConfig(
        input_path=Path("input_images/"),
        output_dir=Path("output_examples/fast_previews/"),
        preset=PipelinePreset.FAST_BATCH,
        
        # Speed optimizations
        cache_models=True,      # Critical for batch speed
        validate_colors=False,  # Skip validation
        generate_report=True    # Still generate report
    )
    
    pipeline = UnifiedLuxuryPipeline(config)
    
    input_dir = Path("input_images")
    if input_dir.exists():
        # Process only JPEG files for speed test
        input_paths = list(input_dir.glob("*.jpg"))
        
        if input_paths:
            print(f"\nSpeed test: Processing {len(input_paths)} JPEG images")
            print("Expected throughput: ~450 images/hour\n")
            
            import time
            start_time = time.time()
            
            results = pipeline.batch_process(input_paths)
            
            elapsed = time.time() - start_time
            throughput = len(results) / (elapsed / 3600)
            
            print(f"\n{'='*70}")
            print(f"Speed Test Results:")
            print(f"  Images:     {len(results)}")
            print(f"  Time:       {elapsed:.1f}s ({elapsed/60:.1f}min)")
            print(f"  Throughput: {throughput:.0f} images/hour")
            print(f"  Avg:        {elapsed/len(results):.1f}s per image")
        else:
            print("⚠️  No JPEG images found for speed test")
    else:
        print("⚠️  input_images/ directory not found")


def example_5_custom_configuration():
    """Example 5: Custom configuration with selective stage control."""
    from unified_luxury_pipeline import (
        UnifiedLuxuryPipeline,
        UnifiedPipelineConfig,
        UpscalingModel
    )
    
    print("\n" + "="*70)
    print("Example 5: Custom Pipeline Configuration")
    print("="*70)
    
    # Create custom configuration (no preset)
    config = UnifiedPipelineConfig(
        input_path=Path("input_images/sample_room.jpg"),
        output_dir=Path("output_examples/custom/"),
        preset=PipelinePreset.PHOTO_REALISTIC,  # Start with base
        
        # Customize each stage
        enable_upscaling=True,
        enable_depth_processing=True,
        enable_material_response=True,
        enable_color_grading=False,  # Disable color grading
        
        # Upscaling: Use Real-ESRGAN for speed
        upscale_model=UpscalingModel.REALESRGAN_4X,
        tile_size=512,
        
        # Material response: Emphasize wood and glass
        material_strength=0.75,
        surface_types=["wood", "glass"],
        
        # Quality: 16-bit with strict validation
        preserve_16bit=True,
        validate_colors=True,
        color_tolerance=0.020,
        
        # Performance: Use Apple Neural Engine
        device="mps"
    )
    
    print("\nCustom Configuration:")
    print(f"  Upscaling: {config.upscale_model.value}")
    print(f"  Depth Processing: {'Enabled' if config.enable_depth_processing else 'Disabled'}")
    print(f"  Material Response: {config.material_strength * 100:.0f}%")
    print(f"  Color Grading: {'Enabled' if config.enable_color_grading else 'Disabled'}")
    print(f"  Device: {config.device}")
    
    pipeline = UnifiedLuxuryPipeline(config)
    
    if config.input_path.exists():
        result = pipeline.process_image(config.input_path)
        print(f"\n{result.summary()}")
    else:
        print(f"\n⚠️  Sample image not found: {config.input_path}")


def example_6_archival_quality():
    """Example 6: Museum-grade archival processing."""
    from unified_luxury_pipeline import (
        UnifiedLuxuryPipeline,
        UnifiedPipelineConfig,
        PipelinePreset
    )
    
    print("\n" + "="*70)
    print("Example 6: Archival Quality Processing")
    print("="*70)
    
    config = UnifiedPipelineConfig(
        input_path=Path("input_images/archival_scan.tif"),
        output_dir=Path("output_examples/archival/"),
        preset=PipelinePreset.ARCHIVAL_QUALITY,
        
        # Archival-specific overrides
        preserve_16bit=True,
        validate_colors=True,
        color_tolerance=0.010,  # Very strict (1%)
        save_intermediate=True,  # Keep all stages
        generate_report=True     # Detailed report
    )
    
    pipeline = UnifiedLuxuryPipeline(config)
    
    if config.input_path.exists():
        print("\nProcessing with archival-grade quality...")
        print("  - 16-bit precision preservation")
        print("  - Strict color validation (1% tolerance)")
        print("  - All intermediate stages saved")
        
        result = pipeline.process_image(config.input_path)
        
        # Validate archival quality
        print(f"\n{'='*70}")
        print("Archival Quality Validation")
        print(f"{'='*70}")
        
        quality_passed = True
        
        # Check bit depth
        if result.bit_depth == 16:
            print("✓ Bit depth: 16-bit (archival grade)")
        else:
            print("✗ Bit depth: 8-bit (not archival grade)")
            quality_passed = False
        
        # Check color consistency
        if result.color_deviation < 0.015:
            print(f"✓ Color deviation: {result.color_deviation:.4f} (excellent)")
        else:
            print(f"⚠️  Color deviation: {result.color_deviation:.4f} (review needed)")
            quality_passed = False
        
        # Check warnings
        if not result.warnings:
            print("✓ No processing warnings")
        else:
            print(f"⚠️  {len(result.warnings)} warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")
            quality_passed = False
        
        if quality_passed:
            print("\n✓ Archival quality criteria met")
        else:
            print("\n⚠️  Review quality issues before archival")
    else:
        print(f"⚠️  Sample image not found: {config.input_path}")


def example_7_interior_exterior_comparison():
    """Example 7: Compare interior vs exterior presets."""
    from unified_luxury_pipeline import (
        UnifiedLuxuryPipeline,
        UnifiedPipelineConfig,
        PipelinePreset
    )
    
    print("\n" + "="*70)
    print("Example 7: Interior vs Exterior Preset Comparison")
    print("="*70)
    
    test_cases = [
        ("input_images/interior.tif", PipelinePreset.INTERIOR_LUXURY),
        ("input_images/exterior.tif", PipelinePreset.EXTERIOR_SHOWCASE),
    ]
    
    for input_path, preset in test_cases:
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"\n⚠️  {input_path.name} not found - skipping")
            continue
        
        print(f"\n{'-'*70}")
        print(f"Processing: {input_path.name}")
        print(f"Preset: {preset.value}")
        print(f"{'-'*70}")
        
        config = UnifiedPipelineConfig(
            input_path=input_path,
            output_dir=Path(f"output_examples/{preset.value}/"),
            preset=preset
        )
        
        pipeline = UnifiedLuxuryPipeline(config)
        result = pipeline.process_image(input_path)
        
        print(f"\nResult:")
        print(f"  Time: {result.processing_time:.1f}s")
        print(f"  Size: {result.final_size[0]}x{result.final_size[1]}")
        print(f"  File: {result.file_size_mb:.1f} MB")
        
        if result.upscaling_metrics:
            print(f"  Color Dev: {result.upscaling_metrics.color_deviation:.4f}")


def main():
    """Run all examples."""
    examples = [
        ("Single Photo-Realistic", example_1_single_photo_realistic),
        ("Architectural Batch", example_2_architectural_batch),
        ("Signature Estate", example_3_signature_estate_showcase),
        ("Fast Batch Preview", example_4_fast_batch_preview),
        ("Custom Configuration", example_5_custom_configuration),
        ("Archival Quality", example_6_archival_quality),
        ("Interior/Exterior Comparison", example_7_interior_exterior_comparison),
    ]
    
    print("\n" + "="*70)
    print("Unified Pipeline Workflow Examples")
    print("="*70)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRun with: python examples/unified_pipeline_workflows.py")
    print("\nOr import individually:")
    print("  from examples.unified_pipeline_workflows import example_1_single_photo_realistic")
    print("  example_1_single_photo_realistic()")
    
    # Uncomment to run all examples
    # for name, func in examples:
    #     try:
    #         func()
    #     except Exception as e:
    #         logger.error(f"Example '{name}' failed: {e}")


if __name__ == "__main__":
    main()
