#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Luxury Pipeline - Usage Examples
========================================

Comprehensive examples demonstrating the unified luxury pipeline capabilities.
"""

from pathlib import Path

from transformation_portal.pipelines.unified_luxury_pipeline import (
    OutputFormat,
    ProcessingProfile,
    SceneType,
    UnifiedLuxuryPipeline,
    UnifiedPipelineConfig,
    batch_process_luxury_renders,
    process_luxury_render,
)


def example_basic_usage():
    """Example 1: Basic single image processing with defaults."""
    print("\n" + "="*70)
    print("Example 1: Basic Single Image Processing")
    print("="*70)

    # Process with balanced profile (default)
    outputs = process_luxury_render(
        Path("input/kitchen.jpg"),
        output_dir=Path("output/example1"),
        profile=ProcessingProfile.BALANCED
    )

    print(f"Generated {len(outputs)} outputs:")
    for fmt, path in outputs.items():
        print(f"  {fmt}: {path}")


def example_premium_quality():
    """Example 2: Premium quality processing for hero shots."""
    print("\n" + "="*70)
    print("Example 2: Premium Quality for Hero Shot")
    print("="*70)

    config = UnifiedPipelineConfig(
        scene_type=SceneType.INTERIOR,
        profile=ProcessingProfile.PREMIUM,
        output_dir=Path("output/hero_shot"),

        # Select specific output formats
        output_formats=[
            OutputFormat.MASTER_TIFF,
            OutputFormat.PRINT_8K,
            OutputFormat.WEB_4K
        ],

        # Enable all enhancements
        enable_depth=True,
        enable_material_response=True,
        enable_vfx=False,  # Optional VFX
        enable_color_grading=True,

        # Fine-tune parameters
        exposure=0.15,
        contrast=1.10,
        saturation=1.05,
        clarity=0.18,

        # Save intermediate stages for review
        save_intermediates=True
    )

    pipeline = UnifiedLuxuryPipeline(config)
    outputs = pipeline.process(Path("input/greatroom.exr"))

    print(f"Generated {len(outputs)} premium outputs")
    print(f"Total processing time: {pipeline.stats.total_time:.2f}s")

    # Save statistics
    stats_path = pipeline.save_stats()
    print(f"Statistics saved to: {stats_path}")


def example_performance_mode():
    """Example 3: Fast processing for quick reviews."""
    print("\n" + "="*70)
    print("Example 3: Performance Mode for Quick Review")
    print("="*70)

    config = UnifiedPipelineConfig(
        profile=ProcessingProfile.PERFORMANCE,
        output_dir=Path("output/quick_review"),

        # Minimal outputs for speed
        output_formats=[OutputFormat.WEB_4K],

        # Disable heavy processing
        enable_depth=False,
        enable_material_response=False,
        enable_vfx=False,

        # Basic color grading only
        enable_color_grading=True,
        exposure=0.1,
        contrast=1.05
    )

    pipeline = UnifiedLuxuryPipeline(config)
    outputs = pipeline.process(Path("input/preview.jpg"))

    print(f"Fast preview generated in {pipeline.stats.total_time:.2f}s")


def example_batch_processing():
    """Example 4: Batch process entire folder."""
    print("\n" + "="*70)
    print("Example 4: Batch Processing")
    print("="*70)

    # Process all images in directory
    results = batch_process_luxury_renders(
        input_dir=Path("input/renders"),
        output_dir=Path("output/batch"),
        profile=ProcessingProfile.BALANCED
    )

    print(f"Processed {len(results)} images:")
    for input_path, outputs in results.items():
        if outputs:
            print(f"  ✓ {input_path.name}: {len(outputs)} outputs")
        else:
            print(f"  ✗ {input_path.name}: failed")


def example_custom_scene_optimization():
    """Example 5: Scene-specific optimization."""
    print("\n" + "="*70)
    print("Example 5: Scene-Specific Optimization")
    print("="*70)

    # Interior scene
    interior_config = UnifiedPipelineConfig(
        scene_type=SceneType.INTERIOR,
        profile=ProcessingProfile.PREMIUM,
        output_dir=Path("output/interior"),
        clarity=0.15,  # Higher clarity for interiors
        contrast=1.12
    )

    # Exterior scene
    exterior_config = UnifiedPipelineConfig(
        scene_type=SceneType.EXTERIOR,
        profile=ProcessingProfile.PREMIUM,
        output_dir=Path("output/exterior"),
        saturation=1.08,  # Higher saturation for exteriors
        enable_vfx=True   # Atmospheric effects
    )

    # Aerial scene
    aerial_config = UnifiedPipelineConfig(
        scene_type=SceneType.AERIAL,
        profile=ProcessingProfile.PREMIUM,
        output_dir=Path("output/aerial"),
        clarity=0.20,  # Maximum clarity for aerials
        saturation=1.10
    )

    print("Configured 3 scene-specific pipelines")


def example_with_lut():
    """Example 6: Processing with custom LUT."""
    print("\n" + "="*70)
    print("Example 6: Processing with Custom LUT")
    print("="*70)

    config = UnifiedPipelineConfig(
        profile=ProcessingProfile.BALANCED,
        output_dir=Path("output/lut_graded"),

        # Apply custom LUT
        lut_path=Path("assets/luts/film_emulation/Kodak_2393.cube"),
        lut_strength=0.7,

        # Additional grading
        exposure=0.05,
        contrast=1.08,
        saturation=1.05
    )

    pipeline = UnifiedLuxuryPipeline(config)
    outputs = pipeline.process(Path("input/exterior.jpg"))

    print("LUT-graded output generated")


def example_runtime_overrides():
    """Example 7: Runtime parameter overrides."""
    print("\n" + "="*70)
    print("Example 7: Runtime Parameter Overrides")
    print("="*70)

    # Create base configuration
    config = UnifiedPipelineConfig(
        profile=ProcessingProfile.BALANCED,
        output_dir=Path("output/overrides")
    )

    pipeline = UnifiedLuxuryPipeline(config)

    # Process first image with base config
    outputs1 = pipeline.process(Path("input/image1.jpg"))

    # Process second image with overrides
    outputs2 = pipeline.process(
        Path("input/image2.jpg"),
        exposure=0.3,      # Override exposure
        saturation=1.15,   # Override saturation
        enable_vfx=True    # Enable VFX for this image only
    )

    print("Processed 2 images with different parameters")


def example_parallel_output_generation():
    """Example 8: Parallel output format generation."""
    print("\n" + "="*70)
    print("Example 8: Parallel Output Generation")
    print("="*70)

    config = UnifiedPipelineConfig(
        output_dir=Path("output/parallel"),
        output_formats=list(OutputFormat),  # All 5 formats
        parallel_outputs=True,  # Generate in parallel (faster)
        enable_depth=False,
        enable_material_response=False
    )

    pipeline = UnifiedLuxuryPipeline(config)
    outputs = pipeline.process(Path("input/large_image.tiff"))

    print(f"Generated {len(outputs)} formats in parallel")


def example_statistics_tracking():
    """Example 9: Detailed statistics tracking."""
    print("\n" + "="*70)
    print("Example 9: Statistics Tracking")
    print("="*70)

    config = UnifiedPipelineConfig(
        profile=ProcessingProfile.BALANCED,
        output_dir=Path("output/stats_demo")
    )

    pipeline = UnifiedLuxuryPipeline(config)

    # Process multiple images
    for i in range(3):
        pipeline.process(Path(f"input/image_{i}.jpg"))

    # Print detailed statistics
    print(pipeline.stats.summary())

    # Save to JSON
    stats_file = pipeline.save_stats()
    print(f"\nStatistics saved to: {stats_file}")


def example_social_media_workflow():
    """Example 10: Social media optimized workflow."""
    print("\n" + "="*70)
    print("Example 10: Social Media Workflow")
    print("="*70)

    config = UnifiedPipelineConfig(
        profile=ProcessingProfile.BALANCED,
        output_dir=Path("output/social"),

        # Social media specific outputs
        output_formats=[
            OutputFormat.SOCIAL,      # 1080p for Instagram
            OutputFormat.WEB_4K       # 4K for website
        ],

        # Optimize for social media aesthetics
        saturation=1.12,  # Slightly boosted saturation
        contrast=1.10,    # Punchy contrast
        clarity=0.15,     # Enhanced clarity

        enable_material_response=True,
        parallel_outputs=True
    )

    pipeline = UnifiedLuxuryPipeline(config)

    # Process and get social-optimized outputs
    outputs = pipeline.process(Path("input/lifestyle_shot.jpg"))

    print(f"Social media outputs:")
    print(f"  Instagram (1080p): {outputs.get('social')}")
    print(f"  Website (4K): {outputs.get('web')}")


def example_auto_scene_detection():
    """Example 11: Automatic scene type detection."""
    print("\n" + "="*70)
    print("Example 11: Automatic Scene Detection")
    print("="*70)

    config = UnifiedPipelineConfig(
        scene_type=SceneType.AUTO,  # Auto-detect scene type
        profile=ProcessingProfile.BALANCED,
        output_dir=Path("output/auto_detect")
    )

    pipeline = UnifiedLuxuryPipeline(config)

    # Pipeline will automatically detect if each image is interior/exterior/aerial
    # and optimize parameters accordingly
    images = [
        Path("input/kitchen.jpg"),      # Will detect as interior
        Path("input/facade.jpg"),       # Will detect as exterior
        Path("input/aerial_view.jpg")   # Will detect as aerial
    ]

    for image_path in images:
        outputs = pipeline.process(image_path)
        print(f"Processed {image_path.name} (auto-detected scene type)")


def example_print_production():
    """Example 12: High-end print production workflow."""
    print("\n" + "="*70)
    print("Example 12: Print Production Workflow")
    print("="*70)

    config = UnifiedPipelineConfig(
        profile=ProcessingProfile.PREMIUM,
        output_dir=Path("output/print_production"),

        # Print-specific outputs
        output_formats=[
            OutputFormat.MASTER_TIFF,  # 16-bit master archive
            OutputFormat.PRINT_8K      # 8K print JPEG
        ],

        # Print-optimized parameters
        contrast=1.08,     # Conservative contrast for print
        saturation=1.05,   # Slightly reduced saturation
        clarity=0.12,      # Controlled clarity

        enable_depth=True,
        enable_material_response=True,
        preserve_metadata=True,  # Preserve all EXIF/IPTC data
        save_intermediates=True  # Keep depth maps, etc.
    )

    pipeline = UnifiedLuxuryPipeline(config)
    outputs = pipeline.process(Path("input/architectural_hero.exr"))

    print("Print production files generated:")
    print(f"  Master archive: {outputs.get('master')}")
    print(f"  Print file: {outputs.get('print')}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIFIED LUXURY PIPELINE - USAGE EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate various ways to use the pipeline.")
    print("Uncomment the examples you want to run.\n")

    # Uncomment to run examples:
    # example_basic_usage()
    # example_premium_quality()
    # example_performance_mode()
    # example_batch_processing()
    # example_custom_scene_optimization()
    # example_with_lut()
    # example_runtime_overrides()
    # example_parallel_output_generation()
    # example_statistics_tracking()
    # example_social_media_workflow()
    # example_auto_scene_detection()
    # example_print_production()

    print("\nAll examples defined. Uncomment specific examples to run them.")
