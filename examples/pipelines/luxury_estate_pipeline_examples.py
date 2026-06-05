#!/usr/bin/env python3
"""
Luxury Estate Master Pipeline - Usage Examples
===============================================

Demonstrates various ways to use the pipeline for different scenarios.

Examples:
1. Single image processing with default preset
2. Single image with custom room type
3. Batch processing with automatic room detection
4. Custom preset configuration
5. Programmatic API usage
6. Performance-optimized processing
7. Quality-optimized processing

Author: Transformation Portal
Date: 2025-11-10
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = REPO_ROOT / "scripts" / "pipelines"
UTILITIES_DIR = REPO_ROOT / "scripts" / "utilities"
os.environ.setdefault("TP_LUXURY_ESTATE_PIPELINE_LOG", "/tmp/tp-luxury-estate-pipeline.log")
for import_root in (PIPELINES_DIR, UTILITIES_DIR):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from luxury_estate_master_pipeline import (
    AIEnhancementConfig,
    ColorGradingConfig,
    DepthConfig,
    LuxuryEstateMasterPipeline,
    MaterialResponseConfig,
    OutputConfig,
    PipelinePreset,
    ToneMappingConfig,
    UpscalingConfig,
    get_750_picacho_preset,
    get_aerial_preset,
)


def example_1_basic_single_image():
    """Example 1: Process single image with default preset."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Single Image Processing")
    print("=" * 80)

    # Initialize pipeline with default preset
    preset = get_750_picacho_preset()
    pipeline = LuxuryEstateMasterPipeline(preset)

    # Process single image
    image_path = Path("input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Great_Room_HDR_32-bit.tif")

    if image_path.exists():
        result = pipeline.process_image(image_path, room_type="great_room")
        print(f"\n✓ Processing complete!")
        print(f"  Output: {result['output_paths']}")
    else:
        print(f"⚠ Image not found: {image_path}")


def example_2_aerial_preset():
    """Example 2: Process aerial image with optimized preset."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Aerial Photography with Specialized Preset")
    print("=" * 80)

    # Use aerial-optimized preset
    preset = get_aerial_preset()
    pipeline = LuxuryEstateMasterPipeline(preset)

    image_path = Path("input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Aerial_HDR_32-bit.tif")

    if image_path.exists():
        result = pipeline.process_image(image_path, room_type="aerial")
        print(f"\n✓ Aerial processing complete with atmospheric effects!")
        print(f"  Total time: {result['total_time']:.1f}s")
    else:
        print(f"⚠ Image not found: {image_path}")


def example_3_batch_processing():
    """Example 3: Batch process all images with room type mapping."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Batch Processing with Room Type Detection")
    print("=" * 80)

    preset = get_750_picacho_preset()
    pipeline = LuxuryEstateMasterPipeline(preset)

    # Find all source images
    source_dir = Path("input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs")
    image_paths = list(source_dir.glob("*.tif"))

    if not image_paths:
        print(f"⚠ No images found in {source_dir}")
        return

    # Define room type mappings
    room_types = {
        "750Picacho_Aerial_HDR_32-bit": "aerial",
        "750Picacho_Bathroom_HDR_32-bit": "bathroom",
        "750Picacho_Bedroom_HDR_32-bit": "bedroom",
        "750Picacho_Great_Room_HDR_32-bit": "great_room",
        "750Picacho_Kitchen_HDR_32-bit": "kitchen",
        "750Picacho_Pool_HDR_32-bit": "pool",
    }

    # Batch process
    results = pipeline.batch_process(image_paths, room_types)

    print(f"\n✓ Batch processing complete!")
    print(f"  Processed: {len(results)} images")
    print(f"  Total time: {pipeline.stats['total_time']:.1f}s")
    print(f"  Average: {pipeline.stats['total_time']/len(results):.1f}s per image")


def example_4_custom_preset():
    """Example 4: Create and use custom preset."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Preset Configuration")
    print("=" * 80)

    # Create custom preset optimized for speed
    custom_preset = PipelinePreset(
        name="Fast Processing",
        description="Speed-optimized configuration for quick turnaround",
        # Enable only essential stages
        depth=DepthConfig(
            enabled=True,
            model_variant="small",
            backend="coreml",  # Fastest on M-series
            clarity_strength=0.5,
        ),
        material_response=MaterialResponseConfig(
            enabled=True,
            strength=0.6,  # Reduced for speed
        ),
        tone_mapping=ToneMappingConfig(
            method="filmic",
            exposure=0.0,
            contrast=1.05,
        ),
        color_grading=ColorGradingConfig(
            enabled=True,
            saturation=1.08,
        ),
        # Disable slow stages
        ai_enhancement=AIEnhancementConfig(
            enabled=False,  # Skip AI for speed
        ),
        upscaling=UpscalingConfig(
            enabled=False,  # Skip upscaling for speed
        ),
        output=OutputConfig(
            save_master_tiff=True,
            save_delivery_jpeg=True,
            master_bit_depth=16,
        ),
    )

    # Use custom preset
    pipeline = LuxuryEstateMasterPipeline(custom_preset)

    print(f"Custom preset created: {custom_preset.name}")
    print(f"  Depth: {custom_preset.depth.enabled}")
    print(f"  AI Enhancement: {custom_preset.ai_enhancement.enabled}")
    print(f"  Upscaling: {custom_preset.upscaling.enabled}")
    print(f"\nExpected speedup: 8-10x faster (~10s per image)")


def example_5_quality_optimized():
    """Example 5: Maximum quality configuration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Maximum Quality Configuration")
    print("=" * 80)

    # Start with default preset
    preset = get_750_picacho_preset()

    # Optimize for maximum quality
    preset.ai_enhancement.num_inference_steps = 40  # Higher quality AI
    preset.ai_enhancement.guidance_scale = 8.5  # Stronger prompt adherence
    preset.upscaling.scale_factor = 4.0  # Maximum upscaling
    preset.output.master_bit_depth = 16  # 16-bit master
    preset.material_response.strength = 0.85  # Enhanced material response

    pipeline = LuxuryEstateMasterPipeline(preset)

    print("Quality-optimized preset created:")
    print(f"  AI steps: {preset.ai_enhancement.num_inference_steps}")
    print(f"  AI guidance: {preset.ai_enhancement.guidance_scale}")
    print(f"  Upscaling: {preset.upscaling.scale_factor}x")
    print(f"  Material strength: {preset.material_response.strength}")
    print(f"\nExpected time: ~120s per image (highest quality)")


def example_6_stage_by_stage():
    """Example 6: Access individual processing stages."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Stage-by-Stage Processing")
    print("=" * 80)

    preset = get_750_picacho_preset()
    pipeline = LuxuryEstateMasterPipeline(preset)

    image_path = Path("input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Kitchen_HDR_32-bit.tif")

    if not image_path.exists():
        print(f"⚠ Image not found: {image_path}")
        return

    print("Processing stages:")
    print("  1. Load 32-bit HDR TIFF")
    print("  2. Depth Anything V2 estimation")
    print("  3. Material Response enhancement")
    print("  4. Filmic tone mapping")
    print("  5. Color grading with LUTs")
    print("  6. ControlNet + SDXL refinement")
    print("  7. Real-ESRGAN 4x upscaling")
    print("\nEach stage can be toggled via preset configuration.")


def example_7_output_comparison():
    """Example 7: Generate comparison outputs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Output Comparison (Multiple Presets)")
    print("=" * 80)

    image_path = Path("input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Pool_HDR_32-bit.tif")

    if not image_path.exists():
        print(f"⚠ Image not found: {image_path}")
        return

    # Create multiple presets for comparison
    presets = [
        ("Standard", get_750_picacho_preset()),
        ("Aerial", get_aerial_preset()),
    ]

    # Modify standard preset for variants
    fast_preset = get_750_picacho_preset()
    fast_preset.name = "Fast"
    fast_preset.ai_enhancement.enabled = False
    fast_preset.upscaling.enabled = False
    fast_preset.output.output_dir = "output_comparison_fast"
    presets.append(("Fast", fast_preset))

    quality_preset = get_750_picacho_preset()
    quality_preset.name = "Ultra Quality"
    quality_preset.ai_enhancement.num_inference_steps = 50
    quality_preset.output.output_dir = "output_comparison_quality"
    presets.append(("Ultra Quality", quality_preset))

    print("Comparison presets:")
    for name, preset in presets:
        print(f"\n  {name}:")
        print(f"    - AI: {'Yes' if preset.ai_enhancement.enabled else 'No'}")
        print(f"    - Upscaling: {'Yes' if preset.upscaling.enabled else 'No'}")
        print(f"    - Output: {preset.output.output_dir}")

    print("\nProcess with each preset to compare results:")
    print("  python luxury_estate_master_pipeline.py image.tif --preset standard")
    print("  python luxury_estate_master_pipeline.py image.tif --preset aerial")


def example_8_report_analysis():
    """Example 8: Analyze processing report."""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Processing Report Analysis")
    print("=" * 80)

    import json

    # Example report path (after batch processing)
    report_path = Path("output_750_picacho_elite/processing_report.json")

    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        print(f"Report loaded: {report_path}")
        print(f"\nBatch Statistics:")
        print(f"  Preset: {report['preset']}")
        print(f"  Images processed: {report['images_processed']}")
        print(f"  Total time: {report['total_time']:.1f}s")
        print(f"  Average time: {report['average_time']:.1f}s per image")

        # Analyze stage times
        if report["results"]:
            first_result = report["results"][0]
            if "stages" in first_result:
                print(f"\nStage breakdown (first image):")
                for stage, duration in first_result["stages"].items():
                    print(f"    {stage}: {duration:.2f}s")
    else:
        print(f"⚠ Report not found: {report_path}")
        print("  Run batch processing first:")
        print("  ./process_750_picacho_elite_batch.sh")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LUXURY ESTATE MASTER PIPELINE - USAGE EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates various pipeline usage patterns.")
    print("Uncomment example calls in main() to run specific examples.")

    # Uncomment to run specific examples:

    # example_1_basic_single_image()
    # example_2_aerial_preset()
    # example_3_batch_processing()
    example_4_custom_preset()
    # example_5_quality_optimized()
    example_6_stage_by_stage()
    example_7_output_comparison()
    # example_8_report_analysis()

    print("\n" + "=" * 80)
    print("For full documentation, see:")
    print("  docs/guides/LUXURY_ESTATE_PIPELINE.md")
    print("  LUXURY_ESTATE_PIPELINE_QUICKSTART.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
