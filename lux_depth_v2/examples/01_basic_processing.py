#!/usr/bin/env python3
"""
Basic Image Processing Example
===============================

Process a single image with default settings.

Requirements:
    - Input image at input_image.jpg
    - (Optional) Depth map at depth_maps/input_image.tif
"""
from pathlib import Path
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig, Preset


def main():
    # Configure pipeline
    config = PipelineConfig(
        preset=Preset.PHOTO_REALISTIC,

        # Input/output paths
        input_dir=Path("input"),
        output_dir=Path("output"),
        depth_dir=Path("depth_maps"),  # Optional

        # Device selection
        device="auto",  # Use CUDA if available, else CPU

        # Processing options
        upscale=4,
        enable_material=True,
        upscaler_backend="none",  # Use bicubic for quick test

        # Output formats
        save_master=True,
        save_upscaled=True,
        save_marketing_png=True,
        save_preview_jpg=True,
        preview_scale=0.25,
    )

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = LuxPipelineV2(config)
    print(f"Device: {pipeline.device}")
    print(f"Autocast: {pipeline.autocast}")

    # Process single image
    input_path = Path("input/input_image.jpg")

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        print("Create an 'input' directory with test images")
        return

    print(f"\nProcessing: {input_path.name}")
    result = pipeline.process_one(input_path)

    # Print results
    print("\nResults:")
    print(f"  Status: {result['status']}")
    print(f"  Processing time: {result['timing_s']:.2f}s")
    print(f"  Zone weights: {result['zone_weights']}")

    if result.get('material_mods'):
        print(f"  Material mods: {result['material_mods']}")

    if result.get('ai_color_diff'):
        print(f"  AI color drift: {result['ai_color_diff']:.4f}")
        print(f"  AI luma drift: {result['ai_luma_diff']:.4f}")

    print(f"\nOutputs saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
