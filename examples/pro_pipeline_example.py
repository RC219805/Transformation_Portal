#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using the Professional Pipeline

This example demonstrates how to use the fully-integrated professional pipeline
for architectural rendering enhancement.
"""

from pathlib import Path

from transformation_portal.pipelines.pro_pipeline import PipelinePreset, ProPipeline, ProPipelineConfig


def example_single_image():
    """Process a single image with the architectural hero preset."""
    print("=" * 60)
    print("Example 1: Single Image Processing")
    print("=" * 60)

    # Create configuration
    config = ProPipelineConfig(
        input_path=Path("data/sample_images/bedroom_1.jpg"),
        output_dir=Path("./output/examples"),
        preset=PipelinePreset.ARCHITECTURAL_HERO,
        device="auto",
        quality="high",
        output_format="tif",
        bit_depth=16,
    )

    # Create and run pipeline
    pipeline = ProPipeline(config)
    result = pipeline.process_image(config.input_path)

    if result:
        print(f"\n✓ Success! Output saved to: {result}")
    else:
        print("\n✗ Processing failed")


def example_batch_processing():
    """Batch process multiple images."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    # Find all images in input directory
    input_dir = Path("data/sample_images")
    image_paths = list(input_dir.glob("*.jpg"))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    # Create configuration
    config = ProPipelineConfig(
        input_path=input_dir,
        output_dir=Path("./output/batch"),
        preset=PipelinePreset.INTERIOR_DRAMATIC,
        device="auto",
        quality="high",
    )

    # Create and run pipeline
    pipeline = ProPipeline(config)
    stats = pipeline.batch_process(image_paths)

    print(f"\n✓ Processed {stats['processed']} images")
    print(f"  Average time: {stats['avg_time']:.2f}s per image")
    print(f"  Throughput: {3600/stats['avg_time']:.1f} images/hour")


def example_custom_configuration():
    """Use custom stage configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    # Create custom configuration
    config = ProPipelineConfig(
        input_path=Path("data/sample_images/aerial_1.jpg"),
        output_dir=Path("./output/custom"),
        preset=PipelinePreset.CUSTOM,
        device="auto",
    )

    # Customize individual stages
    config.depth_stage.enabled = True
    config.depth_stage.config = {
        "atmospheric_haze": True,
        "clarity": 0.20,
    }

    config.ai_stage.enabled = False  # Skip AI for speed

    config.material_stage.enabled = True
    config.material_stage.config = {
        "strength": 0.75,
        "surfaces": ["grass", "water", "roof"],
    }

    config.grading_stage.enabled = True
    config.grading_stage.config = {
        "contrast": 1.15,
        "saturation": 1.12,
    }

    config.finishing_stage.enabled = True
    config.finishing_stage.config = {
        "sharpen": 0.18,
        "clarity": 0.22,
    }

    # Create and run pipeline
    pipeline = ProPipeline(config)
    result = pipeline.process_image(config.input_path)

    if result:
        print(f"\n✓ Custom processing complete: {result}")


def example_preset_comparison():
    """Compare different presets on the same image."""
    print("\n" + "=" * 60)
    print("Example 4: Preset Comparison")
    print("=" * 60)

    input_path = Path("data/sample_images/interior_1.jpg")

    if not input_path.exists():
        print(f"Input image not found: {input_path}")
        return

    presets = [
        PipelinePreset.ARCHITECTURAL_HERO,
        PipelinePreset.INTERIOR_DRAMATIC,
        PipelinePreset.KITCHEN_BRIGHT,
    ]

    for preset in presets:
        print(f"\nProcessing with preset: {preset.value}")

        config = ProPipelineConfig(
            input_path=input_path,
            output_dir=Path(f"./output/comparison/{preset.value}"),
            preset=preset,
        )

        pipeline = ProPipeline(config)
        result = pipeline.process_image(input_path)

        if result:
            print(f"  ✓ Saved to: {result}")


def example_progressive_enhancement():
    """Apply progressive enhancement stages."""
    print("\n" + "=" * 60)
    print("Example 5: Progressive Enhancement")
    print("=" * 60)

    input_path = Path("data/sample_images/exterior_1.jpg")

    if not input_path.exists():
        print(f"Input image not found: {input_path}")
        return

    # Stage 1: Depth-aware processing only
    print("\nStage 1: Depth-aware processing...")
    config = ProPipelineConfig(
        input_path=input_path,
        output_dir=Path("./output/progressive/stage1"),
        preset=PipelinePreset.CUSTOM,
    )
    config.depth_stage.enabled = True
    config.ai_stage.enabled = False
    config.material_stage.enabled = False
    config.grading_stage.enabled = False
    config.finishing_stage.enabled = False

    pipeline = ProPipeline(config)
    stage1_result = pipeline.process_image(input_path)

    # Stage 2: Add material response
    if stage1_result:
        print("\nStage 2: Adding Material Response...")
        config2 = ProPipelineConfig(
            input_path=stage1_result,
            output_dir=Path("./output/progressive/stage2"),
            preset=PipelinePreset.CUSTOM,
        )
        config2.depth_stage.enabled = False
        config2.material_stage.enabled = True
        config2.finishing_stage.enabled = True

        pipeline2 = ProPipeline(config2)
        stage2_result = pipeline2.process_image(stage1_result)

        if stage2_result:
            print(f"\n✓ Progressive enhancement complete: {stage2_result}")


if __name__ == "__main__":
    print("Professional Pipeline Examples")
    print("=" * 60)
    print()
    print("These examples demonstrate various ways to use the")
    print("Transformation Portal Professional Pipeline.")
    print()

    # Uncomment the examples you want to run:

    # example_single_image()
    # example_batch_processing()
    # example_custom_configuration()
    # example_preset_comparison()
    # example_progressive_enhancement()

    print("\n" + "=" * 60)
    print("Note: Uncomment the examples you want to run")
    print("=" * 60)
