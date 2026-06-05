#!/usr/bin/env python3
"""
Elite Architectural Pipeline - Usage Examples
Demonstrates different processing workflows for luxury real estate imagery
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = REPO_ROOT / "scripts" / "pipelines"
UTILITIES_DIR = REPO_ROOT / "scripts" / "utilities"
INPUT_DIR = REPO_ROOT / "input_images" / "750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs"
OUTPUT_ROOT = Path("/tmp/tp-elite-pipeline-examples")
for import_root in (PIPELINES_DIR, UTILITIES_DIR):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from elite_architectural_pipeline import (
    AIEnhancementConfig,
    ColorGradingConfig,
    DepthConfig,
    EliteArchitecturalPipeline,
    PipelinePreset,
    ToneMappingConfig,
    get_750_picacho_preset,
)

# ============================================================================
# Example 1: Simple Single Image Processing
# ============================================================================


def example_1_simple_processing():
    """Process a single image with default settings."""
    print("=" * 80)
    print("Example 1: Simple Single Image Processing")
    print("=" * 80)

    # Get optimized preset for interior spaces
    preset = get_750_picacho_preset(room_type="interior")

    # Disable AI/upscaling for faster processing
    preset.ai_enhancement.enabled = False
    preset.ai_enhancement.upscale_4x = False

    # Initialize pipeline
    pipeline = EliteArchitecturalPipeline(preset=preset, output_dir=OUTPUT_ROOT / "example_1", dry_run=False)

    # Process image
    input_path = INPUT_DIR / "750Picacho_Great_Room_HDR_32-bit.tif"
    if input_path.exists():
        outputs = pipeline.process_image(input_path)
        print(f"\n✅ Outputs: {list(outputs.keys())}")
    else:
        print(f"⚠️ Input file not found: {input_path}")


# ============================================================================
# Example 2: Batch Processing with Custom Settings
# ============================================================================


def example_2_batch_processing():
    """Batch process all 750 Picacho images."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Processing with Custom Settings")
    print("=" * 80)

    # Get aerial preset for outdoor shots
    preset = get_750_picacho_preset(room_type="aerial")

    # Custom tone mapping settings
    preset.tone_mapping.method = "filmic"
    preset.tone_mapping.exposure = 0.2  # Slight brightening

    # Enhanced color grading for golden hour
    preset.color_grading.saturation = 1.15
    preset.color_grading.temperature_shift = (1.08, 1.0, 0.95)

    # Disable upscaling for faster batch processing
    preset.ai_enhancement.upscale_4x = False

    pipeline = EliteArchitecturalPipeline(preset=preset, output_dir=OUTPUT_ROOT / "example_2_batch", dry_run=False)

    input_dir = INPUT_DIR
    if input_dir.exists():
        outputs = pipeline.batch_process(input_dir, pattern="*.tif")
        print(f"\n✅ Processed {len(outputs)} images")
    else:
        print(f"⚠️ Input directory not found: {input_dir}")


# ============================================================================
# Example 3: Maximum Quality Processing
# ============================================================================


def example_3_maximum_quality():
    """Process with all features enabled for maximum quality."""
    print("\n" + "=" * 80)
    print("Example 3: Maximum Quality Processing")
    print("=" * 80)

    preset = get_750_picacho_preset(room_type="aerial")

    # Enable all enhancement features
    preset.depth.clarity_strength = 0.7
    preset.material_response.strength = 0.85
    preset.ai_enhancement.enabled = True
    preset.ai_enhancement.upscale_4x = True
    preset.ai_enhancement.num_steps = 40  # More inference steps
    preset.ai_enhancement.strength = 0.30  # Lower strength for faithfulness

    pipeline = EliteArchitecturalPipeline(
        preset=preset,
        output_dir=OUTPUT_ROOT / "example_3_maximum_quality",
        dry_run=False,
    )

    input_path = INPUT_DIR / "750Picacho_Aerial_HDR_32-bit.tif"
    if input_path.exists():
        outputs = pipeline.process_image(input_path)
        print(f"\n✅ Maximum quality output: {outputs['upscaled']}")
    else:
        print(f"⚠️ Input file not found: {input_path}")


# ============================================================================
# Example 4: Custom Preset from Scratch
# ============================================================================


def example_4_custom_preset():
    """Create a completely custom preset."""
    print("\n" + "=" * 80)
    print("Example 4: Custom Preset from Scratch")
    print("=" * 80)

    # Build custom preset
    preset = PipelinePreset(
        name="Custom Pool Enhancement",
        description="Specialized preset for pool photography with vivid water",
        # Depth processing
        depth=DepthConfig(
            enabled=True,
            num_zones=3,
            atmospheric_haze=False,
            clarity_strength=0.5,
        ),
        # Aggressive tone mapping for drama
        tone_mapping=ToneMappingConfig(
            method="filmic",
            exposure=0.1,
            contrast=1.15,
        ),
        # Cool color grade for water
        color_grading=ColorGradingConfig(
            lut_stack=[
                "assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube",
            ],
            lut_strengths=[0.6],
            saturation=1.15,
            temperature_shift=(0.98, 1.0, 1.05),  # Cool tones
        ),
        # AI enhancement with pool-specific prompt
        ai_enhancement=AIEnhancementConfig(
            enabled=False,  # Disabled for this example
            prompt="luxury infinity pool, crystal clear turquoise water, premium outdoor living, professional architectural photography",
        ),
    )

    pipeline = EliteArchitecturalPipeline(preset=preset, output_dir=OUTPUT_ROOT / "example_4_custom", dry_run=False)

    input_path = INPUT_DIR / "750Picacho_Pool_HDR_32-bit.tif"
    if input_path.exists():
        outputs = pipeline.process_image(input_path)
        print(f"\n✅ Custom processing complete")
    else:
        print(f"⚠️ Input file not found: {input_path}")


# ============================================================================
# Example 5: Dry Run Configuration Preview
# ============================================================================


def example_5_dry_run():
    """Preview configuration without processing."""
    print("\n" + "=" * 80)
    print("Example 5: Dry Run Configuration Preview")
    print("=" * 80)

    preset = get_750_picacho_preset(room_type="interior")

    pipeline = EliteArchitecturalPipeline(preset=preset, output_dir=OUTPUT_ROOT / "example_5", dry_run=True)

    # This will show configuration without processing
    input_path = INPUT_DIR / "750Picacho_Great_Room_HDR_32-bit.tif"
    if input_path.exists():
        print("\n📋 Configuration preview shown above")
    else:
        print(f"⚠️ Input file not found: {input_path}")


# ============================================================================
# Example 6: Fast Processing Mode (Depth + Tone Mapping Only)
# ============================================================================


def example_6_fast_mode():
    """Fast processing without AI enhancement or upscaling."""
    print("\n" + "=" * 80)
    print("Example 6: Fast Processing Mode")
    print("=" * 80)

    preset = get_750_picacho_preset(room_type="interior")

    # Disable expensive operations
    preset.material_response.enabled = False
    preset.color_grading.enabled = False
    preset.ai_enhancement.enabled = False
    preset.ai_enhancement.upscale_4x = False

    pipeline = EliteArchitecturalPipeline(preset=preset, output_dir=OUTPUT_ROOT / "example_6_fast", dry_run=False)

    input_dir = INPUT_DIR
    if input_dir.exists():
        outputs = pipeline.batch_process(input_dir, pattern="*.tif")
        print(f"\n✅ Fast mode: Processed {len(outputs)} images")
        print("⚡ Expected throughput: 400-600 images/hour")
    else:
        print(f"⚠️ Input directory not found: {input_dir}")


# ============================================================================
# Main Runner
# ============================================================================


def main():
    """Run all examples."""
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        Elite Architectural Pipeline - Usage Examples                        ║
║        Demonstrations of different processing workflows                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    )

    examples = [
        ("1", "Simple single image processing", example_1_simple_processing),
        ("2", "Batch processing with custom settings", example_2_batch_processing),
        ("3", "Maximum quality processing", example_3_maximum_quality),
        ("4", "Custom preset from scratch", example_4_custom_preset),
        ("5", "Dry run configuration preview", example_5_dry_run),
        ("6", "Fast processing mode", example_6_fast_mode),
    ]

    print("Available examples:")
    for num, desc, _ in examples:
        print(f"  [{num}] {desc}")
    print("  [all] Run all examples")
    print("  [q] Quit")

    choice = input("\nSelect example to run: ").strip().lower()

    if choice == "q":
        print("Exiting.")
        return

    if choice == "all":
        for _, _, func in examples:
            func()
    else:
        for num, _, func in examples:
            if choice == num:
                func()
                break
        else:
            print(f"Invalid choice: {choice}")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
