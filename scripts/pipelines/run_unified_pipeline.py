#!/usr/bin/env python3
"""
CLI runner for Unified Luxury Pipeline
Process 750 Picacho Lane renderings with production-grade pipeline
"""
from pathlib import Path

from src.transformation_portal.pipelines.unified_luxury_pipeline import (
    OutputFormat,
    ProcessingProfile,
    SceneType,
    UnifiedLuxuryPipeline,
    UnifiedPipelineConfig,
)


def main():
    # Input file from desktop cache - using Master TIFF (16-bit)
    input_file = (
        Path.home()
        / "Desktop"
        / "Cache"
        / "750_LightFiction_Final_Views"
        / "Master_TIFFs_16bit"
        / "750Picacho_Pool.tif"
    )

    # Create output directory
    output_dir = Path.home() / "Desktop" / "Cache" / "750_Picacho_Unified_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Unified Luxury Pipeline - 750 Picacho Lane Pool View")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")
    print()

    # Configure pipeline for premium pool rendering
    config = UnifiedPipelineConfig(
        scene_type=SceneType.EXTERIOR,  # Pool view is exterior
        profile=ProcessingProfile.PREMIUM,  # Highest quality
        enable_material_response=True,  # Physics-based water/tile enhancement
        enable_depth=True,  # Depth-aware atmospheric effects
        enable_vfx=True,  # VFX enhancements (reflections, bloom)
        enable_color_grading=True,  # Professional color grading
        output_formats=[
            OutputFormat.MASTER_TIFF,  # 16-bit master
            OutputFormat.WEB_4K,  # Web delivery
            OutputFormat.PRINT_8K,  # Print quality
            OutputFormat.SOCIAL,  # Social media
        ],
        output_dir=output_dir,
        preserve_metadata=True,  # Keep EXIF/IPTC
        save_intermediates=True,  # Save processing stages
        parallel_outputs=True,  # Fast multi-format generation
    )

    print("Pipeline Configuration:")
    print(f"  Scene Type: {config.scene_type.value}")
    print(f"  Profile: {config.profile.value}")
    print(f"  Material Response: {config.enable_material_response}")
    print(f"  Depth Processing: {config.enable_depth}")
    print(f"  VFX Enhancements: {config.enable_vfx}")
    print(f"  Color Grading: {config.enable_color_grading}")
    print(f"  Output Formats: {[f.value for f in config.output_formats]}")
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = UnifiedLuxuryPipeline(config)
    print("✓ Pipeline ready")
    print()

    # Process image
    print("Processing 750Picacho_Pool.exr...")
    print("-" * 80)

    try:
        results = pipeline.process(input_file)

        print()
        print("=" * 80)
        print("✓ Processing Complete!")
        print("=" * 80)
        print()
        print("Generated Outputs:")
        for format_name, output_path in results.items():
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  [{format_name}] {output_path.name} ({size_mb:.1f} MB)")

        # Print statistics
        stats = pipeline.get_statistics()
        print()
        print("Pipeline Statistics:")
        print(f"  Total Processing Time: {stats.get('total_time', 0):.1f}s")
        print(f"  Stages Completed: {stats.get('stages_completed', 0)}")
        print(f"  Stages Skipped: {stats.get('stages_skipped', 0)}")
        print(f"  Outputs Generated: {stats.get('outputs_generated', 0)}")

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("✗ Processing Failed")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
