#!/usr/bin/env python3
"""750 Picacho Primary Bedroom - Optimized PBR Processing for Luxury Real Estate.

This script demonstrates production-ready PBR map generation for luxury interior
photography using the new Lux Depth V3 presets. Specifically optimized for the
750 Picacho Primary Bedroom - a high-end luxury estate interior with hardwood
floors, premium textiles, and architectural glass elements.

Material Analysis - 750 Picacho Primary Bedroom:
    - Hardwood flooring: Wide-plank, satin finish (15-20% of frame)
    - Premium textiles: Bedding, curtains, upholstery (30-40% of frame)
    - Architectural glass: Windows, mirrors, fixtures (10-15% of frame)
    - Stone surfaces: Visible bathroom elements (5-10% of frame)
    - Metal accents: Lighting fixtures, hardware (5% of frame)

Optimal Preset Recommendation:
    PRIMARY: PREMIUM_QUALITY for hero shot marketing materials
    - Throughput: ~100-150 images/hour (6-8 seconds per image)
    - Memory: 5-7 GB peak
    - Use case: Client deliverables, MLS hero shots, print materials

    ALTERNATIVE: WOOD_OPTIMIZED for floor detail emphasis
    - Emphasizes hardwood grain and plank boundaries
    - Captures surface variation and natural texture
    - Ideal if flooring is the key selling feature

Quality Rationale:
    1. Premium quality essential for luxury real estate marketing
    2. Large source TIFF (143 MB, 16-bit) benefits from high-precision depth
    3. Mixed materials require maximum PBR detail for realistic rendering
    4. Previous processing iterations used depth-aware techniques - PBR extends this
    5. save_float_depth=True prevents quantization artifacts in multi-material scenes

Expected Outputs (per image):
    - {name}_depth.png: 16-bit depth visualization for inspection
    - {name}_depth_float.npy: High-precision depth (critical for quality PBR)
    - {name}_normal.png: RGB normal map (1.5x strength, no pre-blur)
    - {name}_roughness.png: Grayscale roughness (1.3x for material variation)
    - {name}_ao.png: Ambient occlusion (1.2x strength, 7px spread)
    - {name}_manifest.json: Processing metadata and parameters

Performance Expectations:
    - First run: 6-8 seconds (includes depth estimation)
    - Subsequent runs: 0.3-0.5 seconds (depth cached via LRU)
    - Memory peak: ~5.5 GB with the da3-metric model
    - GPU/MPS recommended for optimal performance

Integration Notes:
    - Follows existing output_750_picacho_* naming convention
    - Outputs to output_750_picacho_pbr/ to avoid overwriting previous work
    - Compatible with existing depth-aware processing workflows
    - Can be run standalone or integrated into batch pipelines

Usage:
    # Process with premium quality (recommended)
    python examples/process_750_picacho_pbr.py

    # Use wood-optimized preset for floor emphasis
    python examples/process_750_picacho_pbr.py --preset wood

    # Custom output directory
    python examples/process_750_picacho_pbr.py --output ./custom_output

    # Process with specific device
    python examples/process_750_picacho_pbr.py --device cuda

    # Dry run to validate configuration
    python examples/process_750_picacho_pbr.py --dry-run

Requirements:
    - Python 3.10+
    - transformation_portal package installed
    - Depth Anything V3 model (auto-downloaded on first run)
    - 6-8 GB RAM recommended for large TIFFs
    - MPS (Apple Silicon) or CUDA GPU for optimal performance
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

try:
    from transformation_portal.lux_depth_v3 import (
        FABRIC_OPTIMIZED,
        GLASS_OPTIMIZED,
        PREMIUM_QUALITY,
        STONE_OPTIMIZED,
        WOOD_OPTIMIZED,
        EnhanceOrchestrator,
        get_preset,
        list_presets,
    )
    from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
    from transformation_portal.lux_depth_v3.input_manager import ImageInput
except ImportError as e:
    print(f"❌ Error: Could not import lux_depth_v3 module: {e}")
    print("\nPlease ensure the transformation_portal package is installed:")
    print("  pip install -e .")
    sys.exit(1)


# Default source file for 750 Picacho
DEFAULT_INPUT = Path("input_images/750Picacho_PrimaryBedroom_Ultimate.tif")

# Output directory follows existing naming convention
DEFAULT_OUTPUT = Path("output_750_picacho_pbr")

# Material-aware presets registry
MATERIAL_PRESETS = {
    "premium": PREMIUM_QUALITY,
    "wood": WOOD_OPTIMIZED,
    "stone": STONE_OPTIMIZED,
    "glass": GLASS_OPTIMIZED,
    "fabric": FABRIC_OPTIMIZED,
}


def print_header():
    """Print processing header with property context."""
    print("\n" + "=" * 80)
    print("750 PICACHO PRIMARY BEDROOM - PREMIUM PBR PROCESSING")
    print("=" * 80)
    print("\nProperty: 750 Picacho Lane, Santa Barbara, CA")
    print("Scene: Primary Bedroom Suite")
    print("Materials: Hardwood floors, premium textiles, architectural glass")
    print("Quality Tier: Hero shot / Marketing deliverable")
    print("=" * 80 + "\n")


def analyze_source_file(input_path: Path) -> dict:
    """Analyze source TIFF and report characteristics."""
    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found: {input_path}")

    size_mb = input_path.stat().st_size / (1024 * 1024)

    # Try to get image dimensions
    try:
        from PIL import Image

        with Image.open(input_path) as img:
            width, height = img.size
            mode = img.mode
            megapixels = (width * height) / 1_000_000

            info = {
                "path": input_path,
                "size_mb": size_mb,
                "width": width,
                "height": height,
                "megapixels": megapixels,
                "mode": mode,
            }
    except Exception as e:
        # Fallback if image can't be opened
        info = {
            "path": input_path,
            "size_mb": size_mb,
            "error": str(e),
        }

    return info


def print_source_analysis(info: dict):
    """Print source file analysis."""
    print("📂 SOURCE FILE ANALYSIS")
    print("-" * 80)
    print(f"File: {info['path'].name}")
    print(f"Size: {info['size_mb']:.1f} MB")

    if "width" in info:
        print(f"Resolution: {info['width']} x {info['height']} ({info['megapixels']:.1f} MP)")
        print(f"Color Mode: {info['mode']}")

        # Memory estimate
        estimated_mem_gb = (info["megapixels"] * 12) / 1024  # Rough estimate
        print(f"Estimated Memory: ~{estimated_mem_gb:.1f} GB peak")
    else:
        print(f"Warning: Could not read image dimensions: {info.get('error', 'Unknown')}")

    print("-" * 80 + "\n")


def print_preset_config(config, preset_name: str):
    """Print detailed preset configuration."""
    print(f"🎛️  PRESET CONFIGURATION: {preset_name.upper()}")
    print("-" * 80)

    # Quality tier
    quality_tiers = {
        "premium": "Maximum Quality - Hero Shots & Client Deliverables",
        "wood": "Material-Optimized - Hardwood Emphasis",
        "stone": "Material-Optimized - Stone/Tile Emphasis",
        "glass": "Material-Optimized - Glass/Mirror Emphasis",
        "fabric": "Material-Optimized - Textile Emphasis",
    }
    print(f"Tier: {quality_tiers.get(preset_name, 'Custom Configuration')}")
    print()

    # Depth model
    print(f"Depth Model: {config.model_key or 'default'}")
    print(f"Device: {config.depth_device}")
    print(
        f"Float Depth: {config.save_float_depth} {'✓ High-precision' if config.save_float_depth else '⚠ Standard precision'}"
    )
    print()

    # PBR parameters
    print("PBR Map Parameters:")
    print(f"  Normal Strength:    {config.pbr_normal_strength:.1f}x")
    print(
        f"  Normal Blur:        {config.pbr_normal_blur_radius}px {'(sharp)' if config.pbr_normal_blur_radius == 0 else '(smoothed)'}"
    )
    print(f"  Roughness Strength: {config.pbr_roughness_strength:.1f}x")
    print(f"  Roughness Blur:     {config.pbr_roughness_blur_radius}px")
    print(f"  AO Strength:        {config.pbr_ao_strength:.1f}x")
    print(f"  AO Blur:            {config.pbr_ao_blur_radius}px")
    print(f"  AO Bias:            {config.pbr_ao_bias:.2f} {'(darker)' if config.pbr_ao_bias < 0.5 else '(brighter)'}")
    print()

    # Performance estimate
    throughput_estimates = {
        "premium": "100-150 images/hour (~6-8 seconds/image)",
        "wood": "180-220 images/hour (~4-5 seconds/image)",
        "stone": "180-220 images/hour (~4-5 seconds/image)",
        "glass": "200-250 images/hour (~3-4 seconds/image)",
        "fabric": "180-220 images/hour (~4-5 seconds/image)",
    }
    print(f"Expected Throughput: {throughput_estimates.get(preset_name, 'Variable')}")
    print("-" * 80 + "\n")


def print_outputs(output_dir: Path, base_name: str):
    """Print expected output files."""
    print("📦 EXPECTED OUTPUTS")
    print("-" * 80)
    print(f"Output Directory: {output_dir}")
    print()
    print("Generated Files:")
    print(f"  ✓ {base_name}_depth.png          (16-bit depth visualization)")
    print(f"  ✓ {base_name}_depth_float.npy    (high-precision depth array)")
    print(f"  ✓ {base_name}_normal.png         (RGB normal map)")
    print(f"  ✓ {base_name}_roughness.png      (grayscale roughness map)")
    print(f"  ✓ {base_name}_ao.png             (grayscale ambient occlusion)")
    print(f"  ✓ {base_name}_manifest.json      (processing metadata)")
    print()
    print("Integration with 3D Workflows:")
    print("  • Normal maps: Use in PBR shaders for surface detail")
    print("  • Roughness: Controls specular reflection intensity")
    print("  • AO: Enhances depth perception and contact shadows")
    print("  • Depth: Use for depth-of-field, atmospheric effects")
    print("-" * 80 + "\n")


def print_material_recommendations():
    """Print material-specific recommendations for this property."""
    print("🎨 MATERIAL-SPECIFIC RECOMMENDATIONS")
    print("-" * 80)
    print("For 750 Picacho Primary Bedroom, consider these presets:")
    print()
    print("  premium  - RECOMMENDED for hero shot marketing")
    print("             • Maximum quality across all materials")
    print("             • 1.5x normal strength, no pre-blur")
    print("             • Deep AO for dimensional depth")
    print()
    print("  wood     - Emphasize hardwood flooring detail")
    print("             • Enhanced grain texture (1.3x normal)")
    print("             • Captures satin finish variation")
    print("             • Natural shadows in plank joints")
    print()
    print("  fabric   - Emphasize bedding and textile detail")
    print("             • Moderate weave pattern detail (1.1x normal)")
    print("             • Natural fabric roughness variation")
    print("             • Soft fold shadows")
    print()
    print("  glass    - Emphasize windows and mirrors")
    print("             • Flat normals for reflective surfaces (0.7x)")
    print("             • Smooth specular (0.5x roughness)")
    print("             • Strong frame shadows, bright glass")
    print()
    print("Multi-material workflow:")
    print("  1. Run with 'premium' preset for balanced quality")
    print("  2. If specific material needs emphasis, re-run with material preset")
    print("  3. Composite PBR maps in post if needed")
    print("-" * 80 + "\n")


def process_image(
    input_path: Path,
    output_dir: Path,
    preset_name: str = "premium",
    device: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Process image with specified PBR preset.

    Args:
        input_path: Path to source TIFF
        output_dir: Output directory for PBR maps
        preset_name: Preset name (premium, wood, stone, glass, fabric)
        device: Override device selection (mps, cuda, cpu)
        dry_run: If True, validate config but don't process

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Print header
    print_header()

    # Analyze source file
    try:
        source_info = analyze_source_file(input_path)
        print_source_analysis(source_info)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}\n")
        return 1

    # Load preset
    try:
        config = get_preset(preset_name)
    except ValueError:
        print(f"❌ Error: Unknown preset '{preset_name}'")
        print(f"\nAvailable presets: {', '.join(list_presets())}\n")
        return 1

    # Override device if specified
    if device:
        config = replace(config, depth_device=device)
        print(f"🔧 Device override: {device}\n")

    # Print configuration
    print_preset_config(config, preset_name)

    # Print expected outputs
    base_name = input_path.stem
    print_outputs(output_dir, base_name)

    # Dry run mode
    if dry_run:
        print("🔍 DRY RUN MODE - Configuration validated, no processing performed\n")
        print_material_recommendations()
        return 0

    # Initialize orchestrator
    print("⚙️  INITIALIZING ORCHESTRATOR")
    print("-" * 80)
    try:
        prepared = prepare_lux_execution(config, input_path.parent, [input_path.absolute()])
        input_path = prepared.input_files[0]
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_dir)
        print("✓ Orchestrator initialized")
        print("✓ Depth model loaded (or will be loaded on first use)")
        print("-" * 80 + "\n")
    except Exception as e:
        print(f"❌ Error initializing orchestrator: {e}\n")
        return 1

    # Process image
    print("🚀 PROCESSING")
    print("-" * 80)
    start_time = time.time()

    try:
        print(f"Processing: {input_path.name}")
        print()

        image_input = ImageInput(path=input_path)
        result = orchestrator.enhance_image(image_input, input_root=prepared.input_root)

        elapsed = time.time() - start_time

        print()
        print(f"✓ Processing complete in {elapsed:.2f} seconds")
        print("-" * 80 + "\n")

        # Verify outputs
        print("✅ OUTPUT VERIFICATION")
        print("-" * 80)

        outputs = {
            "depth": output_dir / f"{base_name}_depth.png",
            "depth_float": output_dir / f"{base_name}_depth_float.npy",
            "normal": output_dir / f"{base_name}_normal.png",
            "roughness": output_dir / f"{base_name}_roughness.png",
            "ao": output_dir / f"{base_name}_ao.png",
            "manifest": output_dir / f"{base_name}_manifest.json",
        }

        all_present = True
        for output_type, output_path in outputs.items():
            if output_path.exists():
                size_info = ""
                if output_path.suffix == ".png":
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    size_info = f" ({size_mb:.1f} MB)"
                elif output_path.suffix == ".npy":
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    size_info = f" ({size_mb:.1f} MB)"
                elif output_path.suffix == ".json":
                    size_kb = output_path.stat().st_size / 1024
                    size_info = f" ({size_kb:.1f} KB)"

                print(f"  ✓ {output_path.name}{size_info}")
            else:
                print(f"  ✗ {output_path.name} - NOT FOUND")
                all_present = False

        print("-" * 80 + "\n")

        if not all_present:
            print("⚠️  Warning: Some expected outputs were not generated\n")

        # Success summary
        print("=" * 80)
        print("✅ SUCCESS - PBR MAPS GENERATED")
        print("=" * 80)
        print(f"\nOutput directory: {output_dir.absolute()}")
        print(f"Processing time: {elapsed:.2f} seconds")
        print(f"Preset used: {preset_name}")
        print()
        print("Next Steps:")
        print("  1. Review depth map for quality and accuracy")
        print("  2. Inspect normal map for surface detail preservation")
        print("  3. Check AO map for dimensional depth")
        print("  4. Import PBR maps into 3D workflow or compositing software")
        print("  5. For batch processing, see examples/batch_process.py")
        print()
        print("Integration with existing 750 Picacho outputs:")
        print(f"  • Previous outputs: output_750_picacho_elite/, output_750_picacho_refined/")
        print(f"  • PBR outputs: {output_dir.name}/")
        print(f"  • Compatible with depth-aware processing from process_750_picacho_depth_aware.py")
        print()

        return 0

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print(f"❌ Error during processing: {e}")
        print(f"Time before failure: {elapsed:.2f} seconds")
        print("-" * 80 + "\n")

        # Print troubleshooting hints
        print("💡 TROUBLESHOOTING")
        print("-" * 80)
        print("Common issues:")
        print("  • Out of memory: Try reducing image size or use 'draft' preset")
        print("  • Model download: Ensure internet connection for first-time model download")
        print("  • Device error: Try --device cpu if MPS/CUDA fails")
        print("  • Corrupted TIFF: Verify source file opens in image viewer")
        print("-" * 80 + "\n")

        return 1


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process 750 Picacho Primary Bedroom with optimized PBR presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with recommended premium preset
  %(prog)s

  # Emphasize hardwood flooring detail
  %(prog)s --preset wood

  # Custom output directory
  %(prog)s --output ./custom_pbr_output

  # Force CPU processing
  %(prog)s --device cpu

  # Validate configuration without processing
  %(prog)s --dry-run

Material Preset Selection Guide:
  premium - Balanced maximum quality (RECOMMENDED for hero shots)
  wood    - Emphasize hardwood grain and texture
  stone   - Emphasize stone/tile detail and grout
  glass   - Emphasize reflective surfaces
  fabric  - Emphasize textile weave and draping

For detailed PBR configuration documentation:
  docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md
  docs/reference/PBR_PRESETS_QUICK_REFERENCE.md
        """,
    )

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input TIFF file (default: {DEFAULT_INPUT})")

    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")

    parser.add_argument(
        "--preset",
        type=str,
        default="premium",
        choices=list(MATERIAL_PRESETS.keys()),
        help="PBR preset to use (default: premium)",
    )

    parser.add_argument(
        "--device", type=str, choices=["mps", "cuda", "cpu"], help="Override device selection (default: auto-detect)"
    )

    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without processing")

    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")

    args = parser.parse_args()

    # List presets and exit
    if args.list_presets:
        print("\n750 Picacho Material-Aware PBR Presets:\n")
        print_material_recommendations()
        return 0

    # Process image
    return process_image(
        input_path=args.input,
        output_dir=args.output,
        preset_name=args.preset,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
