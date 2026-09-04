#!/usr/bin/env python3
"""Example: PBR Map Generation for Luxury Real Estate with Presets.

This script demonstrates using the three main PBR presets for architectural
visualization workflows. Run with sample images to generate PBR maps at
different quality tiers.

Usage:
    python examples/pbr_preset_example.py --input ./images --preset standard
    python examples/pbr_preset_example.py --input ./images --preset premium --limit 5
    python examples/pbr_preset_example.py --list-presets

Requirements:
    - Input images in supported formats (JPG, PNG, TIFF)
    - Depth Anything V3 model (auto-downloaded on first run)
    - 4-8 GB RAM depending on preset
"""

import argparse
import sys
from pathlib import Path
from typing import Mapping, Optional

try:
    from transformation_portal.lux_depth_v3 import EnhanceOrchestrator, get_preset, list_presets
    from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
    from transformation_portal.lux_depth_v3.manifest import CombinedManifest
except ImportError as e:
    print(f"Error: Could not import lux_depth_v3 module: {e}")
    print("\nPlease ensure the package is installed:")
    print("  pip install -e .")
    sys.exit(1)


def _authoritative_output_paths(
    result: Mapping[str, object],
    *,
    require_float_depth: bool,
) -> dict[str, Path]:
    """Return only paths carried by the result and its combined manifest."""

    result_fields = {
        "depth": "depth_path",
        "manifest": "manifest",
    }
    if require_float_depth:
        result_fields["depth_float"] = "depth_float_path"

    outputs: dict[str, Path] = {}
    for label, field_name in result_fields.items():
        value = result.get(field_name)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"successful result is missing {field_name}")
        outputs[label] = Path(value)

    combined_manifest = CombinedManifest.load(outputs["manifest"])
    if not isinstance(combined_manifest.pbr_assets, dict):
        raise RuntimeError("successful PBR result is missing combined-manifest PBR assets")
    for label in ("normal", "roughness", "ao"):
        value = combined_manifest.pbr_assets.get(f"{label}_path")
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"combined manifest is missing {label}_path")
        outputs[label] = Path(value)

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate PBR maps using optimized presets for luxury real estate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with standard quality
  %(prog)s --input ./estate_photos --preset standard

  # Premium quality for hero shots only
  %(prog)s --input ./hero_shots --preset premium --limit 10

  # Quick preview draft
  %(prog)s --input ./test_images --preset draft

  # Material-specific processing
  %(prog)s --input ./hardwood_floors --preset wood
  %(prog)s --input ./kitchen_stone --preset stone

  # List all available presets
  %(prog)s --list-presets
        """,
    )

    parser.add_argument("--input", type=Path, help="Input directory containing images (JPG, PNG, TIFF)")
    parser.add_argument("--output", type=Path, help="Output directory for PBR maps (default: ./pbr_output_<preset>)")
    parser.add_argument(
        "--preset",
        type=str,
        default="standard",
        help="Preset name: standard, premium, draft, wood, metal, glass, stone, fabric (default: standard)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of images to process (for testing)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    parser.add_argument(
        "--device", type=str, choices=["mps", "cuda", "cpu"], help="Override device selection (default: auto-detect)"
    )

    args = parser.parse_args()

    # List presets and exit
    if args.list_presets:
        print("Available PBR Presets:\n")

        print("Quality Tiers:")
        print("  draft     - Fast preview (500-700 img/hr)")
        print("  standard  - Balanced quality (200-250 img/hr)")
        print("  premium   - Maximum quality (100-150 img/hr)")

        print("\nMaterial-Specific:")
        print("  wood      - Hardwood floors, cabinetry")
        print("  metal     - Fixtures, appliances")
        print("  glass     - Windows, mirrors")
        print("  stone     - Countertops, tile")
        print("  fabric    - Upholstery, curtains")

        print("\nFor detailed parameter info, see:")
        print("  docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md")
        return 0

    # Validate required arguments
    if not args.input:
        parser.error("--input is required (or use --list-presets)")

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    # Load preset
    try:
        config = get_preset(args.preset)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable presets: {', '.join(list_presets())}")
        sys.exit(1)

    # Override device if specified
    if args.device:
        from dataclasses import replace

        config = replace(config, depth_device=args.device)
        print(f"Using device: {args.device}")

    # Setup output directory
    if args.output:
        output_root = args.output
    else:
        output_root = Path(f"./pbr_output_{args.preset}")

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = [p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
    image_paths.sort()

    if not image_paths:
        print(f"Error: No images found in {args.input}")
        print(f"Looking for: {', '.join(image_extensions)}")
        sys.exit(1)

    # Apply limit if specified
    if args.limit:
        image_paths = image_paths[: args.limit]

    # Print configuration
    print(f"\n{'='*60}")
    print(f"PBR Map Generation - {args.preset.upper()} Preset")
    print(f"{'='*60}")
    print(f"Input:   {args.input}")
    print(f"Output:  {output_root}")
    print(f"Images:  {len(image_paths)}")
    print(f"Preset:  {args.preset}")
    print(f"\nConfiguration:")
    print(f"  Normal strength:    {config.pbr_normal_strength}")
    print(f"  Normal blur:        {config.pbr_normal_blur_radius}")
    print(f"  Roughness strength: {config.pbr_roughness_strength}")
    print(f"  Roughness blur:     {config.pbr_roughness_blur_radius}")
    print(f"  AO strength:        {config.pbr_ao_strength}")
    print(f"  AO blur:            {config.pbr_ao_blur_radius}")
    print(f"  AO bias:            {config.pbr_ao_bias}")
    print(f"  Float depth:        {config.save_float_depth}")
    print(f"  Model:              {config.model_key or 'default'}")
    print(f"{'='*60}\n")

    # Freeze the exact input/model/runtime authority before initialization.
    prepared = prepare_lux_execution(
        config,
        args.input,
        [image_path.absolute() for image_path in image_paths],
    )
    image_paths = list(prepared.input_files)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)

    # Process the complete frozen selection so final batch evidence is emitted.
    batch_results = orchestrator.enhance_batch(
        prepared.input_root,
        input_files=list(prepared.input_files),
    )
    if len(batch_results) != len(image_paths):
        raise RuntimeError(f"batch returned {len(batch_results)} results for {len(image_paths)} prepared inputs")
    successful = 0
    failed = 0

    for i, (img_path, result) in enumerate(zip(image_paths, batch_results), 1):
        print(f"[{i}/{len(image_paths)}] Processed: {img_path.name}")
        if result.get("status") != "ok":
            print(f"  ✗ Failed: {result.get('error') or result.get('reason') or 'non-ok status'}")
            failed += 1
            continue

        try:
            outputs = _authoritative_output_paths(
                result,
                require_float_depth=config.save_float_depth,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"  ✗ Invalid output evidence: {exc}")
            failed += 1
            continue

        missing = [label for label, path in outputs.items() if not path.is_file()]
        if missing:
            print(f"  ✗ Missing evidence-bound outputs: {', '.join(missing)}")
            failed += 1
            continue

        print(f"  ✓ Generated: {', '.join(outputs)}")
        successful += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Complete")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(image_paths)}")
    if failed > 0:
        print(f"Failed:     {failed}/{len(image_paths)}")
    print(f"Output:     {output_root}")
    print("\nExact output paths were read from each batch result and combined manifest.")
    print(f"{'='*60}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
