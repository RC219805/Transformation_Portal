#!/usr/bin/env python3
"""
Script 06: Process Final Output for 750 Picacho Lane.

This script processes all 6 property images to produce final enhanced
4K 16-bit TIFF deliverables.

Features:
- Full 4K resolution processing
- Material-specific enhancement
- 16-bit TIFF output with metadata
- Production-ready deliverables

Usage:
    python scripts/training/750_picacho/06_process_final_output.py [options]

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from training.property_specific.picacho_analyzer import PicachoAnalyzer
from training.property_specific.picacho_inference import (
    PicachoInference,
    InferenceConfig,
    OutputFormat,
    EnhancementLevel
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process final 750 Picacho Lane enhanced outputs"
    )
    parser.add_argument(
        "--property-dir",
        type=Path,
        default=None,
        help="Path to property images directory"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("weights/750_picacho/best_model.pth"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/750_picacho/final_deliverables"),
        help="Output directory for final deliverables"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["16bit_tiff", "32bit_tiff", "16bit_png", "8bit_png", "jpeg_high", "jpeg_web"],
        default="16bit_tiff",
        help="Output format"
    )
    parser.add_argument(
        "--enhancement-level",
        type=str,
        choices=["subtle", "balanced", "strong", "maximum"],
        default="balanced",
        help="Enhancement intensity level"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("750 PICACHO LANE FINAL OUTPUT PROCESSING")
    print("=" * 60 + "\n")

    # Map string arguments to enums
    format_map = {
        "16bit_tiff": OutputFormat.TIFF_16BIT,
        "32bit_tiff": OutputFormat.TIFF_32BIT,
        "16bit_png": OutputFormat.PNG_16BIT,
        "8bit_png": OutputFormat.PNG_8BIT,
        "jpeg_high": OutputFormat.JPEG_HIGH,
        "jpeg_web": OutputFormat.JPEG_WEB,
    }

    level_map = {
        "subtle": EnhancementLevel.SUBTLE,
        "balanced": EnhancementLevel.BALANCED,
        "strong": EnhancementLevel.STRONG,
        "maximum": EnhancementLevel.MAXIMUM,
    }

    # Initialize analyzer to get image paths
    analyzer = PicachoAnalyzer(property_dir=args.property_dir)

    if not analyzer.image_paths:
        print("❌ No property images found.")
        return 1

    # Run analysis to get materials
    print("Analyzing property images...")
    analyzer.analyze_property()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference
    config = InferenceConfig(
        model_path=args.model,
        device=args.device,
        output_format=format_map[args.format],
        enhancement_level=level_map[args.enhancement_level],
        output_dir=output_dir,
        apply_depth_enhancement=True,
        apply_material_enhancement=True,
    )

    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Output format: {args.format}")
    print(f"  Enhancement level: {args.enhancement_level}")
    print(f"  Output directory: {output_dir}")
    print(f"  Images to process: {len(analyzer.image_paths)}")

    # Check if model exists
    model_status = "✓ Found" if args.model.exists() else "✗ Not found (using fallback)"
    print(f"  Model status: {model_status}")

    inference = PicachoInference(config=config)

    # Process each image
    print("\nProcessing property images...")
    results = []
    total_time = 0

    for i, (image_path, analysis) in enumerate(zip(
        analyzer.image_paths, analyzer.analyses
    )):
        print(f"\n[{i + 1}/{len(analyzer.image_paths)}] {image_path.name}")

        # Get materials for this image
        materials = [m.value for m in analysis.materials.primary_materials]
        print(f"  Materials: {', '.join(materials)}")

        try:
            # Process image
            result = inference.process(
                image=image_path,
                materials=materials
            )

            # Save with appropriate format
            output_path = output_dir / f"{image_path.stem}_enhanced.tiff"
            saved_path = result.save(output_path, format=format_map[args.format])

            results.append({
                "source": str(image_path),
                "output": str(saved_path),
                "room_type": analysis.room_type.value,
                "materials": materials,
                "processing_time": result.processing_time,
                "resolution": list(result.resolution),
            })

            total_time += result.processing_time
            print(f"  ✓ Saved: {saved_path.name} ({result.processing_time:.2f}s)")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            logger.error(f"Failed to process {image_path.name}: {e}")

    # Save processing report
    report = {
        "property": "750 Picacho Lane",
        "timestamp": timestamp,
        "model_path": str(args.model),
        "output_format": args.format,
        "enhancement_level": args.enhancement_level,
        "total_images": len(results),
        "total_processing_time": total_time,
        "average_processing_time": total_time / len(results) if results else 0,
        "results": results,
    }

    report_path = output_dir / "processing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL OUTPUT PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n✓ Successfully processed: {len(results)}/{len(analyzer.image_paths)} images")
    print(f"✓ Total processing time: {total_time:.2f}s")
    print(f"✓ Average time per image: {total_time / len(results):.2f}s" if results else "")
    print(f"✓ Output format: {args.format}")
    print(f"✓ Deliverables saved to: {output_dir}")

    print("\nOutput files:")
    for result in results:
        output_name = Path(result["output"]).name
        room = result["room_type"]
        print(f"  • {output_name} ({room})")

    print(f"\n✓ Processing report: {report_path}")

    print("\n" + "=" * 60)
    print("750 PICACHO LANE TRAINING PROTOCOL COMPLETE")
    print("=" * 60)
    print("\nDeliverables ready for client review.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
