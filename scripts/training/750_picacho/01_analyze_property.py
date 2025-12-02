#!/usr/bin/env python3
"""
Script 01: Analyze Property Images for 750 Picacho Lane.

This script performs comprehensive analysis of all property images including:
- Material detection
- Color palette extraction
- Architectural feature identification
- Quality metrics assessment

Usage:
    python scripts/training/750_picacho/01_analyze_property.py [options]

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from training.property_specific.picacho_analyzer import PicachoAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze 750 Picacho Lane property images"
    )
    parser.add_argument(
        "--property-dir",
        type=Path,
        default=None,
        help="Path to property images directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/750_picacho/property_analysis.json"),
        help="Output path for analysis report"
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
    print("750 PICACHO LANE PROPERTY ANALYSIS")
    print("=" * 60 + "\n")

    # Initialize analyzer
    analyzer = PicachoAnalyzer(property_dir=args.property_dir)

    print(f"Property directory: {analyzer.property_dir}")
    print(f"Images found: {len(analyzer.image_paths)}")

    if not analyzer.image_paths:
        print("\n❌ No images found. Please check the property directory.")
        return 1

    # Run analysis
    print("\nAnalyzing property images...")
    report = analyzer.analyze_property()

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.save(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n✓ Total images analyzed: {report.total_images}")
    print(f"✓ Average quality score: {report.average_quality_score:.3f}")
    print(f"✓ Report saved to: {args.output}")

    print("\nMaterials detected:")
    for material, coverage in sorted(
        report.property_materials.items(), key=lambda x: -x[1]
    )[:5]:
        print(f"  • {material}: {coverage:.1%}")

    print("\nRoom distribution:")
    for room, count in report.room_distribution.items():
        print(f"  • {room}: {count}")

    print("\nRecommendations:")
    for rec in report.recommendations[:3]:
        print(f"  → {rec}")

    print("\nNext step: Run 02_synthesize_depth.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
