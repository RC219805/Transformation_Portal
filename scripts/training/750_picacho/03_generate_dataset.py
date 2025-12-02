#!/usr/bin/env python3
"""
Script 03: Generate Training Dataset for 750 Picacho Lane.

This script generates augmented training samples from property images:
- Multi-scale crops (512, 1024, 2048)
- Depth-image correspondence
- Material-aware augmentation

Usage:
    python scripts/training/750_picacho/03_generate_dataset.py [options]

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
from training.property_specific.depth_synthesis import DepthSynthesis
from training.property_specific.dataset_generator import DatasetGenerator, DatasetConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate training dataset for 750 Picacho Lane"
    )
    parser.add_argument(
        "--property-dir",
        type=Path,
        default=None,
        help="Path to property images directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training_750picacho"),
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=600,
        help="Total number of training samples to generate"
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Skip depth map generation/inclusion"
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Skip augmentation (raw crops only)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
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
    print("750 PICACHO LANE DATASET GENERATION")
    print("=" * 60 + "\n")

    # Initialize components
    analyzer = PicachoAnalyzer(property_dir=args.property_dir)

    if not analyzer.image_paths:
        print("❌ No images found. Please check the property directory.")
        return 1

    # Analyze property
    print("Analyzing property images...")
    analyzer.analyze_property()

    # Initialize depth synthesis if needed
    depth_synth = None
    if not args.no_depth:
        print("\nInitializing depth synthesis...")
        depth_synth = DepthSynthesis()

    # Configure dataset generation
    config = DatasetConfig(
        output_dir=args.output_dir,
        total_samples=args.num_samples,
        random_seed=args.seed,
        include_depth=not args.no_depth,
        augmentation_enabled=not args.no_augmentation,
    )

    print("\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Total samples: {args.num_samples}")
    print(f"  Include depth: {not args.no_depth}")
    print(f"  Augmentation: {not args.no_augmentation}")
    print(f"  Random seed: {args.seed}")

    # Generate dataset
    generator = DatasetGenerator(
        analyzer=analyzer,
        depth_synthesis=depth_synth,
        config=config
    )

    print("\nGenerating training samples...")
    samples = generator.generate_dataset(num_samples=args.num_samples)

    print("\nSaving dataset...")
    metadata = generator.save_dataset(output_dir=args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n✓ Total samples generated: {len(samples)}")
    print(f"✓ Train samples: {metadata['splits']['train']['count']}")
    print(f"✓ Validation samples: {metadata['splits']['val']['count']}")
    print(f"✓ Test samples: {metadata['splits']['test']['count']}")
    print(f"✓ Dataset saved to: {args.output_dir}")

    print("\nRoom types covered:")
    for room in set(s.room_type for s in samples):
        count = sum(1 for s in samples if s.room_type == room)
        print(f"  • {room}: {count} samples")

    print("\nMaterials covered:")
    all_materials = set(mat for s in samples for mat in s.materials)
    for material in all_materials:
        print(f"  • {material}")

    print("\nNext step: Run 04_train_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
