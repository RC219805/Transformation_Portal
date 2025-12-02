#!/usr/bin/env python3
"""
Script 02: Synthesize Depth Maps for 750 Picacho Lane.

This script generates high-quality depth maps using Depth Anything V2 Large
model ensemble for all property images.

Features:
- Multi-model ensemble for robust estimation
- Architectural priors for improved accuracy
- 16-bit PNG and float32 TIFF export

Usage:
    python scripts/training/750_picacho/02_synthesize_depth.py [options]

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
from training.property_specific.depth_synthesis import (
    DepthSynthesis,
    DepthSynthesisConfig,
    DepthModelVariant,
    DepthBackend
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synthesize depth maps for 750 Picacho Lane"
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
        default=Path("data/training_750picacho/depth"),
        help="Output directory for depth maps"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "base", "large"],
        default="large",
        help="Depth model variant"
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble (use single model)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "pytorch_mps", "pytorch_cuda", "pytorch_cpu"],
        default="auto",
        help="Compute backend"
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
    print("750 PICACHO LANE DEPTH SYNTHESIS")
    print("=" * 60 + "\n")

    # Initialize analyzer to get image paths
    analyzer = PicachoAnalyzer(property_dir=args.property_dir)

    if not analyzer.image_paths:
        print("❌ No images found. Please check the property directory.")
        return 1

    print(f"Found {len(analyzer.image_paths)} images to process")

    # Configure depth synthesis
    variant_map = {
        "small": DepthModelVariant.SMALL,
        "base": DepthModelVariant.BASE,
        "large": DepthModelVariant.LARGE,
    }

    backend_map = {
        "auto": None,
        "pytorch_mps": DepthBackend.PYTORCH_MPS,
        "pytorch_cuda": DepthBackend.PYTORCH_CUDA,
        "pytorch_cpu": DepthBackend.PYTORCH_CPU,
    }

    config = DepthSynthesisConfig(
        primary_model=variant_map[args.model],
        use_ensemble=not args.no_ensemble,
        output_16bit_png=True,
        output_float32_tiff=True,
        colorize_depth=True,
        output_dir=args.output_dir,
    )

    if backend_map[args.backend]:
        config.backend = backend_map[args.backend]

    # Initialize depth synthesis
    depth_synth = DepthSynthesis(config)

    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Ensemble: {not args.no_ensemble}")
    print(f"  Backend: {depth_synth.device}")
    print(f"  Output: {args.output_dir}")

    # Process all images
    print("\nSynthesizing depth maps...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = depth_synth.synthesize_all(
        images=analyzer.image_paths,
        output_dir=args.output_dir
    )

    # Summary
    print("\n" + "=" * 60)
    print("DEPTH SYNTHESIS COMPLETE")
    print("=" * 60)
    print(f"\n✓ Processed {len(results)} images")
    print(f"✓ Depth maps saved to: {args.output_dir}")

    print("\nDepth statistics:")
    for result in results:
        print(
            f"  • {result.source_path.name}: "
            f"min={result.min_depth:.3f}, max={result.max_depth:.3f}"
        )

    print("\nNext step: Run 03_generate_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
