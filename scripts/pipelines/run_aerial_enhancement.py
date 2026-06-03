#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply MBAR board materials to aerial photographs.

Automated aerial photo enhancement pipeline using board material clustering.
Supports batch processing and configurable output formats.

Usage:
    python scripts/pipelines/run_aerial_enhancement.py
    python scripts/pipelines/run_aerial_enhancement.py --input aerial.tiff --output enhanced.jpg
    python scripts/pipelines/run_aerial_enhancement.py --batch input_dir/ --output-dir output/
    python scripts/pipelines/run_aerial_enhancement.py --resolution 2048 --materials 6

Features:
- Error handling for missing files and dependencies
- Batch processing with progress tracking
- Configurable resolution and material clustering
- Multiple output formats (JPG, PNG, TIFF)

Performance: ~2-5 seconds per 4K image (clustering), ~10-20 seconds (full enhancement)
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from transformation_portal.enhancers.board_material_aerial_enhancer import enhance_aerial
except ImportError as exc:
    print(f"Error: transformation_portal enhancer module not found: {exc}", file=sys.stderr)
    print("Run this script from the repository checkout or install the package first.", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhance aerial photographs with MBAR-approved materials",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input aerial photograph (TIFF, JPG, PNG)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output enhanced image path",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="Batch process all images in directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch processing",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=4096,
        help="Target width in pixels (maintains aspect ratio)",
    )
    parser.add_argument(
        "--materials",
        type=int,
        default=8,
        help="Number of material clusters (k-means)",
    )
    parser.add_argument(
        "--analysis-resolution",
        type=int,
        default=1280,
        help="Resolution for clustering analysis (lower = faster)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png", "tif"],
        default="jpg",
        help="Output format",
    )
    return parser.parse_args()


def process_single_image(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    """Process a single aerial image.

    Args:
        input_path: Input image path
        output_path: Output image path
        args: Parsed arguments

    Returns:
        Path to output file if successful, None otherwise
    """
    # Validate input
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return None

    if not input_path.suffix.lower() in [".tif", ".ti", ".jpg", ".jpeg", ".png"]:
        print(f"⚠️  Warning: Unexpected file format: {input_path.suffix}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process
    print(f"Processing: {input_path.name}")
    print(f"  Output: {output_path}")
    print(f"  Resolution: {args.resolution}px width")
    print(f"  Materials: {args.materials} clusters")

    try:
        result = enhance_aerial(
            input_path,
            output_path,
            analysis_max_dim=args.analysis_resolution,
            k=args.materials,
            seed=args.seed,
            target_width=args.resolution,
        )

        size_mb = result.stat().st_size / (1024**2)
        print(f"  ✅ Enhanced: {result.name} ({size_mb:.2f} MB)")
        return result

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return None


def process_batch(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    """Process all images in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        args: Parsed arguments

    Returns:
        List of successfully processed output paths
    """
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        return []

    # Find all images
    image_exts = {".tif", ".ti", ".jpg", ".jpeg", ".png"}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in image_exts]

    if not images:
        print(f"⚠️  No images found in {input_dir}")
        return []

    print("=" * 70)
    print("BATCH AERIAL ENHANCEMENT")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Images found: {len(images)}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, input_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}]")
        output_name = "{}_MBAR_Enhanced.{}".format(input_path.stem, args.format)
        output_path = output_dir / output_name

        result = process_single_image(input_path, output_path, args)
        if result:
            results.append(result)

    print("\n" + "=" * 70)
    print(f"✅ Batch complete: {len(results)}/{len(images)} successful")
    print("=" * 70)
    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Batch processing mode
    if args.batch:
        if not args.output_dir:
            print("❌ Error: --output-dir required for batch processing")
            sys.exit(1)
        process_batch(args.batch, args.output_dir, args)
        return

    # Single image mode
    # Use default paths if not specified (for backward compatibility)
    if not args.input:
        args.input = Path("/workspaces/800-Picacho-Lane-LUTs/input_images/RC-office750Picacho_Aerial.tif")
    if not args.output:
        args.output = Path("/workspaces/800-Picacho-Lane-LUTs/processed_images/750_Picacho_Aerial_MBAR_Enhanced.jpg")

    print("=" * 70)
    print("AERIAL ENHANCEMENT - MBAR Materials")
    print("=" * 70)

    result = process_single_image(args.input, args.output, args)

    if result:
        print("=" * 70)
        print("✅ Enhancement complete")
        print("=" * 70)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
