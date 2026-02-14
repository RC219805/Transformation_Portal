#!/usr/bin/env python3
"""Diagnose sky region degradation issue.

Analyzes depth maps to verify sky region depth values and identifies
the root cause of color grading regression.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def analyze_depth_map(depth_path: Path, name: str):
    """Analyze depth map statistics."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    # Load depth map
    depth_img = Image.open(depth_path)

    # Check mode
    print(f"Mode: {depth_img.mode}")
    print(f"Size: {depth_img.size}")

    # Convert to numpy
    if depth_img.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        depth = np.array(depth_img, dtype=np.uint16).astype(np.float32) / 65535.0
        print("Loaded as 16-bit depth map, normalized to [0, 1]")
    else:
        depth = np.array(depth_img.convert("L"), dtype=np.float32) / 255.0
        print("Loaded as 8-bit depth map, normalized to [0, 1]")

    # Statistics
    print(f"\nDepth Statistics:")
    print(f"  Min: {depth.min():.4f}")
    print(f"  Max: {depth.max():.4f}")
    print(f"  Mean: {depth.mean():.4f}")
    print(f"  Median: {np.median(depth):.4f}")
    print(f"  Std: {depth.std():.4f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p}: {np.percentile(depth, p):.4f}")

    # Zone analysis (matching enhancement.py logic)
    print(f"\nZone Analysis (Current Enhancement Logic):")
    foreground = depth > 0.7  # Current code: treats as FOREGROUND
    background = depth < 0.3  # Current code: treats as BACKGROUND
    midground = ~foreground & ~background

    total_pixels = depth.size
    print(f"  'Foreground' (depth > 0.7): {foreground.sum():,} pixels ({foreground.sum()/total_pixels*100:.1f}%)")
    print(f"  'Background' (depth < 0.3): {background.sum():,} pixels ({background.sum()/total_pixels*100:.1f}%)")
    print(f"  'Midground' (0.3-0.7): {midground.sum():,} pixels ({midground.sum()/total_pixels*100:.1f}%)")

    print(f"\n⚠️  CRITICAL ANALYSIS:")
    print(f"  In depth maps: LOW values = FAR, HIGH values = NEAR")
    print(f"  Sky regions should have LOW depth (far away)")
    print(f"  Buildings/objects should have HIGH depth (near camera)")
    print(f"")
    print(f"  Current enhancement.py logic:")
    print(f"    - foreground = depth_map > 0.7  ← Treats HIGH depth as foreground ✓")
    print(f"    - background = depth_map < 0.3  ← Treats LOW depth as background ✓")
    print(f"    - Applies fg_boost = 1.15 to foreground (HIGH depth)")
    print(f"    - Applies bg_compress = 0.92 to background (LOW depth)")
    print(f"")

    # Sample sky region (top 20% of image for aerial)
    if "Aerial" in name or "Pool" in name:
        sky_region_height = int(depth.shape[0] * 0.3)  # Top 30%
        sky_sample = depth[:sky_region_height, :]

        print(f"\nSky Region Analysis (top 30% of image):")
        print(f"  Sky depth mean: {sky_sample.mean():.4f}")
        print(f"  Sky depth median: {np.median(sky_sample):.4f}")
        print(f"  Sky depth range: [{sky_sample.min():.4f}, {sky_sample.max():.4f}]")

        # What percentage of sky is in "background" zone?
        sky_in_bg = (sky_sample < 0.3).sum() / sky_sample.size * 100
        sky_in_mg = ((sky_sample >= 0.3) & (sky_sample <= 0.7)).sum() / sky_sample.size * 100
        sky_in_fg = (sky_sample > 0.7).sum() / sky_sample.size * 100

        print(f"\n  Sky pixels in zones:")
        print(f"    Background zone (<0.3): {sky_in_bg:.1f}% ← Gets COMPRESSED by 0.92")
        print(f"    Midground zone (0.3-0.7): {sky_in_mg:.1f}% ← No change")
        print(f"    Foreground zone (>0.7): {sky_in_fg:.1f}% ← Gets BOOSTED by 1.15")

        if sky_in_bg > 50:
            print(f"\n  ✓ CORRECT: Most sky is in background zone (gets compressed)")
            print(f"     This is actually CORRECT behavior - sky should be subtle")
        else:
            print(f"\n  ⚠️  WARNING: Sky not primarily in background zone!")


def main():
    """Run diagnostic analysis."""
    parser = argparse.ArgumentParser(
        description="Diagnose sky detection and color grading issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default hardcoded paths (original investigation)
  python diagnose_sky_issue.py

  # Analyze a specific directory
  python diagnose_sky_issue.py --input depth_maps/ --output debug_sky/

  # Analyze specific depth map files
  python diagnose_sky_issue.py --input depth_maps/ --files aerial.png pool.png
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("depth_maps_apex"),
        help="Directory containing depth maps (default: depth_maps_apex)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for diagnostic results (optional)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["V2_750Picacho_Aerial_depth.png", "V2_750Picacho_Pool_depth.png"],
        help="Depth map filenames to analyze (default: Aerial and Pool images)",
    )

    args = parser.parse_args()
    depth_dir = args.input

    # Analyze key images with sky
    images = [(f, f.replace("_depth.png", "").replace("_", " ")) for f in args.files]

    for depth_file, name in images:
        depth_path = depth_dir / depth_file
        if depth_path.exists():
            analyze_depth_map(depth_path, name)
        else:
            print(f"\n⚠️  Depth map not found: {depth_path}")

    # Load and check JSON metadata (use first file as reference)
    print(f"\n{'='*60}")
    print("Depth Map Metadata Check")
    print(f"{'='*60}")

    # Try to find corresponding JSON for first depth file
    first_depth_file = args.files[0]
    json_file = first_depth_file.replace(".png", ".json")
    json_path = depth_dir / json_file

    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)

        print(f"\nDepth estimation metadata:")
        print(f"  Engine: {meta.get('engine')}")
        print(f"  Depth stats (raw):")
        stats = meta.get("depth_stats", {})
        for k, v in stats.items():
            print(f"    {k}: {v}")

        print(f"\n  PNG16 normalization:")
        norm = meta.get("outputs", {}).get("png16_normalization", {})
        for k, v in norm.items():
            print(f"    {k}: {v}")

        print(f"\n  NOTE: PNG visualization uses p01-p99 normalization")
        print(f"        This may affect the depth range loaded by enhancement")
    else:
        print(f"\n⚠️  No metadata JSON found at: {json_path}")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Output directory ready: {args.output}")


if __name__ == "__main__":
    main()
