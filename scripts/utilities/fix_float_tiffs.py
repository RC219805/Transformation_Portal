#!/usr/bin/env python3
"""
Fix float32 TIFF files with values outside [0,1] range.

Converts improperly saved float32 TIFFs to proper 16-bit uint16 TIFFs.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile


def fix_float_tiff(input_path: Path, output_path: Path = None, dry_run: bool = False):
    """
    Fix a float32 TIFF with values outside [0,1].

    Args:
        input_path: Path to problematic TIFF
        output_path: Optional output path (defaults to overwriting input)
        dry_run: If True, only report issues without fixing
    """
    if output_path is None:
        # Create backup and overwrite original
        backup_path = input_path.with_suffix(".tif.backup")
        output_path = input_path
    else:
        backup_path = None

    print(f"\nAnalyzing: {input_path.name}")
    print("=" * 70)

    with tifffile.TiffFile(input_path) as tif:
        page = tif.pages[0]
        data = page.asarray()

        print("Current format:")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Mean: {data.mean():.4f}")

        # Check for problems
        neg_pct = 100 * (data < 0).sum() / data.size
        over_pct = 100 * (data > 1.0).sum() / data.size

        print("\nIssues detected:")
        print(f"  Negative values: {neg_pct:.2f}%")
        print(f"  Values > 1.0: {over_pct:.2f}%")

        if data.dtype != np.float32 and data.dtype != np.float64:
            print(f"\n✓ File is already {data.dtype}, no fix needed")
            return False

        if neg_pct == 0 and over_pct == 0 and data.max() <= 1.0 and data.min() >= 0:
            print("\n✓ Float values are already in [0,1], converting to uint16")
        else:
            print("\n⚠️  Float values need clipping/normalization")

        if dry_run:
            print("\n[DRY RUN] - Would fix this file")
            return True

        # Fix the data
        # Clip to [0, 1] range
        data_clipped = np.clip(data, 0.0, 1.0)

        # Convert to 16-bit
        data_16bit = (data_clipped * 65535).astype(np.uint16)

        # Create backup if overwriting
        if backup_path:
            print(f"\nCreating backup: {backup_path.name}")
            input_path.replace(backup_path)

        # Save fixed version
        print(f"Saving fixed version: {output_path.name}")
        tifffile.imwrite(output_path, data_16bit, photometric="rgb", compression="lzw")

        # Verify
        new_size = output_path.stat().st_size / (1024**2)
        old_size = input_path.stat().st_size / (1024**2) if input_path.exists() else backup_path.stat().st_size / (1024**2)

        print("\n✓ Fixed!")
        print("  New format: uint16, 16-bit")
        print(f"  File size: {old_size:.1f} MB → {new_size:.1f} MB")

        return True


def main():
    """Fix all float TIFFs in TIFFs/_TIFFs directory."""
    tiff_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "TIFFs" / "_TIFFs"

    if not tiff_dir.exists():
        print(f"Error: Directory not found: {tiff_dir}")
        sys.exit(1)

    # Find all TIF files
    tiff_files = sorted(tiff_dir.glob("*.ti"))

    if not tiff_files:
        print(f"No .tif files found in {tiff_dir}")
        sys.exit(0)

    print(f"Found {len(tiff_files)} TIFF files")
    print("=" * 70)

    fixed_count = 0
    for tiff_file in tiff_files:
        try:
            was_fixed = fix_float_tiff(tiff_file, dry_run=False)
            if was_fixed:
                fixed_count += 1
        except Exception as e:
            print(f"\n✗ Error processing {tiff_file.name}: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"SUMMARY: Fixed {fixed_count}/{len(tiff_files)} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
