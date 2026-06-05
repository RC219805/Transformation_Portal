#!/usr/bin/env python3
"""
Verify TIFF file quality - check bit depth and data integrity
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def verify_tiff(tiff_path: Path):
    """Verify a TIFF file is properly 16-bit RGB."""
    print(f"\n{'='*80}")
    print(f"Verifying: {tiff_path.name}")
    print(f"{'='*80}\n")

    # Load with tifffile
    print("Loading with tifffile...")
    img_array = tifffile.imread(tiff_path)

    print(f"✓ Data type: {img_array.dtype}")
    print(f"✓ Shape: {img_array.shape}")
    print(f"✓ Min value: {img_array.min()}")
    print(f"✓ Max value: {img_array.max()}")
    print(f"✓ Mean value: {img_array.mean():.2f}")

    # Verify it's 16-bit
    if img_array.dtype == np.uint16:
        print("\n✅ CORRECT: File is properly 16-bit")

        # Check if full range is being used
        if img_array.max() > 256:
            print("✅ CORRECT: Using full 16-bit range (values > 256)")
        else:
            print("⚠️  WARNING: Values seem to be in 8-bit range only")

    elif img_array.dtype == np.uint8:
        print("\n❌ ERROR: File is only 8-bit!")
    else:
        print(f"\n⚠️  WARNING: Unexpected data type: {img_array.dtype}")

    # Check dimensions
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        print("✅ CORRECT: RGB image (3 channels)")
    elif len(img_array.shape) == 2:
        print("⚠️  WARNING: Grayscale image")
    else:
        print(f"⚠️  WARNING: Unexpected shape: {img_array.shape}")

    # Compare with PIL loading
    print("\nComparing with PIL Image loading...")
    try:
        pil_img = Image.open(tiff_path)
        print(f"PIL mode: {pil_img.mode}")
        print(f"PIL size: {pil_img.size}")

        if pil_img.mode == "RGB":
            print("⚠️  Note: PIL shows as 'RGB' (8-bit), but tifffile confirms 16-bit")
        elif pil_img.mode == "I;16":
            print("✓ PIL confirms 16-bit mode")
    except Exception as e:
        print(f"Error loading with PIL: {e}")

    # File size check
    file_size_mb = tiff_path.stat().st_size / (1024 * 1024)
    print(f"\nFile size: {file_size_mb:.2f} MB")

    # Estimate expected size for 16-bit RGB
    h, w = img_array.shape[:2]
    expected_size_mb = (h * w * 3 * 2) / (1024 * 1024)  # 2 bytes per channel
    print(f"Expected size (uncompressed): {expected_size_mb:.2f} MB")

    compression_ratio = expected_size_mb / file_size_mb if file_size_mb > 0 else 0
    print(f"Compression ratio: {compression_ratio:.2f}x")

    print(f"\n{'='*80}\n")

    return img_array.dtype == np.uint16 and len(img_array.shape) == 3


def main():
    """Verify TIFF files in output directory."""

    # Check multiple possible locations
    search_paths = [
        Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Ultimate_Quality",
        Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "TIFFs" / "_TIFFs",
    ]

    tiff_files = []
    for search_path in search_paths:
        if search_path.exists():
            tiff_files.extend(list(search_path.glob("*.ti")))
            tiff_files.extend(list(search_path.glob("*.tif")))

    if not tiff_files:
        print("No TIFF files found to verify.")
        print("\nSearched in:")
        for p in search_paths:
            print(f"  - {p}")
        return

    print(f"\nFound {len(tiff_files)} TIFF files to verify\n")

    results = []
    for tiff_file in tiff_files[:5]:  # Check first 5 files
        is_valid = verify_tiff(tiff_file)
        results.append((tiff_file.name, is_valid))

    # Summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}\n")

    valid_count = sum(1 for _, is_valid in results if is_valid)
    print(f"Valid 16-bit RGB TIFFs: {valid_count}/{len(results)}")

    for name, is_valid in results:
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    main()
