#!/usr/bin/env python3
"""Convert all 8-bit TIFFs to proper 16-bit TIFFs."""

import sys
from pathlib import Path

import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utilities.fix_tiff_16bit import convert_8bit_to_16bit_tiff

# Find all TIFF files
output_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "TIFFs" / "_TIFFs"
tiff_files = list(output_dir.glob("*.ti")) + list(output_dir.glob("*.tif"))

print(f"Found {len(tiff_files)} TIFF files to convert\n")

for tiff_path in sorted(tiff_files):
    # Skip if already has _16bit suffix
    if "_16bit" in tiff_path.stem:
        continue

    # Convert in-place (overwrite original)
    print(f"\n{'='*80}")
    print(f"Converting: {tiff_path.name}")
    print(f"{'='*80}")

    try:
        # Read original
        img_array = tifffile.imread(tiff_path)

        # Save as 16-bit (overwrite)
        from scripts.utilities.fix_tiff_16bit import save_16bit_tiff_tifffile

        save_16bit_tiff_tifffile(img_array, tiff_path, compression="lzw")

        # Verify
        verify_array = tifffile.imread(tiff_path)
        if verify_array.dtype == "uint16":
            print(f"✅ VERIFIED: {tiff_path.name} is now 16-bit")
        else:
            print(f"❌ ERROR: {tiff_path.name} is still {verify_array.dtype}")

    except Exception as e:
        print(f"❌ ERROR converting {tiff_path.name}: {e}")

print(f"\n\n{'='*80}")
print("CONVERSION COMPLETE")
print(f"{'='*80}")
