#!/usr/bin/env python3
"""
750 Picacho Lane - Process from Canonical JPEG Sources
Processes the 6 canonical high-quality JPEG files with luxury enhancements
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def apply_luxury_enhancements(img: np.ndarray, scene_name: str) -> np.ndarray:
    """
    Apply subtle luxury enhancements:
    - Slight exposure lift in shadows
    - Micro-contrast enhancement
    - Color refinement
    """
    img_float = img.astype(np.float32)

    # 1. Subtle shadow lift (preserve highlights)
    shadows_mask = (img_float < 0.3).astype(np.float32)
    img_float += shadows_mask * 0.02 * (0.3 - img_float / 0.3)

    # 2. Micro-contrast (local)
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(img_float, sigma=2)
    img_float = img_float + 0.15 * (img_float - blurred)

    # 3. Saturation refinement (+3%)
    if img_float.shape[2] == 3:
        luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
        for c in range(3):
            img_float[:, :, c] = luminance + 1.03 * (img_float[:, :, c] - luminance)

    return np.clip(img_float, 0, 1).astype(np.float32)


def save_16bit_tiff_proper(img: np.ndarray, output_path: Path, compression: str = "lzw"):
    """
    Save as proper 16-bit TIFF using tifffile (not PIL).
    Input: float32 array [0-1]
    Output: uint16 TIFF [0-65535]
    """
    # Convert to 16-bit
    img_16bit = (img * 65535.0).astype(np.uint16)

    # Save with tifffile for proper 16-bit handling
    tifffile.imwrite(output_path, img_16bit, compression=compression, photometric="rgb")

    return img_16bit


def process_single_jpeg(input_path: Path, output_dir: Path, formats: list = ["jpeg", "png", "tiff"]):
    """Process a single JPEG with luxury enhancements."""

    print(f"\n{'='*80}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*80}\n")

    # Load JPEG
    img = Image.open(input_path).convert("RGB")
    img = np.array(img)
    print(f"✓ Loaded JPEG: {img.shape[1]}x{img.shape[0]}")

    # Convert to float
    img_float = img.astype(np.float32) / 255.0

    # Apply enhancements
    scene_name = input_path.stem
    img_enhanced = apply_luxury_enhancements(img_float, scene_name)
    print(f"✓ Applied luxury enhancements for {scene_name}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    base_name = input_path.stem

    # Save in requested formats
    if "tiff" in formats:
        tiff_path = output_dir / f"{base_name}_luxury.ti"
        img_16bit = save_16bit_tiff_proper(img_enhanced, tiff_path)
        outputs["tiff"] = tiff_path
        print(f"✓ Saved 16-bit TIFF: {tiff_path.name}")
        print(f"  - Shape: {img_16bit.shape}")
        print(f"  - Range: [{img_16bit.min()}, {img_16bit.max()}]")
        print(f"  - Size: {tiff_path.stat().st_size / 1024 / 1024:.2f} MB")

    if "jpeg" in formats:
        jpeg_path = output_dir / f"{base_name}_luxury.jpg"
        img_uint8 = (img_enhanced * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode="RGB")
        img_pil.save(jpeg_path, quality=95, optimize=True)
        outputs["jpeg"] = jpeg_path
        print(f"✓ Saved JPEG: {jpeg_path.name}")

    if "png" in formats:
        png_path = output_dir / f"{base_name}_luxury.png"
        img_uint8 = (img_enhanced * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode="RGB")
        img_pil.save(png_path, optimize=True)
        outputs["png"] = png_path
        print(f"✓ Saved PNG: {png_path.name}")

    print(f"\n✅ Completed: {input_path.name}")
    print(f"   Outputs: {len(outputs)} files in {output_dir.name}/\n")

    return outputs


def main():
    """Process the 6 canonical 750 Picacho Lane JPEG sources."""

    # Define the 6 canonical source files
    source_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "JPEGs"
    output_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Final_Production"

    canonical_files = [
        "750Picacho_Aerial.jpg",
        "750Picacho_GreatRoom.jpg",
        "750Picacho_Kitchen.jpg",
        "750Picacho_Pool.jpg",
        "750Picacho_PrimaryBathroom.jpg",
        "750Picacho_PrimaryBedroom.jpg",
    ]

    print("#" * 80)
    print("  750 PICACHO LANE - CANONICAL JPEG PROCESSING")
    print("  Maximum Quality from 6 Source Files")
    print("#" * 80)
    print(f"\nSource: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(canonical_files)} canonical sources\n")

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return 1

    # Verify all 6 files exist
    missing = []
    for filename in canonical_files:
        if not (source_dir / filename).exists():
            missing.append(filename)

    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"   - {f}")
        return 1

    # Process each file
    total_outputs = 0
    for i, filename in enumerate(canonical_files, 1):
        print(f"\n[{i}/{len(canonical_files)}] " + "=" * 70)
        input_path = source_dir / filename
        outputs = process_single_jpeg(input_path, output_dir, formats=["jpeg", "png", "tiff"])
        total_outputs += len(outputs)

    print("\n" + "#" * 80)
    print("  PROCESSING COMPLETE")
    print("#" * 80)
    print(f"\nTotal files processed: {len(canonical_files)}")
    print(f"Total outputs created: {total_outputs}")
    print(f"\nOutputs saved to: {output_dir}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
