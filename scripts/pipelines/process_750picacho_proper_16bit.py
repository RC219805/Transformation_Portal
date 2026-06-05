#!/usr/bin/env python3
"""
Proper 16-bit TIFF conversion from EXR sources for 750 Picacho Lane
Preserves full dynamic range and applies luxury real estate grading
"""
from pathlib import Path
from typing import Optional

import Imath
import numpy as np
import OpenEXR
import typer
from PIL import Image

app = typer.Typer()


def load_exr_as_16bit(exr_path: Path) -> np.ndarray:
    """Load EXR and convert to 16-bit preserving dynamic range"""
    exr_file = OpenEXR.InputFile(str(exr_path))
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read RGB channels as float32
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = {}
    for c in ["R", "G", "B"]:
        channel_str = exr_file.channel(c, FLOAT)
        channel = np.frombuffer(channel_str, dtype=np.float32)
        channel = channel.reshape((height, width))
        channels[c] = channel

    # Stack into RGB array
    rgb = np.stack([channels["R"], channels["G"], channels["B"]], axis=-1)

    print(f"  EXR loaded: {width}x{height}, range [{rgb.min():.3f}, {rgb.max():.3f}]")

    # Tone map for luxury real estate (preserve highlights, lift shadows slightly)
    # Apply gentle S-curve to maintain visual appeal
    rgb_clipped = np.clip(rgb, 0, None)

    # Gentle exposure adjustment if needed (preserve EXR tonality)
    exposure_factor = 1.0
    if rgb_clipped.max() < 0.5:
        exposure_factor = 0.8 / np.percentile(rgb_clipped, 98)
        print(f"  Applying exposure boost: {exposure_factor:.2f}x")

    rgb_exposed = rgb_clipped * exposure_factor

    # Soft clipping at highlights
    rgb_tone = np.where(rgb_exposed < 1.0, rgb_exposed, 1.0 - np.exp(-(rgb_exposed - 1.0)))

    # Convert to 16-bit integer (0-65535 range)
    rgb_16bit = (np.clip(rgb_tone, 0, 1) * 65535).astype(np.uint16)

    print(f"  Converted to 16-bit: range [{rgb_16bit.min()}, {rgb_16bit.max()}]")

    return rgb_16bit


def apply_luxury_grade_16bit(image: np.ndarray) -> np.ndarray:
    """Apply luxury real estate color grading in 16-bit space"""
    img_float = image.astype(np.float32) / 65535.0

    # Subtle warmth (golden hour aesthetic)
    warm_r = img_float[:, :, 0] * 1.02
    warm_g = img_float[:, :, 1] * 1.00
    warm_b = img_float[:, :, 2] * 0.98

    img_float = np.stack([warm_r, warm_g, warm_b], axis=-1)

    # Gentle contrast boost (preserve highlights and shadows)
    mid = 0.5
    contrast = 1.08
    img_float = (img_float - mid) * contrast + mid

    # Clip and convert back
    img_float = np.clip(img_float, 0, 1)
    return (img_float * 65535).astype(np.uint16)


@app.command()
def process_view(
    view_name: str,
    exr_dir: Path = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "16-Bit_EXRs",
    output_dir: Path = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Master_TIFFs_16bit",
    apply_grade: bool = True,
):
    """Process a single view from EXR to proper 16-bit TIFF"""

    # Find EXR file
    exr_path = exr_dir / f"{view_name}.exr"
    if not exr_path.exists():
        print(f"❌ EXR not found: {exr_path}")
        raise typer.Exit(1)

    print(f"\n🎨 Processing: {view_name}")
    print(f"  Source: {exr_path.name}")

    # Load EXR as 16-bit
    rgb_16bit = load_exr_as_16bit(exr_path)

    # Apply luxury grading if requested
    if apply_grade:
        print("  Applying luxury color grade...")
        rgb_16bit = apply_luxury_grade_16bit(rgb_16bit)

    # Save as 16-bit TIFF with proper metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{view_name}.tiff"

    # Use tifffile for proper 16-bit RGB TIFF
    try:
        import tifffile

        tifffile.imwrite(
            output_path,
            rgb_16bit,
            photometric="rgb",
            compression="lzw",
            metadata={
                "Software": "Transformation Portal",
                "ImageDescription": f"750 Picacho Lane - {view_name} - 16-bit Master",
            },
        )
        print(f"  Saved: {output_path.name}")

        # Verify the save (use tifffile to properly read 16-bit)
        verify_arr = tifffile.imread(output_path)
        print(
            f"  Verification: dtype={verify_arr.dtype}, "
            f"range=[{verify_arr.min()}, {verify_arr.max()}], shape={verify_arr.shape}"
        )

        if verify_arr.dtype == np.uint16 and verify_arr.max() > 255:
            print("  ✅ Successfully saved as 16-bit TIFF")
        else:
            print(f"  ⚠️  Warning: Unexpected format: {verify_arr.dtype}")

    except ImportError:
        print("  ❌ tifffile not available")
        print("     Install with: pip install tifffile imagecodecs")
        raise

    return output_path


@app.command()
def process_all(
    exr_dir: Path = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "16-Bit_EXRs",
    output_dir: Path = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Master_TIFFs_16bit",
    apply_grade: bool = True,
):
    """Process all EXR files to proper 16-bit TIFFs"""

    exr_files = list(exr_dir.glob("*.exr"))

    if not exr_files:
        print(f"❌ No EXR files found in {exr_dir}")
        raise typer.Exit(1)

    print(f"\n📊 Found {len(exr_files)} EXR files")
    print(f"   Output: {output_dir}\n")

    for exr_path in sorted(exr_files):
        view_name = exr_path.stem
        try:
            process_view(view_name, exr_dir, output_dir, apply_grade)
        except Exception as e:
            print(f"❌ Error processing {view_name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n✅ Processing complete!")
    print(f"   Output directory: {output_dir}")


if __name__ == "__main__":
    app()
