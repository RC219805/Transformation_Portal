#!/usr/bin/env python3
"""
Process 750 Picacho Lane luxury estate renderings.
Handles 16-bit EXR input with proper tone mapping and outputs high-quality TIFFs + JPEGs.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def load_exr_with_tone_mapping(exr_path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load EXR file and apply proper tone mapping for luxury real estate.

    Args:
        exr_path: Path to EXR file
        target_size: Optional (width, height) for resizing

    Returns:
        Float32 array in [0, 1] range, tone-mapped
    """
    print(f"  Loading EXR: {exr_path.name}")

    # Load EXR using OpenEXR library
    import Imath
    import OpenEXR

    # Open the EXR file
    exr_file = OpenEXR.InputFile(str(exr_path))

    # Get the header
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read RGB channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ["R", "G", "B"]

    # Read channel data
    channel_data = {}
    for channel in channels:
        if channel in header["channels"]:
            channel_data[channel] = exr_file.channel(channel, FLOAT)
        else:
            # Fallback: try lowercase
            channel_lower = channel.lower()
            if channel_lower in header["channels"]:
                channel_data[channel] = exr_file.channel(channel_lower, FLOAT)
            else:
                print(f"    Warning: Channel {channel} not found, using zeros")
                channel_data[channel] = b"\x00" * (width * height * 4)

    # Convert to numpy arrays
    r = np.frombuffer(channel_data["R"], dtype=np.float32).reshape(height, width)
    g = np.frombuffer(channel_data["G"], dtype=np.float32).reshape(height, width)
    b = np.frombuffer(channel_data["B"], dtype=np.float32).reshape(height, width)

    # Stack into RGB image
    img = np.stack([r, g, b], axis=-1)

    # Ensure float32
    img = img.astype(np.float32)

    print(f"    Raw range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"    Shape: {img.shape}, dtype: {img.dtype}")

    # Handle HDR tone mapping with exposure adjustment
    # Conservative tone mapping for luxury real estate

    # 1. Exposure adjustment (slight boost for interior scenes)
    exposure_boost = 1.15  # Subtle lift
    img = img * exposure_boost

    # 2. Filmic tone mapping (preserves highlights better than simple clipping)
    # Using a smooth S-curve response
    def filmic_tonemap(
        x,
        shoulder_strength=0.22,
        linear_strength=0.3,
        linear_angle=0.1,
        toe_strength=0.2,
        toe_numerator=0.01,
        toe_denominator=0.3,
        linear_white=11.2,
    ):
        """Filmic tone mapping operator (Hable/Uncharted 2)"""

        def tonemap_curve(x):
            return (
                (x * (shoulder_strength * x + linear_angle * linear_strength) + toe_strength * toe_numerator)
                / (x * (shoulder_strength * x + linear_strength) + toe_strength * toe_denominator)
            ) - toe_numerator / toe_denominator

        curr = tonemap_curve(x)
        white_scale = 1.0 / tonemap_curve(linear_white)
        return curr * white_scale

    # Apply filmic tone mapping
    img_tonemapped = filmic_tonemap(img)

    # 3. Gentle highlight compression for very bright areas
    img_tonemapped = np.clip(img_tonemapped, 0, 1)

    # 4. Subtle gamma correction (sRGB-like but gentler)
    gamma = 1.0 / 2.0  # Slightly less aggressive than 2.2
    img_tonemapped = np.power(img_tonemapped, gamma)

    print(f"    Tone-mapped range: [{img_tonemapped.min():.3f}, {img_tonemapped.max():.3f}]")

    # Resize if requested
    if target_size is not None:
        from PIL import Image

        # Convert to uint8 for PIL processing
        img_uint8 = (img_tonemapped * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        img_tonemapped = np.array(pil_img).astype(np.float32) / 255.0

    return img_tonemapped


def save_16bit_tiff(img_float: np.ndarray, output_path: Path, compression: str = "lzw"):
    """
    Save float array as 16-bit TIFF with proper scaling.

    Args:
        img_float: Float32 array in [0, 1] range
        output_path: Output TIFF path
        compression: Compression method (lzw, deflate, none)
    """
    # Convert to 16-bit
    img_16bit = (np.clip(img_float, 0, 1) * 65535).astype(np.uint16)

    # Save with tifffile for best quality
    try:
        import tifffile

        tifffile.imwrite(output_path, img_16bit, compression=compression, photometric="rgb", planarconfig="contig")
        print(f"    Saved 16-bit TIFF: {output_path.name} ({output_path.stat().st_size / (1024*1024):.1f} MB)")
    except ImportError:
        # Fallback to imageio if available
        try:
            import imageio.v3 as iio  # noqa: F401

            iio.imwrite(output_path, img_16bit, compression=1 if compression != "none" else 0)
            print(f"    Saved 16-bit TIFF (imageio): {output_path.name}")
        except ImportError as exc:
            raise ImportError("Neither tifffile nor imageio available for TIFF writing") from exc


def save_jpeg(img_float: np.ndarray, output_path: Path, quality: int = 95):
    """
    Save float array as high-quality JPEG.

    Args:
        img_float: Float32 array in [0, 1] range
        output_path: Output JPEG path
        quality: JPEG quality (0-100)
    """
    # Convert to 8-bit
    img_8bit = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)

    # Save with PIL for quality control
    pil_img = Image.fromarray(img_8bit)
    pil_img.save(output_path, "JPEG", quality=quality, optimize=True, subsampling=0)
    print(f"    Saved JPEG: {output_path.name} ({output_path.stat().st_size / (1024*1024):.1f} MB)")


def apply_luxury_adjustments(img: np.ndarray, scene_type: str = "interior") -> np.ndarray:
    """
    Apply subtle luxury real estate adjustments.

    Args:
        img: Float32 array in [0, 1] range
        scene_type: 'interior', 'exterior', 'aerial', 'pool'

    Returns:
        Adjusted float32 array
    """
    # Scene-specific adjustments
    adjustments = {
        "interior": {
            "brightness": 1.02,  # Very subtle lift
            "contrast": 1.05,
            "saturation": 1.03,
            "warmth": 1.01,  # Slight warm bias
        },
        "exterior": {"brightness": 1.0, "contrast": 1.08, "saturation": 1.08, "warmth": 1.02},
        "aerial": {"brightness": 1.0, "contrast": 1.12, "saturation": 1.12, "warmth": 1.0},
        "pool": {"brightness": 1.03, "contrast": 1.06, "saturation": 1.10, "warmth": 0.98},  # Boost blues  # Slightly cool
    }

    params = adjustments.get(scene_type, adjustments["interior"])

    # Apply brightness
    img = img * params["brightness"]

    # Apply contrast (around midpoint)
    img = ((img - 0.5) * params["contrast"]) + 0.5

    # Apply saturation
    # Convert to HSV-like space
    luminance = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    luminance = luminance[..., np.newaxis]

    # Blend between grayscale and color
    sat_factor = params["saturation"]
    img = luminance + (img - luminance) * sat_factor

    # Apply warmth (shift red/blue balance)
    if params["warmth"] != 1.0:
        img[..., 0] = img[..., 0] * params["warmth"]  # Red
        img[..., 2] = img[..., 2] / params["warmth"]  # Blue

    # Clip to valid range
    img = np.clip(img, 0, 1)

    return img


def detect_scene_type(filename: str) -> str:
    """Detect scene type from filename."""
    filename_lower = filename.lower()

    if "aerial" in filename_lower:
        return "aerial"
    elif "pool" in filename_lower:
        return "pool"
    elif any(x in filename_lower for x in ["kitchen", "greatroom", "bedroom", "bathroom", "interior"]):
        return "interior"
    else:
        return "exterior"


def process_single_exr(exr_path: Path, output_dir: Path, create_preview: bool = True):
    """
    Process single EXR file into high-quality TIFF and JPEG.

    Args:
        exr_path: Path to input EXR
        output_dir: Output directory
        create_preview: Also create JPEG preview
    """
    print(f"\nProcessing: {exr_path.name}")

    # Detect scene type
    scene_type = detect_scene_type(exr_path.stem)
    print(f"  Scene type: {scene_type}")

    # Load and tone map
    img = load_exr_with_tone_mapping(exr_path)

    # Apply luxury adjustments
    img = apply_luxury_adjustments(img, scene_type)

    # Create output paths
    tiff_dir = output_dir / "Master_TIFFs"
    jpeg_dir = output_dir / "Web_JPEGs"
    tiff_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir.mkdir(parents=True, exist_ok=True)

    base_name = exr_path.stem
    tiff_path = tiff_dir / f"{base_name}_Master.ti"
    jpeg_path = jpeg_dir / f"{base_name}_Web.jpg"

    # Save outputs
    save_16bit_tiff(img, tiff_path, compression="lzw")

    if create_preview:
        save_jpeg(img, jpeg_path, quality=95)

    print(f"  ✓ Completed: {base_name}")


def main():
    """Main processing function."""
    # Setup paths
    exr_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "16-Bit_EXRs"
    output_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Processed_Output"

    print("=" * 80)
    print("750 Picacho Lane - Luxury Estate Rendering Pipeline")
    print("=" * 80)

    # Get all EXR files
    exr_files = sorted(exr_dir.glob("*.exr"))

    if not exr_files:
        print("ERROR: No EXR files found!")
        return 1

    print(f"\nFound {len(exr_files)} EXR files to process")
    print(f"Output directory: {output_dir}")

    # Process each file
    success_count = 0
    for exr_path in exr_files:
        try:
            process_single_exr(exr_path, output_dir, create_preview=True)
            success_count += 1
        except Exception as e:
            print(f"  ✗ ERROR processing {exr_path.name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print(f"Processing complete: {success_count}/{len(exr_files)} files")
    print(f"Master TIFFs: {output_dir / 'Master_TIFFs'}")
    print(f"Web JPEGs: {output_dir / 'Web_JPEGs'}")
    print("=" * 80)

    return 0 if success_count == len(exr_files) else 1


if __name__ == "__main__":
    sys.exit(main())
