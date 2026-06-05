#!/usr/bin/env python3
"""
Ultimate Quality Pipeline for 750 Picacho Lane Renderings
Optimized for Apple M4 Max with MPS GPU acceleration
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tifffile
import torch
from PIL import Image

# Add transformation_portal to path
sys.path.insert(0, str(Path(__file__).parent))

from transformation_portal.utils.error_handling import safe_execute
from transformation_portal.utils.image_utils import load_image, save_image


def get_optimal_device() -> str:
    """Get the best available device for processing."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple M4 Max GPU
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def estimate_depth_mps(image: Image.Image, device: str = "mps") -> np.ndarray:
    """
    Estimate depth using Depth Anything V2 Large with MPS acceleration.
    Optimized for maximum quality on M4 Max.
    """
    from transformers import pipeline

    print(f"Loading Depth Anything V2 Large on {device}...")

    depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-h", device=device)

    print("Estimating depth...")
    result = depth_estimator(image)
    depth = result["depth"]

    # Convert to numpy array and normalize
    depth_array = np.array(depth, dtype=np.float32)
    depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

    return depth_array


def apply_depth_aware_clarity(image_array: np.ndarray, depth_map: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Apply depth-aware clarity enhancement.
    Stronger sharpening on foreground, gentler on background.
    """
    from scipy.ndimage import gaussian_filter

    # Create depth zones
    foreground_mask = depth_map > 0.7
    midground_mask = (depth_map >= 0.4) & (depth_map <= 0.7)
    background_mask = depth_map < 0.4

    # Apply gaussian blur for unsharp mask
    blurred = gaussian_filter(image_array, sigma=2.0)
    unsharp = image_array - blurred

    # Zone-based strength
    clarity_map = np.zeros_like(depth_map)
    clarity_map[foreground_mask] = strength * 1.5  # Strong on foreground
    clarity_map[midground_mask] = strength * 1.0  # Medium on midground
    clarity_map[background_mask] = strength * 0.5  # Gentle on background

    # Apply clarity
    result = image_array.copy()
    for c in range(3):
        result[:, :, c] += unsharp[:, :, c] * clarity_map

    return np.clip(result, 0, 1)


def apply_luxury_color_grade(image_array: np.ndarray) -> np.ndarray:
    """
    Apply luxury color grading optimized for architectural renders.
    """
    # Slightly cool highlights, warm shadows (luxury aesthetic)

    # Split into luminance and color
    luminance = 0.2126 * image_array[:, :, 0] + 0.7152 * image_array[:, :, 1] + 0.0722 * image_array[:, :, 2]

    # Color balance adjustment
    result = image_array.copy()

    # Cool highlights
    highlight_mask = luminance > 0.7
    result[highlight_mask, 2] *= 1.02  # Slight blue boost in highlights

    # Warm shadows
    shadow_mask = luminance < 0.3
    result[shadow_mask, 0] *= 1.03  # Slight red boost in shadows
    result[shadow_mask, 1] *= 1.01  # Slight green boost in shadows

    # Saturation boost in midtones
    midtone_mask = (luminance >= 0.3) & (luminance <= 0.7)
    saturation_boost = 1.08
    for c in range(3):
        result[midtone_mask, c] = (
            luminance[midtone_mask] + (result[midtone_mask, c] - luminance[midtone_mask]) * saturation_boost
        )

    return np.clip(result, 0, 1)


def apply_material_response(image_array: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """
    Apply material response enhancements based on depth and luminance.
    """
    from scipy.ndimage import gaussian_filter

    # Detect potential material areas based on local contrast and depth
    luminance = 0.2126 * image_array[:, :, 0] + 0.7152 * image_array[:, :, 1] + 0.0722 * image_array[:, :, 2]

    # Local contrast (potential material edges)
    lum_smooth = gaussian_filter(luminance, sigma=3.0)
    local_contrast = np.abs(luminance - lum_smooth)

    # Material areas: high local contrast in foreground/midground
    material_mask = (local_contrast > 0.02) & (depth_map > 0.4)

    # Enhance micro-contrast in material areas
    result = image_array.copy()
    detail = image_array - gaussian_filter(image_array, sigma=1.0, axes=(0, 1))

    for c in range(3):
        result[:, :, c][material_mask] += detail[:, :, c][material_mask] * 0.3

    return np.clip(result, 0, 1)


def process_ultimate_quality(input_path: Path, output_dir: Path, device: str = "mps") -> Dict[str, Any]:
    """
    Process image with ultimate quality settings.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {input_path.name}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Load image
    print("Loading image...")
    image = load_image(str(input_path))
    print(f"Image size: {image.size}")

    # Convert to float array
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Estimate depth
    depth_map = estimate_depth_mps(image, device=device)
    print(f"Depth map shape: {depth_map.shape}")

    # Save depth map
    depth_output = output_dir / f"{input_path.stem}_depth.png"
    depth_vis = (depth_map * 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(depth_output)
    print(f"Saved depth map: {depth_output}")

    # Apply enhancements
    print("\nApplying enhancements...")

    print("  1. Depth-aware clarity...")
    enhanced = apply_depth_aware_clarity(image_array, depth_map, strength=0.3)

    print("  2. Luxury color grade...")
    enhanced = apply_luxury_color_grade(enhanced)

    print("  3. Material response...")
    enhanced = apply_material_response(enhanced, depth_map)

    # Convert back to uint8
    result = (enhanced * 255).astype(np.uint8)
    result_image = Image.fromarray(result, mode="RGB")

    # Save outputs
    outputs = {}

    # TIFF (16-bit for maximum quality) - Use tifffile for proper 16-bit RGB
    tiff_output = output_dir / f"{input_path.stem}_ultimate.tif"
    result_16bit = (enhanced * 65535).astype(np.uint16)

    # tifffile properly handles 16-bit RGB TIFFs
    tifffile.imwrite(
        tiff_output,
        result_16bit,
        photometric="rgb",
        compression="lzw",
        metadata={"Software": "Transformation Portal Ultimate Quality Pipeline"},
    )
    outputs["tiff"] = tiff_output
    print(f"\nSaved 16-bit TIFF: {tiff_output}")

    # PNG (8-bit for preview)
    png_output = output_dir / f"{input_path.stem}_ultimate.png"
    result_image.save(png_output, format="PNG", compress_level=1)
    outputs["png"] = png_output
    print(f"Saved PNG: {png_output}")

    # JPEG (high quality for delivery)
    jpg_output = output_dir / f"{input_path.stem}_ultimate.jpg"
    result_image.save(jpg_output, format="JPEG", quality=98, subsampling=0)
    outputs["jpg"] = jpg_output
    print(f"Saved JPEG: {jpg_output}")

    return outputs


def main():
    """Process 750 Picacho renderings with ultimate quality."""

    # Setup
    device = get_optimal_device()
    print(f"Using device: {device}")

    if device == "mps":
        print("✓ Apple M4 Max GPU acceleration enabled")

    # Input/output directories
    input_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "16-Bit_EXRs"
    output_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Ultimate_Quality"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find TIFF files
    tiff_files = list(input_dir.glob("*.ti")) + list(input_dir.glob("*.tif"))

    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        # Try alternate location
        alt_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "TIFFs" / "_TIFFs"
        tiff_files = list(alt_dir.glob("*.ti")) + list(alt_dir.glob("*.tif"))

        if tiff_files:
            input_dir = alt_dir
            print(f"Found {len(tiff_files)} TIFF files in {input_dir}")

    print(f"\nFound {len(tiff_files)} images to process")

    # Process each image
    results = {}
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}]")

        try:
            outputs = process_ultimate_quality(tiff_file, output_dir, device=device)
            results[tiff_file.name] = {"status": "success", "outputs": {k: str(v) for k, v in outputs.items()}}
        except Exception as e:
            print(f"ERROR processing {tiff_file.name}: {e}")
            results[tiff_file.name] = {"status": "error", "error": str(e)}

    # Save processing log
    log_file = output_dir / "processing_log.json"
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"Output directory: {output_dir}")
    print(f"Processing log: {log_file}")
    print(f"{'='*80}\n")

    # Summary
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = len(results) - successful
    print(f"Successfully processed: {successful}")
    if failed > 0:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
