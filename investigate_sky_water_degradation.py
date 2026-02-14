#!/usr/bin/env python3
"""
Investigate color degradation in sky/water regions.

Compares input vs output images for the Aerial and Pool images to detect
color shifts, saturation changes, or brightness degradation in sky/water regions.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_16bit_tiff(path: Path) -> np.ndarray:
    """Load 16-bit TIFF and convert to float [0,1] for analysis."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    # Normalize to [0,1]
    if arr.max() > 255:
        arr = arr / 65535.0
    else:
        arr = arr / 255.0
    return arr


def analyze_region(image: np.ndarray, mask: np.ndarray = None, region_name: str = "region") -> dict:
    """Analyze color statistics for a region."""
    if mask is not None:
        pixels = image[mask > 0.5]
    else:
        pixels = image.reshape(-1, 3)

    if len(pixels) == 0:
        return {
            "region": region_name,
            "pixel_count": 0,
            "mean_rgb": [0, 0, 0],
            "std_rgb": [0, 0, 0],
            "mean_brightness": 0,
            "mean_saturation": 0,
        }

    # RGB statistics
    mean_rgb = pixels.mean(axis=0).tolist()
    std_rgb = pixels.std(axis=0).tolist()

    # Brightness (mean of RGB)
    brightness = pixels.mean(axis=1)
    mean_brightness = float(brightness.mean())
    std_brightness = float(brightness.std())

    # Saturation (max-min of RGB per pixel)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)
    mean_saturation = float(saturation.mean())
    std_saturation = float(saturation.std())

    return {
        "region": region_name,
        "pixel_count": len(pixels),
        "mean_rgb": mean_rgb,
        "std_rgb": std_rgb,
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "mean_saturation": mean_saturation,
        "std_saturation": std_saturation,
    }


def create_sky_mask_top_region(image_shape: tuple, top_fraction: float = 0.3) -> np.ndarray:
    """Create a mask for the top portion of the image (likely sky)."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    top_rows = int(h * top_fraction)
    mask[:top_rows, :] = 1.0
    return mask


def create_water_mask_bottom_region(image_shape: tuple, bottom_fraction: float = 0.3) -> np.ndarray:
    """Create a mask for the bottom portion of the image (likely pool/water)."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    bottom_rows = int(h * bottom_fraction)
    mask[-bottom_rows:, :] = 1.0
    return mask


def detect_sky_pixels(image: np.ndarray, blue_threshold: float = 0.4, brightness_threshold: float = 0.3) -> np.ndarray:
    """Detect sky-like pixels based on color characteristics."""
    # Sky is typically blue-ish and bright
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Sky criteria: blue channel dominant, relatively bright
    is_blue = (b > r) & (b > g)
    is_bright = ((r + g + b) / 3.0) > brightness_threshold
    is_blue_enough = b > blue_threshold

    sky_mask = (is_blue & is_bright & is_blue_enough).astype(np.float32)
    return sky_mask


def detect_water_pixels(image: np.ndarray, blue_threshold: float = 0.25) -> np.ndarray:
    """Detect water-like pixels based on color characteristics."""
    # Water is typically blue-ish or cyan-ish
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Water criteria: blue channel strong, may be darker than sky
    is_blue_dominant = (b >= r) & (b >= g)
    is_blue_enough = b > blue_threshold
    not_too_dark = ((r + g + b) / 3.0) > 0.15

    water_mask = (is_blue_dominant & is_blue_enough & not_too_dark).astype(np.float32)
    return water_mask


def compare_regions(input_stats: dict, output_stats: dict) -> dict:
    """Compare input vs output statistics and report degradation."""
    delta = {
        "region": input_stats["region"],
        "pixel_count": input_stats["pixel_count"],
        "delta_mean_rgb": [output_stats["mean_rgb"][i] - input_stats["mean_rgb"][i] for i in range(3)],
        "delta_brightness": output_stats["mean_brightness"] - input_stats["mean_brightness"],
        "delta_saturation": output_stats["mean_saturation"] - input_stats["mean_saturation"],
        "brightness_change_pct": (
            100
            * (output_stats["mean_brightness"] - input_stats["mean_brightness"])
            / max(input_stats["mean_brightness"], 0.001)
        ),
        "saturation_change_pct": (
            100
            * (output_stats["mean_saturation"] - input_stats["mean_saturation"])
            / max(input_stats["mean_saturation"], 0.001)
        ),
    }
    return delta


def main():
    # Paths
    input_dir = Path("/Users/rc/Projects/Transformation_Portal/input_images/750Picacho_16-bit_TIFFs")
    output_dir = Path("/Users/rc/Projects/Transformation_Portal/output_bugfix_validation_final/v2")

    images_to_check = [
        {
            "name": "Aerial",
            "input": input_dir / "750Picacho_Aerial_master16.tif",
            "output": output_dir / "750Picacho_Aerial_master16_tif_abd152a0_materials_v3_enhanced.tif",
            "check_sky": True,
            "check_water": False,
        },
        {
            "name": "Pool",
            "input": input_dir / "750Picacho_Pool_master16.tif",
            "output": output_dir / "750Picacho_Pool_master16_tif_c91cb832_materials_v3_enhanced.tif",
            "check_sky": False,
            "check_water": True,
        },
    ]

    results = []

    for img_config in images_to_check:
        print(f"\n{'='*80}")
        print(f"Analyzing: {img_config['name']}")
        print(f"{'='*80}")

        # Load images
        input_img = load_16bit_tiff(img_config["input"])
        output_img = load_16bit_tiff(img_config["output"])

        print(f"Input shape: {input_img.shape}, dtype: {input_img.dtype}")
        print(f"Output shape: {output_img.shape}, dtype: {output_img.dtype}")
        print(f"Input range: [{input_img.min():.4f}, {input_img.max():.4f}]")
        print(f"Output range: [{output_img.min():.4f}, {output_img.max():.4f}]")

        # Resize output to match input if needed (for mask alignment)
        if input_img.shape != output_img.shape:
            print(f"⚠️  Shape mismatch: Input={input_img.shape}, Output={output_img.shape}")
            # Simple approach: crop to minimum dimensions
            min_h = min(input_img.shape[0], output_img.shape[0])
            min_w = min(input_img.shape[1], output_img.shape[1])
            print(f"   Cropping both to ({min_h}, {min_w}, 3) for comparison")
            input_img = input_img[:min_h, :min_w, :]
            output_img = output_img[:min_h, :min_w, :]

        img_result = {
            "image": img_config["name"],
            "input_path": str(img_config["input"]),
            "output_path": str(img_config["output"]),
            "regions": [],
        }

        # Check sky if applicable
        if img_config["check_sky"]:
            print("\n--- SKY ANALYSIS ---")
            # Method 1: Top 30% of image
            sky_mask_top = create_sky_mask_top_region(input_img.shape, top_fraction=0.3)
            input_sky_top = analyze_region(input_img, sky_mask_top, "sky_top30pct")
            output_sky_top = analyze_region(output_img, sky_mask_top, "sky_top30pct")
            delta_sky_top = compare_regions(input_sky_top, output_sky_top)

            print(f"\nSky (Top 30% region):")
            print(
                f"  Input:  Mean RGB={input_sky_top['mean_rgb']}, Brightness={input_sky_top['mean_brightness']:.4f}, Saturation={input_sky_top['mean_saturation']:.4f}"
            )
            print(
                f"  Output: Mean RGB={output_sky_top['mean_rgb']}, Brightness={output_sky_top['mean_brightness']:.4f}, Saturation={output_sky_top['mean_saturation']:.4f}"
            )
            print(
                f"  Delta:  RGB={delta_sky_top['delta_mean_rgb']}, Brightness={delta_sky_top['delta_brightness']:+.4f} ({delta_sky_top['brightness_change_pct']:+.2f}%), Saturation={delta_sky_top['delta_saturation']:+.4f} ({delta_sky_top['saturation_change_pct']:+.2f}%)"
            )

            img_result["regions"].append(
                {
                    "type": "sky_top30pct",
                    "input": input_sky_top,
                    "output": output_sky_top,
                    "delta": delta_sky_top,
                }
            )

            # Method 2: Color-based detection
            sky_mask_color = detect_sky_pixels(input_img)
            sky_coverage_pct = 100 * sky_mask_color.sum() / (sky_mask_color.shape[0] * sky_mask_color.shape[1])
            print(f"\nSky (Color-based detection): {sky_coverage_pct:.1f}% of image")

            if sky_coverage_pct > 1.0:
                input_sky_color = analyze_region(input_img, sky_mask_color, "sky_color_detected")
                output_sky_color = analyze_region(output_img, sky_mask_color, "sky_color_detected")
                delta_sky_color = compare_regions(input_sky_color, output_sky_color)

                print(
                    f"  Input:  Mean RGB={input_sky_color['mean_rgb']}, Brightness={input_sky_color['mean_brightness']:.4f}, Saturation={input_sky_color['mean_saturation']:.4f}"
                )
                print(
                    f"  Output: Mean RGB={output_sky_color['mean_rgb']}, Brightness={output_sky_color['mean_brightness']:.4f}, Saturation={output_sky_color['mean_saturation']:.4f}"
                )
                print(
                    f"  Delta:  RGB={delta_sky_color['delta_mean_rgb']}, Brightness={delta_sky_color['delta_brightness']:+.4f} ({delta_sky_color['brightness_change_pct']:+.2f}%), Saturation={delta_sky_color['delta_saturation']:+.4f} ({delta_sky_color['saturation_change_pct']:+.2f}%)"
                )

                img_result["regions"].append(
                    {
                        "type": "sky_color_detected",
                        "coverage_pct": sky_coverage_pct,
                        "input": input_sky_color,
                        "output": output_sky_color,
                        "delta": delta_sky_color,
                    }
                )

        # Check water if applicable
        if img_config["check_water"]:
            print("\n--- WATER ANALYSIS ---")
            # Method 1: Bottom 30% of image
            water_mask_bottom = create_water_mask_bottom_region(input_img.shape, bottom_fraction=0.3)
            input_water_bottom = analyze_region(input_img, water_mask_bottom, "water_bottom30pct")
            output_water_bottom = analyze_region(output_img, water_mask_bottom, "water_bottom30pct")
            delta_water_bottom = compare_regions(input_water_bottom, output_water_bottom)

            print(f"\nWater (Bottom 30% region):")
            print(
                f"  Input:  Mean RGB={input_water_bottom['mean_rgb']}, Brightness={input_water_bottom['mean_brightness']:.4f}, Saturation={input_water_bottom['mean_saturation']:.4f}"
            )
            print(
                f"  Output: Mean RGB={output_water_bottom['mean_rgb']}, Brightness={output_water_bottom['mean_brightness']:.4f}, Saturation={output_water_bottom['mean_saturation']:.4f}"
            )
            print(
                f"  Delta:  RGB={delta_water_bottom['delta_mean_rgb']}, Brightness={delta_water_bottom['delta_brightness']:+.4f} ({delta_water_bottom['brightness_change_pct']:+.2f}%), Saturation={delta_water_bottom['delta_saturation']:+.4f} ({delta_water_bottom['saturation_change_pct']:+.2f}%)"
            )

            img_result["regions"].append(
                {
                    "type": "water_bottom30pct",
                    "input": input_water_bottom,
                    "output": output_water_bottom,
                    "delta": delta_water_bottom,
                }
            )

            # Method 2: Color-based detection
            water_mask_color = detect_water_pixels(input_img)
            water_coverage_pct = 100 * water_mask_color.sum() / (water_mask_color.shape[0] * water_mask_color.shape[1])
            print(f"\nWater (Color-based detection): {water_coverage_pct:.1f}% of image")

            if water_coverage_pct > 1.0:
                input_water_color = analyze_region(input_img, water_mask_color, "water_color_detected")
                output_water_color = analyze_region(output_img, water_mask_color, "water_color_detected")
                delta_water_color = compare_regions(input_water_color, output_water_color)

                print(
                    f"  Input:  Mean RGB={input_water_color['mean_rgb']}, Brightness={input_water_color['mean_brightness']:.4f}, Saturation={input_water_color['mean_saturation']:.4f}"
                )
                print(
                    f"  Output: Mean RGB={output_water_color['mean_rgb']}, Brightness={output_water_color['mean_brightness']:.4f}, Saturation={output_water_color['mean_saturation']:.4f}"
                )
                print(
                    f"  Delta:  RGB={delta_water_color['delta_mean_rgb']}, Brightness={delta_water_color['delta_brightness']:+.4f} ({delta_water_color['brightness_change_pct']:+.2f}%), Saturation={delta_water_color['delta_saturation']:+.4f} ({delta_water_color['saturation_change_pct']:+.2f}%)"
                )

                img_result["regions"].append(
                    {
                        "type": "water_color_detected",
                        "coverage_pct": water_coverage_pct,
                        "input": input_water_color,
                        "output": output_water_color,
                        "delta": delta_water_color,
                    }
                )

        # Whole image comparison
        print("\n--- WHOLE IMAGE ANALYSIS ---")
        input_whole = analyze_region(input_img, None, "whole_image")
        output_whole = analyze_region(output_img, None, "whole_image")
        delta_whole = compare_regions(input_whole, output_whole)

        print(f"\nWhole Image:")
        print(
            f"  Input:  Mean RGB={input_whole['mean_rgb']}, Brightness={input_whole['mean_brightness']:.4f}, Saturation={input_whole['mean_saturation']:.4f}"
        )
        print(
            f"  Output: Mean RGB={output_whole['mean_rgb']}, Brightness={output_whole['mean_brightness']:.4f}, Saturation={output_whole['mean_saturation']:.4f}"
        )
        print(
            f"  Delta:  RGB={delta_whole['delta_mean_rgb']}, Brightness={delta_whole['delta_brightness']:+.4f} ({delta_whole['brightness_change_pct']:+.2f}%), Saturation={delta_whole['delta_saturation']:+.4f} ({delta_whole['saturation_change_pct']:+.2f}%)"
        )

        img_result["regions"].append(
            {
                "type": "whole_image",
                "input": input_whole,
                "output": output_whole,
                "delta": delta_whole,
            }
        )

        results.append(img_result)

    # Save results
    output_json = Path("sky_water_degradation_analysis.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_json}")
    print(f"{'='*80}")

    # Summary
    print("\n\nSUMMARY OF FINDINGS:")
    print("=" * 80)
    for img_result in results:
        print(f"\n{img_result['image']}:")
        for region in img_result["regions"]:
            delta = region["delta"]
            region_type = region["type"]
            print(f"  {region_type}:")
            print(f"    Brightness change: {delta['delta_brightness']:+.4f} ({delta['brightness_change_pct']:+.2f}%)")
            print(f"    Saturation change: {delta['delta_saturation']:+.4f} ({delta['saturation_change_pct']:+.2f}%)")

            # Flag significant degradation
            if abs(delta["brightness_change_pct"]) > 5.0:
                print(f"    ⚠️  SIGNIFICANT BRIGHTNESS CHANGE DETECTED")
            if abs(delta["saturation_change_pct"]) > 10.0:
                print(f"    ⚠️  SIGNIFICANT SATURATION CHANGE DETECTED")


if __name__ == "__main__":
    main()
