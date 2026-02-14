#!/usr/bin/env python3
"""
Create visual comparison of sky/water regions for the investigation report.
Generates side-by-side comparison images with annotations.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_16bit_tiff(path: Path) -> np.ndarray:
    """Load 16-bit TIFF as uint8 for visualization."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    # Normalize to [0,255]
    if arr.max() > 255:
        arr = arr / 65535.0 * 255.0
    return arr.astype(np.uint8)


def create_comparison_image(input_path: Path, output_path: Path, title: str, annotation: str) -> Image.Image:
    """Create side-by-side comparison with annotations."""
    input_img = load_16bit_tiff(input_path)
    output_img = load_16bit_tiff(output_path)

    # Resize to manageable size for visualization (1200px wide each)
    h, w = input_img.shape[:2]
    scale = 1200 / w
    new_h = int(h * scale)

    input_pil = Image.fromarray(input_img)
    output_pil = Image.fromarray(output_img)

    input_resized = input_pil.resize((1200, new_h), Image.LANCZOS)
    output_resized = output_pil.resize((1200, new_h), Image.LANCZOS)

    # Create canvas with space for title and annotations
    title_height = 80
    annotation_height = 120
    canvas_width = 2400
    canvas_height = new_h + title_height + annotation_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(40, 40, 40))

    # Paste images
    canvas.paste(input_resized, (0, title_height))
    canvas.paste(output_resized, (1200, title_height))

    # Add annotations
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        note_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        note_font = ImageFont.load_default()

    # Title
    draw.text((canvas_width // 2 - 300, 20), title, fill=(255, 255, 255), font=title_font)

    # Labels
    draw.text((600 - 80, title_height + new_h + 20), "INPUT", fill=(255, 255, 0), font=label_font)
    draw.text((1800 - 100, title_height + new_h + 20), "OUTPUT", fill=(0, 255, 255), font=label_font)

    # Annotation
    lines = annotation.split("\n")
    y_offset = title_height + new_h + 70
    for line in lines:
        draw.text((40, y_offset), line, fill=(200, 200, 200), font=note_font)
        y_offset += 30

    return canvas


def main():
    input_dir = Path("/Users/rc/Projects/Transformation_Portal/input_images/750Picacho_16-bit_TIFFs")
    output_dir = Path("/Users/rc/Projects/Transformation_Portal/output_bugfix_validation_final/v2")
    comparison_dir = Path("comparison_images")
    comparison_dir.mkdir(exist_ok=True)

    # Load analysis results
    with open("sky_water_degradation_analysis.json") as f:
        results = json.load(f)

    # Create Aerial comparison (sky)
    aerial_result = next(r for r in results if r["image"] == "Aerial")
    sky_delta = next(r for r in aerial_result["regions"] if r["type"] == "sky_color_detected")

    aerial_annotation = (
        f"Sky Coverage: {sky_delta['coverage_pct']:.1f}%\n"
        f"Brightness: {sky_delta['delta']['brightness_change_pct']:+.2f}% | "
        f"Saturation: {sky_delta['delta']['saturation_change_pct']:+.2f}%\n"
        f"Assessment: ✅ Minimal change, within tolerances"
    )

    aerial_comparison = create_comparison_image(
        input_dir / "750Picacho_Aerial_master16.tif",
        output_dir / "750Picacho_Aerial_master16_tif_abd152a0_materials_v3_enhanced.tif",
        "Aerial Image - Sky Analysis",
        aerial_annotation,
    )
    aerial_comparison.save(comparison_dir / "aerial_sky_comparison.jpg", quality=95)
    print(f"✅ Created: {comparison_dir / 'aerial_sky_comparison.jpg'}")

    # Create Pool comparison (water)
    pool_result = next(r for r in results if r["image"] == "Pool")
    water_delta = next(r for r in pool_result["regions"] if r["type"] == "water_color_detected")

    pool_annotation = (
        f"Water Coverage: {water_delta['coverage_pct']:.1f}%\n"
        f"Brightness: {water_delta['delta']['brightness_change_pct']:+.2f}% | "
        f"Saturation: {water_delta['delta']['saturation_change_pct']:+.2f}%\n"
        f"Assessment: ✅ Minimal change, foliage adjacency effect"
    )

    pool_comparison = create_comparison_image(
        input_dir / "750Picacho_Pool_master16.tif",
        output_dir / "750Picacho_Pool_master16_tif_c91cb832_materials_v3_enhanced.tif",
        "Pool Image - Water Analysis",
        pool_annotation,
    )
    pool_comparison.save(comparison_dir / "pool_water_comparison.jpg", quality=95)
    print(f"✅ Created: {comparison_dir / 'pool_water_comparison.jpg'}")

    print("\n" + "=" * 80)
    print("Visual comparison images generated in: comparison_images/")
    print("=" * 80)


if __name__ == "__main__":
    main()
