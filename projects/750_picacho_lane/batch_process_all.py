#!/usr/bin/env python3
"""
750 Picacho Lane - Batch Process All Renderings
Processes all 7 views with scene-specific Material Response settings
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

# Import OpenEXR for reading
try:
    import OpenEXR
    import Imath
except ImportError:
    print("ERROR: OpenEXR not installed. Run: pip install OpenEXR")
    sys.exit(1)


# Scene-specific configurations
SCENE_CONFIGS = {
    "Pool": {
        "description": "Pool & Aquatic Features",
        "water_enhance": True,
        "water_saturation": 1.25,
        "contrast": 1.08,
        "saturation": 1.05,
        "temperature": 5,
        "materials": ["water", "stone", "concrete"]
    },
    "GreatRoom": {
        "description": "Great Room - Interior Living",
        "water_enhance": False,
        "wood_enhance": True,
        "fabric_enhance": True,
        "contrast": 1.10,
        "saturation": 1.03,
        "temperature": 3,
        "warmth": 8,
        "materials": ["wood", "fabric", "glass", "stone"]
    },
    "Kitchen": {
        "description": "Kitchen - Culinary Space",
        "water_enhance": False,
        "metal_enhance": True,
        "stone_enhance": True,
        "contrast": 1.12,
        "saturation": 1.02,
        "temperature": 2,
        "clarity": 1.15,
        "materials": ["metal", "stone", "glass", "wood"]
    },
    "PrimaryBedroom": {
        "description": "Primary Bedroom Suite",
        "water_enhance": False,
        "fabric_enhance": True,
        "wood_enhance": True,
        "contrast": 1.05,
        "saturation": 1.02,
        "temperature": 6,
        "warmth": 10,
        "softness": 0.95,
        "materials": ["fabric", "wood", "glass"]
    },
    "PrimaryBathroom": {
        "description": "Primary Bathroom - Spa",
        "water_enhance": False,
        "stone_enhance": True,
        "glass_enhance": True,
        "contrast": 1.08,
        "saturation": 1.04,
        "temperature": 4,
        "materials": ["stone", "glass", "metal", "water"]
    },
    "Aerial": {
        "description": "Aerial View - Estate Overview",
        "water_enhance": True,
        "landscape_enhance": True,
        "contrast": 1.15,
        "saturation": 1.08,
        "temperature": 7,
        "clarity": 1.20,
        "atmospheric_depth": True,
        "materials": ["water", "stone", "vegetation", "roof"]
    },
    "Aerial-2": {
        "description": "Aerial View 2 - Neighborhood Context",
        "water_enhance": True,
        "landscape_enhance": True,
        "contrast": 1.15,
        "saturation": 1.08,
        "temperature": 7,
        "clarity": 1.20,
        "atmospheric_depth": True,
        "materials": ["water", "stone", "vegetation", "roof"]
    }
}


def linear_to_srgb(linear):
    """Convert linear RGB to sRGB gamma"""
    linear = np.clip(linear, 0, 1)
    srgb = np.where(linear <= 0.0031308,
                    12.92 * linear,
                    1.055 * np.power(linear, 1/2.4) - 0.055)
    return srgb


def load_exr(filepath):
    """Load EXR file and convert to numpy array"""
    exr_file = OpenEXR.InputFile(str(filepath))
    header = exr_file.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = ['R', 'G', 'B']
    channel_data = [exr_file.channel(c, FLOAT) for c in channels]

    img_array = np.zeros((height, width, 3), dtype=np.float32)
    for i, data in enumerate(channel_data):
        channel = np.frombuffer(data, dtype=np.float32).reshape(height, width)
        img_array[:, :, i] = channel

    return img_array


def detect_water_mask(img, threshold=0.4):
    """Detect water regions (blue-dominant areas)"""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Water tends to have: blue > green and blue > red
    blue_dominant = (b > r * 1.1) & (b > g * 1.05)

    # Medium-to-high saturation
    intensity = (r + g + b) / 3.0
    chroma = b - np.minimum(r, g)
    saturation = np.where(intensity > 0.01, chroma / (intensity + 1e-6), 0)

    water_mask = blue_dominant & (saturation > threshold)
    return water_mask.astype(np.float32)


def enhance_water(img, mask, saturation_boost=1.25):
    """Enhance water regions"""
    enhanced = img.copy()

    # Boost blue saturation
    enhanced[:, :, 2] = np.clip(
        enhanced[:, :, 2] * (1.0 + (saturation_boost - 1.0) * mask),
        0, 1
    )

    # Slightly cool the color (reduce red)
    enhanced[:, :, 0] = np.clip(
        enhanced[:, :, 0] * (1.0 - 0.08 * mask),
        0, 1
    )

    return enhanced


def apply_color_grading(img, config):
    """Apply color grading based on scene config"""
    graded = img.copy()

    # Contrast
    contrast = config.get('contrast', 1.0)
    if contrast != 1.0:
        graded = np.clip((graded - 0.5) * contrast + 0.5, 0, 1)

    # Saturation
    saturation = config.get('saturation', 1.0)
    if saturation != 1.0:
        luminance = 0.2126 * graded[:, :, 0] + 0.7152 * graded[:, :, 1] + 0.0722 * graded[:, :, 2]
        luminance = luminance[:, :, np.newaxis]
        graded = np.clip(luminance + (graded - luminance) * saturation, 0, 1)

    # Temperature (warm/cool)
    temperature = config.get('temperature', 0)
    if temperature != 0:
        temp_factor = temperature / 100.0
        graded[:, :, 0] = np.clip(graded[:, :, 0] * (1.0 + temp_factor * 0.1), 0, 1)
        graded[:, :, 2] = np.clip(graded[:, :, 2] * (1.0 - temp_factor * 0.05), 0, 1)

    # Warmth (additional for interiors)
    warmth = config.get('warmth', 0)
    if warmth != 0:
        warmth_factor = warmth / 100.0
        graded[:, :, 0] = np.clip(graded[:, :, 0] * (1.0 + warmth_factor * 0.08), 0, 1)
        graded[:, :, 1] = np.clip(graded[:, :, 1] * (1.0 + warmth_factor * 0.04), 0, 1)

    # Clarity (mid-tone contrast) - FIXED
    clarity = config.get('clarity', 1.0)
    if clarity != 1.0:
        # Simple clarity: boost mid-tones
        # Calculate per-channel mid-tone mask
        mid_mask = 1.0 - np.abs(graded - 0.5) * 2.0
        graded = np.clip(graded * (1.0 + (clarity - 1.0) * mid_mask), 0, 1)

    return graded


def process_scene(input_path, output_dir, scene_name, config):
    """Process a single scene with its specific configuration"""

    print(f"\n{'='*70}")
    print(f"🏡 PROCESSING: {scene_name}")
    print(f"{'='*70}")

    start_time = time.time()

    # Load EXR
    print(f"📂 Loading: {input_path.name} ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")
    img_linear = load_exr(input_path)
    height, width = img_linear.shape[:2]
    print(f"  • Resolution: {width} x {height}")
    print(f"  • Range: [{img_linear.min():.3f}, {img_linear.max():.3f}]")

    # Convert to sRGB
    print("🎨 Converting linear → sRGB")
    img_srgb = linear_to_srgb(img_linear)

    # Material Response enhancements
    print(f"💎 Applying Material Response - {config['description']}")
    enhanced = img_srgb.copy()

    # Water enhancement if applicable
    if config.get('water_enhance', False):
        water_mask = detect_water_mask(img_srgb)
        water_percent = (water_mask > 0.5).sum() / water_mask.size * 100
        print(f"  💧 Water regions: {water_percent:.1f}%")
        enhanced = enhance_water(enhanced, water_mask, config.get('water_saturation', 1.25))

    # Color grading
    print("🎨 Applying color grading")
    print(f"  • Contrast: {config.get('contrast', 1.0)}")
    print(f"  • Saturation: {config.get('saturation', 1.0)}")
    print(f"  • Temperature: +{config.get('temperature', 0)}")
    if config.get('warmth'):
        print(f"  • Warmth: +{config.get('warmth', 0)}")
    if config.get('clarity'):
        print(f"  • Clarity: {config.get('clarity', 1.0)}")

    graded = apply_color_grading(enhanced, config)

    # Save outputs
    print("💾 Saving deliverables")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = scene_name.replace('-', '_')

    # Master TIFF (16-bit)
    img_uint16 = (np.clip(graded, 0, 1) * 65535).astype(np.uint16)
    master_path = output_dir / f"750Picacho_{base_name}_Master.tif"
    Image.fromarray(img_uint16, mode='RGB').save(
        master_path,
        format='TIFF',
        compression='lzw'
    )
    master_size = master_path.stat().st_size / 1024 / 1024
    print(f"  ✅ Master TIFF: {master_path.name} ({master_size:.1f} MB)")

    # Web JPEG
    img_uint8 = (np.clip(graded, 0, 1) * 255).astype(np.uint8)
    web_path = output_dir / f"750Picacho_{base_name}_Web.jpg"
    Image.fromarray(img_uint8, mode='RGB').save(
        web_path,
        format='JPEG',
        quality=95,
        optimize=True
    )
    web_size = web_path.stat().st_size / 1024 / 1024
    print(f"  ✅ Web JPEG: {web_path.name} ({web_size:.1f} MB)")

    # Thumbnail
    thumb_img = Image.fromarray(img_uint8)
    thumb_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    thumb_path = output_dir / f"750Picacho_{base_name}_Thumbnail.jpg"
    thumb_img.save(thumb_path, format='JPEG', quality=90, optimize=True)
    thumb_size = thumb_path.stat().st_size / 1024
    print(f"  ✅ Thumbnail: {thumb_path.name} ({thumb_size:.0f} KB)")

    elapsed = time.time() - start_time
    print(f"✅ Complete in {elapsed:.1f}s")

    return {
        'scene': scene_name,
        'config': config,
        'resolution': (width, height),
        'master_size': master_size,
        'web_size': web_size,
        'thumb_size': thumb_size,
        'time': elapsed
    }


def main():
    """Batch process all renderings"""

    print("="*70)
    print("🏛️  750 PICACHO LANE - BATCH PROCESSING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Paths
    source_dir = Path("/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/16-Bit_EXRs")
    output_dir = Path("/Users/rc/Transformation_Portal/projects/750_picacho_lane/output")

    # Find all EXR files
    exr_files = list(source_dir.glob("*.exr"))
    print(f"Found {len(exr_files)} EXR files")

    # Match files to scenes
    scenes_to_process = []
    for exr_path in exr_files:
        # Extract scene name from filename
        filename = exr_path.stem  # e.g., "750Picacho_Pool" or "2-750Picacho_Aerial-2"

        for scene_key, config in SCENE_CONFIGS.items():
            if scene_key.replace('-', '_') in filename or scene_key in filename:
                scenes_to_process.append((exr_path, scene_key, config))
                break

    print(f"Matched {len(scenes_to_process)} scenes")
    print()

    # Process each scene
    results = []
    total_start = time.time()

    for i, (exr_path, scene_name, config) in enumerate(scenes_to_process, 1):
        print(f"\n[{i}/{len(scenes_to_process)}]")
        result = process_scene(exr_path, output_dir, scene_name, config)
        results.append(result)

    # Summary
    total_time = time.time() - total_start

    print("\n" + "="*70)
    print("✅ BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Processed: {len(results)} scenes")
    print(f"Average: {total_time/len(results):.1f}s per scene")
    print()
    print("📊 Scene Summary:")
    print("-" * 70)

    total_master_size = 0
    total_web_size = 0

    for result in results:
        total_master_size += result['master_size']
        total_web_size += result['web_size']
        print(f"  {result['scene']:20} | {result['resolution'][0]}x{result['resolution'][1]:4} | "
              f"Master: {result['master_size']:5.1f}MB | Web: {result['web_size']:4.1f}MB | "
              f"{result['time']:4.1f}s")

    print("-" * 70)
    print(f"  {'TOTAL':20} |          | Master: {total_master_size:5.1f}MB | "
          f"Web: {total_web_size:4.1f}MB |")
    print()
    print(f"📦 All deliverables saved to:")
    print(f"   {output_dir}")
    print()
    print("🎯 Ready for client delivery!")


if __name__ == "__main__":
    main()
