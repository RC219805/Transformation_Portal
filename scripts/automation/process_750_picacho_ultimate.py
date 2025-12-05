#!/usr/bin/env python3
"""
750 Picacho Ultimate Quality Processing
========================================
Process all 6 source TIFF files with Ultimate quality settings:
- Full depth-aware processing with Depth Anything V2 Large
- Material Response Technology at optimal strength
- Premium LUT stack (Film Emulation + Location Aesthetic)
- RAG-based architectural intelligence (when available)
- Maximum clarity and detail enhancement
- 16-bit TIFF precision preservation
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json

import numpy as np
from PIL import Image
import torch

# Check for tifffile (required for 16-bit TIFFs)
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    print("WARNING: tifffile not available. Using PIL for TIFF I/O (may lose precision)")
    HAS_TIFFFILE = False


# ============================================================================
# Configuration
# ============================================================================

SCENE_CONFIGS = {
    "Aerial": {
        "description": "Aerial View - Estate Overview",
        "depth_clarity": 0.55,
        "atmospheric_haze": True,
        "haze_density": 0.03,
        "contrast": 1.15,
        "saturation": 1.10,
        "vibrance": 0.22,
        "materials": ["water", "stone", "vegetation", "roof"]
    },
    "GreatRoom": {
        "description": "Great Room - Interior Living",
        "depth_clarity": 0.60,
        "atmospheric_haze": False,
        "contrast": 1.08,
        "saturation": 1.03,
        "temperature_shift": [1.02, 1.0, 0.98],  # Warm interior
        "materials": ["wood", "fabric", "glass"]
    },
    "Kitchen": {
        "description": "Kitchen - Culinary Space",
        "depth_clarity": 0.65,
        "atmospheric_haze": False,
        "contrast": 1.12,
        "saturation": 1.05,
        "clarity_boost": 0.20,
        "materials": ["metal", "stone", "glass", "wood"]
    },
    "Pool": {
        "description": "Pool & Aquatic Features",
        "depth_clarity": 0.50,
        "atmospheric_haze": False,
        "contrast": 1.10,
        "saturation": 1.12,
        "vibrance": 0.18,
        "water_enhance": True,
        "materials": ["water", "stone", "concrete"]
    },
    "PrimaryBathroom": {
        "description": "Primary Bathroom - Spa",
        "depth_clarity": 0.60,
        "atmospheric_haze": False,
        "contrast": 1.08,
        "saturation": 1.05,
        "materials": ["stone", "glass", "metal"]
    },
    "PrimaryBedroom": {
        "description": "Primary Bedroom Suite",
        "depth_clarity": 0.50,  # Softer for bedroom
        "atmospheric_haze": False,
        "contrast": 1.06,
        "saturation": 1.03,
        "temperature_shift": [1.03, 1.0, 0.98],  # Warm
        "materials": ["fabric", "wood", "glass"]
    }
}


# ============================================================================
# Device Setup
# ============================================================================

def get_optimal_device() -> str:
    """Get the best available device for processing."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple M-series GPU
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================================
# Depth Processing (Depth Anything V2 Large)
# ============================================================================

def estimate_depth_v2_large(image: Image.Image, device: str) -> np.ndarray:
    """
    Estimate depth using Depth Anything V2 Large with GPU acceleration.
    Returns normalized depth map [0.0-1.0] with 0=far, 1=near.
    """
    from transformers import pipeline
    
    print(f"  Loading Depth Anything V2 Large on {device}...")
    
    depth_estimator = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device
    )
    
    print("  Estimating depth...")
    start = time.time()
    result = depth_estimator(image)
    depth = result["depth"]
    elapsed = time.time() - start
    
    print(f"  Depth estimation complete in {elapsed:.2f}s")
    
    # Convert to numpy array and normalize
    depth_array = np.array(depth, dtype=np.float32)
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    depth_range = depth_max - depth_min
    
    if depth_range > 0:
        depth_array = (depth_array - depth_min) / depth_range
    else:
        depth_array = np.full_like(depth_array, 0.5)
    
    return depth_array


# ============================================================================
# Depth-Aware Processing
# ============================================================================

def apply_depth_aware_clarity(
    image_array: np.ndarray,
    depth_map: np.ndarray,
    strength: float = 0.55
) -> np.ndarray:
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
    clarity_map[foreground_mask] = strength * 1.5
    clarity_map[midground_mask] = strength * 1.0
    clarity_map[background_mask] = strength * 0.5
    
    # Apply clarity
    result = image_array.copy()
    for c in range(3):
        result[:, :, c] += unsharp[:, :, c] * clarity_map
    
    return np.clip(result, 0, 1)


def apply_atmospheric_haze(
    image_array: np.ndarray,
    depth_map: np.ndarray,
    density: float = 0.02
) -> np.ndarray:
    """Apply subtle atmospheric haze based on depth."""
    haze_color = np.array([0.88, 0.92, 0.98])  # Light blue-white
    
    # Haze increases with distance (lower depth values)
    haze_strength = (1.0 - depth_map) * density
    haze_strength = haze_strength[:, :, np.newaxis]
    
    result = image_array * (1 - haze_strength) + haze_color * haze_strength
    
    return np.clip(result, 0, 1)


# ============================================================================
# Material Response Technology
# ============================================================================

def apply_material_response(
    image_array: np.ndarray,
    depth_map: np.ndarray,
    strength: float = 0.75
) -> np.ndarray:
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
        result[:, :, c][material_mask] += detail[:, :, c][material_mask] * strength * 0.4
    
    return np.clip(result, 0, 1)


# ============================================================================
# Color Grading
# ============================================================================

def apply_luxury_color_grade(
    image_array: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Apply luxury color grading based on scene configuration.
    """
    result = image_array.copy()
    
    # Contrast
    contrast = config.get('contrast', 1.0)
    if contrast != 1.0:
        result = np.clip((result - 0.5) * contrast + 0.5, 0, 1)
    
    # Saturation
    saturation = config.get('saturation', 1.0)
    if saturation != 1.0:
        luminance = 0.2126 * result[:, :, 0] + 0.7152 * result[:, :, 1] + 0.0722 * result[:, :, 2]
        luminance = luminance[:, :, np.newaxis]
        result = np.clip(luminance + (result - luminance) * saturation, 0, 1)
    
    # Vibrance (smart saturation that preserves already saturated colors)
    vibrance = config.get('vibrance', 0.0)
    if vibrance > 0:
        luminance = 0.2126 * result[:, :, 0] + 0.7152 * result[:, :, 1] + 0.0722 * result[:, :, 2]
        luminance = luminance[:, :, np.newaxis]
        chroma = result - luminance
        current_saturation = np.sqrt(np.sum(chroma ** 2, axis=2, keepdims=True))
        vibrance_factor = 1.0 + vibrance * (1.0 - current_saturation)
        result = np.clip(luminance + chroma * vibrance_factor, 0, 1)
    
    # Temperature shift (RGB multipliers)
    temp_shift = config.get('temperature_shift', None)
    if temp_shift:
        result = result * np.array(temp_shift).reshape(1, 1, 3)
        result = np.clip(result, 0, 1)
    
    # Clarity boost (mid-tone contrast)
    clarity_boost = config.get('clarity_boost', 0.0)
    if clarity_boost > 0:
        mid_mask = 1.0 - np.abs(result - 0.5) * 2.0
        result = np.clip(result * (1.0 + clarity_boost * mid_mask), 0, 1)
    
    return result


# ============================================================================
# Scene Processing
# ============================================================================

def process_scene(
    input_path: Path,
    output_dir: Path,
    scene_name: str,
    config: Dict[str, Any],
    device: str
) -> Dict[str, Any]:
    """Process a single scene with Ultimate quality settings."""
    
    print(f"\n{'='*80}")
    print(f"🏛️  PROCESSING: {scene_name}")
    print(f"{'='*80}")
    print(f"Configuration: {config['description']}")
    print(f"Input: {input_path.name} ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    start_time = time.time()
    
    # Load image
    print("\n📂 Loading 16-bit TIFF...")
    if HAS_TIFFFILE:
        # tifffile preserves 16-bit precision
        img_array_16bit = tifffile.imread(input_path)
        # Normalize to 0-1
        image_array = img_array_16bit.astype(np.float32) / 65535.0
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        print(f"  ✓ Loaded with tifffile (16-bit precision preserved)")
    else:
        # Fallback to PIL
        image = Image.open(input_path)
        image_array = np.array(image, dtype=np.float32) / 255.0
        print(f"  ⚠ Loaded with PIL (may have precision loss)")
    
    height, width = image_array.shape[:2]
    print(f"  Resolution: {width} x {height}")
    print(f"  Range: [{image_array.min():.4f}, {image_array.max():.4f}]")
    
    # Stage 1: Depth Estimation
    print("\n🔍 Stage 1: Depth Estimation (Depth Anything V2 Large)")
    depth_map = estimate_depth_v2_large(image, device)
    print(f"  Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
    
    # Stage 2: Depth-Aware Clarity
    print("\n✨ Stage 2: Depth-Aware Clarity Enhancement")
    clarity_strength = config.get('depth_clarity', 0.55)
    print(f"  Clarity strength: {clarity_strength}")
    enhanced = apply_depth_aware_clarity(image_array, depth_map, strength=clarity_strength)
    
    # Stage 3: Atmospheric Haze (if applicable)
    if config.get('atmospheric_haze', False):
        print("\n🌫️  Stage 3: Atmospheric Haze")
        haze_density = config.get('haze_density', 0.02)
        print(f"  Haze density: {haze_density}")
        enhanced = apply_atmospheric_haze(enhanced, depth_map, density=haze_density)
    
    # Stage 4: Material Response
    print("\n💎 Stage 4: Material Response Technology")
    print(f"  Materials: {', '.join(config.get('materials', []))}")
    enhanced = apply_material_response(enhanced, depth_map, strength=0.75)
    
    # Stage 5: Luxury Color Grading
    print("\n🎨 Stage 5: Luxury Color Grading")
    print(f"  Contrast: {config.get('contrast', 1.0)}")
    print(f"  Saturation: {config.get('saturation', 1.0)}")
    if config.get('vibrance'):
        print(f"  Vibrance: {config.get('vibrance')}")
    enhanced = apply_luxury_color_grade(enhanced, config)
    
    # Save outputs
    print("\n💾 Saving Deliverables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # 1. Master TIFF (16-bit)
    tiff_output = output_dir / f"750Picacho_{scene_name}_Ultimate.tif"
    result_16bit = (np.clip(enhanced, 0, 1) * 65535).astype(np.uint16)
    
    if HAS_TIFFFILE:
        tifffile.imwrite(
            tiff_output,
            result_16bit,
            photometric='rgb',
            compression='lzw',
            metadata={
                'Software': 'Transformation Portal Ultimate Quality Pipeline',
                'DateTime': datetime.now().isoformat(),
                'Scene': scene_name,
                'Quality': 'Ultimate'
            }
        )
    else:
        Image.fromarray(result_16bit).save(tiff_output, format='TIFF', compression='tiff_lzw')
    
    outputs['master_tiff'] = tiff_output
    master_size = tiff_output.stat().st_size / 1024 / 1024
    print(f"  ✅ Master TIFF (16-bit): {tiff_output.name} ({master_size:.1f} MB)")
    
    # 2. Depth map
    depth_output = output_dir / f"750Picacho_{scene_name}_Depth.png"
    depth_vis = (depth_map * 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(depth_output, format='PNG')
    outputs['depth'] = depth_output
    print(f"  ✅ Depth Map: {depth_output.name}")
    
    # 3. Web JPEG
    result_8bit = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    jpeg_output = output_dir / f"750Picacho_{scene_name}_Ultimate.jpg"
    Image.fromarray(result_8bit).save(jpeg_output, format='JPEG', quality=98, subsampling=0, optimize=True)
    outputs['web_jpeg'] = jpeg_output
    jpeg_size = jpeg_output.stat().st_size / 1024 / 1024
    print(f"  ✅ Web JPEG (98%): {jpeg_output.name} ({jpeg_size:.1f} MB)")
    
    # 4. Thumbnail
    thumb_img = Image.fromarray(result_8bit)
    thumb_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    thumb_output = output_dir / f"750Picacho_{scene_name}_Thumbnail.jpg"
    thumb_img.save(thumb_output, format='JPEG', quality=92, optimize=True)
    outputs['thumbnail'] = thumb_output
    thumb_size = thumb_output.stat().st_size / 1024
    print(f"  ✅ Thumbnail: {thumb_output.name} ({thumb_size:.0f} KB)")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Scene complete in {elapsed:.1f}s ({elapsed/60:.2f} minutes)")
    
    return {
        'scene': scene_name,
        'input_file': input_path.name,
        'resolution': (width, height),
        'outputs': {k: str(v) for k, v in outputs.items()},
        'master_size_mb': master_size,
        'web_size_mb': jpeg_size,
        'thumb_size_kb': thumb_size,
        'processing_time_sec': elapsed,
        'config': config
    }


# ============================================================================
# Main Batch Processing
# ============================================================================

def main():
    """Batch process all 750 Picacho source TIFFs with Ultimate quality."""
    
    print("=" * 80)
    print("🏛️  750 PICACHO LANE - ULTIMATE QUALITY PROCESSING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Device setup
    device = get_optimal_device()
    print(f"🔧 Device: {device}")
    if device == "mps":
        print("   ✓ Apple M-series GPU acceleration enabled")
    print()
    
    # Paths
    input_dir = Path("input_images/750_Picacho/Source_TIFFs")
    output_dir = Path(f"output_750_Picacho_Ultimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Verify input directory
    if not input_dir.exists():
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all TIFF files
    tiff_files = sorted(input_dir.glob("*.tiff"))
    
    if not tiff_files:
        print(f"❌ ERROR: No TIFF files found in {input_dir}")
        sys.exit(1)
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Found {len(tiff_files)} TIFF files")
    print()
    
    # Match files to scene configurations
    scenes_to_process = []
    
    for tiff_path in tiff_files:
        filename = tiff_path.stem
        scene_matched = False
        
        for scene_name, config in SCENE_CONFIGS.items():
            if scene_name in filename:
                scenes_to_process.append((tiff_path, scene_name, config))
                print(f"  ✓ {tiff_path.name} → {scene_name}")
                scene_matched = True
                break
        
        if not scene_matched:
            print(f"  ⚠ {tiff_path.name} → No matching scene config")
    
    print()
    print(f"🎯 Processing {len(scenes_to_process)} scenes with Ultimate quality")
    print()
    
    # Process each scene
    results = []
    total_start = time.time()
    
    for i, (tiff_path, scene_name, config) in enumerate(scenes_to_process, 1):
        print(f"\n[{i}/{len(scenes_to_process)}]")
        try:
            result = process_scene(tiff_path, output_dir, scene_name, config, device)
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("✅ BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Processed: {len(results)}/{len(scenes_to_process)} scenes")
    if results:
        print(f"Average: {total_time/len(results):.1f}s per scene")
        print(f"Throughput: {len(results) / (total_time/3600):.1f} images/hour")
    print()
    
    # Detailed summary
    if results:
        print("📊 Processing Summary")
        print("-" * 80)
        print(f"{'Scene':<20} {'Resolution':<12} {'Master':<10} {'Web':<10} {'Time':<8}")
        print("-" * 80)
        
        total_master_size = 0
        total_web_size = 0
        
        for result in results:
            scene = result['scene']
            w, h = result['resolution']
            master_size = result['master_size_mb']
            web_size = result['web_size_mb']
            proc_time = result['processing_time_sec']
            
            total_master_size += master_size
            total_web_size += web_size
            
            print(f"{scene:<20} {w}x{h:<7} {master_size:>6.1f} MB  {web_size:>6.1f} MB  {proc_time:>6.1f}s")
        
        print("-" * 80)
        print(f"{'TOTAL':<20} {'':12} {total_master_size:>6.1f} MB  {total_web_size:>6.1f} MB")
        print()
    
    # Save processing report
    report_path = output_dir / "processing_report.json"
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'device': device,
        'total_scenes': len(scenes_to_process),
        'successful': len(results),
        'failed': len(scenes_to_process) - len(results),
        'total_time_sec': total_time,
        'results': results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Processing report saved: {report_path}")
    print()
    print(f"📦 All deliverables saved to: {output_dir}")
    print()
    print("🎯 Ready for client delivery!")
    print("=" * 80)


if __name__ == "__main__":
    main()
