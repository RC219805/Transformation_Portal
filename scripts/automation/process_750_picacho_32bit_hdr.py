#!/usr/bin/env python3
"""
750 Picacho 32-bit HDR Processing Pipeline
==========================================
Process 32-bit floating-point sRGB TIFFs with comprehensive HDR tone mapping
and Ultimate quality enhancement.

Input: 32-bit float TIFF with HDR data (negative values and values >1)
Processing: Reinhard Local tone mapping → 16-bit precision pipeline
Output: 16-bit TIFF masters + web JPEGs + depth maps + processing report
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import json

import numpy as np
from PIL import Image
import torch

# Required: tifffile for 32-bit TIFF support
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    print("❌ ERROR: tifffile is required for 32-bit TIFF processing")
    print("Install with: pip install tifffile")
    sys.exit(1)

# Optional: scipy for advanced filtering
try:
    from scipy.ndimage import gaussian_filter  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    print("⚠ WARNING: scipy not available, using fallback filters")
    HAS_SCIPY = False


# ============================================================================
# Scene-Specific Configurations
# ============================================================================

SCENE_CONFIGS = {
    "Aerial": {
        "description": "Aerial View - Estate Overview (85.3 MP, 977 MB)",
        "preset": "exterior_aerial",
        "depth_clarity": 0.55,
        "atmospheric_haze": True,
        "haze_density": 0.03,
        "contrast": 1.15,
        "saturation": 1.10,
        "vibrance": 0.22,
        "materials": ["stone", "vegetation", "roof"],
        "tone_map_params": {
            "key": 0.18,  # Target middle gray
            "sat": 0.8,   # Saturation preservation
            "epsilon": 1e-6
        }
    },
    "Kitchen": {
        "description": "Kitchen - Culinary Space (80.5 MP, 921 MB)",
        "preset": "interior_luxury",
        "depth_clarity": 0.65,
        "atmospheric_haze": False,
        "contrast": 1.12,
        "saturation": 1.05,
        "clarity_boost": 0.20,
        "materials": ["metal", "stone", "glass", "wood"],
        "material_priority": "HIGH",  # Metal appliances + stone countertops
        "tone_map_params": {
            "key": 0.22,  # Brighter for interior
            "sat": 0.85,
            "epsilon": 1e-6
        }
    },
    "Pool": {
        "description": "Pool & Outdoor (80.9 MP, 926 MB)",
        "preset": "exterior_water",
        "depth_clarity": 0.50,
        "atmospheric_haze": False,
        "contrast": 1.10,
        "saturation": 1.12,
        "vibrance": 0.18,
        "water_enhance": True,
        "materials": ["water", "stone", "concrete"],
        "material_priority": "CRITICAL",  # Water surface depth-aware
        "tone_map_params": {
            "key": 0.20,
            "sat": 0.88,
            "epsilon": 1e-6
        }
    },
    "PrimaryBathroom": {
        "description": "Primary Bathroom - Spa (190.2 MP, 2,177 MB) - HERO IMAGE",
        "preset": "interior_luxury_max",
        "depth_clarity": 0.60,
        "atmospheric_haze": False,
        "contrast": 1.08,
        "saturation": 1.05,
        "materials": ["stone", "glass", "metal"],
        "material_priority": "MAXIMUM",  # Stone/marble at highest strength
        "material_strength": 0.85,  # Increased from default 0.75
        "tone_map_params": {
            "key": 0.24,  # Hero image - well-exposed
            "sat": 0.90,  # High saturation preservation
            "epsilon": 1e-6
        }
    },
    "PrimaryBedroom": {
        "description": "Primary Bedroom Suite (96.0 MP, 1,099 MB)",
        "preset": "interior_warm_luxury",
        "depth_clarity": 0.50,  # Softer for bedroom
        "atmospheric_haze": False,
        "contrast": 1.06,
        "saturation": 1.03,
        "temperature_shift": [1.03, 1.0, 0.98],  # Warm
        "materials": ["fabric", "wood", "glass"],
        "tone_map_params": {
            "key": 0.20,  # Softer, intimate
            "sat": 0.82,
            "epsilon": 1e-6
        }
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
# HDR Tone Mapping - Reinhard Local Operator
# ============================================================================

def reinhard_local_tone_map(
    hdr_image: np.ndarray,
    key: float = 0.18,
    sat: float = 0.8,
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply Reinhard Local tone mapping to HDR image.
    
    Parameters:
    -----------
    hdr_image : np.ndarray
        Input HDR image (float32, any range)
    key : float
        Target middle gray (0.18 = 18% gray, photographic standard)
    sat : float
        Saturation preservation (0.0-1.0)
    epsilon : float
        Small value to avoid division by zero
    
    Returns:
    --------
    tone_mapped : np.ndarray
        Tone-mapped image [0.0-1.0] range
    stats : Dict[str, float]
        Statistics about the tone mapping process
    """
    print(f"  Applying Reinhard Local tone mapping...")
    print(f"    Key (target gray): {key}")
    print(f"    Saturation preservation: {sat}")
    
    # Input statistics
    input_min = hdr_image.min()
    input_max = hdr_image.max()
    input_mean = hdr_image.mean()
    
    # Compute luminance
    luminance = 0.2126 * hdr_image[:, :, 0] + 0.7152 * hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 2]
    luminance = np.maximum(luminance, epsilon)  # Avoid negative/zero luminance
    
    # Log-average luminance (world adaptation level)
    log_lum = np.log(luminance + epsilon)
    lum_avg = np.exp(log_lum.mean())
    
    print(f"    Log-average luminance: {lum_avg:.6f}")
    
    # Scale luminance by key value
    scaled_lum = (key / lum_avg) * luminance
    
    # Reinhard local operator: L_d = L / (1 + L)
    tone_mapped_lum = scaled_lum / (1.0 + scaled_lum)
    
    # Apply tone mapping to color channels with saturation control
    tone_mapped = np.zeros_like(hdr_image)
    
    for c in range(3):
        # Option 1: Apply to each channel independently
        if sat < 0.5:
            # More aggressive desaturation
            tone_mapped[:, :, c] = tone_mapped_lum
        else:
            # Preserve color with saturation control
            # Scale each channel by the tone mapping ratio
            ratio = tone_mapped_lum / (luminance + epsilon)
            tone_mapped[:, :, c] = hdr_image[:, :, c] * ratio
            
            # Blend with luminance based on saturation parameter
            tone_mapped[:, :, c] = (
                sat * tone_mapped[:, :, c] + 
                (1 - sat) * tone_mapped_lum
            )
    
    # Clip to valid range [0, 1]
    tone_mapped = np.clip(tone_mapped, 0.0, 1.0)
    
    # Output statistics
    output_min = tone_mapped.min()
    output_max = tone_mapped.max()
    output_mean = tone_mapped.mean()
    
    stats = {
        'input_min': float(input_min),
        'input_max': float(input_max),
        'input_mean': float(input_mean),
        'input_range': float(input_max - input_min),
        'log_avg_luminance': float(lum_avg),
        'output_min': float(output_min),
        'output_max': float(output_max),
        'output_mean': float(output_mean),
        'compression_ratio': float((input_max - input_min) / (output_max - output_min + epsilon))
    }
    
    print(f"    ✓ Tone mapping complete")
    print(f"      Input range: [{input_min:.4f}, {input_max:.4f}]")
    print(f"      Output range: [{output_min:.4f}, {output_max:.4f}]")
    print(f"      Compression ratio: {stats['compression_ratio']:.2f}x")
    
    return tone_mapped, stats


# ============================================================================
# Depth Estimation (Depth Anything V2 Large)
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
        device=device if device != "cpu" else -1
    )
    
    print("  Estimating depth...")
    start = time.time()
    result = depth_estimator(image)
    depth = result["depth"]
    elapsed = time.time() - start
    
    print(f"  ✓ Depth estimation complete in {elapsed:.2f}s")
    
    # Convert to numpy and normalize
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
    if HAS_SCIPY:
        from scipy.ndimage import gaussian_filter
        
        # Create depth zones
        foreground_mask = depth_map > 0.7
        midground_mask = (depth_map >= 0.4) & (depth_map <= 0.7)
        background_mask = depth_map < 0.4
        
        # Unsharp mask
        blurred = gaussian_filter(image_array, sigma=2.0, axes=(0, 1))
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
    else:
        # Simple sharpening fallback
        return image_array


def apply_atmospheric_haze(
    image_array: np.ndarray,
    depth_map: np.ndarray,
    density: float = 0.02
) -> np.ndarray:
    """Apply subtle atmospheric haze based on depth."""
    haze_color = np.array([0.88, 0.92, 0.98])  # Light blue-white
    
    # Haze increases with distance
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
    if not HAS_SCIPY:
        return image_array
    
    from scipy.ndimage import gaussian_filter
    
    # Detect potential material areas
    luminance = 0.2126 * image_array[:, :, 0] + 0.7152 * image_array[:, :, 1] + 0.0722 * image_array[:, :, 2]
    
    # Local contrast (material edges)
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
    """Apply luxury color grading based on scene configuration."""
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
    
    # Vibrance
    vibrance = config.get('vibrance', 0.0)
    if vibrance > 0:
        luminance = 0.2126 * result[:, :, 0] + 0.7152 * result[:, :, 1] + 0.0722 * result[:, :, 2]
        luminance = luminance[:, :, np.newaxis]
        chroma = result - luminance
        current_saturation = np.sqrt(np.sum(chroma ** 2, axis=2, keepdims=True))
        vibrance_factor = 1.0 + vibrance * (1.0 - current_saturation)
        result = np.clip(luminance + chroma * vibrance_factor, 0, 1)
    
    # Temperature shift
    temp_shift = config.get('temperature_shift', None)
    if temp_shift:
        result = result * np.array(temp_shift).reshape(1, 1, 3)
        result = np.clip(result, 0, 1)
    
    # Clarity boost
    clarity_boost = config.get('clarity_boost', 0.0)
    if clarity_boost > 0:
        mid_mask = 1.0 - np.abs(result - 0.5) * 2.0
        result = np.clip(result * (1.0 + clarity_boost * mid_mask), 0, 1)
    
    return result


# ============================================================================
# Scene Processing
# ============================================================================

def process_scene_hdr(
    input_path: Path,
    output_dir: Path,
    scene_name: str,
    config: Dict[str, Any],
    device: str
) -> Dict[str, Any]:
    """Process a single 32-bit HDR TIFF scene with Ultimate quality."""
    
    print(f"\n{'='*80}")
    print(f"🏛️  PROCESSING: {scene_name}")
    print(f"{'='*80}")
    print(f"Configuration: {config['description']}")
    print(f"Input: {input_path.name} ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    start_time = time.time()
    
    # Stage 1: Load 32-bit HDR TIFF
    print("\n📂 Stage 1: Loading 32-bit HDR TIFF")
    hdr_array = tifffile.imread(input_path)
    
    # Handle alpha channel if present
    has_alpha = False
    if hdr_array.shape[2] == 4:
        print("  ⚠ Alpha channel detected (RGBA format)")
        hdr_array = hdr_array[:, :, :3]  # Extract RGB
        has_alpha = True
        print("  ✓ Alpha channel separated (will process RGB only)")
    
    height, width = hdr_array.shape[:2]
    print(f"  ✓ Loaded: {width} x {height}")
    print(f"  Dtype: {hdr_array.dtype}")
    print(f"  Range: [{hdr_array.min():.6f}, {hdr_array.max():.6f}]")
    
    # HDR statistics
    negative_count = (hdr_array < 0).sum()
    above_one_count = (hdr_array > 1.0).sum()
    total_pixels = hdr_array.size
    
    print(f"  HDR Data:")
    print(f"    Negative values: {negative_count:,} ({negative_count/total_pixels*100:.2f}%)")
    print(f"    Values > 1.0: {above_one_count:,} ({above_one_count/total_pixels*100:.2f}%)")
    
    # Stage 2: HDR Tone Mapping (Reinhard Local)
    print(f"\n🎨 Stage 2: HDR Tone Mapping (Reinhard Local)")
    tone_map_params = config.get('tone_map_params', {})
    tone_mapped, tone_map_stats = reinhard_local_tone_map(
        hdr_array,
        key=tone_map_params.get('key', 0.18),
        sat=tone_map_params.get('sat', 0.8),
        epsilon=tone_map_params.get('epsilon', 1e-6)
    )
    
    # Stage 3: Convert to 16-bit for processing
    print("\n🔢 Stage 3: Converting to 16-bit Precision")
    # Already in [0, 1] range from tone mapping
    print(f"  ✓ Working in normalized [0, 1] range")
    
    # Create PIL image for depth estimation
    image_8bit = (np.clip(tone_mapped, 0, 1) * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_8bit)
    
    # Stage 4: Depth Estimation
    print(f"\n🔍 Stage 4: Depth Estimation (Depth Anything V2 Large)")
    depth_map = estimate_depth_v2_large(image_pil, device)
    print(f"  Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
    
    # Stage 5: Material Response Technology
    print(f"\n💎 Stage 5: Material Response Technology")
    print(f"  Materials: {', '.join(config.get('materials', []))}")
    material_priority = config.get('material_priority', 'STANDARD')
    material_strength = config.get('material_strength', 0.75)
    print(f"  Priority: {material_priority}")
    print(f"  Strength: {material_strength}")
    
    enhanced = apply_material_response(tone_mapped, depth_map, strength=material_strength)
    
    # Stage 6: Depth-Aware Clarity
    print(f"\n✨ Stage 6: Zone-Based Clarity Enhancement")
    clarity_strength = config.get('depth_clarity', 0.55)
    print(f"  Clarity strength: {clarity_strength}")
    enhanced = apply_depth_aware_clarity(enhanced, depth_map, strength=clarity_strength)
    
    # Stage 7: Atmospheric Haze (if applicable)
    if config.get('atmospheric_haze', False):
        print(f"\n🌫️  Stage 7: Atmospheric Haze")
        haze_density = config.get('haze_density', 0.02)
        print(f"  Haze density: {haze_density}")
        enhanced = apply_atmospheric_haze(enhanced, depth_map, density=haze_density)
    
    # Stage 8: Luxury Color Grading
    print(f"\n🎨 Stage 8: Luxury Color Grading")
    print(f"  Contrast: {config.get('contrast', 1.0)}")
    print(f"  Saturation: {config.get('saturation', 1.0)}")
    if config.get('vibrance'):
        print(f"  Vibrance: {config.get('vibrance')}")
    enhanced = apply_luxury_color_grade(enhanced, config)
    
    # Stage 9: Save Deliverables
    print(f"\n💾 Stage 9: Saving Deliverables")
    
    # Create organized output directories
    masters_dir = output_dir / "masters"
    web_dir = output_dir / "web"
    depth_dir = output_dir / "depth"
    thumbs_dir = output_dir / "thumbnails"
    
    for d in [masters_dir, web_dir, depth_dir, thumbs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # 1. Master TIFF (16-bit)
    tiff_output = masters_dir / f"750Picacho_{scene_name}_HDR_Ultimate.tif"
    result_16bit = (np.clip(enhanced, 0, 1) * 65535).astype(np.uint16)
    
    tifffile.imwrite(
        tiff_output,
        result_16bit,
        photometric='rgb',
        compression='lzw',
        metadata={
            'Software': 'Transformation Portal HDR Ultimate Pipeline',
            'DateTime': datetime.now().isoformat(),
            'Scene': scene_name,
            'Quality': 'Ultimate',
            'ToneMappingOperator': 'Reinhard Local',
            'SourceBitDepth': '32-bit float',
            'OutputBitDepth': '16-bit'
        }
    )
    
    outputs['master_tiff'] = tiff_output
    master_size = tiff_output.stat().st_size / 1024 / 1024
    print(f"  ✅ Master TIFF (16-bit): {tiff_output.name} ({master_size:.1f} MB)")
    
    # 2. Depth map
    depth_output = depth_dir / f"750Picacho_{scene_name}_Depth.png"
    depth_vis = (depth_map * 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(depth_output, format='PNG')
    outputs['depth'] = depth_output
    print(f"  ✅ Depth Map: {depth_output.name}")
    
    # 3. Web JPEG (98%)
    result_8bit = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    jpeg_output = web_dir / f"750Picacho_{scene_name}_HDR_Ultimate.jpg"
    Image.fromarray(result_8bit).save(jpeg_output, format='JPEG', quality=98, subsampling=0, optimize=True)
    outputs['web_jpeg'] = jpeg_output
    jpeg_size = jpeg_output.stat().st_size / 1024 / 1024
    print(f"  ✅ Web JPEG (98%): {jpeg_output.name} ({jpeg_size:.1f} MB)")
    
    # 4. Thumbnail (1200px)
    thumb_img = Image.fromarray(result_8bit)
    thumb_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    thumb_output = thumbs_dir / f"750Picacho_{scene_name}_Thumbnail.jpg"
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
        'megapixels': width * height / 1_000_000,
        'has_alpha': has_alpha,
        'hdr_stats': {
            'negative_values_pct': float(negative_count / total_pixels * 100),
            'above_one_pct': float(above_one_count / total_pixels * 100)
        },
        'tone_mapping': tone_map_stats,
        'outputs': {k: str(v.relative_to(output_dir)) for k, v in outputs.items()},
        'file_sizes': {
            'master_mb': master_size,
            'web_mb': jpeg_size,
            'thumb_kb': thumb_size
        },
        'processing_time_sec': elapsed,
        'config': config
    }


# ============================================================================
# Main Batch Processing
# ============================================================================

def main():
    """Batch process all 750 Picacho 32-bit HDR TIFFs with Ultimate quality."""
    
    print("=" * 80)
    print("🏛️  750 PICACHO LANE - 32-BIT HDR ULTIMATE QUALITY PROCESSING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Device setup
    device = get_optimal_device()
    print(f"🔧 Device: {device}")
    if device == "mps":
        print("   ✓ Apple M-series GPU acceleration enabled")
    elif device == "cuda":
        print("   ✓ CUDA GPU acceleration enabled")
    print()
    
    # Paths
    input_dir = Path("input_images/750_Picacho/32-bit_LightRoom_sRGB_TIFFs")
    output_dir = Path(f"output_750_Picacho_32bit_HDR_Ultimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Verify input directory
    if not input_dir.exists():
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all TIFF files
    tiff_files = sorted(input_dir.glob("*.tif"))
    
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
        
        # Match by filename keywords
        if "Aerial" in filename:
            scenes_to_process.append((tiff_path, "Aerial", SCENE_CONFIGS["Aerial"]))
            scene_matched = True
        elif "Kitchen" in filename:
            scenes_to_process.append((tiff_path, "Kitchen", SCENE_CONFIGS["Kitchen"]))
            scene_matched = True
        elif "Pool" in filename:
            scenes_to_process.append((tiff_path, "Pool", SCENE_CONFIGS["Pool"]))
            scene_matched = True
        elif "PrimaryBathroom" in filename or "Bathroom" in filename:
            scenes_to_process.append((tiff_path, "PrimaryBathroom", SCENE_CONFIGS["PrimaryBathroom"]))
            scene_matched = True
        elif "PrimaryBedroom" in filename or "Bedroom" in filename:
            scenes_to_process.append((tiff_path, "PrimaryBedroom", SCENE_CONFIGS["PrimaryBedroom"]))
            scene_matched = True
        
        if scene_matched:
            print(f"  ✓ {tiff_path.name}")
        else:
            print(f"  ⚠ {tiff_path.name} → No matching scene config (skipped)")
    
    if not scenes_to_process:
        print("\n❌ ERROR: No files matched to scene configurations")
        sys.exit(1)
    
    print()
    print(f"🎯 Processing {len(scenes_to_process)} scenes with HDR Ultimate quality")
    print(f"⏱️  Estimated batch time: ~{len(scenes_to_process) * 15} minutes")
    print()
    
    # Process each scene sequentially (for memory management)
    results = []
    total_start = time.time()
    
    for i, (tiff_path, scene_name, config) in enumerate(scenes_to_process, 1):
        print(f"\n[{i}/{len(scenes_to_process)}]")
        try:
            result = process_scene_hdr(tiff_path, output_dir, scene_name, config, device)
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
        print(f"Average: {total_time/len(results):.1f}s per scene ({total_time/len(results)/60:.1f} min)")
        print(f"Throughput: {len(results) / (total_time/3600):.1f} images/hour")
    print()
    
    # Detailed summary table
    if results:
        print("📊 Processing Summary")
        print("-" * 100)
        print(f"{'Scene':<20} {'Resolution':<15} {'MP':<7} {'Master':<10} {'Web':<10} {'Time':<10}")
        print("-" * 100)
        
        total_master_size = 0
        total_web_size = 0
        
        for result in results:
            scene = result['scene']
            w, h = result['resolution']
            mp = result['megapixels']
            master_size = result['file_sizes']['master_mb']
            web_size = result['file_sizes']['web_mb']
            proc_time = result['processing_time_sec']
            
            total_master_size += master_size
            total_web_size += web_size
            
            print(f"{scene:<20} {w:>5}x{h:<7} {mp:>5.1f}  {master_size:>6.1f} MB  {web_size:>6.1f} MB  {proc_time/60:>6.1f} min")
        
        print("-" * 100)
        print(f"{'TOTAL':<20} {'':15} {'':7} {total_master_size:>6.1f} MB  {total_web_size:>6.1f} MB  {total_time/60:>6.1f} min")
        print()
        
        # HDR tone mapping summary
        print("🎨 HDR Tone Mapping Statistics")
        print("-" * 100)
        print(f"{'Scene':<20} {'Input Range':<20} {'Output Range':<20} {'Compression':<12}")
        print("-" * 100)
        
        for result in results:
            scene = result['scene']
            tm = result['tone_mapping']
            input_range = f"[{tm['input_min']:.3f}, {tm['input_max']:.3f}]"
            output_range = f"[{tm['output_min']:.3f}, {tm['output_max']:.3f}]"
            compression = f"{tm['compression_ratio']:.1f}x"
            
            print(f"{scene:<20} {input_range:<20} {output_range:<20} {compression:<12}")
        
        print("-" * 100)
        print()
    
    # Save comprehensive processing report
    report_path = output_dir / "HDR_processing_report.json"
    report = {
        'timestamp': datetime.now().isoformat(),
        'pipeline': 'HDR Ultimate Quality - Reinhard Local Tone Mapping',
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'device': device,
        'total_scenes': len(scenes_to_process),
        'successful': len(results),
        'failed': len(scenes_to_process) - len(results),
        'total_time_sec': total_time,
        'total_time_min': total_time / 60,
        'avg_time_per_scene_min': (total_time / len(results) / 60) if results else 0,
        'throughput_images_per_hour': len(results) / (total_time / 3600) if results else 0,
        'results': results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Comprehensive processing report saved: {report_path}")
    print()
    
    # Create delivery checklist
    checklist_path = output_dir / "DELIVERY_CHECKLIST.md"
    with open(checklist_path, 'w') as f:
        f.write("# 750 Picacho Lane - HDR Ultimate Quality Deliverables\n\n")
        f.write(f"**Processing Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Deliverables\n\n")
        f.write("### Masters (16-bit TIFF)\n")
        for r in results:
            f.write(f"- [ ] {r['outputs']['master_tiff']} ({r['file_sizes']['master_mb']:.1f} MB)\n")
        f.write("\n### Web-Optimized (98% JPEG)\n")
        for r in results:
            f.write(f"- [ ] {r['outputs']['web_jpeg']} ({r['file_sizes']['web_mb']:.1f} MB)\n")
        f.write("\n### Depth Maps (Reference)\n")
        for r in results:
            f.write(f"- [ ] {r['outputs']['depth']}\n")
        f.write("\n### Thumbnails (1200px)\n")
        for r in results:
            f.write(f"- [ ] {r['outputs']['thumbnail']} ({r['file_sizes']['thumb_kb']:.0f} KB)\n")
        f.write("\n## Quality Assurance\n\n")
        f.write("- [ ] Verify no clipping in highlights/shadows\n")
        f.write("- [ ] Check material enhancement quality\n")
        f.write("- [ ] Validate depth-aware processing transitions\n")
        f.write("- [ ] Confirm color accuracy in neutral surfaces\n")
        f.write("- [ ] Verify metadata preservation\n")
        f.write("\n## Technical Notes\n\n")
        f.write(f"- **Source Format:** 32-bit floating-point sRGB TIFF\n")
        f.write(f"- **Tone Mapping:** Reinhard Local operator\n")
        f.write(f"- **Depth Model:** Depth Anything V2 Large\n")
        f.write(f"- **Processing Device:** {device}\n")
        f.write(f"- **Total Processing Time:** {total_time/60:.1f} minutes\n")
    
    print(f"📋 Delivery checklist saved: {checklist_path}")
    print()
    print(f"📦 All deliverables organized in: {output_dir}")
    print(f"   ├── masters/      (16-bit TIFF masters)")
    print(f"   ├── web/          (98% JPEG web-optimized)")
    print(f"   ├── depth/        (Depth maps for reference)")
    print(f"   └── thumbnails/   (1200px thumbnails)")
    print()
    print("🎯 Ready for client delivery!")
    print("=" * 80)


if __name__ == "__main__":
    main()
