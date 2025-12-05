#!/usr/bin/env python3
"""
750 Picacho 32-bit HDR Processing Pipeline - ENHANCED VERSION
==============================================================
Enhanced version with Phase 1 strategic improvements:
- Adaptive tone mapping
- HDR visualizations
- Time prediction
- QA validation
- Enhanced reporting
- Alpha channel handling
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image
import torch

# Phase 1 enhancements
sys.path.insert(0, str(Path(__file__).parent))
from utils.adaptive_tone_mapping import AdaptiveToneMapper
from utils.alpha_compositor import AlphaCompositor
from utils.enhanced_reporter import ProcessingReport, create_client_deliverable_summary
from tools.time_predictor import ProcessingTimePredictor, ImageMetadata
from tools.hdr_visualizer import HDRVisualizer
from tools.qa_validator import QAValidator

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
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    print("⚠ WARNING: scipy not available, using fallback filters")
    HAS_SCIPY = False


# ============================================================================
# Scene-Specific Configurations (same as original)
# ============================================================================

SCENE_CONFIGS = {
    "Aerial": {
        "description": "Aerial View - Estate Overview",
        "preset": "exterior_aerial",
        "depth_clarity": 0.55,
        "atmospheric_haze": True,
        "haze_density": 0.03,
        "contrast": 1.15,
        "saturation": 1.10,
        "vibrance": 0.22,
        "materials": ["stone", "vegetation", "roof"],
    },
    "Kitchen": {
        "description": "Kitchen - Culinary Space",
        "preset": "interior_luxury",
        "depth_clarity": 0.65,
        "contrast": 1.12,
        "saturation": 1.05,
        "clarity_boost": 0.20,
        "materials": ["metal", "stone", "glass", "wood"],
        "material_priority": "HIGH",
    },
    "Pool": {
        "description": "Pool & Outdoor",
        "preset": "exterior_water",
        "depth_clarity": 0.50,
        "contrast": 1.10,
        "saturation": 1.12,
        "vibrance": 0.18,
        "water_enhance": True,
        "materials": ["water", "stone", "concrete"],
        "material_priority": "CRITICAL",
    },
    "PrimaryBathroom": {
        "description": "Primary Bathroom - Spa (HERO IMAGE)",
        "preset": "interior_luxury_max",
        "depth_clarity": 0.60,
        "contrast": 1.08,
        "saturation": 1.05,
        "materials": ["stone", "glass", "metal"],
        "material_priority": "MAXIMUM",
        "material_strength": 0.85,
    },
    "PrimaryBedroom": {
        "description": "Primary Bedroom Suite",
        "preset": "interior_warm_luxury",
        "depth_clarity": 0.50,
        "contrast": 1.06,
        "saturation": 1.03,
        "temperature_shift": [1.03, 1.0, 0.98],
        "materials": ["fabric", "wood", "glass"],
    }
}


# ============================================================================
# Device Setup
# ============================================================================

def get_optimal_device() -> str:
    """Get the best available device for processing."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================================
# Depth Estimation
# ============================================================================

def estimate_depth_v2_large(image: Image.Image, device: str) -> np.ndarray:
    """Estimate depth using Depth Anything V2 Large."""
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
# Processing Functions (simplified - using adaptive tone mapping)
# ============================================================================

def apply_depth_aware_clarity(
    image_array: np.ndarray,
    depth_map: np.ndarray,
    strength: float = 0.55
) -> np.ndarray:
    """Apply depth-aware clarity enhancement."""
    if HAS_SCIPY:
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
        return image_array


def apply_luxury_color_grade(
    image_array: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """Apply luxury color grading."""
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
    
    return result


# ============================================================================
# Enhanced Scene Processing
# ============================================================================

def process_scene_hdr_enhanced(
    input_path: Path,
    output_dir: Path,
    scene_name: str,
    config: Dict[str, Any],
    device: str,
    tone_mapper: AdaptiveToneMapper,
    visualizer: HDRVisualizer,
    compositor: AlphaCompositor
) -> Dict[str, Any]:
    """Process scene with Phase 1 enhancements."""
    
    print(f"\n{'='*80}")
    print(f"🏛️  PROCESSING: {scene_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Stage 1: Load HDR TIFF
    print("\n📂 Stage 1: Loading 32-bit HDR TIFF")
    hdr_array = tifffile.imread(input_path)
    
    # Handle alpha channel
    has_alpha = False
    alpha_channel = None
    if hdr_array.shape[2] == 4:
        print("  ⚠ Alpha channel detected")
        alpha_channel = hdr_array[:, :, 3]
        hdr_array = hdr_array[:, :, :3]
        has_alpha = True
    
    height, width = hdr_array.shape[:2]
    megapixels = width * height / 1_000_000
    
    print(f"  ✓ Loaded: {width} x {height} ({megapixels:.1f} MP)")
    
    # Stage 2: ADAPTIVE Tone Mapping
    print(f"\n🎨 Stage 2: Adaptive HDR Tone Mapping")
    tone_mapped, tone_metadata = tone_mapper.apply_adaptive_tone_mapping(hdr_array)
    
    # Print adaptive analysis
    analysis = tone_metadata['analysis']
    print(f"  Scene type: {analysis['scene_classification']}")
    params = analysis['recommended_params']
    print(f"  Key: {params['key']:.4f}, Saturation: {params['sat']:.4f}")
    
    # Stage 3: Generate HDR visualizations
    print(f"\n📊 Stage 3: Generating HDR Visualizations")
    
    # Save temporary files for visualization
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Save HDR input (for visualization)
    hdr_temp = temp_dir / f"{scene_name}_hdr_input.tif"
    tifffile.imwrite(hdr_temp, hdr_array.astype(np.float32))
    
    # Save tone-mapped (for visualization)
    tm_temp = temp_dir / f"{scene_name}_tone_mapped.tif"
    tifffile.imwrite(tm_temp, (np.clip(tone_mapped, 0, 1) * 65535).astype(np.uint16))
    
    # Generate visualizations
    viz_dir = output_dir / "visualizations"
    visualizer_scene = HDRVisualizer(viz_dir)
    visualizer_scene.generate_histogram_comparison(hdr_temp, tm_temp, scene_name, is_hdr=True)
    visualizer_scene.generate_luminance_distribution(hdr_temp, tm_temp, scene_name)
    visualizer_scene.generate_dynamic_range_comparison(hdr_temp, tm_temp, scene_name)
    
    # Stage 4: Depth Estimation
    print(f"\n🔍 Stage 4: Depth Estimation")
    image_8bit = (np.clip(tone_mapped, 0, 1) * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_8bit)
    depth_map = estimate_depth_v2_large(image_pil, device)
    
    # Stage 5: Depth-Aware Clarity
    print(f"\n✨ Stage 5: Zone-Based Clarity Enhancement")
    clarity_strength = config.get('depth_clarity', 0.55)
    enhanced = apply_depth_aware_clarity(tone_mapped, depth_map, strength=clarity_strength)
    
    # Stage 6: Color Grading
    print(f"\n🎨 Stage 6: Luxury Color Grading")
    enhanced = apply_luxury_color_grade(enhanced, config)
    
    # Stage 7: Alpha Channel Handling
    if has_alpha:
        print(f"\n🎭 Stage 7: Alpha Channel Processing")
        # Restore alpha for compositing
        enhanced_rgba = np.dstack([enhanced, alpha_channel])
        
        # Generate alpha variants
        alpha_dir = output_dir / "alpha_variants" / scene_name
        alpha_paths = compositor.save_variants(
            enhanced_rgba,
            alpha_dir,
            f"750Picacho_{scene_name}",
            modes=['preserve', 'flatten-white', 'flatten-black']
        )
    
    # Stage 8: Save Deliverables
    print(f"\n💾 Stage 8: Saving Deliverables")
    
    masters_dir = output_dir / "masters"
    web_dir = output_dir / "web"
    depth_dir = output_dir / "depth"
    thumbs_dir = output_dir / "thumbnails"
    
    for d in [masters_dir, web_dir, depth_dir, thumbs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # Master TIFF
    tiff_output = masters_dir / f"750Picacho_{scene_name}_HDR_Ultimate.tif"
    result_16bit = (np.clip(enhanced, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(tiff_output, result_16bit, photometric='rgb', compression='lzw')
    outputs['master_tiff'] = tiff_output
    
    # Web JPEG
    result_8bit = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    jpeg_output = web_dir / f"750Picacho_{scene_name}_HDR_Ultimate.jpg"
    Image.fromarray(result_8bit).save(jpeg_output, format='JPEG', quality=98, optimize=True)
    outputs['web_jpeg'] = jpeg_output
    
    # Depth map
    depth_output = depth_dir / f"750Picacho_{scene_name}_Depth.png"
    depth_vis = (depth_map * 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(depth_output)
    outputs['depth'] = depth_output
    
    # Thumbnail
    thumb_img = Image.fromarray(result_8bit)
    thumb_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    thumb_output = thumbs_dir / f"750Picacho_{scene_name}_Thumbnail.jpg"
    thumb_img.save(thumb_output, format='JPEG', quality=92, optimize=True)
    outputs['thumbnail'] = thumb_output
    
    elapsed = time.time() - start_time
    print(f"\n✅ Scene complete in {elapsed:.1f}s ({elapsed/60:.2f} minutes)")
    
    return {
        'scene_name': scene_name,
        'input_file': input_path,
        'output_files': outputs,
        'processing_time_sec': elapsed,
        'metrics': {
            'width': width,
            'height': height,
            'megapixels': megapixels,
            'bit_depth': 32,
            'is_hdr': True,
            'has_alpha': has_alpha
        },
        'tone_mapping': {
            **analysis['luminance_stats'],
            **analysis['histogram_stats'],
            'parameters_used': params,
            'reasoning': analysis['reasoning']
        },
        'depth': {
            'min': float(depth_map.min()),
            'max': float(depth_map.max()),
            'mean': float(depth_map.mean())
        }
    }


# ============================================================================
# Main Pipeline with Phase 1 Enhancements
# ============================================================================

def main():
    """Enhanced batch processing with Phase 1 improvements."""
    
    print("=" * 80)
    print("🏛️  750 PICACHO LANE - ENHANCED HDR PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Device setup
    device = get_optimal_device()
    print(f"🔧 Device: {device}")
    print()
    
    # Paths
    input_dir = Path("input_images/750_Picacho/32-bit_LightRoom_sRGB_TIFFs")
    output_dir = Path(f"output_750_Picacho_32bit_HDR_Enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if not input_dir.exists():
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    tiff_files = sorted(input_dir.glob("*.tif"))
    
    if not tiff_files:
        print(f"❌ ERROR: No TIFF files found")
        sys.exit(1)
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Found {len(tiff_files)} TIFF files")
    print()
    
    # Phase 1 Enhancement: QA Validation
    print("=" * 80)
    print("🔍 PHASE 1: PRE-FLIGHT QA VALIDATION")
    print("=" * 80)
    validator = QAValidator()
    qa_summary = validator.validate_batch(tiff_files)
    
    if qa_summary['invalid'] > 0:
        print(f"⚠️ WARNING: {qa_summary['invalid']} files failed validation")
        print("Proceeding with valid files only...")
    
    # Filter to valid files only
    valid_files = [
        Path(v['path']) for v in validator.validations if v['is_valid']
    ]
    
    print(f"\n✅ Proceeding with {len(valid_files)} valid files")
    print()
    
    # Phase 1 Enhancement: Time Prediction
    print("=" * 80)
    print("⏱️  PHASE 1: PROCESSING TIME PREDICTION")
    print("=" * 80)
    predictor = ProcessingTimePredictor()
    time_prediction = predictor.predict_batch(valid_files, include_depth=True)
    
    print(f"Estimated total time: {time_prediction['total_predicted_hours']:.2f} hours")
    print(f"Estimated completion: {time_prediction['estimated_completion']}")
    print()
    
    # Match files to configurations
    scenes_to_process = []
    for tiff_path in valid_files:
        filename = tiff_path.stem
        for scene_key in SCENE_CONFIGS:
            if scene_key in filename:
                scenes_to_process.append((tiff_path, scene_key, SCENE_CONFIGS[scene_key]))
                break
    
    print(f"🎯 Processing {len(scenes_to_process)} scenes with enhanced pipeline")
    print()
    
    # Initialize Phase 1 tools
    tone_mapper = AdaptiveToneMapper()
    visualizer = HDRVisualizer(output_dir / "visualizations")
    compositor = AlphaCompositor()
    reporter = ProcessingReport(output_dir, "750 Picacho Lane - HDR Enhanced")
    
    # Process scenes
    results = []
    total_start = time.time()
    
    for i, (tiff_path, scene_name, config) in enumerate(scenes_to_process, 1):
        print(f"\n[{i}/{len(scenes_to_process)}]")
        try:
            result = process_scene_hdr_enhanced(
                tiff_path, output_dir, scene_name, config, device,
                tone_mapper, visualizer, compositor
            )
            results.append(result)
            
            # Add to report
            reporter.add_result(
                scene_name=result['scene_name'],
                input_file=result['input_file'],
                output_files=result['output_files'],
                processing_time_sec=result['processing_time_sec'],
                metrics=result['metrics'],
                tone_mapping_stats=result['tone_mapping'],
                depth_stats=result['depth']
            )
            
        except Exception as e:
            print(f"\n❌ ERROR processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Processed: {len(results)}/{len(scenes_to_process)} scenes")
    if results:
        print(f"Throughput: {len(results) / (total_time/3600):.1f} images/hour")
    print()
    
    # Phase 1 Enhancement: Comprehensive Reports
    print("=" * 80)
    print("📄 PHASE 1: GENERATING ENHANCED REPORTS")
    print("=" * 80)
    
    report_paths = reporter.finalize(include_thumbnails=True)
    
    # Generate client summary
    client_summary = create_client_deliverable_summary(output_dir, "750 Picacho Lane", results)
    print(f"  ✓ Client summary: {client_summary.name}")
    
    print()
    print(f"📦 All deliverables in: {output_dir}")
    print(f"   ├── masters/           (16-bit TIFF masters)")
    print(f"   ├── web/               (98% JPEG web-optimized)")
    print(f"   ├── depth/             (Depth maps)")
    print(f"   ├── thumbnails/        (1200px thumbnails)")
    print(f"   ├── visualizations/    (HDR analysis charts)")
    print(f"   └── alpha_variants/    (Alpha channel variants)")
    print()
    print("🎯 Ready for client delivery!")
    print("=" * 80)


if __name__ == "__main__":
    main()
