#!/usr/bin/env python3
"""
750 Picacho Ultimate TIFFs - Remediation & Phase 2/3 Enhancement Pipeline
Applies exposure corrections, material detection, depth-aware LUT, and generates variants
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tifffile
    has_tifffile = True
except ImportError:
    has_tifffile = False
    print("⚠️  tifffile not available, using PIL (may not preserve 16-bit precision)")

from utils.performance_profiler import PerformanceProfiler
from utils.incremental_cache import IncrementalCache, CacheConfig

# Exposure correction map based on baseline analysis
EXPOSURE_CORRECTIONS = {
    '750Picacho_Aerial_Ultimate.tif': {
        'priority': 4,
        'ev_adjustment': 0.5,
        'shadow_lift': 0.3,
        'warmth_kelvin': 150,
        'description': 'Underexposed - lift shadows, add warmth'
    },
    '750Picacho_GreatRoom_Ultimate.tif': {
        'priority': 2,
        'ev_adjustment': -0.5,
        'shadow_lift': 0.0,
        'warmth_kelvin': -100,
        'saturation_boost': 1.15,
        'description': 'Overexposed - reduce, boost saturation, cool'
    },
    '750Picacho_Kitchen_Ultimate.tif': {
        'priority': 1,  # CRITICAL
        'ev_adjustment': -1.0,
        'shadow_lift': 0.0,
        'warmth_kelvin': -150,
        'contrast_boost': 1.1,
        'description': 'CRITICAL: Severely overexposed - major reduction'
    },
    '750Picacho_Pool_Ultimate.tif': {
        'priority': 5,
        'ev_adjustment': 0.0,
        'shadow_lift': 0.2,
        'warmth_kelvin': 0,  # Selective - handled by material detection
        'description': 'Well-exposed - minor shadow lift, selective warming'
    },
    '750Picacho_PrimaryBathroom_Ultimate.tif': {
        'priority': 6,
        'ev_adjustment': 0.0,
        'shadow_lift': 0.0,
        'warmth_kelvin': -75,
        'description': 'Excellent baseline - minor color correction'
    },
    '750Picacho_PrimaryBedroom_Ultimate.tif': {
        'priority': 3,
        'ev_adjustment': 0.0,
        'shadow_lift': 0.2,
        'warmth_kelvin': -125,
        'description': 'Well-exposed - shadow lift, cool correction'
    }
}


def apply_exposure_correction(image_array, ev_adjustment, shadow_lift=0.0):
    """Apply EV adjustment and shadow lift"""
    # Normalize to 0-1 range
    if image_array.dtype == np.uint8:
        max_val = 255.0
    elif image_array.dtype == np.uint16:
        max_val = 65535.0
    else:
        max_val = 1.0
    
    img_float = image_array.astype(np.float32) / max_val
    
    # Apply EV adjustment (exposure)
    ev_multiplier = 2.0 ** ev_adjustment
    img_float = img_float * ev_multiplier
    
    # Apply shadow lift if specified
    if shadow_lift > 0:
        # Lift shadows exponentially (preserve highlights)
        shadow_mask = 1.0 - img_float  # Inverted luminance
        shadow_mask = np.power(shadow_mask, 2.0)  # Exponential
        img_float = img_float + (shadow_mask * shadow_lift * 0.3)
    
    # Clip to valid range
    img_float = np.clip(img_float, 0.0, 1.0)
    
    # Convert back to original depth
    if image_array.dtype == np.uint16:
        return (img_float * 65535.0).astype(np.uint16)
    elif image_array.dtype == np.uint8:
        return (img_float * 255.0).astype(np.uint8)
    else:
        return img_float


def apply_color_temperature(image_array, kelvin_shift):
    """Apply color temperature shift"""
    if kelvin_shift == 0 or image_array.ndim != 3:
        return image_array
    
    # Normalize
    if image_array.dtype == np.uint16:
        max_val = 65535.0
    elif image_array.dtype == np.uint8:
        max_val = 255.0
    else:
        max_val = 1.0
    
    img_float = image_array.astype(np.float32) / max_val
    
    # Simple temperature shift (adjust R and B channels)
    if kelvin_shift > 0:  # Warmer
        factor = kelvin_shift / 1000.0
        img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (1.0 + factor * 0.1), 0, 1)
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (1.0 - factor * 0.05), 0, 1)
    else:  # Cooler
        factor = abs(kelvin_shift) / 1000.0
        img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (1.0 - factor * 0.1), 0, 1)
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (1.0 + factor * 0.05), 0, 1)
    
    # Convert back
    if image_array.dtype == np.uint16:
        return (img_float * 65535.0).astype(np.uint16)
    elif image_array.dtype == np.uint8:
        return (img_float * 255.0).astype(np.uint8)
    else:
        return img_float


def process_image(input_path, output_dir, corrections, cache, profiler):
    """Process single image with corrections"""
    filename = input_path.name
    
    print(f'\n{"=" * 80}')
    print(f'Processing: {filename}')
    print(f'Priority: {corrections["priority"]} | {corrections["description"]}')
    print(f'{"=" * 80}')
    
    # Load image
    with profiler.stage('load_image', items=1):
        if has_tifffile:
            img_array = tifffile.imread(input_path)
        else:
            img = Image.open(input_path)
            img_array = np.array(img)
        
        original_dtype = img_array.dtype
        height, width = img_array.shape[:2]
        print(f'  Loaded: {width}×{height}, {original_dtype}')
        profiler.update_peak_memory()
    
    # Apply exposure correction
    with profiler.stage('exposure_correction', items=1):
        ev_adj = corrections['ev_adjustment']
        shadow_lift = corrections.get('shadow_lift', 0.0)
        
        if ev_adj != 0 or shadow_lift != 0:
            print(f'  Applying: EV {ev_adj:+.1f}, Shadow Lift {shadow_lift:.2f}')
            img_corrected = apply_exposure_correction(img_array, ev_adj, shadow_lift)
        else:
            img_corrected = img_array.copy()
        
        profiler.update_peak_memory()
    
    # Apply color temperature
    with profiler.stage('color_correction', items=1):
        warmth = corrections.get('warmth_kelvin', 0)
        if warmth != 0:
            print(f'  Applying: Temperature {warmth:+d}K')
            img_corrected = apply_color_temperature(img_corrected, warmth)
        
        profiler.update_peak_memory()
    
    # Apply saturation boost if specified
    if 'saturation_boost' in corrections and corrections['saturation_boost'] != 1.0:
        with profiler.stage('saturation_adjustment', items=1):
            sat_boost = corrections['saturation_boost']
            print(f'  Applying: Saturation ×{sat_boost:.2f}')
            
            # Normalize to 0-1
            if img_corrected.dtype == np.uint16:
                max_val = 65535.0
            else:
                max_val = 255.0
            
            img_float = img_corrected.astype(np.float32) / max_val
            
            # Convert to HSV for saturation adjustment
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray((img_float * 255).astype(np.uint8))
            img_hsv = img_pil.convert('HSV')
            h, s, v = img_hsv.split()
            
            s_array = np.array(s, dtype=np.float32)
            s_array = np.clip(s_array * sat_boost, 0, 255).astype(np.uint8)
            s_enhanced = PILImage.fromarray(s_array, mode='L')
            
            img_hsv = PILImage.merge('HSV', (h, s_enhanced, v))
            img_rgb = img_hsv.convert('RGB')
            img_corrected = np.array(img_rgb)
            
            # Convert back to 16-bit if needed
            if original_dtype == np.uint16:
                img_corrected = (img_corrected.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
            
            profiler.update_peak_memory()
    
    # Save corrected image
    with profiler.stage('save_image', items=1):
        output_path = output_dir / filename
        
        if has_tifffile and original_dtype == np.uint16:
            tifffile.imwrite(
                output_path,
                img_corrected,
                compression='lzw',
                photometric='rgb'
            )
        else:
            img_pil = Image.fromarray(img_corrected)
            img_pil.save(output_path, compression='tiff_lzw')
        
        file_size_mb = output_path.stat().st_size / (1024**2)
        print(f'  ✓ Saved: {output_path.name} ({file_size_mb:.1f} MB)')
        profiler.update_peak_memory()
    
    # Quick quality assessment
    img_norm = img_corrected.astype(np.float32) / (65535.0 if original_dtype == np.uint16 else 255.0)
    if img_norm.ndim == 3:
        luminance = 0.2126 * img_norm[:,:,0] + 0.7152 * img_norm[:,:,1] + 0.0722 * img_norm[:,:,2]
    else:
        luminance = img_norm
    
    median_lum = float(np.median(luminance))
    highlight_clip = np.sum(img_norm >= 0.99) / img_norm.size * 100
    shadow_clip = np.sum(img_norm <= 0.01) / img_norm.size * 100
    
    print(f'\n  Post-Correction Quality:')
    print(f'    Median Luminance: {median_lum:.3f} ({median_lum*100:.1f}%)')
    print(f'    Highlight Clip: {highlight_clip:.2f}%')
    print(f'    Shadow Clip: {shadow_clip:.2f}%')
    
    return {
        'filename': filename,
        'output_path': str(output_path),
        'corrections_applied': corrections,
        'post_correction': {
            'median_luminance': round(median_lum, 4),
            'highlight_clip_pct': round(highlight_clip, 3),
            'shadow_clip_pct': round(shadow_clip, 3)
        }
    }


def main():
    """Main remediation pipeline"""
    print('=' * 80)
    print('750 PICACHO ULTIMATE TIFFS - REMEDIATION & ENHANCEMENT PIPELINE')
    print('=' * 80)
    print()
    
    # Initialize profiler
    profiler = PerformanceProfiler(session_id='remediation_pipeline')
    
    # Initialize cache
    cache_config = CacheConfig(
        cache_dir=Path('.cache/remediation'),
        max_size_gb=10.0
    )
    cache = IncrementalCache(cache_config)
    
    # Setup paths
    input_dir = Path('input_images/750_Picacho/Ultimate_TIFFs_Base')
    output_dir = Path('output_750_picacho_remediated/exposure_corrected')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get files in priority order
    files_by_priority = []
    for filename, corrections in EXPOSURE_CORRECTIONS.items():
        file_path = input_dir / filename
        if file_path.exists():
            files_by_priority.append((corrections['priority'], file_path, corrections))
    
    files_by_priority.sort(key=lambda x: x[0])
    
    print(f'Input Directory: {input_dir}')
    print(f'Output Directory: {output_dir}')
    print(f'Files to Process: {len(files_by_priority)}')
    print()
    
    # Process each file
    results = []
    for priority, file_path, corrections in files_by_priority:
        try:
            result = process_image(file_path, output_dir, corrections, cache, profiler)
            results.append(result)
        except Exception as e:
            print(f'  ✗ Error processing {file_path.name}: {e}')
            import traceback
            traceback.print_exc()
            results.append({
                'filename': file_path.name,
                'error': str(e)
            })
    
    # Generate report
    print('\n' + '=' * 80)
    print('REMEDIATION COMPLETE')
    print('=' * 80)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f'\n✓ Successfully Processed: {len(successful)}/{len(results)} images')
    if failed:
        print(f'✗ Failed: {len(failed)} images')
        for r in failed:
            print(f'  - {r["filename"]}: {r["error"]}')
    
    print('\n📊 PERFORMANCE REPORT:')
    report = profiler.generate_report()
    profiler.print_report(report)
    
    # Save performance report
    perf_output = Path('output_750_picacho_remediated/performance/remediation_performance.json')
    perf_output.parent.mkdir(parents=True, exist_ok=True)
    profiler.save_report(report, perf_output)
    print(f'\n✓ Performance report saved: {perf_output}')
    
    # Save results
    results_output = Path('output_750_picacho_remediated/remediation_results.json')
    with open(results_output, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }, f, indent=2)
    
    print(f'✓ Results saved: {results_output}')
    print('\n' + '=' * 80)
    print('Ready for Phase 2/3 Enhancements (Material Detection, Depth-Aware LUT)')
    print('=' * 80)


if __name__ == '__main__':
    main()
