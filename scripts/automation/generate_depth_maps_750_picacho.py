#!/usr/bin/env python3
"""
Enhanced Multi-Zone Depth Map Generator for 750 Picacho
Uses Depth Anything V2 Large model with MPS acceleration

Features:
- 16-bit TIFF support (preserves dynamic range)
- Multi-zone depth analysis (foreground, midground, background)
- Enhanced visualization with color-coded depth zones
- Batch processing with progress tracking
- MPS (Apple Silicon) GPU acceleration
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

# Color maps for visualization
DEPTH_COLORMAP = {
    'viridis': [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    'magma': [(0, 0, 4), (40, 11, 84), (119, 31, 109), (186, 54, 85), (252, 253, 191)],
    'turbo': [(48, 18, 59), (62, 73, 137), (68, 134, 194), (134, 190, 169), (253, 231, 37)],
}


def apply_colormap(depth: np.ndarray, colormap: str = 'turbo') -> np.ndarray:
    """Apply perceptually uniform colormap to depth map."""
    # Normalize to 0-1
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # Get colormap
    colors = DEPTH_COLORMAP.get(colormap, DEPTH_COLORMAP['turbo'])
    num_colors = len(colors)
    
    # Interpolate colors
    indices = depth_norm * (num_colors - 1)
    lower = np.floor(indices).astype(int)
    upper = np.ceil(indices).astype(int)
    frac = indices - lower
    
    # Ensure indices are in bounds
    lower = np.clip(lower, 0, num_colors - 1)
    upper = np.clip(upper, 0, num_colors - 1)
    
    # Get RGB values
    h, w = depth.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(3):  # RGB channels
        lower_colors = np.array([colors[idx][i] for idx in lower.flat]).reshape(h, w)
        upper_colors = np.array([colors[idx][i] for idx in upper.flat]).reshape(h, w)
        rgb[:, :, i] = (lower_colors * (1 - frac) + upper_colors * frac).astype(np.uint8)
    
    return rgb


def analyze_depth_zones(depth: np.ndarray, num_zones: int = 3) -> dict:
    """Analyze depth map into multiple zones (foreground, midground, background)."""
    # Compute percentiles for zone boundaries
    percentiles = np.linspace(0, 100, num_zones + 1)
    thresholds = np.percentile(depth, percentiles)
    
    zones = {}
    zone_names = ['foreground', 'midground', 'background'][:num_zones]
    
    for i, name in enumerate(zone_names):
        mask = (depth >= thresholds[i]) & (depth < thresholds[i + 1])
        zones[name] = {
            'mask': mask,
            'mean_depth': depth[mask].mean() if mask.any() else 0,
            'std_depth': depth[mask].std() if mask.any() else 0,
            'coverage': mask.sum() / mask.size * 100,
            'min_depth': thresholds[i],
            'max_depth': thresholds[i + 1],
        }
    
    return zones


def create_multi_zone_visualization(
    depth: np.ndarray,
    zones: dict,
    original_size: Tuple[int, int]
) -> Image.Image:
    """Create enhanced visualization with color-coded depth zones."""
    # Apply colormap
    colored_depth = apply_colormap(depth, 'turbo')
    
    # Overlay zone boundaries
    zone_overlay = colored_depth.copy()
    
    # Add zone highlights (subtle overlay)
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # Red, Green, Blue
    for idx, (name, zone_info) in enumerate(zones.items()):
        mask = zone_info['mask']
        if mask.any():
            # Create semi-transparent overlay
            overlay = zone_overlay.copy()
            overlay[mask] = (overlay[mask] * 0.7 + np.array(colors[idx]) * 0.3).astype(np.uint8)
            zone_overlay = overlay
    
    # Convert to PIL Image
    viz = Image.fromarray(zone_overlay)
    
    # Resize to original dimensions if needed
    if viz.size != original_size:
        viz = viz.resize(original_size, Image.Resampling.LANCZOS)
    
    return viz


def process_image(
    image_path: Path,
    output_dir: Path,
    depth_estimator,
    device: str = 'mps',
    save_raw: bool = True,
    save_visualization: bool = True,
    save_zones: bool = True,
) -> dict:
    """Process single image and generate depth maps."""
    print(f"\n{'=' * 70}")
    print(f"Processing: {image_path.name}")
    print(f"{'=' * 70}")
    
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        print(f"Image size: {original_size[0]}x{original_size[1]}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return {'error': str(e)}
    
    # Estimate depth
    print("Estimating depth with Depth Anything V2 Large...")
    start_time = time.time()
    
    try:
        result = depth_estimator(img)
        depth_map = result['depth']
        depth_array = np.array(depth_map)
        
        inference_time = time.time() - start_time
        print(f"✅ Depth estimation completed in {inference_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error during depth estimation: {e}")
        return {'error': str(e)}
    
    # Analyze depth zones
    print("\nAnalyzing depth zones...")
    zones = analyze_depth_zones(depth_array, num_zones=3)
    
    print("\n📊 Depth Zone Analysis:")
    for name, info in zones.items():
        print(f"  {name.capitalize()}:")
        print(f"    Coverage: {info['coverage']:.1f}%")
        print(f"    Mean depth: {info['mean_depth']:.3f}")
        print(f"    Std dev: {info['std_depth']:.3f}")
        print(f"    Range: [{info['min_depth']:.3f}, {info['max_depth']:.3f}]")
    
    # Create output filenames
    stem = image_path.stem
    results = {}
    
    # Save raw depth map (16-bit for precision)
    if save_raw:
        raw_path = output_dir / f"{stem}_depth_raw_16bit.tiff"
        # Normalize to 16-bit range
        depth_16bit = ((depth_array - depth_array.min()) / 
                       (depth_array.max() - depth_array.min()) * 65535).astype(np.uint16)
        Image.fromarray(depth_16bit).save(raw_path)
        print(f"\n✅ Saved raw depth map: {raw_path.name}")
        results['raw_depth'] = raw_path
    
    # Save visualization
    if save_visualization:
        viz_path = output_dir / f"{stem}_depth_visualization.png"
        viz = create_multi_zone_visualization(depth_array, zones, original_size)
        viz.save(viz_path, optimize=True)
        print(f"✅ Saved visualization: {viz_path.name}")
        results['visualization'] = viz_path
    
    # Save individual zone maps
    if save_zones:
        for zone_name, zone_info in zones.items():
            zone_path = output_dir / f"{stem}_depth_zone_{zone_name}.png"
            # Create zone mask visualization
            zone_mask = zone_info['mask'].astype(np.uint8) * 255
            zone_img = Image.fromarray(zone_mask)
            zone_img = zone_img.resize(original_size, Image.Resampling.NEAREST)
            zone_img.save(zone_path)
            print(f"✅ Saved {zone_name} zone: {zone_path.name}")
            results[f'zone_{zone_name}'] = zone_path
    
    # Save analysis metadata
    metadata_path = output_dir / f"{stem}_depth_analysis.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Depth Map Analysis for {image_path.name}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Image size: {original_size[0]}x{original_size[1]}\n")
        f.write(f"Inference time: {inference_time:.2f}s\n")
        f.write(f"Model: Depth Anything V2 Large\n")
        f.write(f"Device: {device.upper()}\n\n")
        f.write("Zone Analysis:\n")
        for name, info in zones.items():
            f.write(f"\n{name.capitalize()}:\n")
            f.write(f"  Coverage: {info['coverage']:.1f}%\n")
            f.write(f"  Mean depth: {info['mean_depth']:.3f}\n")
            f.write(f"  Std dev: {info['std_depth']:.3f}\n")
            f.write(f"  Range: [{info['min_depth']:.3f}, {info['max_depth']:.3f}]\n")
    
    print(f"✅ Saved metadata: {metadata_path.name}")
    results['metadata'] = metadata_path
    
    return results


def main():
    """Main entry point."""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "Enhanced Depth Map Generator - 750 Picacho" + " " * 20 + "║")
    print("║" + " " * 20 + "Depth Anything V2 Large + MPS Acceleration" + " " * 14 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Setup paths
    input_dir = Path("input_images/750_Picacho")
    output_dir = Path("output_750_Picacho_Depth_Maps")
    output_dir.mkdir(exist_ok=True)
    
    # Find TIFF files
    tiff_files = sorted(input_dir.glob("*.tif*"))
    
    if not tiff_files:
        print(f"❌ No TIFF files found in {input_dir}")
        return 1
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Found {len(tiff_files)} TIFF file(s)")
    
    # Check device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Using MPS (Apple Silicon GPU) acceleration")
    else:
        device = 'cpu'
        print("⚠️  MPS not available, using CPU")
    
    # Load model
    print("\n🔄 Loading Depth Anything V2 Large model...")
    try:
        depth_estimator = pipeline(
            task='depth-estimation',
            model='depth-anything/Depth-Anything-V2-Large-hf',
            device=device
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Process images
    results = []
    print("\n" + "=" * 70)
    print("Starting batch processing...")
    print("=" * 70)
    
    for img_path in tiff_files:
        result = process_image(
            img_path,
            output_dir,
            depth_estimator,
            device=device,
            save_raw=True,
            save_visualization=True,
            save_zones=True,
        )
        results.append({
            'input': img_path,
            'outputs': result
        })
    
    # Summary
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 30 + "PROCESSING COMPLETE!" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print(f"✅ Processed: {len(tiff_files)} image(s)")
    print(f"📁 Output directory: {output_dir}")
    print(f"\nGenerated files per image:")
    print(f"  • Raw depth map (16-bit TIFF)")
    print(f"  • Color visualization (PNG)")
    print(f"  • Zone masks (foreground/midground/background)")
    print(f"  • Analysis metadata (TXT)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
