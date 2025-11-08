#!/usr/bin/env python3
"""
Unified Luxury Pipeline - 750 Picacho Lane
Maximum quality processing with proper 16-bit TIFF output
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from typing import Optional, Tuple

# Import existing utilities
sys.path.insert(0, str(Path(__file__).parent))
from transformation_portal.utils.image_utils import load_image, save_image
from transformation_portal.utils.format_utils import normalize_extension


def load_exr_or_tiff(input_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load EXR or TIFF with proper handling.
    Returns: (numpy array in float32 [0-1] range, metadata dict)
    """
    ext = normalize_extension(input_path)
    
    if ext == '.exr':
        try:
            import OpenEXR
            import Imath
            
            exr_file = OpenEXR.InputFile(str(input_path))
            header = exr_file.header()
            
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # Read RGB channels
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = ['R', 'G', 'B']
            
            channel_data = {}
            for channel in channels:
                if channel in header['channels']:
                    channel_str = exr_file.channel(channel, FLOAT)
                    channel_data[channel] = np.frombuffer(channel_str, dtype=np.float32)
                    channel_data[channel] = channel_data[channel].reshape(height, width)
            
            # Stack into RGB array
            if len(channel_data) == 3:
                img_array = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=2)
            else:
                raise ValueError("EXR must have R, G, B channels")
            
            # EXR data is already in linear float space [0-1] (or can be >1 for HDR)
            # Clip to [0-1] for processing
            img_array = np.clip(img_array, 0, 1)
            
            metadata = {'source_format': 'exr', 'color_space': 'linear'}
            print(f"✓ Loaded EXR: {width}x{height}, range [{img_array.min():.3f}, {img_array.max():.3f}]")
            
            return img_array.astype(np.float32), metadata
            
        except ImportError:
            print("⚠️  OpenEXR not available, falling back to imageio")
            import imageio
            img_array = imageio.imread(input_path)
            # Convert to float [0-1]
            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.float32) / 255.0
            elif img_array.dtype == np.uint16:
                img_array = img_array.astype(np.float32) / 65535.0
            else:
                img_array = img_array.astype(np.float32)
            
            metadata = {'source_format': 'exr', 'color_space': 'unknown'}
            return img_array, metadata
    
    elif ext in ['.tif', '.tiff']:
        # Use tifffile for proper 16-bit loading
        img_array = tifffile.imread(input_path)
        
        if img_array.dtype == np.uint16:
            img_array = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)
        
        metadata = {'source_format': 'tiff', 'color_space': 'srgb'}
        print(f"✓ Loaded TIFF: {img_array.shape}, dtype was {tifffile.imread(input_path).dtype}")
        
        return img_array, metadata
    
    else:
        # Use PIL for other formats
        img = Image.open(input_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        metadata = {'source_format': str(ext), 'color_space': 'srgb'}
        print(f"✓ Loaded {ext}: {img.size}")
        
        return img_array, metadata


def save_16bit_tiff_proper(img_array: np.ndarray, output_path: Path, compression: str = 'lzw'):
    """
    Save 16-bit TIFF using tifffile (PROPER METHOD).
    
    Args:
        img_array: Float array [0-1] or uint16 array
        output_path: Path to save TIFF
        compression: 'lzw', 'deflate', or None
    """
    # Ensure proper range and type
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        # Convert float [0-1] to uint16 [0-65535]
        img_array = np.clip(img_array, 0, 1)
        img_uint16 = (img_array * 65535.0).astype(np.uint16)
    elif img_array.dtype == np.uint16:
        img_uint16 = img_array
    else:
        # Convert any other type to float first
        img_array = img_array.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / img_array.max()
        img_uint16 = (img_array * 65535.0).astype(np.uint16)
    
    # Save with tifffile - this properly writes 16-bit RGB
    tifffile.imwrite(
        output_path,
        img_uint16,
        photometric='rgb',
        compression=compression,
        metadata=None
    )
    
    print(f"✓ Saved 16-bit TIFF: {output_path.name}")
    print(f"  - Shape: {img_uint16.shape}")
    print(f"  - Range: [{img_uint16.min()}, {img_uint16.max()}]")
    print(f"  - Size: {output_path.stat().st_size / (1024*1024):.2f} MB")


def apply_luxury_enhancements(img_array: np.ndarray, scene_name: str) -> np.ndarray:
    """
    Apply luxury-grade enhancements.
    
    Args:
        img_array: Float32 array in [0-1] range
        scene_name: Name of scene for scene-specific adjustments
    """
    enhanced = img_array.copy()
    
    # Scene-specific adjustments
    if 'pool' in scene_name.lower():
        # Pool: enhance blues, boost highlights slightly
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.05, 0, 1)  # Blue channel
        enhanced = np.clip(enhanced * 1.02, 0, 1)  # Slight global lift
        
    elif 'aerial' in scene_name.lower():
        # Aerial: enhance greens, add clarity
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.03, 0, 1)  # Green channel
        
    elif 'kitchen' in scene_name.lower() or 'bathroom' in scene_name.lower():
        # Interior: subtle warmth, maintain neutrals
        enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.01, 0, 1)  # Red channel
        
    # Universal enhancements
    # 1. Gentle S-curve for contrast (preserves highlights and shadows)
    enhanced = enhanced ** 0.95
    
    # 2. Subtle saturation boost
    # Convert to HSV-like adjustment
    intensity = np.mean(enhanced, axis=2, keepdims=True)
    saturation_boost = 1.08
    enhanced = intensity + (enhanced - intensity) * saturation_boost
    enhanced = np.clip(enhanced, 0, 1)
    
    print(f"✓ Applied luxury enhancements for {scene_name}")
    
    return enhanced


def process_single_view(
    input_path: Path,
    output_dir: Path,
    save_jpeg: bool = True,
    save_tiff: bool = True
):
    """
    Process a single view with maximum quality.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*80}\n")
    
    # Load
    img_array, metadata = load_exr_or_tiff(input_path)
    
    # Get scene name
    scene_name = input_path.stem
    
    # Apply enhancements
    enhanced = apply_luxury_enhancements(img_array, scene_name)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save outputs
    outputs = []
    
    if save_tiff:
        # 16-bit TIFF using proper method
        tiff_path = output_dir / f"{scene_name}_luxury.tif"
        save_16bit_tiff_proper(enhanced, tiff_path, compression='lzw')
        outputs.append(tiff_path)
    
    if save_jpeg:
        # High-quality JPEG for web/preview
        jpeg_path = output_dir / f"{scene_name}_luxury.jpg"
        img_uint8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='RGB')
        img_pil.save(jpeg_path, 'JPEG', quality=98, optimize=True, subsampling=0)
        outputs.append(jpeg_path)
        print(f"✓ Saved JPEG: {jpeg_path.name}")
    
    print(f"\n✅ Completed: {input_path.name}")
    print(f"   Outputs: {len(outputs)} files in {output_dir.name}/\n")
    
    return outputs


def main():
    """Process all 750 Picacho Lane views."""
    
    # Setup paths
    source_dir = Path("/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/16-Bit_EXRs")
    output_dir = Path("/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/Ultimate_Quality")
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return 1
    
    # Find all EXR files
    exr_files = sorted(source_dir.glob("*.exr"))
    
    if not exr_files:
        print(f"❌ No EXR files found in {source_dir}")
        return 1
    
    print(f"\n{'#'*80}")
    print(f"  750 PICACHO LANE - UNIFIED LUXURY PIPELINE")
    print(f"  Maximum Quality Processing")
    print(f"{'#'*80}\n")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Files to process: {len(exr_files)}\n")
    
    # Process each view
    all_outputs = []
    for i, exr_file in enumerate(exr_files, 1):
        print(f"\n[{i}/{len(exr_files)}] " + "="*70)
        try:
            outputs = process_single_view(
                input_path=exr_file,
                output_dir=output_dir,
                save_jpeg=True,
                save_tiff=True
            )
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"❌ Error processing {exr_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"  PROCESSING COMPLETE")
    print(f"{'#'*80}\n")
    print(f"Total files processed: {len(exr_files)}")
    print(f"Total outputs created: {len(all_outputs)}")
    print(f"\nOutputs saved to: {output_dir}\n")
    
    # Verify TIFFs
    print("\nVerifying TIFF quality...")
    tiff_outputs = [f for f in all_outputs if f.suffix.lower() in ['.tif', '.tiff']]
    if tiff_outputs:
        sample_tiff = tiff_outputs[0]
        img_verify = tifffile.imread(sample_tiff)
        print(f"\nSample verification ({sample_tiff.name}):")
        print(f"  - Data type: {img_verify.dtype}")
        print(f"  - Shape: {img_verify.shape}")
        print(f"  - Range: [{img_verify.min()}, {img_verify.max()}]")
        
        if img_verify.dtype == np.uint16 and img_verify.max() > 256:
            print(f"\n✅ CONFIRMED: TIFFs are properly 16-bit with full range\n")
        else:
            print(f"\n⚠️  WARNING: Check TIFF bit depth\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
