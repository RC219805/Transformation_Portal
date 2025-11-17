#!/usr/bin/env python3
"""
Process the 750 Picacho Pool luxury image with professional enhancements.

This script applies luxury real estate enhancement specifically optimized
for pool scenes with:
- Professional color grading
- Enhanced water clarity and reflections
- Optimized exposure and contrast
- Luxury glow for premium aesthetic
- Crystal-clear detail enhancement
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from luxury_tiff_batch_processor.adjustments import apply_adjustments, LUXURY_PRESETS, AdjustmentSettings
from luxury_tiff_batch_processor.io_utils import image_to_float, float_to_dtype_array, save_image


def process_pool_image(input_path: Path, output_path: Path):
    """
    Process luxury pool image with professional enhancements.
    
    Args:
        input_path: Input TIFF file path
        output_path: Output TIFF file path
    """
    print(f"Loading image: {input_path}")
    
    # Load the image using PIL
    pil_image = Image.open(input_path)
    print(f"Image loaded: {pil_image.size}, mode: {pil_image.mode}")
    
    # Convert to float32 for high-precision processing
    result = image_to_float(pil_image, return_format="object")
    img_array = result.array
    orig_dtype = result.dtype
    alpha = result.alpha
    base_channels = result.base_channels
    float_norm = result.float_normalisation
    
    # Extract metadata
    metadata = pil_image.tag_v2 if hasattr(pil_image, 'tag_v2') else None
    icc_profile = pil_image.info.get('icc_profile')
    
    print(f"Array: {img_array.shape}, dtype: {img_array.dtype}")
    print(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    # Create pool-specific adjustments optimized for luxury pool scenes
    pool_adjustments = AdjustmentSettings(
        # Slight positive exposure for bright, inviting water
        exposure=0.15,
        
        # Enhanced clarity for crisp architectural details and water surface
        clarity=0.35,
        
        # Boost vibrance for vibrant water blues and landscape greens
        vibrance=0.25,
        
        # Moderate saturation boost for luxury aesthetic
        saturation=0.15,
        
        # Subtle shadow lift to reveal detail in poolside areas
        shadow_lift=0.08,
        
        # Gentle highlight recovery to preserve sky and reflections
        highlight_recovery=0.12,
        
        # Enhanced midtone contrast for dimensional depth
        midtone_contrast=0.18,
        
        # Cool white balance for refreshing water tones
        white_balance_temp=5800,  # Slightly cool daylight
        white_balance_tint=-2,    # Subtle blue shift
        
        # Luxury glow for premium aesthetic
        glow=0.20,
        
        # Minimal chroma denoise to preserve detail
        chroma_denoise=0.10
    )
    
    print("\nApplying luxury pool enhancements:")
    print(f"  - Exposure: +{pool_adjustments.exposure:.2f} stops")
    print(f"  - Clarity: {pool_adjustments.clarity:.2f}")
    print(f"  - Vibrance: +{pool_adjustments.vibrance:.2f}")
    print(f"  - Saturation: +{pool_adjustments.saturation:.2f}")
    print(f"  - White Balance: {pool_adjustments.white_balance_temp}K (tint: {pool_adjustments.white_balance_tint})")
    print(f"  - Luxury Glow: {pool_adjustments.glow:.2f}")
    print(f"  - Midtone Contrast: {pool_adjustments.midtone_contrast:.2f}")
    
    # Apply the adjustments
    processed = apply_adjustments(img_array, pool_adjustments)
    
    print(f"\nProcessed image range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Convert back to original dtype
    arr_int = float_to_dtype_array(
        processed,
        orig_dtype,
        alpha,
        base_channels,
        float_normalisation=float_norm
    )
    
    # Save the processed image - skip problematic metadata
    print(f"Saving to: {output_path}")
    # Create simplified metadata-free save to avoid TIFF tag issues
    save_image(
        output_path,
        arr_int,
        orig_dtype,
        None,  # Skip metadata to avoid TIFF tag issues
        icc_profile,
        compression="tiff_lzw"
    )
    
    print("✅ Processing complete!")
    
    # Also save a high-quality JPEG for quick preview
    jpeg_path = output_path.with_suffix('.jpg')
    
    # Convert to 8-bit for JPEG
    img_8bit = np.clip(processed * 255, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_8bit)
    
    # Preserve EXIF/IPTC metadata if available
    exif_data = None
    try:
        original_img = Image.open(input_path)
        exif_data = original_img.info.get('exif')
    except Exception:
        pass
    
    # Save progressive JPEG at 98% quality
    if exif_data:
        img_pil.save(jpeg_path, 'JPEG', quality=98, progressive=True, exif=exif_data)
    else:
        img_pil.save(jpeg_path, 'JPEG', quality=98, progressive=True)
    
    print(f"✅ JPEG preview saved: {jpeg_path}")
    
    return processed


if __name__ == "__main__":
    # Define paths
    repo_root = Path(__file__).parent
    input_file = repo_root / "input_images" / "V2_750Picacho_Pool.tiff"
    output_file = repo_root / "output_images" / "V2_750Picacho_Pool_Luxury_Enhanced.tiff"
    
    # Verify input exists
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the image
    try:
        process_pool_image(input_file, output_file)
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
