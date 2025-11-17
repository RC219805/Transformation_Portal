#!/usr/bin/env python3
"""
Luxury TIFF Processor for UHNW Real Estate Deliverables
Montecito Shores Interior Processing

Professional image enhancement without depth processing for maximum reliability.
"""

import sys
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def auto_white_balance(img):
    """
    Apply automatic white balance using gray world assumption.
    Adjusts color channels to neutralize color casts.
    """
    img_array = np.array(img, dtype=np.float32)

    # Calculate average for each channel
    avg_r = np.mean(img_array[:, :, 0])
    avg_g = np.mean(img_array[:, :, 1])
    avg_b = np.mean(img_array[:, :, 2])

    # Gray world assumption - aim for equal averages
    gray = (avg_r + avg_g + avg_b) / 3

    # Calculate scaling factors
    scale_r = gray / avg_r if avg_r > 0 else 1.0
    scale_g = gray / avg_g if avg_g > 0 else 1.0
    scale_b = gray / avg_b if avg_b > 0 else 1.0

    # Apply scaling with limits to prevent over-correction
    scale_r = np.clip(scale_r, 0.8, 1.2)
    scale_g = np.clip(scale_g, 0.8, 1.2)
    scale_b = np.clip(scale_b, 0.8, 1.2)

    # Apply white balance
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * scale_r, 0, 255)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * scale_g, 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * scale_b, 0, 255)

    return Image.fromarray(img_array.astype(np.uint8))

def enhance_luxury_interior(image_path, output_dir, preset='signature', auto_wb=False):
    """
    Process high-resolution TIFF for luxury real estate delivery.

    Enhancements:
    - Clarity and sharpness for architectural details
    - Subtle contrast enhancement
    - Color saturation boost
    - Preserve 16-bit depth when possible
    """
    print(f"\nProcessing: {image_path.name}")

    # Load image
    img = Image.open(image_path)
    print(f"  Dimensions: {img.size}, Mode: {img.mode}")

    # Convert to RGB if needed
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')

    # Apply auto white balance if requested
    if auto_wb:
        print(f"  Applying auto white balance...")
        img = auto_white_balance(img)

    # Enhancement parameters for luxury interiors
    params = {
        'signature': {
            'contrast': 1.12,
            'brightness': 1.02,
            'saturation': 1.08,
            'sharpness': 1.25,
        },
        'natural': {
            'contrast': 1.08,
            'brightness': 1.01,
            'saturation': 1.05,
            'sharpness': 1.15,
        },
        'dramatic': {
            'contrast': 1.18,
            'brightness': 0.98,
            'saturation': 1.12,
            'sharpness': 1.35,
        },
        'seaview': {
            'contrast': 1.15,      # Enhanced clarity for architectural details
            'brightness': 1.04,    # Slight lift to brighten darker interiors
            'saturation': 0.98,    # Reduce saturation to counteract warm cast
            'sharpness': 1.30,     # Strong sharpness for high-res JPEGs
            'warmth_reduction': True,  # Flag for targeted warm cast reduction
        }
    }

    p = params.get(preset, params['signature'])

    # Apply enhancements
    print(f"  Applying '{preset}' preset...")

    # Seaview-specific: Reduce warm cast before other enhancements
    if p.get('warmth_reduction', False):
        print(f"  Reducing warm cast...")
        img_array = np.array(img, dtype=np.float32)
        # Reduce red channel by 5%, increase blue by 5%
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.95, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.08, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))

    # Contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(p['contrast'])

    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(p['brightness'])

    # Color saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(p['saturation'])

    # Sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(p['sharpness'])

    # Optional: Slight unsharp mask for clarity
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))

    # Save with maximum quality
    wb_suffix = '_awb' if auto_wb else ''
    output_path = output_dir / f"{image_path.stem}_{preset}{wb_suffix}_enhanced.tif"

    # Try to preserve 16-bit if tifffile is available
    try:
        import tifffile
        img_array = np.array(img, dtype=np.uint16 if img.mode == 'I;16' else np.uint8)
        tifffile.imwrite(output_path, img_array, compression='adobe_deflate')
        print(f"  Saved (16-bit): {output_path.name}")
    except ImportError:
        # Fallback to PIL
        img.save(output_path, 'TIFF', compression='tiff_adobe_deflate', quality=100)
        print(f"  Saved (PIL): {output_path.name}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process luxury interior TIFFs for UHNW clients')
    parser.add_argument('input_dir', type=str, help='Input directory with TIFF files')
    parser.add_argument('--output', '-o', type=str, help='Output directory (default: auto-generated)')
    parser.add_argument('--preset', '-p', choices=['signature', 'natural', 'dramatic', 'seaview'],
                        default='signature', help='Enhancement preset')
    parser.add_argument('--pattern', default='*.tif', help='File pattern to process')
    parser.add_argument('--auto-wb', action='store_true',
                        help='Apply automatic white balance (gray world algorithm)')

    args = parser.parse_args()

    # Setup directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output_images') / f'montecito_shores_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    images = list(input_dir.glob(args.pattern))
    # Filter out macOS resource fork files
    images = [img for img in images if not img.name.startswith('._')]

    if not images:
        print(f"No images found matching '{args.pattern}' in {input_dir}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"LUXURY TIFF PROCESSOR - UHNW DELIVERABLES")
    print(f"{'='*70}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Preset: {args.preset}")
    print(f"Images found: {len(images)}")
    print(f"{'='*70}\n")

    # Process images
    processed = []
    failed = []

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}]", end=' ')
        try:
            output_path = enhance_luxury_interior(img_path, output_dir, args.preset, args.auto_wb)
            processed.append(output_path)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append((img_path, str(e)))

    # Summary
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully processed: {len(processed)}/{len(images)}")
    print(f"Output directory: {output_dir}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for path, error in failed:
            print(f"  - {path.name}: {error}")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
