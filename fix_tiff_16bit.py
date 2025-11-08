#!/usr/bin/env python3
"""
Fix TIFF 16-bit Saving Issue

Proper method to save 16-bit TIFFs with full quality preservation.
"""

import numpy as np
from PIL import Image
import tifffile
from pathlib import Path


def save_16bit_tiff_pil(image_array: np.ndarray, output_path: Path, compression='lzw'):
    """
    Save 16-bit TIFF using PIL (proper method).
    
    Args:
        image_array: numpy array in [0, 1] float range or [0, 255] uint8
        output_path: Path to save TIFF
        compression: 'lzw', 'tiff_deflate', or None
    """
    # Convert to 16-bit range
    if image_array.dtype == np.uint8:
        # Convert uint8 [0, 255] to uint16 [0, 65535]
        array_16bit = (image_array.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    elif image_array.dtype in (np.float32, np.float64):
        # Convert float [0, 1] to uint16 [0, 65535]
        array_16bit = (np.clip(image_array, 0, 1) * 65535.0).astype(np.uint16)
    elif image_array.dtype == np.uint16:
        # Already 16-bit
        array_16bit = image_array
    else:
        raise ValueError(f"Unsupported dtype: {image_array.dtype}")
    
    # PIL doesn't handle RGB uint16 directly - need to save per-channel or use mode I;16
    # Best approach: use tifffile for RGB 16-bit
    if array_16bit.ndim == 3 and array_16bit.shape[2] == 3:
        # RGB image - use tifffile (PIL struggles with RGB uint16)
        save_16bit_tiff_tifffile(array_16bit, output_path, compression)
    else:
        # Grayscale - PIL can handle this
        img = Image.fromarray(array_16bit, mode='I;16')
        img.save(output_path, compression=compression)


def save_16bit_tiff_tifffile(image_array: np.ndarray, output_path: Path, compression='lzw'):
    """
    Save 16-bit TIFF using tifffile (RECOMMENDED for RGB).
    
    This is the most reliable method for RGB 16-bit TIFFs.
    
    Args:
        image_array: numpy array in [0, 1] float range or [0, 255] uint8 or uint16
        output_path: Path to save TIFF
        compression: 'lzw', 'deflate', 'zstd', or None
    """
    # Convert to 16-bit range
    if image_array.dtype == np.uint8:
        array_16bit = (image_array.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    elif image_array.dtype in (np.float32, np.float64):
        array_16bit = (np.clip(image_array, 0, 1) * 65535.0).astype(np.uint16)
    elif image_array.dtype == np.uint16:
        array_16bit = image_array
    else:
        raise ValueError(f"Unsupported dtype: {image_array.dtype}")
    
    # Map compression parameter
    compress_map = {
        'lzw': 'lzw',
        'tiff_deflate': 'deflate',
        'deflate': 'deflate',
        'zstd': 'zstd',
        None: None
    }
    compress = compress_map.get(compression, compression)
    
    # Save with tifffile
    tifffile.imwrite(
        output_path,
        array_16bit,
        photometric='rgb' if array_16bit.ndim == 3 else 'minisblack',
        compression=compress,
        metadata={'axes': 'YXC' if array_16bit.ndim == 3 else 'YX'}
    )
    
    print(f"✅ Saved 16-bit TIFF: {output_path.name}")
    print(f"   Shape: {array_16bit.shape}, dtype: {array_16bit.dtype}")
    print(f"   Range: [{array_16bit.min()}, {array_16bit.max()}]")


def convert_8bit_to_16bit_tiff(input_path: Path, output_path: Path = None):
    """
    Convert existing 8-bit TIFF to proper 16-bit TIFF.
    
    Args:
        input_path: Path to 8-bit TIFF
        output_path: Path for 16-bit TIFF (defaults to input_path with _16bit suffix)
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_16bit{input_path.suffix}"
    
    # Load image
    img = Image.open(input_path)
    array = np.array(img)
    
    print(f"Converting {input_path.name}...")
    print(f"  Input: {array.dtype}, shape: {array.shape}, range: [{array.min()}, {array.max()}]")
    
    # Save as 16-bit
    save_16bit_tiff_tifffile(array, output_path)
    
    # Verify
    array_16 = tifffile.imread(output_path)
    print(f"  Output: {array_16.dtype}, shape: {array_16.shape}, range: [{array_16.min()}, {array_16.max()}]")
    
    return output_path


# Example usage
if __name__ == "__main__":
    # Test with a sample array
    test_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    test_path = Path("/tmp/test_16bit.tif")
    save_16bit_tiff_tifffile(test_array, test_path)
    
    # Verify
    loaded = tifffile.imread(test_path)
    print(f"\nVerification:")
    print(f"  Saved dtype: {loaded.dtype}")
    print(f"  Saved shape: {loaded.shape}")
    print(f"  Saved range: [{loaded.min()}, {loaded.max()}]")
    
    if loaded.dtype == np.uint16:
        print("✅ SUCCESS: Proper 16-bit TIFF!")
    else:
        print("❌ FAILED: Still 8-bit")
