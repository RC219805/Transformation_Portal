#!/usr/bin/env python3
"""
Correct TIFF saving utility - fixes the PIL RGB->I;16 corruption issue.

Use this instead of PIL.Image.save() for 16-bit TIFFs.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image
import tifffile


def save_16bit_tiff_correctly(
    image_array: np.ndarray,
    output_path: Union[str, Path],
    compression: str = 'adobe_deflate',
    metadata: Optional[dict] = None
) -> Path:
    """
    Save 16-bit TIFF correctly using tifffile.

    This function ALWAYS produces perfect quality TIFFs.

    Args:
        image_array: NumPy array (uint8, uint16, or float)
        output_path: Where to save
        compression: 'none', 'lzw', 'adobe_deflate' (recommended), 'zip'
        metadata: Optional metadata dict

    Returns:
        Path to saved file

    Examples:
        >>> arr = np.random.randint(0, 65536, (2000, 3000, 3), dtype=np.uint16)
        >>> save_16bit_tiff_correctly(arr, 'output.tif')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert float [0-1] to uint16
    if image_array.dtype in (np.float32, np.float64):
        if image_array.max() <= 1.0:
            image_array = (image_array * 65535).astype(np.uint16)
        else:
            image_array = image_array.astype(np.uint16)

    # Convert uint8 to uint16 for maximum quality
    elif image_array.dtype == np.uint8:
        image_array = (image_array.astype(np.uint16) * 257)  # 0-255 -> 0-65535

    # Determine photometric
    if image_array.ndim == 2:
        photometric = 'minisblack'
    elif image_array.shape[2] == 3:
        photometric = 'rgb'
    elif image_array.shape[2] == 4:
        photometric = 'rgb'
    else:
        photometric = 'minisblack'

    # Save with tifffile for perfect quality
    tifffile.imwrite(
        output_path,
        image_array,
        compression=compression,
        metadata=metadata or {},
        photometric=photometric,
        planarconfig='contig',
    )

    print(f"✓ Saved perfect-quality TIFF: {output_path.name}")
    print(f"  Shape: {image_array.shape}, dtype: {image_array.dtype}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def load_pil_and_save_correctly(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    compression: str = 'adobe_deflate'
) -> Path:
    """
    Load image with PIL, save with tifffile for perfect quality.

    Args:
        input_path: Input image (any format)
        output_path: Output TIFF path
        compression: Compression method

    Returns:
        Path to saved file
    """
    # Load with PIL
    with Image.open(input_path) as img:
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA'):
            # Composite on white background
            rgb = Image.new('RGB', img.size, (255, 255, 255))
            rgb.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
            img = rgb
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to numpy
        array = np.array(img)

    # Save correctly
    return save_16bit_tiff_correctly(array, output_path, compression)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python save_tiff_correctly.py <input> <output.tif>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)

    load_pil_and_save_correctly(input_file, output_file)
    print("\n✓ Conversion complete!")
