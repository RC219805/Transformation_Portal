#!/usr/bin/env python3
"""
Utility to save 16-bit TIFF files correctly.
This fixes the PIL bug where 16-bit data is saved as 8-bit.
"""

import numpy as np
from pathlib import Path
from typing import Union
from PIL import Image


def save_16bit_tiff(
    image: Union[np.ndarray, Image.Image],
    output_path: Union[str, Path],
    compression: str = 'lzw',
    dpi: tuple = (300, 300),
    metadata: dict = None
) -> Path:
    """
    Save image as true 16-bit TIFF.

    Args:
        image: numpy array (float [0,1] or uint16) or PIL Image
        output_path: Where to save the file
        compression: 'lzw', 'deflate', or None
        dpi: Tuple of (x_dpi, y_dpi)
        metadata: Optional metadata dict

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)

    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure float32 in [0, 1] range
    if image.dtype == np.uint16:
        image_float = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        print("  ⚠️  Warning: Input is 8-bit, upconverting to 16-bit")
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.astype(np.float32)

    # Convert to 16-bit
    image_16bit = (np.clip(image_float, 0, 1) * 65535).astype(np.uint16)

    # Try tifffile first (best quality)
    try:
        import tifffile

        # Prepare metadata
        resolution_unit = 2  # inches
        x_resolution = (dpi[0], 1)
        y_resolution = (dpi[1], 1)

        tifffile.imwrite(
            output_path,
            image_16bit,
            compression=compression,
            photometric='rgb',
            resolution=(x_resolution, y_resolution),
            resolutionunit=resolution_unit,
            metadata=metadata or {}
        )

        file_size_mb = output_path.stat().st_size / (1024**2)
        print(f"✓ Saved 16-bit TIFF: {output_path.name} ({file_size_mb:.1f} MB)")
        return output_path

    except ImportError:
        print("  ⚠️  tifffile not available, using PIL (may lose 16-bit depth)")

        # Fallback to PIL - this will save as 8-bit!
        image_8bit = (np.clip(image_float, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(image_8bit)
        pil_img.save(output_path, format='TIFF', compression=f'tiff_{compression}', dpi=dpi)

        file_size_mb = output_path.stat().st_size / (1024**2)
        print(f"  Saved 8-bit TIFF: {output_path.name} ({file_size_mb:.1f} MB)")
        print("  ⚠️  Install tifffile for 16-bit: pip install tifffile")
        return output_path


def verify_tiff_depth(tiff_path: Union[str, Path]) -> dict:
    """
    Verify the bit depth and quality of a TIFF file.

    Returns:
        Dictionary with verification results
    """
    tiff_path = Path(tiff_path)

    try:
        import tifffile

        with tifffile.TiffFile(str(tiff_path)) as tif:
            page = tif.pages[0]
            img_array = page.asarray()

            result = {
                'path': tiff_path,
                'shape': page.shape,
                'dtype': str(page.dtype),
                'bits_per_sample': page.bitspersample,
                'compression': page.compression,
                'photometric': page.photometric,
                'data_range': (img_array.min(), img_array.max()),
                'is_16bit': page.dtype == np.uint16,
                'file_size_mb': tiff_path.stat().st_size / (1024**2)
            }

            # Check if 16-bit data is properly utilized
            if result['is_16bit']:
                unique_values = len(np.unique(img_array))
                result['unique_values'] = unique_values
                result['bit_utilization'] = (unique_values / 65536) * 100

                # Check for incorrect scaling
                if img_array.max() < 300:
                    result['warning'] = "Data appears 8-bit scaled to 16-bit range"

            return result

    except ImportError:
        return {'error': 'tifffile not available for verification'}
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Verify mode
        tiff_path = Path(sys.argv[1])
        if tiff_path.exists():
            print(f"Verifying: {tiff_path}")
            print("=" * 80)
            result = verify_tiff_depth(tiff_path)
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"File not found: {tiff_path}")
    else:
        print("Usage:")
        print("  python fix_tiff_saving.py <tiff_file>  # Verify TIFF")
        print("\nOr import in your code:")
        print("  from fix_tiff_saving import save_16bit_tiff")
