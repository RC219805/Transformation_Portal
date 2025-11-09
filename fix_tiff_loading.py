#!/usr/bin/env python3
"""
Fix TIFF loading to preserve 16-bit depth.
PIL automatically converts 16-bit TIFFs to 8-bit when using np.array() or .convert().
This module provides correct 16-bit loading functions.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple
from PIL import Image


def load_16bit_tiff(path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """
    Load a TIFF file preserving 16-bit depth.

    Args:
        path: Path to TIFF file

    Returns:
        Tuple of (float array [0,1], metadata dict)
    """
    path = Path(path)

    try:
        import tifffile

        # Load with tifffile to preserve 16-bit
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            image = page.asarray()

            metadata = {
                'original_dtype': str(image.dtype),
                'original_shape': image.shape,
                'bits_per_sample': page.bitspersample,
                'compression': page.compression,
            }

            # Convert to float [0, 1]
            if image.dtype == np.uint16:
                image_float = image.astype(np.float32) / 65535.0
            elif image.dtype == np.uint8:
                image_float = image.astype(np.float32) / 255.0
            else:
                image_float = image.astype(np.float32)

            print(f"✓ Loaded 16-bit TIFF: {path.name} as {metadata['original_dtype']}")
            return image_float, metadata

    except ImportError:
        print(f"  ⚠️  tifffile not available, using PIL (will lose 16-bit depth)")

        # PIL fallback - this will convert to 8-bit!
        pil_img = Image.open(str(path))
        image = np.array(pil_img)

        metadata = {
            'original_dtype': str(image.dtype),
            'original_shape': image.shape,
            'bits_per_sample': 8,
            'warning': 'Loaded via PIL - 16-bit depth lost',
        }

        image_float = image.astype(np.float32) / 255.0
        print(f"  Loaded 8-bit (from PIL): {path.name}")
        print(f"  ⚠️  Install tifffile for 16-bit: pip install tifffile")

        return image_float, metadata


def convert_8bit_to_16bit_tiff(input_path: Union[str, Path], output_path: Union[str, Path]):
    """
    Re-save an 8-bit TIFF as proper 16-bit.
    Note: This cannot recover lost bit depth, only prevents further loss.
    """
    from fix_tiff_saving import save_16bit_tiff

    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Converting {input_path.name} to 16-bit...")

    # Load (may be 8-bit or 16-bit)
    image, metadata = load_16bit_tiff(input_path)

    # Save as 16-bit
    save_16bit_tiff(image, output_path)

    print(f"✓ Converted to: {output_path}")
    print(f"  Note: If source was 8-bit, no data recovery possible")


def verify_and_fix_directory(input_dir: Union[str, Path], output_dir: Union[str, Path] = None):
    """
    Verify all TIFFs in a directory and re-save any that are incorrectly 8-bit.
    """
    from fix_tiff_saving import verify_tiff_depth, save_16bit_tiff

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "fixed_16bit"
    output_dir.mkdir(exist_ok=True, parents=True)

    tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))

    print(f"\nVerifying {len(tiff_files)} TIFF files in {input_dir}")
    print("=" * 80)

    needs_fixing = []

    for tiff_path in tiff_files:
        result = verify_tiff_depth(tiff_path)

        if result.get('is_16bit'):
            print(f"✓ {tiff_path.name}: Correct 16-bit")
        else:
            print(f"✗ {tiff_path.name}: Incorrect 8-bit - NEEDS FIX")
            needs_fixing.append(tiff_path)

    if needs_fixing:
        print(f"\n{len(needs_fixing)} files need fixing:")
        for path in needs_fixing:
            output_path = output_dir / path.name
            convert_8bit_to_16bit_tiff(path, output_path)
    else:
        print("\n✓ All TIFFs are correctly 16-bit!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])

        if input_path.is_dir():
            # Directory mode
            output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
            verify_and_fix_directory(input_path, output_dir)
        elif input_path.exists():
            # Single file mode
            output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.parent / f"{input_path.stem}_16bit.tif"
            convert_8bit_to_16bit_tiff(input_path, output_path)
        else:
            print(f"Path not found: {input_path}")
    else:
        print("Usage:")
        print("  python fix_tiff_loading.py <input.tif> [output.tif]")
        print("  python fix_tiff_loading.py <directory> [output_directory]")
        print("\nOr import in your code:")
        print("  from fix_tiff_loading import load_16bit_tiff")
