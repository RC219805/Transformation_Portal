#!/usr/bin/env python3
"""
convert_problem_tiffs.py

Recursively find TIFF files under IMAGES_ROOT and produce safe PNG conversions
named <origstem>_conv.png next to each original. Uses tifffile for robust loading.

Usage:
    python convert_problem_tiffs.py /path/to/images
"""
import sys
from pathlib import Path
import numpy as np

try:
    import tifffile
except Exception as e:
    print("tifffile required. pip install tifffile", file=sys.stderr)
    raise

from PIL import Image

def normalize_float_to_uint8(arr: np.ndarray) -> np.ndarray:
    # map array to uint8 range 0..255
    amin, amax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if amax - amin < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - amin) / (amax - amin)
    out = (scaled * 255.0).clip(0, 255).astype(np.uint8)
    return out

def to_image_array(arr: np.ndarray) -> np.ndarray:
    """Normalize and reshape arr to either (H,W) or (H,W,3), dtype uint8 or uint16."""
    # collapse leading singleton dims
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    # if still >3 dims, try to take first page/frame
    if arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], -1)  # fallback flatten pages
        arr = arr[0]
    # Now handle common cases:
    # - (H,W)
    # - (H,W,1)
    # - (H,W,3)
    # - (1,H,W) or (C,H,W)
    if arr.ndim == 3 and arr.shape[0] in (1,3,4):  # (C,H,W) -> (H,W,C)
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[2] == 1:  # (H,W,1) -> (H,W)
        arr = arr[...,0]
    # dtype handling
    if np.issubdtype(arr.dtype, np.floating):
        arr = normalize_float_to_uint8(arr)
    elif np.issubdtype(arr.dtype, np.integer):
        # if already fits in uint8, convert; if large (e.g. 16-bit) keep appropriate type
        if arr.dtype == np.uint8:
            pass
        else:
            # choose uint16 if max>255, else uint8
            if int(np.nanmax(arr)) > 255:
                arr = arr.astype(np.uint16)
            else:
                arr = arr.astype(np.uint8)
    else:
        # fallback: cast to uint8 via normalization
        arr = normalize_float_to_uint8(arr.astype('float32'))
    return arr

def convert_file(p: Path):
    try:
        arr = tifffile.imread(p)
    except Exception as e:
        print(f"ERROR reading {p}: {e}")
        return False
    arr2 = to_image_array(arr)
    # prepare PIL Image
    try:
        if arr2.ndim == 2:
            mode = "I;16" if arr2.dtype == np.uint16 else None
            im = Image.fromarray(arr2, mode=mode) if mode else Image.fromarray(arr2)
        elif arr2.ndim == 3 and arr2.shape[2] in (3,4):
            # if uint16 use 'I;16' per channel isn't supported; convert to uint8 for PNG
            if arr2.dtype == np.uint16:
                # scale down to 0..255 for RGB PNG (preserve approx)
                arr_out = (arr2 / 257).astype(np.uint8)  # 65535/255 ≈ 257
            else:
                arr_out = arr2.astype(np.uint8)
            im = Image.fromarray(arr_out)
        else:
            print(f"Unsupported post-processed array shape/dtype: {arr2.shape} {arr2.dtype} for {p}")
            return False
    except Exception as e:
        print(f"PIL.fromarray failed for {p}: {e}")
        return False

    out = p.with_name(p.stem + "_conv.png")
    try:
        im.save(out)
        print("Converted:", p, "->", out)
        return True
    except Exception as e:
        print(f"Failed to save {out}: {e}")
        return False

def main(root):
    root = Path(root)
    tifs = list(root.rglob("*.tif")) + list(root.rglob("*.tiff"))
    if not tifs:
        print("No TIFF files found under", root)
        return
    for p in tifs:
        convert_file(p)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_problem_tiffs.py /path/to/images")
        sys.exit(1)
    main(sys.argv[1])
