#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realize_v8_unified.py - Base enhancement pipeline for Transformation Portal

Provides core image enhancement functionality with support for presets,
color management, and ICC profiles. This serves as the foundation for
VFX extensions.

Usage:
    from realize_v8_unified import enhance, PRESETS
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image


# ==================== Logging Utilities ====================

def _info(msg: str) -> None:
    """Print info message."""
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    """Print warning message."""
    print(f"[WARN] {msg}")


def _error(msg: str) -> None:
    """Print error message."""
    print(f"[ERROR] {msg}")


# ==================== Image I/O ====================

def _open_any(path: Union[str, Path]) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Open an image and extract metadata.

    Args:
        path: Path to image file

    Returns:
        Tuple of (PIL Image, metadata dict)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)

    # Extract metadata
    meta = {
        'format': img.format,
        'mode': img.mode,
        'size': img.size,
        'info': img.info.copy() if hasattr(img, 'info') else {},
    }

    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return img, meta


def _save_with_meta(
    img: Image.Image,
    arr: Optional[np.ndarray],
    path: Union[str, Path],
    meta: Dict[str, Any],
    out_bitdepth: int = 16
) -> int:
    """
    Save image with metadata preservation.

    Args:
        img: PIL Image to save
        arr: Optional numpy array (for bit depth conversion)
        path: Output path
        meta: Metadata dictionary to preserve
        out_bitdepth: Output bit depth (8, 16, or 32)

    Returns:
        Actual bit depth used (may differ from requested if not supported)

    Note:
        16-bit RGB is not supported by PIL and will be saved as 8-bit with a warning.
        For true 16-bit RGB support, use a library like tifffile.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Track actual bit depth used
    actual_bitdepth = out_bitdepth

    # Convert array to appropriate bit depth if provided
    if arr is not None:
        if out_bitdepth == 16:
            # For 16-bit images, we need to handle RGB and grayscale differently
            # PIL's native 16-bit support is limited, so we use a workaround
            if arr.ndim == 3 and arr.shape[2] == 3:
                # RGB image - convert to 8-bit for PIL since PIL doesn't support 16-bit RGB natively
                # For true 16-bit RGB, use tifffile library (not a dependency here)
                _warn("16-bit RGB not fully supported by PIL - saving as 8-bit RGB instead. "
                      "For true 16-bit RGB support, install tifffile: pip install tifffile")
                arr_uint = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(arr_uint, mode='RGB')
                actual_bitdepth = 8  # Downgraded
            elif arr.ndim == 2:
                # Grayscale image - PIL supports 16-bit grayscale
                arr_uint = (np.clip(arr, 0, 1) * 65535).astype(np.uint16)
                img = Image.fromarray(arr_uint, mode='I;16')
            else:
                raise ValueError(f"Unsupported array shape for 16-bit image: {arr.shape}")
        elif out_bitdepth == 32:
            # Mode 'F' only supports 2D (grayscale) float32 arrays in PIL
            if arr.ndim == 2:
                img = Image.fromarray(arr.astype(np.float32), mode='F')
            elif arr.ndim == 3 and arr.shape[2] == 3:
                raise ValueError(
                    "Cannot save 32-bit float RGB images with PIL. "
                    "Mode 'F' only supports 2D (grayscale) float32 arrays. "
                    "Use out_bitdepth=8 for RGB images, or install tifffile for 16-bit support."
                )
            else:
                raise ValueError(f"Unsupported array shape for 32-bit float image: {arr.shape}")
        else:  # 8-bit
            arr_uint = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(arr_uint, mode='RGB')

    # Preserve metadata
    info = meta.get('info', {})
    img.save(path, **info)

    _info(f"Saved: {path}")
    return actual_bitdepth


def _image_to_float_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to float32 numpy array in [0, 1] range.

    Args:
        img: PIL Image

    Returns:
        Float32 numpy array (H, W, 3)
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ==================== Preset Configuration ====================

@dataclass
class Preset:
    """Configuration preset for image enhancement."""
    name: str
    description: str
    exposure: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    clarity: float = 0.0
    grain: float = 0.0
    vignette: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary."""
        return asdict(self)


# Predefined presets
PRESETS = {
    "signature_estate": Preset(
        name="Signature Estate",
        description="Balanced enhancement for luxury real estate",
        exposure=0.1,
        contrast=1.08,
        saturation=1.05,
        clarity=0.15,
    ),
    "signature_estate_agx": Preset(
        name="Signature Estate AGX",
        description="AgX tone mapping for luxury real estate",
        exposure=0.05,
        contrast=1.12,
        saturation=1.08,
        clarity=0.18,
    ),
    "natural": Preset(
        name="Natural",
        description="Minimal enhancement preserving original look",
        exposure=0.0,
        contrast=1.02,
        saturation=1.0,
        clarity=0.05,
    ),
}


# ==================== Core Enhancement Functions ====================

def enhance(
    img_or_arr: Union[Image.Image, np.ndarray, str, Path],
    exposure: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    clarity: float = 0.0,
    grain: float = 0.0,
    vignette: float = 0.0,
    random_seed: Optional[int] = None,
    **kwargs
) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """
    Apply basic enhancement to an image.

    Args:
        img_or_arr: Input image (PIL Image, numpy array, or path)
        exposure: Exposure adjustment in stops (-2 to +2)
        contrast: Contrast multiplier (0.5 to 2.0)
        saturation: Saturation multiplier (0 to 2.0)
        clarity: Local contrast enhancement (0 to 1.0)
        grain: Film grain amount (0 to 1.0)
        vignette: Vignette strength (0 to 1.0)
        random_seed: Optional random seed for reproducible grain (None for random)
        **kwargs: Additional parameters (ignored)

    Returns:
        Tuple of (preview PIL Image, working numpy array, metrics dict)
    """
    import time
    t_start = time.perf_counter()

    # Load image
    if isinstance(img_or_arr, (str, Path)):
        img, _ = _open_any(img_or_arr)
        arr = _image_to_float_array(img)
    elif isinstance(img_or_arr, Image.Image):
        arr = _image_to_float_array(img_or_arr)
    elif isinstance(img_or_arr, np.ndarray):
        arr = img_or_arr.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(img_or_arr)}")

    # Apply adjustments
    result = arr.copy()

    # Exposure
    if exposure != 0.0:
        result = result * (2.0 ** exposure)

    # Contrast (around middle gray)
    if contrast != 1.0:
        result = (result - 0.5) * contrast + 0.5

    # Saturation
    if saturation != 1.0:
        # Convert to HSV-like saturation adjustment
        gray = 0.299 * result[..., 0] + 0.587 * result[..., 1] + 0.114 * result[..., 2]
        gray = gray[..., None]
        result = gray + (result - gray) * saturation

    # Clarity (local contrast via unsharp mask)
    if clarity > 0.0:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(result, sigma=5.0, mode='reflect')
        result = result + (result - blurred) * clarity

    # Grain
    if grain > 0.0:
        rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()
        noise = rng.normal(0, grain * 0.05, result.shape).astype(np.float32)
        result = result + noise

    # Vignette
    if vignette > 0.0:
        h, w = result.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        vignette_mask = 1.0 - (dist / max_dist) * vignette
        vignette_mask = np.clip(vignette_mask, 0, 1)
        result = result * vignette_mask[..., None]

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    # Convert to PIL Image for preview
    preview = Image.fromarray((result * 255).astype(np.uint8))

    # Metrics
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    metrics = {
        'total_time_ms': elapsed_ms,
        'exposure': exposure,
        'contrast': contrast,
        'saturation': saturation,
        'clarity': clarity,
    }

    return preview, result, metrics


# ==================== CLI Entry Point ====================

def main():
    """Basic CLI for testing - actual CLI should be in separate module."""
    import argparse

    parser = argparse.ArgumentParser(description="Realize V8 Unified Enhancement")
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=list(PRESETS.keys()), default='signature_estate')

    args = parser.parse_args()

    # Load preset
    preset = PRESETS[args.preset]

    # Enhance
    img, meta = _open_any(args.input)
    preview, arr, metrics = enhance(img, **preset.to_dict())

    # Save
    _save_with_meta(preview, arr, args.output, meta)

    _info(f"Processing complete: {metrics['total_time_ms']}ms")


if __name__ == "__main__":
    main()
