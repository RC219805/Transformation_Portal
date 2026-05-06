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

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Optional imports for 16-bit TIFF support
try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


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
        "format": img.format,
        "mode": img.mode,
        "size": img.size,
        "info": img.info.copy() if hasattr(img, "info") else {},
    }

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img, meta


def _save_with_meta(
    img: Image.Image, arr: Optional[np.ndarray], path: Union[str, Path], meta: Dict[str, Any], out_bitdepth: int = 16
) -> None:
    """
    Save image with metadata preservation.

    Args:
        img: PIL Image to save
        arr: Optional numpy array (for bit depth conversion)
        path: Output path
        meta: Metadata dictionary to preserve
        out_bitdepth: Output bit depth (8, 16, or 32)

    Note:
        16-bit RGB images require TIFF format with tifffile library.
        Other formats will automatically fall back to 8-bit.
        For 16-bit grayscale, PNG is supported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert array to appropriate bit depth if provided
    if arr is not None:
        if out_bitdepth == 16:
            arr_uint = (np.clip(arr, 0, 1) * 65535).astype(np.uint16)

            if arr_uint.ndim == 3 and arr_uint.shape[2] == 3:
                # 16-bit RGB - requires tifffile for TIFF
                ext = path.suffix.lower()
                if ext in [".ti", ".tiff"]:
                    if HAS_TIFFFILE:
                        try:
                            # Preserve metadata if present
                            tifffile_kwargs = {"photometric": "rgb"}
                            # tifffile supports 'description' (string) and 'metadata' (dict)
                            if meta:
                                # If 'description' is present, use it; else, serialize all meta as description
                                description = meta.get("description")
                                if description is None and meta:
                                    # Fallback: serialize meta as a string for description
                                    from transformation_portal.ingest.canonical_json import dumps_json

                                    try:
                                        description = dumps_json(meta)
                                    except Exception:
                                        description = str(meta)
                                if description is not None:
                                    tifffile_kwargs["description"] = description
                                # If 'metadata' is present and a dict, pass it
                                metadata_dict = meta.get("metadata")
                                if isinstance(metadata_dict, dict):
                                    tifffile_kwargs["metadata"] = metadata_dict
                            tifffile.imwrite(path, arr_uint, **tifffile_kwargs)
                            _info(f"Saved: {path} (16-bit TIFF via tifffile, metadata preserved if possible)")
                            return
                        except Exception as e:
                            _warn(f"Failed to write 16-bit TIFF: {e}")
                            _warn("Falling back to 8-bit")
                    else:
                        _warn("tifffile not available for 16-bit TIFF. Install: pip install tifffile")
                        _warn("Falling back to 8-bit")
                else:
                    _warn(f"Format {ext} doesn't support 16-bit RGB, falling back to 8-bit")

                # Fall back to 8-bit
                out_bitdepth = 8

            elif arr_uint.ndim == 2:
                # Grayscale 16-bit - PNG and TIFF supported
                img = Image.fromarray(arr_uint, mode="I;16")
                # Try saving the 16-bit grayscale image to ensure PIL supports it
                try:
                    # Save to a temporary in-memory file to test support
                    from io import BytesIO

                    tmp = BytesIO()
                    img.save(tmp, format=path.suffix.lstrip(".").upper() or "PNG")
                except Exception as e:
                    _warn(f"Failed to save 16-bit grayscale image: {e}")
                    _warn("Falling back to 8-bit grayscale")
                    arr_uint_8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                    img = Image.fromarray(arr_uint_8, mode="L")
                    out_bitdepth = 8

        elif out_bitdepth == 32:
            # Mode 'F' only supports 2D (grayscale) float32 arrays in PIL.
            if arr.ndim == 2:
                img = Image.fromarray(arr.astype(np.float32), mode="F")
            elif arr.ndim == 3 and arr.shape[2] == 3:
                raise ValueError(
                    "Cannot save 32-bit float RGB images with PIL. "
                    "Mode 'F' only supports 2D (grayscale) float32 arrays. "
                    "Use out_bitdepth=8 or 16 for RGB images."
                )
            else:
                raise ValueError(f"Unsupported array shape for 32-bit float image: {arr.shape}")

        else:
            # 8-bit (default)
            arr_uint = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(arr_uint, mode="RGB")

    # Preserve metadata
    info = meta.get("info", {})

    # Save the image
    try:
        img.save(path, **info)
        _info(f"Saved: {path}")
    except Exception as e:
        # Fallback: save without metadata
        _warn(f"Failed to save with metadata, trying without: {e}")
        img.save(path)
        _info(f"Saved: {path} (without metadata)")


def _image_to_float_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to float32 numpy array in [0, 1] range.

    Args:
        img: PIL Image

    Returns:
        Float32 numpy array (H, W, 3)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

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
    **kwargs,
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
        result = result * (2.0**exposure)

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

        blurred = gaussian_filter(result, sigma=5.0, mode="reflect")
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
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
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
        "total_time_ms": elapsed_ms,
        "exposure": exposure,
        "contrast": contrast,
        "saturation": saturation,
        "clarity": clarity,
    }

    return preview, result, metrics


# ==================== CLI Entry Point ====================


def main():
    """CLI for Realize V8 with VFX extension support."""
    import argparse

    parser = argparse.ArgumentParser(description="Realize V8 Unified Enhancement")

    # Check if VFX extension is available
    try:
        # pylint: disable=cyclic-import
        # Lazy import to allow VFX extension to import from this module
        from realize_v8_unified_cli_extension import add_vfx_commands

        has_vfx = True
    except ImportError:
        has_vfx = False

    # Use subparsers if VFX is available, otherwise simple args
    if has_vfx:
        subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

        # Basic enhance command
        # Note: Basic enhance outputs 8/16-bit RGB images (input images of any type are supported; conversion to RGB is automatic)
        # 32-bit output is only supported for grayscale images (used by VFX commands for depth maps)
        p_enhance = subparsers.add_parser("enhance", help="Basic enhancement")
        p_enhance.add_argument("--input", type=Path, required=True)
        p_enhance.add_argument("--output", type=Path, required=True)
        p_enhance.add_argument("--preset", choices=list(PRESETS.keys()), default="signature_estate")
        p_enhance.add_argument(
            "--out-bitdepth",
            type=int,
            choices=[8, 16, 32],
            default=8,
            help="Output bit depth: 8 or 16 for RGB images, 32 for grayscale only",
        )
        p_enhance.set_defaults(func=_handle_basic_enhance)

        # Add VFX commands
        add_vfx_commands(subparsers)

        args = parser.parse_args()

        # Execute command (func is always set by set_defaults)
        return args.func(args) or 0
    else:
        # Fallback to simple CLI without VFX
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--preset", choices=list(PRESETS.keys()), default="signature_estate")
        parser.add_argument(
            "--out-bitdepth",
            type=int,
            choices=[8, 16, 32],
            default=8,
            help="Output bit depth: 8 or 16 for RGB images, 32 for grayscale only",
        )

        args = parser.parse_args()

        # Load preset
        preset = PRESETS[args.preset]

        # Enhance
        img, meta = _open_any(args.input)
        preview, arr, metrics = enhance(img, **preset.to_dict())

        # Save
        _save_with_meta(preview, arr, args.output, meta, out_bitdepth=args.out_bitdepth)

        _info(f"Processing complete: {metrics['total_time_ms']}ms")
        return 0


def _handle_basic_enhance(args):
    """Handle basic enhancement command."""
    # Load preset
    preset = PRESETS[args.preset]

    # Enhance
    img, meta = _open_any(args.input)
    preview, arr, metrics = enhance(img, **preset.to_dict())

    # Save
    out_bitdepth = getattr(args, "out_bitdepth", 8)
    _save_with_meta(preview, arr, args.output, meta, out_bitdepth=out_bitdepth)

    _info(f"Processing complete: {metrics['total_time_ms']}ms")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
