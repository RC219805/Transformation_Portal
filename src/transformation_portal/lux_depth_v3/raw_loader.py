"""RAW camera file loader for APEX pipeline.

Provides high-quality RAW → RGB conversion using rawpy (LibRaw wrapper).

Supported formats:
- Canon: .CR2, .CRW
- Nikon: .NEF, .NRW
- Sony: .ARW, .SRF, .SR2
- Adobe: .DNG
- Olympus: .ORF
- Fujifilm: .RAF
- Pentax: .PEF
- TIFF: .TIF, .TIFF (when used as RAW container)

Design principles:
- Optional dependency: graceful ImportError if rawpy not installed
- High-quality defaults: camera white balance, full resolution, sRGB
- Deterministic: same file → same RGB output
- Metadata preservation: EXIF data retained where possible
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import rawpy

logger = logging.getLogger(__name__)

# RAW file extensions (case-insensitive)
RAW_EXTENSIONS = {
    # Canon
    ".cr2",
    ".crw",
    # Nikon
    ".nef",
    ".nrw",
    # Sony
    ".arw",
    ".srf",
    ".sr2",
    # Adobe
    ".dng",
    # Olympus
    ".orf",
    # Fujifilm
    ".raf",
    # Pentax
    ".pef",
    # Panasonic
    ".rw2",
    # Phase One
    ".iiq",
    # Hasselblad
    ".3fr",
    # TIFF as RAW container
    ".tif",
    ".tiff",
}


def is_raw_file(path: Path) -> bool:
    """Check if file extension indicates RAW format.

    Args:
        path: File path to check

    Returns:
        True if file extension matches known RAW formats
    """
    return path.suffix.lower() in RAW_EXTENSIONS


def load_raw_as_rgb(
    raw_path: Path,
    use_camera_wb: bool = True,
    half_size: bool = False,
    output_bps: int = 8,
) -> np.ndarray:
    """Load RAW camera file and convert to RGB numpy array.

    Args:
        raw_path: Path to RAW camera file
        use_camera_wb: Use camera white balance (vs. auto-calculated)
        half_size: Half-size interpolation (faster, lower quality)
        output_bps: Output bits per sample (8 or 16)

    Returns:
        RGB numpy array (uint8 or uint16 depending on output_bps)
        Shape: (H, W, 3)

    Raises:
        ImportError: If rawpy not installed
        FileNotFoundError: If RAW file doesn't exist
        ValueError: If RAW file cannot be parsed

    Example:
        >>> rgb = load_raw_as_rgb(Path("IMG_1234.CR2"))
        >>> print(rgb.shape, rgb.dtype)
        (4000, 6000, 3) uint8
    """
    try:
        import rawpy
    except ImportError as e:
        raise ImportError(
            "rawpy required for RAW file support.\n" "Install with: pip install rawpy\n" "Or: pip install -e '.[raw]'"
        ) from e

    if not raw_path.exists():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    logger.debug(f"Loading RAW file: {raw_path.name} (use_camera_wb={use_camera_wb}, half_size={half_size})")

    try:
        with rawpy.imread(str(raw_path)) as raw:
            # High-quality postprocessing with sensible defaults
            rgb = raw.postprocess(
                use_camera_wb=use_camera_wb,  # Use embedded camera WB (vs. auto)
                half_size=half_size,  # Full resolution by default
                no_auto_bright=False,  # Apply auto-brightness (normalize exposure)
                output_color=rawpy.ColorSpace.sRGB,  # sRGB color space (standard for web/display)
                output_bps=output_bps,  # 8-bit or 16-bit output
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # High-quality AHD algorithm
                use_auto_wb=not use_camera_wb,  # Auto WB if not using camera WB
            )

            logger.debug(
                f"RAW converted: {raw_path.name} → RGB {rgb.shape} {rgb.dtype} "
                f"(sensor: {raw.raw_image.shape}, ISO: {raw.camera_iso_speed if hasattr(raw, 'camera_iso_speed') else 'N/A'})"
            )

            return rgb

    except Exception as e:
        raise ValueError(f"Failed to load RAW file {raw_path.name}: {e}") from e


def load_raw_as_pil(
    raw_path: Path,
    use_camera_wb: bool = True,
    half_size: bool = False,
) -> Image.Image:
    """Load RAW camera file and convert to PIL Image.

    Convenience wrapper around load_raw_as_rgb() that returns PIL Image
    instead of numpy array.

    Args:
        raw_path: Path to RAW camera file
        use_camera_wb: Use camera white balance (vs. auto-calculated)
        half_size: Half-size interpolation (faster, lower quality)

    Returns:
        PIL Image in RGB mode

    Raises:
        ImportError: If rawpy not installed
        FileNotFoundError: If RAW file doesn't exist
        ValueError: If RAW file cannot be parsed
    """
    rgb = load_raw_as_rgb(raw_path, use_camera_wb=use_camera_wb, half_size=half_size, output_bps=8)

    return Image.fromarray(rgb, mode="RGB")
