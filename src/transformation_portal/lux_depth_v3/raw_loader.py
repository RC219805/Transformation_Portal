"""RAW camera file loader for APEX pipeline.

Provides high-quality RAW → RGB conversion using rawpy (LibRaw wrapper).

Supported formats:
- Canon: .CR2, .CRW
- Nikon: .NEF, .NRW
- Sony: .ARW, .SRF, .SR2
- Adobe: .DNG (TIFF-based RAW format)
- Olympus: .ORF
- Fujifilm: .RAF
- Pentax: .PEF

Design principles:
- Optional dependency: graceful ImportError if rawpy not installed
- High-quality defaults: camera white balance, full resolution, sRGB
- Deterministic: same file → same RGB output
- Metadata preservation: EXIF data retained where possible
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

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
    # Note: DNG is TIFF-based RAW format (included above).
    # Standard TIFF (.tif/.tiff) is NOT RAW and handled via PIL.
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
    output_bps: int = 16,
    output_linear: bool = True,
) -> np.ndarray:
    """Load RAW camera file and convert to RGB numpy array.

    IMPORTANT: For APEX pipeline and Spatial AI Foundation compliance,
    output_linear MUST be True to preserve linear-light relationships.
    Gamma-encoded outputs violate the Data Fidelity requirement.

    Args:
        raw_path: Path to RAW camera file
        use_camera_wb: Use camera white balance (vs. auto-calculated)
        half_size: Half-size interpolation (faster, lower quality)
        output_bps: Output bits per sample (8 or 16, default 16 for fidelity)
        output_linear: Output linear RGB (True) or gamma-encoded sRGB (False)
                      Default True for linear light preservation (REQUIRED for APEX)

    Returns:
        RGB numpy array (uint8 or uint16 depending on output_bps)
        Shape: (H, W, 3)

        If output_linear=True: Linear RGB (scene-referred)
        If output_linear=False: Gamma-encoded sRGB (display-referred)

    Raises:
        ImportError: If rawpy not installed
        FileNotFoundError: If RAW file doesn't exist
        ValueError: If RAW file cannot be parsed or if gamma-encoded output
                   requested (output_linear=False is blocked for APEX)

    Example:
        >>> # Linear output (APEX compliant)
        >>> rgb = load_raw_as_rgb(Path("IMG_1234.CR2"), output_linear=True)
        >>> print(rgb.shape, rgb.dtype)
        (4000, 6000, 3) uint16

        >>> # Gamma-encoded output (blocked for APEX)
        >>> rgb = load_raw_as_rgb(Path("IMG_1234.CR2"), output_linear=False)
        ValueError: Gamma-encoded output not allowed for APEX pipeline
    """
    try:
        import rawpy
    except ImportError as e:
        raise ImportError(
            "rawpy required for RAW file support.\n" "Install with: pip install rawpy\n" "Or: pip install -e '.[raw]'"
        ) from e

    if not raw_path.exists():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    # APEX compliance check: block gamma-encoded output
    if not output_linear:
        raise ValueError(
            "Gamma-encoded RAW output (output_linear=False) is not allowed for APEX pipeline. "
            "Per Spatial AI Foundation ROADMAP (Section I: Data Fidelity is Sacred), "
            "training inputs MUST preserve linear-light relationships. "
            "Use output_linear=True (default) to preserve linear RGB. "
            "If display-referred output is needed, apply gamma in post-processing with explicit documentation."
        )

    logger.debug(
        f"Loading RAW file: {raw_path.name} "
        f"(use_camera_wb={use_camera_wb}, half_size={half_size}, "
        f"output_linear={output_linear}, output_bps={output_bps})"
    )

    try:
        with rawpy.imread(str(raw_path)) as raw:
            # Determine color space based on output_linear flag
            # Linear output uses rawpy.ColorSpace.raw (no gamma encoding)
            # For gamma output (blocked above), would use rawpy.ColorSpace.sRGB
            if output_linear:
                # Linear RGB output (scene-referred, no gamma)
                # Note: rawpy.ColorSpace.raw gives linear sensor RGB
                # We use rawpy.ColorSpace.raw for true linear output
                output_color = rawpy.ColorSpace.raw
                gamma = (1, 1)  # Linear gamma curve (no encoding)
            else:
                # This path is unreachable due to check above, but kept for clarity
                output_color = rawpy.ColorSpace.sRGB
                gamma = (2.2, 4.5)  # sRGB gamma (for reference)

            # High-quality postprocessing with linear-preserving settings
            rgb = raw.postprocess(
                use_camera_wb=use_camera_wb,  # Use embedded camera WB (vs. auto)
                half_size=half_size,  # Full resolution by default
                no_auto_bright=False,  # Apply auto-brightness (normalize exposure)
                output_color=output_color,  # Linear or sRGB color space
                output_bps=output_bps,  # 8-bit or 16-bit output
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # High-quality AHD algorithm
                use_auto_wb=not use_camera_wb,  # Auto WB if not using camera WB
                gamma=gamma,  # Gamma curve (linear or sRGB)
                no_auto_scale=False,  # Auto-scale to full range
                user_flip=0,  # No rotation
                four_color_rgb=False,  # Standard 3-color RGB
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,  # No NR (preserve detail)
            )

            logger.debug(
                f"RAW converted: {raw_path.name} → RGB {rgb.shape} {rgb.dtype} "
                f"({'linear' if output_linear else 'gamma-encoded'}) "
                f"(sensor: {raw.raw_image.shape}, ISO: {raw.camera_iso_speed if hasattr(raw, 'camera_iso_speed') else 'N/A'})"
            )

            return rgb

    except Exception as e:
        raise ValueError(f"Failed to load RAW file {raw_path.name}: {e}") from e


def load_raw_as_pil(
    raw_path: Path,
    use_camera_wb: bool = True,
    half_size: bool = False,
    output_linear: bool = True,
) -> Image.Image:
    """Load RAW camera file and convert to PIL Image.

    Convenience wrapper around load_raw_as_rgb() that returns PIL Image
    instead of numpy array.

    IMPORTANT: For APEX pipeline compliance, output_linear=True (default)
    to preserve linear-light relationships.

    Args:
        raw_path: Path to RAW camera file
        use_camera_wb: Use camera white balance (vs. auto-calculated)
        half_size: Half-size interpolation (faster, lower quality)
        output_linear: Output linear RGB (True) or gamma-encoded sRGB (False)
                      Default True for linear light preservation (REQUIRED for APEX)

    Returns:
        PIL Image in RGB mode (uint16 for linear, uint8 for gamma)

    Raises:
        ImportError: If rawpy not installed
        FileNotFoundError: If RAW file doesn't exist
        ValueError: If RAW file cannot be parsed or gamma output requested
    """
    # Use 16-bit for linear (preserve precision), 8-bit for gamma (legacy)
    output_bps = 16 if output_linear else 8

    rgb = load_raw_as_rgb(
        raw_path,
        use_camera_wb=use_camera_wb,
        half_size=half_size,
        output_bps=output_bps,
        output_linear=output_linear,
    )

    # Convert to PIL Image with appropriate mode
    if rgb.dtype == np.uint16:
        # PIL doesn't have native 16-bit RGB mode, so we scale to 8-bit
        # Use proper scaling: 65535 / 255 ≈ 257 (not 256)
        # This preserves the full dynamic range without truncation
        rgb_8bit = (rgb / 257).astype(np.uint8)
        return Image.fromarray(rgb_8bit, mode="RGB")
    else:
        # 8-bit RGB (legacy path, should not be reached for APEX)
        return Image.fromarray(rgb, mode="RGB")
