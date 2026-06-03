"""RAW camera file loader for APEX pipeline.

Provides high-quality RAW → RGB conversion using rawpy (LibRaw wrapper).

Supported formats:
- Canon: .CR2, .CR3, .CRW
- Nikon: .NEF, .NRW
- Sony: .ARW, .SRF, .SR2, .SRW
- Adobe: .DNG (TIFF-based RAW format)
- Olympus: .ORF
- Fujifilm: .RAF
- Pentax: .PEF
- Panasonic: .RW2
- Phase One: .IIQ
- Hasselblad: .3FR

Design principles:
- Optional dependency: graceful ImportError if rawpy not installed
- High-quality defaults: camera white balance, full resolution, sRGB
- Deterministic: same file → same RGB output
- Metadata preservation: EXIF data retained where possible
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from transformation_portal.core.raw_formats import RAW_EXTENSIONS as RAW_EXTENSIONS  # noqa: F401  (re-export)
from transformation_portal.core.raw_runtime import (  # noqa: F401  (re-exports)
    is_valid_demosaic_name,
    resolve_demosaic_algorithm,
    run_raw_worker,
)

logger = logging.getLogger(__name__)

# RAW_EXTENSIONS is re-exported from transformation_portal.core.raw_formats
# (a stdlib-only module so format-classifiers like input_manager — which
# must not pull PIL/rawpy at import time — can share this constant).
#
# is_valid_demosaic_name and resolve_demosaic_algorithm are re-exported from
# transformation_portal.core.raw_runtime so the rendering and research/training
# paths share a single source of truth without violating ADR-023's import
# isolation between those surfaces.


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
    output_linear: bool = False,
    python_executable: Optional[str] = None,
    demosaic: str = "AHD",
) -> np.ndarray:
    """Load RAW camera file and convert to RGB numpy array.

    IMPORTANT: For APEX pipeline compliance, use output_linear=True with output_bps=16.
    The defaults (8-bit gamma-encoded) are for legacy compatibility only.

    Args:
        raw_path: Path to RAW camera file
        use_camera_wb: Use camera white balance (vs. auto-calculated)
        half_size: Half-size interpolation (faster, lower quality)
        output_bps: Output bits per sample (8 or 16, default 8 for legacy compatibility)
        output_linear: Output linear RGB (True) or gamma-encoded sRGB (False)
                      Default False for legacy compatibility
                      MUST be True for APEX pipeline
        python_executable: Optional interpreter for an isolated RAW runtime.
            When provided, the decode runs in a subprocess backed by that
            interpreter instead of importing rawpy in-process.
        demosaic: rawpy.DemosaicAlgorithm name (e.g. "AHD", "AMAZE", "DCB",
            "LMMSE", "VNG", "PPG"). Default "AHD" preserves prior behavior.
            Unknown names fail closed via resolve_demosaic_algorithm().

    Returns:
        RGB numpy array (uint8 or uint16 depending on output_bps)
        Shape: (H, W, 3)

        If output_linear=True: Linear RGB (scene-referred)
        If output_linear=False (default): Gamma-encoded sRGB (display-referred)

    Raises:
        ImportError: If rawpy not installed
        FileNotFoundError: If RAW file doesn't exist
        ValueError: If RAW file cannot be parsed or demosaic name is unknown

    Example:
        >>> # Legacy usage (8-bit gamma-encoded)
        >>> rgb = load_raw_as_rgb(Path("IMG_1234.CR2"))
        >>> print(rgb.shape, rgb.dtype)
        (4000, 6000, 3) uint8

        >>> # APEX usage (16-bit linear)
        >>> rgb = load_raw_as_rgb(Path("IMG_1234.CR2"),
        ...                       output_bps=16,
        ...                       output_linear=True)
        >>> print(rgb.shape, rgb.dtype)
        (4000, 6000, 3) uint16
    """
    if python_executable is not None:
        if not raw_path.exists():
            raise FileNotFoundError(f"RAW file not found: {raw_path}")
        rgb, _ = run_raw_worker(
            python_executable=python_executable,
            command_name="load_rgb",
            input_path=raw_path,
            payload={
                "use_camera_wb": bool(use_camera_wb),
                "half_size": bool(half_size),
                "output_bps": int(output_bps),
                "output_linear": bool(output_linear),
                "demosaic": str(demosaic),
            },
            start=Path(__file__),
        )
        return np.asarray(rgb)

    try:
        import rawpy
    except ImportError as e:
        raise ImportError(
            "rawpy required for RAW file support.\n"
            "Use ./scripts/setup/install_raw_runtime.sh for the isolated RAW runtime, "
            "or deliberately install the RAW extra into this active interpreter for development."
        ) from e

    if not raw_path.exists():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    demosaic_enum = resolve_demosaic_algorithm(demosaic)

    logger.debug(
        f"Loading RAW file: {raw_path.name} "
        f"(use_camera_wb={use_camera_wb}, half_size={half_size}, "
        f"output_linear={output_linear}, output_bps={output_bps}, "
        f"demosaic={demosaic})"
    )

    try:
        with rawpy.imread(str(raw_path)) as raw:
            # Determine color space based on output_linear flag
            if output_linear:
                # Linear RGB output (scene-referred, no gamma)
                # Use raw color space for true linear output
                output_color = rawpy.ColorSpace.raw
                gamma: Tuple[float, float] = (1.0, 1.0)  # Linear gamma curve (no encoding)
            else:
                # Gamma-encoded sRGB output (display-referred, legacy)
                output_color = rawpy.ColorSpace.sRGB
                gamma = (2.2, 4.5)  # sRGB gamma

            # High-quality postprocessing with linear-preserving settings
            rgb = raw.postprocess(
                use_camera_wb=use_camera_wb,  # Use embedded camera WB (vs. auto)
                half_size=half_size,  # Full resolution by default
                no_auto_bright=False,  # Apply auto-brightness (normalize exposure)
                output_color=output_color,  # Linear or sRGB color space
                output_bps=output_bps,  # 8-bit or 16-bit output
                demosaic_algorithm=demosaic_enum,
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
    output_linear: bool = False,
    python_executable: Optional[str] = None,
    demosaic: str = "AHD",
) -> Image.Image:
    """Load RAW camera file and convert to PIL Image.

    Convenience wrapper around load_raw_as_rgb() that returns PIL Image
    instead of numpy array.

    IMPORTANT: For APEX pipeline compliance, use output_linear=True.
    The default (False) is for legacy compatibility only.

    Args:
        raw_path: Path to RAW camera file
        use_camera_wb: Use camera white balance (vs. auto-calculated)
        half_size: Half-size interpolation (faster, lower quality)
        output_linear: Output linear RGB (True) or gamma-encoded sRGB (False)
                      Default False for legacy compatibility
                      MUST be True for APEX pipeline
        python_executable: Optional interpreter for an isolated RAW runtime.
        demosaic: rawpy.DemosaicAlgorithm name (default "AHD"). See
            load_raw_as_rgb for details.

    Returns:
        PIL Image in RGB mode (uint8)

    Raises:
        ImportError: If rawpy not installed
        FileNotFoundError: If RAW file doesn't exist
        ValueError: If RAW file cannot be parsed or demosaic name is unknown
    """
    # Use 16-bit for linear (preserve precision), 8-bit for gamma (legacy)
    output_bps = 16 if output_linear else 8

    rgb = load_raw_as_rgb(
        raw_path,
        use_camera_wb=use_camera_wb,
        half_size=half_size,
        output_bps=output_bps,
        output_linear=output_linear,
        python_executable=python_executable,
        demosaic=demosaic,
    )

    # Convert to PIL Image with appropriate mode
    if rgb.dtype == np.uint16:
        # PIL doesn't have native 16-bit RGB mode, so we scale to 8-bit
        # Use proper scaling: 65535 / 255 ≈ 257 (not 256)
        # This preserves the full dynamic range without truncation
        rgb_8bit = (rgb / 257).astype(np.uint8)
        return Image.fromarray(rgb_8bit, mode="RGB")
    else:
        # 8-bit RGB (legacy path)
        return Image.fromarray(rgb, mode="RGB")
