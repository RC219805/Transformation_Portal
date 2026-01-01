"""Depth writer for V2 contract compliance.

Writes depth maps in uint16 PNG format matching V2's expected input contract:
- filename: {stem}_depth.png
- dtype: uint16
- shape: (H, W) - single channel, no RGB depth
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def write_depth_u16_png(
    path: Path,
    depth: np.ndarray,
    method: str = "p1p99",
    debug_verify: bool = False,
) -> Tuple[float, float]:
    """Write depth map as uint16 PNG for V2 consumption.

    Args:
        path: Output path (should end with _depth.png)
        depth: Depth array (float32 or uint16)
        method: Quantization method if float input ("p1p99", "p0.5p99.5", "minmax")
        debug_verify: If True, read back and verify shape/dtype

    Returns:
        Tuple of (p1, p99) percentile values used for quantization

    Raises:
        ValueError: If depth shape is invalid or contains NaN/Inf
    """
    # Validate input
    if depth.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D depth, got shape {depth.shape}")

    # Handle 3D depth (take first channel)
    if depth.ndim == 3:
        if depth.shape[2] != 1:
            logger.warning(
                f"Depth has {depth.shape[2]} channels, taking first channel only. "
                "V2 requires single-channel depth."
            )
        depth = depth[:, :, 0]

    # Check for invalid values
    if not np.isfinite(depth).all():
        raise ValueError("Depth contains NaN or Inf values")

    # Convert to uint16 if needed
    if depth.dtype == np.uint16:
        # Already quantized: preserve raw uint16 values but compute true statistics
        depth_u16 = depth
        depth_f32 = depth.astype(np.float32)

        if method == "p1p99":
            p1 = float(np.percentile(depth_f32, 1.0))
            p99 = float(np.percentile(depth_f32, 99.0))
        elif method == "p0.5p99.5":
            p1 = float(np.percentile(depth_f32, 0.5))
            p99 = float(np.percentile(depth_f32, 99.5))
        elif method == "minmax":
            p1 = float(depth_f32.min())
            p99 = float(depth_f32.max())
        else:
            raise ValueError(f"Unknown quantization method: {method}")
    else:
        # Quantize float depth to uint16
        depth_f32 = depth.astype(np.float32)

        if method == "p1p99":
            p1 = np.percentile(depth_f32, 1.0)
            p99 = np.percentile(depth_f32, 99.0)
        elif method == "p0.5p99.5":
            p1 = np.percentile(depth_f32, 0.5)
            p99 = np.percentile(depth_f32, 99.5)
        elif method == "minmax":
            p1 = float(depth_f32.min())
            p99 = float(depth_f32.max())
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        # Prevent division by zero
        if p99 <= p1 + 1e-6:
            logger.warning(f"Depth range too small (p1={p1:.3f}, p99={p99:.3f}), using zeros")
            depth_u16 = np.zeros_like(depth_f32, dtype=np.uint16)
        else:
            # Clip and map to [0, 65535]
            depth_normalized = np.clip((depth_f32 - p1) / (p99 - p1), 0.0, 1.0)
            depth_u16 = (depth_normalized * 65535.0 + 0.5).astype(np.uint16)

    # Ensure output directory exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write single-channel PNG (mode='I;16' for 16-bit grayscale)
    img = Image.fromarray(depth_u16, mode='I;16')
    img.save(str(path))

    logger.info(f"Wrote depth to {path} (shape={depth_u16.shape}, dtype=uint16)")

    # Optional verification
    if debug_verify:
        verify_depth = np.array(Image.open(path))
        assert verify_depth.shape == depth_u16.shape, \
            f"Shape mismatch: wrote {depth_u16.shape}, read {verify_depth.shape}"
        assert verify_depth.dtype == np.uint16, \
            f"Dtype mismatch: expected uint16, got {verify_depth.dtype}"
        logger.debug(f"Verified depth write: shape={verify_depth.shape}, dtype={verify_depth.dtype}")

    return float(p1), float(p99)


def read_depth_u16_png(path: Path) -> np.ndarray:
    """Read uint16 depth PNG written by write_depth_u16_png.

    Args:
        path: Path to depth PNG

    Returns:
        Depth array as uint16 (H, W)

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If depth is not single-channel uint16
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Depth file not found: {path}")

    img = Image.open(path)
    depth = np.array(img)

    # Validate
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth, got shape {depth.shape}")
    if depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth, got {depth.dtype}")

    return depth
