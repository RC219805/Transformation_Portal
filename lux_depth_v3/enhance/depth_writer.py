"""Depth writer for V2 contract compliance.

Writes depth maps in uint16 PNG format matching V2's expected input contract:
- filename: {stem}_depth.png
- dtype: uint16
- shape: (H, W) - single channel, no RGB depth
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, NamedTuple
import logging
import os

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DepthScalingStats(NamedTuple):
    """Detailed statistics from depth quantization for provenance."""

    method: str
    p_low_percentile: float
    p_high_percentile: float
    v_low_value: float
    v_high_value: float
    clipped_low_frac: float
    clipped_high_frac: float
    invalid_frac: float


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
                f"Depth has {depth.shape[2]} channels, taking first channel only. V2 requires single-channel depth."
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
    img = Image.fromarray(depth_u16, mode="I;16")
    img.save(str(path))

    logger.info(f"Wrote depth to {path} (shape={depth_u16.shape}, dtype=uint16)")

    # Optional verification
    if debug_verify:
        verify_depth = np.array(Image.open(path))
        assert verify_depth.shape == depth_u16.shape, f"Shape mismatch: wrote {depth_u16.shape}, read {verify_depth.shape}"
        assert verify_depth.dtype == np.uint16, f"Dtype mismatch: expected uint16, got {verify_depth.dtype}"
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


def write_depth_u16_png_with_stats(
    path: Path,
    depth: np.ndarray,
    method: str = "p1p99",
    debug_verify: bool = False,
) -> Tuple[float, float, DepthScalingStats]:
    """Write depth map with detailed scaling statistics for provenance.

    Enhanced version of write_depth_u16_png that computes clipping fractions
    and invalid pixel statistics.

    Args:
        path: Output path (should end with _depth.png)
        depth: Depth array (float32 or uint16)
        method: Quantization method ("p1p99", "p0.5p99.5", "minmax")
        debug_verify: If True, read back and verify shape/dtype

    Returns:
        Tuple of (p1, p99, DepthScalingStats)

    Raises:
        ValueError: If depth shape is invalid
    """
    # Validate input
    if depth.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D depth, got shape {depth.shape}")

    # Handle 3D depth (take first channel)
    if depth.ndim == 3:
        if depth.shape[2] != 1:
            logger.warning(
                f"Depth has {depth.shape[2]} channels, taking first channel only. V2 requires single-channel depth."
            )
        depth = depth[:, :, 0]

    # Convert to float32 for statistics
    depth_f32 = depth.astype(np.float32)

    # Compute invalid fraction BEFORE cleaning
    invalid_mask = ~np.isfinite(depth_f32)
    invalid_frac = float(invalid_mask.sum() / depth_f32.size)

    # Clean invalid values (replace with median)
    if invalid_mask.any():
        valid_depth = depth_f32[~invalid_mask]
        if valid_depth.size > 0:
            median_value = float(np.median(valid_depth))
            depth_f32[invalid_mask] = median_value
            logger.warning(f"Replaced {invalid_mask.sum()} invalid values with median {median_value:.3f}")
        else:
            # All invalid: use zeros
            depth_f32 = np.zeros_like(depth_f32)
            logger.warning("All depth values invalid, using zeros")

    # Determine percentiles
    if method == "p1p99":
        p_low_percentile, p_high_percentile = 1.0, 99.0
    elif method == "p0.5p99.5":
        p_low_percentile, p_high_percentile = 0.5, 99.5
    elif method == "minmax":
        p_low_percentile, p_high_percentile = 0.0, 100.0
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # Compute percentile values
    p1 = float(np.percentile(depth_f32, p_low_percentile))
    p99 = float(np.percentile(depth_f32, p_high_percentile))

    # Compute clipping fractions
    clipped_low_frac = float((depth_f32 < p1).sum() / depth_f32.size)
    clipped_high_frac = float((depth_f32 > p99).sum() / depth_f32.size)

    # Quantize to uint16
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
    img = Image.fromarray(depth_u16, mode="I;16")
    img.save(str(path))

    logger.info(f"Wrote depth to {path} (shape={depth_u16.shape}, dtype=uint16)")

    # Optional verification
    if debug_verify:
        verify_depth = np.array(Image.open(path))
        assert verify_depth.shape == depth_u16.shape, f"Shape mismatch: wrote {depth_u16.shape}, read {verify_depth.shape}"
        assert verify_depth.dtype == np.uint16, f"Dtype mismatch: expected uint16, got {verify_depth.dtype}"
        logger.debug(f"Verified depth write: shape={verify_depth.shape}, dtype={verify_depth.dtype}")

    # Create detailed statistics
    stats = DepthScalingStats(
        method=method,
        p_low_percentile=p_low_percentile,
        p_high_percentile=p_high_percentile,
        v_low_value=p1,
        v_high_value=p99,
        clipped_low_frac=clipped_low_frac,
        clipped_high_frac=clipped_high_frac,
        invalid_frac=invalid_frac,
    )

    return p1, p99, stats


def atomic_write_depth_u16_png(
    path: Path,
    depth: np.ndarray,
    method: str = "p1p99",
    debug_verify: bool = False,
) -> Tuple[float, float]:
    """Write depth with atomic rename to prevent partial files on crash.

    This ensures that if the process crashes during write, the output
    directory will not contain corrupt/partial files. Uses write-to-temp
    then atomic rename pattern.

    Args:
        path: Final output path
        depth: Depth array to write (float32 or uint16)
        method: Quantization method ("p1p99", "p0.5p99.5", "minmax")
        debug_verify: Enable read-back verification (slower)

    Returns:
        Tuple of (p1, p99) percentile values used for quantization

    Raises:
        ValueError: If depth is invalid
        IOError: If write fails

    Notes:
        - Temp file is written in same directory (same filesystem)
        - os.replace() provides atomic rename on POSIX systems
        - Cleanup is guaranteed via finally block
    """
    path = Path(path)

    # Ensure parent directory exists BEFORE temp file write
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in SAME directory (ensures same filesystem)
    tmp_path = path.with_suffix(".tmp.png")

    try:
        # Write depth to temp file
        p1, p99 = write_depth_u16_png(
            tmp_path,
            depth,
            method=method,
            debug_verify=False,  # Don't verify temp file
        )

        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        # Using os.replace() for cross-platform compatibility
        os.replace(str(tmp_path), str(path))

        # Optional verification on final file
        if debug_verify:
            verify_depth = np.array(Image.open(path))
            assert verify_depth.shape == depth.shape[:2], (
                f"Shape mismatch: expected {depth.shape[:2]}, got {verify_depth.shape}"
            )
            assert verify_depth.dtype == np.uint16, f"Dtype mismatch: expected uint16, got {verify_depth.dtype}"
            logger.debug(f"Verified depth write: {path}")

        logger.debug(f"Atomically wrote depth to {path}")
        return p1, p99

    except Exception as e:
        # Clean up partial write
        if tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.debug(f"Cleaned up partial write: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up {tmp_path}: {cleanup_error}")
        raise IOError(f"Failed to write depth to {path}: {e}") from e


def atomic_write_depth_u16_png_with_stats(
    path: Path,
    depth: np.ndarray,
    method: str = "p1p99",
    debug_verify: bool = False,
) -> Tuple[float, float, DepthScalingStats]:
    """Write depth atomically with detailed scaling statistics.

    Combines atomic write pattern with enhanced provenance statistics.

    Args:
        path: Final output path
        depth: Depth array to write (float32 or uint16)
        method: Quantization method ("p1p99", "p0.5p99.5", "minmax")
        debug_verify: Enable read-back verification (slower)

    Returns:
        Tuple of (p1, p99, DepthScalingStats)

    Raises:
        ValueError: If depth is invalid
        IOError: If write fails
    """
    path = Path(path)

    # Ensure parent directory exists BEFORE temp file write
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in SAME directory (ensures same filesystem)
    tmp_path = path.with_suffix(".tmp.png")

    try:
        # Write depth to temp file with stats
        p1, p99, stats = write_depth_u16_png_with_stats(
            tmp_path,
            depth,
            method=method,
            debug_verify=False,  # Don't verify temp file
        )

        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        os.replace(str(tmp_path), str(path))

        # Optional verification on final file
        if debug_verify:
            verify_depth = np.array(Image.open(path))
            assert verify_depth.shape == depth.shape[:2], (
                f"Shape mismatch: expected {depth.shape[:2]}, got {verify_depth.shape}"
            )
            assert verify_depth.dtype == np.uint16, f"Dtype mismatch: expected uint16, got {verify_depth.dtype}"
            logger.debug(f"Verified depth write: {path}")

        logger.debug(f"Atomically wrote depth to {path}")
        return p1, p99, stats

    except Exception as e:
        # Clean up partial write
        if tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.debug(f"Cleaned up partial write: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up {tmp_path}: {cleanup_error}")
        raise IOError(f"Failed to write depth to {path}: {e}") from e
