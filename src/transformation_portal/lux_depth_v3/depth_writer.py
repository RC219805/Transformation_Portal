"""Depth map writer with atomic operations and statistics.

Provides robust, atomic depth map I/O with 16-bit precision:
- Atomic writes via shared atomic write primitives
- Statistics calculation on original float data
- Optional verification after write
- Read/write cycle preserves precision within quantization error
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import numpy as np

from .io_atomic import atomic_temp_file

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthWriteStats:
    """Statistics from depth map write operation.

    Provides _asdict() for backward compatibility with orchestrator.
    """
    min: float
    max: float
    mean: float
    std: float
    shape: tuple[int, ...]
    dtype: str
    method: str

    def _asdict(self) -> dict:
        """Return stats as dict (orchestrator compatibility)."""
        return asdict(self)


def atomic_write_depth_u16_png_with_stats(
    output_path: Path,
    depth_map: np.ndarray,
    method: str = "u16",
    debug_verify: bool = False,
    **kwargs
) -> tuple[Path, Optional[Path], DepthWriteStats]:
    """Atomically write depth map as 16-bit PNG with statistics.

    Performs safe atomic write via temporary file + rename.
    Calculates statistics on the raw float data before quantization.

    Args:
        output_path: Output file path
        depth_map: Depth map as numpy array (float32, range [0.0, 1.0])
        method: Quantization method (only "u16" supported)
        debug_verify: Whether to verify write integrity by reading back
        **kwargs: Additional arguments (reserved for future use)

    Returns:
        Tuple of (output_path, verification_path_or_none, statistics)

    Raises:
        ImportError: If opencv-python not installed
        ValueError: If unsupported quantization method specified
        IOError: If write or verification fails
    """
    if not HAS_CV2:
        raise ImportError(
            "opencv-python required for depth_writer. Install with: pip install opencv-python"
        )

    # Normalize legacy/config values
    # EnhanceConfig defaults to "none", which means "default behavior" (u16 for this writer)
    if method in (None, "", "none"):
        method = "u16"

    # Validate method
    if method != "u16":
        raise ValueError(
            f"Unsupported depth quantization method: {method!r}. Only 'u16' is supported."
        )

    # 1. Calculate statistics on original data
    stats = DepthWriteStats(
        min=float(np.min(depth_map)),
        max=float(np.max(depth_map)),
        mean=float(np.mean(depth_map)),
        std=float(np.std(depth_map)),
        shape=tuple(depth_map.shape),
        dtype=str(depth_map.dtype),
        method=method
    )

    # 2. Normalize to 16-bit (0-65535)
    # Assumes input is 0.0-1.0 float. Clip just in case.
    depth_clipped = np.clip(depth_map, 0.0, 1.0)
    depth_u16 = (depth_clipped * 65535.0).astype(np.uint16)

    # 3. Atomic Write using shared helper
    # cv2.imwrite requires a file path, so we use atomic_temp_file context manager
    try:
        # cv2.imwrite is path-based and creates file with umask permissions
        with atomic_temp_file(output_path, suffix=".png", create_file=False) as temp_path:
            # Use explicit PNG compression parameters
            success = cv2.imwrite(
                str(temp_path),
                depth_u16,
                [cv2.IMWRITE_PNG_COMPRESSION, 3]  # Compression level 0-9
            )
            if not success:
                raise IOError(f"cv2.imwrite returned False for {temp_path}")
            # atomic_temp_file will handle os.replace on success

    except Exception as e:
        # atomic_temp_file handles cleanup, but we re-raise with context
        raise IOError(f"Failed to write depth map to {output_path}") from e

    # 4. Verification (Optional)
    verification_path = None
    if debug_verify:
        # Read back and compare
        check_img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if check_img is None:
            raise IOError(f"Verification failed: Could not read back {output_path}")

        # Check for bit-exactness
        if not np.array_equal(depth_u16, check_img):
            logger.warning(
                f"Verification WARNING: Readback of {output_path} does not match written data!"
            )
            # Note: Compression shouldn't change pixel values for PNG
        else:
            logger.debug(f"Verification successful for {output_path}")

    return output_path, verification_path, stats


def read_depth_u16_png(depth_path: Path) -> np.ndarray:
    """Read depth map from 16-bit PNG.

    Returns normalized float32 array [0.0, 1.0].
    Falls back to PIL if opencv-python is not available.

    Args:
        depth_path: Path to depth map PNG

    Returns:
        Depth map as float32 numpy array, normalized to [0.0, 1.0]

    Raises:
        FileNotFoundError: If depth file doesn't exist
        IOError: If read fails
    """
    if not Path(depth_path).exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    # Prefer opencv for performance, fallback to PIL for CI compatibility
    if HAS_CV2:
        # Read raw 16-bit with opencv
        img_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if img_u16 is None:
            raise IOError(f"Failed to read depth map: {depth_path}")
        img_f32 = img_u16.astype(np.float32) / 65535.0
    else:
        # Fallback to PIL (CI-compatible)
        from PIL import Image
        logger.debug(f"Using PIL fallback for PNG read (opencv not available): {depth_path}")
        img = Image.open(depth_path)
        img_array = np.array(img)

        # Normalize based on bit depth
        if img_array.dtype == np.uint16:
            img_f32 = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            img_f32 = img_array.astype(np.float32) / 255.0
        else:
            # Already float, ensure [0, 1] range
            img_f32 = img_array.astype(np.float32)
            maxv = float(np.nanmax(img_f32)) if img_f32.size else 1.0
            if maxv > 1.0:
                img_f32 /= maxv

    return img_f32
