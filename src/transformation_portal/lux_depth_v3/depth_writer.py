"""Depth map writer with atomic operations and statistics.

Provides atomic write operations for 16-bit depth maps with statistics.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import tempfile
import shutil
import numpy as np

logger = logging.getLogger(__name__)

# Try importing dependencies with graceful fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available, install with: pip install opencv-python")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, install with: pip install Pillow")


def atomic_write_depth_u16_png_with_stats(
    output_path: Path,
    depth_map: np.ndarray,
    method: str = "u16",
    debug_verify: bool = False,
    **kwargs
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    """Atomically write depth map as 16-bit PNG with statistics.

    Args:
        output_path: Output file path
        depth_map: Depth map as numpy array (float32, normalized [0, 1])
        method: Quantization method ("u16", "none", etc.)
        debug_verify: Whether to verify write integrity
        **kwargs: Additional arguments

    Returns:
        Tuple of (output_path, verification_path, statistics_dict)
    """
    # Ensure output_path is a Path object
    output_path = Path(output_path)

    # Calculate statistics before conversion
    stats = {
        "min": float(np.min(depth_map)),
        "max": float(np.max(depth_map)),
        "mean": float(np.mean(depth_map)),
        "std_dev": float(np.std(depth_map)),
        "shape": depth_map.shape,
        "dtype": str(depth_map.dtype),
        "method": method,
    }

    # Convert to 16-bit unsigned integer
    if method == "u16":
        # Normalize to [0, 65535] range
        depth_u16 = (depth_map * 65535).astype(np.uint16)
    elif method == "none":
        # Use as-is (assume already uint16)
        if depth_map.dtype != np.uint16:
            # Auto-convert if needed
            depth_u16 = (depth_map * 65535).astype(np.uint16)
        else:
            depth_u16 = depth_map
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first (atomic operation)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".png",
        dir=output_path.parent,
        prefix=f".{output_path.stem}_"
    )

    try:
        # Close the file descriptor (we'll use the path)
        import os
        os.close(temp_fd)

        # Write depth map
        if CV2_AVAILABLE:
            # Prefer cv2 for 16-bit PNG writing
            success = cv2.imwrite(str(temp_path), depth_u16)
            if not success:
                raise IOError(f"Failed to write depth map to {temp_path}")
        elif PIL_AVAILABLE:
            # Fallback to PIL
            img = Image.fromarray(depth_u16, mode='I;16')
            img.save(temp_path)
        else:
            raise ImportError(
                "Either opencv-python or Pillow is required for depth writing. "
                "Install with: pip install opencv-python or pip install Pillow"
            )

        # Atomic rename
        shutil.move(str(temp_path), str(output_path))

        logger.info("Wrote depth map to %s (method=%s)", output_path, method)

    except Exception as e:
        # Clean up temp file on error
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        raise IOError(f"Failed to write depth map: {e}") from e

    # Verification step
    verification_path = None
    if debug_verify:
        # Read back and verify
        try:
            depth_verify = read_depth_u16_png(output_path)

            # Check shape matches
            if depth_verify.shape != depth_u16.shape:
                raise ValueError(
                    f"Verification failed: shape mismatch "
                    f"(expected {depth_u16.shape}, got {depth_verify.shape})"
                )

            # Check values match (allow small rounding errors)
            max_diff = np.max(np.abs(depth_verify.astype(float) - depth_u16.astype(float)))
            if max_diff > 1.0:
                raise ValueError(
                    f"Verification failed: max difference {max_diff} exceeds threshold"
                )

            # Write verification report
            verification_path = output_path.parent / f"{output_path.stem}_verify.txt"
            with open(verification_path, 'w') as f:
                f.write("Verification successful\n")
                f.write(f"Output: {output_path}\n")
                f.write(f"Shape: {depth_verify.shape}\n")
                f.write(f"Max difference: {max_diff}\n")

            logger.info("Depth map verification passed: %s", output_path)

        except Exception as e:
            logger.error("Depth map verification failed: %s", e)
            raise

    return output_path, verification_path, stats


def read_depth_u16_png(depth_path: Path) -> np.ndarray:
    """Read depth map from 16-bit PNG.

    Args:
        depth_path: Path to depth map PNG

    Returns:
        Depth map as numpy array (uint16)
    """
    depth_path = Path(depth_path)

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    # Read depth map
    if CV2_AVAILABLE:
        # Prefer cv2 for 16-bit PNG reading
        depth_map = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            raise IOError(f"Failed to read depth map from {depth_path}")
    elif PIL_AVAILABLE:
        # Fallback to PIL
        img = Image.open(depth_path)
        depth_map = np.array(img)
    else:
        raise ImportError(
            "Either opencv-python or Pillow is required for depth reading. "
            "Install with: pip install opencv-python or pip install Pillow"
        )

    # Ensure uint16
    if depth_map.dtype != np.uint16:
        logger.warning(
            "Depth map has dtype %s, expected uint16. Converting...",
            depth_map.dtype
        )
        depth_map = depth_map.astype(np.uint16)

    return depth_map
