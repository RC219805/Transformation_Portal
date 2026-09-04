"""Depth map writer with atomic operations and statistics.

Provides robust, atomic depth map I/O with 16-bit precision:
- Atomic writes via shared atomic write primitives
- Statistics calculation on original float data
- Optional verification after write
- Read/write cycle preserves precision within quantization error
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .io_atomic import atomic_temp_file

try:
    import cv2 as opencv

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    opencv = None  # type: ignore

logger = logging.getLogger(__name__)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_DEPTH_PNG_DECODED_PIXELS = 67_108_864


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
    encoding: str = "u16"
    normalization: Optional[dict[str, Any]] = None
    encoded_min: Optional[int] = None
    encoded_max: Optional[int] = None
    encoded_unique_values: Optional[int] = None

    def _asdict(self) -> dict:
        """Return stats as dict (orchestrator compatibility)."""
        return asdict(self)


def _finite_float_array(depth_map: np.ndarray) -> np.ndarray:
    """Return a float32 array with non-finite values replaced by 0."""
    arr = np.asarray(depth_map, dtype=np.float32)
    if np.isfinite(arr).all():
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def normalize_depth_for_u16_png(depth_map: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the deterministic normalized preview encoded into depth_u16 PNG."""
    arr = _finite_float_array(depth_map)
    finite = np.isfinite(np.asarray(depth_map))
    finite_values = np.asarray(depth_map, dtype=np.float32)[finite]
    if finite_values.size == 0:
        normalized = np.zeros_like(arr, dtype=np.float32)
        return normalized, {
            "mode": "no_finite_values",
            "source_min": None,
            "source_max": None,
            "encoded_min": 0.0,
            "encoded_max": 0.0,
        }

    source_min = float(np.min(finite_values))
    source_max = float(np.max(finite_values))
    if source_min >= 0.0 and source_max <= 1.0:
        normalized = np.clip(arr, 0.0, 1.0)
        return normalized.astype(np.float32, copy=False), {
            "mode": "identity_0_1",
            "source_min": source_min,
            "source_max": source_max,
            "encoded_min": float(np.min(normalized)),
            "encoded_max": float(np.max(normalized)),
        }

    if source_min >= -0.5 and source_max <= 1.5:
        normalized = np.clip(arr, 0.0, 1.0)
        return normalized.astype(np.float32, copy=False), {
            "mode": "clip_0_1",
            "source_min": source_min,
            "source_max": source_max,
            "encoded_min": float(np.min(normalized)),
            "encoded_max": float(np.max(normalized)),
        }

    p01 = float(np.percentile(finite_values, 1.0))
    p99 = float(np.percentile(finite_values, 99.0))
    if not np.isfinite(p01) or not np.isfinite(p99) or p99 <= p01:
        normalized = np.zeros_like(arr, dtype=np.float32)
        return normalized, {
            "mode": "degenerate_percentile_1_99",
            "source_min": source_min,
            "source_max": source_max,
            "percentile_1": p01 if np.isfinite(p01) else None,
            "percentile_99": p99 if np.isfinite(p99) else None,
            "encoded_min": 0.0,
            "encoded_max": 0.0,
        }

    normalized = np.clip((arr - p01) / (p99 - p01), 0.0, 1.0).astype(np.float32, copy=False)
    return normalized, {
        "mode": "percentile_1_99",
        "source_min": source_min,
        "source_max": source_max,
        "percentile_1": p01,
        "percentile_99": p99,
        "encoded_min": float(np.min(normalized)),
        "encoded_max": float(np.max(normalized)),
    }


def _count_u16_unique_values(depth_u16: np.ndarray) -> int:
    """Return exact u16 cardinality with bounded memory."""
    counts = np.bincount(depth_u16.reshape(-1), minlength=65536)
    return int(np.count_nonzero(counts))


def atomic_write_depth_u16_png_with_stats(
    output_path: Path,
    depth_map: np.ndarray,
    method: str = "u16",
    debug_verify: bool = False,
    compute_encoded_unique_values: bool = False,
    **kwargs: Any,
) -> tuple[Path, Optional[Path], DepthWriteStats]:
    """Atomically write depth map as 16-bit PNG with statistics.

    Performs safe atomic write via temporary file + rename.
    Calculates statistics on the raw float data before quantization.

    Args:
        output_path: Output file path
        depth_map: Depth map as numpy array (float32, range [0.0, 1.0])
        method: Quantization method (only "u16" supported)
        debug_verify: Whether to verify write integrity by reading back
        compute_encoded_unique_values: Whether to compute exact u16 cardinality.
            This scans the encoded image and is intended for APEX audit paths.
        **kwargs: Additional arguments (reserved for future use)

    Returns:
        Tuple of (output_path, verification_path_or_none, statistics)

    Raises:
        ImportError: If OpenCV is not installed
        ValueError: If unsupported quantization method specified
        IOError: If write or verification fails
    """
    if not HAS_CV2:
        raise ImportError(
            "OpenCV (cv2) required for depth_writer. "
            "Install with: pip install opencv-python-headless on Linux "
            "or pip install opencv-python on other platforms."
        )

    # Normalize legacy/config values
    # EnhanceConfig defaults to "none", which
    # means "default behavior" (u16 for this writer)
    if method in (None, "", "none"):
        method = "u16"

    # Validate method
    if method != "u16":
        raise ValueError("Unsupported depth quantization" f" method: {method!r}." " Only 'u16' is supported.")

    finite_depth = _finite_float_array(depth_map)

    # 1. Calculate statistics on original data
    stats = DepthWriteStats(
        min=float(np.min(finite_depth)),
        max=float(np.max(finite_depth)),
        mean=float(np.mean(finite_depth)),
        std=float(np.std(finite_depth)),
        shape=tuple(depth_map.shape),
        dtype=str(depth_map.dtype),
        method=method,
    )

    # 2. Normalize to 16-bit (0-65535)
    depth_normalized, normalization_stats = normalize_depth_for_u16_png(depth_map)
    depth_u16 = np.rint(depth_normalized * 65535.0).astype(np.uint16)
    stats = DepthWriteStats(
        min=stats.min,
        max=stats.max,
        mean=stats.mean,
        std=stats.std,
        shape=stats.shape,
        dtype=stats.dtype,
        method=stats.method,
        encoding="normalized_u16_png",
        normalization=normalization_stats,
        encoded_min=int(np.min(depth_u16)),
        encoded_max=int(np.max(depth_u16)),
        encoded_unique_values=(_count_u16_unique_values(depth_u16) if compute_encoded_unique_values else None),
    )

    # 3. Atomic Write using shared helper
    # cv2.imwrite requires a file path, so we
    # use atomic_temp_file context manager
    try:
        # cv2.imwrite is path-based and creates
        # file with umask permissions
        with atomic_temp_file(
            output_path,
            suffix=".png",
            create_file=False,
        ) as temp_path:
            # Use explicit PNG compression parameters
            # Compression level 0-9
            success = opencv.imwrite(
                str(temp_path),
                depth_u16,
                [opencv.IMWRITE_PNG_COMPRESSION, 3],
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
        check_img = opencv.imread(str(output_path), opencv.IMREAD_UNCHANGED)
        if check_img is None:
            raise IOError(f"Verification failed:" f" Could not read back" f" {output_path}")

        # Check for bit-exactness
        if not np.array_equal(depth_u16, check_img):
            logger.warning(
                "Verification WARNING:" " Readback of %s does not" " match written data!",
                output_path,
            )
            # Note: Compression shouldn't change pixel values for PNG
        else:
            logger.debug(f"Verification successful for {output_path}")

    return output_path, verification_path, stats


def _validated_depth_png_header(payload: bytes, *, max_decoded_pixels: int) -> tuple[int, int]:
    if isinstance(max_decoded_pixels, bool) or not isinstance(max_decoded_pixels, int) or max_decoded_pixels <= 0:
        raise ValueError("max_decoded_pixels must be a positive integer")
    if len(payload) < 29 or payload[:8] != _PNG_SIGNATURE:
        raise IOError("Depth map payload is not a PNG")
    if int.from_bytes(payload[8:12], "big") != 13 or payload[12:16] != b"IHDR":
        raise IOError("Depth map payload has no canonical PNG IHDR")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    bit_depth = payload[24]
    color_type = payload[25]
    if width <= 0 or height <= 0 or width * height > max_decoded_pixels:
        raise IOError(f"Depth map decoded pixels exceed the bounded limit of {max_decoded_pixels}")
    if bit_depth != 16 or color_type != 0:
        raise IOError("Depth map payload must be a 16-bit grayscale PNG")
    return height, width


def read_depth_u16_png_bytes(
    payload: bytes,
    *,
    max_decoded_pixels: int = MAX_DEPTH_PNG_DECODED_PIXELS,
) -> np.ndarray:
    """Decode a depth PNG from the exact supplied bytes.

    Returns a normalized float32 array using the same decoder-specific
    normalization as :func:`read_depth_u16_png`.
    """
    try:
        expected_shape = _validated_depth_png_header(payload, max_decoded_pixels=max_decoded_pixels)
        if HAS_CV2:
            encoded = np.frombuffer(payload, dtype=np.uint8)
            img_u16 = opencv.imdecode(encoded, opencv.IMREAD_UNCHANGED)
            if img_u16 is None:
                raise IOError("Failed to decode depth map payload")
            if img_u16.ndim != 2 or img_u16.dtype != np.uint16 or tuple(img_u16.shape) != expected_shape:
                raise IOError("Decoded depth map is not the declared 16-bit grayscale image")
            return img_u16.astype(np.float32) / 65535.0

        from PIL import Image

        with Image.open(BytesIO(payload)) as img:
            if img.size != (expected_shape[1], expected_shape[0]):
                raise IOError("Decoded depth map dimensions do not match PNG IHDR")
            img_array = np.array(img)

        if img_array.ndim != 2 or tuple(img_array.shape) != expected_shape or img_array.dtype.kind not in {"u", "i"}:
            raise IOError("Decoded depth map is not a 16-bit grayscale image")
        if img_array.size and (int(img_array.min()) < 0 or int(img_array.max()) > 65535):
            raise IOError("Decoded depth map values exceed the uint16 range")
        return img_array.astype(np.float32) / 65535.0
    except IOError:
        raise
    except Exception as exc:
        raise IOError("Failed to decode depth map payload") from exc


def read_depth_u16_png(depth_path: Path) -> np.ndarray:
    """Read depth map from 16-bit PNG.

    Returns normalized float32 array [0.0, 1.0].
    Falls back to PIL if OpenCV is not available.

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
        img_u16 = opencv.imread(str(depth_path), opencv.IMREAD_UNCHANGED)
        if img_u16 is None:
            raise IOError(f"Failed to read depth map: {depth_path}")
        img_f32 = img_u16.astype(np.float32) / 65535.0
    else:
        # Fallback to PIL (CI-compatible)
        from PIL import Image

        logger.debug(
            "Using PIL fallback for PNG read" " (opencv not available): %s",
            depth_path,
        )
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
