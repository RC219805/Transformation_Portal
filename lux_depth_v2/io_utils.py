from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import tifffile  # type: ignore
except Exception:  # pragma: no cover
    tifffile = None  # type: ignore


# Slice 3 PR-2: Compression validation constants
VALID_TIFF_COMPRESSION = frozenset({None, "lzw", "zstd", "deflate"})


def validate_tiff_compression(compression: Optional[str]) -> None:
    """
    Validate TIFF compression parameter.

    Raises:
        ValueError: If compression is not in VALID_TIFF_COMPRESSION
    """
    if compression not in VALID_TIFF_COMPRESSION:
        valid_str = ", ".join(sorted(str(c) for c in VALID_TIFF_COMPRESSION if c))
        raise ValueError(f"tiff_compression={compression!r} is invalid. Valid options: {valid_str}, or None")


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int
    dtype: str
    bit_depth: int


@dataclass
class DepthInfo:
    """Metadata about a loaded depth file (before/after coercion + normalization)."""

    file_format: str  # e.g. 'png', 'tif', 'tiff'
    source_dtype: str  # dtype as loaded from disk (e.g. 'uint16', 'uint8')
    dtype: str  # dtype after coercion (expected 'uint16')
    shape: Tuple[int, int]  # (H, W)
    channels: int  # 1 for grayscale, >1 if source was multi-channel
    channel_collapsed: bool  # True if we collapsed identical channels to 2D
    u16_min: int
    u16_max: int
    p1: float  # 1st percentile used for robust normalization
    p99: float  # 99th percentile used for robust normalization


def ensure_deps() -> None:
    if cv2 is None or tifffile is None:
        raise RuntimeError("Missing deps. Install: opencv-python tifffile numpy")


def _is_tiff(p: Path) -> bool:
    return p.suffix.lower() in (".tif", ".tiff")


def read_rgb_any(path: Path) -> Tuple[np.ndarray, ImageInfo]:
    """Read common image formats into float32 RGB 0..1, shape HxWx3."""
    ensure_deps()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if _is_tiff(p):
        arr = tifffile.imread(str(p))
        if arr.ndim == 2:
            # grayscale to RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        # tiff may be RGB already
        if arr.dtype == np.uint16:
            rgb01 = (arr.astype(np.float32) / 65535.0).clip(0.0, 1.0)
            info = ImageInfo(p, arr.shape[1], arr.shape[0], "uint16", 16)
            return rgb01, info
        if arr.dtype == np.uint8:
            rgb01 = (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
            info = ImageInfo(p, arr.shape[1], arr.shape[0], "uint8", 8)
            return rgb01, info
        # other types
        arrf = arr.astype(np.float32)
        mx = float(np.max(arrf)) if arrf.size else 1.0
        rgb01 = (arrf / max(mx, 1e-6)).clip(0.0, 1.0)
        info = ImageInfo(p, arr.shape[1], arr.shape[0], str(arr.dtype), 16)
        return rgb01, info

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {p}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] > 3:
        img = img[:, :, :3]

    # OpenCV is BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype == np.uint8:
        rgb01 = (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
        info = ImageInfo(p, img.shape[1], img.shape[0], "uint8", 8)
        return rgb01, info
    if img.dtype == np.uint16:
        rgb01 = (img.astype(np.float32) / 65535.0).clip(0.0, 1.0)
        info = ImageInfo(p, img.shape[1], img.shape[0], "uint16", 16)
        return rgb01, info

    imgf = img.astype(np.float32)
    mx = float(np.max(imgf)) if imgf.size else 1.0
    rgb01 = (imgf / max(mx, 1e-6)).clip(0.0, 1.0)
    info = ImageInfo(p, img.shape[1], img.shape[0], str(img.dtype), 16)
    return rgb01, info


def _collapse_depth_channels(d: np.ndarray, path: Path) -> Tuple[np.ndarray, int, bool]:
    """Ensure depth is 2D. If multi-channel, only accept identical channels.

    Handles both channel-last (H,W,C) and channel-first (C,H,W) layouts.
    For RGB/RGBA, requires all color channels to be identical (alpha ignored).
    Rejects stacks or other unexpected shapes to avoid silent misinterpretation.
    """
    if d.ndim == 2:
        return d, 1, False
    if d.ndim != 3:
        raise ValueError(
            f"Depth must be 2D or 3D, got shape={d.shape}: {path}. This looks like a stack or unsupported layout."
        )

    # Try channel-last (H,W,C) - most common for PNG/TIFF
    if d.shape[-1] in (1, 3, 4):
        channels = d.shape[-1]
        if channels == 1:
            return d[..., 0], 1, True
        # RGB/RGBA: verify channels are identical (grayscale saved as RGB)
        c0 = d[..., 0]
        if not np.array_equal(c0, d[..., 1]) or not np.array_equal(c0, d[..., 2]):
            raise ValueError(
                f"Depth must be single-channel; got multi-channel with differing channels: {path} "
                f"(shape={d.shape}, dtype={d.dtype}). "
                "This looks like an RGB image or colormap, not a depth map."
            )
        return c0, channels, True

    # Try channel-first (C,H,W) - sometimes seen in TIFF
    if d.shape[0] in (1, 3, 4):
        channels = d.shape[0]
        if channels == 1:
            return d[0, ...], 1, True
        # RGB/RGBA: verify channels are identical
        c0 = d[0, ...]
        if not np.array_equal(c0, d[1, ...]) or not np.array_equal(c0, d[2, ...]):
            raise ValueError(
                f"Depth must be single-channel; got multi-channel with differing channels: {path} "
                f"(shape={d.shape}, dtype={d.dtype}). "
                "This looks like an RGB image or colormap, not a depth map."
            )
        return c0, channels, True

    raise ValueError(
        f"Depth must be 2D or a 1/3/4-channel image, got shape {d.shape}: {path}. "
        "Unexpected channel dimension - cannot determine layout."
    )


def read_depth_u16_with_info(
    path: Path,
    expected_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, DepthInfo]:
    """
    Read depth from TIFF/PNG and return (depth01, info).

    Depth is normalized to float32 0..1 using robust percentiles (p1, p99).
    """
    ensure_deps()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    if _is_tiff(p):
        d = tifffile.imread(str(p))
        file_format = ext.lstrip(".")
    elif ext == ".png":
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise RuntimeError(f"Failed to read depth PNG: {p}")
        file_format = "png"
    else:
        raise ValueError(f"Unsupported depth file extension '{p.suffix}' for {p}")

    source_dtype = str(d.dtype)
    d, channels, channel_collapsed = _collapse_depth_channels(d, p)
    if d.ndim != 2:
        raise ValueError(f"Depth must be 2D after channel handling, got shape={d.shape} for {p}")

    # Handle uint8 depth with warning (common export mistake)
    if d.dtype == np.uint8:
        warnings.warn(
            f"Depth map is 8-bit (uint8) for {p}. "
            "Upscaling to 16-bit will preserve quantization artifacts. "
            "Re-export depth as 16-bit PNG from Depth Anything 3 for best results.",
            RuntimeWarning,
            stacklevel=2,
        )
        d = d.astype(np.uint16) * 257  # 0-255 -> 0-65535
    # Coerce other integer types to uint16
    elif d.dtype != np.uint16:
        # Reject obviously wrong types
        if np.issubdtype(d.dtype, np.floating):
            raise TypeError(
                f"Depth must be uint16/uint8 integer, got floating point {d.dtype}: {p}. "
                "Depth maps should be integer grayscale."
            )
        # Cache min/max for performance (avoid multiple passes over large arrays)
        d_min, d_max = d.min(), d.max()
        if np.issubdtype(d.dtype, np.signedinteger):
            if d_min < 0:
                raise ValueError(
                    f"Depth has negative values (dtype={d.dtype}, min={d_min}): {p}. Depth maps must be non-negative."
                )
        # Prevent overflow when casting to uint16
        max_val = int(d_max)
        if max_val > 65535:
            raise ValueError(
                f"Depth values exceed uint16 range (max={max_val} > 65535) for {p}. "
                f"dtype={d.dtype}. Cannot safely cast to 16-bit."
            )
        # Cast other integer types (uint32, int16, etc.) to uint16
        d = d.astype(np.uint16)
    if expected_hw is not None:
        eh, ew = expected_hw
        if d.shape != (eh, ew):
            raise ValueError(f"Depth shape mismatch for {p}: got {d.shape}, expected {(eh, ew)}")

    # robust percentile normalization (like V1)
    df = d.astype(np.float32)
    if df.size == 0:
        depth01 = np.zeros_like(df, dtype=np.float32)
        p1 = 0.0
        p99 = 0.0
    else:
        p1 = float(np.percentile(df, 1.0))
        p99 = float(np.percentile(df, 99.0))
        if p99 <= p1 + 1.0:
            depth01 = np.zeros_like(df, dtype=np.float32)
        else:
            depth01 = ((df - p1) / (p99 - p1)).clip(0.0, 1.0).astype(np.float32)

    # Cache min/max for DepthInfo (reuse or compute once)
    d_min_final = int(d.min()) if d.size else 0
    d_max_final = int(d.max()) if d.size else 0

    info = DepthInfo(
        file_format=file_format,
        source_dtype=source_dtype,
        dtype=str(d.dtype),
        shape=(int(d.shape[0]), int(d.shape[1])),
        channels=int(channels),
        channel_collapsed=bool(channel_collapsed),
        u16_min=d_min_final,
        u16_max=d_max_final,
        p1=float(p1),
        p99=float(p99),
    )
    return depth01, info


def read_depth_u16(path: Path, expected_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read depth TIFF/PNG into float32 0..1 normalized."""
    depth01, _ = read_depth_u16_with_info(path, expected_hw=expected_hw)
    return depth01


def read_mask_any(path: Path) -> np.ndarray:
    """Load a single-channel mask into float32 [0,1]. Supports TIFF/PNG/JPG."""
    ensure_deps()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if _is_tiff(p):
        m = tifffile.imread(str(p))
        if m.ndim == 3:
            m = m[..., 0]
        if m.dtype == np.uint16:
            return (m.astype(np.float32) / 65535.0).clip(0.0, 1.0)
        if m.dtype == np.uint8:
            return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)
        mf = m.astype(np.float32)
        mx = float(np.max(mf)) if mf.size else 1.0
        return (mf / max(mx, 1e-6)).clip(0.0, 1.0)

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read mask: {p}")
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 65535.0).clip(0.0, 1.0)
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def atomic_write_rgb16_tiff(path: Path, rgb01: np.ndarray, compression: str = "deflate") -> None:
    """Write uint16 RGB TIFF atomically."""
    ensure_deps()
    validate_tiff_compression(compression)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rgb01 = np.asarray(rgb01, dtype=np.float32)
    rgb_u16 = (np.clip(rgb01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)

    tmp = p.with_suffix(p.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    tifffile.imwrite(str(tmp), rgb_u16, photometric="rgb", compression=compression, metadata=None)
    os.replace(str(tmp), str(p))


def atomic_write_png8(path: Path, rgb01: np.ndarray, compression: int = 6) -> None:
    """
    Write 8-bit PNG atomically with configurable compression (M1.1).

    Args:
        path: Output PNG path
        rgb01: RGB float32 array in [0, 1]
        compression: PNG compression level (0-9, default 6)
                    0 = no compression (fastest, largest files)
                    1 = best speed
                    6 = balanced (default)
                    9 = best compression (slowest)
    """
    ensure_deps()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    # Keep .png extension for OpenCV to recognize the format
    tmp = p.parent / (p.stem + ".tmp" + p.suffix)
    success = cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not success:
        raise RuntimeError(f"Failed to write PNG to {tmp}")
    os.replace(str(tmp), str(p))


def atomic_write_jpg8(path: Path, rgb01: np.ndarray, quality: int = 92) -> None:
    ensure_deps()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    # Keep .jpg extension for OpenCV to recognize the format
    tmp = p.parent / (p.stem + ".tmp" + p.suffix)
    success = cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not success:
        raise RuntimeError(f"Failed to write JPEG to {tmp}")
    os.replace(str(tmp), str(p))


# ------------------------------------------------------------------ #
# Slice 3 PR-2: Tiled BigTIFF + non-atomic legacy writer
# ------------------------------------------------------------------ #


def write_tiff16_legacy(path: Path, rgb01: np.ndarray, compression: Optional[str] = "deflate") -> None:
    """
    Write uint16 RGB TIFF (non-atomic, direct write).

    Slice 3 PR-2: Legacy behavior for backward compatibility.
    This is the default when use_atomic_image_writes=False.

    Args:
        path: Output TIFF path
        rgb01: RGB float32 array in [0, 1], shape (H, W, 3)
        compression: TIFF compression ("deflate", "lzw", "zstd", None)

    Raises:
        ValueError: If compression is invalid
    """
    ensure_deps()
    validate_tiff_compression(compression)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rgb01 = np.asarray(rgb01, dtype=np.float32)
    rgb_u16 = (np.clip(rgb01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)

    tifffile.imwrite(str(p), rgb_u16, photometric="rgb", compression=compression, metadata=None)


def write_tiff16_tiled(
    path: Path,
    rgb01: np.ndarray,
    tile_size: int,
    compression: Optional[str] = None,
) -> None:
    """
    Write uint16 RGB TIFF with tiling for large images (Slice 3 PR-2 optimization).

    Tiling reduces peak memory and improves write performance for large TIFFs.
    Automatically uses BigTIFF format for files >2GB uncompressed.

    Args:
        path: Output TIFF path
        rgb01: RGB float32 array in [0, 1], shape (H, W, 3)
        tile_size: Tile dimension in pixels (e.g., 512), must be positive int
        compression: TIFF compression ("lzw", "zstd", "deflate", None)

    Raises:
        ValueError: If tile_size is invalid or compression unsupported
    """
    ensure_deps()
    validate_tiff_compression(compression)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint16
    rgb01 = np.asarray(rgb01, dtype=np.float32)
    rgb_u16 = (np.clip(rgb01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)

    # Validate tile_size
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError(f"tile_size must be a positive int, got {tile_size!r}")

    h, w = rgb_u16.shape[:2]
    if tile_size > max(h, w):
        raise ValueError(f"tile_size={tile_size} cannot exceed max image dimension={max(h, w)}")

    # Determine if BigTIFF is needed (>2GB uncompressed size threshold)
    channels = rgb_u16.shape[2] if rgb_u16.ndim == 3 else 1
    uncompressed_size = h * w * channels * 2  # 2 bytes per uint16
    use_bigtiff = uncompressed_size > (2 * 1024 * 1024 * 1024)  # >2GB

    # Write tiled TIFF
    tifffile.imwrite(
        str(p),
        rgb_u16,
        bigtiff=use_bigtiff,
        tile=(tile_size, tile_size),
        compression=compression,
        photometric="rgb" if channels == 3 else None,
        planarconfig="contig",  # Interleaved RGB
        metadata=None,
    )
