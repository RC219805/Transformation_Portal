from __future__ import annotations

import os
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


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int
    dtype: str
    bit_depth: int


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


def read_depth_u16(path: Path) -> np.ndarray:
    """Read 16-bit depth TIFF into float32 0..1 normalized."""
    ensure_deps()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    d = tifffile.imread(str(p))
    if d.ndim != 2:
        d = d[..., 0]
    if d.dtype != np.uint16:
        d = d.astype(np.uint16)
    # robust percentile normalization (like V1)
    df = d.astype(np.float32)
    lo, hi = np.percentile(df, 1.0), np.percentile(df, 99.0)
    if hi <= lo + 1.0:
        return np.zeros_like(df, dtype=np.float32)
    return ((df - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)


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


def atomic_write_png8(path: Path, rgb01: np.ndarray) -> None:
    ensure_deps()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    tmp = p.with_suffix(p.suffix + ".tmp")
    cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 7])
    os.replace(str(tmp), str(p))


def atomic_write_jpg8(path: Path, rgb01: np.ndarray, quality: int = 92) -> None:
    ensure_deps()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    tmp = p.with_suffix(p.suffix + ".tmp")
    cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    os.replace(str(tmp), str(p))
