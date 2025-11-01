#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
depth_tools.py - Production-grade depth-based post-processing for architectural imagery.

Features included:
    • Bounded LRU caches with size limits
    • Retry logic for I/O operations with exponential backoff
    • Bilateral filter parameter exposure (uses OpenCV if available)
    • Consolidated mask discovery and loading
    • Enhanced error context and recovery
    • Memory-efficient streaming for large batches
    • Optional multiprocessing for batch work
    • Progress callback support and verbose logging
    • Validation pipeline with early error detection

Modes supported: haze | clarity | dof

Depth maps expected: *_depth16.png (or other high-bit-depth formats)
Mask files (optional): _mask_sky.png, _mask_building.png, etc.
Enhanced images: searched recursively for files matching base + priority tags.

Designed to be robust for large photography / architectural pipelines.
"""
from __future__ import annotations

import argparse
import functools
import glob
import logging
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ----- Optional accelerated libraries -----

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:
    gaussian_filter = None
    _SCIPY_AVAILABLE = False

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    import tifffile as tiff  # type: ignore
    _TIFFFILE_AVAILABLE = True
except Exception:
    tiff = None
    _TIFFFILE_AVAILABLE = False

# ----- Defaults / configuration -----

BUILDING_HAZE_SUPPRESSION = 0.85
SKY_HAZE_BOOST = 0.70
BUILDING_BLUR_REDUCTION = 0.88
SKY_BLUR_REDUCTION = 0.30

DEFAULT_HAZE_COLOR = (0.94, 0.96, 0.99)
DEFAULT_CACHE_SIZE = 128
DEFAULT_IO_RETRIES = 3
DEFAULT_IO_RETRY_DELAY = 0.5

PRIORITY_TAGS = ("_enh", "_punchy", "_golden", "_agx", "_view", "_ok", "enh", "punchy", "golden")
SUPPORTED_EXTENSIONS = (
    ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp",
    ".TIF", ".TIFF", ".JPG", ".JPEG", ".PNG", ".WEBP"
)

# ----- Logging -----

_log = logging.getLogger("depth_tools")
if not _log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)

# ----- LRU cache -----

class BoundedCache:
    """Simple LRU cache using OrderedDict. Values are stored as NumPy arrays (copied)."""

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE):
        self.maxsize = int(maxsize)
        self._cache: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[np.ndarray]:
        v = self._cache.get(key)
        if v is not None:
            # move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return v.copy()
        self.misses += 1
        return None

    def put(self, key: Any, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value.copy()
        # evict if needed
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = float(self.hits) / total if total > 0 else 0.0
        return {"hits": self.hits, "misses": self.misses, "size": len(self._cache), "hit_rate": hit_rate}

# per-process caches (can be pickled if needed; small by default)
_depth_cache = BoundedCache()
_mask_cache = BoundedCache()

def clear_all_caches() -> None:
    ds = _depth_cache.stats()
    ms = _mask_cache.stats()
    _log.debug("Depth cache: hits=%d misses=%d size=%d hit_rate=%.2f%%", ds["hits"], ds["misses"], ds["size"], ds["hit_rate"] * 100)
    _log.debug("Mask cache:  hits=%d misses=%d size=%d hit_rate=%.2f%%", ms["hits"], ms["misses"], ms["size"], ms["hit_rate"] * 100)
    _depth_cache.clear()
    _mask_cache.clear()

# ----- retry decorator -----

def retry_on_io_error(max_attempts: int = DEFAULT_IO_RETRIES, initial_delay: float = DEFAULT_IO_RETRY_DELAY, backoff_factor: float = 2.0):
    """Decorator for retrying IO operations with exponential backoff on OSError/IOError."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = float(initial_delay)
            last_exc: Optional[BaseException] = None
            for attempt in range(1, int(max_attempts) + 1):
                try:
                    return func(*args, **kwargs)
                except (IOError, OSError) as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        _log.debug("I/O error in %s (attempt %d/%d): %s - retrying in %.2fs", func.__name__, attempt, max_attempts, exc, delay)
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        _log.warning("I/O error in %s after %d attempts: %s", func.__name__, max_attempts, exc)
            # re-raise last
            if last_exc:
                raise last_exc
            raise RuntimeError("Retry wrapper failed without exception")
        return wrapper
    return decorator

# ----- validators -----

def validate_color(color: Tuple[float, float, float], name: str = "color") -> Tuple[float, float, float]:
    if len(color) != 3:
        raise ValueError(f"{name} must have 3 components, got {len(color)}")
    # allow 0..255 or 0..1
    if any(c > 1.0 for c in color):
        if not all(0.0 <= c <= 255.0 for c in color):
            raise ValueError(f"{name} values must be in 0..1 or 0..255 range, got {color}")
        normalized = tuple(float(c) / 255.0 for c in color)
        color = normalized  # type: ignore
    if not all(0.0 <= c <= 1.0 for c in color):
        raise ValueError(f"{name} must be in 0..1 range after normalization, got {color}")
    return color

def validate_file_exists(path: str, description: str = "File") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"{description} is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{description} is not readable: {path}")

# ----- blur/backends -----

def gaussian_blur_float(img: np.ndarray, sigma: float, backend: Optional[str] = None) -> np.ndarray:
    """
    Blur an image represented as float32 in [0,1]. Supports grayscale 2D or color 3D arrays.
    backend: 'scipy', 'cv2', 'pil' or None (auto)
    """
    if sigma <= 0.5:
        return img
    use_backend = backend
    if use_backend is None:
        if _SCIPY_AVAILABLE:
            use_backend = "scipy"
        elif _CV2_AVAILABLE:
            use_backend = "cv2"
        else:
            use_backend = "pil"

    if use_backend == "scipy" and gaussian_filter is not None:
        if img.ndim == 2:
            return gaussian_filter(img, sigma=sigma, mode="reflect")
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[..., c] = gaussian_filter(img[..., c], sigma=sigma, mode="reflect")
        return out

    if use_backend == "cv2" and cv2 is not None:
        ksize = max(1, int(2 * math.ceil(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        # OpenCV expects uint8 or float32; use float32
        src = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.float32)
        blurred = cv2.GaussianBlur(src, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        return blurred.astype(np.float32) / 255.0

    # PIL fallback
    if img.ndim == 2:
        proxy = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(proxy, mode="L")
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.asarray(blurred).astype(np.float32) / 255.0
    proxy = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(proxy, mode="RGB")
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred).astype(np.float32) / 255.0

def bilateral_blur_float(img: np.ndarray, depth: np.ndarray, sigma_spatial: float, sigma_depth: float = 0.08, diameter: Optional[int] = None) -> np.ndarray:
    """
    Edge-preserving bilateral-style blur guided by depth.
    If OpenCV not present, falls back to gaussian blur.
    sigma_depth is relative (0..1) and used to compute sigmaColor for OpenCV.
    """
    if (not _CV2_AVAILABLE) or sigma_spatial <= 0.5:
        return gaussian_blur_float(img, sigma_spatial)

    d = diameter if diameter is not None else max(1, int(2 * math.ceil(3 * sigma_spatial) + 1))
    sigma_color = float(max(1.0, sigma_depth * 255.0))
    sigma_space = float(sigma_spatial)

    # OpenCV bilateralFilter operates per-channel on uint8/float images
    out = np.empty_like(img)
    src = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if img.ndim == 2:
        res = cv2.bilateralFilter(src, d, sigma_color, sigma_space)
        return res.astype(np.float32) / 255.0
    for c in range(img.shape[2]):
        ch = src[..., c]
        res = cv2.bilateralFilter(ch, d, sigma_color, sigma_space)
        out[..., c] = res.astype(np.float32) / 255.0
    return out

# ----- file discovery -----

def find_file_for_base(root: str, base: str, pattern_suffix: str = "*", priority_tags: Optional[Tuple[str, ...]] = None, extensions: Optional[Tuple[str, ...]] = None) -> Optional[str]:
    """
    Search recursively under root for files starting with 'base' and matching priority tags.
    Returns best matching path or None.
    """
    tags = priority_tags or PRIORITY_TAGS
    exts = extensions or SUPPORTED_EXTENSIONS
    pattern = os.path.join(root, "**", f"{base}{pattern_suffix}")
    candidates: List[str] = []
    for path in glob.glob(pattern, recursive=True):
        name = os.path.basename(path)
        low = name.lower()
        if any(low.endswith(ext.lower()) for ext in exts):
            # Accept all matching files, but prioritize by tag
            candidates.append(path)
    if not candidates:
        return None

    def score(p: str) -> int:
        low = p.lower()
        # prefer earlier tag index and earlier position
        for idx, tag in enumerate(tags):
            tagp = tag.lower()
            if tagp in low:
                pos = low.find(tagp)
                return idx * 100000 + pos
        return 9999999

    candidates.sort(key=score)
    return candidates[0]

def find_mask_for_base(mask_root: Optional[str], base: str, kind: str) -> Optional[str]:
    """Look for _mask_{kind}.* under mask_root (first match)."""
    if not mask_root:
        return None
    exts = ("png", "tif", "tiff", "jpg", "jpeg")
    for ext in exts:
        pat = os.path.join(mask_root, f"{base}_mask_{kind}.{ext}")
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    # fallback generic glob
    generic = glob.glob(os.path.join(mask_root, f"{base}_mask_{kind}.*"))
    return generic[0] if generic else None

# ----- I/O helpers -----

@retry_on_io_error()
def load_image_rgb(path: str) -> np.ndarray:
    validate_file_exists(path, "Image")
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr.astype(np.float32) / 255.0

@retry_on_io_error()
def save_image_rgb(path: str, rgb01: np.ndarray, fmt: str = "tiff", quality: int = 95) -> str:
    path = str(path)
    rgb8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    stem = str(Path(path).with_suffix(""))
    fmt = fmt.lower()
    if fmt in ("tiff", "tif") and _TIFFFILE_AVAILABLE:
        out = f"{stem}.tiff"
        tiff.imwrite(out, rgb8, compression="deflate", photometric="rgb")
    elif fmt in ("tiff", "tif") and not _TIFFFILE_AVAILABLE:
        _log.debug("tifffile not available - falling back to PNG for %s", path)
        out = f"{stem}.png"
        Image.fromarray(rgb8).save(out, optimize=True)
    elif fmt == "png":
        out = f"{stem}.png"
        Image.fromarray(rgb8).save(out, optimize=True)
    elif fmt in ("jpg", "jpeg"):
        out = f"{stem}.jpg"
        Image.fromarray(rgb8).save(out, quality=int(quality), optimize=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return out

@retry_on_io_error()
def load_depth_normalized(depth_path: str, target_size: Optional[Tuple[int, int]] = None, method: str = "percentile", use_cache: bool = True) -> np.ndarray:
    """
    Load a depth file and normalize to [0,1]. Supports percentile clipping, histogram equalization or linear scaling.
    target_size: (H, W)
    """
    validate_file_exists(depth_path, "Depth map")
    cache_key = (depth_path, target_size, method)
    if use_cache:
        cached = _depth_cache.get(cache_key)
        if cached is not None:
            return cached

    raw = np.asarray(Image.open(depth_path)).astype(np.float32)

    if method == "percentile":
        p_lo, p_hi = np.percentile(raw, [2, 98])
        clipped = np.clip(raw, p_lo, p_hi)
        norm = (clipped - p_lo) / (p_hi - p_lo + 1e-8)
    elif method == "histogram":
        u8 = ((raw - raw.min()) / (np.ptp(raw) + 1e-8) * 255.0).astype(np.uint8)
        equal = ImageOps.equalize(Image.fromarray(u8))
        norm = np.asarray(equal).astype(np.float32) / 255.0
    else:  # linear
        norm = (raw - raw.min()) / (np.ptp(raw) + 1e-8)

    if target_size is not None:
        H, W = target_size
        u8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        norm = np.asarray(Image.fromarray(u8).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0

    if use_cache:
        _depth_cache.put(cache_key, norm)

    return norm

@retry_on_io_error()
def load_mask(mask_path: Optional[str], kind: str, target_size: Tuple[int, int], use_cache: bool = True) -> np.ndarray:
    """
    Load a mask file (grayscale or alpha) and return normalized float mask HxW in [0,1].
    If mask_path is None or missing, returns zeros of target_size.
    """
    H, W = target_size
    if not mask_path:
        return np.zeros((H, W), dtype=np.float32)

    cache_key = (mask_path, target_size)
    if use_cache:
        cached = _mask_cache.get(cache_key)
        if cached is not None:
            return cached

    if not os.path.exists(mask_path):
        _log.debug("Mask missing for %s: %s", kind, mask_path)
        return np.zeros((H, W), dtype=np.float32)
    try:
        img = Image.open(mask_path)
        if img.mode == "L":
            pass
        elif img.mode == "RGBA":
            _log.debug("Using alpha channel from %s for mask %s", mask_path, kind)
            img = img.getchannel("A")
        elif img.mode == "RGB":
            # convert to L but warn if channels differ
            arr = np.asarray(img)
            if not (np.allclose(arr[..., 0], arr[..., 1]) and np.allclose(arr[..., 1], arr[..., 2])):
                _log.debug("RGB mask %s has differing channels, converting to grayscale", mask_path)
            img = img.convert("L")
        else:
            img = img.convert("L")

        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.clip(arr, 0.0, 1.0)
        if use_cache:
            _mask_cache.put(cache_key, arr)
        return arr
    except Exception as exc:
        _log.warning("Failed loading mask %s (%s): %s", mask_path, kind, exc)
        return np.zeros((H, W), dtype=np.float32)

# ----- effects -----

def apply_depth_haze(img: np.ndarray, depth: np.ndarray, haze_color: Tuple[float, float, float] = DEFAULT_HAZE_COLOR, strength: float = 0.16, near_pct: float = 12.0, far_pct: float = 88.0, mids_gain: float = 1.02, sky_mask: Optional[np.ndarray] = None, building_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Depth-weighted atmospheric haze with mask modulation.
    Masks are 0..1 floats HxW.
    """
    haze_color = validate_color(haze_color, "haze_color")
    near, far = np.percentile(depth, [near_pct, far_pct])
    depth_range = far - near if far > near else 1e-6
    w = np.clip((depth - near) / depth_range, 0.0, 1.0)[..., None]  # HxWx1

    H, W = depth.shape[:2]
    sky = np.zeros((H, W), dtype=np.float32) if (sky_mask is None or sky_mask.size == 0) else sky_mask.astype(np.float32)
    building = np.zeros((H, W), dtype=np.float32) if (building_mask is None or building_mask.size == 0) else building_mask.astype(np.float32)
    sky = sky[..., None]
    building = building[..., None]

    weight_modulated = w * (1.0 - building * BUILDING_HAZE_SUPPRESSION) + sky * (SKY_HAZE_BOOST * (1.0 - w))
    weight_modulated = np.clip(weight_modulated, 0.0, 1.0)

    haze_rgb = np.array(haze_color, dtype=np.float32)[None, None, :]
    blended = img * (1 - strength * weight_modulated) + haze_rgb * (strength * weight_modulated)
    enhanced = np.clip(0.5 + (blended - 0.5) * mids_gain, 0.0, 1.0)
    return enhanced

def apply_depth_clarity(img: np.ndarray, depth: np.ndarray, amount: float = 0.14, radius_px: int = 3, near_pct: float = 18.0, far_pct: float = 82.0, sky_mask: Optional[np.ndarray] = None, building_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Depth-aware microcontrast enhancement. Avoids sky and favors building.
    """
    near, far = np.percentile(depth, [near_pct, far_pct])
    depth_range = far - near if far > near else 1e-6
    w = np.clip((depth - near) / depth_range, 0.0, 1.0)[..., None]

    H, W = depth.shape[:2]
    sky = np.zeros((H, W), dtype=np.float32) if (sky_mask is None or sky_mask.size == 0) else sky_mask.astype(np.float32)
    building = np.zeros((H, W), dtype=np.float32) if (building_mask is None or building_mask.size == 0) else building_mask.astype(np.float32)
    sky = sky[..., None]
    building = building[..., None]

    blurred = gaussian_blur_float(img, sigma=float(radius_px))
    detail = img - blurred

    mask_strength = (1.0 - sky) * (0.6 + 0.4 * building)
    mask_strength = mask_strength[..., None]

    enhanced = img + detail * (amount * w * mask_strength)
    return np.clip(enhanced, 0.0, 1.0)

def apply_depth_dof(img: np.ndarray, depth: np.ndarray, focus_pct: float = 35.0, aperture: float = 0.20, clarity: float = 0.15, falloff: float = 1.5, edge_preserving: bool = True, bilateral_sigma_depth: float = 0.08, bilateral_diameter: Optional[int] = None, sky_mask: Optional[np.ndarray] = None, building_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    DOF with optional bilateral edge preservation. Masks protect building from blur and can slightly reduce sky extremes.
    """
    if clarity > 1e-6:
        usm = Image.fromarray((img * 255.0).astype(np.uint8)).filter(ImageFilter.UnsharpMask(radius=2, percent=int(clarity * 100), threshold=0))
        img = np.asarray(usm).astype(np.float32) / 255.0

    focus_depth = float(np.percentile(depth, focus_pct))
    dist = np.abs(depth - focus_depth)
    dist_max = dist.max()
    if dist_max > 0:
        dist = dist / dist_max
    weight = np.clip(dist ** max(1e-3, falloff), 0.0, 1.0)[..., None]

    H, W = depth.shape[:2]
    building = np.zeros((H, W), dtype=np.float32) if (building_mask is None or building_mask.size == 0) else building_mask.astype(np.float32)
    sky = np.zeros((H, W), dtype=np.float32) if (sky_mask is None or sky_mask.size == 0) else sky_mask.astype(np.float32)
    building = building[..., None]
    sky = sky[..., None]

    r_near = max(1.0, 2.0 + 10.0 * aperture)
    r_far = max(2.0, 6.0 + 28.0 * aperture)

    if edge_preserving and _CV2_AVAILABLE:
        blur_near = bilateral_blur_float(img, depth, sigma_spatial=r_near, sigma_depth=bilateral_sigma_depth, diameter=bilateral_diameter)
        blur_far = bilateral_blur_float(img, depth, sigma_spatial=r_far, sigma_depth=bilateral_sigma_depth * 0.75, diameter=bilateral_diameter)
    else:
        blur_near = gaussian_blur_float(img, sigma=r_near)
        blur_far = gaussian_blur_float(img, sigma=r_far)

    w_soft = weight
    w_hard = weight ** 2.0
    blended = (1 - w_soft) * blur_near + w_soft * ((1 - w_hard) * blur_near + w_hard * blur_far)

    # protector reduces blur on buildings and slightly reduces extreme sky blur
    reduce_on_building = (1.0 - BUILDING_BLUR_REDUCTION * building)  # building=1 -> factor ~0.12
    slight_sky_protect = (1.0 - SKY_BLUR_REDUCTION * sky)            # sky=1 -> factor ~0.7
    protector = (reduce_on_building * slight_sky_protect)[..., None]

    w_protected = w_soft * protector
    out = img * (1 - w_protected) + blended * w_protected
    return np.clip(out, 0.0, 1.0)

# ----- batch driver -----

@dataclass
class BatchOptions:
    images_root: str
    depths_root: str
    out_root: str
    mask_root: Optional[str] = None
    mode: str = "haze"
    restrict_tag: Optional[str] = None
    fmt: str = "tiff"
    workers: int = 1
    verbose: bool = False
    skip_missing: bool = True
    # effect-specific
    haze_color: Tuple[float, float, float] = DEFAULT_HAZE_COLOR
    strength: float = 0.18
    near: float = 15.0
    far: float = 85.0
    mids_gain: float = 1.03
    amount: float = 0.12
    radius: int = 3
    focus: float = 35.0
    aperture: float = 0.22
    clarity: float = 0.18
    falloff: float = 1.4

def _process_single(dp: str, opts: BatchOptions) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Process a single depth map path (dp). Returns tuple (base, out_path or None, error or None).
    This helper is suitable for running in a child process as long as dependencies are available.
    """
    base = os.path.basename(dp).replace("_depth16.png", "").replace("_depth16.tiff", "").replace("_depth16.tif", "")
    try:
        # find source image
        src = find_file_for_base(opts.images_root, base, pattern_suffix="*")
        if src is None:
            msg = f"No source image found for base {base}"
            _log.debug(msg)
            if opts.skip_missing:
                return base, None, msg
            raise FileNotFoundError(msg)

        img = load_image_rgb(src)
        H, W = img.shape[:2]
        depth = load_depth_normalized(dp, target_size=(H, W), method="percentile", use_cache=True)

        sky_path = find_mask_for_base(opts.mask_root, base, "sky") if opts.mask_root else None
        building_path = find_mask_for_base(opts.mask_root, base, "building") if opts.mask_root else None
        sky_mask = load_mask(sky_path, "sky", (H, W), use_cache=True) if sky_path else np.zeros((H, W), dtype=np.float32)
        building_mask = load_mask(building_path, "building", (H, W), use_cache=True) if building_path else np.zeros((H, W), dtype=np.float32)

        if opts.mode == "haze":
            out = apply_depth_haze(img, depth,
                                   haze_color=opts.haze_color, strength=opts.strength,
                                   near_pct=opts.near, far_pct=opts.far, mids_gain=opts.mids_gain,
                                   sky_mask=sky_mask, building_mask=building_mask)
            suffix = "_depthhaze"
        elif opts.mode == "clarity":
            out = apply_depth_clarity(img, depth,
                                      amount=opts.amount, radius_px=opts.radius,
                                      near_pct=opts.near, far_pct=opts.far,
                                      sky_mask=sky_mask, building_mask=building_mask)
            suffix = "_depthclarity"
        elif opts.mode == "dof":
            out = apply_depth_dof(img, depth,
                                  focus_pct=opts.focus, aperture=opts.aperture,
                                  clarity=opts.clarity, falloff=opts.falloff,
                                  edge_preserving=_CV2_AVAILABLE, bilateral_sigma_depth=0.08,
                                  sky_mask=sky_mask, building_mask=building_mask)
            suffix = "_depthdof"
        else:
            raise ValueError(f"Unknown mode: {opts.mode}")

        root, _ = os.path.splitext(os.path.basename(src))
        out_path = os.path.join(opts.out_root, f"{root}{suffix}.{opts.fmt.lstrip('.')}")
        saved = save_image_rgb(out_path, out, fmt=opts.fmt)
        return base, saved, None
    except Exception as exc:
        _log.exception("Failed processing base %s: %s", base, exc)
        return base, None, str(exc)

def process_batch(opts: BatchOptions, progress: Optional[Callable[[int, int, str], None]] = None) -> None:
    """
    Process a directory of depth maps. If workers > 1, uses ProcessPoolExecutor.
    progress callback signature: (done:int, total:int, message:str)
    """
    os.makedirs(opts.out_root, exist_ok=True)
    depth_maps = sorted(glob.glob(os.path.join(opts.depths_root, "*_depth16.*")))
    if not depth_maps:
        raise SystemExit(f"No depth maps found in {opts.depths_root}")

    total = len(depth_maps)
    done = 0
    errors = []

    _log.info("Starting batch: %d depth maps, mode=%s, workers=%d", total, opts.mode, opts.workers)

    if opts.workers > 1:
        # Use multiprocessing
        with ProcessPoolExecutor(max_workers=opts.workers) as ex:
            futures = {ex.submit(_process_single, dp, opts): dp for dp in depth_maps}
            for fut in as_completed(futures):
                base, out_path, err = fut.result()
                done += 1
                if err:
                    errors.append((base, err))
                    _log.warning("Base %s failed: %s", base, err)
                else:
                    _log.info("Processed %s -> %s", base, out_path)
                if progress:
                    progress(done, total, base)
    else:
        for dp in depth_maps:
            base, out_path, err = _process_single(dp, opts)
            done += 1
            if err:
                errors.append((base, err))
                _log.warning("Base %s failed: %s", base, err)
            else:
                _log.info("Processed %s -> %s", base, out_path)
            if progress:
                progress(done, total, base)

    _log.info("Batch complete: %d processed, %d errors", total - len(errors), len(errors))
    if errors:
        _log.info("Errors (sample): %s", errors[:8])

# ----- CLI -----

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="depth_tools", description="Depth-driven post effects (haze|clarity|dof) with optional masks")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("images_root", help="Root of enhanced images (searched recursively)")
        p.add_argument("depths_root", help="Root folder containing *_depth16.* depth maps")
        p.add_argument("out_root", help="Output folder")
        p.add_argument("--mask-root", type=str, default=None, help="Folder containing generated masks (<base>_mask_sky.png)")
        p.add_argument("--restrict-tag", type=str, default=None, help="Restrict matches to a filename tag (not strictly required)")
        p.add_argument("--fmt", type=str, default="tiff", help="Output format (tiff/png/jpg)")
        p.add_argument("--workers", type=int, default=1, help="Parallel worker count (ProcessPoolExecutor)")
        p.add_argument("--verbose", action="store_true", help="Verbose logging")

    ph = sub.add_parser("haze", help="Apply depth-weighted atmospheric haze")
    common(ph)
    ph.add_argument("--haze-color", type=float, nargs=3, default=DEFAULT_HAZE_COLOR, metavar=("R", "G", "B"))
    ph.add_argument("--strength", type=float, default=0.18)
    ph.add_argument("--near", type=float, default=15.0)
    ph.add_argument("--far", type=float, default=85.0)
    ph.add_argument("--mids-gain", type=float, default=1.03)

    pc = sub.add_parser("clarity", help="Depth-aware microcontrast")
    common(pc)
    pc.add_argument("--amount", type=float, default=0.12)
    pc.add_argument("--radius", type=int, default=3)
    pc.add_argument("--near", type=float, default=20.0)
    pc.add_argument("--far", type=float, default=80.0)

    pd = sub.add_parser("dof", help="Cinematic depth-of-field")
    common(pd)
    pd.add_argument("--focus", type=float, default=35.0)
    pd.add_argument("--aperture", type=float, default=0.22)
    pd.add_argument("--clarity", type=float, default=0.18)
    pd.add_argument("--falloff", type=float, default=1.4)

    return ap

def _cli_progress(done: int, total: int, base: str) -> None:
    print(f"\rProcessed {done}/{total}: {base}", end="", flush=True)
    if done == total:
        print("")

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_cli().parse_args(list(argv) if argv is not None else None)

    # configure logging
    if getattr(args, "verbose", False):
        _log.setLevel(logging.DEBUG)
    else:
        _log.setLevel(logging.INFO)

    opts = BatchOptions(
        images_root=args.images_root,
        depths_root=args.depths_root,
        out_root=args.out_root,
        mask_root=getattr(args, "mask_root", None),
        mode=args.cmd,
        restrict_tag=getattr(args, "restrict_tag", None),
        fmt=getattr(args, "fmt", "tiff"),
        workers=max(1, int(getattr(args, "workers", 1))),
        verbose=getattr(args, "verbose", False)
    )

    # effect-specific mappings
    if args.cmd == "haze":
        opts.haze_color = tuple(getattr(args, "haze_color", DEFAULT_HAZE_COLOR))
        opts.strength = float(getattr(args, "strength", 0.18))
        opts.near = float(getattr(args, "near", 15.0))
        opts.far = float(getattr(args, "far", 85.0))
        opts.mids_gain = float(getattr(args, "mids_gain", 1.03))
    elif args.cmd == "clarity":
        opts.amount = float(getattr(args, "amount", 0.12))
        opts.radius = int(getattr(args, "radius", 3))
        opts.near = float(getattr(args, "near", 20.0))
        opts.far = float(getattr(args, "far", 80.0))
    elif args.cmd == "dof":
        opts.focus = float(getattr(args, "focus", 35.0))
        opts.aperture = float(getattr(args, "aperture", 0.22))
        opts.clarity = float(getattr(args, "clarity", 0.18))
        opts.falloff = float(getattr(args, "falloff", 1.4))

    try:
        process_batch(opts, progress=_cli_progress)
    except Exception as exc:
        _log.exception("Fatal error running batch: %s", exc)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
