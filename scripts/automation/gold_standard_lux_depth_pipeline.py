#!/usr/bin/env python3
"""
Gold-Standard Depth-Aware 16-bit Luxury Enhancement Pipeline

Built on the "Depth-Aware 16-bit Enhancement Pipeline (Robust Edition)" core, with
select strengths integrated from the "Unified Luxury Rendering Pipeline":

Integrated (quality-first) features:
- Luxury presets (consistent, conservative, real-estate oriented).
- Optional *material-aware* response via **explicit user masks** (no guessing).
- Optional 1D/3D LUT (.cube) support with midtone-weighted application, luma preservation,
  and highlight/black protection.
- Stage timings + guard-rail metrics (helps you keep photorealism and avoid overshoot).

Quality-over-convenience policy:
- No runtime downloads.
- Depth inference is intentionally **not** included; depth assets must be provided.
- Material response is applied only when masks exist; otherwise it is skipped.

Required depth assets (stored in --depth-dir):
- <stem>_depth_raw_16bit.tiff

Optional depth zone masks (stored in --depth-dir):
- <stem>_depth_zone_foreground.png
- <stem>_depth_zone_midground.png
- <stem>_depth_zone_background.png

Optional material masks (stored in --depth-dir), 8-bit grayscale PNG (0..255):
- <stem>_material_wood.png
- <stem>_material_metal.png
- <stem>_material_glass.png
- <stem>_material_stone.png
(You can add more surfaces; see --surfaces.)

Outputs (in --output-dir):
- <stem>_MASTER_16bit.tiff
- <stem>_UPSCALED_16bit.tiff
- <stem>_MARKETING.png
- <stem>_PREVIEW.jpg
- <stem>_report.json
- _batch_report.json

Notes:
- AI upscaling (Real-ESRGAN / ONNX) is used as a *detail layer* (luma only) to keep
  color/tonal fidelity anchored to the original image.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import cv2  # opencv-python
except Exception:
    cv2 = None  # type: ignore

try:
    import tifffile
except Exception:
    tifffile = None  # type: ignore

EPS = 1e-12


# --------------------------- Presets ---------------------------

class Preset(str, Enum):
    """Curated looks (conservative; tuned for real estate)."""
    PHOTO_REALISTIC = "photo_realistic"
    ARCHITECTURAL = "architectural"
    ARCHIVAL_QUALITY = "archival_quality"
    SIGNATURE_ESTATE = "signature_estate"
    INTERIOR_LUXURY = "interior_luxury"
    EXTERIOR_SHOWCASE = "exterior_showcase"


# --------------------------- Config ---------------------------

@dataclass
class Config:
    input_dir: Path
    depth_dir: Path
    output_dir: Path

    preset: Preset = Preset.PHOTO_REALISTIC

    upscale: int = 4                 # 2 or 4
    backend: str = "realesrgan"      # realesrgan | onnx | none
    device: str = "auto"            # auto | cuda | cpu (Real-ESRGAN)
    model_path: Optional[Path] = None
    model_sha256: Optional[str] = None

    tile: int = 512
    tile_pad: int = 16
    half: bool = False

    strict_depth: bool = False

    # Output toggles
    save_master: bool = True
    save_upscaled_16bit: bool = True
    save_marketing_png: bool = True
    save_preview_jpg: bool = True
    preview_scale: float = 0.25
    save_report: bool = True

    skip_existing: bool = True
    overwrite: bool = False

    warn_float_gb: float = 6.0  # warn if a single float32 output buffer exceeds this

    # Depth weight synthesis (if masks missing)
    fg_q: float = 0.35
    bg_q: float = 0.65
    transition: float = 0.08
    mask_soften_sigma: float = 4.0  # in ORIGINAL px

    # AI detail transfer (final-res)
    detail_sigma: float = 1.2
    detail_strength: float = 0.65
    detail_clip: float = 0.075
    detail_fg: float = 1.00
    detail_mid: float = 0.70
    detail_bg: float = 0.25

    # Clarity + sharpen (final-res, luma only)
    clarity_sigma: float = 2.2
    clarity_clip: float = 0.05
    clarity_fg: float = 0.18
    clarity_mid: float = 0.10
    clarity_bg: float = 0.05

    sharpen_sigma: float = 0.9
    sharpen_thresh: float = 0.010
    sharpen_fg: float = 0.10
    sharpen_mid: float = 0.06
    sharpen_bg: float = 0.02

    # Grading (subtle, luma-preserving temperature)
    temp_fg: float = 0.012
    temp_mid: float = 0.004
    temp_bg: float = -0.002

    sat_fg: float = 1.040
    sat_mid: float = 1.020
    sat_bg: float = 1.000

    exp_fg: float = 1.010
    exp_mid: float = 1.000
    exp_bg: float = 0.995

    con_fg: float = 1.030
    con_mid: float = 1.020
    con_bg: float = 1.010

    soft_clip_knee: float = 0.92

    # Luxury: material response (quality-first: only via explicit masks)
    enable_material: bool = True
    material_strength: float = 0.75
    surfaces: Tuple[str, ...] = ("wood", "metal", "glass", "stone")
    material_mask_soften_sigma: float = 2.0  # in ORIGINAL px

    # Luxury: LUT (.cube)
    enable_lut: bool = False
    lut_path: Optional[Path] = None
    lut_strength: float = 0.70
    lut_preserve_luma: bool = True
    lut_protect_highlights: bool = True
    lut_protect_blacks: bool = True
    lut_midtone_bias: float = 0.85  # 0..1, higher = more midtone-only

    # Guard-rails
    validate_ai: bool = True
    ai_color_warn: float = 0.06     # mean abs RGB diff (0..1)
    ai_color_fail: float = 0.12     # if exceeded, skip AI details
    ai_luma_warn: float = 0.06
    ai_luma_fail: float = 0.12

    def apply_preset(self) -> None:
        """Mutate config in-place based on preset."""
        p = self.preset

        # Baseline (PHOTO_REALISTIC): conservative, clean, minimal creative bias.
        if p == Preset.PHOTO_REALISTIC:
            self.enable_lut = False
            self.lut_strength = 0.0
            self.material_strength = 0.70
            self.temp_fg, self.temp_mid, self.temp_bg = 0.010, 0.003, -0.002
            self.sat_fg, self.sat_mid, self.sat_bg = 1.030, 1.015, 1.000
            self.con_fg, self.con_mid, self.con_bg = 1.025, 1.015, 1.010
            self.detail_strength = 0.65

        elif p == Preset.ARCHITECTURAL:
            self.enable_lut = False
            self.material_strength = 0.65
            self.temp_fg, self.temp_mid, self.temp_bg = 0.006, 0.002, -0.003
            self.sat_fg, self.sat_mid, self.sat_bg = 1.020, 1.010, 1.000
            self.con_fg, self.con_mid, self.con_bg = 1.040, 1.030, 1.020
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.20, 0.11, 0.05
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.11, 0.07, 0.02
            self.detail_strength = 0.70

        elif p == Preset.ARCHIVAL_QUALITY:
            # Minimal "look"; mostly technical refinement. Great for archives / further finishing.
            self.enable_lut = False
            self.enable_material = False
            self.material_strength = 0.0
            self.temp_fg, self.temp_mid, self.temp_bg = 0.004, 0.001, -0.001
            self.sat_fg, self.sat_mid, self.sat_bg = 1.010, 1.005, 1.000
            self.con_fg, self.con_mid, self.con_bg = 1.015, 1.010, 1.005
            self.detail_strength = 0.55
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.14, 0.08, 0.04
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.08, 0.05, 0.015
            self.soft_clip_knee = 0.94

        elif p == Preset.SIGNATURE_ESTATE:
            # Subtle warmth + presence. LUT optional (recommended).
            self.enable_material = True
            self.material_strength = 0.80
            self.enable_lut = bool(self.lut_path)
            self.lut_strength = 0.70
            self.temp_fg, self.temp_mid, self.temp_bg = 0.014, 0.006, 0.000
            self.sat_fg, self.sat_mid, self.sat_bg = 1.060, 1.030, 1.010
            self.con_fg, self.con_mid, self.con_bg = 1.035, 1.025, 1.015
            self.detail_strength = 0.68
            self.soft_clip_knee = 0.91

        elif p == Preset.INTERIOR_LUXURY:
            # Warmer mid/fg, protect windows (glass) from oversharpen.
            self.enable_material = True
            self.material_strength = 0.85
            self.enable_lut = bool(self.lut_path)
            self.lut_strength = 0.65
            self.temp_fg, self.temp_mid, self.temp_bg = 0.018, 0.010, 0.002
            self.sat_fg, self.sat_mid, self.sat_bg = 1.070, 1.040, 1.010
            self.con_fg, self.con_mid, self.con_bg = 1.030, 1.020, 1.010
            self.detail_strength = 0.65
            self.soft_clip_knee = 0.91

        elif p == Preset.EXTERIOR_SHOWCASE:
            # Slightly cooler background; protect sky/foliage from harsh microcontrast.
            self.enable_material = True
            self.material_strength = 0.78
            self.enable_lut = bool(self.lut_path)
            self.lut_strength = 0.60
            self.temp_fg, self.temp_mid, self.temp_bg = 0.010, 0.002, -0.006
            self.sat_fg, self.sat_mid, self.sat_bg = 1.040, 1.020, 1.000
            self.con_fg, self.con_mid, self.con_bg = 1.035, 1.020, 1.008
            self.detail_strength = 0.70
            self.soft_clip_knee = 0.90


def _need_deps() -> None:
    if cv2 is None or tifffile is None:
        raise RuntimeError("Missing dependencies. Install: opencv-python tifffile numpy tqdm")


# --------------------------- IO ---------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model(path: Path, expected_sha256: Optional[str]) -> None:
    if expected_sha256:
        got = sha256_file(path)
        if got.lower() != expected_sha256.lower():
            raise ValueError(f"SHA256 mismatch for {path.name}: expected {expected_sha256}, got {got}")


def read_rgb_tiff(path: Path) -> np.ndarray:
    _need_deps()
    arr = tifffile.imread(str(path))
    if getattr(arr, "ndim", 0) == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Unsupported TIFF dtype: {arr.dtype}")
    if arr.shape[2] == 4:
        # Composite over white
        rgb = arr[..., :3].astype(np.float32)
        a = arr[..., 3:4].astype(np.float32) / np.iinfo(arr.dtype).max
        white = np.ones_like(rgb, dtype=np.float32) * np.iinfo(arr.dtype).max
        comp = rgb * a + white * (1.0 - a)
        arr = comp.astype(arr.dtype)
    return arr[..., :3]


def to01(rgb_u: np.ndarray) -> np.ndarray:
    if rgb_u.dtype == np.uint8:
        return (rgb_u.astype(np.float32) / 255.0).astype(np.float32)
    if rgb_u.dtype == np.uint16:
        return (rgb_u.astype(np.float32) / 65535.0).astype(np.float32)
    raise ValueError("to01 expects uint8 or uint16")


def u16(rgb01: np.ndarray) -> np.ndarray:
    arr = np.clip(rgb01, 0, 1)
    return (arr * 65535.0 + 0.5).astype(np.uint16)


def u8(rgb01: np.ndarray) -> np.ndarray:
    arr = np.clip(rgb01, 0, 1)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def write_tiff_u16(path: Path, rgb_u16: np.ndarray, compression: str = "zlib") -> None:
    _need_deps()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tifffile.imwrite(str(tmp), rgb_u16, photometric="rgb", compression=compression, metadata=None)
    tmp.replace(path)


def write_png_u8(path: Path, rgb_u8: np.ndarray) -> None:
    _need_deps()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    success = cv2.imwrite(str(tmp), rgb_u8[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not success:
        raise RuntimeError(f"Failed to write PNG: {tmp}")
    tmp.replace(path)


def write_preview_jpg(path: Path, rgb_u8: np.ndarray, scale: float) -> None:
    _need_deps()
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb_u8.shape[:2]
    s = max(0.05, min(1.0, float(scale)))
    nh, nw = int(round(h * s)), int(round(w * s))
    small = cv2.resize(rgb_u8[..., ::-1], (nw, nh), interpolation=cv2.INTER_AREA)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    success = cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError(f"Failed to write JPEG: {tmp}")
    tmp.replace(path)


# --------------------------- Utility math ---------------------------

def luma(rgb01: np.ndarray) -> np.ndarray:
    return (rgb01[..., 0] * 0.2126 + rgb01[..., 1] * 0.7152 + rgb01[..., 2] * 0.0722).astype(np.float32)


def blur(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    k = int(round(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(x.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def resize(x: np.ndarray, w: int, h: int, interp: int) -> np.ndarray:
    return cv2.resize(x, (int(w), int(h)), interpolation=interp)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, EPS), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def soft_clip01(rgb01: np.ndarray, knee: float) -> np.ndarray:
    # gentle highlight rolloff starting at knee (0..1)
    k = float(np.clip(knee, 0.0, 1.0))
    out = rgb01.astype(np.float32)
    if k >= 1.0:
        return np.clip(out, 0.0, 1.0)
    x = np.maximum(out - k, 0.0) / max(1.0 - k, EPS)
    out = out - x * x * 0.5 * (1.0 - k)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def edge_map(l: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(l.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    p = np.percentile(mag, 99.0)
    if p <= EPS:
        return np.zeros_like(mag, dtype=np.float32)
    return np.clip(mag / p, 0.0, 1.0).astype(np.float32)


def midtone_map(l: np.ndarray) -> np.ndarray:
    # 0 in deep shadows/highlights, 1 in midtones
    a = smoothstep(0.08, 0.35, l)
    b = 1.0 - smoothstep(0.65, 0.98, l)
    return (a * b).astype(np.float32)


def apply_luma_ratio(rgb01: np.ndarray, new_l: np.ndarray, old_l: Optional[np.ndarray] = None) -> np.ndarray:
    if old_l is None:
        old_l = luma(rgb01)
    ratio = np.clip(new_l / np.maximum(old_l, EPS), 0.0, 8.0).astype(np.float32)
    out = rgb01.astype(np.float32) * ratio[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def param_map(wfg: np.ndarray, wmid: np.ndarray, wbg: np.ndarray, fg: float, mid: float, bg: float) -> np.ndarray:
    return (wfg * fg + wmid * mid + wbg * bg).astype(np.float32)


# --------------------------- Depth / Weights ---------------------------

@dataclass
class Weights:
    wfg: np.ndarray
    wmid: np.ndarray
    wbg: np.ndarray
    source: str


def norm_depth(depth_u16: np.ndarray) -> np.ndarray:
    d = depth_u16.astype(np.float32)
    lo, hi = np.percentile(d, 1.0), np.percentile(d, 99.0)
    if hi <= lo + 1.0:
        return np.zeros_like(d, dtype=np.float32)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_mask_any(p: Path) -> np.ndarray:
    """
    Load a single-channel mask into float32 [0,1].
    Supports:
    - PNG/JPG via OpenCV (8-bit)
    - TIFF via tifffile (8/16-bit)
    """
    _need_deps()
    suf = p.suffix.lower()
    if suf in (".tif", ".tiff"):
        m = tifffile.imread(str(p))
        if getattr(m, "ndim", 0) > 2:
            m = m[..., 0]
        if m.dtype == np.uint8:
            return (m.astype(np.float32) / 255.0).astype(np.float32)
        if m.dtype == np.uint16:
            return (m.astype(np.float32) / 65535.0).astype(np.float32)
        return np.clip(m.astype(np.float32), 0.0, 1.0).astype(np.float32)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(str(p))
    return (m.astype(np.float32) / 255.0).astype(np.float32)


def weights_from_assets(h: int, w: int, depth: Optional[np.ndarray], masks: Dict[str, np.ndarray], cfg: Config) -> Weights:
    have = all(k in masks for k in ("foreground", "midground", "background"))
    if have:
        wfg, wmid, wbg = masks["foreground"], masks["midground"], masks["background"]
        if wfg.shape != (h, w): wfg = resize(wfg, w, h, cv2.INTER_LINEAR)
        if wmid.shape != (h, w): wmid = resize(wmid, w, h, cv2.INTER_LINEAR)
        if wbg.shape != (h, w): wbg = resize(wbg, w, h, cv2.INTER_LINEAR)
        s = max(0.0, cfg.mask_soften_sigma)
        if s > 0:
            wfg, wmid, wbg = blur(wfg, s), blur(wmid, s), blur(wbg, s)
        sm = np.maximum(wfg + wmid + wbg, EPS)
        wfg, wmid, wbg = wfg / sm, wmid / sm, wbg / sm
        return Weights(wfg=wfg.astype(np.float32), wmid=wmid.astype(np.float32), wbg=wbg.astype(np.float32), source="zone_masks")
    if depth is None:
        # fall back to uniform
        u = np.ones((h, w), np.float32)
        return Weights(wfg=u * 0.34, wmid=u * 0.33, wbg=u * 0.33, source="uniform_no_depth")
    # derive from depth percentiles
    fg_t = float(np.quantile(depth, cfg.fg_q))
    bg_t = float(np.quantile(depth, cfg.bg_q))
    tr = float(max(cfg.transition, 1e-3))
    wfg = 1.0 - smoothstep(fg_t - tr, fg_t + tr, depth)
    wbg = smoothstep(bg_t - tr, bg_t + tr, depth)
    wmid = np.clip(1.0 - wfg - wbg, 0.0, 1.0)
    s = max(0.0, cfg.mask_soften_sigma)
    if s > 0:
        wfg, wmid, wbg = blur(wfg, s), blur(wmid, s), blur(wbg, s)
        sm = np.maximum(wfg + wmid + wbg, EPS)
        wfg, wmid, wbg = wfg / sm, wmid / sm, wbg / sm
    return Weights(wfg=wfg.astype(np.float32), wmid=wmid.astype(np.float32), wbg=wbg.astype(np.float32), source="depth_quantiles")


def upsample_weights(W: Weights, h2: int, w2: int) -> Weights:
    wfg = resize(W.wfg, w2, h2, cv2.INTER_LINEAR).astype(np.float32)
    wmid = resize(W.wmid, w2, h2, cv2.INTER_LINEAR).astype(np.float32)
    wbg = resize(W.wbg, w2, h2, cv2.INTER_LINEAR).astype(np.float32)
    sm = np.maximum(wfg + wmid + wbg, EPS)
    return Weights(wfg=wfg / sm, wmid=wmid / sm, wbg=wbg / sm, source=W.source)


# --------------------------- Upscalers ---------------------------

class Upscaler:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scale = cfg.upscale

    def upscale(self, rgb01: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoneUpscaler(Upscaler):
    def upscale(self, rgb01: np.ndarray) -> np.ndarray:
        h, w = rgb01.shape[:2]
        out = resize(rgb01, w * self.scale, h * self.scale, cv2.INTER_LANCZOS4)
        return np.clip(out.astype(np.float32), 0.0, 1.0)


class RealESRGAN(Upscaler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        if not cfg.model_path:
            raise ValueError("RealESRGAN backend requires --model-path to local .pth")
        verify_model(cfg.model_path, cfg.model_sha256)

        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # RRDBNet configuration matches Real-ESRGAN x4plus
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=cfg.upscale)

        self._er = RealESRGANer(
            scale=cfg.upscale,
            model_path=str(cfg.model_path),
            model=model,
            tile=cfg.tile if cfg.tile > 0 else 0,
            tile_pad=cfg.tile_pad,
            pre_pad=0,
            half=bool(cfg.half),
            gpu_id=0 if device == "cuda" else None,
        )

    def upscale(self, rgb01: np.ndarray) -> np.ndarray:
        # Real-ESRGAN expects BGR uint8
        inp = u8(rgb01)[..., ::-1]
        out, _ = self._er.enhance(inp, outscale=self.scale)
        out = out[..., ::-1]
        return np.clip(out.astype(np.float32) / 255.0, 0.0, 1.0)


class ONNXUpscaler(Upscaler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        if not cfg.model_path:
            raise ValueError("ONNX backend requires --model-path to local .onnx")
        verify_model(cfg.model_path, cfg.model_sha256)
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if cfg.device in ("auto", "cuda"):
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                providers = ["CPUExecutionProvider"]

        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(cfg.model_path), sess_options=sess_opt, providers=providers)
        self._in_name = self._sess.get_inputs()[0].name
        self._out_name = self._sess.get_outputs()[0].name

    def upscale(self, rgb01: np.ndarray) -> np.ndarray:
        # Assumes NCHW float32 0..1 input
        x = rgb01.transpose(2, 0, 1)[None, ...].astype(np.float32)
        y = self._sess.run([self._out_name], {self._in_name: x})[0]
        y = np.clip(y[0].transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)
        return y


def build_upscaler(cfg: Config) -> Upscaler:
    b = cfg.backend.lower()
    if b == "none":
        return NoneUpscaler(cfg)
    if b == "realesrgan":
        return RealESRGAN(cfg)
    if b == "onnx":
        return ONNXUpscaler(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")


# --------------------------- Luxury: material response ---------------------------

@dataclass(frozen=True)
class SurfaceProfile:
    temp_offset: float = 0.0     # additive (approx)
    sat_mult: float = 1.0        # multiplicative
    exp_mult: float = 1.0
    con_mult: float = 1.0
    detail_mult: float = 1.0
    clarity_mult: float = 1.0
    sharpen_mult: float = 1.0
    highlight_compress: float = 0.0  # 0..1, luma rolloff in highlights


SURFACE_PROFILES: Dict[str, SurfaceProfile] = {
    # Subtle defaults; they're meant to be *barely noticeable*.
    "wood":  SurfaceProfile(temp_offset=0.006, sat_mult=1.05, con_mult=1.015, detail_mult=1.05, clarity_mult=1.08, sharpen_mult=1.03),
    "metal": SurfaceProfile(temp_offset=-0.001, sat_mult=0.99, con_mult=1.010, detail_mult=1.03, clarity_mult=1.06, sharpen_mult=1.03),
    "glass": SurfaceProfile(temp_offset=-0.002, sat_mult=0.98, con_mult=0.995, detail_mult=0.80, clarity_mult=0.82, sharpen_mult=0.78, highlight_compress=0.20),
    "stone": SurfaceProfile(temp_offset=0.000, sat_mult=1.00, con_mult=1.012, detail_mult=1.02, clarity_mult=1.05, sharpen_mult=1.02),

    # Optional extra surfaces you can add by providing matching masks.
    "foliage": SurfaceProfile(temp_offset=0.000, sat_mult=1.02, con_mult=1.005, detail_mult=0.95, clarity_mult=0.92, sharpen_mult=0.92),
    "sky":     SurfaceProfile(temp_offset=-0.003, sat_mult=1.00, con_mult=0.995, detail_mult=0.85, clarity_mult=0.78, sharpen_mult=0.75, highlight_compress=0.15),
}


@dataclass
class MaterialMods:
    # Per-pixel maps
    temp_offset: np.ndarray
    sat_mult: np.ndarray
    exp_mult: np.ndarray
    con_mult: np.ndarray
    detail_mult: np.ndarray
    clarity_mult: np.ndarray
    sharpen_mult: np.ndarray
    highlight_compress: np.ndarray  # 0..1
    source: str


def _blend_mult(cur: np.ndarray, mult: float, w: np.ndarray) -> np.ndarray:
    # cur * lerp(1, mult, w) == cur * (1 + w*(mult-1))
    return (cur * (1.0 + w * (mult - 1.0))).astype(np.float32)


def _blend_add(cur: np.ndarray, add: float, w: np.ndarray) -> np.ndarray:
    return (cur + w * add).astype(np.float32)


def load_material_masks(stem: str, depth_dir: Path, surfaces: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for s in surfaces:
        # Prefer PNG; accept TIFF if provided.
        candidates = [
            depth_dir / f"{stem}_material_{s}.png",
            depth_dir / f"{stem}_material_{s}.tif",
            depth_dir / f"{stem}_material_{s}.tiff",
        ]
        for p in candidates:
            if p.exists():
                masks[s] = load_mask_any(p)
                break
    return masks


def build_material_mods(h: int, w: int, masks: Dict[str, np.ndarray], cfg: Config) -> Optional[MaterialMods]:
    if not cfg.enable_material:
        return None
    if cfg.material_strength <= 0:
        return None
    if not masks:
        return None

    # Base maps
    temp_off = np.zeros((h, w), np.float32)
    sat_mult = np.ones((h, w), np.float32)
    exp_mult = np.ones((h, w), np.float32)
    con_mult = np.ones((h, w), np.float32)
    det_mult = np.ones((h, w), np.float32)
    cla_mult = np.ones((h, w), np.float32)
    sha_mult = np.ones((h, w), np.float32)
    hi_comp = np.zeros((h, w), np.float32)

    s = max(0.0, cfg.material_mask_soften_sigma)
    strength = float(np.clip(cfg.material_strength, 0.0, 1.0))
    used: List[str] = []

    for name, m in masks.items():
        prof = SURFACE_PROFILES.get(name)
        if prof is None:
            # Unknown surface -> ignore (quality-first).
            continue
        if m.shape != (h, w):
            m = resize(m, w, h, cv2.INTER_LINEAR)
        m = np.clip(m.astype(np.float32), 0.0, 1.0)
        if s > 0:
            m = blur(m, s)
        wgt = np.clip(m * strength, 0.0, 1.0)

        temp_off = _blend_add(temp_off, prof.temp_offset, wgt)
        sat_mult = _blend_mult(sat_mult, prof.sat_mult, wgt)
        exp_mult = _blend_mult(exp_mult, prof.exp_mult, wgt)
        con_mult = _blend_mult(con_mult, prof.con_mult, wgt)
        det_mult = _blend_mult(det_mult, prof.detail_mult, wgt)
        cla_mult = _blend_mult(cla_mult, prof.clarity_mult, wgt)
        sha_mult = _blend_mult(sha_mult, prof.sharpen_mult, wgt)
        hi_comp = _blend_add(hi_comp, prof.highlight_compress, wgt)
        used.append(name)

    if not used:
        return None

    # Safety clamps (prevents accidental mask overshoot)
    sat_mult = np.clip(sat_mult, 0.85, 1.25)
    exp_mult = np.clip(exp_mult, 0.90, 1.10)
    con_mult = np.clip(con_mult, 0.90, 1.15)
    det_mult = np.clip(det_mult, 0.70, 1.25)
    cla_mult = np.clip(cla_mult, 0.70, 1.35)
    sha_mult = np.clip(sha_mult, 0.60, 1.35)
    hi_comp = np.clip(hi_comp, 0.0, 1.0)

    return MaterialMods(
        temp_offset=temp_off,
        sat_mult=sat_mult,
        exp_mult=exp_mult,
        con_mult=con_mult,
        detail_mult=det_mult,
        clarity_mult=cla_mult,
        sharpen_mult=sha_mult,
        highlight_compress=hi_comp,
        source="material_masks:" + ",".join(sorted(set(used))),
    )


def apply_material_highlight_compression(rgb01: np.ndarray, hi_comp: np.ndarray, knee: float = 0.85) -> np.ndarray:
    """
    Gentle highlight rolloff inside masked regions (primarily for glass/sky).
    This is intentionally subtle: it reduces only the top end.
    """
    if hi_comp is None:
        return rgb01
    if float(np.max(hi_comp)) <= 0:
        return rgb01
    l = luma(rgb01)
    t = smoothstep(knee, 1.0, l)
    # up to ~4-5% reduction at the very top for strength=1
    new_l = np.clip(l - (hi_comp * t * t) * 0.05, 0.0, 1.0)
    return apply_luma_ratio(rgb01, new_l, old_l=l)


# --------------------------- Luxury: LUT (.cube) ---------------------------

@dataclass(frozen=True)
class CubeLUT:
    title: str
    domain_min: np.ndarray  # (3,)
    domain_max: np.ndarray  # (3,)
    lut1d: Optional[np.ndarray]  # (N,3) or None
    lut3d: Optional[np.ndarray]  # (N,N,N,3) or None
    size1d: int
    size3d: int


def _parse_floats(parts: List[str]) -> List[float]:
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


def load_cube_lut(path: Path) -> CubeLUT:
    """
    Minimal .cube reader:
    - Supports LUT_1D_SIZE and/or LUT_3D_SIZE.
    - DOMAIN_MIN/MAX is respected.
    - 3D data is assumed in the standard order: B fastest, then G, then R.
      (This matches the majority of .cube exports.)
    """
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
    title = ""
    dom_min = np.array([0.0, 0.0, 0.0], np.float32)
    dom_max = np.array([1.0, 1.0, 1.0], np.float32)
    n1d = 0
    n3d = 0
    data: List[List[float]] = []
    for ln in txt:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.upper().startswith("TITLE"):
            # TITLE "name"
            if '"' in s:
                title = s.split('"', 1)[1].rsplit('"', 1)[0]
            continue
        parts = s.split()
        key = parts[0].upper()
        if key == "DOMAIN_MIN" and len(parts) >= 4:
            dom_min = np.array(_parse_floats(parts[1:4]), np.float32)
            continue
        if key == "DOMAIN_MAX" and len(parts) >= 4:
            dom_max = np.array(_parse_floats(parts[1:4]), np.float32)
            continue
        if key == "LUT_1D_SIZE" and len(parts) >= 2:
            n1d = int(parts[1])
            continue
        if key == "LUT_3D_SIZE" and len(parts) >= 2:
            n3d = int(parts[1])
            continue

        vals = _parse_floats(parts)
        if len(vals) == 3:
            data.append(vals)

    lut1d = None
    lut3d = None
    idx = 0

    if n1d > 0:
        if len(data) < idx + n1d:
            raise ValueError(f"Invalid .cube: not enough 1D entries (need {n1d})")
        lut1d = np.array(data[idx:idx + n1d], np.float32)
        idx += n1d

    if n3d > 0:
        need = n3d ** 3
        if len(data) < idx + need:
            raise ValueError(f"Invalid .cube: not enough 3D entries (need {need})")
        raw = np.array(data[idx:idx + need], np.float32)
        idx += need
        lut3d = np.zeros((n3d, n3d, n3d, 3), np.float32)
        k = 0
        for r in range(n3d):
            for g in range(n3d):
                for b in range(n3d):
                    lut3d[r, g, b, :] = raw[k]
                    k += 1

    if n1d <= 0 and n3d <= 0:
        raise ValueError("Invalid .cube: missing LUT_1D_SIZE or LUT_3D_SIZE")

    return CubeLUT(
        title=title or path.stem,
        domain_min=dom_min.astype(np.float32),
        domain_max=dom_max.astype(np.float32),
        lut1d=lut1d,
        lut3d=lut3d,
        size1d=n1d,
        size3d=n3d,
    )


def _lut_apply_1d(rgb01: np.ndarray, lut: np.ndarray, dom_min: np.ndarray, dom_max: np.ndarray) -> np.ndarray:
    n = lut.shape[0]
    x = (rgb01 - dom_min[None, None, :]) / np.maximum(dom_max - dom_min, EPS)[None, None, :]
    x = np.clip(x, 0.0, 1.0)
    t = x * (n - 1)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, n - 1)
    f = (t - i0).astype(np.float32)

    out = np.empty_like(rgb01, dtype=np.float32)
    for c in range(3):
        v0 = lut[i0[..., c], c]
        v1 = lut[i1[..., c], c]
        out[..., c] = (v0 + (v1 - v0) * f[..., c]).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _lut_apply_3d(rgb01: np.ndarray, lut3d: np.ndarray, dom_min: np.ndarray, dom_max: np.ndarray) -> np.ndarray:
    n = lut3d.shape[0]
    x = (rgb01 - dom_min[None, None, :]) / np.maximum(dom_max - dom_min, EPS)[None, None, :]
    x = np.clip(x, 0.0, 1.0)
    t = x * (n - 1)
    r = t[..., 0]
    g = t[..., 1]
    b = t[..., 2]

    r0 = np.floor(r).astype(np.int32); r1 = np.clip(r0 + 1, 0, n - 1)
    g0 = np.floor(g).astype(np.int32); g1 = np.clip(g0 + 1, 0, n - 1)
    b0 = np.floor(b).astype(np.int32); b1 = np.clip(b0 + 1, 0, n - 1)

    fr = (r - r0).astype(np.float32)
    fg = (g - g0).astype(np.float32)
    fb = (b - b0).astype(np.float32)

    c000 = lut3d[r0, g0, b0]
    c001 = lut3d[r0, g0, b1]
    c010 = lut3d[r0, g1, b0]
    c011 = lut3d[r0, g1, b1]
    c100 = lut3d[r1, g0, b0]
    c101 = lut3d[r1, g0, b1]
    c110 = lut3d[r1, g1, b0]
    c111 = lut3d[r1, g1, b1]

    # Trilinear interpolation
    c00 = c000 + (c001 - c000) * fb[..., None]
    c01 = c010 + (c011 - c010) * fb[..., None]
    c10 = c100 + (c101 - c100) * fb[..., None]
    c11 = c110 + (c111 - c110) * fb[..., None]

    c0 = c00 + (c01 - c00) * fg[..., None]
    c1 = c10 + (c11 - c10) * fg[..., None]

    out = c0 + (c1 - c0) * fr[..., None]
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def apply_lut(rgb01: np.ndarray, lut: CubeLUT, cfg: Config) -> np.ndarray:
    strength = float(np.clip(cfg.lut_strength, 0.0, 1.0))
    if strength <= 0:
        return rgb01

    x = rgb01.astype(np.float32)
    if lut.lut1d is not None:
        x = _lut_apply_1d(x, lut.lut1d, lut.domain_min, lut.domain_max)
    if lut.lut3d is not None:
        x = _lut_apply_3d(x, lut.lut3d, lut.domain_min, lut.domain_max)

    # Protect highlights/blacks by applying LUT mainly in midtones.
    if cfg.lut_protect_highlights or cfg.lut_protect_blacks:
        l = luma(rgb01)
        m = midtone_map(l)
        # Bias controls how "midtone only" this is.
        bias = float(np.clip(cfg.lut_midtone_bias, 0.0, 1.0))
        w = (bias * m + (1.0 - bias)).astype(np.float32)
        strength_map = (strength * w)[..., None]
    else:
        strength_map = np.full_like(rgb01, strength, dtype=np.float32)

    if cfg.lut_preserve_luma:
        # Luma-preserving blend: blend in LUT'd chroma while keeping original luma.
        out = rgb01 * (1.0 - strength_map) + x * strength_map
        l0 = luma(rgb01)
        l1 = luma(out)
        r = np.clip(l0 / np.maximum(l1, EPS), 0.0, 8.0)
        out = np.clip(out * r[..., None], 0.0, 1.0)
        return out.astype(np.float32)

    out = rgb01 * (1.0 - strength_map) + x * strength_map
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------- Enhancements ---------------------------

def detail_transfer(base01: np.ndarray, ai01: np.ndarray, W: Weights, cfg: Config, mods: Optional[MaterialMods]) -> np.ndarray:
    if ai01.shape != base01.shape:
        ai01 = resize(ai01, base01.shape[1], base01.shape[0], cv2.INTER_LANCZOS4)

    lb, la = luma(base01), luma(ai01)
    hp_b = lb - blur(lb, cfg.detail_sigma)
    hp_a = la - blur(la, cfg.detail_sigma)
    d = np.clip(hp_a - hp_b, -cfg.detail_clip, cfg.detail_clip)

    z = param_map(W.wfg, W.wmid, W.wbg, cfg.detail_fg, cfg.detail_mid, cfg.detail_bg)
    e = edge_map(lb)
    m = midtone_map(lb)
    s = cfg.detail_strength * z * (0.35 + 0.65 * e) * (0.40 + 0.60 * m)

    if mods is not None:
        if mods.detail_mult.shape != lb.shape:
            dm = resize(mods.detail_mult, lb.shape[1], lb.shape[0], cv2.INTER_LINEAR)
        else:
            dm = mods.detail_mult
        s = (s * dm).astype(np.float32)

    new_l = np.clip(lb + d * s, 0.0, 1.0)
    return apply_luma_ratio(base01, new_l, old_l=lb)


def apply_clarity(rgb01: np.ndarray, W: Weights, cfg: Config, mods: Optional[MaterialMods]) -> np.ndarray:
    l = luma(rgb01)
    hp = np.clip(l - blur(l, cfg.clarity_sigma), -cfg.clarity_clip, cfg.clarity_clip)
    z = param_map(W.wfg, W.wmid, W.wbg, cfg.clarity_fg, cfg.clarity_mid, cfg.clarity_bg)
    if mods is not None:
        cm = mods.clarity_mult
        if cm.shape != l.shape:
            cm = resize(cm, l.shape[1], l.shape[0], cv2.INTER_LINEAR)
        z = (z * cm).astype(np.float32)
    new_l = np.clip(l + hp * z * midtone_map(l), 0.0, 1.0)
    return apply_luma_ratio(rgb01, new_l, old_l=l)


def apply_sharpen(rgb01: np.ndarray, W: Weights, cfg: Config, mods: Optional[MaterialMods]) -> np.ndarray:
    l = luma(rgb01)
    det = l - blur(l, cfg.sharpen_sigma)
    det = det * (np.abs(det) >= cfg.sharpen_thresh)
    z = param_map(W.wfg, W.wmid, W.wbg, cfg.sharpen_fg, cfg.sharpen_mid, cfg.sharpen_bg)
    if mods is not None:
        sm = mods.sharpen_mult
        if sm.shape != l.shape:
            sm = resize(sm, l.shape[1], l.shape[0], cv2.INTER_LINEAR)
        z = (z * sm).astype(np.float32)
    new_l = np.clip(l + det * z * (0.30 + 0.70 * edge_map(l)), 0.0, 1.0)
    return apply_luma_ratio(rgb01, new_l, old_l=l)


def apply_temperature(rgb01: np.ndarray, temp: np.ndarray) -> np.ndarray:
    l0 = luma(rgb01)
    out = rgb01.copy().astype(np.float32)
    out[..., 0] *= (1.0 + temp)  # R
    out[..., 2] *= (1.0 - temp)  # B
    out = np.clip(out, 0.0, 1.5)
    l1 = luma(out)
    r = np.clip(l0 / np.maximum(l1, EPS), 0.0, 8.0)
    return np.clip((out * r[..., None]).astype(np.float32), 0.0, 1.0)


def apply_saturation(rgb01: np.ndarray, sat: np.ndarray) -> np.ndarray:
    l = luma(rgb01)[..., None]
    return np.clip((l + (rgb01 - l) * sat[..., None]).astype(np.float32), 0.0, 1.0)


def apply_exp_con(rgb01: np.ndarray, exp: np.ndarray, con: np.ndarray) -> np.ndarray:
    l0 = luma(rgb01)
    l = np.clip(l0 * exp, 0.0, 1.0)
    mid = midtone_map(l)
    c = 1.0 + (con - 1.0) * (0.35 + 0.65 * mid)
    new_l = np.clip(0.5 + (l - 0.5) * c, 0.0, 1.0)
    return apply_luma_ratio(rgb01, new_l, old_l=l0)


def grade_core(rgb01: np.ndarray, W: Weights, cfg: Config, mods: Optional[MaterialMods]) -> np.ndarray:
    temp = param_map(W.wfg, W.wmid, W.wbg, cfg.temp_fg, cfg.temp_mid, cfg.temp_bg)
    sat  = param_map(W.wfg, W.wmid, W.wbg, cfg.sat_fg,  cfg.sat_mid,  cfg.sat_bg)
    exp  = param_map(W.wfg, W.wmid, W.wbg, cfg.exp_fg,  cfg.exp_mid,  cfg.exp_bg)
    con  = param_map(W.wfg, W.wmid, W.wbg, cfg.con_fg,  cfg.con_mid,  cfg.con_bg)

    if mods is not None:
        # Ensure shapes match (they should)
        if mods.temp_offset.shape != temp.shape:
            t_off = resize(mods.temp_offset, temp.shape[1], temp.shape[0], cv2.INTER_LINEAR)
            s_mul = resize(mods.sat_mult, temp.shape[1], temp.shape[0], cv2.INTER_LINEAR)
            e_mul = resize(mods.exp_mult, temp.shape[1], temp.shape[0], cv2.INTER_LINEAR)
            c_mul = resize(mods.con_mult, temp.shape[1], temp.shape[0], cv2.INTER_LINEAR)
        else:
            t_off, s_mul, e_mul, c_mul = mods.temp_offset, mods.sat_mult, mods.exp_mult, mods.con_mult

        temp = (temp + t_off).astype(np.float32)
        sat = (sat * s_mul).astype(np.float32)
        exp = (exp * e_mul).astype(np.float32)
        con = (con * c_mul).astype(np.float32)

        # Safety clamps
        sat = np.clip(sat, 0.80, 1.35)
        exp = np.clip(exp, 0.90, 1.10)
        con = np.clip(con, 0.90, 1.15)
        temp = np.clip(temp, -0.05, 0.05)

    out = apply_temperature(rgb01, temp)
    out = apply_saturation(out, sat)
    out = apply_exp_con(out, exp, con)
    return out.astype(np.float32)


# --------------------------- Metrics ---------------------------

def metrics(rgb01: np.ndarray) -> Dict[str, float]:
    hi = float(np.mean(rgb01 >= 1.0 - 1e-6))
    lo = float(np.mean(rgb01 <= 0.0 + 1e-6))
    l = luma(rgb01)
    return {
        "clip_hi": hi,
        "clip_lo": lo,
        "l_mean": float(l.mean()),
        "l_p1": float(np.percentile(l, 1)),
        "l_p99": float(np.percentile(l, 99)),
    }


def mean_abs_rgb(a: np.ndarray, b: np.ndarray, max_samples: int = 250_000) -> float:
    """
    Fast, robust color distance proxy in RGB space.
    Samples pixels if the image is huge.
    """
    if a.shape != b.shape:
        raise ValueError("mean_abs_rgb requires same shape")
    h, w = a.shape[:2]
    n = h * w
    if n <= max_samples:
        return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    # stride sampling
    step = int(math.ceil(math.sqrt(n / max_samples)))
    return float(np.mean(np.abs(a[::step, ::step].astype(np.float32) - b[::step, ::step].astype(np.float32))))


def mean_abs_luma(a: np.ndarray, b: np.ndarray, max_samples: int = 250_000) -> float:
    if a.shape != b.shape:
        raise ValueError("mean_abs_luma requires same shape")
    la = luma(a)
    lb = luma(b)
    h, w = la.shape[:2]
    n = h * w
    if n <= max_samples:
        return float(np.mean(np.abs(la - lb)))
    step = int(math.ceil(math.sqrt(n / max_samples)))
    return float(np.mean(np.abs(la[::step, ::step] - lb[::step, ::step])))


# --------------------------- Runner ---------------------------

@dataclass
class Context:
    lut: Optional[CubeLUT] = None


def _serialize_config(cfg: Config) -> Dict[str, Any]:
    """Convert Config to JSON-serializable dict."""
    d = dataclasses.asdict(cfg)
    # Convert Path objects to strings (recursively handle nested structures)
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_paths(item) for item in obj)
        else:
            return obj
    
    return convert_paths(d)


def process_one(path: Path, cfg: Config, up: Upscaler, ctx: Context) -> Tuple[bool, Dict[str, Any]]:
    t0 = time.time()
    stage: Dict[str, float] = {}

    rep: Dict[str, Any] = {
        "input": str(path),
        "cfg": _serialize_config(cfg),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": getattr(cv2, "__version__", None),
            "tifffile": getattr(tifffile, "__version__", None),
        },
        "warnings": [],
        "outputs": {},
        "metrics": {},
        "stage_times_sec": stage,
    }

    try:
        t = time.time()
        rgb_u = read_rgb_tiff(path)
        rgb01 = to01(rgb_u)
        h, w = rgb01.shape[:2]
        stem = path.stem
        stage["read_input"] = time.time() - t

        out_h, out_w = h * cfg.upscale, w * cfg.upscale
        est_gb = (out_h * out_w * 3 * 4) / (1024 ** 3)  # float32 RGB buffer
        if est_gb > cfg.warn_float_gb:
            rep["warnings"].append(
                f"High memory risk: {out_w}x{out_h} float32 RGB ~{est_gb:.2f}GB (consider --upscale 2 or smaller --tile)"
            )

        # Depth + zone masks
        t = time.time()
        depth_path = cfg.depth_dir / f"{stem}_depth_raw_16bit.tiff"
        masks: Dict[str, np.ndarray] = {}
        for k in ("foreground", "midground", "background"):
            mp = cfg.depth_dir / f"{stem}_depth_zone_{k}.png"
            if mp.exists():
                try:
                    masks[k] = load_mask_any(mp)
                except Exception as e:
                    rep["warnings"].append(f"mask {mp.name}: {e}")

        depth = None
        if depth_path.exists():
            dpth = tifffile.imread(str(depth_path))
            if getattr(dpth, "ndim", 0) == 2:
                if dpth.dtype == np.uint16:
                    depth = norm_depth(dpth)
                else:
                    # normalize anyway
                    maxv = float(np.iinfo(dpth.dtype).max) if hasattr(np, "iinfo") and np.issubdtype(dpth.dtype, np.integer) else float(np.max(dpth))
                    if maxv <= 0:
                        maxv = 1.0
                    dd = np.clip(dpth.astype(np.float32) / maxv, 0.0, 1.0)
                    depth = dd.astype(np.float32)
            else:
                rep["warnings"].append(f"depth map not 2D: {depth_path.name}")
        else:
            if cfg.strict_depth:
                raise FileNotFoundError(f"Missing depth map: {depth_path}")

        W0 = weights_from_assets(h, w, depth, masks, cfg)
        rep["depth_weights_source"] = W0.source
        stage["depth_weights"] = time.time() - t

        # Material masks -> per-pixel mods
        t = time.time()
        m_masks = load_material_masks(stem, cfg.depth_dir, cfg.surfaces) if cfg.enable_material else {}
        mods0 = build_material_mods(h, w, m_masks, cfg)
        if mods0 is not None:
            rep["material_source"] = mods0.source
        stage["material_mods"] = time.time() - t

        # MASTER (original resolution)
        t = time.time()
        master = grade_core(rgb01.copy(), W0, cfg, mods0)
        if mods0 is not None:
            master = apply_material_highlight_compression(master, mods0.highlight_compress)
        if cfg.enable_lut and ctx.lut is not None:
            master = apply_lut(master, ctx.lut, cfg)
        master = soft_clip01(master, cfg.soft_clip_knee)
        rep["metrics"]["master"] = metrics(master)
        stage["master_grade"] = time.time() - t

        # UPSCALED (optional)
        out_up = None
        if cfg.save_upscaled_16bit or cfg.save_marketing_png or cfg.save_preview_jpg:
            t = time.time()
            base = resize(rgb01, w * cfg.upscale, h * cfg.upscale, cv2.INTER_LANCZOS4).astype(np.float32)
            base = np.clip(base, 0.0, 1.0)
            W = upsample_weights(W0, base.shape[0], base.shape[1])
            stage["base_resize"] = time.time() - t

            # Upsample material modifiers to final-res
            mods = None
            if mods0 is not None:
                mods = MaterialMods(
                    temp_offset=resize(mods0.temp_offset, base.shape[1], base.shape[0], cv2.INTER_LINEAR),
                    sat_mult=resize(mods0.sat_mult, base.shape[1], base.shape[0], cv2.INTER_LINEAR),
                    exp_mult=resize(mods0.exp_mult, base.shape[1], base.shape[0], cv2.INTER_LINEAR),
                    con_mult=resize(mods0.con_mult, base.shape[1], base.shape[0], cv2.INTER_LINEAR),
                    detail_mult=resize(mods0.detail_mult, base.shape[1], base.shape[0], cv2.INTER_LINEAR),
                    clarity_mult=resize(mods0.clarity_mult, base.shape[1], base.shape[0], cv2.INTER_LINEAR),
                    sharpen_mult=resize(mods0.sharpen_mult, base.shape[1], base.shape[0], cv2.INTER_LINEAR),
                    highlight_compress=resize(mods0.highlight_compress, base.shape[1], base.shape[0], cv2.INTER_LINEAR),
                    source=mods0.source,
                )

            # AI upscale -> detail layer
            if cfg.backend.lower() != "none":
                t = time.time()
                ai = up.upscale(rgb01)
                stage["ai_upscale"] = time.time() - t

                # Guard-rail: if AI deviates wildly, do not inject its details.
                if cfg.validate_ai:
                    try:
                        ai_r = ai
                        if ai_r.shape != base.shape:
                            ai_r = resize(ai_r, base.shape[1], base.shape[0], cv2.INTER_LANCZOS4)
                        cdev = mean_abs_rgb(ai_r, base)
                        ldev = mean_abs_luma(ai_r, base)
                        rep["metrics"]["ai_color_mean_abs"] = float(cdev)
                        rep["metrics"]["ai_luma_mean_abs"] = float(ldev)
                        if cdev > cfg.ai_color_warn or ldev > cfg.ai_luma_warn:
                            rep["warnings"].append(f"AI deviation warn: mean_abs_rgb={cdev:.3f}, mean_abs_luma={ldev:.3f}")
                        if cdev > cfg.ai_color_fail or ldev > cfg.ai_luma_fail:
                            rep["warnings"].append("AI deviation too high; skipping AI detail injection (using base resize).")
                            out_up = base
                        else:
                            t = time.time()
                            out_up = detail_transfer(base, ai, W, cfg, mods)
                            stage["detail_transfer"] = time.time() - t
                    except Exception as e:
                        rep["warnings"].append(f"AI validation failed; continuing without validation: {e}")
                        t = time.time()
                        out_up = detail_transfer(base, ai, W, cfg, mods)
                        stage["detail_transfer"] = time.time() - t
                else:
                    t = time.time()
                    out_up = detail_transfer(base, ai, W, cfg, mods)
                    stage["detail_transfer"] = time.time() - t
            else:
                out_up = base

            # Clarity + sharpen + grade + luxury stack
            t = time.time()
            out_up = apply_clarity(out_up, W, cfg, mods)
            out_up = apply_sharpen(out_up, W, cfg, mods)
            out_up = grade_core(out_up, W, cfg, mods)
            if mods is not None:
                out_up = apply_material_highlight_compression(out_up, mods.highlight_compress)
            if cfg.enable_lut and ctx.lut is not None:
                out_up = apply_lut(out_up, ctx.lut, cfg)
            out_up = soft_clip01(out_up, cfg.soft_clip_knee)
            rep["metrics"]["upscaled"] = metrics(out_up)
            stage["final_grade"] = time.time() - t

        # Writes
        t = time.time()
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.save_master:
            op = cfg.output_dir / f"{stem}_MASTER_16bit.tiff"
            if not (cfg.skip_existing and op.exists() and not cfg.overwrite):
                write_tiff_u16(op, u16(master))
                rep["outputs"]["master_16bit"] = str(op)
            else:
                rep["warnings"].append(f"skip existing {op.name}")

        if out_up is not None and cfg.save_upscaled_16bit:
            op = cfg.output_dir / f"{stem}_UPSCALED_16bit.tiff"
            if not (cfg.skip_existing and op.exists() and not cfg.overwrite):
                write_tiff_u16(op, u16(out_up))
                rep["outputs"]["upscaled_16bit"] = str(op)
            else:
                rep["warnings"].append(f"skip existing {op.name}")

        if out_up is not None and cfg.save_marketing_png:
            op = cfg.output_dir / f"{stem}_MARKETING.png"
            if not (cfg.skip_existing and op.exists() and not cfg.overwrite):
                write_png_u8(op, u8(out_up))
                rep["outputs"]["marketing_png"] = str(op)
            else:
                rep["warnings"].append(f"skip existing {op.name}")

        if out_up is not None and cfg.save_preview_jpg:
            op = cfg.output_dir / f"{stem}_PREVIEW.jpg"
            if not (cfg.skip_existing and op.exists() and not cfg.overwrite):
                write_preview_jpg(op, u8(out_up), cfg.preview_scale)
                rep["outputs"]["preview_jpg"] = str(op)
            else:
                rep["warnings"].append(f"skip existing {op.name}")

        stage["write_outputs"] = time.time() - t

        rep["elapsed_sec"] = float(time.time() - t0)

        if cfg.save_report:
            rp = cfg.output_dir / f"{stem}_report.json"
            tmp = rp.with_suffix(rp.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
            tmp.replace(rp)
            rep["outputs"]["report_json"] = str(rp)

        return True, rep

    except Exception as e:
        rep["elapsed_sec"] = float(time.time() - t0)
        rep["error"] = f"{type(e).__name__}: {e}"
        return False, rep




def write_batch_report_md(cfg: Config, reps: List[Dict[str, Any]], ok: int, total: int, out_path: Path) -> None:
    """
    Human-friendly summary (non-authoritative). JSON remains the source of truth.
    """
    lines: List[str] = []
    lines.append("# Batch Processing Report")
    lines.append("")
    lines.append(f"- Preset: `{cfg.preset.value}`")
    lines.append(f"- Upscale: `{cfg.upscale}x`")
    lines.append(f"- Backend: `{cfg.backend}`")
    if cfg.enable_lut and cfg.lut_path:
        lines.append(f"- LUT: `{cfg.lut_path.name}` (strength {cfg.lut_strength:.2f})")
    else:
        lines.append(f"- LUT: `disabled`")
    if cfg.enable_material and cfg.material_strength > 0:
        lines.append(f"- Material response: `enabled` (strength {cfg.material_strength:.2f}) surfaces={list(cfg.surfaces)}")
    else:
        lines.append(f"- Material response: `disabled`")
    lines.append("")
    lines.append(f"## Summary")
    lines.append(f"- Succeeded: **{ok} / {total}**")
    lines.append("")

    # Table
    lines.append("| Status | Image | Time (s) | Warnings | Outputs |")
    lines.append("|---:|---|---:|---:|---|")
    for rep in reps:
        name = Path(rep.get("input", "")).name
        status = "✅" if "error" not in rep else "❌"
        sec = float(rep.get("elapsed_sec", 0.0) or 0.0)
        warn_n = len(rep.get("warnings", []) or [])
        outs = rep.get("outputs", {}) or {}
        # show just filenames to keep readable
        out_names = []
        for k in ("master_16bit", "upscaled_16bit", "marketing_png", "preview_jpg"):
            if k in outs:
                out_names.append(Path(outs[k]).name)
        out_s = ", ".join(out_names) if out_names else ""
        lines.append(f"| {status} | {name} | {sec:.2f} | {warn_n} | {out_s} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(out_path)


# --------------------------- CLI ---------------------------

def parse(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=None, help="Input TIFF file or directory")
    p.add_argument("--input-dir", type=Path, default=None, help="Input directory of TIFFs (legacy)")
    p.add_argument("--depth-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--preset", type=str, default=Preset.PHOTO_REALISTIC.value,
                   choices=[e.value for e in Preset], help="Curated look preset")

    p.add_argument("--upscale", type=int, default=4, choices=[2, 4])
    p.add_argument("--backend", type=str, default="realesrgan", choices=["realesrgan", "onnx", "none"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--model-sha256", type=str, default=None)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--tile-pad", type=int, default=16)
    p.add_argument("--half", action="store_true")

    p.add_argument("--strict-depth", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-skip-existing", action="store_true")

    # Outputs
    p.add_argument("--no-master", action="store_true")
    p.add_argument("--no-upscaled-16bit", action="store_true")
    p.add_argument("--no-marketing", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--preview-scale", type=float, default=0.25)

    # Material
    p.add_argument("--no-material", action="store_true")
    p.add_argument("--material-strength", type=float, default=0.75)
    p.add_argument("--surfaces", type=str, default="wood,metal,glass,stone",
                   help="Comma-separated surface masks to search for (e.g. wood,metal,glass,stone,foliage,sky)")
    p.add_argument("--material-mask-soften-sigma", type=float, default=2.0)

    # LUT
    p.add_argument("--lut-path", type=Path, default=None, help="Path to .cube LUT")
    p.add_argument("--lut-strength", type=float, default=0.70)
    p.add_argument("--lut-no-preserve-luma", action="store_true")
    p.add_argument("--lut-no-protect-highlights", action="store_true")
    p.add_argument("--lut-no-protect-blacks", action="store_true")
    p.add_argument("--lut-midtone-bias", type=float, default=0.85)

    # Guard rails
    p.add_argument("--no-validate-ai", action="store_true")
    p.add_argument("--ai-color-warn", type=float, default=0.06)
    p.add_argument("--ai-color-fail", type=float, default=0.12)
    p.add_argument("--ai-luma-warn", type=float, default=0.06)
    p.add_argument("--ai-luma-fail", type=float, default=0.12)

    a = p.parse_args(argv)

    inp = a.input if a.input is not None else a.input_dir
    if inp is None:
        p.error('You must provide --input (file/dir) or --input-dir')

    cfg = Config(
        input_dir=inp,
        depth_dir=a.depth_dir,
        output_dir=a.output_dir,
        preset=Preset(a.preset),
        upscale=a.upscale,
        backend=a.backend,
        device=a.device,
        model_path=a.model_path,
        model_sha256=a.model_sha256,
        tile=a.tile,
        tile_pad=a.tile_pad,
        half=bool(a.half),
        strict_depth=bool(a.strict_depth),
        overwrite=bool(a.overwrite),
        skip_existing=not bool(a.no_skip_existing),
        save_master=not bool(a.no_master),
        save_upscaled_16bit=not bool(a.no_upscaled_16bit),
        save_marketing_png=not bool(a.no_marketing),
        save_preview_jpg=not bool(a.no_preview),
        save_report=not bool(a.no_report),
        preview_scale=float(a.preview_scale),

        enable_material=not bool(a.no_material),
        material_strength=float(a.material_strength),
        surfaces=tuple([s.strip() for s in str(a.surfaces).split(",") if s.strip()]),
        material_mask_soften_sigma=float(a.material_mask_soften_sigma),

        enable_lut=bool(a.lut_path),
        lut_path=a.lut_path,
        lut_strength=float(a.lut_strength),
        lut_preserve_luma=not bool(a.lut_no_preserve_luma),
        lut_protect_highlights=not bool(a.lut_no_protect_highlights),
        lut_protect_blacks=not bool(a.lut_no_protect_blacks),
        lut_midtone_bias=float(a.lut_midtone_bias),

        validate_ai=not bool(a.no_validate_ai),
        ai_color_warn=float(a.ai_color_warn),
        ai_color_fail=float(a.ai_color_fail),
        ai_luma_warn=float(a.ai_luma_warn),
        ai_luma_fail=float(a.ai_luma_fail),
    )

    cfg.apply_preset()
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    _need_deps()
    cfg = parse(argv)

    if not cfg.input_dir.exists():
        print(f"Input path not found: {cfg.input_dir}")
        return 2
    if not cfg.depth_dir.exists():
        print(f"Depth dir not found: {cfg.depth_dir}")
        return 2
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load LUT once (if enabled)
    ctx = Context(lut=None)
    if cfg.enable_lut and cfg.lut_path:
        if not cfg.lut_path.exists():
            print(f"⚠️  LUT not found: {cfg.lut_path} (continuing without LUT)")
        else:
            try:
                ctx.lut = load_cube_lut(cfg.lut_path)
            except Exception as e:
                print(f"⚠️  LUT failed to load ({cfg.lut_path.name}): {e} (continuing without LUT)")

    up = build_upscaler(cfg)

    if cfg.input_dir.is_file():
        tiffs = [cfg.input_dir]
    else:
        tiffs = sorted([p for p in cfg.input_dir.glob("*.tif*") if p.is_file()])

    if not tiffs:
        print(f"No TIFFs found in {cfg.input_dir}")
        return 2

    reps: List[Dict[str, Any]] = []
    ok = 0
    for pth in tqdm(tiffs, desc="Processing", unit="img"):
        success, rep = process_one(pth, cfg, up, ctx)
        reps.append(rep)
        if success:
            ok += 1
            print(f"\n✅ {pth.name}: {rep.get('elapsed_sec', 0):.2f}s")
        else:
            print(f"\n❌ {pth.name}: {rep.get('error')}")

    batch = {"ok": ok, "total": len(tiffs), "output_dir": str(cfg.output_dir), "images": reps, "epoch": time.time()}
    bp = cfg.output_dir / "_batch_report.json"
    tmp = bp.with_suffix(bp.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2)
    tmp.replace(bp)

    # Optional human-readable summary
    try:
        write_batch_report_md(cfg, reps, ok, len(tiffs), cfg.output_dir / "batch_report.md")
    except Exception as e:
        print(f"⚠️  batch_report.md failed: {e}")

    print(f"Done: {ok}/{len(tiffs)} succeeded. Batch report: {bp.name}")
    return 0 if ok == len(tiffs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
