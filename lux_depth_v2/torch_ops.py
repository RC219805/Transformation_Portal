from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


EPS = 1e-6


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for V2 GPU pipeline. Install torch.")


def pick_device(device: str = "auto") -> "torch.device":
    require_torch()
    d = (device or "auto").lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda")
    if d == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch(cudnn_benchmark: bool = True) -> None:
    require_torch()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


@contextmanager
def maybe_autocast(enabled: bool, device: "torch.device"):
    require_torch()
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def to_torch_rgb(rgb01: np.ndarray, device: "torch.device") -> "torch.Tensor":
    """HxWx3 float32 -> 1x3xHxW float32"""
    require_torch()
    x = np.asarray(rgb01, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB float array")
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device=device, dtype=torch.float32)


def from_torch_rgb(rgb: "torch.Tensor") -> np.ndarray:
    """1x3xHxW -> HxWx3 float32"""
    require_torch()
    x = rgb.detach().clamp(0.0, 1.0)
    x = x[0].permute(1, 2, 0).contiguous()
    return x.to("cpu").numpy().astype(np.float32)


def smoothstep(edge0: float, edge1: float, x: "torch.Tensor") -> "torch.Tensor":
    require_torch()
    t = ((x - edge0) / max(edge1 - edge0, EPS)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luma(rgb: "torch.Tensor") -> "torch.Tensor":
    """1x3xHxW -> 1x1xHxW"""
    require_torch()
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return (r * 0.2126 + g * 0.7152 + b * 0.0722).to(dtype=torch.float32)


def midtone_map(l: "torch.Tensor") -> "torch.Tensor":
    a = smoothstep(0.08, 0.35, l)
    b = 1.0 - smoothstep(0.65, 0.98, l)
    return (a * b).to(dtype=torch.float32)


def _gaussian_kernel1d(sigma: float, device: "torch.device", dtype: "torch.dtype") -> "torch.Tensor":
    require_torch()
    sigma = float(max(0.0, sigma))
    if sigma <= 0.0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = int(round(sigma * 3.0))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / torch.sum(k)
    return k


def gaussian_blur(x: "torch.Tensor", sigma: float, autocast: bool = False) -> "torch.Tensor":
    """Gaussian blur for 1xCxHxW tensors."""
    require_torch()
    if sigma <= 0:
        return x
    device = x.device
    # CPU fast-path (OpenCV) – dramatically faster than torch conv on many CPU runtimes.
    if device.type == "cpu" and 'cv2' in globals() and cv2 is not None:
        arr = x.detach().to(dtype=torch.float32).numpy()  # 1xCxHxW
        k = int(round(float(sigma) * 3.0)) * 2 + 1
        k = max(3, k)
        out = np.empty_like(arr, dtype=np.float32)
        for ch in range(arr.shape[1]):
            out[0, ch] = cv2.GaussianBlur(arr[0, ch], (k, k), float(sigma), borderType=cv2.BORDER_REFLECT)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)
    # Always compute kernel in float32 for stability.
    k1 = _gaussian_kernel1d(sigma, device=device, dtype=torch.float32)
    k1 = k1.to(dtype=x.dtype)
    radius = (k1.numel() - 1) // 2

    c = x.shape[1]
    # Depthwise separable convolution
    # Horizontal
    kh = k1.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    # Vertical
    kv = k1.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

    with maybe_autocast(autocast, device):
        xh = F.pad(x, (radius, radius, 0, 0), mode="reflect")
        xh = F.conv2d(xh, kh, groups=c)
        xv = F.pad(xh, (0, 0, radius, radius), mode="reflect")
        xv = F.conv2d(xv, kv, groups=c)
    return xv


def resize(x: "torch.Tensor", size_hw: Tuple[int, int], mode: str = "bicubic", autocast: bool = False) -> "torch.Tensor":
    require_torch()
    h, w = int(size_hw[0]), int(size_hw[1])
    device = x.device
    with maybe_autocast(autocast, device):
        return F.interpolate(x, size=(h, w), mode=mode, align_corners=False)


def soft_clip01(rgb: "torch.Tensor", knee: float) -> "torch.Tensor":
    require_torch()
    k = float(max(0.0, min(1.0, knee)))
    out = rgb.to(dtype=torch.float32)
    if k >= 1.0:
        return out.clamp(0.0, 1.0)
    x = (out - k).clamp(min=0.0) / max(1.0 - k, EPS)
    out = out - x * x * 0.5 * (1.0 - k)
    return out.clamp(0.0, 1.0)


def _percentile_approx(t: "torch.Tensor", q: float, max_samples: int = 200_000) -> "torch.Tensor":
    """Approximate percentile by random sampling to avoid full sort cost."""
    require_torch()
    flat = t.reshape(-1)
    n = flat.numel()
    if n <= max_samples:
        return torch.quantile(flat, q)
    # sample without replacement would be ideal but replacement is fine here
    idx = torch.randint(0, n, (max_samples,), device=flat.device)
    sample = flat[idx]
    return torch.quantile(sample, q)


def edge_map(l: "torch.Tensor", autocast: bool = False) -> "torch.Tensor":
    """Compute an edge magnitude map in 0..1 from 1x1xHxW luma."""
    require_torch()
    device = l.device
    # CPU fast-path (OpenCV)
    if device.type == "cpu" and 'cv2' in globals() and cv2 is not None:
        arr = l.detach().to(dtype=torch.float32)[0, 0].numpy()
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy + float(EPS))
        p = np.percentile(mag, 99.0)
        if p <= float(EPS):
            em = np.zeros_like(mag, dtype=np.float32)
        else:
            em = np.clip(mag / float(p), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(em).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=torch.float32) / 8.0
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=torch.float32) / 8.0
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    with maybe_autocast(autocast, device):
        lp = F.pad(l, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(lp, kx)
        gy = F.conv2d(lp, ky)
        mag = torch.sqrt(gx * gx + gy * gy + EPS)
    p = _percentile_approx(mag, 0.99)
    if float(p.item()) <= EPS:
        return torch.zeros_like(mag, dtype=torch.float32)
    return (mag / p).clamp(0.0, 1.0).to(dtype=torch.float32)


def apply_luma_ratio(rgb: "torch.Tensor", new_l: "torch.Tensor", old_l: Optional["torch.Tensor"] = None) -> "torch.Tensor":
    require_torch()
    old = luma(rgb) if old_l is None else old_l
    ratio = (new_l / (old + EPS)).clamp(0.0, 8.0).to(dtype=torch.float32)
    out = rgb.to(dtype=torch.float32) * ratio
    return out.clamp(0.0, 1.0)


def param_map(wfg: "torch.Tensor", wmid: "torch.Tensor", wbg: "torch.Tensor", fg: float, mid: float, bg: float) -> "torch.Tensor":
    require_torch()
    return (wfg * float(fg) + wmid * float(mid) + wbg * float(bg)).to(dtype=torch.float32)


def apply_temperature(rgb: "torch.Tensor", temp: "torch.Tensor") -> "torch.Tensor":
    """Luma-preserving warm/cool shift."""
    require_torch()
    l0 = luma(rgb)
    out = rgb.to(dtype=torch.float32).clone()
    out[:, 0:1] *= (1.0 + temp)
    out[:, 2:3] *= (1.0 - temp)
    out = out.clamp(0.0, 1.5)
    l1 = luma(out)
    r = (l0 / (l1 + EPS)).clamp(0.0, 8.0)
    return (out * r).clamp(0.0, 1.0)


def apply_saturation(rgb: "torch.Tensor", sat: "torch.Tensor") -> "torch.Tensor":
    require_torch()
    l = luma(rgb)
    return (l + (rgb - l) * sat).clamp(0.0, 1.0).to(dtype=torch.float32)


def apply_exp_con(rgb: "torch.Tensor", exp: "torch.Tensor", con: "torch.Tensor") -> "torch.Tensor":
    require_torch()
    l0 = luma(rgb)
    l = (l0 * exp).clamp(0.0, 1.0)
    mid = midtone_map(l)
    c = 1.0 + (con - 1.0) * (0.35 + 0.65 * mid)
    new_l = (0.5 + (l - 0.5) * c).clamp(0.0, 1.0)
    return apply_luma_ratio(rgb, new_l, old_l=l0)


@dataclass
class GradeMaps:
    temp: "torch.Tensor"  # 1x1xHxW
    sat: "torch.Tensor"
    exp: "torch.Tensor"
    con: "torch.Tensor"


def grade_core(rgb: "torch.Tensor", wfg: "torch.Tensor", wmid: "torch.Tensor", wbg: "torch.Tensor",
              cfg, mods: Optional[object] = None) -> "torch.Tensor":
    """Depth-aware grading core (temp, sat, exp, con)."""
    require_torch()
    temp = param_map(wfg, wmid, wbg, cfg.temp_fg, cfg.temp_mid, cfg.temp_bg)
    sat  = param_map(wfg, wmid, wbg, cfg.sat_fg,  cfg.sat_mid,  cfg.sat_bg)
    exp  = param_map(wfg, wmid, wbg, cfg.exp_fg,  cfg.exp_mid,  cfg.exp_bg)
    con  = param_map(wfg, wmid, wbg, cfg.con_fg,  cfg.con_mid,  cfg.con_bg)

    # Material mods (if present) are expected to have torch tensors matching (1,1,H,W)
    if mods is not None:
        try:
            temp = temp + mods.temp_offset
            sat = sat * mods.sat_mult
            exp = exp * mods.exp_mult
            con = con * mods.con_mult
        except Exception:
            pass

        # Safety clamps
        sat = sat.clamp(0.80, 1.35)
        exp = exp.clamp(0.90, 1.10)
        con = con.clamp(0.90, 1.15)
        temp = temp.clamp(-0.05, 0.05)

    out = apply_temperature(rgb, temp)
    out = apply_saturation(out, sat)
    out = apply_exp_con(out, exp, con)
    return out.to(dtype=torch.float32)


def detail_transfer(base: "torch.Tensor", ai: "torch.Tensor", wfg: "torch.Tensor", wmid: "torch.Tensor", wbg: "torch.Tensor",
                    cfg, mods: Optional[object] = None, autocast: bool = False) -> "torch.Tensor":
    """Inject controlled AI detail into base luminance."""
    require_torch()
    if ai.shape != base.shape:
        ai = resize(ai, (base.shape[2], base.shape[3]), mode="bicubic", autocast=autocast)

    lb, la = luma(base), luma(ai)
    hp_b = lb - gaussian_blur(lb, cfg.detail_sigma, autocast=autocast)
    hp_a = la - gaussian_blur(la, cfg.detail_sigma, autocast=autocast)
    d = (hp_a - hp_b).clamp(-cfg.detail_clip, cfg.detail_clip)

    z = param_map(wfg, wmid, wbg, cfg.detail_fg, cfg.detail_mid, cfg.detail_bg)
    e = edge_map(lb, autocast=autocast)
    m = midtone_map(lb)
    s = float(cfg.detail_strength) * z * (0.35 + 0.65 * e) * (0.40 + 0.60 * m)

    if mods is not None:
        try:
            s = s * mods.detail_mult
        except Exception:
            pass

    new_l = (lb + d * s).clamp(0.0, 1.0)
    return apply_luma_ratio(base, new_l, old_l=lb)


def apply_clarity(rgb: "torch.Tensor", wfg: "torch.Tensor", wmid: "torch.Tensor", wbg: "torch.Tensor",
                  cfg, mods: Optional[object] = None, autocast: bool = False) -> "torch.Tensor":
    require_torch()
    l = luma(rgb)
    hp = (l - gaussian_blur(l, cfg.clarity_sigma, autocast=autocast)).clamp(-cfg.clarity_clip, cfg.clarity_clip)
    z = param_map(wfg, wmid, wbg, cfg.clarity_fg, cfg.clarity_mid, cfg.clarity_bg)

    if mods is not None:
        try:
            z = z * mods.clarity_mult
        except Exception:
            pass

    new_l = (l + hp * z * midtone_map(l)).clamp(0.0, 1.0)
    return apply_luma_ratio(rgb, new_l, old_l=l)


def apply_sharpen(rgb: "torch.Tensor", wfg: "torch.Tensor", wmid: "torch.Tensor", wbg: "torch.Tensor",
                  cfg, mods: Optional[object] = None, autocast: bool = False) -> "torch.Tensor":
    require_torch()
    l = luma(rgb)
    det = l - gaussian_blur(l, cfg.sharpen_sigma, autocast=autocast)
    det = det * (det.abs() >= float(cfg.sharpen_thresh))
    z = param_map(wfg, wmid, wbg, cfg.sharpen_fg, cfg.sharpen_mid, cfg.sharpen_bg)

    if mods is not None:
        try:
            z = z * mods.sharpen_mult
        except Exception:
            pass

    new_l = (l + det * z * (0.30 + 0.70 * edge_map(l, autocast=autocast))).clamp(0.0, 1.0)
    return apply_luma_ratio(rgb, new_l, old_l=l)


def material_highlight_compress(rgb: "torch.Tensor", hi_comp: Optional["torch.Tensor"], knee: float = 0.85) -> "torch.Tensor":
    require_torch()
    if hi_comp is None:
        return rgb
    if float(hi_comp.max().item()) <= 0:
        return rgb
    l = luma(rgb)
    t = smoothstep(knee, 1.0, l)
    new_l = (l - (hi_comp * t * t) * 0.05).clamp(0.0, 1.0)
    return apply_luma_ratio(rgb, new_l, old_l=l)


def mean_abs_rgb(a: "torch.Tensor", b: "torch.Tensor", max_samples: int = 250_000) -> float:
    """Mean absolute RGB difference (approx by sampling)."""
    require_torch()
    if a.shape != b.shape:
        raise ValueError("mean_abs_rgb: shapes differ")
    # 1x3xHxW
    diff = (a.to(dtype=torch.float32) - b.to(dtype=torch.float32)).abs()
    flat = diff.permute(0,2,3,1).reshape(-1, 3)
    n = flat.shape[0]
    if n <= max_samples:
        return float(flat.mean().item())
    idx = torch.randint(0, n, (max_samples,), device=a.device)
    return float(flat[idx].mean().item())


def mean_abs_luma(a: "torch.Tensor", b: "torch.Tensor", max_samples: int = 250_000) -> float:
    require_torch()
    if a.shape != b.shape:
        raise ValueError("mean_abs_luma: shapes differ")
    la = luma(a)
    lb = luma(b)
    diff = (la - lb).abs().reshape(-1)
    n = diff.numel()
    if n <= max_samples:
        return float(diff.mean().item())
    idx = torch.randint(0, n, (max_samples,), device=a.device)
    return float(diff[idx].mean().item())


class Tiler:
    """Utility to run post-processing on large images in overlapping tiles.

    The callback receives:
      - tile tensor: 1xCx(ya1-ya0)x(xa1-xa0) including overlap
      - ya0, xa0, ya1, xa1: absolute coords of tile-with-overlap in the full image
      - y0, x0, y1, x1: absolute coords of core (non-overlap) area to write back
    """

    def __init__(self, tile: int, overlap: int):
        self.tile = int(tile)
        self.overlap = int(max(0, overlap))

    def run(self, rgb: "torch.Tensor", fn: Callable[["torch.Tensor", int, int, int, int, int, int, int, int], "torch.Tensor"]) -> "torch.Tensor":
        require_torch()
        if self.tile <= 0:
            # degrade gracefully
            return fn(rgb, 0, 0, rgb.shape[2], rgb.shape[3], 0, 0, rgb.shape[2], rgb.shape[3])

        b, c, h, w = rgb.shape
        tile = self.tile
        ov = self.overlap

        out = torch.empty_like(rgb, dtype=torch.float32)
        # Process tiles sequentially to keep VRAM bounded
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)

                # region with overlap
                ya0 = max(y0 - ov, 0)
                xa0 = max(x0 - ov, 0)
                ya1 = min(y1 + ov, h)
                xa1 = min(x1 + ov, w)

                tile_in = rgb[:, :, ya0:ya1, xa0:xa1]
                tile_out = fn(tile_in, ya0, xa0, ya1, xa1, y0, x0, y1, x1)

                # crop back to core
                cy0 = y0 - ya0
                cx0 = x0 - xa0
                cy1 = cy0 + (y1 - y0)
                cx1 = cx0 + (x1 - x0)
                out[:, :, y0:y1, x0:x1] = tile_out[:, :, cy0:cy1, cx0:cx1]
        return out
