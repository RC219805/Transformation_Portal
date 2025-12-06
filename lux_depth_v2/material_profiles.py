from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from . import torch_ops


@dataclass(frozen=True)
class SurfaceProfile:
    # Additive offset or multiplicative scales applied per-surface.
    temp_offset: float = 0.0
    sat_mult: float = 1.0
    exp_mult: float = 1.0
    con_mult: float = 1.0
    detail_mult: float = 1.0
    clarity_mult: float = 1.0
    sharpen_mult: float = 1.0
    highlight_compress: float = 0.0  # 0..1


# Conservatively tuned; intended to be subtle and photorealistic.
SURFACE_PROFILES: Dict[str, SurfaceProfile] = {
    "wood":    SurfaceProfile(temp_offset=0.004, sat_mult=1.05, exp_mult=1.000, con_mult=1.010, detail_mult=1.08, clarity_mult=1.04, sharpen_mult=1.03, highlight_compress=0.00),
    "metal":   SurfaceProfile(temp_offset=-0.002, sat_mult=0.98, exp_mult=1.000, con_mult=1.025, detail_mult=1.05, clarity_mult=1.02, sharpen_mult=1.05, highlight_compress=0.00),
    "glass":   SurfaceProfile(temp_offset=-0.002, sat_mult=0.98, exp_mult=1.000, con_mult=1.010, detail_mult=0.95, clarity_mult=0.82, sharpen_mult=0.78, highlight_compress=0.20),
    "stone":   SurfaceProfile(temp_offset=-0.001, sat_mult=0.99, exp_mult=1.000, con_mult=1.020, detail_mult=1.05, clarity_mult=1.03, sharpen_mult=1.02, highlight_compress=0.00),
    "sky":     SurfaceProfile(temp_offset=-0.003, sat_mult=1.02, exp_mult=1.000, con_mult=1.010, detail_mult=0.90, clarity_mult=0.78, sharpen_mult=0.75, highlight_compress=0.15),
    "foliage": SurfaceProfile(temp_offset=0.000, sat_mult=1.03, exp_mult=1.000, con_mult=1.015, detail_mult=1.02, clarity_mult=1.02, sharpen_mult=1.02, highlight_compress=0.00),
}


@dataclass
class MaterialMods:
    temp_offset: "torch_ops.torch.Tensor"      # 1x1xHxW
    sat_mult: "torch_ops.torch.Tensor"         # 1x1xHxW
    exp_mult: "torch_ops.torch.Tensor"
    con_mult: "torch_ops.torch.Tensor"
    detail_mult: "torch_ops.torch.Tensor"
    clarity_mult: "torch_ops.torch.Tensor"
    sharpen_mult: "torch_ops.torch.Tensor"
    highlight_compress: "torch_ops.torch.Tensor"  # 1x1xHxW (0..1)
    source: str


def _blend_add(cur: "torch_ops.torch.Tensor", add: float, w: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
    return (cur + w * float(add)).to(dtype=torch_ops.torch.float32)


def _blend_mult(cur: "torch_ops.torch.Tensor", mult: float, w: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
    # cur * lerp(1, mult, w) == cur * (1 + w*(mult-1))
    return (cur * (1.0 + w * (float(mult) - 1.0))).to(dtype=torch_ops.torch.float32)


def build_material_mods(
    masks: Dict[str, "torch_ops.torch.Tensor"],
    cfg,
) -> Optional[MaterialMods]:
    """Build per-pixel material modification maps from masks."""
    torch_ops.require_torch()
    if not bool(getattr(cfg, "enable_material", True)):
        return None
    if not masks:
        return None

    # choose shape from any mask
    any_mask = next(iter(masks.values()))
    device = any_mask.device
    _, _, h, w = any_mask.shape

    z = torch_ops.torch.zeros((1, 1, h, w), device=device, dtype=torch_ops.torch.float32)
    o = torch_ops.torch.ones((1, 1, h, w), device=device, dtype=torch_ops.torch.float32)

    temp_off = z.clone()
    sat = o.clone()
    exp = o.clone()
    con = o.clone()
    detail = o.clone()
    clarity = o.clone()
    sharp = o.clone()
    hi = z.clone()

    strength = float(getattr(cfg, "material_strength", 0.75))
    strength = max(0.0, min(1.5, strength))

    for name, mask in masks.items():
        if name not in SURFACE_PROFILES:
            continue
        prof = SURFACE_PROFILES[name]
        wgt = (mask.clamp(0.0, 1.0) * strength).to(dtype=torch_ops.torch.float32)
        if float(wgt.max().item()) <= 0.0:
            continue
        temp_off = _blend_add(temp_off, prof.temp_offset, wgt)
        sat = _blend_mult(sat, prof.sat_mult, wgt)
        exp = _blend_mult(exp, prof.exp_mult, wgt)
        con = _blend_mult(con, prof.con_mult, wgt)
        detail = _blend_mult(detail, prof.detail_mult, wgt)
        clarity = _blend_mult(clarity, prof.clarity_mult, wgt)
        sharp = _blend_mult(sharp, prof.sharpen_mult, wgt)
        hi = _blend_add(hi, prof.highlight_compress, wgt)

    # Clamp highlight compress to 0..1
    hi = hi.clamp(0.0, 1.0)

    # Safety clamps on multipliers
    sat = sat.clamp(0.80, 1.35)
    exp = exp.clamp(0.90, 1.10)
    con = con.clamp(0.90, 1.15)
    detail = detail.clamp(0.70, 1.40)
    clarity = clarity.clamp(0.60, 1.60)
    sharp = sharp.clamp(0.60, 1.80)
    temp_off = temp_off.clamp(-0.05, 0.05)

    return MaterialMods(
        temp_offset=temp_off,
        sat_mult=sat,
        exp_mult=exp,
        con_mult=con,
        detail_mult=detail,
        clarity_mult=clarity,
        sharpen_mult=sharp,
        highlight_compress=hi,
        source="material_segmentation",
    )
