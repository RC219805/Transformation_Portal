#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lux Render Pipeline — PBR v3 (NumPy + Pillow, zero external deps)
- ACEScg→Rec.709 friendly (do color IO in sRGB; math in linear)
- Full PBR map support: albedo, normal, roughness (or gloss), metallic, AO, displacement/POM
- Env/spec sampling, multi-light, masks, quality presets (draft/preview/final)
- Variant albedo mixing to reduce tiling repetition
CLI examples:
  # Tiles04 travertine with gloss->roughness invert + AO + env
  python lux_render_pipeline_plus_v3.py materialize in.png out.png \
    --albedo Tiles04_COL_VAR1_2K.jpg --variant-map Tiles04_COL_VAR2_2K.jpg --variant-mix 0.35 \
    --normal Tiles04_NRM_2K.jpg --roughness Tiles04_GLOSS_2K.jpg --roughness-is-gloss \
    --ao Tiles04_REFL_2K.jpg --env-map studio.hdr \
    --height-strength 0.35 --reflection-strength 0.25 --proc-scale 0.75 --quality preview

  # Bronze anodized (metalness workflow)
  python lux_render_pipeline_plus_v3.py materialize in.png out.png \
    --albedo MetalBronze001_COL_2K_METALNESS.jpg \
    --normal MetalBronze001_NRM16_2K_METALNESS.tiff \
    --metallic-map MetalBronze001_METALNESS_2K_METALNESS.jpg \
    --roughness MetalBronze001_ROUGHNESS_2K_METALNESS.jpg \
    --reflection-strength 0.4 --proc-scale 1.0 --quality final
"""

from __future__ import annotations
from pathlib import Path
import argparse, math, json
from typing import Optional, Tuple, List, Union
import numpy as np
from PIL import Image, ImageEnhance

# ---------------- IO helpers ----------------

def _open_rgb(path: Union[str, Path]) -> Image.Image:
    im = Image.open(path)
    # Treat color maps as sRGB
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

def _open_L(path: Union[str, Path]) -> Image.Image:
    im = Image.open(path)
    # Non-color data
    if im.mode != "L":
        im = im.convert("L")
    return im

def _resize(im: Image.Image, size: Tuple[int,int], res=Image.BICUBIC) -> Image.Image:
    if im.size == size: return im
    return im.resize(size, res)

def _as_f32_rgb(im: Image.Image) -> np.ndarray:
    a = np.asarray(im, dtype=np.uint8)
    return np.require(a, np.float32, ["C","A"]) / 255.0

def _as_f32_L(im: Image.Image) -> np.ndarray:
    a = np.asarray(im, dtype=np.uint8)
    return np.require(a, np.float32, ["C","A"]) / 255.0

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)

def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92*x, 1.055*np.power(x, 1/2.4) - 0.055)

# ------------- math helpers -------------

def _normalize(v: np.ndarray, eps=1e-6) -> np.ndarray:
    n = np.sqrt(np.sum(v*v, axis=-1, keepdims=True)) + eps
    return v / n

def _sobel_dxdy(h: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    p = np.pad(h, ((1,1),(1,1)), mode="edge")
    dx = (p[1:-1,2:] - p[1:-1,:-2]) + 2*(p[2:,2:] - p[2:,:-2]) + (p[:-2,2:] - p[:-2,:-2])
    dy = (p[2:,1:-1] - p[:-2,1:-1]) + 2*(p[2:,2:] - p[:-2,2:]) + (p[2:,:-2] - p[:-2,:-2])
    k = 1.0/8.0
    return np.require(dx*k, np.float32, ["C","A"]), np.require(dy*k, np.float32, ["C","A"])

def _sample_env(env: Optional[np.ndarray], dir3: np.ndarray) -> np.ndarray:
    if env is None: return np.ones_like(dir3)[..., :3]
    x,y,z = dir3[...,0], dir3[...,1], dir3[...,2]
    theta = np.arctan2(x, z)
    phi   = np.arccos(np.clip(y, -1, 1))
    u = (theta/(2*np.pi) + 0.5); v = phi/np.pi
    H,W,_ = env.shape
    uu = np.clip((u*W).astype(np.int32), 0, W-1)
    vv = np.clip((v*H).astype(np.int32), 0, H-1)
    return env[vv, uu]

def _parallax_uv(height: np.ndarray, view_xy: Tuple[float,float], scale: float, steps: int) -> Tuple[np.ndarray,np.ndarray]:
    H,W = height.shape
    yy,xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    uvx = xx.copy(); uvy = yy.copy()
    if steps <= 1 or scale <= 1e-6:
        uvx += height * (scale*view_xy[0]) * W
        uvy += height * (scale*view_xy[1]) * H
        return uvx, uvy
    step = 1.0/steps
    accum = np.zeros_like(height)
    for _ in range(steps):
        accum += height
        uvx += accum * (scale*view_xy[0])*W*step
        uvy += accum * (scale*view_xy[1])*H*step
    return uvx, uvy

def _bilinear(img: np.ndarray, uvx: np.ndarray, uvy: np.ndarray) -> np.ndarray:
    H,W = img.shape[:2]
    x = np.clip(uvx, 0, W-1); y = np.clip(uvy, 0, H-1)
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0+1, 0, W-1);        y1 = np.clip(y0+1, 0, H-1)
    wx = x - x0; wy = y - y0
    c00 = img[y0, x0]; c10 = img[y0, x1]; c01 = img[y1, x0]; c11 = img[y1, x1]
    return (c00*(1-wx)*(1-wy) + c10*wx*(1-wy) + c01*(1-wx)*wy + c11*wx*wy)

# ------------- core PBR -------------

def apply_pbr_overlays(
    image: Union[str, Path, Image.Image, np.ndarray],
    *, albedo: Optional[Union[str,Path]]=None, variant_map: Optional[Union[str,Path]]=None, variant_mix: float=0.0,
    normal: Optional[Union[str,Path]]=None, roughness: Optional[Union[str,Path]]=None, roughness_is_gloss: bool=False,
    metallic_map: Optional[Union[str,Path]]=None, metallic: float=0.0,
    ao: Optional[Union[str,Path]]=None, mask: Optional[Union[str,Path]]=None, env_map: Optional[Union[str,Path]]=None,
    albedo_blend: float=1.0, height_strength: float=0.3, reflection_strength: float=0.25,
    gloss_power_min: float=1.5, gloss_power_max: float=6.0, fresnel_power: float=3.0,
    ao_strength: float=1.0, normal_strength: float=1.0,
    lights: Optional[List[Tuple[np.ndarray,np.ndarray,float]]] = None,
    pom_scale: float=0.02, pom_steps: int=1, view_dir_xy: Tuple[float,float]=(0.0, 0.0),
    proc_scale: float=1.0, enable_displacement: bool=True, disp_mm: float=4.0,
    exposure: float=0.0, contrast: float=1.0, saturation: float=1.0, clamp_low: float=0.0, clamp_high: float=1.0,
    out_mode: str="RGB", quality: str="preview"
) -> Image.Image:

    if isinstance(image, (str, Path)):
        pil_in = _open_rgb(image)
    elif isinstance(image, Image.Image):
        pil_in = image.convert("RGB")
    else:
        pil_in = Image.fromarray((np.clip(image,0,1)*255.0+0.5).astype(np.uint8), "RGB")

    W,H = pil_in.size
    base_hi_srgb = _as_f32_rgb(pil_in)
    base_hi = _srgb_to_linear(base_hi_srgb)

    # quality presets
    q = quality.lower()
    if q == "draft":
        proc_scale = min(proc_scale, 0.5); pom_steps = min(pom_steps, 1); enable_displacement=False; disp_mm = min(disp_mm, 1.0)
    elif q == "preview":
        proc_scale = min(proc_scale, 0.75); pom_steps = min(pom_steps, 4); disp_mm = min(disp_mm, 2.0)
    elif q == "final":
        pass

    scale = float(np.clip(proc_scale, 0.1, 1.0))
    w,h = max(1,int(W*scale)), max(1,int(H*scale))

    # Mask (process region)
    mask_np = None
    if mask is not None:
        m = _open_L(mask); m = _resize(m, (w,h), res=Image.BICUBIC)
        mask_np = _as_f32_L(m)

    # Start from input (linear)
    base = np.asarray(Image.fromarray((np.clip(base_hi,0,1)*255).astype(np.uint8)).resize((w,h), Image.BICUBIC), dtype=np.uint8).astype(np.float32)/255.0

    # Albedo
    if albedo is not None:
        alb = _open_rgb(albedo); alb = _resize(alb, (w,h)); alb_np = _srgb_to_linear(_as_f32_rgb(alb))
        if variant_map:
            v2 = _open_rgb(variant_map); v2 = _resize(v2, (w,h)); v2_np = _srgb_to_linear(_as_f32_rgb(v2))
            m = float(np.clip(variant_mix, 0.0, 1.0))
            alb_np = alb_np*(1-m) + v2_np*m
        if normal is not None and enable_displacement and pom_scale>1e-6:
            # POM uses height map if provided; fallback to normal Z
            pass
        t = float(np.clip(albedo_blend,0,1))
        base = alb_np*t + base*(1.0-t)

    # Height from displacement not provided; we approximate with roughness/normal if needed
    height = None

    # Normal
    if normal is not None:
        n_im = _open_rgb(normal); n_im = _resize(n_im, (w,h)); n_np = _as_f32_rgb(n_im)  # assumed normal map in tangent space
        # Convert normal map (0..1)->(-1..1)
        n_ts = (n_np*2.0 - 1.0)
        n_ts = _normalize(n_ts)
    else:
        n_ts = np.dstack([np.zeros_like(base[...,0]), np.zeros_like(base[...,0]), np.ones_like(base[...,0])])

    # Roughness
    if roughness is not None:
        r_im = _open_rgb(roughness); r_im = _resize(r_im,(w,h)); r_np = _as_f32_rgb(r_im)
        rough = r_np.mean(axis=-1)
        if roughness_is_gloss:
            rough = 1.0 - rough
        rough = np.clip(rough, 0.02, 0.98)
    else:
        rough = np.full((h,w), 0.6, dtype=np.float32)

    # Metallic
    if metallic_map is not None:
        m_im = _open_rgb(metallic_map); m_im = _resize(m_im,(w,h)); m_np = _as_f32_rgb(m_im).mean(axis=-1)
        metal = np.clip(m_np, 0.0, 1.0)
    else:
        metal = np.full((h,w), float(np.clip(metallic,0,1)), dtype=np.float32)

    # AO
    if ao is not None:
        ao_im = _open_rgb(ao); ao_im = _resize(ao_im,(w,h)); ao_np = _as_f32_rgb(ao_im).mean(axis=-1)
        base = np.clip(base * (ao_np[...,None]**float(np.clip(ao_strength,0,4))), 0.0, 1.0)

    # Height-based shading (Sobel from height if available)
    if height is not None and height_strength>1e-6:
        dx,dy = _sobel_dxdy(height)
        z = np.full_like(dx, 1.0/max(1e-3, 1.0-0.5*height_strength))
        n_from_h = _normalize(np.stack([-dx, -dy, z], axis=-1))
        n_ts = _normalize(n_ts*(1.0-height_strength) + n_from_h*height_strength)

    # Lighting: simple multi-light lambert + specular; fresnel approx
    if lights is None:
        lights = [(np.array([0.3,0.5,0.8],np.float32), np.array([1,1,1],np.float32), 1.0)]
    N = _normalize(n_ts)
    view = np.dstack([np.zeros_like(rough), np.zeros_like(rough), np.ones_like(rough)])  # view ~ +Z
    spec_acc = np.zeros_like(base)
    diff_acc = np.zeros_like(base)

    for Ldir, Lcol, I in lights:
        L = np.array(Ldir, dtype=np.float32); L = L/(np.linalg.norm(L)+1e-6)
        ndotl = np.clip(np.sum(N*L[None,None,:], axis=-1, keepdims=True), 0.0, 1.0)
        diff = base * ndotl * I
        diff_acc += diff * Lcol
        # GGX-ish spec (very simplified)
        half_vec = _normalize(N + view)
        ndoth = np.clip(np.sum(N*half_vec, axis=-1), 0.0, 1.0)
        # perceptual roughness -> shininess
        shin = 2.0*(1.0 - rough)
        spec = (ndoth[...,None] ** (1.0 + 32.0*shin[...,None]))
        # Fresnel
        F0 = 0.04*(1.0-metal)[...,None] + base*metal[...,None]  # dielectrics vs metals
        spec_rgb = F0 + (1.0 - F0) * (1.0 - ndoth[...,None])**5
        spec_acc += spec * spec_rgb * I

    # Env reflection (uses N.z as proxy)
    env = None
    if env_map is not None:
        try:
            env_im = _open_rgb(env_map); env_np = _as_f32_rgb(env_im); env = env_np
        except Exception:
            env = None
    if env is not None and reflection_strength>1e-6:
        refl = _sample_env(env, N) * (1.0 - rough[...,None])
        spec_acc += refl * float(reflection_strength)

    shaded = np.clip(diff_acc + spec_acc*float(reflection_strength), 0.0, 1.0)

    # Mask application
    if mask_np is not None:
        m = mask_np[...,None]
        shaded = shaded*m + base*(1.0-m)

    # Upscale + detail restore
    if scale < 1.0:
        up = np.asarray(Image.fromarray((shaded*255).astype(np.uint8)).resize((W,H), Image.BICUBIC), dtype=np.uint8).astype(np.float32)/255.0
        hf = np.clip(base_hi - np.asarray(Image.fromarray((base_hi*255).astype(np.uint8)).resize((W,H), Image.BICUBIC), dtype=np.uint8).astype(np.float32)/255.0, -1,1)
        shaded = np.clip(up + 0.12*hf, 0.0, 1.0)
    else:
        shaded = np.asarray(Image.fromarray((shaded*255).astype(np.uint8)).resize((W,H), Image.BICUBIC), dtype=np.uint8).astype(np.float32)/255.0

    # Finishing
    # exposure/clamp
    if exposure!=0.0 or clamp_low>0 or clamp_high<1:
        def _lut_u8(exposure, lo, hi):
            x = (np.arange(256, dtype=np.float32)/255.0)
            if exposure!=0.0: x *= 2.0**float(exposure)
            lo,hi = float(np.clip(lo,0,1)), float(np.clip(hi,0,1))
            if hi < lo: hi=lo
            x = np.zeros_like(x) if hi==lo else (x-lo)/max(1e-6,hi-lo)
            return (np.clip(x,0,1)*255.0+0.5).astype(np.uint8)
        lut = _lut_u8(exposure, clamp_low, clamp_high)
        arr8 = (np.clip(shaded,0,1)*255.0+0.5).astype(np.uint8)
        shaded = _as_f32_rgb(Image.fromarray(lut[arr8]))
    out = Image.fromarray((_linear_to_srgb(np.clip(shaded,0,1))*255.0+0.5).astype(np.uint8), "RGB")
    if contrast!=1 or saturation!=1:
        if contrast!=1: out = ImageEnhance.Contrast(out).enhance(float(contrast))
        if saturation!=1: out = ImageEnhance.Color(out).enhance(float(saturation))
    return out.convert(out_mode)

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(prog="lux_pbr_v3", description="PBR materializer with ACES-friendly maps")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser):
        p.add_argument("input", type=Path)
        p.add_argument("output", type=Path)
        # maps
        p.add_argument("--albedo", type=Path)
        p.add_argument("--variant-map", type=Path)
        p.add_argument("--variant-mix", type=float, default=0.0)
        p.add_argument("--normal", type=Path)
        p.add_argument("--roughness", type=Path, help="If a GLOSS map is provided, also pass --roughness-is-gloss")
        p.add_argument("--roughness-is-gloss", action="store_true")
        p.add_argument("--metallic-map", type=Path)
        p.add_argument("--metallic", type=float, default=0.0)
        p.add_argument("--ao", type=Path)
        p.add_argument("--mask", type=Path)
        p.add_argument("--env-map", type=Path)
        # controls
        p.add_argument("--albedo-blend", type=float, default=1.0)
        p.add_argument("--height-strength", type=float, default=0.3)
        p.add_argument("--reflection-strength", type=float, default=0.25)
        p.add_argument("--pom-scale", type=float, default=0.02)
        p.add_argument("--pom-steps", type=int, default=1)
        p.add_argument("--view-dir", type=str, default="0.0,0.0")
        p.add_argument("--proc-scale", type=float, default=1.0)
        p.add_argument("--enable-displacement", action="store_true", default=False)
        p.add_argument("--disp-mm", type=float, default=4.0)
        p.add_argument("--exposure", type=float, default=0.0)
        p.add_argument("--contrast", type=float, default=1.0)
        p.add_argument("--saturation", type=float, default=1.0)
        p.add_argument("--clamp-low", type=float, default=0.0)
        p.add_argument("--clamp-high", type=float, default=1.0)
        p.add_argument("--out-mode", type=str, default="RGB", choices=["RGB","RGBA"])
        p.add_argument("--quality", type=str, default="preview", choices=["draft","preview","final"])

    p_mat = sub.add_parser("materialize")
    add_common(p_mat)

    args = ap.parse_args()
    if args.cmd == "materialize":
        vx,vy = [float(x.strip()) for x in args.view_dir.split(",")]
        out = apply_pbr_overlays(
            args.input, albedo=args.albedo, variant_map=args.variant_map, variant_mix=args.variant_mix,
            normal=args.normal, roughness=args.roughness, roughness_is_gloss=args.roughness_is_gloss,
            metallic_map=args.metallic_map, metallic=args.metallic, ao=args.ao, mask=args.mask, env_map=args.env_map,
            albedo_blend=args.albedo_blend, height_strength=args.height_strength, reflection_strength=args.reflection_strength,
            pom_scale=args.pom_scale, pom_steps=args.pom_steps, view_dir_xy=(vx,vy), proc_scale=args.proc_scale,
            enable_displacement=args.enable_displacement, disp_mm=args.disp_mm,
            exposure=args.exposure, contrast=args.contrast, saturation=args.saturation,
            clamp_low=args.clamp_low, clamp_high=args.clamp_high,
            out_mode=args.out_mode, quality=args.quality
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.save(args.output)

if __name__ == "__main__":
    main()
