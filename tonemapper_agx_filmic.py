
"""
tonemapper_agx_filmic.py
------------------------
Drop-in tone mapping helpers for architectural render pipelines.

Features
- AgX via OpenColorIO v2 (exact, if config.ocio is provided)
- Filmic (Hable "Uncharted 2") fallback with white-point normalization
- Safe sRGB encoding, gamut compression (optional), and utility helpers

Usage (minimal):
    from tonemapper_agx_filmic import apply_agx_ocio, apply_filmic_hable, list_ocio_views
    # img_lin: float32 numpy array (H, W, 3) in *scene-linear* working space
    out = apply_agx_ocio(img_lin, config_path="/path/to/agx/config.ocio",
                         in_colorspace="Utility - Linear - sRGB",  # or "ACES - ACEScg"
                         display=None, view=None)  # uses defaults if None

    # Fallback (no OCIO):
    out = apply_filmic_hable(img_lin, exposure=1.0, white_point=11.2)

Notes
- AgX paths & names depend on the config you load. Use list_ocio_views() to introspect.
- Input must be scene-linear (no gamma). If your buffer is sRGB-encoded, decode it first.

Author: ChatGPT (GPT-5 Pro), 2025
"""

from __future__ import annotations
import os
import numpy as np

# Optional dependency: PyOpenColorIO (pip install opencolorio)
try:
    import PyOpenColorIO as ocio  # type: ignore
    _HAVE_OCIO = True
except Exception:
    ocio = None  # type: ignore
    _HAVE_OCIO = False


# ---------- Utility: sRGB OETF/EOTF ----------

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Decode sRGB to linear (expects 0..1)."""
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    out = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    return out.astype(np.float32)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """Encode linear to sRGB (clamped to 0..1)."""
    lin = np.clip(lin, 0.0, 1.0).astype(np.float32)
    out = np.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * (lin ** (1.0 / 2.4)) - 0.055,
    )
    return out.astype(np.float32)


# ---------- Utility: luminance & gamut compression ----------

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

def luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute relative luminance from linear RGB."""
    return np.tensordot(rgb, _LUMA, axes=([rgb.ndim - 1], [0])).astype(np.float32)


def highlight_desat(rgb: np.ndarray, start: float = 0.8, strength: float = 0.85) -> np.ndarray:
    """
    Simple highlight desaturation: reduces saturation as luminance approaches 1.
    This mimics film-like chroma roll-off; useful for per-channel TMOs like Hable.
    """
    y = luminance(np.clip(rgb, 0.0, None))
    # Weight grows from 0 at 'start' to 1 at 1.0
    w = np.clip((y - start) / max(1e-6, (1.0 - start)), 0.0, 1.0).astype(np.float32)
    w = w[..., None]  # broadcast to channels
    grey = y[..., None]
    return (1.0 - strength * w) * rgb + (strength * w) * grey


# ---------- Filmic (Hable) Tone Mapper ----------

def _hable_curve(x: np.ndarray,
                 A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30) -> np.ndarray:
    """
    John Hable's Uncharted 2 filmic curve for HDR tone mapping.
    Operates on linear scene values.
    """
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - (E / F)


def apply_filmic_hable(img_lin: np.ndarray,
                       exposure: float = 1.0,
                       white_point: float = 11.2,
                       desat_highlights: bool = True,
                       desat_start: float = 0.85,
                       desat_strength: float = 0.85,
                       encode_srgb: bool = True) -> np.ndarray:
    """
    Apply Hable filmic tonemapper to a scene-linear RGB image (float32).
    Parameters
    - exposure: linear exposure multiplier before tone mapping.
    - white_point: scene value mapped to display white. (11.2 is Hable's ref.)
    - desat_highlights: optional chroma roll-off to avoid neon highlights.
    - encode_srgb: if True, returns sRGB-encoded 0..1 image; else display-linear.
    """
    if img_lin.dtype != np.float32:
        img = img_lin.astype(np.float32)
    else:
        img = img_lin

    # per-channel mapping (standard practice for Hable)
    x = np.maximum(img * exposure, 0.0).astype(np.float32)
    numer = _hable_curve(x)
    denom = _hable_curve(np.array([white_point], dtype=np.float32))[0]
    mapped = np.clip(numer / max(1e-6, denom), 0.0, 1.0)

    if desat_highlights:
        mapped = highlight_desat(mapped, start=desat_start, strength=desat_strength)

    return linear_to_srgb(mapped) if encode_srgb else mapped


# ---------- AgX via OpenColorIO (exact transform if config provided) ----------

def _resolve_config(config_path: str | None):
    """
    Resolve an OCIO config: prefer explicit path, then OCIO env var, else None.
    """
    if not _HAVE_OCIO:
        raise RuntimeError("PyOpenColorIO not available. Install: pip install opencolorio")

    path = config_path or os.environ.get("OCIO", "")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "OpenColorIO config not found. Provide config_path to an AgX config. "
            "Tip: https://github.com/sobotka/AgX (download config.ocio)"
        )
    return ocio.Config.CreateFromFile(path)


def list_ocio_views(config_path: str | None = None) -> dict:
    """
    Inspect displays and views in the provided OCIO config.
    Returns: { display: [view1, view2, ...], ... }
    """
    cfg = _resolve_config(config_path)
    out = {}
    displays = cfg.getDisplaysAll() if hasattr(cfg, 'getDisplaysAll') else cfg.getDisplays()
    for display in displays:
        views = [v for v in cfg.getViews(display)]
        out[display] = views
    return out


def apply_agx_ocio(img_lin: np.ndarray,
                   config_path: str | None = None,
                   in_colorspace: str | None = None,
                   display: str | None = None,
                   view: str | None = None,
                   encode_srgb: bool = True) -> np.ndarray:
    """
    Apply AgX (or any display/view) via OpenColorIO v2.
    - img_lin: scene-linear RGB in the named 'in_colorspace' of the config.
    - in_colorspace: e.g. 'Utility - Linear - sRGB' or 'ACES - ACEScg'.
      If None, attempts to use config.getSceneLinearColorSpaceName().
    - display: OCIO display name. If None, uses config.getDefaultDisplay().
    - view: OCIO view name. If None, uses config.getDefaultView(display).
      For AgX configs, typical views include 'AgX - Base', 'AgX - Medium Contrast'.
    - encode_srgb: If True and the display is sRGB-like, return sRGB-encoded values.
      Otherwise, result is whatever the display transform returns (often display-referred).

    Returns a float32 numpy array (H, W, 3) in 0..1 (clipped).
    """
    if img_lin.dtype != np.float32:
        img = img_lin.astype(np.float32)
    else:
        img = img_lin

    cfg = _resolve_config(config_path)
    src_cs = in_colorspace or (cfg.getSceneLinearColorSpaceName() if hasattr(cfg, 'getSceneLinearColorSpaceName') else cfg.getRole(ocio.ROLE_SCENE_LINEAR))

    # Fallback to defaults if not specified
    display = display or cfg.getDefaultDisplay()
    view = view or cfg.getDefaultView(display)

    # Construct a DisplayViewTransform from scene-linear to display/view
    dvt = ocio.DisplayViewTransform(src=src_cs, display=display, view=view)
    proc = cfg.getProcessor(dvt)
    cpu = proc.getDefaultCPUProcessor()

    # Apply in-place using a PackedImageDesc
    h, w, c = img.shape
    if c != 3:
        raise ValueError("Expected RGB image with 3 channels.")
    # Copy to avoid mutating caller's buffer
    out = np.ascontiguousarray(img.copy())
    desc = ocio.PackedImageDesc(out, w, h, c)
    cpu.apply(desc)

    # OCIO output is typically display-referred; clamp to [0,1]
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    # If the display is an sRGB-like display, OCIO already includes the OETF.
    # We keep encode_srgb for API symmetry with filmic; no extra step needed here.
    return out


# ---------- Convenience: soft check for common AgX view names ----------

def guess_agx_view(config_path: str | None = None) -> tuple[str, str]:
    """
    Return (display, view) picking an AgX-like view if present.
    Otherwise returns (default_display, default_view).
    """
    cfg = _resolve_config(config_path)
    display = cfg.getDefaultDisplay() if hasattr(cfg, 'getDefaultDisplay') else cfg.getDefaultDisplayDeviceName()
    candidates = []
    for disp in cfg.getDisplaysAll():
        for v in cfg.getViews(disp):
            name = v.lower()
            if "agx" in name:
                candidates.append((disp, v))
    if candidates:
        # Prefer the default display if any candidate belongs to it
        for d, v in candidates:
            if d == display:
                return d, v
        return candidates[0]
    # Fallback to defaults
    return display, cfg.getDefaultView(display)


# ---------- Minimal self-test (disabled by default) ----------

if __name__ == "__main__":
    print("tonemapper_agx_filmic.py: run inside your pipeline. See docstrings for usage.")
