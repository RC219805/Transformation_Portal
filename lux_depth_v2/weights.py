from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import torch_ops


@dataclass
class Weights:
    wfg: "torch_ops.torch.Tensor"
    wmid: "torch_ops.torch.Tensor"
    wbg: "torch_ops.torch.Tensor"
    source: str


def _to_1x1(depth_or_mask: np.ndarray, device) -> "torch_ops.torch.Tensor":
    t = torch_ops.torch.from_numpy(depth_or_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).contiguous()
    return t.to(device=device, dtype=torch_ops.torch.float32)


def weights_from_assets(
    h: int,
    w: int,
    device,
    depth01: Optional[np.ndarray],
    masks: Dict[str, np.ndarray],
    cfg,
) -> Weights:
    """Build (foreground/midground/background) weights.

    Priority:
      1) explicit zone masks: foreground/midground/background
      2) depth percentiles
      3) uniform fallback
    """
    torch_ops.require_torch()

    have_zone = all(k in masks for k in ("foreground", "midground", "background"))
    if have_zone:
        wfg = masks["foreground"]
        wmid = masks["midground"]
        wbg = masks["background"]

        # basic resize with numpy if needed; torch resize is in torch_ops but expects tensor
        if wfg.shape != (h, w):
            import cv2
            wfg = cv2.resize(wfg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        if wmid.shape != (h, w):
            import cv2
            wmid = cv2.resize(wmid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        if wbg.shape != (h, w):
            import cv2
            wbg = cv2.resize(wbg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        wfg_t = _to_1x1(wfg, device)
        wmid_t = _to_1x1(wmid, device)
        wbg_t = _to_1x1(wbg, device)

        s = float(max(0.0, cfg.mask_soften_sigma))
        if s > 0:
            wfg_t = torch_ops.gaussian_blur(wfg_t, s, autocast=False)
            wmid_t = torch_ops.gaussian_blur(wmid_t, s, autocast=False)
            wbg_t = torch_ops.gaussian_blur(wbg_t, s, autocast=False)

        sm = (wfg_t + wmid_t + wbg_t).clamp(min=torch_ops.EPS)
        wfg_t, wmid_t, wbg_t = wfg_t / sm, wmid_t / sm, wbg_t / sm
        return Weights(wfg=wfg_t, wmid=wmid_t, wbg=wbg_t, source="zone_masks")

    if depth01 is None:
        if bool(getattr(cfg, "strict_depth", False)):
            raise FileNotFoundError("Depth missing and strict_depth=True")
        u = torch_ops.torch.ones((1, 1, h, w), device=device, dtype=torch_ops.torch.float32)
        return Weights(wfg=u * 0.34, wmid=u * 0.33, wbg=u * 0.33, source="uniform_no_depth")

    d = depth01.astype(np.float32)
    t = _to_1x1(d, device)

    # percentiles via sampling
    flat = t.reshape(-1)
    n = flat.numel()
    max_s = 250_000
    if n > max_s:
        idx = torch_ops.torch.randint(0, n, (max_s,), device=device)
        samp = flat[idx]
    else:
        samp = flat
    fg_t = float(torch_ops.torch.quantile(samp, float(cfg.fg_q)).item())
    bg_t = float(torch_ops.torch.quantile(samp, float(cfg.bg_q)).item())
    tr = float(max(getattr(cfg, "transition", 0.08), 1e-3))

    wfg_t = 1.0 - torch_ops.smoothstep(fg_t - tr, fg_t + tr, t)
    wbg_t = torch_ops.smoothstep(bg_t - tr, bg_t + tr, t)
    wmid_t = (1.0 - wfg_t - wbg_t).clamp(0.0, 1.0)

    s = float(max(0.0, cfg.mask_soften_sigma))
    if s > 0:
        wfg_t = torch_ops.gaussian_blur(wfg_t, s, autocast=False)
        wmid_t = torch_ops.gaussian_blur(wmid_t, s, autocast=False)
        wbg_t = torch_ops.gaussian_blur(wbg_t, s, autocast=False)
        sm = (wfg_t + wmid_t + wbg_t).clamp(min=torch_ops.EPS)
        wfg_t, wmid_t, wbg_t = wfg_t / sm, wmid_t / sm, wbg_t / sm

    return Weights(wfg=wfg_t, wmid=wmid_t, wbg=wbg_t, source="depth_percentiles")
