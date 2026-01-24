from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DA3ModelBackendConfig:
    model_id: str = "depth-anything/DA3METRIC-LARGE"
    device: str = "cpu"  # cpu|cuda|mps - default to cpu for safety
    dtype: str = "float32"  # float32 recommended (DA3 uses quantile in some paths)
    max_side: int = 896  # safety cap to avoid giant attention buffers
    cache_dir: Optional[Path] = None
    offline: bool = False  # for HF_HUB_OFFLINE


def _require(modname: str):
    try:
        return __import__(modname, fromlist=["*"])
    except Exception as e:
        raise RuntimeError(f"Required module missing: {modname} ({type(e).__name__}: {e})") from e


def _get_patch_hw(model: torch.nn.Module) -> Tuple[int, int]:
    # Best effort: DinoV2 patch embed commonly has proj.kernel_size
    backbone = getattr(model, "backbone", None)
    pretrained = getattr(backbone, "pretrained", None) if backbone is not None else None
    patch_embed = getattr(pretrained, "patch_embed", None) if pretrained is not None else None
    proj = getattr(patch_embed, "proj", None) if patch_embed is not None else None
    ks = getattr(proj, "kernel_size", None)
    if isinstance(ks, tuple) and len(ks) == 2:
        return int(ks[0]), int(ks[1])
    # conservative fallback (DINOv2 ViT uses 14)
    return 14, 14


def _pad_to_multiple(x: torch.Tensor, patch_h: int, patch_w: int) -> Tuple[torch.Tensor, int, int]:
    # x: (B,S,C,H,W)
    H, W = x.shape[-2], x.shape[-1]
    pad_h = (patch_h - (H % patch_h)) % patch_h
    pad_w = (patch_w - (W % patch_w)) % patch_w
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x, pad_h, pad_w


def _resize_long_side_rgb01(rgb01: np.ndarray, max_side: int) -> np.ndarray:
    # rgb01: HxWx3 float32 [0,1]
    if max_side <= 0:
        return rgb01
    h, w = rgb01.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return rgb01
    scale = max_side / float(long_side)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    # torch interpolate for simplicity
    t = torch.from_numpy(rgb01).permute(2, 0, 1)[None]  # 1x3xHxW
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    out = t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return out


class DA3ModelBackend:
    """
    Official DA3 model backend using:
      - HF snapshot_download (config.json + model.safetensors)
      - depth_anything_3.model.da3.create_object
    Avoids depth_anything_3.api to keep optional deps (pycolmap/moviepy/open3d/gsplat) out.
    """

    def __init__(self, cfg: Optional[DA3ModelBackendConfig] = None):
        self.cfg = cfg or DA3ModelBackendConfig()
        self._model: Optional[torch.nn.Module] = None

    def is_available(self) -> bool:
        try:
            _require("depth_anything_3.model.da3")
            _require("omegaconf")
            _require("huggingface_hub")
            _require("safetensors.torch")
            return True
        except Exception:
            return False

    def _build_and_load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        da3_mod = _require("depth_anything_3.model.da3")
        OmegaConf = _require("omegaconf").OmegaConf
        snapshot_download = _require("huggingface_hub").snapshot_download
        load_file = _require("safetensors.torch").load_file

        # Respect cache dir and offline mode
        cache_kwargs = {}
        if self.cfg.cache_dir:
            cache_kwargs["cache_dir"] = str(self.cfg.cache_dir)
        if self.cfg.offline:
            cache_kwargs["local_files_only"] = True

        snap_dir = Path(snapshot_download(self.cfg.model_id, **cache_kwargs))
        raw = json.loads((snap_dir / "config.json").read_text())
        cfg = OmegaConf.create(raw["config"])
        model = da3_mod.create_object(cfg)

        state = load_file(str(snap_dir / "model.safetensors"))
        # strip common prefixes
        state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        device = torch.device(self.cfg.device)
        dtype = torch.float32 if self.cfg.dtype == "float32" else torch.float16
        model.eval().to(device=device, dtype=dtype)
        self._model = model
        return model

    def _prepare_input(self, rgb01: np.ndarray, model: nn.Module) -> torch.Tensor:
        """
        Prepare input tensor in the exact format DA3 expects.

        DA3's DINOv2 backbone expects: (B, N, 3, H, W)
        - B: batch size (1 for single image)
        - N: number of views (1 for single image)
        - 3: RGB channels
        - H, W: height, width (padded to patch size multiples)

        Args:
            rgb01: HxWx3 float32 [0,1] numpy array
            model: DA3 model instance

        Returns:
            Tensor in shape (1, 1, 3, H, W), padded and on correct device/dtype
        """
        # (B=1, N=1, C=3, H, W) is what DA3 expects
        x = torch.from_numpy(rgb01).permute(2, 0, 1)[None, None]  # HxWx3 -> 1x1x3xHxW
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        x = x.to(device=device, dtype=dtype)
        return x

    @torch.inference_mode()
    def predict_depth01_from_rgb01(self, rgb01: np.ndarray) -> np.ndarray:
        """
        rgb01: HxWx3 float32 [0,1]
        returns: depth float32 HxW (not quantized)
        """
        model = self._build_and_load()

        # downscale for safety
        rgb01_small = _resize_long_side_rgb01(rgb01, self.cfg.max_side)

        # Prepare input in DA3's expected format
        x = self._prepare_input(rgb01_small, model)

        ph, pw = _get_patch_hw(model)
        x, pad_h, pad_w = _pad_to_multiple(x, ph, pw)

        out = model(x)
        depth = out["depth"].squeeze(0).squeeze(0).detach()  # HxW (maybe padded)

        # remove pad
        if pad_h or pad_w:
            depth = depth[: depth.shape[0] - pad_h, : depth.shape[1] - pad_w]

        depth_np = depth.float().cpu().numpy().astype(np.float32)

        # Resize back to original rgb01 size for downstream quantization/alignment
        if depth_np.shape[:2] != rgb01.shape[:2]:
            t = torch.from_numpy(depth_np)[None, None]
            t = F.interpolate(t, size=rgb01.shape[:2], mode="bilinear", align_corners=False)
            depth_np = t[0, 0].cpu().numpy().astype(np.float32)

        # Normalize defensively to [0,1] if it’s not already stable
        dmin, dmax = float(np.nanmin(depth_np)), float(np.nanmax(depth_np))
        if np.isfinite(dmin) and np.isfinite(dmax) and dmax > dmin:
            depth_np = (depth_np - dmin) / (dmax - dmin)
        depth_np = np.clip(depth_np, 0.0, 1.0)
        return depth_np
