from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---- DA3 input shape helpers (module scope) ----
def _get_patch_hw(model: torch.nn.Module) -> tuple[int, int]:
    """Return patch size (H,W) used by DinoV2 patch embedding."""
    try:
        pe = model.backbone.pretrained.patch_embed  # type: ignore[attr-defined]
        ps = pe.patch_size
        if isinstance(ps, (tuple, list)):
            return int(ps[0]), int(ps[1])
        return int(ps), int(ps)
    except Exception:
        return 14, 14  # DinoV2 vitl default


def _pad_to_multiple(x: torch.Tensor, patch_h: int, patch_w: int) -> tuple[torch.Tensor, int, int]:
    """Pad bottom/right so H,W become divisible by patch_h/patch_w. x is (B,S,C,H,W)."""
    H, W = x.shape[-2], x.shape[-1]
    pad_h = (patch_h - (H % patch_h)) % patch_h
    pad_w = (patch_w - (W % patch_w)) % patch_w
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x, pad_h, pad_w


# ----------------------------------------------

# We intentionally avoid importing depth_anything_3.api (it pulls in pycolmap/open3d/etc at import time).
from depth_anything_3.model import da3 as da3_mod  # provides DepthAnything3Net, NestedDepthAnything3Net

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    raise RuntimeError("huggingface_hub is required (pip install huggingface_hub).") from e


def _load_image_as_tensor(path: Path, device: torch.device, dtype: torch.dtype, max_side: int, min_side: int):
    # Returns: tensor (B,S,C,H,W), (origH, origW), (inH, inW)
    img = Image.open(path).convert("RGB")
    origW, origH = img.size

    scale = 1.0
    if max_side and max(origW, origH) > max_side:
        scale = max_side / float(max(origW, origH))
    if min_side and min(origW, origH) < min_side:
        scale = max(scale, min_side / float(min(origW, origH)))

    if scale != 1.0:
        inW = int(round(origW * scale))
        inH = int(round(origH * scale))
        img = img.resize((inW, inH), resample=Image.BICUBIC)
    else:
        inW, inH = origW, origH

    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # 1x1xCxHxW
    return t.to(device=device, dtype=dtype), (origH, origW), (inH, inW)


def _find_checkpoint(snapshot_dir: Path) -> Path:
    # Try common checkpoint patterns
    patterns = [
        "*.safetensors",
        "*.pt",
        "*.pth",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    for pat in patterns:
        hits = list(snapshot_dir.glob(pat))
        if hits:
            # prefer larger files (likely actual weights vs small metadata)
            hits.sort(key=lambda p: p.stat().st_size, reverse=True)
            return hits[0]
    # Also search recursively, but keep it shallow
    for pat in ["**/*.safetensors", "**/*.pth", "**/*.pt", "**/pytorch_model.bin"]:
        hits = list(snapshot_dir.glob(pat))
        if hits:
            hits.sort(key=lambda p: p.stat().st_size, reverse=True)
            return hits[0]
    raise FileNotFoundError(f"No checkpoint found in {snapshot_dir}")


def _load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as e:
            raise RuntimeError("safetensors is required for .safetensors checkpoints (pip install safetensors).") from e
        state = load_file(str(ckpt_path))
        # Normalize keys (safetensors path) to match model.state_dict() naming.
        out = {}
        for k, v in state.items():
            nk = k
            for pref in ("module.", "model.", "net."):
                if nk.startswith(pref):
                    nk = nk[len(pref) :]
            out[nk] = v
        return out
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(obj)}")
    # Strip common prefixes
    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _build_model_from_hf_config(model_id: str, snapshot_dir: Path):
    # Build model using DA3's own Hydra-style config + factory.
    import json
    from omegaconf import OmegaConf
    from depth_anything_3.model import da3 as da3_mod

    cfg_path = snapshot_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in snapshot: {snapshot_dir}")

    raw = json.loads(cfg_path.read_text())
    if "config" not in raw:
        raise RuntimeError(f"Unexpected config.json format. Top-level keys: {list(raw.keys())}")

    cfg = OmegaConf.create(raw["config"])  # contains __object__ blocks
    model = da3_mod.create_object(cfg)  # returns a DepthAnything3Net instance
    return model


def _quantize_p1p99(depth: np.ndarray) -> np.ndarray:
    # depth: HxW float32
    p1, p99 = np.percentile(depth, [1, 99])
    if p99 <= p1:
        p1, p99 = float(depth.min()), float(depth.max() + 1e-6)
    d = np.clip((depth - p1) / (p99 - p1), 0.0, 1.0)
    return (d * 65535.0).round().astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-depth-dir", required=True)
    ap.add_argument("--model-id", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--device", default="mps")  # mps|cpu
    ap.add_argument("--dtype", default="float16")  # float16|float32
    ap.add_argument("--max-side", type=int, default=1024, help="Max image side for DA3 inference (prevents attention OOM).")
    ap.add_argument("--min-side", type=int, default=0, help="Optional min side; 0 disables.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_depth_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Download snapshot (cached)
    snap_path = Path(snapshot_download(repo_id=args.model_id, allow_patterns=["*"], local_dir=None))
    ckpt = _find_checkpoint(snap_path)
    state = _load_state_dict(ckpt)

    model = _build_model_from_hf_config(args.model_id, snap_path).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded checkpoint:", ckpt.name)
    if missing:
        print("WARN missing keys:", len(missing))
    if unexpected:
        print("WARN unexpected keys:", len(unexpected))

    model.eval()

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}
    images = [p for p in sorted(in_dir.iterdir()) if p.suffix.lower() in exts]
    if not images:
        raise SystemExit(f"No images found in {in_dir}")

    for p in images:
        x, (origH, origW), (inH, inW) = _load_image_as_tensor(
            p, device=device, dtype=dtype, max_side=args.max_side, min_side=args.min_side
        )
        patch_h, patch_w = _get_patch_hw(model)
        x, pad_h, pad_w = _pad_to_multiple(x, patch_h, patch_w)

        with torch.no_grad():
            y = model(x)

        # Crop back to original size if we padded
        if pad_h or pad_w:
            # y could be tensor/dict; cropping happens after we extract the depth tensor
            orig_h = x.shape[-2] - pad_h
            orig_w = x.shape[-1] - pad_w
        else:
            orig_h = x.shape[-2]
            orig_w = x.shape[-1]
        # y could be (B,1,H,W) or similar; normalize to HxW
        if isinstance(y, (list, tuple)):
            y = y[0]
        if isinstance(y, dict) and "depth" in y:
            y = y["depth"]
        if not isinstance(y, torch.Tensor):
            raise RuntimeError(f"Unexpected model output type: {type(y)}")

        depth = y.squeeze().detach().float().cpu().numpy()
        depth = depth[:orig_h, :orig_w]
        if (orig_h, orig_w) != (origH, origW):
            # depth currently at inference resolution; resize to original H/W
            dimg = Image.fromarray(depth.astype(np.float32), mode="F")
            dimg = dimg.resize((origW, origH), resample=Image.BILINEAR)
            depth = np.asarray(dimg).astype(np.float32)
        # Resize depth back to original image size
        if (orig_h, orig_w) != (origH, origW):
            pass
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected depth shape after squeeze: {depth.shape}")

        depth_u16 = _quantize_p1p99(depth)

        stem = p.stem
        out_png = out_dir / f"{stem}_depth.png"
        Image.fromarray(depth_u16, mode="I;16").save(out_png)

        stats: Dict[str, Any] = {
            "file": p.name,
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "p50": float(np.percentile(depth, 50)),
            "p01": float(np.percentile(depth, 1)),
            "p99": float(np.percentile(depth, 99)),
            "checkpoint": str(ckpt),
            "model_id": args.model_id,
            "device": str(device),
            "dtype": str(dtype),
        }
        (out_dir / f"{stem}_depth_stats.json").write_text(json.dumps(stats, indent=2))

        print("Wrote:", out_png.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
