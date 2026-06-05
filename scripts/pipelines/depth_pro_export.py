#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import PIL  # for __version__
import torch
from PIL import Image

try:
    import depth_pro
except ImportError:  # pragma: no cover - optional model runtime
    depth_pro = None

try:
    import importlib.metadata as importlib_metadata  # py3.8+
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DepthProExport")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = REPO_ROOT / "input_images" / "750_picacho" / "source_jpegs" / "750Picacho_Pool.jpg"
DEFAULT_CKPT = Path("checkpoints/depth_pro.pt")
DEPTHPRO_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"

EXIF_ORIENTATION_TAG = 274  # standard EXIF Orientation tag ID


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_version(mod: Any, fallback: str = "unknown") -> str:
    v = getattr(mod, "__version__", None)
    if isinstance(v, str) and v.strip():
        return v
    return fallback


def pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"


def get_exif_orientation(img: Image.Image) -> int | None:
    """
    Returns EXIF Orientation tag value if present, else None.
    NOTE: We do NOT apply exif transpose here (behavior unchanged).
    """
    try:
        exif = img.getexif()
        if exif:
            val = exif.get(EXIF_ORIENTATION_TAG)
            return int(val) if val is not None else None
    except Exception:
        pass
    return None


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _safe_repr(obj: Any, limit: int = 500) -> str:
    """
    Return a bounded repr() so provenance doesn't explode in size.
    """
    try:
        s = repr(obj)
    except Exception:
        s = f"<unreprable {obj.__class__.__name__}>"
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else (s[:limit] + "...<truncated>")


def extract_depthpro_meta(model: Any, transform: Any) -> Dict[str, Any]:
    """
    Best-effort introspection of Depth Pro internals WITHOUT changing behavior.
    We only read attributes if they exist; otherwise we omit or set null.
    """
    candidate_attr_names = [
        "patch_encoder_preset",
        "image_encoder_preset",
        "decoder_features",
        "checkpoint_uri",
        "checkpoint_path",
        "model_id",
        "variant",
        "preset",
        "name",
    ]

    meta: Dict[str, Any] = {
        "model": {
            "class": model.__class__.__name__,
            "id": _safe_getattr(model, "model_id") or _safe_getattr(model, "id") or _safe_getattr(model, "name"),
            "attributes": {},
        },
        "transform": {
            "class": transform.__class__.__name__ if transform is not None else None,
            "repr": _safe_repr(transform) if transform is not None else None,
        },
    }

    for k in candidate_attr_names:
        v = _safe_getattr(model, k)
        if v is not None:
            meta["model"]["attributes"][k] = v

    cfg = _safe_getattr(model, "config") or _safe_getattr(model, "cfg")
    if cfg is not None:
        cfg_attrs: Dict[str, Any] = {}
        for k in candidate_attr_names:
            v = _safe_getattr(cfg, k)
            if v is not None:
                cfg_attrs[k] = v
        if cfg_attrs:
            meta["model"]["config_attributes"] = cfg_attrs
            meta["model"]["id"] = (
                meta["model"]["id"] or cfg_attrs.get("model_id") or cfg_attrs.get("name") or cfg_attrs.get("preset")
            )

    return meta


def depth_stats(depth: np.ndarray) -> Dict[str, Any]:
    """
    Adds non-invasive stats for sanity checks (no behavior change).
    Reports over finite values only.
    """
    d = depth.astype(np.float32, copy=False)
    finite = np.isfinite(d)
    finite_pct = float(finite.mean() * 100.0)
    if not finite.any():
        return {
            "finite_pct": finite_pct,
            "min": None,
            "median": None,
            "p95": None,
        }

    vals = d[finite]
    return {
        "finite_pct": round(finite_pct, 6),
        "min": float(np.min(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95.0)),
    }


def normalize_to_uint16(depth: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        raise ValueError("Depth contains no finite values")

    vmin = float(np.percentile(d[finite], 1.0))
    vmax = float(np.percentile(d[finite], 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    x = (d - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)

    u16 = (x * 65535.0 + 0.5).astype(np.uint16)
    meta = {"norm": "p01_p99", "vmin": vmin, "vmax": vmax}
    return u16, meta


def ensure_checkpoint(ckpt: Path) -> None:
    if ckpt.exists() and ckpt.is_file():
        return
    raise FileNotFoundError(
        f"Missing checkpoint: {ckpt}\n"
        f"Download (atomic + resumable):\n"
        f"  mkdir -p {ckpt.parent}\n"
        f"  curl -L --retry 10 --retry-all-errors --retry-delay 2 --continue-at - "
        f"-o {ckpt}.part {DEPTHPRO_URL}\n"
        f"  unzip -t {ckpt}.part | tail -n 5\n"
        f"  mv -f {ckpt}.part {ckpt}\n"
    )


def run_depth_pro(img_path: Path, device: torch.device) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Runs Depth Pro inference via official API.
    Behavior unchanged: PIL->transform->model.infer.
    """
    if depth_pro is None:
        raise RuntimeError(
            "depth_pro package is not installed. Install the Depth Pro runtime before running inference."
        )

    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device).eval()

    img_raw = Image.open(img_path)
    # EXIF orientation is NOT applied (behavior unchanged); we record the tag in provenance later.
    img = img_raw.convert("RGB")

    x = transform(img) if callable(transform) else img

    if isinstance(x, torch.Tensor) and x.ndim == 3:
        x = x.unsqueeze(0)
    if isinstance(x, torch.Tensor):
        x = x.to(device, dtype=torch.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.infer(x)
    dt = time.perf_counter() - t0

    depth_t = out["depth"]
    if depth_t.ndim == 3 and depth_t.shape[0] == 1:
        depth_t = depth_t[0]

    depth = depth_t.detach().float().cpu().numpy().astype(np.float32)
    meta = extract_depthpro_meta(model, transform)
    return depth, dt, meta


def save_depth_products(
    img_path: Path,
    depth: np.ndarray,
    ckpt_path: Path,
    device: torch.device,
    inference_sec: float,
    depthpro_meta: Dict[str, Any] | None = None,
) -> Dict[str, Path]:
    stem = img_path.with_suffix("")
    png16_path = Path(str(stem) + "_depthpro_depth16.png")
    npy_path = Path(str(stem) + "_depthpro_depth.npy")
    json_path = Path(str(stem) + "_depthpro_provenance.json")

    # Save float depth (source of truth)
    np.save(npy_path, depth.astype(np.float32))

    # Save 16-bit visualization PNG (no deprecated Pillow mode arg)
    u16, norm_meta = normalize_to_uint16(depth)
    Image.fromarray(u16.astype(np.uint16)).save(png16_path)

    img_raw = Image.open(img_path)
    exif_orientation = get_exif_orientation(img_raw)
    img = img_raw.convert("RGB")
    w, h = img.size

    # Output integrity
    npy_sha256 = sha256_file(npy_path)
    png_sha256 = sha256_file(png16_path)
    npy_bytes = npy_path.stat().st_size
    png_bytes = png16_path.stat().st_size

    # Run timestamp (both epoch + ISO UTC)
    ts_epoch = int(time.time())
    ts_iso_utc = datetime.now(timezone.utc).isoformat()

    provenance: Dict[str, Any] = {
        "status": "ok",
        "engine": "apple_depth_pro",
        "device": str(device),
        "input": {
            "path": str(img_path),
            "resolution": [w, h],
            "exif": {
                "orientation_tag": exif_orientation,
                "orientation_applied": False,
                "notes": "EXIF orientation is recorded but not applied (behavior unchanged).",
            },
        },
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": sha256_file(ckpt_path),
            "bytes": ckpt_path.stat().st_size,
        },
        "outputs": {
            "depth_npy": str(npy_path),
            "depth_png16": str(png16_path),
            "depth_shape": list(depth.shape),
            "depth_dtype": str(depth.dtype),
            "png16_normalization": norm_meta,
            "notes": "depth_npy is raw float depth; depth_png16 is percentile-normalized visualization (p01-p99).",
            "integrity": {
                "depth_npy": {"sha256": npy_sha256, "bytes": npy_bytes},
                "depth_png16": {"sha256": png_sha256, "bytes": png_bytes},
            },
        },
        "depth_stats": depth_stats(depth),
        "timing": {
            "inference_sec": round(float(inference_sec), 6),
        },
        "units": "unknown",
        "run": {
            "timestamp_epoch": ts_epoch,
            "timestamp_iso_utc": ts_iso_utc,
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_mps_available": bool(torch.backends.mps.is_available()),
            "numpy": np.__version__,
            "pillow": safe_version(PIL),
            # installed package version (preferred over __version__)
            "depth_pro_pkg": pkg_version("depth_pro"),
            # keep __version__ too, if it exists (often "unknown" for git installs)
            "depth_pro": safe_version(depth_pro),
        },
    }

    if depthpro_meta:
        provenance["depth_pro_meta"] = depthpro_meta

    json_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"png16": png16_path, "npy": npy_path, "json": json_path}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Apple Depth Pro and export depth + provenance.")
    p.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE), help="Input image path.")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT), help="Checkpoint path (default: checkpoints/depth_pro.pt).")
    p.add_argument("--cpu", action="store_true", help="Force CPU (ignore MPS).")
    p.add_argument("--no-save", action="store_true", help="Run inference but do not save outputs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    img_path = Path(args.image).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    device = pick_device(force_cpu=bool(args.cpu))

    logger.info("--- Depth Pro Export ---")
    logger.info(f"ℹ️  Device: {device}")
    logger.info(f"ℹ️  Image: {img_path}")
    logger.info(f"ℹ️  Checkpoint: {ckpt_path}")

    try:
        ensure_checkpoint(ckpt_path)
    except Exception as e:
        logger.error(f"❌ {e}")
        return 1

    if not img_path.exists():
        logger.error(f"❌ Missing image: {img_path}")
        return 1

    try:
        depth, dt, dp_meta = run_depth_pro(img_path, device=device)
        logger.info(f"✅ Inference OK | depth_shape={tuple(depth.shape)} | dtype={depth.dtype} | sec={dt:.4f}")
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        return 1

    if args.no_save:
        return 0

    try:
        outputs = save_depth_products(
            img_path=img_path,
            depth=depth,
            ckpt_path=ckpt_path,
            device=device,
            inference_sec=dt,
            depthpro_meta=dp_meta,
        )
        logger.info("✅ Saved depth products")
        logger.info(f"16-bit PNG: {outputs['png16']}")
        logger.info(f"float NPY : {outputs['npy']}")
        logger.info(f"JSON      : {outputs['json']}")
        return 0
    except Exception as e:
        logger.error(f"❌ Saving outputs failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
