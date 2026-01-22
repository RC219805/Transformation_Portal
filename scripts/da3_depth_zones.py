#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure repo root is importable even when running "python scripts/..."
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lux_depth_v3.enhance.depth_zones import DepthZoneConfig, DepthZoneGenerator  # noqa: E402


def load_u16_depth_png(p: Path) -> np.ndarray:
    d = np.array(Image.open(p))
    if d.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth png, got {d.dtype} for {p}")
    return (d.astype(np.float32) / 65535.0).clip(0.0, 1.0)


def load_rgb01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_u16_mask(mask01: np.ndarray, out_path: Path) -> None:
    m = (np.clip(mask01, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
    Image.fromarray(m, mode="I;16").save(out_path)


def save_preview(zones: np.ndarray, rgb01: np.ndarray | None, out_path: Path) -> None:
    # zones: (H, W, 4) weights
    if rgb01 is None:
        z1 = (np.clip(zones[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(z1, mode="L").save(out_path)
        return

    # simple overlay: map Z1..Z4 to RGBA-ish colors, blend with image
    # Z1=red, Z2=green, Z3=blue, Z4=yellow
    z1, z2, z3, z4 = [zones[..., i] for i in range(4)]
    overlay = np.stack([z1 + z4, z2 + z4, z3, np.ones_like(z1)], axis=-1)  # RGBA-ish
    overlay_rgb = overlay[..., :3]
    overlay_rgb = np.clip(overlay_rgb / (overlay_rgb.max() + 1e-6), 0.0, 1.0)

    alpha = 0.45
    comp = np.clip((1 - alpha) * rgb01 + alpha * overlay_rgb, 0.0, 1.0)
    out = (comp * 255.0).round().astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate depth zones from DA3 depth maps")
    ap.add_argument("--depth-dir", required=True, type=Path, help="Directory containing *_depth.png (uint16)")
    ap.add_argument("--output-dir", required=True, type=Path, help="Directory to write zone artifacts")
    ap.add_argument("--input-dir", type=Path, default=None, help="Optional directory of images (renders_safe) for previews")
    ap.add_argument("--percentiles", default="10,35,65")
    ap.add_argument("--blend-sigma", type=float, default=5.0)
    ap.add_argument("--apply-sky-heuristic", action="store_true")
    ap.add_argument("--sky-brightness-threshold", type=float, default=0.85)
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pcts = tuple(int(x.strip()) for x in args.percentiles.split(","))
    cfg = DepthZoneConfig(
        percentiles=pcts,
        blend_sigma=args.blend_sigma,
        apply_sky_heuristic=args.apply_sky_heuristic,
        sky_brightness_threshold=args.sky_brightness_threshold,
    )
    gen = DepthZoneGenerator(config=cfg)

    depth_maps = sorted(args.depth_dir.glob("*_depth.png"))
    if not depth_maps:
        raise SystemExit(f"No *_depth.png found in {args.depth_dir}")

    wrote = 0
    for dp in depth_maps:
        stem = dp.name.replace("_depth.png", "")
        depth = load_u16_depth_png(dp)

        rgb01 = None
        if args.input_dir is not None:
            # Accept common image extensions; match by stem
            for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"):
                cand = args.input_dir / f"{stem}{ext}"
                if cand.exists():
                    rgb01 = load_rgb01(cand)
                    break

        zones, stats = gen.generate_zones(depth=depth, image=rgb01)

        # Save per-zone masks
        save_u16_mask(zones[..., 0], out_dir / f"{stem}_zones_Z1.png")
        save_u16_mask(zones[..., 1], out_dir / f"{stem}_zones_Z2.png")
        save_u16_mask(zones[..., 2], out_dir / f"{stem}_zones_Z3.png")
        save_u16_mask(zones[..., 3], out_dir / f"{stem}_zones_Z4.png")

        # Preview + stats
        save_preview(zones, rgb01, out_dir / f"{stem}_zones_preview.png")
        (out_dir / f"{stem}_zone_stats.json").write_text(json.dumps(stats, indent=2))

        wrote += 1
        print(f"WROTE zones: {stem}")

    print(f"✓ Done. Wrote zones for {wrote} images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
