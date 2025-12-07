#!/usr/bin/env python3
"""
Lux Depth v2/v2.1 performance benchmark: CPU vs MPS vs CUDA.

Key properties:
- Forces specific device (cpu / mps / cuda[:idx]), avoids "auto" ambiguity.
- Correct timing for async backends (cuda/mps synchronize).
- Warmup + multiple trials → median + p90.
- Optional "compute-only" mode to avoid disk I/O dominating results.
- Writes JSON/CSV summary for regression tracking.

Usage examples:
  # Compute-only (recommended for device comparisons)
  python tools/bench_perf.py \
    --input input_images/.../image.tif \
    --depth-dir output_depth_maps \
    --devices cpu mps \
    --upscales 2 4 \
    --runs 5 --warmup 1 \
    --io compute

  # End-to-end (includes read/write)
  python tools/bench_perf.py \
    --input ... \
    --depth-dir ... \
    --devices cpu mps \
    --upscales 2 4 \
    --runs 3 --warmup 1 \
    --io end_to_end
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# If you want to benchmark v2.1, swap these imports accordingly.
# from lux_depth_v21.pipeline import LuxPipelineV21 as LuxPipeline
# from lux_depth_v21.config import PipelineConfig, Preset

from lux_depth_v2.pipeline import LuxPipelineV2 as LuxPipeline
from lux_depth_v2.config import PipelineConfig, Preset


def _now() -> float:
    return time.perf_counter()


def _torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


_TORCH = _torch()


def sync_if_needed(device_str: str) -> None:
    """Synchronize async devices so timing reflects real work."""
    if _TORCH is None:
        return

    d = str(device_str).lower()
    if "cuda" in d and hasattr(_TORCH, "cuda") and _TORCH.cuda.is_available():
        _TORCH.cuda.synchronize()
    elif d.startswith("mps") and hasattr(_TORCH, "mps") and _TORCH.backends.mps.is_available():
        _TORCH.mps.synchronize()


def pct(values: List[float], q: float) -> float:
    """Percentile using simple nearest-rank."""
    if not values:
        return float("nan")
    v = sorted(values)
    k = int(round((len(v) - 1) * q))
    return v[max(0, min(len(v) - 1, k))]


def maybe_disable_outputs(config: PipelineConfig) -> None:
    """
    Best-effort disable disk outputs for compute-only benchmarking.
    Works across slightly different config schemas.
    """
    # Common toggles (set only if present)
    for attr in (
        "write_outputs",
        "save_master",
        "save_upscaled",
        "save_marketing_png",
        "save_preview_jpg",
        "save_reports",
        "save_debug",
        "dump_debug",
    ):
        if hasattr(config, attr):
            setattr(config, attr, False)

    # Some pipelines always write at least report/master;
    # keep output_dir but put it in a temp-ish location.
    if hasattr(config, "output_dir"):
        setattr(config, "output_dir", Path("_bench_outputs__compute_only"))


def configure_pipeline(
    *,
    device: str,
    upscale: int,
    input_path: Path,
    depth_dir: Path,
    output_root: Path,
    preset: str,
    upscaler_backend: str,
    io_mode: str,
) -> LuxPipeline:
    cfg = PipelineConfig()

    # preset
    preset_map = {
        "exterior_showcase": Preset.EXTERIOR_SHOWCASE,
        "interior_luxury": Preset.INTERIOR_LUXURY,
        "photo_realistic": Preset.PHOTO_REALISTIC,
        "archival_quality": Preset.ARCHIVAL_QUALITY,
        "architectural": Preset.ARCHITECTURAL,
    }
    cfg.preset = preset_map.get(preset.lower(), Preset.EXTERIOR_SHOWCASE)

    # device and io
    cfg.device = device
    cfg.depth_dir = depth_dir
    cfg.upscaler_backend = upscaler_backend
    cfg.upscale = upscale

    # outputs: keep minimal by default (benchmark mode should not be I/O-bound)
    cfg.save_upscaled = False
    cfg.save_marketing_png = False
    cfg.save_preview_jpg = False

    # output dir per test when end-to-end is requested
    cfg.output_dir = output_root / f"bench_{device.replace(':','_')}_{upscale}x"

    if io_mode == "compute":
        # Use the new write_outputs kill-switch for true compute-only mode
        if hasattr(cfg, "write_outputs"):
            cfg.write_outputs = False
        else:
            # Fallback for older pipeline versions
            maybe_disable_outputs(cfg)
    elif io_mode == "end_to_end":
        # allow saving master/upscaled if your pipeline requires it; otherwise stay minimal
        if hasattr(cfg, "write_outputs"):
            cfg.write_outputs = True
        if hasattr(cfg, "save_upscaled"):
            cfg.save_upscaled = True

    pipe = LuxPipeline(cfg)
    return pipe


def run_trials(
    pipe: LuxPipeline,
    input_path: Path,
    runs: int,
    warmup: int,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    times: List[float] = []
    reports: List[Dict[str, Any]] = []

    resolved_device = str(getattr(pipe, "device", "unknown"))

    # warmup (do not record)
    for _ in range(warmup):
        sync_if_needed(resolved_device)
        _ = pipe.process_one(input_path)
        sync_if_needed(resolved_device)

    for _ in range(runs):
        sync_if_needed(resolved_device)
        t0 = _now()
        report = pipe.process_one(input_path)
        sync_if_needed(resolved_device)
        t1 = _now()
        times.append(t1 - t0)
        reports.append(report if isinstance(report, dict) else {"report": str(report)})

    return times, reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--depth-dir", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=Path("bench_outputs"))
    ap.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    ap.add_argument("--upscales", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--preset", type=str, default="exterior_showcase")
    ap.add_argument("--upscaler-backend", type=str, default="torch")  # "torch" is interpolation in many builds
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--io", choices=["compute", "end_to_end"], default="compute")
    ap.add_argument("--csv", type=Path, default=Path("bench_results.csv"))
    ap.add_argument("--json", type=Path, default=Path("bench_results.json"))
    args = ap.parse_args()

    input_path: Path = args.input
    depth_dir: Path = args.depth_dir
    output_root: Path = args.output_root

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    if not depth_dir.exists():
        raise SystemExit(f"Depth dir not found: {depth_dir}")

    # Warn about MPS fallback (can silently move ops to CPU and ruin comparisons)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        print("WARNING: PYTORCH_ENABLE_MPS_FALLBACK=1 is set. "
              "Unsupported ops may silently run on CPU, skewing MPS results.")

    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    print("=" * 88)
    print(f"Lux Depth Benchmark | io={args.io} | runs={args.runs} | warmup={args.warmup} | backend={args.upscaler_backend}")
    print("=" * 88)

    for dev in args.devices:
        for up in args.upscales:
            print(f"\nRunning: device={dev} upscale={up}x ...")
            pipe = configure_pipeline(
                device=dev,
                upscale=up,
                input_path=input_path,
                depth_dir=depth_dir,
                output_root=output_root,
                preset=args.preset,
                upscaler_backend=args.upscaler_backend,
                io_mode=args.io,
            )

            times, reports = run_trials(pipe, input_path, runs=args.runs, warmup=args.warmup)

            med = stats.median(times)
            p90 = pct(times, 0.90)
            imgs_hr = 3600.0 / med if med > 0 else 0.0

            # Pull a few useful diagnostics if present
            last = reports[-1] if reports else {}
            ai_color = float(last.get("ai_color_diff", 0.0) or 0.0)
            ai_luma = float(last.get("ai_luma_diff", 0.0) or 0.0)
            stage_times = last.get("stage_times_sec") or last.get("stage_times") or {}

            resolved_device = str(getattr(pipe, "device", dev))

            print(f"  resolved_device: {resolved_device}")
            print(f"  median: {med:.3f}s | p90: {p90:.3f}s | {imgs_hr:.1f} img/hr")
            if ai_color or ai_luma:
                print(f"  ai drift: color={ai_color:.6f} luma={ai_luma:.6f}")
            if isinstance(stage_times, dict) and stage_times:
                # show top 5 stages
                top = sorted(stage_times.items(), key=lambda kv: float(kv[1]), reverse=True)[:5]
                print("  top stages:")
                for k, v in top:
                    print(f"    {k:24s} {float(v):.3f}s")

            rows.append({
                "device_requested": dev,
                "device_resolved": resolved_device,
                "upscale": up,
                "io_mode": args.io,
                "upscaler_backend": args.upscaler_backend,
                "runs": args.runs,
                "warmup": args.warmup,
                "median_sec": med,
                "p90_sec": p90,
                "img_per_hour": imgs_hr,
                "ai_color_diff": ai_color,
                "ai_luma_diff": ai_luma,
            })

    # Write CSV/JSON
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"results": rows}, indent=2))

    print("\nDone.")
    print(f"CSV:  {args.csv}")
    print(f"JSON: {args.json}")


if __name__ == "__main__":
    main()
