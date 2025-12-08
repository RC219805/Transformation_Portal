#!/usr/bin/env python3
"""
Run a single-image hardened processing pass.

Example:
  python scripts/hardening/run_hardened_smoke.py \
    --input input_images/.../image.tif \
    --output out_smoke \
    --preset INTERIOR_LUXURY
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.hardening import HardeningPolicy, LuxPipelineV2Hardened


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--preset", default="PHOTO_REALISTIC")
    ap.add_argument("--policy", default=None, help="Path to hardening policy JSON")
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.output_dir = Path(args.output)
    cfg.preset = getattr(Preset, args.preset)

    policy = HardeningPolicy.load(Path(args.policy)) if args.policy else HardeningPolicy.load()
    pipe = LuxPipelineV2Hardened(cfg, policy=policy)

    rep = pipe.process_one(Path(args.input))
    print(rep)


if __name__ == "__main__":
    main()
