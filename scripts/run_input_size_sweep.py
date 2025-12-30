#!/usr/bin/env python3
"""
Run a sweep over multiple inference input sizes using Depth Anything V2 Large HF.
Usage:
  python scripts/run_input_size_sweep.py \
      --input-dir data/structure_subset \
      --output-dir outputs/sweep_202512XX \
      --sizes 518 768 896 1022
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
from transformers import pipeline
from PIL import Image
import torch


def run_depth(image_path, model_id, input_size, out_dir, device):
    """Invoke depth estimation pipeline and save metrics + output."""
    print(f"  Processing {Path(image_path).name} at size {input_size}...")

    # Load pipeline
    pipe = pipeline("depth-estimation", model=model_id, device=device)

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Infer depth
    result = pipe(img)
    depth = np.array(result["depth"])

    # Save depth map
    out_npy = out_dir / f"{Path(image_path).stem}_depth_{input_size}.npy"
    np.save(out_npy, depth)

    # Save metadata
    meta = {
        "image": Path(image_path).name,
        "input_size": input_size,
        "depth_shape": depth.shape,
        "depth_range": [float(depth.min()), float(depth.max())],
        "depth_mean": float(depth.mean()),
        "depth_std": float(depth.std()),
    }

    meta_path = out_dir / f"{Path(image_path).stem}_meta_{input_size}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(out_npy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-V2-Large-hf")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    images = list(Path(args.input_dir).glob("*.[jp][pn]g"))
    images.extend(list(Path(args.input_dir).glob("*.jpeg")))
    images = sorted(set(images))

    print(f"\n{'=' * 60}")
    print(f"INPUT SIZE SWEEP - Depth Anything V2 Large HF")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Images: {len(images)}")
    print(f"Sizes: {args.sizes}")
    print(f"Output: {out_base}")

    for size in args.sizes:
        print(f"\n{'=' * 60}")
        print(f"Running input_size={size}")
        print(f"{'=' * 60}")

        sweep_dir = out_base / f"input_{size}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            try:
                run_depth(str(img), args.model_id, size, sweep_dir, args.device)
            except Exception as e:
                print(f"  ERROR on {img.name}: {e}")
                continue

    print(f"\n{'=' * 60}")
    print("Sweep complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
