"""Export a custom material segmentation PyTorch model to ONNX.

This is a scaffold to standardize production ONNX I/O expected by lux_depth_v2:

  input:  1x3xHxW float32 RGB in [0,1]
  output: 1xCxHxW logits (C == number of materials/classes)

You will need to edit `load_model()` to construct your model and load weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn


class InputNorm(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,3,1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def load_model(ckpt: Path) -> nn.Module:
    # Replace with actual model definition when implementing custom ONNX export
    raise NotImplementedError("Edit load_model() to return your material segmentation model.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--onnx-out", type=str, required=True)
    ap.add_argument("--labels-out", type=str, required=True, help="JSON mapping class index->surface name")
    ap.add_argument("--surfaces", type=str, required=True, help="Comma-separated list of material names matching model channels")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    onnx_out = Path(args.onnx_out)
    labels_out = Path(args.labels_out)
    surfaces: List[str] = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    if not surfaces:
        raise ValueError("No surfaces provided")

    model = load_model(ckpt)
    model.eval()

    wrapped = nn.Sequential(InputNorm(), model)

    dummy = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    dynamic_axes = {"input": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}}

    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
    )

    import json
    labels = {str(i): surfaces[i] for i in range(len(surfaces))}
    labels_out.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print(f"Wrote: {onnx_out}")
    print(f"Wrote: {labels_out}")


if __name__ == "__main__":
    main()
