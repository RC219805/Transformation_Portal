"""Standalone CLI for advisory FastVLM caption sidecars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fastvlm_runtime import (
    FastVLMRuntimeConfig,
    build_fastvlm_sidecar,
    dumps_sidecar,
    run_fastvlm_caption,
)
from .image_proxy import build_vlm_image_proxy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an advisory FastVLM caption sidecar.")
    parser.add_argument("--input-image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--fastvlm-python", required=True, type=Path)
    parser.add_argument("--mlx-vlm-dir", required=True, type=Path)
    parser.add_argument("--proxy-format", choices=("png", "jpeg"), default="png")
    parser.add_argument("--max-side-px", type=int, default=1600)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    proxy = build_vlm_image_proxy(
        args.input_image,
        args.output_dir,
        max_side_px=args.max_side_px,
        format=args.proxy_format,
        output_name="image_proxy.jpg" if args.proxy_format == "jpeg" else "image_proxy.png",
    )

    config = FastVLMRuntimeConfig(
        enabled=True,
        python_path=args.fastvlm_python,
        mlx_vlm_dir=args.mlx_vlm_dir,
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )
    result = run_fastvlm_caption(config, proxy.proxy_path)
    raw_path = args.output_dir / "vlm_captioning.raw.txt"
    raw_path.write_text(result.raw_stdout, encoding="utf-8")
    sidecar = build_fastvlm_sidecar(
        enabled=True,
        model_path=args.model_path,
        image_proxy=proxy,
        runtime_result=result,
    )
    (args.output_dir / "vlm_captioning.sidecar.json").write_text(dumps_sidecar(sidecar), encoding="utf-8")
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
