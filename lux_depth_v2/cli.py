from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, Preset
from .logging_utils import setup_logging
from .pipeline import LuxPipelineV2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gold Standard Lux Depth Pipeline V2 (GPU-accelerated, modular).")

    # IO
    p.add_argument("--input", type=str, default=None, help="Single input image path.")
    p.add_argument("--input-dir", type=str, default=None, help="Input directory of images.")
    p.add_argument("--depth-dir", type=str, default=None, help="Depth directory (depth TIFFs and optional zone masks).")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory.")

    # Look
    p.add_argument("--preset", type=str, default=Preset.PHOTO_REALISTIC.value, choices=[e.value for e in Preset])

    # Device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"])

    # Upscaling
    p.add_argument("--upscale", type=int, default=4, choices=[2,4])
    p.add_argument("--upscaler-backend", type=str, default="torch", choices=["torch", "realesrgan", "onnx", "none"])
    p.add_argument("--model-path", type=str, default=None, help="Upscaler model path (.pth or .onnx).")
    p.add_argument("--model-sha256", type=str, default=None, help="Optional SHA256 for verifying model file.")
    p.add_argument("--tile", type=int, default=512, help="Real-ESRGAN tile size (0 disables).")
    p.add_argument("--tile-pad", type=int, default=16)
    p.add_argument("--half", action="store_true", help="Use half precision inside Real-ESRGAN (GPU only).")

    # Segmentation
    p.add_argument("--seg-backend", type=str, default="auto", choices=["auto","onnx","segformer","heuristic","none"])
    p.add_argument("--seg-onnx-model", type=str, default=None, help="Material segmentation ONNX model path.")
    p.add_argument("--seg-onnx-labels", type=str, default=None, help="Optional JSON label mapping for ONNX model.")
    p.add_argument("--seg-segformer-model", type=str, default=None, help="Local dir or HF model id for segformer backend.")
    p.add_argument("--seg-allow-downloads", action="store_true", help="Allow downloading pretrained segformer weights.")
    p.add_argument("--seg-long-side", type=int, default=768)
    p.add_argument("--seg-min-conf", type=float, default=0.25)

    # Service mode (optional)
    p.add_argument("--service", action="store_true", help="Run as HTTP service (FastAPI).")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8088)

    return p


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("INFO")

    cfg = PipelineConfig(
        input_dir=Path(args.input_dir) if args.input_dir else None,
        depth_dir=Path(args.depth_dir) if args.depth_dir else None,
        output_dir=Path(args.output_dir),
        preset=Preset(args.preset),
        device=args.device,
        precision=args.precision,
        upscale=int(args.upscale),
        upscaler_backend=args.upscaler_backend,
        model_path=Path(args.model_path) if args.model_path else None,
        model_sha256=args.model_sha256,
        tile=int(args.tile),
        tile_pad=int(args.tile_pad),
        half=bool(args.half),
    )
    cfg.segmentation.backend = args.seg_backend
    cfg.segmentation.onnx_model_path = Path(args.seg_onnx_model) if args.seg_onnx_model else None
    cfg.segmentation.onnx_labels_path = Path(args.seg_onnx_labels) if args.seg_onnx_labels else None
    cfg.segmentation.segformer_model = args.seg_segformer_model
    cfg.segmentation.allow_downloads = bool(args.seg_allow_downloads)
    cfg.segmentation.input_long_side = int(args.seg_long_side)
    cfg.segmentation.min_confidence = float(args.seg_min_conf)

    if args.service:
        # Defer imports so batch-only users don't need server deps installed.
        from .service import run_service
        run_service(cfg, host=args.host, port=int(args.port), logger=logger)
        return

    pipe = LuxPipelineV2(cfg, logger=logger)

    if args.input:
        rep = pipe.process_one(Path(args.input))
        logger.info(f"Done: {rep.get('status')} | {Path(args.input).name}")
    else:
        pipe.process_directory()


if __name__ == "__main__":
    main()
