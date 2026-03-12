"""Subprocess worker for isolated Depth Pro inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ...ingest.canonical_json import dump_json


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the Depth Pro worker."""
    parser = argparse.ArgumentParser(
        description="Run Depth Pro in an isolated Python environment.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate that Depth Pro imports and the checkpoint exists.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the Depth Pro checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device to pass to Depth Pro.",
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        help="Input image path for inference mode.",
    )
    parser.add_argument(
        "--output-depth",
        type=Path,
        help="Output .npy path for the depth map.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON path for structured metadata.",
    )
    return parser


def _check_availability(checkpoint: Path) -> int:
    """Validate imports and checkpoint presence for subprocess mode."""
    import depth_pro  # noqa: F401

    from ...stage_graph.stages.depth_pro import DepthProStage

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    _ = DepthProStage
    return 0


def _run_inference(
    input_image: Path,
    output_depth: Path,
    output_json: Path,
    checkpoint: Path,
    device: str,
) -> int:
    """Run Depth Pro inference and persist structured outputs."""
    from ...stage_graph.stage import StageContext, StageStatus
    from ...stage_graph.stages.depth_pro import DepthProStage

    image = Image.open(input_image).convert("RGB")
    stage = DepthProStage(
        checkpoint_path=checkpoint,
        device=device,
        strict_validation=True,
    )
    result = stage.compute(
        StageContext(
            artifacts={"image": image},
            device=device,
        )
    )

    if result.status != StageStatus.COMPLETED:
        message = result.error or "Depth Pro inference failed"
        if result.error_traceback:
            message = f"{message}\nTraceback:\n{result.error_traceback}"
        print(message, file=sys.stderr)
        return 1

    depth_map = result.artifacts.get("depth_map")
    provenance = result.artifacts.get("depth_provenance")
    if depth_map is None:
        print("Depth Pro worker did not receive depth_map output.", file=sys.stderr)
        return 1
    if not isinstance(provenance, dict):
        print("Depth Pro worker did not receive provenance output.", file=sys.stderr)
        return 1

    output_depth.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_depth, np.asarray(depth_map, dtype=np.float32), allow_pickle=False)

    payload = {
        "depth_units": "meters",
        "device": device,
        "dtype": "float32",
        "input_size": [int(image.height), int(image.width)],
        "focal_length_px": result.metadata.get("focal_length_px"),
        "field_of_view_deg": result.metadata.get("fov_deg"),
        "provenance": provenance,
        "warnings": [],
    }
    with output_json.open("w", encoding="utf-8") as handle:
        dump_json(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for subprocess-backed Depth Pro execution."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser()
    if args.check:
        return _check_availability(checkpoint)

    if args.input_image is None or args.output_depth is None or args.output_json is None:
        parser.error("--input-image, --output-depth, and --output-json are required unless --check is used.")

    return _run_inference(
        input_image=args.input_image.expanduser(),
        output_depth=args.output_depth.expanduser(),
        output_json=args.output_json.expanduser(),
        checkpoint=checkpoint,
        device=str(args.device),
    )


if __name__ == "__main__":
    raise SystemExit(main())
