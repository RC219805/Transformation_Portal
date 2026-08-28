"""Subprocess worker for isolated Depth Anything 3 inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ...ingest.canonical_json import dump_json


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the DA3 worker."""
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 in an isolated Python environment.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate that DA3 imports are available.",
    )
    parser.add_argument(
        "--model-variant",
        required=True,
        help="ModelVariant enum member name (for example METRIC_LARGE).",
    )
    parser.add_argument(
        "--model-key",
        help="Canonical Lux Depth V3 registry key (for example da3_metric).",
    )
    parser.add_argument(
        "--model-revision",
        help=(
            "Planned model revision from the parent's resolved contract "
            "(P0-1, issue #2065): pins the worker's resolution and model "
            "load to the revision the plan recorded."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device to pass to DA3.",
    )
    parser.add_argument(
        "--use-coreml",
        action="store_true",
        help="Enable the Apple CoreML opt-in when supported.",
    )
    parser.add_argument(
        "--non-commercial-ok",
        action="store_true",
        help="Acknowledge non-commercial registry-selected models for subprocess execution.",
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


def _resolve_model_variant(model_variant_name: str):
    """Resolve a ModelVariant enum member from its symbolic name."""
    from ...lux_depth_v3.config import ModelVariant

    try:
        return ModelVariant[model_variant_name]
    except KeyError as exc:
        raise ValueError(f"Unknown DA3 model variant: {model_variant_name}") from exc


def _requires_custom_da3_library(model_variant) -> bool:
    """Return whether the selected model requires depth-anything-3 imports."""
    model_id = str(model_variant.value.huggingface_id).lower()
    return model_id.startswith("depth-anything/da3") or "da3nested" in model_id


def _check_availability(model_variant_name: str) -> int:
    """Validate imports for subprocess mode."""
    import torch  # noqa: F401
    import transformers  # noqa: F401

    model_variant = _resolve_model_variant(model_variant_name)
    if _requires_custom_da3_library(model_variant):
        try:
            from depth_anything_3.api import DepthAnything3  # noqa: F401
        except ImportError:
            from depth_anything_3 import DepthAnything3  # noqa: F401
    return 0


def _run_inference(
    *,
    input_image: Path,
    output_depth: Path,
    output_json: Path,
    model_variant_name: str,
    model_key: str | None,
    model_revision: str | None,
    device: str,
    use_coreml: bool,
    non_commercial_ok: bool,
) -> int:
    """Run DA3 inference and persist structured outputs."""
    from ...lux_depth_v3.config import DA3Config, DeviceConfig
    from ...lux_depth_v3.inference import DA3InferenceEngine

    image = Image.open(input_image).convert("RGB")
    model_variant = _resolve_model_variant(model_variant_name)
    config = DA3Config(
        model_variant=model_variant,
        model_key=model_key,
        non_commercial_ok=non_commercial_ok,
        model_revision=model_revision,
        device=DeviceConfig(device=device, use_coreml=use_coreml),
    )
    engine = DA3InferenceEngine(
        config=config,
        commercial_use=True,
        validate_license_strict=False,
        model_key=model_key,
        non_commercial_ok=non_commercial_ok,
    )
    result = engine.predict(image)

    output_depth.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_depth, np.asarray(result.depth_map, dtype=np.float32), allow_pickle=False)

    payload = {
        "metadata": result.metadata,
        "device": result.metadata.get("device", device),
        "dtype": "float32",
        "input_size": [
            int(result.original_image.shape[0]),
            int(result.original_image.shape[1]),
        ],
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
    """Entry point for subprocess-backed DA3 execution."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.check:
        return _check_availability(args.model_variant)

    if args.input_image is None or args.output_depth is None or args.output_json is None:
        parser.error("--input-image, --output-depth, and --output-json are required unless --check is used.")

    return _run_inference(
        input_image=args.input_image.expanduser(),
        output_depth=args.output_depth.expanduser(),
        output_json=args.output_json.expanduser(),
        model_variant_name=str(args.model_variant),
        model_key=str(args.model_key) if args.model_key else None,
        model_revision=str(args.model_revision) if args.model_revision else None,
        device=str(args.device),
        use_coreml=bool(args.use_coreml),
        non_commercial_ok=bool(args.non_commercial_ok),
    )


if __name__ == "__main__":
    raise SystemExit(main())
