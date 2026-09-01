"""Subprocess worker for isolated Depth Anything 3 inference."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...ingest.canonical_json import dump_json


@dataclass(frozen=True)
class _CanonicalWorkerAuthority:
    """Revalidated execution authority selected for this worker process."""

    plan: Any
    candidate: Any
    model_contract: Any
    resolved_model_contract: Any
    device: str
    use_coreml: bool
    non_commercial_ok: bool

    @property
    def plan_fingerprint_sha256(self) -> str:
        return str(self.plan.plan_fingerprint_sha256)

    @property
    def candidate_id(self) -> str:
        return str(self.candidate.candidate_id)

    @property
    def model_backend_id(self) -> str | None:
        value = getattr(self.candidate, "constituent_backend_id", None)
        return None if value is None else str(value)


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
        "--execution-plan-stdin",
        action="store_true",
        help="Read one canonical tp.execution.plan.v1 object from stdin.",
    )
    parser.add_argument(
        "--candidate-id",
        help="Exact carried backend candidate identifier for canonical plan mode.",
    )
    parser.add_argument(
        "--model-backend-id",
        help="Exact ensemble constituent backend identifier for canonical plan mode.",
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


def _validate_execution_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Enforce a closed canonical mode while preserving the legacy argv path."""

    if args.execution_plan_stdin:
        if not args.candidate_id:
            parser.error("--candidate-id is required with --execution-plan-stdin")
        mixed_flags = []
        for option, value in (
            ("--model-variant", args.model_variant),
            ("--model-key", args.model_key),
            ("--model-revision", args.model_revision),
            ("--device", args.device),
        ):
            if value is not None:
                mixed_flags.append(option)
        if args.use_coreml:
            mixed_flags.append("--use-coreml")
        if args.non_commercial_ok:
            mixed_flags.append("--non-commercial-ok")
        if mixed_flags:
            parser.error(
                "--execution-plan-stdin cannot be combined with legacy model, license, or device selectors: "
                + ", ".join(mixed_flags)
            )
        return

    if args.candidate_id is not None or args.model_backend_id is not None:
        parser.error("--candidate-id and --model-backend-id require --execution-plan-stdin")
    if args.model_variant is None:
        parser.error("--model-variant is required unless --execution-plan-stdin is used")


def _consume_canonical_worker_authority(
    *,
    candidate_id: str,
    model_backend_id: str | None,
) -> _CanonicalWorkerAuthority:
    """Bounded-read and revalidate the exact execution authority from stdin."""

    from ...core.execution_plan import MAX_PLAN_BODY_BYTES
    from ...lux_depth_v3.execution_lifecycle import (
        backend_candidate_authority,
        consume_lux_worker_execution_plan,
        runtime_model_contract_from_candidate,
    )

    plan_bytes = sys.stdin.buffer.read(MAX_PLAN_BODY_BYTES + 1)
    plan = consume_lux_worker_execution_plan(plan_bytes)
    candidate = backend_candidate_authority(
        plan,
        candidate_id,
        model_backend_id=model_backend_id,
    )
    model_contract = candidate.model_contract
    if model_contract is None or model_contract.backend_id != "da3":
        raise ValueError("Selected canonical worker authority is not a DA3 model contract")
    resolved_model_contract = runtime_model_contract_from_candidate(candidate)
    if resolved_model_contract is None:
        raise ValueError("Selected canonical DA3 authority has no executable model contract")
    device = str(model_contract.device or "cpu")
    if device == "auto":
        device = "cpu"
    return _CanonicalWorkerAuthority(
        plan=plan,
        candidate=candidate,
        model_contract=model_contract,
        resolved_model_contract=resolved_model_contract,
        device=device,
        use_coreml=str(model_contract.model.accelerator_kind or "") == "coreml",
        non_commercial_ok=bool(plan.license_acknowledgements.non_commercial_ok),
    )


def _resolve_model_variant(model_variant_name: str) -> Any:
    """Resolve a ModelVariant enum member from its symbolic name."""
    from ...lux_depth_v3.config import ModelVariant

    try:
        return ModelVariant[model_variant_name]
    except KeyError as exc:
        raise ValueError(f"Unknown DA3 model variant: {model_variant_name}") from exc


def _requires_custom_da3_library(model_variant: Any) -> bool:
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
    model_revision: str | None = None,
    device: str,
    use_coreml: bool,
    non_commercial_ok: bool,
    canonical_authority: _CanonicalWorkerAuthority | None = None,
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
        resolved_model_contract=(canonical_authority.resolved_model_contract if canonical_authority is not None else None),
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
    if canonical_authority is not None:
        payload["execution_authority"] = {
            "plan_fingerprint_sha256": canonical_authority.plan_fingerprint_sha256,
            "candidate_id": canonical_authority.candidate_id,
            "model_backend_id": canonical_authority.model_backend_id,
            "executed_backend_id": "da3",
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
    _validate_execution_mode(parser, args)

    canonical_authority = None
    if args.execution_plan_stdin:
        try:
            canonical_authority = _consume_canonical_worker_authority(
                candidate_id=str(args.candidate_id),
                model_backend_id=str(args.model_backend_id) if args.model_backend_id else None,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))

    if canonical_authority is not None:
        resolved = canonical_authority.resolved_model_contract
        from ...lux_depth_v3.config_resolver import _compat_model_variant_for_resolved_key

        model_variant_name = _compat_model_variant_for_resolved_key(resolved.canonical_key).name
        model_key = resolved.canonical_key
        model_revision = resolved.revision
        device = canonical_authority.device
        use_coreml = canonical_authority.use_coreml
        non_commercial_ok = canonical_authority.non_commercial_ok
    else:
        model_variant_name = str(args.model_variant)
        model_key = str(args.model_key) if args.model_key else None
        model_revision = str(args.model_revision) if args.model_revision else None
        device = str(args.device or "cpu")
        use_coreml = bool(args.use_coreml)
        non_commercial_ok = bool(args.non_commercial_ok)

    if args.check:
        return _check_availability(model_variant_name)

    if args.input_image is None or args.output_depth is None or args.output_json is None:
        parser.error("--input-image, --output-depth, and --output-json are required unless --check is used.")

    return _run_inference(
        input_image=args.input_image.expanduser(),
        output_depth=args.output_depth.expanduser(),
        output_json=args.output_json.expanduser(),
        model_variant_name=model_variant_name,
        model_key=model_key,
        model_revision=model_revision,
        device=device,
        use_coreml=use_coreml,
        non_commercial_ok=non_commercial_ok,
        canonical_authority=canonical_authority,
    )


if __name__ == "__main__":
    raise SystemExit(main())
