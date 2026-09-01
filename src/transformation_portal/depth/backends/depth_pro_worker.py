"""Subprocess worker for isolated Depth Pro inference."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...ingest.canonical_json import dump_json, dumps_json


@dataclass(frozen=True)
class _CanonicalWorkerAuthority:
    """Revalidated execution authority selected for this worker process."""

    plan: Any
    candidate: Any
    checkpoint: Path
    device: str

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
        help="Path to the Depth Pro checkpoint.",
    )
    parser.add_argument(
        "--device",
        help="Inference device to pass to Depth Pro.",
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
        if args.checkpoint is not None:
            mixed_flags.append("--checkpoint")
        if args.device is not None:
            mixed_flags.append("--device")
        if mixed_flags:
            parser.error(
                "--execution-plan-stdin cannot be combined with legacy checkpoint or device selectors: "
                + ", ".join(mixed_flags)
            )
        return

    if args.candidate_id is not None or args.model_backend_id is not None:
        parser.error("--candidate-id and --model-backend-id require --execution-plan-stdin")
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --execution-plan-stdin is used")


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
    )

    plan_bytes = sys.stdin.buffer.read(MAX_PLAN_BODY_BYTES + 1)
    plan = consume_lux_worker_execution_plan(plan_bytes)
    candidate = backend_candidate_authority(
        plan,
        candidate_id,
        model_backend_id=model_backend_id,
    )
    model_contract = candidate.model_contract
    if model_contract is None or model_contract.backend_id != "depth_pro":
        raise ValueError("Selected canonical worker authority is not a Depth Pro model contract")
    if model_contract.artifact_path is None:
        raise ValueError("Selected canonical Depth Pro authority has no checkpoint path")
    device = str(candidate.device or model_contract.device or "cpu")
    if device == "auto":
        device = "cpu"
    return _CanonicalWorkerAuthority(
        plan=plan,
        candidate=candidate,
        checkpoint=Path(model_contract.artifact_path),
        device=device,
    )


def _torch_diagnostics(device: str) -> dict[str, Any]:
    """Collect structured device diagnostics for readiness checks."""
    diagnostics: dict[str, Any] = {
        "device": device,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    macos_version = platform.mac_ver()[0]
    if macos_version:
        diagnostics["macos_version"] = macos_version

    try:
        import torch
    except ImportError as exc:
        diagnostics["torch_import_error"] = str(exc)
        return diagnostics

    def _safe_bool_call(callback: Any) -> bool:
        if not callable(callback):
            return False
        try:
            return bool(callback())
        except Exception:
            return False

    torch_backends = getattr(torch, "backends", None)
    mps_backend = getattr(torch_backends, "mps", None)
    torch_cuda = getattr(torch, "cuda", None)

    diagnostics["torch_version"] = getattr(torch, "__version__", "unknown")
    diagnostics["mps_built"] = _safe_bool_call(getattr(mps_backend, "is_built", None))
    diagnostics["mps_available"] = _safe_bool_call(getattr(mps_backend, "is_available", None))
    diagnostics["cuda_available"] = _safe_bool_call(getattr(torch_cuda, "is_available", None))
    return diagnostics


def _emit_check_failure(reason: str, diagnostics: dict[str, Any]) -> int:
    """Emit a structured readiness failure for subprocess availability checks."""
    payload = {
        "status": "unavailable",
        "reason": reason,
        **diagnostics,
    }
    print(dumps_json(payload, sort_keys=True), file=sys.stderr)
    return 1


def _check_device_availability(device: str) -> int:
    """Validate that the requested device is actually usable."""
    normalized_device = str(device or "cpu").strip().lower() or "cpu"
    diagnostics = _torch_diagnostics(normalized_device)

    if normalized_device == "cpu":
        return 0

    if "torch_import_error" in diagnostics:
        return _emit_check_failure("PyTorch import failed for device readiness check.", diagnostics)

    if normalized_device == "mps":
        if not diagnostics.get("mps_built"):
            return _emit_check_failure("PyTorch was not built with MPS support.", diagnostics)
        if not diagnostics.get("mps_available"):
            return _emit_check_failure("PyTorch MPS backend is not available in this runtime.", diagnostics)
        return 0

    if normalized_device == "cuda":
        if not diagnostics.get("cuda_available"):
            return _emit_check_failure("PyTorch CUDA backend is not available in this runtime.", diagnostics)
        return 0

    return _emit_check_failure(f"Unsupported depth device: {normalized_device}", diagnostics)


def _check_availability(checkpoint: Path, device: str) -> int:
    """Validate imports, checkpoint presence, and requested device readiness."""
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    import depth_pro  # noqa: F401

    from ...stage_graph.stages.depth_pro import DepthProStage

    _ = DepthProStage
    return _check_device_availability(device)


def _run_inference(
    input_image: Path,
    output_depth: Path,
    output_json: Path,
    checkpoint: Path,
    device: str,
    canonical_authority: _CanonicalWorkerAuthority | None = None,
) -> int:
    """Run Depth Pro inference and persist structured outputs."""
    import numpy as np
    from PIL import Image

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
    if canonical_authority is not None:
        payload["execution_authority"] = {
            "plan_fingerprint_sha256": canonical_authority.plan_fingerprint_sha256,
            "candidate_id": canonical_authority.candidate_id,
            "model_backend_id": canonical_authority.model_backend_id,
            "executed_backend_id": "depth_pro",
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
        checkpoint = canonical_authority.checkpoint.expanduser()
        device = canonical_authority.device
    else:
        assert args.checkpoint is not None
        checkpoint = args.checkpoint.expanduser()
        device = str(args.device or "cpu")

    if args.check:
        return _check_availability(checkpoint, device)

    if args.input_image is None or args.output_depth is None or args.output_json is None:
        parser.error("--input-image, --output-depth, and --output-json are required unless --check is used.")

    return _run_inference(
        input_image=args.input_image.expanduser(),
        output_depth=args.output_depth.expanduser(),
        output_json=args.output_json.expanduser(),
        checkpoint=checkpoint,
        device=device,
        canonical_authority=canonical_authority,
    )


if __name__ == "__main__":
    raise SystemExit(main())
