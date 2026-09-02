"""Subprocess worker for isolated Depth Anything 3 inference."""

from __future__ import annotations

import argparse
import hashlib
import os
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
        "--prepare-runtime-identity",
        action="store_true",
        help="Prepare local-only, fail-closed runtime evidence without inference.",
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
    parser.add_argument(
        "--output-runtime-identity",
        type=Path,
        help="Output JSON path for prepared DA3 runtime evidence.",
    )
    parser.add_argument(
        "--expected-runtime-identity-sha256",
        help="Prepared runtime digest that inference must re-materialize and echo.",
    )
    parser.add_argument(
        "--runtime-verification-token",
        type=Path,
        help="Canonical stat token produced by the parent-verified preparation worker.",
    )
    parser.add_argument(
        "--runtime-verification-token-sha256",
        help="Parent-verified canonical digest of --runtime-verification-token.",
    )
    return parser


def _validate_execution_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Enforce a closed canonical mode while preserving the legacy argv path."""

    if args.check and args.prepare_runtime_identity:
        parser.error("--check and --prepare-runtime-identity are mutually exclusive")
    if args.prepare_runtime_identity and args.output_runtime_identity is None:
        parser.error("--output-runtime-identity is required with --prepare-runtime-identity")
    if args.output_runtime_identity is not None and not args.prepare_runtime_identity:
        parser.error("--output-runtime-identity requires --prepare-runtime-identity")
    expected_runtime_identity = args.expected_runtime_identity_sha256
    if expected_runtime_identity is not None:
        normalized = str(expected_runtime_identity)
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            parser.error("--expected-runtime-identity-sha256 must be a lowercase SHA-256 digest")
        if args.check or args.prepare_runtime_identity:
            parser.error("--expected-runtime-identity-sha256 is only valid for inference")
        if args.runtime_verification_token is None or args.runtime_verification_token_sha256 is None:
            parser.error("prepared inference requires the runtime verification token and its digest")
    elif args.runtime_verification_token is not None or args.runtime_verification_token_sha256 is not None:
        parser.error("runtime verification token arguments require --expected-runtime-identity-sha256")
    token_sha256 = args.runtime_verification_token_sha256
    if token_sha256 is not None and (
        len(token_sha256) != 64 or any(character not in "0123456789abcdef" for character in token_sha256)
    ):
        parser.error("--runtime-verification-token-sha256 must be a lowercase SHA-256 digest")

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


def _build_inference_engine(
    *,
    model_variant_name: str,
    model_key: str | None,
    model_revision: str | None,
    device: str,
    use_coreml: bool,
    non_commercial_ok: bool,
    canonical_authority: _CanonicalWorkerAuthority | None,
) -> Any:
    """Construct the lazy DA3 engine without materializing model tensors."""

    from ...lux_depth_v3.config import DA3Config, DeviceConfig
    from ...lux_depth_v3.inference import DA3InferenceEngine

    model_variant = _resolve_model_variant(model_variant_name)
    config = DA3Config(
        model_variant=model_variant,
        model_key=model_key,
        non_commercial_ok=non_commercial_ok,
        model_revision=model_revision,
        resolved_model_contract=(canonical_authority.resolved_model_contract if canonical_authority is not None else None),
        device=DeviceConfig(device=device, use_coreml=use_coreml),
    )
    return DA3InferenceEngine(
        config=config,
        commercial_use=True,
        validate_license_strict=False,
        model_key=model_key,
        non_commercial_ok=non_commercial_ok,
    )


def _prepare_worker_runtime_identity(
    *,
    engine: Any,
    model_key: str | None,
    model_revision: str | None,
    requested_device: str,
    use_coreml: bool,
    canonical_authority: _CanonicalWorkerAuthority | None,
) -> Any:
    """Prepare local-only worker evidence and adapt canonical plans to core."""

    from .da3_runtime_identity import (
        build_prepared_cache_runtime_evidence,
        prepare_da3_runtime_identity_with_verification_token,
    )

    if canonical_authority is not None:
        resolved = canonical_authority.resolved_model_contract
    else:
        resolved = engine._resolve_model_contract(use_coreml_backend=use_coreml)  # noqa: SLF001
    if model_revision and resolved.revision != model_revision:
        raise RuntimeError("DA3 runtime preparation resolved a different model revision")

    backend_value = getattr(getattr(engine, "backend", None), "value", None)
    executed_backend = str(backend_value or getattr(engine, "backend", "unknown"))
    actual_device = str(getattr(engine, "device", "unknown"))
    evidence, verification_token = prepare_da3_runtime_identity_with_verification_token(
        model_canonical_key=str(resolved.canonical_key),
        model_repo_id=str(resolved.spec.repo_id),
        model_lock_revision=resolved.revision,
        requested_device=requested_device,
        actual_device=actual_device,
        executed_backend=executed_backend,
    )
    prepared = None
    if canonical_authority is not None:
        prepared = build_prepared_cache_runtime_evidence(
            evidence,
            plan=canonical_authority.plan,
            candidate_authority=canonical_authority.candidate,
        )
    return evidence, prepared, verification_token


def _require_cache_authorized_engine_runtime(engine: Any, *, expected_device: str) -> tuple[str, str]:
    """Reject a fresh inference engine that differs from prepared authority."""

    expected_backends = {"cpu": "pytorch_cpu", "mps": "pytorch_mps"}
    expected_backend = expected_backends.get(expected_device)
    backend_value = getattr(getattr(engine, "backend", None), "value", None)
    actual_backend = str(backend_value or getattr(engine, "backend", "unknown"))
    actual_device = str(getattr(engine, "device", "unknown"))
    if expected_backend is None or actual_device != expected_device or actual_backend != expected_backend:
        raise RuntimeError(
            "DA3 inference engine differs from prepared cache authority: "
            f"expected {expected_backend or 'unsupported'}/{expected_device}, "
            f"observed {actual_backend}/{actual_device}"
        )
    return actual_backend, actual_device


def _write_runtime_identity_report(
    output_path: Path,
    *,
    evidence: Any,
    prepared: Any | None,
    verification_token: Any | None,
) -> None:
    """Write the closed worker preparation response."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    from .da3_runtime_identity import runtime_verification_token_sha256

    payload = {
        "schema": "tp.da3.worker-runtime-handshake.v1",
        "runtime_evidence": evidence.to_mapping(),
        "prepared_cache_runtime": None if prepared is None else prepared.to_payload(),
        "runtime_identity_sha256": None if prepared is None else prepared.runtime_identity_sha256,
        "runtime_verification_token": verification_token,
        "runtime_verification_token_sha256": (
            None if verification_token is None else runtime_verification_token_sha256(verification_token)
        ),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        dump_json(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


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
    expected_runtime_identity_sha256: str | None = None,
    runtime_verification_token_path: Path | None = None,
    runtime_verification_token_sha256: str | None = None,
) -> int:
    """Run DA3 inference and persist structured outputs."""
    verification_token = None
    bound_runtime_identity_sha256: str | None = None
    expected_bound_backend = {"cpu": "pytorch_cpu", "mps": "pytorch_mps"}.get(device)
    if expected_runtime_identity_sha256 is not None:
        from .da3_runtime_identity import (
            load_da3_worker_runtime_handshake,
            verify_runtime_verification_token,
        )

        if runtime_verification_token_path is None or runtime_verification_token_sha256 is None:
            raise RuntimeError("DA3 inference is missing its runtime verification token")
        verification_token = load_da3_worker_runtime_handshake(
            runtime_verification_token_path,
            maximum_bytes=32 * 1024 * 1024,
        )
        worker_runtime_digest = verification_token.get("worker_runtime_identity_sha256")
        if (
            expected_bound_backend is None
            or not isinstance(worker_runtime_digest, str)
            or not verify_runtime_verification_token(
                verification_token,
                expected_token_sha256=runtime_verification_token_sha256,
                expected_worker_runtime_identity_sha256=worker_runtime_digest,
                expected_prepared_runtime_identity_sha256=expected_runtime_identity_sha256,
                expected_requested_device=device,
                expected_actual_device=device,
                expected_executed_backend=expected_bound_backend,
                revalidate_worker_import_environment=True,
            )
        ):
            raise RuntimeError("DA3 worker runtime verification token is stale or invalid")
        bound_runtime_identity_sha256 = str(verification_token["prepared_runtime"]["runtime_identity_sha256"])

    image = Image.open(input_image).convert("RGB")
    engine = _build_inference_engine(
        model_variant_name=model_variant_name,
        model_key=model_key,
        model_revision=model_revision,
        device=device,
        use_coreml=use_coreml,
        non_commercial_ok=non_commercial_ok,
        canonical_authority=canonical_authority,
    )

    expected_engine_runtime: tuple[str, str] | None = None
    if expected_runtime_identity_sha256 is not None:
        expected_engine_runtime = _require_cache_authorized_engine_runtime(engine, expected_device=device)

    if expected_runtime_identity_sha256 is not None:
        # The preparation path is deliberately local-only.  Once it has
        # authorized a miss execution, prevent the normal loader from changing
        # the snapshot over the network between preparation and inference.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    result = engine.predict(image)

    if expected_runtime_identity_sha256 is not None:
        from .da3_runtime_identity import verify_runtime_verification_token

        if verification_token is None or runtime_verification_token_sha256 is None:
            raise RuntimeError("DA3 worker runtime verification token was not retained")
        worker_runtime_digest = str(verification_token["worker_runtime_identity_sha256"])
        if not verify_runtime_verification_token(
            verification_token,
            expected_token_sha256=runtime_verification_token_sha256,
            expected_worker_runtime_identity_sha256=worker_runtime_digest,
            expected_prepared_runtime_identity_sha256=expected_runtime_identity_sha256,
            expected_requested_device=device,
            expected_actual_device=device,
            expected_executed_backend=expected_bound_backend,
            revalidate_worker_import_environment=True,
        ):
            raise RuntimeError("DA3 worker runtime identity changed during inference")
        observed_engine_runtime = _require_cache_authorized_engine_runtime(engine, expected_device=device)
        result_metadata = getattr(result, "metadata", None)
        if (
            expected_engine_runtime != observed_engine_runtime
            or not isinstance(result_metadata, dict)
            or result_metadata.get("backend") != expected_engine_runtime[0]
            or result_metadata.get("device") != expected_engine_runtime[1]
        ):
            raise RuntimeError("DA3 inference result differs from prepared cache authority")

    output_depth.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    depth_array = np.asarray(result.depth_map, dtype=np.float32)
    np.save(output_depth, depth_array, allow_pickle=False)
    depth_digest = hashlib.sha256()
    with output_depth.open("rb") as depth_handle:
        while True:
            chunk = depth_handle.read(1024 * 1024)
            if not chunk:
                break
            depth_digest.update(chunk)
    depth_size_bytes = output_depth.stat().st_size

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
    if expected_runtime_identity_sha256 is not None:
        if bound_runtime_identity_sha256 is None:
            raise RuntimeError("DA3 inference did not retain its authenticated runtime identity")
        payload["runtime_identity_sha256"] = bound_runtime_identity_sha256
        payload["depth_artifact"] = {
            "sha256": depth_digest.hexdigest(),
            "size_bytes": depth_size_bytes,
            "shape": [int(value) for value in depth_array.shape],
            "dtype": str(depth_array.dtype),
            "fortran_order": bool(depth_array.flags.f_contiguous and not depth_array.flags.c_contiguous),
        }
    with output_json.open("w", encoding="utf-8") as handle:
        dump_json(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
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

    if args.prepare_runtime_identity:
        engine = _build_inference_engine(
            model_variant_name=model_variant_name,
            model_key=model_key,
            model_revision=model_revision,
            device=device,
            use_coreml=use_coreml,
            non_commercial_ok=non_commercial_ok,
            canonical_authority=canonical_authority,
        )
        evidence, prepared, verification_token = _prepare_worker_runtime_identity(
            engine=engine,
            model_key=model_key,
            model_revision=model_revision,
            requested_device=device,
            use_coreml=use_coreml,
            canonical_authority=canonical_authority,
        )
        if args.output_runtime_identity is None:  # pragma: no cover - parser enforces this
            parser.error("--output-runtime-identity is required with --prepare-runtime-identity")
        _write_runtime_identity_report(
            args.output_runtime_identity.expanduser(),
            evidence=evidence,
            prepared=prepared,
            verification_token=verification_token,
        )
        return 0

    if args.input_image is None or args.output_depth is None or args.output_json is None:
        parser.error(
            "--input-image, --output-depth, and --output-json are required unless "
            "--check or --prepare-runtime-identity is used."
        )

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
        expected_runtime_identity_sha256=(
            str(args.expected_runtime_identity_sha256) if args.expected_runtime_identity_sha256 else None
        ),
        runtime_verification_token_path=(
            args.runtime_verification_token.expanduser() if args.runtime_verification_token else None
        ),
        runtime_verification_token_sha256=(
            str(args.runtime_verification_token_sha256) if args.runtime_verification_token_sha256 else None
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
