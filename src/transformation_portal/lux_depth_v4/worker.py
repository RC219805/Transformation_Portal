"""Isolated, governed DA3 native-depth worker for the V4 photography profile.

The process owns one model and accepts only bounded requests from its parent.
No arbitrary module, model, command, or network download is accepted.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v3 import PhotographyPlan, parse_photography_plan
from transformation_portal.ingest.canonical_json import canonicalize_json, dumps_json
from transformation_portal.lux_depth_v4.lifecycle import authorize_model


def device_probe(requested: str) -> str:
    """Probe the actual worker runtime without constructing model tensors."""
    if requested == "cpu":
        return "cpu"
    import torch

    available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not available:
        raise RuntimeError("Explicit MPS execution is unavailable in the governed runtime")
    return "mps" if available else "cpu"


class NativeDepthWorker:
    """Worker lifetime holds materialized authority and one lazily loaded model."""

    def __init__(self, plan: PhotographyPlan) -> None:
        from transformation_portal.depth.backends.da3_runtime_identity import (
            prepare_da3_runtime_identity_with_verification_token,
            runtime_verification_token_sha256,
        )
        from transformation_portal.lux_depth_v3.config import DA3Config, DeviceConfig
        from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine

        self.plan = plan
        payload = plan.to_payload()
        resolved = authorize_model(plan)
        device = device_probe(payload["device"])
        if device != payload["device"]:
            raise RuntimeError("Worker device differs from prepared plan")
        self.engine = DA3InferenceEngine(
            DA3Config(
                model_key=resolved.canonical_key,
                model_revision=resolved.revision,
                resolved_model_contract=resolved,
                device=DeviceConfig(device=device, use_coreml=False),
            ),
            model_key=resolved.canonical_key,
        )
        expected_backend = {"cpu": "pytorch_cpu", "mps": "pytorch_mps"}[device]
        if self.engine.device != device or self.engine.backend.value != expected_backend:
            raise RuntimeError("Worker backend differs from prepared plan")
        if device == "mps":
            import torch

            torch.mps.set_per_process_memory_fraction(0.7)
        self.evidence, token = prepare_da3_runtime_identity_with_verification_token(
            model_canonical_key=resolved.canonical_key,
            model_repo_id=resolved.spec.repo_id,
            model_lock_revision=resolved.revision,
            requested_device=device,
            actual_device=device,
            executed_backend=expected_backend,
        )
        runtime_digest = self.evidence.runtime_identity_sha256
        if not self.evidence.cacheable or token is None or runtime_digest is None:
            reasons = self.evidence.to_mapping().get("incomplete_reasons")
            raise RuntimeError(f"Governed DA3 runtime is incomplete: {reasons}")
        self.token = token
        self.runtime_identity_sha256 = runtime_digest
        self.token_digest = runtime_verification_token_sha256(self.token)
        self.verify()

    def verify(self) -> None:
        from transformation_portal.depth.backends.da3_runtime_identity import verify_runtime_verification_token

        if not verify_runtime_verification_token(
            self.token,
            expected_token_sha256=self.token_digest,
            expected_worker_runtime_identity_sha256=self.runtime_identity_sha256,
            revalidate_worker_import_environment=True,
        ):
            raise RuntimeError("Worker runtime changed after materialization")

    def infer(self, input_path: Path, output_path: Path) -> dict:
        from PIL import Image

        from transformation_portal.depth.backends.da3_worker import _seed_isolated_inference

        self.verify()
        with Image.open(input_path) as image:
            if image.mode != "RGB" or max(image.size) > 2044 or any(size % 14 for size in image.size):
                raise ValueError("Worker input must be a bounded RGB model proxy")
            image.load()
            proxy = image.copy()
        _seed_isolated_inference(proxy)
        self.engine._load_model()
        model = self.engine.model
        if model is None:
            raise RuntimeError("Governed DA3 model did not load")
        # Model initialization consumes RNG only on the first request. Reset at
        # the actual inference seam so cold and warm execution share one seed.
        _seed_isolated_inference(proxy)
        prediction = model.inference([proxy], process_res=max(proxy.size), process_res_method="upper_bound_resize")
        native = np.asarray(prediction.depth[0], dtype=np.float32)
        if native.shape != (proxy.height, proxy.width) or not np.isfinite(native).all():
            raise RuntimeError("DA3 native output does not match the exact proxy grid")
        # The pinned metric model contract does not define confidence semantics.
        # A numerically bounded output is not evidence of calibrated confidence.
        self.verify()
        with output_path.open("xb") as handle:
            np.savez(handle, native_depth=native)
        raw = output_path.read_bytes()
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "runtime_identity_sha256": self.evidence.runtime_identity_sha256,
            "plan_fingerprint_sha256": self.plan.plan_fingerprint_sha256,
            "native_semantics": "da3_metric_uncalibrated",
            "confidence_available": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-device", choices=("cpu", "mps", "auto"))
    args = parser.parse_args(argv)
    if args.probe_device:
        print(dumps_json({"device": device_probe(args.probe_device)}))
        return 0
    worker = None
    for line in iter(lambda: sys.stdin.buffer.readline(8 * 1024 * 1024 + 1), b""):
        try:
            if len(line) > 8 * 1024 * 1024:
                raise ValueError("Worker request exceeds byte limit")
            request = decode_bounded_json_object(line)
            result: dict[str, Any]
            with contextlib.redirect_stdout(sys.stderr):
                command = request.get("command")
                if command == "prepare" and worker is None and set(request) == {"command", "plan"}:
                    worker = NativeDepthWorker(parse_photography_plan(canonicalize_json(request["plan"])))
                    result = {"runtime_evidence": worker.evidence.to_mapping()}
                elif command == "verify" and worker is not None and set(request) == {"command"}:
                    worker.verify()
                    result = {"verified": True}
                elif command == "infer" and worker is not None and set(request) == {"command", "input", "output"}:
                    result = worker.infer(Path(request["input"]), Path(request["output"]))
                else:
                    raise ValueError("Unsupported worker command or lifecycle transition")
            print(dumps_json({"ok": True, "result": result}, allow_nan=False), flush=True)
        except Exception as exc:
            print(dumps_json({"ok": False, "error": str(exc)}), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
