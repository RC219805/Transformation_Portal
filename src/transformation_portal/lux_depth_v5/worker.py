"""Pinned DA3 inference with explicit compute precision and sky substitution evidence.

The V4 worker's governed model lifetime and source verification remain authority.
This version deliberately bypasses the upstream API's automatic autocast wrapper;
its pinned preprocessing, core network, and output conversion are retained.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v3 import PhotographyPlan
from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4, parse_plan_v4
from transformation_portal.ingest.canonical_json import canonicalize_json, dumps_json
from transformation_portal.lux_depth_v4.worker import NativeDepthWorker as GovernedNativeDepthWorker
from transformation_portal.lux_depth_v4.worker import device_probe

from .backend import INFERENCE_RECIPE, SKY_MASK_POLICY

# Changes to this API recipe require re-auditing the pinned implementation before
# advancing this constant. The materialized runtime also verifies the source bytes.
SUPPORTED_DA3_SOURCE_REVISION = "95a2adea1a8180104bf51937409034bdec70a244"


def _validate_api_contract(model: Any) -> None:
    """Reject upstream API drift before bypassing its implicit autocast wrapper."""
    if (type(model).__module__, type(model).__name__) != ("depth_anything_3.api", "DepthAnything3") or (
        type(model.model).__module__,
        type(model.model).__name__,
    ) != ("depth_anything_3.model.da3", "DepthAnything3Net"):
        raise RuntimeError("V5 explicit precision requires the pinned DA3 metric API and network")
    interfaces = (
        (model._preprocess_inputs, ("image", "extrinsics", "intrinsics", "process_res", "process_res_method")),
        (model._prepare_model_inputs, ("imgs_cpu", "extrinsics", "intrinsics")),
        (model._convert_to_prediction, ("raw_output",)),
        (
            model.model.forward,
            ("x", "extrinsics", "intrinsics", "export_feat_layers", "infer_gs", "use_ray_pose", "ref_view_strategy"),
        ),
    )
    if any(tuple(inspect.signature(method).parameters) != expected for method, expected in interfaces):
        raise RuntimeError("Pinned DA3 inference interfaces changed; the explicit precision recipe must be re-audited")


def _predict_native(model: Any, proxy: Any, precision: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Execute the pinned network without entering DepthAnything3.forward/inference.

    fp16 means mixed precision with the upstream float32 depth head; fp32 disables
    autocast for the complete network. Storage is float32 in both cases.
    """
    if precision not in {"fp32", "fp16"}:
        raise ValueError("Unsupported V5 compute precision")
    _validate_api_contract(model)
    import torch

    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("V5 precision recipe requires float32 model weights")
    with torch.inference_mode():
        imgs_cpu, extrinsics, intrinsics = model._preprocess_inputs([proxy], None, None, max(proxy.size), "upper_bound_resize")
        images, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
        if (
            images.shape != (1, 1, 3, proxy.height, proxy.width)
            or images.dtype != torch.float32
            or images.device.type not in {"cpu", "mps"}
            or ex_t is not None
            or in_t is not None
        ):
            raise RuntimeError("Pinned DA3 preprocessing differs from the prepared proxy")
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=precision == "fp16"):
            # The upstream outer forward chooses FP16 even on CPU/MPS. Calling
            # the pinned network directly gives this recipe sole outer authority.
            raw_output = model.model(images, None, None, [], False, False, "saddle_balanced")
        prediction = model._convert_to_prediction(raw_output)
        native = np.asarray(prediction.depth[0], dtype=np.float32)
        if native.shape != (proxy.height, proxy.width) or not np.isfinite(native).all():
            raise RuntimeError("DA3 native output does not match the exact proxy grid")
        raw_sky = raw_output.get("sky")
        sky_mask = None
        if raw_sky is not None:
            if raw_sky.shape != (1, 1, proxy.height, proxy.width) or not bool(torch.isfinite(raw_sky).all()):
                raise RuntimeError("DA3 sky output does not match the exact proxy grid")
            # Mono depth substitutes sky at >=0.3, while Prediction.sky uses
            # >=0.5. Preserve every substituted region, including [0.3, 0.5).
            sky_mask = (raw_sky[0, 0] >= 0.3).cpu().numpy().astype(np.bool_)
    return native, sky_mask


class NativeDepthWorker(GovernedNativeDepthWorker):
    """Add a versioned precision recipe to the existing verified model lifetime."""

    def __init__(self, plan: ExecutionPlanV4) -> None:
        self.precision = plan.to_payload()["configuration"]["depth"]["precision"]
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("Unsupported V5 compute precision")
        # The governed constructor consumes only model/device/fingerprint fields,
        # shared by both immutable plan types. V5 parsing happens at IPC ingress.
        super().__init__(cast(PhotographyPlan, plan))
        if self.evidence.to_mapping()["evidence"]["source_revision"] != SUPPORTED_DA3_SOURCE_REVISION:
            raise RuntimeError("V5 precision recipe does not authorize this DA3 source revision")

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
        _seed_isolated_inference(proxy)
        native, sky_mask = _predict_native(model, proxy, self.precision)
        self.verify()
        with output_path.open("xb") as handle:
            if sky_mask is None:
                np.savez(handle, native_depth=native)
            else:
                np.savez(handle, native_depth=native, sky_mask=sky_mask)
        raw = output_path.read_bytes()
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "runtime_identity_sha256": self.evidence.runtime_identity_sha256,
            "plan_fingerprint_sha256": self.plan.plan_fingerprint_sha256,
            "native_semantics": "da3_metric_uncalibrated",
            "precision": self.precision,
            "inference_recipe": INFERENCE_RECIPE,
            "sky_mask_policy": SKY_MASK_POLICY,
            "sky_available": sky_mask is not None,
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
                    worker = NativeDepthWorker(parse_plan_v4(canonicalize_json(request["plan"])))
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
