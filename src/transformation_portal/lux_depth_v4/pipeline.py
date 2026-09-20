"""Sequential photography graph on the hardened core executor.

This explicit V4 candidate never changes V3 admission or executor selection.
Only the native-depth node is cacheable; all photographic masters remain local.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from transformation_portal.core.cas_dag_executor import AuthoritativeStageIdentity, CASDAGConfig, CASDAGExecutor
from transformation_portal.core.depth_artifact import DepthArtifact
from transformation_portal.core.execution_identity_v4 import MaterializedExecutionIdentityV4
from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2, digest_payload
from transformation_portal.core.image_artifact import ImageProxy, artifact_content_hash
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.stage_graph.graph import StageGraph
from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus
from transformation_portal.storage.cas_store import ArtifactStore

from .backend import DA3Session, require_process_supervisor
from .companions import calibrated_depth, freeze_companions, load_materials
from .io import snapshot, write_evidence
from .lifecycle import PreparedLuxExecutionV4, authorize_model, validate_prepared_bindings
from .photography import (
    apply_materials,
    create_proxy,
    decode_master,
    enhance_master,
    restore_depth_with_validity,
    write_delivery,
)
from .raw import decode_raw
from .runtime import PhotographyRuntime

if TYPE_CHECKING:
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher


@dataclass(frozen=True)
class LuxDepthV4Result:
    """Completed local attempt; managed visibility still requires a dispatch fence."""

    output_root: Path
    evidence_path: Path
    plan_fingerprint_sha256: str
    artifact_paths: tuple[str, ...]
    input_count: int
    depth_cache_hits: int
    depth_cache_misses: int


class _PhotographyStage(Stage):
    def __init__(self, node: dict, operation: Callable):
        super().__init__(node["id"], node["stage"])
        self.node = node
        self.operation = operation

    def get_dependencies(self) -> list[str]:
        return sorted({binding.split(".")[0] for binding in self.node["inputs"].values() if not binding.startswith("$")})

    def get_cache_key(self, _context: StageContext) -> str:
        raise RuntimeError("V4 forbids the legacy stage cache")

    def compute(self, context: StageContext) -> StageResult:
        outputs = self.operation(context)
        if set(outputs) != {f"{self.name}.{name}" for name in self.node["outputs"]}:
            raise RuntimeError("Stage output inventory differs from the canonical plan")
        return StageResult(self.name, self.version, StageStatus.COMPLETED, artifacts=outputs)


def _proxy_hash(proxy: ImageProxy) -> str:
    return artifact_content_hash(
        {"transform": proxy.transform.to_payload(), "master_content_hash": proxy.master_content_hash}, {"pixels": proxy.pixels}
    )


def _depth(context: StageContext, source_sha256: str, companion: Mapping[str, Any] | None = None) -> DepthArtifact:
    proxy = context.artifacts["preprocess.proxy"]
    wire = context.artifacts["depth.depth"]
    if not isinstance(wire, dict) or set(wire) != {"descriptor", "native_depth", "valid_mask", "confidence"}:
        raise ValueError("Depth artifact has an invalid wire representation")
    native = wire["native_depth"]
    confidence = wire["confidence"]
    if native.shape != proxy.transform.padded_shape or native.dtype != np.float32:
        raise ValueError("Cached native depth does not match the prepared proxy")
    artifact = calibrated_depth(native, proxy, companion, source_sha256, confidence=confidence)
    if not np.array_equal(artifact.valid_mask, wire["valid_mask"]):
        raise ValueError("Depth validity differs from the native depth artifact")
    if artifact.to_payload() != wire["descriptor"]:
        raise ValueError("Depth artifact semantics differ from prepared alignment")
    return artifact


def run(
    prepared: PreparedLuxExecutionV4,
    *,
    cancellation: Callable[[], bool] | None = None,
    publisher: GenerationPublisher | None = None,
) -> LuxDepthV4Result:
    """Execute the exact prepared plan, publishing completion only after verification."""
    if type(prepared) is not PreparedLuxExecutionV4:
        raise TypeError("run requires PreparedLuxExecutionV4; V3 plans cannot be downgraded")
    validate_prepared_bindings(prepared)
    plan = ExecutionPlanV2(prepared.canonical_plan_bytes)
    payload = plan.to_payload()
    if "publication" in payload or publisher is not None:
        from .publication import validate_publication_plan

        if publisher is None:
            raise ValueError("Managed execution requires its publisher before backend initialization")
        validate_publication_plan(payload, publisher.limits)
    authorize_model(plan)
    resources = payload["resources"]
    configuration = payload["configuration"]
    started = time.monotonic()

    def rebind_companion_manifest(*, check_semantics: bool = False) -> None:
        if prepared.companion_root is None:
            return
        expected = payload["companions_manifest"]
        _, observed = snapshot(
            prepared.companion_root,
            prepared.companion_root / expected["path"],
            maximum_bytes=min(resources["max_input_bytes"], 1024 * 1024),
            retain_bytes=False,
        )
        if observed != expected:
            raise RuntimeError("Companion manifest changed after preparation")
        if check_semantics:
            records, _, receipt = freeze_companions(
                prepared.companion_root / expected["path"],
                payload["inputs"],
                max_input_bytes=resources["max_input_bytes"],
                max_pixels=resources["max_pixels"],
            )
            planned = {item["path"]: item["companions"] for item in payload["inputs"] if "companions" in item}
            if receipt != expected or records != planned:
                raise RuntimeError("Companion manifest semantics differ from the prepared plan")

    # Rebind every input before model initialization or creating any output.
    rebind_companion_manifest(check_semantics=True)
    for item in payload["inputs"]:
        _, record = snapshot(
            prepared.input_root,
            prepared.input_root / item["path"],
            maximum_bytes=resources["max_input_bytes"],
            retain_bytes=False,
        )
        if record != {key: item[key] for key in ("path", "sha256", "size_bytes")}:
            raise RuntimeError("Input bytes changed after preparation")
    root = prepared.output_root
    artifacts: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    hits = misses = written = 0

    def check() -> None:
        if publisher is not None and publisher.limits.to_payload() != payload["publication"]:
            raise RuntimeError("Publisher limits changed during managed execution")
        if cancellation is not None and cancellation():
            raise RuntimeError("Photographic execution cancelled")
        if time.monotonic() - started >= resources["wall_time_seconds"]:
            raise TimeoutError("Photographic execution exceeded its wall-time budget")
        psutil = require_process_supervisor()

        if psutil.Process().memory_info().rss > resources["memory_mib"] * 1024 * 1024:
            raise RuntimeError("Parent exceeded its observed memory budget")

    def observe(path: Path, kind: str, input_id: str | None = None) -> None:
        nonlocal written
        check()
        maximum_bytes = resources["max_output_bytes"] - written
        if publisher is not None:
            maximum_bytes = min(maximum_bytes, payload["publication"]["max_file_bytes"])
        _, record = snapshot(root, path, maximum_bytes=maximum_bytes, retain_bytes=False)
        written += record["size_bytes"]
        artifacts.append({**record, "kind": kind, "input_id": input_id})

    check()

    # Runtime verification precedes output creation; no network model downloads.
    def poll_cancellation() -> bool:
        check()
        return False

    with DA3Session(prepared.runtime_python, plan, cancellation=poll_cancellation) as session:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="tp-lux-v4-cache-") as temporary:
            try:
                write_evidence(root, "execution-plan.json", prepared.canonical_plan_bytes)
                observe(root / "execution-plan.json", "plan")
                cache = (prepared.cache_root or Path(temporary)) / "identity-v4"
                executor = CASDAGExecutor(
                    ArtifactStore(cache / "cas"),
                    cache / "results",
                    CASDAGConfig(
                        enable_caching=prepared.cache_root is not None,
                        enable_provenance=False,
                        lock_timeout=min(1.0, float(resources["wall_time_seconds"])),
                        code_paths=[str(Path(__file__).parent)],
                        parallel=False,
                    ),
                )
                # Establish the admitted output/cache namespaces before freezing
                # import directories; our own directory creation is not drift.
                parent_runtime = PhotographyRuntime()
                combined_runtime = digest_payload(
                    {"parent": parent_runtime.sha256, "worker": session.runtime.runtime_identity_sha256}
                )
                model_digest = digest_payload(session.runtime.to_mapping()["backend_identity"])
                for item in payload["inputs"]:
                    check()
                    data, record = snapshot(
                        prepared.input_root, prepared.input_root / item["path"], maximum_bytes=resources["max_input_bytes"]
                    )
                    if record["sha256"] != item["sha256"] or record["size_bytes"] != item["size_bytes"]:
                        raise RuntimeError("Input changed during execution")
                    current_id = item["id"]
                    rebind_companion_manifest()
                    companion = item.get("companions")
                    masks: dict[str, np.ndarray] = {}
                    confidences: dict[str, float] = {}
                    if companion is not None:
                        assert prepared.companion_root is not None
                        masks, confidences = load_materials(
                            prepared.companion_root,
                            companion,
                            max_input_bytes=resources["max_input_bytes"],
                            max_pixels=resources["max_pixels"],
                        )

                    def preprocess(_context: StageContext) -> dict[str, Any]:
                        master = decode_master(
                            data,
                            source_name=item["path"],
                            input_color=configuration["input_color"],
                            max_pixels=resources["max_pixels"],
                            raw_decoder=lambda content, name: decode_raw(
                                content,
                                name,
                                python=prepared.raw_python or "",
                                resources=resources,
                                cancellation=poll_cancellation,
                            ),
                        )
                        if companion is not None and "calibration" in companion:
                            calibration = companion["calibration"]
                            if master.shape != (calibration["height"], calibration["width"]):
                                raise ValueError("Calibration does not match orientation-normalized master geometry")
                        if any(mask.shape != master.shape for mask in masks.values()):
                            raise ValueError("Material masks do not match orientation-normalized master geometry")
                        return {
                            "preprocess.master": master,
                            "preprocess.proxy": create_proxy(master, configuration["target_size"]),
                        }

                    def depth(context: StageContext) -> dict[str, Any]:
                        proxy = context.artifacts["preprocess.proxy"]
                        arrays, response = session.compute(proxy.pixels)
                        if response["native_semantics"] != "da3_metric_uncalibrated":
                            raise RuntimeError("Unexpected native depth semantics")
                        native = arrays["native_depth"]
                        confidence = arrays.get("confidence")
                        artifact = calibrated_depth(native, proxy, companion, item["sha256"], confidence=confidence)
                        return {
                            "depth.depth": {
                                "descriptor": artifact.to_payload(),
                                "native_depth": artifact.native_depth,
                                "valid_mask": artifact.valid_mask,
                                "confidence": artifact.confidence,
                            }
                        }

                    def enhance(context: StageContext) -> dict[str, Any]:
                        artifact = _depth(context, item["sha256"], companion)
                        aligned, aligned_valid = restore_depth_with_validity(
                            artifact.relative_depth(), artifact.valid_mask, context.artifacts["preprocess.proxy"].transform
                        )
                        master = enhance_master(
                            context.artifacts["preprocess.master"],
                            aligned,
                            strength=configuration["strength"],
                            clarity=configuration["clarity"],
                            valid_mask=aligned_valid,
                        )
                        outputs: dict[str, Any] = {"enhance.master": master}
                        if "materials" in payload["nodes"][2]["inputs"]:
                            master, materials_report = apply_materials(master, masks, confidences)
                            outputs = {"enhance.master": master, "enhance.materials": materials_report}
                        return outputs

                    def output(context: StageContext) -> dict[str, Any]:
                        master = context.artifacts["enhance.master"]
                        original = context.artifacts["preprocess.master"]
                        artifact = _depth(context, item["sha256"], companion)
                        transform = context.artifacts["preprocess.proxy"].transform
                        aligned, aligned_valid = restore_depth_with_validity(
                            artifact.relative_depth(), artifact.valid_mask, transform
                        )
                        destination = root / current_id
                        destination.mkdir(mode=0o700)
                        arrays = {
                            "source-master.npy": original.pixels,
                            "master.npy": master.pixels,
                            "native-depth.npy": artifact.native_depth,
                            "depth-valid.npy": artifact.valid_mask,
                            "relative-depth.npy": aligned,
                            "aligned-depth-valid.npy": aligned_valid,
                        }
                        if artifact.metric_map_m is not None:
                            arrays["metric-depth-m.npy"] = artifact.metric_map_m
                            arrays["aligned-metric-depth-m.npy"], _ = restore_depth_with_validity(
                                artifact.metric_map_m, artifact.valid_mask, transform
                            )
                        if master.alpha is not None:
                            arrays["alpha.npy"] = master.alpha
                        if artifact.confidence is not None:
                            arrays["depth-confidence.npy"] = artifact.confidence
                        estimate = (
                            sum(array.nbytes + 256 for array in arrays.values())
                            + master.pixels.shape[0] * master.pixels.shape[1] * 8
                            + 1024 * 1024
                        )
                        if estimate + written > resources["max_output_bytes"] or estimate > shutil.disk_usage(root).free:
                            raise RuntimeError("Photographic output exceeds the disk budget")
                        if configuration["preview_maps"]:
                            from .photography import generate_preview_maps

                            maps, preview_report = generate_preview_maps(artifact)
                            arrays.update({f"preview-{name}.npy": value for name, value in maps.items()})
                            estimate += sum(value.nbytes + 256 for value in maps.values())
                            if estimate + written > resources["max_output_bytes"] or estimate > shutil.disk_usage(root).free:
                                raise RuntimeError("Preview maps exceed the disk budget")
                        else:
                            preview_report = {"status": "disabled", "classification": "depth_derived_preview"}
                        for name, array in arrays.items():
                            check()
                            with (destination / name).open("xb") as handle:
                                np.save(handle, array, allow_pickle=False)
                                handle.flush()
                                os.fsync(handle.fileno())
                            observe(destination / name, "array", current_id)
                        delivery = write_delivery(master, destination / "delivery.tif")
                        delivery["path"] = f"{current_id}/delivery.tif"
                        observe(destination / "delivery.tif", "image", current_id)
                        descriptor = {
                            "schema": "tp.lux.photograph.v1",
                            "input_id": current_id,
                            "source": original.to_payload(),
                            "master": master.to_payload(),
                            "depth": artifact.to_payload(),
                            "depth_content_sha256": artifact.content_hash(),
                            "aligned_depth": {
                                "shape": list(master.shape),
                                "validity_path": f"{current_id}/aligned-depth-valid.npy",
                                "relative_path": f"{current_id}/relative-depth.npy",
                                "metric_path": (
                                    f"{current_id}/aligned-metric-depth-m.npy" if artifact.metric_map_m is not None else None
                                ),
                                "invalid_value": 0,
                                "interpolation": "valid_weighted_bilinear_nearest_validity",
                            },
                            "delivery": delivery,
                            "materials": context.artifacts.get(
                                "enhance.materials", {"status": "abstained", "reason": "no_authoritative_masks"}
                            ),
                            "preview_maps": preview_report,
                        }
                        write_evidence(root, f"{current_id}/photograph.json", canonicalize_json(descriptor))
                        observe(destination / "photograph.json", "descriptor", current_id)
                        return {
                            "output.delivery": delivery,
                            "output.master": f"{current_id}/master.npy",
                            "output.evidence": f"{current_id}/photograph.json",
                        }

                    operations = {"preprocess": preprocess, "depth": depth, "enhance": enhance, "output": output}
                    graph = StageGraph()
                    for node in payload["nodes"]:
                        graph.add_stage(_PhotographyStage(node, operations[node["id"]]))

                    def identity(
                        stage: Stage, context: StageContext, _upstream: Mapping[str, Any]
                    ) -> AuthoritativeStageIdentity:
                        bound = {}
                        node = next(node for node in payload["nodes"] if node["id"] == stage.name)
                        for name, binding in node["inputs"].items():
                            if binding == "$input":
                                digest = item["sha256"]
                            elif binding in {"$calibration", "$materials"}:
                                digest = digest_payload((companion or {}).get(binding[1:]))
                            elif binding == "enhance.materials":
                                digest = digest_payload(context.artifacts[binding])
                            elif binding == "depth.depth":
                                digest = _depth(context, item["sha256"], companion).content_hash()
                            else:
                                value = context.artifacts[binding]
                                digest = _proxy_hash(value) if isinstance(value, ImageProxy) else value.content_hash()
                            bound[name] = digest
                        materialized = MaterializedExecutionIdentityV4.from_plan(
                            plan,
                            node_id=stage.name,
                            input_id=current_id,
                            inputs=bound,
                            source_identity_sha256=parent_runtime.source_sha256,
                            runtime_identity_sha256=combined_runtime,
                            model_identity_sha256=model_digest if stage.name == "depth" else None,
                        )
                        return AuthoritativeStageIdentity(
                            stage.name, stage.version, materialized.execution_identity_sha256, materialized.canonical_bytes
                        )

                    def checkpoint(stage_name: str, phase: str, _context: StageContext) -> None:
                        check()
                        session.checkpoint()
                        if phase in {"before_cache", "before_store", "before_compute", "before_propagate"}:
                            parent_runtime.verify()
                            session.verify()

                    execution = executor.execute(
                        graph,
                        StageContext(device=payload["device"]),
                        identity_provider=identity,
                        checkpoint=checkpoint,
                        cache_policy=lambda stage, _context: stage.name == "depth",
                    )
                    if not execution.success:
                        raise RuntimeError(execution.error)
                    hit = execution.stage_results["depth"].cache_hit
                    hits += int(hit)
                    misses += int(not hit)
                    image_records.append(
                        {
                            "input_id": current_id,
                            "depth_cache_hit": hit,
                            "identities": {
                                name: value.cas_id.removeprefix("sha256:") for name, value in execution.identities.items()
                            },
                        }
                    )
                check()
                parent_runtime.verify()
                session.verify()
                for artifact in artifacts:
                    check()
                    _, actual = snapshot(
                        root, root / artifact["path"], maximum_bytes=artifact["size_bytes"], retain_bytes=False
                    )
                    if any(actual[key] != artifact[key] for key in actual):
                        raise RuntimeError("Output changed before completion publication")
                evidence = {
                    "schema": "tp.lux.execution.evidence.v2",
                    "complete": True,
                    "synthetic": False,
                    "plan_schema": plan.schema,
                    "plan_fingerprint_sha256": plan.plan_fingerprint_sha256,
                    "parent_runtime_sha256": parent_runtime.sha256,
                    "worker_runtime": session.runtime.to_mapping(),
                    "inputs": payload["inputs"],
                    "artifacts": artifacts,
                    "executions": image_records,
                    "duration_seconds": time.monotonic() - started,
                    "cache": {
                        "namespace": "identity-v4",
                        "enabled": prepared.cache_root is not None,
                        "hits": hits,
                        "misses": misses,
                    },
                    "production_acceptance": "pending",
                }
                completion_bytes = canonicalize_json(evidence)
                if written + len(completion_bytes) > resources["max_output_bytes"]:
                    raise RuntimeError("Completion evidence exceeds the output byte budget")
                if publisher is not None and len(completion_bytes) > payload["publication"]["max_file_bytes"]:
                    raise RuntimeError("Completion evidence exceeds the publisher file-byte limit")
                check()
                parent_runtime.verify()
                session.verify()
                write_evidence(root, "execution-evidence.json", completion_bytes)
            except BaseException as exc:
                write_evidence(
                    root,
                    "failure.json",
                    canonicalize_json(
                        {
                            "schema": "tp.lux.execution.failure.v2",
                            "complete": False,
                            "plan_fingerprint_sha256": plan.plan_fingerprint_sha256,
                            "error": str(exc),
                        }
                    ),
                )
                raise
    return LuxDepthV4Result(
        root,
        root / "execution-evidence.json",
        plan.plan_fingerprint_sha256,
        tuple(record["path"] for record in artifacts) + ("execution-evidence.json",),
        len(image_records),
        hits,
        misses,
    )
