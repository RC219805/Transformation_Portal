"""Opt-in native Depth Pro execution and model-free photographic replay for V6.

This independent contract never impersonates a retained DA3/V5 generation.
Replay establishes consistency of retained inference, not measured scene accuracy.
"""

from __future__ import annotations

import importlib
import io
import math
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import numpy as np

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import require_digest
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import (
    _pin_output_root,
    _secure_atomic_write_bytes,
    _validate_pinned_root_namespace,
)
from transformation_portal.lux_depth_v4.backend import require_process_supervisor
from transformation_portal.lux_depth_v4.evidence import _inventory
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot
from transformation_portal.lux_depth_v4.photography import decode_master
from transformation_portal.lux_depth_v5.evidence import _array

from . import depth_pro_backend as backend
from .color import GradeRecipe, RenderRecipe
from .depth_pro_products import model_input_bytes, native_image_products
from .evidence import VerifiedGradeEvidence
from .pipeline import LuxDepthV6Result
from .plan import MAX_PLAN_BYTES, OutputLimits, digest
from .source import SourceLimits

PLAN_SCHEMA = "tp.lux.depth_pro.plan.v1"
EVIDENCE_SCHEMA = "tp.lux.depth_pro.evidence.v1"
_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
_MODULES = (
    "transformation_portal.lux_depth_v6.depth_pro",
    "transformation_portal.lux_depth_v6.depth_pro_backend",
    "transformation_portal.lux_depth_v6.depth_pro_products",
    "transformation_portal.lux_depth_v6.color",
    "transformation_portal.lux_depth_v6.products",
    "transformation_portal.core.image_artifact",
    "transformation_portal.lux_depth_v4.photography",
    "transformation_portal.lux_depth_v4.io",
    "transformation_portal.lux_depth_v5.preview",
)
_DEPENDENCIES = ("numpy", "Pillow", "tifffile", "imagecodecs")


def processing_identity() -> dict[str, Any]:
    """Bind the exact processing sources and encoding environment used by replay."""
    return {
        "modules": {name: digest(Path(importlib.import_module(name).__file__).read_bytes()) for name in _MODULES},
        "dependencies": {name: version(name) for name in _DEPENDENCIES},
    }


def _resources(payload: dict[str, Any], limits: SourceLimits) -> None:
    total = sum(item["size_bytes"] for item in payload["inputs"])
    if total > limits.max_input_bytes:
        raise ValueError("Depth Pro sources exceed the admitted input byte budget")
    reserve = 2 * MAX_PLAN_BYTES
    for item in payload["inputs"]:
        pixels = math.prod(item["shape"])
        # Full-resolution decode, float grading, model staging and encoder copies.
        # Runtime supervision independently enforces the actual process budget.
        if pixels > limits.max_pixels or pixels * 512 + item["size_bytes"] + 8 * 1024**3 > limits.memory_mib * 1024**2:
            raise ValueError("Depth Pro geometry exceeds pixel/working-memory admission")
        reserve += pixels * 80 + item["size_bytes"] + 32 * 1024**2
    if reserve > payload["limits"]["max_output_bytes"]:
        raise ValueError("Depth Pro products exceed output byte admission")


@dataclass(frozen=True)
class NativeDepthProPlan:
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes or len(self.canonical_bytes) > MAX_PLAN_BYTES:
            raise ValueError("Depth Pro plan requires bounded canonical bytes")
        payload = self.to_payload()
        keys = {
            "schema",
            "pipeline",
            "backend",
            "worker_authority",
            "runtime",
            "inputs",
            "input_color",
            "grade",
            "render",
            "source_limits",
            "limits",
            "processing",
            "production_acceptance",
        }
        if (
            set(payload) != keys
            or payload["schema"] != PLAN_SCHEMA
            or payload["pipeline"] != "lux_depth_v6"
            or payload["backend"] != "depth_pro"
            or payload["production_acceptance"] != "not_established"
            or canonicalize_json(payload) != self.canonical_bytes
        ):
            raise ValueError("Invalid closed canonical native V6 Depth Pro plan")
        if payload["input_color"] not in {"auto", "srgb", "linear_srgb"}:
            raise ValueError("Invalid Depth Pro input color")
        GradeRecipe.from_payload(payload["grade"])
        RenderRecipe.from_payload(payload["render"])
        if (
            type(payload["source_limits"]) is not dict
            or set(payload["source_limits"]) != {"max_input_bytes", "max_pixels", "memory_mib"}
            or type(payload["limits"]) is not dict
            or set(payload["limits"]) != {"max_output_bytes", "wall_time_seconds"}
        ):
            raise ValueError("Depth Pro requires explicit closed resource limits")
        source_limits = SourceLimits(**payload["source_limits"])
        OutputLimits(**payload["limits"])
        authority = backend.validate_authority(canonicalize_json(payload["worker_authority"]))
        images = payload["inputs"]
        if not isinstance(images, list) or not 1 <= len(images) <= 1024:
            raise ValueError("Depth Pro plan requires a bounded nonempty input selection")
        paths = []
        for index, item in enumerate(images):
            if not isinstance(item, dict) or set(item) != {
                "input_id",
                "path",
                "size_bytes",
                "sha256",
                "shape",
                "master_sha256",
                "model_input_sha256",
            }:
                raise ValueError("Invalid Depth Pro input binding")
            path = item["path"]
            if (
                type(path) is not str
                or Path(path).name != path
                or Path(path).suffix.lower() not in _SUFFIXES
                or item["input_id"] != f"input-{index:04d}"
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] <= 0
            ):
                raise ValueError("Invalid Depth Pro original photograph path")
            if (
                type(item["shape"]) is not list
                or len(item["shape"]) != 2
                or any(type(size) is not int or size <= 0 for size in item["shape"])
            ):
                raise ValueError("Invalid Depth Pro photograph geometry")
            for key in ("sha256", "master_sha256", "model_input_sha256"):
                require_digest(item[key])
            paths.append(path)
        if paths != sorted(set(paths)) or paths != [item.path for item in authority.inputs]:
            raise ValueError("Depth Pro worker authority differs from photographic input selection")
        processing = payload["processing"]
        if (
            not isinstance(processing, dict)
            or set(processing) != {"modules", "dependencies"}
            or set(processing["modules"]) != set(_MODULES)
            or set(processing["dependencies"]) != set(_DEPENDENCIES)
        ):
            raise ValueError("Invalid Depth Pro processing identity")
        for value in processing["modules"].values():
            require_digest(value)
        if any(type(value) is not str or not value for value in processing["dependencies"].values()):
            raise ValueError("Invalid Depth Pro processing dependency")
        runtime = payload["runtime"]
        backend.validate_runtime_identity(runtime)
        config = backend.runtime_config_from_execution_plan(authority)
        model = backend.backend_candidate_authority(authority, "depth_pro").model_contract
        if (
            runtime["python_executable"] != config.depth_pro_python_executable
            or runtime["checkpoint"]["path"] != model.artifact_path
            or runtime["checkpoint"]["sha256"] != model.artifact_sha256
        ):
            raise ValueError("Depth Pro runtime differs from canonical worker authority")
        _resources(payload, source_limits)

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        return digest(self.canonical_bytes)


@dataclass(frozen=True)
class NativeDepthProRequest:
    input_dir: Path
    output_dir: Path
    python_executable: Path
    checkpoint: Path
    device: str = "cpu"
    non_commercial_ok: bool = False
    accept_license: bool = False
    input_color: str = "auto"
    grade: GradeRecipe = field(default_factory=GradeRecipe)
    render: RenderRecipe = field(default_factory=RenderRecipe)
    source_limits: SourceLimits = field(default_factory=SourceLimits)
    output_limits: OutputLimits = field(default_factory=OutputLimits)


@dataclass(frozen=True)
class PreparedNativeDepthPro:
    plan: NativeDepthProPlan
    input_root: Path
    output_root: Path
    python_executable: Path
    checkpoint: Path

    @property
    def canonical_plan_bytes(self) -> bytes:
        return self.plan.canonical_bytes


def _selection(root: Path) -> list[Path]:
    selected = []
    for count, path in enumerate(root.iterdir(), start=1):
        if count > 2048:
            raise ValueError("Depth Pro input directory exceeds entry limit")
        if path.suffix.lower() in _SUFFIXES:
            selected.append(path)
    if not 1 <= len(selected) <= 1024:
        raise ValueError("Depth Pro requires 1..1024 top-level JPEG, PNG, or TIFF originals")
    return sorted(selected)


def _binding(root: Path, path: Path, input_id: str, input_color: str, limits: SourceLimits) -> dict[str, Any]:
    raw, record = snapshot(root, path, maximum_bytes=min(limits.max_input_bytes, limits.memory_mib * 1024**2 // 8))
    # Restrict decode before allocating pixels, using the same conservative
    # per-pixel reserve enforced on the frozen plan.
    available = limits.memory_mib * 1024**2 - 8 * 1024**3 - len(raw)
    admitted_pixels = min(limits.max_pixels, max(0, available // 512))
    if admitted_pixels < 1:
        raise ValueError("Depth Pro working-memory budget is insufficient")
    master = decode_master(raw, source_name=path.name, input_color=input_color, max_pixels=admitted_pixels)
    return {
        **record,
        "input_id": input_id,
        "shape": list(master.shape[:2]),
        "master_sha256": master.content_hash(),
        "model_input_sha256": digest(model_input_bytes(master)),
    }


def prepare(request: NativeDepthProRequest) -> PreparedNativeDepthPro:
    if type(request) is not NativeDepthProRequest:
        raise TypeError("Depth Pro requires an exact native request")
    if request.non_commercial_ok is not True or request.accept_license is not True:
        raise ValueError("Depth Pro requires non-commercial and Apple research-license acknowledgements")
    root = directory_path(request.input_dir)
    output = directory_path(request.output_dir, allow_missing=True)
    if output.exists() or not output.parent.is_dir() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Depth Pro output must be new with an existing parent and disjoint from input")
    with pinned_directory(root):
        selected = _selection(root)
        inputs = []
        remaining = request.source_limits.max_input_bytes
        for index, path in enumerate(selected):
            if remaining <= 0:
                raise ValueError("Depth Pro sources exceed the admitted input byte budget")
            item = _binding(
                root,
                path,
                f"input-{index:04d}",
                request.input_color,
                replace(request.source_limits, max_input_bytes=remaining),
            )
            inputs.append(item)
            remaining -= item["size_bytes"]
        authority = backend.prepare_authority(
            root,
            selected,
            python_executable=request.python_executable,
            checkpoint=request.checkpoint,
            device=request.device,
            non_commercial_ok=request.non_commercial_ok,
            accept_license=request.accept_license,
        )
        runtime = backend.runtime_identity(
            request.python_executable, request.checkpoint, memory_mib=request.source_limits.memory_mib
        )
    plan = NativeDepthProPlan(
        canonicalize_json(
            {
                "schema": PLAN_SCHEMA,
                "pipeline": "lux_depth_v6",
                "backend": "depth_pro",
                "worker_authority": decode_bounded_json_object(authority),
                "runtime": runtime,
                "inputs": inputs,
                "input_color": request.input_color,
                "grade": request.grade.to_payload(),
                "render": request.render.to_payload(),
                "source_limits": request.source_limits.to_payload(),
                "limits": request.output_limits.to_payload(),
                "processing": processing_identity(),
                "production_acceptance": "not_established",
            }
        )
    )
    return PreparedNativeDepthPro(plan, root, output, request.python_executable, request.checkpoint)


def _guard(payload: dict[str, Any], cancellation: Callable[[], bool] | None) -> Callable[[], None]:
    started = time.monotonic()
    process = require_process_supervisor().Process()

    def check() -> None:
        if cancellation is not None and cancellation():
            raise RuntimeError("Depth Pro execution cancelled")
        if time.monotonic() - started > payload["limits"]["wall_time_seconds"]:
            raise RuntimeError("Depth Pro execution exceeded its wall-time budget")
        if process.memory_info().rss > payload["source_limits"]["memory_mib"] * 1024**2:
            raise RuntimeError("Depth Pro execution exceeded its memory budget")

    return check


def _validate_sources(
    root: Path, payload: dict[str, Any], check: Callable[[], None], limits: SourceLimits | None = None
) -> None:
    limits = limits or SourceLimits(**payload["source_limits"])
    with pinned_directory(root):
        if [path.name for path in _selection(root)] != [item["path"] for item in payload["inputs"]]:
            raise ValueError("Depth Pro input selection changed")
        for item in payload["inputs"]:
            check()
            _, record = snapshot(root, root / item["path"], maximum_bytes=item["size_bytes"], retain_bytes=False)
            if record != {key: item[key] for key in ("path", "size_bytes", "sha256")}:
                raise ValueError("Depth Pro source bytes changed")
            bound_limits = replace(limits, max_input_bytes=min(limits.max_input_bytes, item["size_bytes"]))
            if _binding(root, root / item["path"], item["input_id"], payload["input_color"], bound_limits) != item:
                raise ValueError("Depth Pro source bytes or decoded master changed")
            check()


def _npy(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def run(prepared: PreparedNativeDepthPro, *, cancellation: Callable[[], bool] | None = None) -> LuxDepthV6Result:
    if type(prepared) is not PreparedNativeDepthPro or type(prepared.plan) is not NativeDepthProPlan:
        raise TypeError("Depth Pro execution requires an exact prepared carrier")
    plan = NativeDepthProPlan(prepared.canonical_plan_bytes)
    payload = plan.to_payload()
    check = _guard(payload, cancellation)
    check()
    authority = canonicalize_json(payload["worker_authority"])
    if backend.validate_authority(authority).input_root != str(prepared.input_root):
        raise ValueError("Depth Pro prepared input root differs from worker authority")
    if (
        str(prepared.python_executable.expanduser().absolute()) != payload["runtime"]["python_executable"]
        or str(prepared.checkpoint.expanduser().resolve(strict=True)) != payload["runtime"]["checkpoint"]["path"]
    ):
        raise ValueError("Depth Pro prepared runtime paths differ from worker authority")
    if payload["processing"] != processing_identity():
        raise ValueError("Depth Pro processing identity changed")
    _validate_sources(prepared.input_root, payload, check)
    if payload["runtime"] != backend.runtime_identity(
        prepared.python_executable,
        prepared.checkpoint,
        checkpoint_callback=check,
        memory_mib=payload["source_limits"]["memory_mib"],
    ):
        raise ValueError("Depth Pro runtime identity changed")
    root = directory_path(prepared.output_root, allow_missing=True)
    if (
        root != prepared.output_root
        or root.exists()
        or not root.parent.is_dir()
        or root.is_relative_to(prepared.input_root)
        or prepared.input_root.is_relative_to(root)
    ):
        raise ValueError("Depth Pro output must remain new and disjoint")
    with _pin_output_root(root.parent) as parent:
        check()
        os.mkdir(root.name, mode=0o700, dir_fd=parent.descriptor)
        _validate_pinned_root_namespace(parent)
        with _pin_output_root(root) as pinned:
            records = []
            written = 0

            def write(relative: str, data: bytes, *, record: bool = True) -> None:
                nonlocal written
                check()
                _validate_pinned_root_namespace(pinned)
                if written + len(data) > payload["limits"]["max_output_bytes"] or len(data) > shutil.disk_usage(root).free:
                    raise RuntimeError("Depth Pro product exceeds disk/output budget")
                _secure_atomic_write_bytes(pinned, relative, data, maximum_bytes=max(1, len(data)))
                written += len(data)
                if record:
                    records.append({"path": relative, "size_bytes": len(data), "sha256": digest(data)})

            write("plan.json", plan.canonical_bytes)
            for item in payload["inputs"]:
                check()
                input_id = item["input_id"]
                os.mkdir(input_id, mode=0o700, dir_fd=pinned.descriptor)
                raw, observed = snapshot(
                    prepared.input_root, prepared.input_root / item["path"], maximum_bytes=item["size_bytes"]
                )
                if observed != {key: item[key] for key in ("path", "size_bytes", "sha256")}:
                    raise ValueError("Depth Pro original changed before inference")
                master = decode_master(
                    raw,
                    source_name=item["path"],
                    input_color=payload["input_color"],
                    max_pixels=payload["source_limits"]["max_pixels"],
                )
                model_input = model_input_bytes(master)
                if master.content_hash() != item["master_sha256"] or digest(model_input) != item["model_input_sha256"]:
                    raise ValueError("Depth Pro model input changed")
                del master
                native, metadata = backend.infer(
                    authority,
                    model_input,
                    tuple(item["shape"]),
                    memory_mib=payload["source_limits"]["memory_mib"],
                    checkpoint=check,
                )
                del model_input
                backend.validate_worker_metadata(metadata, authority, tuple(item["shape"]))
                write(f"{input_id}/source.bin", raw)
                write(f"{input_id}/native-depth.npy", _npy(native))
                write(f"{input_id}/worker.json", canonicalize_json(metadata))
                for relative, data in native_image_products(
                    raw,
                    item["path"],
                    payload["input_color"],
                    native,
                    metadata,
                    input_id,
                    GradeRecipe.from_payload(payload["grade"]),
                    RenderRecipe.from_payload(payload["render"]),
                    max_pixels=payload["source_limits"]["max_pixels"],
                    checkpoint=check,
                ):
                    write(relative, data)
                del raw, native, metadata, data
            records.sort(key=lambda record: record["path"])
            if payload["runtime"] != backend.runtime_identity(
                prepared.python_executable,
                prepared.checkpoint,
                checkpoint_callback=check,
                memory_mib=payload["source_limits"]["memory_mib"],
            ):
                raise ValueError("Depth Pro runtime changed during execution")
            _verify_artifacts(root, plan, prepared.input_root, records, check, completed=False)
            evidence = canonicalize_json(
                {
                    "schema": EVIDENCE_SCHEMA,
                    "plan_sha256": plan.sha256,
                    "artifacts": records,
                    "input_count": len(payload["inputs"]),
                    "production_acceptance": "not_established",
                }
            )
            write("evidence.json", evidence, record=False)
            _validate_pinned_root_namespace(pinned)
    return LuxDepthV6Result(root, root / "evidence.json", plan.sha256, len(payload["inputs"]), evidence)


def _verify_artifacts(
    root: Path,
    plan: NativeDepthProPlan,
    source_root: Path,
    records: list[dict],
    check: Callable[[], None],
    *,
    completed: bool,
    source_limits: SourceLimits | None = None,
) -> None:
    payload = plan.to_payload()
    if payload["processing"] != processing_identity():
        raise ValueError("Depth Pro processing identity differs from the frozen plan")
    if type(records) is not list or len(records) > 20 * len(payload["inputs"]) + 1:
        raise ValueError("Depth Pro artifact inventory exceeds bound")
    declared = {}
    for record in records:
        if (
            type(record) is not dict
            or set(record) != {"path", "size_bytes", "sha256"}
            or type(record["path"]) is not str
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or record["path"] in declared
        ):
            raise ValueError("Invalid Depth Pro artifact inventory")
        require_digest(record["sha256"])
        declared[record["path"]] = record
    if sum(record["size_bytes"] for record in records) > payload["limits"]["max_output_bytes"]:
        raise ValueError("Depth Pro inventory exceeds output byte budget")
    seen = set()

    def product(relative: str, expected: bytes) -> None:
        check()
        record = {"path": relative, "size_bytes": len(expected), "sha256": digest(expected)}
        if declared.get(relative) != record:
            raise ValueError(f"Depth Pro product differs from semantic replay: {relative}")
        _, observed = snapshot(root, root / relative, maximum_bytes=len(expected), retain_bytes=False)
        if observed != record:
            raise ValueError(f"Depth Pro output bytes changed: {relative}")
        seen.add(relative)

    product("plan.json", plan.canonical_bytes)
    _validate_sources(source_root, payload, check, source_limits)
    authority = canonicalize_json(payload["worker_authority"])
    for item in payload["inputs"]:
        check()
        input_id = item["input_id"]
        raw, observed = snapshot(root, root / input_id / "source.bin", maximum_bytes=item["size_bytes"])
        if observed["sha256"] != item["sha256"] or observed["size_bytes"] != item["size_bytes"]:
            raise ValueError("Retained Depth Pro original differs from admitted source")
        product(f"{input_id}/source.bin", raw)
        relative = f"{input_id}/native-depth.npy"
        record = declared.get(relative)
        if record is None:
            raise ValueError("Missing retained native depth")
        native = _array(
            root,
            relative,
            tuple(item["shape"]),
            np.dtype("float32"),
            {relative: {**record, "kind": "array"}},
            allow_nonfinite=True,
        )
        product(relative, _npy(native))
        metadata_raw, _ = snapshot(root, root / input_id / "worker.json", maximum_bytes=1024 * 1024)
        metadata = decode_bounded_json_object(metadata_raw)
        backend.validate_worker_metadata(metadata, authority, tuple(item["shape"]))
        if canonicalize_json(metadata) != metadata_raw:
            raise ValueError("Noncanonical retained Depth Pro worker metadata")
        product(f"{input_id}/worker.json", metadata_raw)
        for relative, data in native_image_products(
            raw,
            item["path"],
            payload["input_color"],
            native,
            metadata,
            input_id,
            GradeRecipe.from_payload(payload["grade"]),
            RenderRecipe.from_payload(payload["render"]),
            max_pixels=payload["source_limits"]["max_pixels"],
            checkpoint=check,
        ):
            product(relative, data)
    if set(declared) != seen:
        raise ValueError("Depth Pro inventory contains unexpected artifacts")
    _validate_sources(source_root, payload, check, source_limits)
    if payload["processing"] != processing_identity():
        raise ValueError("Depth Pro processing changed during replay")
    for relative in sorted(seen):
        check()
        _, observed = snapshot(root, root / relative, maximum_bytes=declared[relative]["size_bytes"], retain_bytes=False)
        if observed != declared[relative]:
            raise ValueError("Depth Pro output changed during verification")
    if _inventory(root) != seen | ({"evidence.json"} if completed else set()):
        raise ValueError("Depth Pro output namespace differs from exact inventory")


def verify(
    output_root: Path,
    *,
    source_root: Path,
    expected_plan_sha256: str | None = None,
    source_limits: SourceLimits | None = None,
    cancellation: Callable[[], bool] | None = None,
) -> VerifiedGradeEvidence:
    """Replay retained model evidence without loading the model or its runtime."""
    root, sources = directory_path(output_root), directory_path(source_root)
    with pinned_directory(root):
        raw, _ = snapshot(root, root / "plan.json", maximum_bytes=MAX_PLAN_BYTES)
        plan = NativeDepthProPlan(raw)
        if expected_plan_sha256 is not None and plan.sha256 != expected_plan_sha256:
            raise ValueError("Depth Pro plan differs from expected exact bytes")
        payload = plan.to_payload()
        if source_limits is not None:
            limits = SourceLimits(
                **{key: min(value, getattr(source_limits, key)) for key, value in payload["source_limits"].items()}
            )
            _resources(payload, limits)
            guarded = {**payload, "source_limits": limits.to_payload()}
        else:
            guarded = payload
        check = _guard(guarded, cancellation)
        check()
        evidence_raw, evidence_record = snapshot(root, root / "evidence.json", maximum_bytes=MAX_PLAN_BYTES)
        evidence = decode_bounded_json_object(evidence_raw)
        if (
            set(evidence) != {"schema", "plan_sha256", "artifacts", "input_count", "production_acceptance"}
            or evidence["schema"] != EVIDENCE_SCHEMA
            or evidence["plan_sha256"] != plan.sha256
            or type(evidence["input_count"]) is not int
            or evidence["input_count"] != len(payload["inputs"])
            or evidence["production_acceptance"] != "not_established"
            or canonicalize_json(evidence) != evidence_raw
        ):
            raise ValueError("Invalid native V6 Depth Pro completion")
        _verify_artifacts(
            root,
            plan,
            sources,
            evidence["artifacts"],
            check,
            completed=True,
            source_limits=SourceLimits(**guarded["source_limits"]),
        )
        if (
            sum(record["size_bytes"] for record in evidence["artifacts"]) + len(evidence_raw)
            > payload["limits"]["max_output_bytes"]
        ):
            raise ValueError("Depth Pro completion exceeds output byte budget")
        _, observed = snapshot(root, root / "evidence.json", maximum_bytes=len(evidence_raw), retain_bytes=False)
        if observed != evidence_record:
            raise ValueError("Depth Pro completion changed during verification")
        check()
    return VerifiedGradeEvidence(root, evidence_raw, plan.sha256)
