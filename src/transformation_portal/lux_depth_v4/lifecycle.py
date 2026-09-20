"""Resolve photographic requests once, without loading a model or writing outputs."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema

from transformation_portal.core.da3_runtime import repo_local_da3_python_path
from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2, digest_payload, photography_nodes
from transformation_portal.core.execution_plan_v3 import (
    ExecutionPlanV3,
    PhotographyPlan,
    materials_photography_nodes,
    parse_photography_plan,
)
from transformation_portal.lux_depth_v3.model_resolution import ModelRequest, ResolvedModel, resolve_model_contract
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v4.backend import CarriedDepthPlan
    from transformation_portal.materials_v4.engine import ResponsePolicy
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher

_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".cr2", ".nef", ".arw"})


@dataclass(frozen=True)
class LuxDepthV4Request:
    """Explicit request; execution never accepts a second mutable configuration."""

    input_dir: Path
    output_dir: Path
    model_key: str = "da3-metric"
    device: str = "cpu"
    input_color: str = "auto"
    target_size: int = 518
    strength: float = 0.25
    clarity: float = 0.0
    preview_maps: bool = False
    max_pixels: int = 100_000_000
    max_input_bytes: int = 1024 * 1024 * 1024
    max_output_bytes: int = 64 * 1024 * 1024 * 1024
    wall_time_seconds: int = 3600
    memory_mib: int = 16384
    runtime_python: str | None = None
    raw_python: str | None = None
    cache_dir: Path | None = None
    companions_manifest: Path | None = None
    materials_manifest: Path | None = None
    materials_policy: ResponsePolicy | None = None


@dataclass(frozen=True)
class PreparedLuxExecutionV4:
    """Immutable plan and local physical bindings; plan contains no executable path."""

    plan: PhotographyPlan
    input_root: Path
    output_root: Path
    runtime_python: str
    raw_python: str | None
    cache_root: Path | None
    companion_root: Path | None = None
    materials_root: Path | None = None

    @property
    def canonical_plan_bytes(self) -> bytes:
        return self.plan.canonical_bytes


def authorize_model(plan: CarriedDepthPlan) -> ResolvedModel:
    """Independently revalidate the model, revision and commercial-use boundary."""
    carried = plan.to_payload()["model"]
    resolved = resolve_model_contract(
        ModelRequest(model_key=carried["canonical_key"], strict_model_lock=True, requested_revision=carried["revision"])
    )
    expected = {
        "canonical_key": resolved.canonical_key,
        "repo_id": resolved.spec.repo_id,
        "revision": resolved.revision,
        "license": resolved.spec.license_id,
    }
    if carried != expected or resolved.canonical_key != "da3_metric":
        raise ValueError("Carried V4 model does not match the governed photography model")
    return resolved


def _configuration_and_resources(request: LuxDepthV4Request) -> tuple[dict, dict]:
    """Validate governed request fields before discovery, hashing, or device probes."""
    configuration = {
        "input_color": request.input_color,
        "target_size": request.target_size,
        "strength": request.strength,
        "clarity": request.clarity,
        "preview_maps": request.preview_maps,
    }
    resources = {
        "max_pixels": request.max_pixels,
        "max_input_bytes": request.max_input_bytes,
        "max_output_bytes": request.max_output_bytes,
        "wall_time_seconds": request.wall_time_seconds,
        "memory_mib": request.memory_mib,
        "inference_slots": 1,
    }
    if not isinstance(request.device, str) or request.device not in {"cpu", "mps", "auto"}:
        raise ValueError("V4 device must be cpu, mps, or auto")
    for value in (request.strength, request.clarity):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Photographic strength and clarity must be finite numbers")
    schema = json.loads(files("transformation_portal.schemas.execution").joinpath("plan.v2.schema.json").read_text())
    for name, payload in (("configuration", configuration), ("resources", resources)):
        try:
            jsonschema.Draft202012Validator(schema["properties"][name]).validate(payload)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Invalid V4 {name}: {exc.message}") from exc
    return configuration, resources


def _discover_inputs(root: Path) -> list[Path]:
    """Select a bounded, deterministic inventory without silently skipping linked subtrees."""
    selected = []

    def unreadable(error: OSError) -> None:
        raise error

    for current, directories, names in os.walk(root, followlinks=False, onerror=unreadable):
        directory = Path(current)
        if any((directory / name).is_symlink() for name in directories):
            raise ValueError("Input tree must not contain linked directories")
        for name in names:
            path = directory / name
            if path.suffix.lower() not in _SUFFIXES:
                continue
            selected.append(path)
            if len(selected) > 1024:
                raise ValueError("Input count exceeds 1024")
    if not selected:
        raise ValueError("Input selection contains no supported photographic files")
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _overlap(first: Path, second: Path) -> bool:
    return first.is_relative_to(second) or second.is_relative_to(first)


def validate_prepared_bindings(prepared: PreparedLuxExecutionV4) -> None:
    """Revalidate physical bindings without resolving/loading a model or creating paths."""
    if type(prepared) is not PreparedLuxExecutionV4:
        raise TypeError("Expected PreparedLuxExecutionV4")
    if type(prepared.plan) not in (ExecutionPlanV2, ExecutionPlanV3):
        raise TypeError("Prepared execution requires an exact core photography plan carrier")
    parse_photography_plan(prepared.canonical_plan_bytes)
    root = directory_path(prepared.input_root)
    output = directory_path(prepared.output_root, allow_missing=True)
    if root != prepared.input_root or output != prepared.output_root or _overlap(root, output):
        raise ValueError("Prepared input/output roots must be canonical and non-overlapping")
    if prepared.cache_root is not None:
        cache = directory_path(prepared.cache_root, allow_missing=True)
        if cache != prepared.cache_root or _overlap(cache, root) or _overlap(cache, output):
            raise ValueError("Prepared cache must be canonical and separate from input/output roots")

    has_companions = any("companions" in item for item in prepared.plan.to_payload()["inputs"])
    if has_companions != (prepared.companion_root is not None):
        raise ValueError("Prepared companion namespace differs from its input bindings")
    if prepared.companion_root is not None:
        companion_root = directory_path(prepared.companion_root)
        if companion_root != prepared.companion_root or _overlap(companion_root, output):
            raise ValueError("Prepared companion root must be canonical and separate from outputs")
        if prepared.cache_root is not None and _overlap(companion_root, prepared.cache_root):
            raise ValueError("Prepared companion root must be separate from cache")
    has_materials = "materials_manifest" in prepared.plan.to_payload()
    if has_materials != (prepared.materials_root is not None):
        raise ValueError("Prepared material namespace differs from its input bindings")
    if prepared.materials_root is not None:
        materials_root = directory_path(prepared.materials_root)
        if materials_root != prepared.materials_root or _overlap(materials_root, output):
            raise ValueError("Prepared material root must be canonical and separate from outputs")
        if prepared.cache_root is not None and _overlap(materials_root, prepared.cache_root):
            raise ValueError("Prepared material root must be separate from cache")


def prepare(request: LuxDepthV4Request, *, publisher: GenerationPublisher | None = None) -> PreparedLuxExecutionV4:
    """Freeze content selection, model, device and processing policy exactly once."""
    if not isinstance(request, LuxDepthV4Request):
        raise TypeError("prepare requires LuxDepthV4Request")
    return _prepare(request, publisher=publisher)


def _prepare(request: Any, *, publisher: GenerationPublisher | None = None, profile: Any = None) -> Any:
    """Shared discovery and admission; public version boundaries select a fixed profile."""
    configuration, resources = _configuration_and_resources(request)
    if profile is not None:
        configuration.update(profile.configuration(request))
    if request.materials_policy is not None and request.materials_manifest is None:
        raise ValueError("A Materials V4 policy requires an explicit materials manifest")
    publication_limits = None if publisher is None else publisher.limits
    if publication_limits is not None:
        resources["max_output_bytes"] = min(resources["max_output_bytes"], publication_limits.max_total_bytes)
    root = directory_path(request.input_dir)
    output = directory_path(request.output_dir, allow_missing=True)
    if _overlap(root, output):
        raise ValueError("Input and output roots must be separate non-overlapping directories")
    cache = None if request.cache_dir is None else directory_path(request.cache_dir, allow_missing=True)
    if cache is not None and (_overlap(cache, root) or _overlap(cache, output)):
        raise ValueError("Cache must be separate from input and output roots")
    inputs: list[dict[str, Any]] = []
    with pinned_directory(root):
        for path in _discover_inputs(root):
            _, record = snapshot(root, path, maximum_bytes=request.max_input_bytes, retain_bytes=False)
            inputs.append({"id": f"input-{len(inputs):04d}", **record})
    companion_root = None
    companion_manifest = None
    if request.companions_manifest is not None:
        from .companions import freeze_companions

        companion_records, companion_root, companion_manifest = freeze_companions(
            request.companions_manifest,
            inputs,
            max_input_bytes=request.max_input_bytes,
            max_pixels=request.max_pixels,
        )
        if _overlap(companion_root, output) or (cache is not None and _overlap(companion_root, cache)):
            raise ValueError("Companion root must be separate from output and cache roots")
        for item in inputs:
            if item["path"] in companion_records:
                item["companions"] = companion_records[item["path"]]
    materials_root = None
    materials_manifest = None
    if request.materials_manifest is not None:
        from transformation_portal.materials_v4.engine import ResponsePolicy

        from .materials import freeze_materials

        policy = request.materials_policy if request.materials_policy is not None else ResponsePolicy()
        if type(policy) is not ResponsePolicy:
            raise ValueError("Materials V4 requires an explicit ResponsePolicy")
        if any("materials" in item.get("companions", {}) for item in inputs):
            raise ValueError("Cannot combine legacy material masks and Materials V4")
        materials_records, materials_root, materials_manifest = freeze_materials(request.materials_manifest, inputs, resources)
        if _overlap(materials_root, output) or (cache is not None and _overlap(materials_root, cache)):
            raise ValueError("Material root must be separate from output and cache roots")
        for item in inputs:
            if item["path"] in materials_records:
                item["materials_v4"] = materials_records[item["path"]]
        configuration["materials_v4"] = policy.to_payload()
    if profile is not None:
        profile.validate_inputs(inputs)
    if publication_limits is not None:
        from .publication import validate_publication_plan

        (validate_publication_plan if profile is None else profile.validate_publication_plan)(
            {
                "inputs": inputs,
                "configuration": configuration,
                "resources": resources,
                "publication": publication_limits.to_payload(),
            },
            publication_limits,
        )
    resolved = resolve_model_contract(ModelRequest(model_key=request.model_key, strict_model_lock=True))
    if resolved.canonical_key != "da3_metric":
        raise ValueError("Initial V4 photography profile supports only governed da3_metric")
    interpreter = request.runtime_python or os.environ.get("TRANSFORMATION_PORTAL_DA3_PYTHON")
    if interpreter is None:
        candidate = repo_local_da3_python_path(Path(__file__))
        interpreter = str(candidate) if candidate is not None else ""
    # Keep the venv executable spelling: resolving its symlink would launch the base interpreter.
    interpreter = os.path.abspath(os.path.expanduser(interpreter)) if interpreter else ""
    device = request.device
    if device != "cpu":
        from transformation_portal.lux_depth_v4.backend import probe_device

        device = probe_device(interpreter, device)
    payload = {
        "schema": (
            ("tp.execution.plan.v3" if materials_manifest is not None else "tp.execution.plan.v2")
            if profile is None
            else profile.plan_schema
        ),
        "canonicalization": "tp.canonical.json.v1",
        "pipeline": "lux_depth_v4" if profile is None else profile.pipeline,
        "model": {
            "canonical_key": resolved.canonical_key,
            "repo_id": resolved.spec.repo_id,
            "revision": resolved.revision,
            "license": resolved.spec.license_id,
        },
        "device": device,
        "inputs": inputs,
        "configuration": configuration,
        "resources": resources,
        "nodes": (
            (materials_photography_nodes if materials_manifest is not None else photography_nodes)
            if profile is None
            else profile.nodes
        )(configuration, companions=companion_root is not None),
    }
    if companion_manifest is not None:
        payload["companions_manifest"] = companion_manifest
    if materials_manifest is not None:
        payload["materials_manifest"] = materials_manifest
    if publication_limits is not None:
        payload["publication"] = publication_limits.to_payload()
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    carrier = (
        (ExecutionPlanV3 if materials_manifest is not None else ExecutionPlanV2) if profile is None else profile.plan_type
    )
    plan = carrier.from_payload(payload)
    authorize_model(plan)
    raw_python = request.raw_python or os.environ.get("TRANSFORMATION_PORTAL_RAW_PYTHON")
    raw_python = os.path.abspath(os.path.expanduser(raw_python)) if raw_python else None
    prepared_type = PreparedLuxExecutionV4 if profile is None else profile.prepared_type
    return prepared_type(plan, root, output, interpreter, raw_python, cache, companion_root, materials_root)
