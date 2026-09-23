"""Immutable, standalone V6 grading authority bound to a verified V5 parent.

This plan cannot authorize a managed job or new model inference. Physical paths
remain local bindings, while the portable plan identifies exact input evidence.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.io import directory_path
from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES

from .color import GradeRecipe, RenderRecipe
from .depth_maps import DepthMapRecipe
from .source import SourceLimits, VerifiedV5Source, prepare_source

PLAN_SCHEMA = "tp.lux.grade.plan.v1"
DEPTH_PLAN_SCHEMA = "tp.lux.grade.plan.v2"
MAX_PLAN_BYTES = 16 * 1024 * 1024
_MODULES = (
    "transformation_portal.lux_depth_v6.color",
    "transformation_portal.lux_depth_v6.reconstruction",
    "transformation_portal.lux_depth_v6.source",
    "transformation_portal.lux_depth_v6.plan",
    "transformation_portal.lux_depth_v6.pipeline",
    "transformation_portal.lux_depth_v6.products",
    "transformation_portal.lux_depth_v6.evidence",
    "transformation_portal.core.image_artifact",
    "transformation_portal.core.depth_evidence",
    "transformation_portal.lux_depth_v4.photography",
    "transformation_portal.lux_depth_v5.photography",
    "transformation_portal.lux_depth_v5.preview",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def processing_identity(*, depth_maps: bool = False) -> dict[str, Any]:
    """Bind the implementation and numeric/encoding dependencies used by replay."""
    modules = {}
    for name in _processing_modules(depth_maps):
        module = importlib.import_module(name)
        filename = module.__file__
        if filename is None or Path(filename).suffix != ".py":
            raise ValueError("V6 requires auditable Python source for its processing recipe")
        modules[name] = digest(Path(filename).read_bytes())
    return {
        "modules": modules,
        "dependencies": {name: version(name) for name in ("numpy", "Pillow", "scipy", "tifffile", "imagecodecs")},
    }


def _processing_modules(depth_maps: bool) -> tuple[str, ...]:
    return _MODULES + (("transformation_portal.lux_depth_v6.depth_maps",) if depth_maps else ())


def depth_recipe(payload: dict[str, Any]) -> DepthMapRecipe | None:
    """Read the optional recipe only after closed GradePlan validation."""
    return DepthMapRecipe.from_payload(payload["depth_maps"]) if payload["schema"] == DEPTH_PLAN_SCHEMA else None


@dataclass(frozen=True)
class OutputLimits:
    max_output_bytes: int = 64 * 1024**3
    wall_time_seconds: int = 3600

    def __post_init__(self) -> None:
        for name, maximum in (("max_output_bytes", 1024**4), ("wall_time_seconds", 86400)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be a positive bounded integer")

    def to_payload(self) -> dict[str, int]:
        return {"max_output_bytes": self.max_output_bytes, "wall_time_seconds": self.wall_time_seconds}


@dataclass(frozen=True)
class GradePlan:
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes or len(self.canonical_bytes) > MAX_PLAN_BYTES:
            raise ValueError("Grade plan must be bounded canonical JSON bytes")
        payload = decode_bounded_json_object(self.canonical_bytes)
        has_depth = payload.get("schema") == DEPTH_PLAN_SCHEMA
        keys = {"schema", "pipeline", "source", "grade", "render", "limits", "processing", "production_acceptance"}
        if has_depth:
            keys.add("depth_maps")
        if (
            set(payload) != keys
            or payload["schema"] not in (PLAN_SCHEMA, DEPTH_PLAN_SCHEMA)
            or payload["pipeline"] != "lux_depth_v6"
        ):
            raise ValueError("Invalid closed V6 grade plan")
        if has_depth:
            DepthMapRecipe.from_payload(payload["depth_maps"])
        if canonicalize_json(payload) != self.canonical_bytes or payload["production_acceptance"] != "not_established":
            raise ValueError("Grade plan must be canonical and cannot authorize production acceptance")
        GradeRecipe.from_payload(payload["grade"])
        RenderRecipe.from_payload(payload["render"])
        limits = payload["limits"]
        if not isinstance(limits, dict) or set(limits) != {"max_output_bytes", "wall_time_seconds"}:
            raise ValueError("Invalid V6 output limits")
        OutputLimits(**limits)
        source = payload["source"]
        if not isinstance(source, dict) or set(source) != {"digest", "plan_sha256", "evidence_sha256", "images", "limits"}:
            raise ValueError("Invalid V6 source binding")
        source_limits = source["limits"]
        if not isinstance(source_limits, dict) or set(source_limits) != {"max_input_bytes", "max_pixels", "memory_mib"}:
            raise ValueError("Invalid V6 source limits")
        SourceLimits(**source_limits)
        for name in ("digest", "plan_sha256", "evidence_sha256"):
            _require_digest(source[name])
        images = source["images"]
        if not isinstance(images, list) or not 1 <= len(images) <= 1024:
            raise ValueError("V6 requires a bounded nonempty image inventory")
        identifiers = []
        for item in images:
            image_keys = {"input_id", "shape", "master_sha256"} | ({"native_shape"} if has_depth else set())
            if not isinstance(item, dict) or set(item) != image_keys:
                raise ValueError("Invalid V6 source image")
            identifier = item["input_id"]
            if (
                not isinstance(identifier, str)
                or not identifier
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in identifier)
            ):
                raise ValueError("Invalid V6 image identifier")
            identifiers.append(identifier)
            shape = item["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(n) is not int or not 1 <= n <= 100_000_000 for n in shape)
            ):
                raise ValueError("Invalid V6 image dimensions")
            if shape[0] * shape[1] > 100_000_000:
                raise ValueError("V6 image exceeds the pixel ceiling")
            if has_depth:
                native = item["native_shape"]
                if (
                    not isinstance(native, list)
                    or len(native) != 2
                    or any(type(n) is not int or not 1 <= n <= 100_000_000 for n in native)
                    or native[0] * native[1] > 100_000_000
                ):
                    raise ValueError("Invalid V6 native depth dimensions")
            _require_digest(item["master_sha256"])
        if identifiers != sorted(set(identifiers)):
            raise ValueError("V6 image inventory must be sorted and unique")
        processing = payload["processing"]
        if not isinstance(processing, dict) or set(processing) != {"modules", "dependencies"}:
            raise ValueError("Invalid V6 processing identity")
        if not isinstance(processing["modules"], dict) or set(processing["modules"]) != set(_processing_modules(has_depth)):
            raise ValueError("V6 processing source inventory differs")
        for value in processing["modules"].values():
            _require_digest(value)
        dependencies = processing["dependencies"]
        if not isinstance(dependencies, dict) or set(dependencies) != {"numpy", "Pillow", "scipy", "tifffile", "imagecodecs"}:
            raise ValueError("V6 processing dependencies differ")
        if any(type(value) is not str or not value or len(value) > 128 for value in dependencies.values()):
            raise ValueError("V6 dependency versions must be bounded strings")
        validate_resources(payload)

    @property
    def sha256(self) -> str:
        return digest(self.canonical_bytes)

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)


def _require_digest(value: Any) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected lowercase SHA256")


def validate_resources(payload: dict[str, Any]) -> None:
    """Apply the same complete admission at preparation, execution, and replay.

    GradePlan validates payload structure before this cross-field check. Three
    float RGB masters, optional float alpha, and RGBA TIFF consume at most 48
    bytes/pixel. Every image separately reserves its bounded PNG and metadata;
    canonical plan and completion each receive their own full byte allowance.
    """
    source = payload["source"]
    limits = source["limits"]
    reserve = 2 * MAX_PLAN_BYTES
    has_depth = payload["schema"] == DEPTH_PLAN_SCHEMA
    for image in source["images"]:
        pixels = image["shape"][0] * image["shape"][1]
        if pixels > limits["max_pixels"]:
            raise ValueError("V6 image exceeds its admitted per-image pixel budget")
        native_pixels = image["native_shape"][0] * image["native_shape"][1] if has_depth else 0
        memory = pixels * (320 if has_depth else 256) + native_pixels * 128 + 256 * 1024**2
        if memory > limits["memory_mib"] * 1024**2:
            raise ValueError("V6 image exceeds conservative reconstruction/replay memory admission")
        reserve += pixels * 48 + MAX_PREVIEW_BYTES + 2 * 1024**2
        if has_depth:
            # Four native carriers, five aligned carriers (including optional
            # calibrated meters), float TIFF, bounded scalar PNGs and headers.
            reserve += native_pixels * 7 + pixels * 18 + 12 * 1024**2
    if reserve > payload["limits"]["max_output_bytes"]:
        raise ValueError("V6 output reservation exceeds the admitted byte budget")


@dataclass(frozen=True)
class LuxDepthV6Request:
    input_dir: Path
    output_dir: Path
    grade: GradeRecipe = field(default_factory=GradeRecipe)
    render: RenderRecipe = field(default_factory=RenderRecipe)
    source_limits: SourceLimits = field(default_factory=SourceLimits)
    output_limits: OutputLimits = field(default_factory=OutputLimits)
    depth_maps: DepthMapRecipe | None = None


@dataclass(frozen=True)
class PreparedLuxExecutionV6:
    plan: GradePlan
    source: VerifiedV5Source
    output_root: Path

    @property
    def canonical_plan_bytes(self) -> bytes:
        return self.plan.canonical_bytes


def source_binding(source: VerifiedV5Source, *, depth_maps: bool = False) -> dict[str, Any]:
    return {
        "digest": source.source_digest,
        "plan_sha256": digest(source.canonical_plan_bytes),
        "evidence_sha256": digest(source.canonical_evidence_bytes),
        "limits": source.limits.to_payload(),
        "images": [
            {
                "input_id": image.input_id,
                "shape": list(image.shape),
                "master_sha256": image.master_sha256,
                **({"native_shape": image.descriptor["depth"]["shape"]} if depth_maps else {}),
            }
            for image in source.images
        ],
    }


def prepare(request: LuxDepthV6Request, *, cancellation: Callable[[], bool] | None = None) -> PreparedLuxExecutionV6:
    """Read-only planning verifies the complete parent without loading a model."""
    if type(request) is not LuxDepthV6Request:
        raise TypeError("V6 preparation requires LuxDepthV6Request")
    for value, expected in (
        (request.grade, GradeRecipe),
        (request.render, RenderRecipe),
        (request.source_limits, SourceLimits),
        (request.output_limits, OutputLimits),
    ):
        if type(value) is not expected:
            raise TypeError("V6 request requires exact typed recipes and limits")
    if request.depth_maps is not None and type(request.depth_maps) is not DepthMapRecipe:
        raise TypeError("V6 depth maps require an exact typed recipe")
    output = directory_path(request.output_dir, allow_missing=True)
    source_root = directory_path(request.input_dir)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("V6 output must be new with an existing parent")
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("V6 output and parent evidence must be disjoint")
    source = prepare_source(source_root, limits=request.source_limits, cancellation=cancellation)
    payload = {
        "schema": DEPTH_PLAN_SCHEMA if request.depth_maps is not None else PLAN_SCHEMA,
        "pipeline": "lux_depth_v6",
        "source": source_binding(source, depth_maps=request.depth_maps is not None),
        "grade": request.grade.to_payload(),
        "render": request.render.to_payload(),
        "limits": request.output_limits.to_payload(),
        "processing": processing_identity(depth_maps=request.depth_maps is not None),
        "production_acceptance": "not_established",
    }
    if request.depth_maps is not None:
        payload["depth_maps"] = request.depth_maps.to_payload()
    return PreparedLuxExecutionV6(GradePlan(canonicalize_json(payload)), source, output)
