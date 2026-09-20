"""Opt-in V5 preparation on the existing governed discovery/admission machinery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformation_portal.core.execution_plan_v3 import parse_photography_plan
from transformation_portal.core.execution_plan_v4 import (
    ExecutionPlanV4,
    depth_photography_nodes,
    legacy_validation_projection,
    parse_plan_v4,
)
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4 import lifecycle as legacy

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v3.model_resolution import ResolvedModel
    from transformation_portal.materials_v4.engine import ResponsePolicy
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher


@dataclass(frozen=True)
class LuxDepthV5Request:
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
    max_input_bytes: int = 1024**3
    max_output_bytes: int = 64 * 1024**3
    wall_time_seconds: int = 3600
    memory_mib: int = 16384
    runtime_python: str | None = None
    raw_python: str | None = None
    cache_dir: Path | None = None
    companions_manifest: Path | None = None
    materials_manifest: Path | None = None
    materials_policy: ResponsePolicy | None = None
    precision: str = "fp32"
    refinement: str = "guided_bilinear"


@dataclass(frozen=True)
class PreparedLuxExecutionV5:
    plan: ExecutionPlanV4
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


def authorize_model(plan: ExecutionPlanV4) -> ResolvedModel:
    if type(plan) is not ExecutionPlanV4:
        raise TypeError("V5 requires the exact core V4 plan carrier")
    parse_plan_v4(plan.canonical_bytes)
    return legacy.authorize_model(plan)


def validate_prepared_bindings(prepared: PreparedLuxExecutionV5) -> None:
    if type(prepared) is not PreparedLuxExecutionV5 or type(prepared.plan) is not ExecutionPlanV4:
        raise TypeError("Expected PreparedLuxExecutionV5 with the exact core plan")
    plan = parse_plan_v4(prepared.canonical_plan_bytes)
    projection = parse_photography_plan(canonicalize_json(legacy_validation_projection(plan.to_payload())))
    legacy.validate_prepared_bindings(
        legacy.PreparedLuxExecutionV4(
            projection,
            prepared.input_root,
            prepared.output_root,
            prepared.runtime_python,
            prepared.raw_python,
            prepared.cache_root,
            prepared.companion_root,
            prepared.materials_root,
        )
    )


class _PreparationProfile:
    pipeline = "lux_depth_v5"
    plan_schema = "tp.execution.plan.v4"
    plan_type = ExecutionPlanV4
    prepared_type = PreparedLuxExecutionV5
    nodes = staticmethod(depth_photography_nodes)

    @staticmethod
    def configuration(request: LuxDepthV5Request) -> dict:
        if (
            type(request.precision) is not str
            or type(request.refinement) is not str
            or request.precision not in {"fp32", "fp16"}
            or request.refinement not in {"bilinear", "guided_bilinear"}
        ):
            raise ValueError("Unsupported V5 precision or refinement policy")
        if request.model_key not in ("da3-metric", "da3_metric"):
            raise ValueError("Initial V5 baseline supports only governed da3_metric")
        return {"depth": {"precision": request.precision, "refinement": request.refinement}}

    @staticmethod
    def validate_inputs(inputs: list[dict[str, Any]]) -> None:
        if any("materials" in item.get("companions", {}) for item in inputs):
            raise ValueError("V5 requires Materials V4 evidence; legacy companion material masks are unsupported")

    @staticmethod
    def validate_publication_plan(payload: dict[str, Any], limits: Any) -> None:
        from .publication import validate_publication_plan

        return validate_publication_plan(payload, limits)


def prepare(request: LuxDepthV5Request, *, publisher: GenerationPublisher | None = None) -> PreparedLuxExecutionV5:
    if type(request) is not LuxDepthV5Request:
        raise TypeError("prepare requires LuxDepthV5Request")
    prepared = legacy._prepare(request, publisher=publisher, profile=_PreparationProfile)
    validate_prepared_bindings(prepared)
    authorize_model(prepared.plan)
    return prepared
