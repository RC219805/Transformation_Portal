"""Managed unified route authority over unchanged native LuxDepth plans.

The immutable bindings envelope records the admitted public route and workflow.
It adds no executable authority: native plans, physical bindings, verification,
and artifact schemas retain their existing contracts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.photography_adapter import MAX_PHOTOGRAPHY_BINDINGS_BYTES, PhotographyBindings

if TYPE_CHECKING:
    from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4
    from transformation_portal.core.execution_plan_v5 import ExecutionPlanV5
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request
    from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
    from transformation_portal.orchestrator.dispatch import DispatchFence
    from transformation_portal.orchestrator.photography_adapter import PreparedPhotographyDispatch
    from transformation_portal.orchestrator.photography_v6_adapter import PreparedV6Dispatch

UNIFIED_BINDINGS_SCHEMA = "tp.job.lux_depth.bindings.v1"
_WORKFLOW_SCHEMAS = {"process": "tp.execution.plan.v5", "infer": "tp.execution.plan.v4"}


@dataclass(frozen=True)
class UnifiedBindings:
    """Closed, canonical route authority stored under the existing binding digest."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes or len(self.canonical_bytes) > MAX_PHOTOGRAPHY_BINDINGS_BYTES:
            raise ExecutionPlanError("Unified bindings must be bounded canonical bytes")
        payload = decode_bounded_json_object(self.canonical_bytes)
        if (
            set(payload) != {"schema", "pipeline", "workflow", "photography_bindings"}
            or payload["schema"] != UNIFIED_BINDINGS_SCHEMA
            or payload["pipeline"] != "lux-depth"
            or type(payload["workflow"]) is not str
            or payload["workflow"] not in _WORKFLOW_SCHEMAS
            or type(payload["photography_bindings"]) is not dict
            or canonicalize_json(payload) != self.canonical_bytes
        ):
            raise ExecutionPlanError("Invalid closed unified bindings carrier")
        PhotographyBindings(canonicalize_json(payload["photography_bindings"]))

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)

    @property
    def workflow(self) -> str:
        return self.to_payload()["workflow"]

    @property
    def photography_bindings_bytes(self) -> bytes:
        return canonicalize_json(self.to_payload()["photography_bindings"])


def is_unified_bindings(bindings_bytes: bytes | None) -> bool:
    """Recognize only the source-owned envelope; validation remains mandatory."""
    return bindings_bytes is not None and decode_bounded_json_object(bindings_bytes).get("schema") == UNIFIED_BINDINGS_SCHEMA


def validate_unified_dispatch(plan_bytes: bytes, bindings_bytes: bytes) -> ExecutionPlanV4 | ExecutionPlanV5:
    """Bind one public workflow to exactly its supported native plan family."""
    bindings = UnifiedBindings(bindings_bytes)
    if decode_bounded_json_object(plan_bytes).get("schema") != _WORKFLOW_SCHEMAS[bindings.workflow]:
        raise ExecutionPlanError("Unified workflow differs from the admitted native plan family")
    if bindings.workflow == "process":
        from transformation_portal.orchestrator.photography_v6_adapter import validate_v6_dispatch

        return validate_v6_dispatch(plan_bytes, bindings.photography_bindings_bytes)
    from transformation_portal.orchestrator.photography_adapter import validate_photography_dispatch

    return validate_photography_dispatch(plan_bytes, bindings.photography_bindings_bytes)


@dataclass(frozen=True)
class PreparedUnifiedDispatch:
    plan_bytes: bytes
    bindings_bytes: bytes

    def __post_init__(self) -> None:
        validate_unified_dispatch(self.plan_bytes, self.bindings_bytes)


def prepare_unified_dispatch(
    request: LuxDepthV5Request | ManagedLuxDepthV6Request, *, workflow: str, publisher: GenerationPublisher
) -> PreparedUnifiedDispatch:
    """Freeze native authority, then bind the unified route without replanning."""
    native: PreparedPhotographyDispatch | PreparedV6Dispatch
    if workflow == "process":
        from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
        from transformation_portal.orchestrator.photography_v6_adapter import prepare_v6_dispatch

        if type(request) is not ManagedLuxDepthV6Request:
            raise TypeError("Unified process requires an exact PhotographyRequest")
        native = prepare_v6_dispatch(request, publisher=publisher)
    elif workflow == "infer":
        from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request
        from transformation_portal.orchestrator.photography_adapter import prepare_photography_dispatch

        if type(request) is not LuxDepthV5Request:
            raise TypeError("Unified infer requires an exact InferenceRequest")
        native = prepare_photography_dispatch(request, publisher=publisher)
    else:
        raise ExecutionPlanError("Unsupported managed LuxDepth workflow")
    bindings = canonicalize_json(
        {
            "schema": UNIFIED_BINDINGS_SCHEMA,
            "pipeline": "lux-depth",
            "workflow": workflow,
            "photography_bindings": PhotographyBindings(native.bindings_bytes).to_payload(),
        }
    )
    return PreparedUnifiedDispatch(native.plan_bytes, bindings)


async def publish_unified_result(
    plan_bytes: bytes, bindings_bytes: bytes, *, publisher: GenerationPublisher, fence: DispatchFence
) -> dict[str, Any]:
    """Independently verify native output and project its admitted public workflow."""
    validate_unified_dispatch(plan_bytes, bindings_bytes)
    bindings = UnifiedBindings(bindings_bytes)
    if bindings.workflow == "process":
        from transformation_portal.lux_depth_v6.publication import _publication_arguments

        arguments = await asyncio.to_thread(_publication_arguments, plan_bytes, publisher, fence)
    else:
        from transformation_portal.lux_depth_v4.publication import _prepare_publication
        from transformation_portal.lux_depth_v5.publication import _PublicationProfile

        arguments = await asyncio.to_thread(
            _prepare_publication,
            None,
            publisher=publisher,
            fence=fence,
            profile=_PublicationProfile,
            admitted_plan_bytes=plan_bytes,
        )
    summary = arguments["run_summary"]
    arguments["run_summary"] = {
        **summary,
        "pipeline": "lux_depth",
        "workflow": bindings.workflow,
        "engine_pipeline": summary["pipeline"],
    }
    return await publisher.publish(fence, **arguments)
