"""Managed V6 dispatch consumes a frozen V5 inference and finishing contract."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchFence
from transformation_portal.orchestrator.photography_adapter import (
    PHOTOGRAPHY_BINDINGS_SCHEMA,
    ManagedPhotographyPublisher,
    PhotographyBindings,
    consume_photography_dispatch,
    server_runtime_bindings,
    validate_photography_dispatch,
)

if TYPE_CHECKING:
    from transformation_portal.core.execution_plan_v5 import ExecutionPlanV5
    from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request, PreparedManagedLuxExecutionV6


@dataclass(frozen=True)
class PreparedV6Dispatch:
    plan_bytes: bytes
    bindings_bytes: bytes

    def __post_init__(self) -> None:
        validate_v6_dispatch(self.plan_bytes, self.bindings_bytes)


def validate_v6_dispatch(plan_bytes: bytes, bindings_bytes: bytes) -> ExecutionPlanV5:
    """Validate the closed composition and the same physical V5 input authority."""
    from transformation_portal.core.execution_plan_v5 import ExecutionPlanV5

    plan = ExecutionPlanV5(plan_bytes)
    validate_photography_dispatch(canonicalize_json(plan.to_payload()["inference"]), bindings_bytes)
    return plan


def prepare_v6_dispatch(request: ManagedLuxDepthV6Request, *, publisher: GenerationPublisher) -> PreparedV6Dispatch:
    """Freeze raw-photo inference and V6 finishing using server-owned runtimes."""
    from transformation_portal.lux_depth.lifecycle import prepare

    runtime, raw = server_runtime_bindings()
    inference = request.inference
    if inference.runtime_python not in (None, runtime) or inference.raw_python not in (None, raw):
        raise ExecutionPlanError("Photography interpreters must be selected by the server")
    request = replace(request, inference=replace(inference, runtime_python=runtime, raw_python=raw))
    prepared = prepare(request, publisher=publisher)
    prepared_inference = prepared.inference
    bindings = PhotographyBindings.from_payload(
        {
            "schema": PHOTOGRAPHY_BINDINGS_SCHEMA,
            "input_root": str(prepared_inference.input_root),
            "runtime_python": prepared_inference.runtime_python,
            "raw_python": prepared_inference.raw_python,
            "cache_root": None if prepared_inference.cache_root is None else str(prepared_inference.cache_root),
            "companion_root": None if prepared_inference.companion_root is None else str(prepared_inference.companion_root),
            "materials_root": None if prepared_inference.materials_root is None else str(prepared_inference.materials_root),
        }
    )
    return PreparedV6Dispatch(prepared.plan.canonical_bytes, bindings.canonical_bytes)


def consume_v6_dispatch(plan_bytes: bytes, bindings_bytes: bytes, *, execution_root: Path) -> PreparedManagedLuxExecutionV6:
    """Hydrate exact admitted inference; V6 recipes remain frozen in the outer plan."""
    from transformation_portal.lux_depth_v6.managed import PreparedManagedLuxExecutionV6

    plan = validate_v6_dispatch(plan_bytes, bindings_bytes)
    inference = consume_photography_dispatch(
        canonicalize_json(plan.to_payload()["inference"]), bindings_bytes, execution_root=execution_root / "source-v5"
    )
    return PreparedManagedLuxExecutionV6(plan, inference)


class ManagedV6PhotographyPublisher(ManagedPhotographyPublisher):
    """Publish verified TIFF/PNG photographs ahead of their supporting depth maps."""

    delivery_schema = "tp.lux.delivery.v4"
    delivery_pattern = r"v6/input-[0-9]{4}/delivery\.tif"
    pipeline_version = "V6"

    def _decorate_item(self, relative: str, item: dict[str, Any], files: Mapping[str, Path], fence: DispatchFence) -> None:
        from transformation_portal.portal import job_artifacts

        hint = item["display_hint"]
        group = f"lux-depth-v6|{Path(relative).parent.name}"
        if re.fullmatch(self.delivery_pattern, relative):
            hint.update(
                role="primary_preview",
                priority=1200,
                label="V6 photographic delivery · 16-bit TIFF",
                compare_group=f"{group}|photograph",
            )
        elif re.fullmatch(r"v6/input-[0-9]{4}/preview\.png", relative):
            hint.update(
                role="supporting_preview",
                priority=1100,
                label="V6 photographic draft · PNG",
                compare_group=f"{group}|photograph",
            )
        elif relative.startswith("source-v5/"):
            source_input = re.match(r"source-v5/(input-[0-9]{4})/", relative)
            source_group = f"lux-depth-v6|{source_input[1]}|source-v5" if source_input else "lux-depth-v6|source-v5"
            hint.update(role="file", priority=50, label="Retained V5 verification evidence", compare_group=source_group)
        elif re.fullmatch(r"v6/input-[0-9]{4}/depth-relative\.tif", relative):
            hint.update(
                role="supporting_preview",
                priority=650,
                label="Relative depth · float32 TIFF",
                compare_group=f"{group}|relative-depth",
            )
            preview = str(Path(relative).parent / "depth-preview.png")
            if preview in files:
                item["preview_url"] = job_artifacts._artifact_url(fence.locator.job_id, preview)
                item["preview_mime_type"] = "image/png"
        elif re.fullmatch(r"v6/input-[0-9]{4}/depth-preview\.png", relative):
            hint.update(
                role="supporting_preview",
                priority=600,
                label="Relative depth preview",
                compare_group=f"{group}|relative-depth",
            )
        elif re.fullmatch(r"v6/input-[0-9]{4}/depth-preview-valid\.png", relative):
            hint.update(
                role="supporting_preview",
                priority=550,
                label="Depth preview validity",
                compare_group=f"{group}|depth-validity",
            )
