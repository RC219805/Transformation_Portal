"""Closed, opt-in HTTP requests for fresh LuxDepthV6 photography and depth maps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, ValidationError, field_validator

from transformation_portal.core.execution_plan_v5 import ENVELOPE_RESERVE
from transformation_portal.orchestrator.execution_policy import ExecutionPolicy
from transformation_portal.orchestrator.photography_adapter import server_runtime_bindings
from transformation_portal.portal.photography_jobs import (
    PhotographyJobArgs,
    _json_safe_rejected_value,
    photography_config_preview,
    photography_request,
)

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request


class PhotographyV6JobArgs(PhotographyJobArgs):
    """Flat JSON controls; depth products are mandatory and bindings stay server-owned."""

    max_pixels: int = Field(default=100_000_000, ge=1, le=100_000_000)
    max_output_bytes: int = Field(default=64 * 1024**3, ge=ENVELOPE_RESERVE + 2, le=1024**4)
    exposure_stops: float = Field(default=0.0, ge=-8, le=8)
    white_balance: list[Annotated[float, Field(ge=0.25, le=4)]] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0], min_length=3, max_length=3
    )
    contrast: float = Field(default=1.0, ge=0.25, le=4)
    pivot: float = Field(default=0.18, ge=0.001, le=1)
    saturation: float = Field(default=1.0, ge=0, le=2)
    render: Literal["perceptual_srgb", "soft_srgb", "clip_srgb"] = "perceptual_srgb"
    shoulder: float = Field(default=0.8, ge=0.1, le=0.95)
    depth_refinement: Literal["guided_bilinear_v4", "guided_bilinear_v3", "bilinear"] = "guided_bilinear_v4"

    @field_validator("materials_manifest", "materials_policy", mode="before")
    @classmethod
    def reject_materials(cls, value: Any) -> None:
        if value is not None:
            raise ValueError("Managed LuxDepthV6 does not support materials evidence or response policies.")
        return None


def managed_v6_readiness() -> dict[str, Any]:
    """Cheap readiness independent of V5 activation; execution verifies model authority."""
    issues = []
    if os.getenv("TP_LUX_V6_MANAGED_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        issues.append(("photography_v6_disabled", "Managed LuxDepthV6 is not enabled on this server."))
    if (
        os.getenv("TP_ORCHESTRATOR_STATE_BACKEND", "memory").strip().lower() != "postgres"
        or os.getenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "memory").strip().lower() != "redis"
    ):
        issues.append(("photography_dispatch_required", "Managed LuxDepthV6 requires Postgres and Redis dispatch."))
    runtime, _ = server_runtime_bindings()
    if not Path(runtime).is_file() or not os.access(runtime, os.X_OK):
        issues.append(("photography_runtime_unavailable", "The server photography runtime is unavailable."))
    return {
        "status": "blocked" if issues else "ready",
        "canonical_command": "lux-depth-v6",
        "missing_prerequisites": [
            {"reason": reason, "severity": "blocked", "field": "pipeline", "message": message} for reason, message in issues
        ],
        "runner_details": {"adapter": "tp.job.photography.bindings.v1", "plan_schema": "tp.execution.plan.v5"},
        "notes": ["Model and runtime authority are verified during execution; readiness does not prove native inference."],
    }


def v6_config_preview(args: dict[str, Any], *, policy: ExecutionPolicy) -> dict[str, Any]:
    """Validate the complete V6 request before reusing common photography path guards."""
    try:
        values = PhotographyV6JobArgs.model_validate(args).model_dump()
    except ValidationError as exc:
        normalized = _json_safe_rejected_value(dict(args))
        return {
            "pipeline": "lux-depth-v6",
            "normalized_args": normalized,
            "execution_args": dict(normalized),
            "argv_preview": "",
            "field_errors": [
                {"field": str(issue["loc"][0]), "code": "invalid_argument", "message": issue["msg"]}
                for issue in exc.errors(include_input=False, include_url=False)
            ],
            "field_warnings": [],
            "inactive_fields": [],
            "readiness": managed_v6_readiness(),
            "estimate_summary": {},
            "debug_bundle_summary": {},
            "next_best_action": None,
        }
    common = {name: values[name] for name in PhotographyJobArgs.model_fields}
    preview = photography_config_preview(common, policy=policy)
    finishing = {name: value for name, value in values.items() if name not in PhotographyJobArgs.model_fields}
    preview["normalized_args"].update(finishing)
    preview["execution_args"].update(finishing)
    preview["pipeline"] = "lux-depth-v6"
    preview["readiness"] = managed_v6_readiness()
    return preview


def v6_request(args: dict[str, Any], *, tenant_id: str) -> ManagedLuxDepthV6Request:
    """Freeze inference and finishing controls after the HTTP boundary authorizes paths."""
    from transformation_portal.lux_depth_v6.color import GradeRecipe, RenderRecipe
    from transformation_portal.lux_depth_v6.depth_maps import DepthMapRecipe
    from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request

    values = PhotographyV6JobArgs.model_validate(args).model_dump()
    inference = photography_request({name: values[name] for name in PhotographyJobArgs.model_fields}, tenant_id=tenant_id)
    return ManagedLuxDepthV6Request(
        inference=inference,
        grade=GradeRecipe(
            exposure_stops=values["exposure_stops"],
            white_balance=tuple(values["white_balance"]),
            contrast=values["contrast"],
            pivot=values["pivot"],
            saturation=values["saturation"],
        ),
        render=RenderRecipe(mode=values["render"], shoulder=values["shoulder"]),
        depth_maps=DepthMapRecipe(refinement=values["depth_refinement"]),
    )
