"""Closed HTTP workflows for unified LuxDepth with independent managed access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from transformation_portal.orchestrator.execution_policy import ExecutionPolicy, managed_lux_depth_enabled
from transformation_portal.orchestrator.photography_adapter import server_runtime_bindings
from transformation_portal.portal.photography_jobs import (
    PhotographyJobArgs,
    _json_safe_rejected_value,
    photography_config_preview,
    photography_request,
)
from transformation_portal.portal.photography_v6_jobs import PhotographyV6JobArgs, v6_config_preview, v6_request

if TYPE_CHECKING:
    from transformation_portal.core.security.tenant import TenantContext
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request
    from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request


class LuxDepthProcessArgs(PhotographyV6JobArgs):
    """Original photographs through depth inference and verified finishing."""

    workflow: Literal["process"] = "process"


class LuxDepthInferArgs(PhotographyJobArgs):
    """Original photographs through depth inference with optional Materials."""

    workflow: Literal["infer"] = "infer"


def _parse_args(args: dict[str, Any]) -> LuxDepthProcessArgs | LuxDepthInferArgs:
    # An unknown/non-string workflow reaches the process Literal validator;
    # it is never silently normalized to an executable workflow.
    model = LuxDepthInferArgs if args.get("workflow", "process") == "infer" else LuxDepthProcessArgs
    return model.model_validate(args)


def managed_lux_depth_readiness(workflow: Any = "process") -> dict[str, Any]:
    """Report cheap prerequisites without importing an inference runtime."""
    issues = []
    supported = isinstance(workflow, str) and workflow in ("process", "infer")
    if not supported:
        issues.append(("workflow", "unsupported_workflow", "Choose the process or infer workflow."))
    if not managed_lux_depth_enabled():
        issues.append(("pipeline", "lux_depth_disabled", "Managed Lux Depth Unified is not enabled on this server."))
    if (
        os.getenv("TP_ORCHESTRATOR_STATE_BACKEND", "memory").strip().lower() != "postgres"
        or os.getenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "memory").strip().lower() != "redis"
    ):
        issues.append(("pipeline", "photography_dispatch_required", "Managed Lux Depth Unified requires Postgres and Redis."))
    runtime, _ = server_runtime_bindings()
    if not Path(runtime).is_file() or not os.access(runtime, os.X_OK):
        issues.append(("pipeline", "photography_runtime_unavailable", "The server photography runtime is unavailable."))
    return {
        "status": "blocked" if issues else "ready",
        "canonical_command": f"lux-depth {workflow}" if supported else "lux-depth",
        "missing_prerequisites": [
            {"reason": reason, "severity": "blocked", "field": field, "message": message} for field, reason, message in issues
        ],
        "runner_details": {
            "adapter": "tp.job.lux_depth.bindings.v1",
            "workflow": workflow if supported else None,
            "supported_workflows": ["process", "infer"],
            "plan_schema": (
                ("tp.execution.plan.v5" if workflow == "process" else "tp.execution.plan.v4") if supported else None
            ),
        },
        "notes": ["Model and runtime authority are verified during execution; readiness does not prove native inference."],
    }


def lux_depth_config_preview(
    args: dict[str, Any], *, policy: ExecutionPolicy, tenant: TenantContext | None = None
) -> dict[str, Any]:
    """Validate workflow-specific options and retain the established path guards."""
    try:
        values = _parse_args(args).model_dump()
    except ValidationError as exc:
        normalized = _json_safe_rejected_value(dict(args))
        return {
            "pipeline": "lux-depth",
            "normalized_args": normalized,
            "execution_args": dict(normalized),
            "argv_preview": "",
            "field_errors": [
                {"field": str(issue["loc"][0]), "code": "invalid_argument", "message": issue["msg"]}
                for issue in exc.errors(include_input=False, include_url=False)
            ],
            "field_warnings": [],
            "inactive_fields": [],
            "readiness": managed_lux_depth_readiness(args.get("workflow", "process")),
            "estimate_summary": {},
            "debug_bundle_summary": {},
            "next_best_action": None,
        }
    workflow = values.pop("workflow")
    preview = (v6_config_preview if workflow == "process" else photography_config_preview)(
        values, policy=policy, tenant=tenant
    )
    preview["pipeline"] = "lux-depth"
    preview["normalized_args"]["workflow"] = workflow
    preview["execution_args"]["workflow"] = workflow
    preview["readiness"] = managed_lux_depth_readiness(workflow)
    return preview


def lux_depth_request(args: dict[str, Any], *, tenant_id: str) -> ManagedLuxDepthV6Request | LuxDepthV5Request:
    """Construct only an exact native carrier after HTTP tenant/path authorization."""
    values = _parse_args(args).model_dump()
    workflow = values.pop("workflow")
    return (v6_request if workflow == "process" else photography_request)(values, tenant_id=tenant_id)


def lux_depth_config_metadata() -> dict[str, Any]:
    """Discoverable closed public controls; runtime and cache paths stay private."""
    return {
        "pipeline": "lux-depth",
        "default_workflow": "process",
        "workflows": {
            "process": {
                "label": "Process photographs",
                "description": "Depth inference, photographic finishing, and depth products.",
                "plan_schema": "tp.execution.plan.v5",
                "args_schema": LuxDepthProcessArgs.model_json_schema(),
            },
            "infer": {
                "label": "Infer depth and Materials",
                "description": "Depth inference with optional source-bound Materials evidence.",
                "plan_schema": "tp.execution.plan.v4",
                "args_schema": LuxDepthInferArgs.model_json_schema(),
            },
        },
        "production_acceptance": "not_established",
    }
