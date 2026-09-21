"""Closed, opt-in HTTP request adapter for managed LuxDepthV5 photography."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from transformation_portal.orchestrator.execution_policy import (
    ExecutionPolicy,
    managed_photography_enabled,
    server_photography_cache_root,
)
from transformation_portal.orchestrator.photography_adapter import server_runtime_bindings

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request


class PhotographyJobArgs(BaseModel):
    """Public options; physical runtime and cache bindings are server-owned."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    input_dir: str = Field(min_length=1, max_length=4096)
    output_dir: str = Field(min_length=1, max_length=4096)
    model_key: Literal["da3-metric", "da3_metric"] = "da3-metric"
    device: Literal["cpu", "mps", "auto"] = "cpu"
    input_color: Literal["auto", "srgb", "linear_srgb"] = "auto"
    target_size: int = Field(default=518, ge=14, le=2044, multiple_of=14)
    strength: float = Field(default=0.25, ge=0, le=1)
    clarity: float = Field(default=0.0, ge=0, le=1)
    preview_maps: bool = False
    max_pixels: int = Field(default=100_000_000, ge=1, le=200_000_000)
    max_input_bytes: int = Field(default=1024**3, ge=1, le=2 * 1024**3)
    max_output_bytes: int = Field(default=64 * 1024**3, ge=1, le=1024**4)
    wall_time_seconds: int = Field(default=3600, ge=1, le=86400)
    memory_mib: int = Field(default=16384, ge=256, le=262144)
    precision: Literal["fp32", "fp16"] = "fp32"
    refinement: Literal["guided_bilinear", "bilinear"] = "guided_bilinear"
    companions_manifest: str | None = Field(default=None, min_length=1, max_length=4096)
    materials_manifest: str | None = Field(default=None, min_length=1, max_length=4096)
    materials_policy: dict[str, Any] | None = None


def managed_photography_readiness() -> dict[str, Any]:
    issues = []
    if not managed_photography_enabled():
        issues.append(("photography_disabled", "Managed LuxDepthV5 is not enabled on this server."))
    if (
        os.getenv("TP_ORCHESTRATOR_STATE_BACKEND", "memory").strip().lower() != "postgres"
        or os.getenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "memory").strip().lower() != "redis"
    ):
        issues.append(("photography_dispatch_required", "Managed LuxDepthV5 requires Postgres and Redis dispatch."))
    runtime, _ = server_runtime_bindings()
    if not Path(runtime).is_file() or not os.access(runtime, os.X_OK):
        issues.append(("photography_runtime_unavailable", "The server photography runtime is unavailable."))
    return {
        "status": "blocked" if issues else "ready",
        "canonical_command": "lux-depth-v5",
        "missing_prerequisites": [
            {"reason": reason, "severity": "blocked", "field": "pipeline", "message": message} for reason, message in issues
        ],
        "runner_details": {"adapter": "tp.job.photography.bindings.v1", "plan_schema": "tp.execution.plan.v4"},
        "notes": ["Model and runtime authority are verified during execution; readiness does not prove native inference."],
    }


def _json_safe_rejected_value(value: Any) -> Any:
    """Keep invalid numeric input from breaking the validation response itself."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe_rejected_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_rejected_value(item) for item in value]
    return value


def _photography_path_errors(paths: dict[str, Path], *, inspect_files: bool) -> list[dict[str, str]]:
    """Check inexpensive path prerequisites without discovering or reading media."""
    errors = []
    for name, path in paths.items():
        if not inspect_files:
            continue
        if name == "output_dir":
            ancestor = path
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
                errors.append(
                    {"field": name, "code": "output_dir_unwritable", "message": "Choose a writable output directory."}
                )
        elif name == "input_dir":
            if not path.is_dir():
                errors.append({"field": name, "code": "input_dir_required", "message": "Choose an existing input directory."})
        elif not path.is_file():
            errors.append({"field": name, "code": "not_a_file", "message": "Choose an existing manifest file."})
    output = paths.get("output_dir")
    if output is not None:
        for name, path in paths.items():
            if name == "output_dir":
                continue
            root = path if name == "input_dir" else path.parent
            if output.is_relative_to(root) or root.is_relative_to(output):
                errors.append(
                    {
                        "field": "output_dir",
                        "code": "invalid_path",
                        "message": "Output must be separate from the input and manifest directories.",
                    }
                )
                break
    return errors


def photography_config_preview(args: dict[str, Any], *, policy: ExecutionPolicy) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    normalized: dict[str, Any] = dict(args)
    try:
        parsed = PhotographyJobArgs.model_validate(args)
        normalized = parsed.model_dump()
    except ValidationError as exc:
        errors.extend(
            {"field": str(issue["loc"][0]), "code": "invalid_argument", "message": issue["msg"]}
            for issue in exc.errors(include_input=False, include_url=False)
        )
    if not errors:
        paths: dict[str, Path] = {}
        for name in ("input_dir", "output_dir", "companions_manifest", "materials_manifest"):
            value = normalized[name]
            if value is None:
                continue
            try:
                roots = policy.output_roots if name == "output_dir" else policy.input_roots
                paths[name] = policy.resolve_path(value, roots)
                normalized[name] = str(paths[name])
            except (OSError, ValueError):
                errors.append({"field": name, "code": "invalid_path", "message": "Path is outside authorized roots."})
        # Managed tenant paths are authorized by the HTTP boundary again before
        # preparation; avoid probing normalized targets across that interval.
        try:
            errors.extend(_photography_path_errors(paths, inspect_files=not policy.pilot_enabled))
        except (OSError, ValueError):
            errors.append({"field": "input_dir", "code": "invalid_path", "message": "Input paths are unavailable."})
        try:
            if normalized["materials_policy"] is not None:
                from transformation_portal.materials_v4.engine import ResponsePolicy

                if normalized["materials_manifest"] is None:
                    raise ValueError("Materials policy requires a materials manifest.")
                ResponsePolicy.from_payload(normalized["materials_policy"])
        except (TypeError, ValueError):
            errors.append({"field": "materials_policy", "code": "invalid_argument", "message": "Invalid materials policy."})
    if errors:
        normalized = {name: _json_safe_rejected_value(value) for name, value in normalized.items()}
    return {
        "pipeline": "lux-depth-v5",
        "normalized_args": normalized,
        "execution_args": dict(normalized),
        "argv_preview": "",
        "field_errors": errors,
        "field_warnings": [],
        "inactive_fields": [],
        "readiness": managed_photography_readiness(),
        "estimate_summary": {},
        "debug_bundle_summary": {},
        "next_best_action": None,
    }


def photography_request(args: dict[str, Any], *, tenant_id: str) -> LuxDepthV5Request:
    """Build the typed request from already-authorized paths; reject extra options again."""
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request

    values = PhotographyJobArgs.model_validate(args).model_dump()
    for name in ("input_dir", "output_dir", "companions_manifest", "materials_manifest"):
        if values[name] is not None:
            values[name] = Path(values[name])
    if values["materials_policy"] is not None:
        from transformation_portal.materials_v4.engine import ResponsePolicy

        values["materials_policy"] = ResponsePolicy.from_payload(values["materials_policy"])
    runtime, raw = server_runtime_bindings()
    return LuxDepthV5Request(
        **values, runtime_python=runtime, raw_python=raw, cache_dir=server_photography_cache_root(tenant_id)
    )
