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
    from transformation_portal.core.security.tenant import TenantContext
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request


class PhotographyJobArgs(BaseModel):
    """Public options; physical runtime and cache bindings are server-owned."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    input_dir: str = Field(min_length=1, max_length=4096)
    output_dir: str = Field(min_length=1, max_length=4096)
    model_key: Literal["da3-metric", "da3_metric"] = "da3-metric"
    device: Literal["cpu", "mps", "auto"] = "cpu"
    input_color: Literal["auto", "auto_assume_srgb", "srgb", "linear_srgb"] = "auto"
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


def _photography_input_color_preflight(
    args: dict[str, Any], *, policy: ExecutionPolicy, tenant: TenantContext | None
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Inspect authorized input headers, preserving tenant and ingest color policy."""
    from PIL.Image import DecompressionBombError

    from transformation_portal.core.security.tenant import TenantError
    from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
    from transformation_portal.lux_depth_v4.input_preflight import validate_input_directory_colors
    from transformation_portal.lux_depth_v4.photography import InputColorError
    from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

    root = Path(args["input_dir"])
    if policy.pilot_enabled:
        # Internal callers without a verified tenant may normalize arguments,
        # but cannot authorize a new content probe.
        if tenant is None:
            return [], []
        roots = tuple(
            base / tenant.tenant_id
            for configured in (tenant.workspace_root, tenant.cas_root)
            for base in (configured.absolute(), configured.resolve())
        )
        if ".." in root.parts or not any(root.absolute().is_relative_to(base) for base in roots):
            return [
                {"field": "input_dir", "code": "invalid_path", "message": "Input is outside the authorized workspace."}
            ], []
        try:
            policy.tenant_guard(tenant).enforce_path(root)
        except (TenantError, DispatchAuthorityLost, OSError, ValueError):
            return [
                {"field": "input_dir", "code": "invalid_path", "message": "Input is outside the authorized workspace."}
            ], []
    try:
        preparations = validate_input_directory_colors(
            root,
            input_color=args["input_color"],
            max_input_bytes=args["max_input_bytes"],
            max_pixels=args["max_pixels"],
        )
    except InputColorError as exc:
        return [{"field": "input_color", "code": exc.code, "message": str(exc)}], []
    except (OSError, ValueError, TypeError, IndexError, RecursionError, ArtifactEvidenceError, DecompressionBombError):
        return [
            {
                "field": "input_dir",
                "code": "input_metadata_invalid",
                "message": "Input metadata could not be validated within the configured limits. "
                "Check the image files and size limits, or re-export a smaller batch of photographs.",
            }
        ], []
    return [], preparations


def _color_preparation_summary(
    preparations: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Describe planned color handling without claiming pixels were processed."""
    if not preparations:
        return [], {}
    converted = sum(item["engine"] == "imagecodecs_lcms" for item in preparations)
    assumed = sum(item["assumed_srgb"] is True for item in preparations)
    overrides = sum(item["action"] == "explicit" and item["source_icc_sha256"] is not None for item in preparations)
    warnings = []
    if assumed:
        warnings.append(
            {
                "field": "input_color",
                "code": "input_color_assumed_srgb",
                "message": f"{assumed} image(s) have no usable color metadata. Preparation will assume sRGB; "
                "this is an assumption, not a detected profile. The choice will be recorded in the photographic evidence.",
            }
        )
    if overrides:
        warnings.append(
            {
                "field": "input_color",
                "code": "input_color_profile_override",
                "message": f"Explicit Input color overrides {overrides} embedded profile(s) without converting their pixels. "
                "Choose Auto to convert supported profiles into the working color space.",
            }
        )
    return warnings, {
        "summary_label": f"Color preparation: {len(preparations)} image(s), {converted} to convert, {assumed} sRGB assumption(s).",
        "color_preparation": {
            "scope": "metadata_preflight",
            "inspected_images": len(preparations),
            "planned_profile_conversions": converted,
            "assumed_srgb_images": assumed,
            "explicit_profile_overrides": overrides,
        },
    }


def photography_config_preview(
    args: dict[str, Any], *, policy: ExecutionPolicy, tenant: TenantContext | None = None
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    preparations: list[dict[str, Any]] = []
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
        if not errors:
            color_errors, preparations = _photography_input_color_preflight(normalized, policy=policy, tenant=tenant)
            errors.extend(color_errors)
    if errors:
        normalized = {name: _json_safe_rejected_value(value) for name, value in normalized.items()}
    warnings, summary = _color_preparation_summary(preparations) if not errors else ([], {})
    return {
        "pipeline": "lux-depth-v5",
        "normalized_args": normalized,
        "execution_args": dict(normalized),
        "argv_preview": "",
        "field_errors": errors,
        "field_warnings": warnings,
        "inactive_fields": [],
        "readiness": managed_photography_readiness(),
        "estimate_summary": summary,
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
