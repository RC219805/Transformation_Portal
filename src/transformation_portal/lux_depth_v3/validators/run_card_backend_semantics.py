"""Shared backend semantic validation for Lux run cards."""

from __future__ import annotations

from typing import Any


def collect_run_card_backend_semantic_errors(payload: dict[str, Any]) -> list[str]:
    """Return backend-selection semantic errors for a run-card payload.

    This helper is intentionally artifact-agnostic. Integrity verification may
    enrich these errors with artifact-derived diagnostics, but the pass/fail
    semantics live here.
    """
    backend_selection = payload.get("backend_selection")
    backend_summary = payload.get("backend_summary")
    if not isinstance(backend_selection, dict) or not isinstance(backend_summary, dict):
        return []

    errors: list[str] = []
    final_backends_used = backend_summary.get("final_backends_used")
    if not isinstance(final_backends_used, list):
        return ["backend_summary.final_backends_used must be an array"]

    success_count = payload.get("success_count")
    if not isinstance(success_count, int):
        success_count = 0

    if not final_backends_used:
        if success_count > 0:
            errors.append("backend_summary.final_backends_used must be non-empty when success_count > 0")
        return errors

    primary_backend = final_backends_used[0]
    if not isinstance(primary_backend, str) or not primary_backend:
        errors.append("backend_summary.final_backends_used[0] must be a non-empty string")
        return errors

    summary_primary = backend_summary.get("primary_backend")
    if summary_primary != primary_backend:
        errors.append("backend_summary.primary_backend must equal backend_summary.final_backends_used[0]")

    resolved = backend_selection.get("resolved")
    if not isinstance(resolved, str) or not resolved:
        errors.append("backend_selection.resolved must be a non-empty string")
        return errors
    if resolved != primary_backend:
        errors.append("backend_selection.resolved must match backend_summary.final_backends_used[0]")

    requested_backend = backend_selection.get("requested") or backend_summary.get("requested_backend")
    fallback_images = backend_summary.get("fallback_images")
    requested_backend_defect = backend_summary.get("requested_backend_defect")
    requested_backend_status = backend_summary.get("requested_backend_status")
    explicit_requested_backend_defect = (
        requested_backend_status == "not_honored"
        and isinstance(requested_backend_defect, str)
        and bool(requested_backend_defect.strip())
    )
    total_images = payload.get("total_images")
    error_count = payload.get("error_count")
    run_failed = (
        isinstance(error_count, int) and error_count > 0 or isinstance(total_images, int) and success_count < total_images
    )
    if (
        requested_backend == "depth_pro"
        and isinstance(fallback_images, int)
        and success_count > 0
        and fallback_images == success_count
        and primary_backend != requested_backend
        and not run_failed
        and not explicit_requested_backend_defect
    ):
        errors.append(
            "requested backend 'depth_pro' was not honored: "
            f"all successful images ({success_count}/{success_count}) used fallback backend '{primary_backend}'"
        )

    logical_backend = backend_selection.get("logical_backend")
    resolved_engine = backend_selection.get("resolved_engine")
    wrapper_declared = logical_backend is not None or resolved_engine is not None
    if not wrapper_declared:
        return errors

    if not isinstance(logical_backend, str) or not logical_backend:
        errors.append("backend_selection.logical_backend must be a non-empty string when wrapper semantics are declared")
    if not isinstance(resolved_engine, str) or not resolved_engine:
        errors.append("backend_selection.resolved_engine must be a non-empty string when wrapper semantics are declared")
    if isinstance(logical_backend, str) and isinstance(resolved_engine, str):
        if logical_backend == resolved_engine:
            errors.append("backend_selection.logical_backend and backend_selection.resolved_engine must differ")
        if resolved_engine != primary_backend:
            errors.append("backend_selection.resolved_engine must match backend_summary.final_backends_used[0]")

    if isinstance(fallback_images, int) and fallback_images != 0:
        errors.append("wrapper semantics are only valid when backend_summary.fallback_images == 0")

    return errors
