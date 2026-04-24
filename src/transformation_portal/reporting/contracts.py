"""Small shared reporting builders for capability, quality-gate, and stage status payloads.

Capability describes backend availability/execution truth. Quality-gate describes
output-validity truth. Result-row status captures the final artifact outcome.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional


def _copy_mapping(value: Any) -> Optional[dict[str, Any]]:
    """Return a deep-copied dict when the input is mapping-like."""
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return None


def _normalize_optional_string(value: Any) -> Optional[str]:
    """Normalize values to stripped strings or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_quality_gate_report(
    gate_payload: Any,
    *,
    default_kind: str = "apex_depth",
) -> Optional[dict[str, Any]]:
    """Project gate payloads into a stable serialized quality-gate shape."""
    if not isinstance(gate_payload, Mapping):
        return None

    existing_details = gate_payload.get("details")
    if isinstance(existing_details, Mapping):
        metrics = _copy_mapping(existing_details.get("metrics")) or {}
        thresholds = _copy_mapping(existing_details.get("thresholds")) or {}
        shape_context = _copy_mapping(existing_details.get("shape_context")) or {}
        demoted_failure_codes = existing_details.get("demoted_failure_codes")
    else:
        metrics = _copy_mapping(gate_payload.get("metrics")) or {}
        thresholds = _copy_mapping(gate_payload.get("thresholds")) or {}
        shape_context = _copy_mapping(gate_payload.get("shape_context")) or {}
        demoted_failure_codes = gate_payload.get("demoted_failure_codes")

    details = {
        "metrics": metrics,
        "thresholds": thresholds,
        "shape_context": shape_context,
    }
    if demoted_failure_codes not in (None, [], {}):
        details["demoted_failure_codes"] = copy.deepcopy(demoted_failure_codes)

    return {
        "kind": _normalize_optional_string(gate_payload.get("kind")) or default_kind,
        "passed": bool(gate_payload.get("passed", False)),
        "failure_codes": list(gate_payload.get("failure_codes") or []),
        "warnings": list(gate_payload.get("warnings") or []),
        "details": details,
    }


def build_capability_report(
    *,
    requested_backend: Any,
    executed_backend: Any,
    availability_state: Any,
    reason: Any = None,
    synthetic_output: bool = False,
    stub_mode: bool = False,
    fallback_executed: bool = False,
    model_repo_id: Any = None,
    model_revision: Any = None,
    asset_bundle_version: Any = None,
) -> dict[str, Any]:
    """Build a normalized backend capability report."""
    return {
        "requested_backend": _normalize_optional_string(requested_backend),
        "executed_backend": _normalize_optional_string(executed_backend),
        "availability_state": _normalize_optional_string(availability_state),
        "reason": _normalize_optional_string(reason),
        "synthetic_output": bool(synthetic_output),
        "stub_mode": bool(stub_mode),
        "fallback_executed": bool(fallback_executed),
        "model_repo_id": _normalize_optional_string(model_repo_id),
        "model_revision": _normalize_optional_string(model_revision),
        "asset_bundle_version": _normalize_optional_string(asset_bundle_version),
    }


def select_result_attempt(result: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Return the selected attempt for a result row when available."""
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None

    selected_attempt_index = result.get("selected_attempt_index")
    if isinstance(selected_attempt_index, int) and 0 <= selected_attempt_index < len(attempts):
        attempt = attempts[selected_attempt_index]
        if isinstance(attempt, Mapping):
            return copy.deepcopy(dict(attempt))

    backend = result.get("backend")
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        if backend and attempt.get("backend") != backend:
            continue
        if attempt.get("status") == "success":
            return copy.deepcopy(dict(attempt))

    for attempt in attempts:
        if isinstance(attempt, Mapping):
            return copy.deepcopy(dict(attempt))

    return None


def _list_result_attempts(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deep-copied attempt records for rows with structured history."""
    attempts = result.get("attempts")
    if not isinstance(attempts, list):
        return []
    return [copy.deepcopy(dict(attempt)) for attempt in attempts if isinstance(attempt, Mapping)]


def _find_failed_attempt_for_backend(
    attempts: list[dict[str, Any]],
    executed_backend: Optional[str],
) -> Optional[dict[str, Any]]:
    """Return the most relevant failed attempt for the executed backend."""
    if executed_backend:
        for attempt in reversed(attempts):
            if attempt.get("status") == "failed" and attempt.get("backend") == executed_backend:
                return attempt
    for attempt in reversed(attempts):
        if attempt.get("status") == "failed":
            return attempt
    return None


def _is_semantic_gate_failure(
    failed_attempt: Optional[dict[str, Any]],
    gate_report: Optional[Mapping[str, Any]],
) -> bool:
    """Return True when a failed row reflects output-quality rejection, not backend loss."""
    if isinstance(failed_attempt, Mapping) and failed_attempt.get("failure_kind") == "semantic":
        return True
    if isinstance(gate_report, Mapping) and gate_report.get("passed") is False and isinstance(failed_attempt, Mapping):
        return failed_attempt.get("failure_kind") == "semantic"
    return False


def _resolve_fallback_reason(
    attempts: list[dict[str, Any]],
    resolution_reason: Any,
) -> Optional[str]:
    """Return the best available explanation for backend fallback execution."""
    normalized_reason = _normalize_optional_string(resolution_reason)
    if normalized_reason:
        return normalized_reason

    for attempt in attempts:
        if attempt.get("status") != "failed":
            continue
        if attempt.get("failure_kind") not in {"operational", "license"}:
            continue
        reason = _normalize_optional_string(attempt.get("error_message")) or _normalize_optional_string(
            attempt.get("error_code")
        )
        if reason:
            return reason

    for attempt in attempts:
        if attempt.get("status") != "failed":
            continue
        reason = _normalize_optional_string(attempt.get("error_message")) or _normalize_optional_string(
            attempt.get("error_code")
        )
        if reason:
            return reason

    return None


def _select_capability_metadata_attempt(
    selected_attempt: Optional[dict[str, Any]],
    failed_attempt: Optional[dict[str, Any]],
    executed_backend: Optional[str],
) -> Optional[dict[str, Any]]:
    """Return the attempt that best represents executed-backend provenance."""
    if isinstance(selected_attempt, Mapping) and (
        executed_backend is None or selected_attempt.get("backend") == executed_backend
    ):
        return selected_attempt
    if isinstance(failed_attempt, Mapping):
        return failed_attempt
    return selected_attempt


def build_orchestrator_result_capability_report(
    result: Mapping[str, Any],
    *,
    requested_backend: Any = None,
    resolution_reason: Any = None,
) -> dict[str, Any]:
    """Build the normalized capability record for an orchestrator result row.

    Capability describes backend truth. Quality-gate and row status carry the
    separate question of whether the produced output was ultimately accepted.
    """
    selected_attempt = select_result_attempt(result)
    attempts = _list_result_attempts(result)
    requested = (
        result.get("requested_backend")
        or requested_backend
        or result.get("backend")
        or (selected_attempt.get("backend") if isinstance(selected_attempt, Mapping) else None)
        or "auto"
    )
    executed_backend = result.get("backend") or (
        selected_attempt.get("backend") if isinstance(selected_attempt, Mapping) else None
    )
    fallback_executed = bool(result.get("fallback_used"))
    status = result.get("status")
    gate_report = resolve_result_quality_gate(result)
    failed_attempt = _find_failed_attempt_for_backend(attempts, executed_backend)
    semantic_gate_failure = _is_semantic_gate_failure(failed_attempt, gate_report)
    selected_attempt_succeeded = isinstance(selected_attempt, Mapping) and selected_attempt.get("status") == "success"
    metadata_attempt = _select_capability_metadata_attempt(
        selected_attempt,
        failed_attempt,
        _normalize_optional_string(executed_backend),
    )

    if status == "skipped":
        availability_state = "skipped"
        reason = result.get("reason")
    elif fallback_executed and (selected_attempt_succeeded or semantic_gate_failure):
        availability_state = "fallback_executed"
        reason = _resolve_fallback_reason(attempts, resolution_reason)
    elif selected_attempt_succeeded or semantic_gate_failure:
        availability_state = "available"
        reason = None
    elif status == "error":
        availability_state = "failed"
        reason = (
            (_normalize_optional_string(failed_attempt.get("error_message")) if isinstance(failed_attempt, Mapping) else None)
            or (_normalize_optional_string(failed_attempt.get("error_code")) if isinstance(failed_attempt, Mapping) else None)
            or _normalize_optional_string(result.get("error_code"))
            or _normalize_optional_string(result.get("error"))
        )
    elif executed_backend:
        availability_state = "available"
        reason = None
    else:
        availability_state = "unknown"
        reason = None

    model_repo_id = result.get("model_id")
    if not model_repo_id and isinstance(metadata_attempt, Mapping):
        model_repo_id = metadata_attempt.get("model_id")

    model_revision = metadata_attempt.get("model_revision") if isinstance(metadata_attempt, Mapping) else None
    asset_bundle_version = metadata_attempt.get("asset_bundle_version") if isinstance(metadata_attempt, Mapping) else None

    return build_capability_report(
        requested_backend=requested,
        executed_backend=executed_backend,
        availability_state=availability_state,
        reason=reason,
        synthetic_output=False,
        stub_mode=False,
        fallback_executed=fallback_executed,
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        asset_bundle_version=asset_bundle_version,
    )


def resolve_result_quality_gate(
    result: Mapping[str, Any],
    *,
    default_kind: str = "apex_depth",
) -> Optional[dict[str, Any]]:
    """Resolve the normalized quality gate for a result row."""
    normalized = build_quality_gate_report(result.get("quality_gate"), default_kind=default_kind)
    if normalized is not None:
        return normalized
    return build_quality_gate_report(result.get("error_details"), default_kind=default_kind)


def build_stage_report(
    *,
    stage: str,
    status: str,
    capability: Optional[Mapping[str, Any]] = None,
    quality_gate: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a normalized stage-report payload."""
    report = {
        "stage": stage,
        "status": status,
        "capability": _copy_mapping(capability),
        "quality_gate": _copy_mapping(quality_gate),
    }
    metadata_copy = _copy_mapping(metadata)
    if metadata_copy is not None:
        report["metadata"] = metadata_copy
    return report


def derive_stage_report_map(stage_reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index stage reports by stage name, preserving the last report per stage."""
    report_map: dict[str, dict[str, Any]] = {}
    for report in stage_reports:
        stage_name = report.get("stage")
        if isinstance(stage_name, str) and stage_name:
            report_map[stage_name] = copy.deepcopy(report)
    return report_map
