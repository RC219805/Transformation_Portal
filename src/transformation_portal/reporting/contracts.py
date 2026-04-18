"""Small shared reporting builders for capability, quality-gate, and stage status payloads."""

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

    details = {
        "metrics": _copy_mapping(gate_payload.get("metrics")) or {},
        "thresholds": _copy_mapping(gate_payload.get("thresholds")) or {},
        "shape_context": _copy_mapping(gate_payload.get("shape_context")) or {},
    }
    demoted_failure_codes = gate_payload.get("demoted_failure_codes")
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


def build_orchestrator_result_capability_report(
    result: Mapping[str, Any],
    *,
    requested_backend: Any = None,
    resolution_reason: Any = None,
) -> dict[str, Any]:
    """Build the normalized capability record for an orchestrator result row."""
    selected_attempt = select_result_attempt(result)
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

    if status == "skipped":
        availability_state = "skipped"
        reason = result.get("reason")
    elif status == "error":
        availability_state = "failed"
        reason = result.get("error_code") or result.get("error")
    elif fallback_executed:
        availability_state = "fallback_executed"
        reason = resolution_reason or None
        if not reason:
            attempts = result.get("attempts")
            if isinstance(attempts, list):
                for attempt in attempts:
                    if not isinstance(attempt, Mapping):
                        continue
                    if attempt.get("status") == "failed":
                        reason = attempt.get("error_message") or attempt.get("error_code")
                        if reason:
                            break
    elif executed_backend:
        availability_state = "available"
        reason = None
    else:
        availability_state = "unknown"
        reason = None

    model_repo_id = result.get("model_id")
    if not model_repo_id and isinstance(selected_attempt, Mapping):
        model_repo_id = selected_attempt.get("model_id")

    model_revision = selected_attempt.get("model_revision") if isinstance(selected_attempt, Mapping) else None
    asset_bundle_version = selected_attempt.get("asset_bundle_version") if isinstance(selected_attempt, Mapping) else None

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
) -> dict[str, Any]:
    """Build a normalized stage-report payload."""
    return {
        "stage": stage,
        "status": status,
        "capability": _copy_mapping(capability),
        "quality_gate": _copy_mapping(quality_gate),
    }


def derive_stage_report_map(stage_reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index stage reports by stage name, preserving the last report per stage."""
    report_map: dict[str, dict[str, Any]] = {}
    for report in stage_reports:
        stage_name = report.get("stage")
        if isinstance(stage_name, str) and stage_name:
            report_map[stage_name] = copy.deepcopy(report)
    return report_map
