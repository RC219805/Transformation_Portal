"""Shared legacy run summaries for managed execution and HTTP projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from transformation_portal.orchestrator.job_execution import ExecutionJob, execution_error
from transformation_portal.portal import job_artifacts as _artifacts
from transformation_portal.vlm_captioning.fastvlm_runtime import FASTVLM_CHECKPOINT_DIRS

_coerce_nonnegative_int = _artifacts._coerce_nonnegative_int
_captioning_artifact_counts_from_run_card = _artifacts._captioning_artifact_counts_from_run_card
_captioning_artifact_counts_from_job_artifacts = _artifacts._captioning_artifact_counts_from_job_artifacts
ALLOWED_VLM_CAPTIONING_MODEL_ROLES = frozenset(FASTVLM_CHECKPOINT_DIRS)
FASTVLM_RUN_STATUS_VALUES = {
    "off",
    "requested",
    "succeeded",
    "failed",
    "skipped",
    "missing_runtime",
    "invalid_config",
    "unsupported_backend",
}
FASTVLM_RUNTIME_STATUS_ALIASES = {
    "ok": "succeeded",
    "success": "succeeded",
    "successful": "succeeded",
    "succeeded": "succeeded",
    "error": "failed",
    "failed": "failed",
    "failure": "failed",
    "proxy_error": "failed",
    "timeout": "failed",
    "missing_model": "missing_runtime",
    "missing_runtime": "missing_runtime",
    "invalid_config": "invalid_config",
    "unsupported_backend": "unsupported_backend",
    "skipped": "skipped",
    "disabled": "off",
    "off": "off",
    "requested": "requested",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _fastvlm_model_role_from_value(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ALLOWED_VLM_CAPTIONING_MODEL_ROLES else "custom"


def _normalize_fastvlm_run_status(
    raw_status: Any,
    *,
    artifact_counts: Optional[Mapping[str, int]] = None,
    requested: bool = False,
) -> Optional[Dict[str, Any]]:
    counts = {
        "sidecar_count": _coerce_nonnegative_int((artifact_counts or {}).get("sidecar_count")) or 0,
        "raw_count": _coerce_nonnegative_int((artifact_counts or {}).get("raw_count")) or 0,
        "proxy_count": _coerce_nonnegative_int((artifact_counts or {}).get("proxy_count")) or 0,
    }
    if not isinstance(raw_status, Mapping):
        if not requested and not any(counts.values()):
            return None
        raw_status = {"enabled": True, "status": "requested" if requested else "succeeded"}

    enabled = _as_bool(raw_status.get("enabled"), requested or any(counts.values()))
    backend = str(raw_status.get("backend") or "fastvlm").strip().lower() or "fastvlm"
    status_text = str(raw_status.get("status") or "").strip().lower()
    normalized_status = FASTVLM_RUNTIME_STATUS_ALIASES.get(status_text, "")
    policy_violation = raw_status.get("used_for_quality_gate") is True

    sidecar_count = max(counts["sidecar_count"], _coerce_nonnegative_int(raw_status.get("sidecar_count")) or 0)
    raw_count = max(counts["raw_count"], _coerce_nonnegative_int(raw_status.get("raw_count")) or 0)
    proxy_count = max(counts["proxy_count"], _coerce_nonnegative_int(raw_status.get("proxy_count")) or 0)
    failed_count = _coerce_nonnegative_int(raw_status.get("failed_count")) or 0

    if policy_violation:
        normalized_status = "failed"
        failed_count = max(failed_count, 1)
    elif backend != "fastvlm":
        normalized_status = "unsupported_backend"
    elif normalized_status not in FASTVLM_RUN_STATUS_VALUES:
        if not enabled:
            normalized_status = "off"
        elif failed_count > 0:
            normalized_status = "failed"
        elif sidecar_count > 0:
            normalized_status = "succeeded"
        elif requested:
            normalized_status = "requested"
        else:
            normalized_status = "skipped"

    if normalized_status == "off":
        enabled = False
    if normalized_status == "succeeded" and sidecar_count == 0 and not requested:
        normalized_status = "skipped"

    normalized: Dict[str, Any] = {
        "status": normalized_status,
        "enabled": bool(enabled),
        "backend": backend,
        "model_role": str(
            raw_status.get("model_role") or _fastvlm_model_role_from_value(raw_status.get("model") or "")
        ).strip()
        or "custom",
        "model_id": raw_status.get("model_id") if raw_status.get("model_id") is not None else None,
        "model_path": str(raw_status.get("model_path") or "").strip() or None,
        "role": "advisory",
        "sidecar_count": sidecar_count,
        "raw_count": raw_count,
        "proxy_count": proxy_count,
        "failed_count": failed_count,
        "used_for_quality_gate": False,
    }
    if policy_violation:
        normalized["policy_violation"] = True
        normalized["quality_gate_claimed"] = True
        normalized["error"] = "captioning_status.used_for_quality_gate must be false"
    elif raw_status.get("error"):
        normalized["error"] = str(raw_status.get("error"))
    return normalized


def _summarize_run_card_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"source": "run_card"}

    batch_id = str(payload.get("batch_id") or "").strip()
    if batch_id:
        summary["batch_id"] = batch_id

    total_images = _coerce_nonnegative_int(payload.get("total_images"))
    success_count = _coerce_nonnegative_int(payload.get("success_count"))
    error_count = _coerce_nonnegative_int(payload.get("error_count"))
    artifact_index = payload.get("artifact_index")
    artifact_index_count = len(artifact_index) if isinstance(artifact_index, list) else None

    if total_images is None and success_count is not None and error_count is not None:
        total_images = success_count + error_count

    if total_images is not None:
        summary["total_images"] = total_images
    if success_count is not None:
        summary["success_count"] = success_count
    if error_count is not None:
        summary["error_count"] = error_count
    if artifact_index_count is not None:
        summary["artifact_index_count"] = artifact_index_count

    reviewable_outputs = bool((success_count or 0) > 0)
    partial = reviewable_outputs and bool((error_count or 0) > 0)
    summary["reviewable_outputs"] = reviewable_outputs
    summary["partial"] = partial
    captioning_status = _normalize_fastvlm_run_status(
        payload.get("captioning_status"),
        artifact_counts=_captioning_artifact_counts_from_run_card(payload),
    )
    if captioning_status is not None:
        summary["captioning_status"] = captioning_status

    return summary


def _summarize_batch_manifest_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"source": "batch_manifest"}

    batch_id = str(payload.get("batch_id") or "").strip()
    if batch_id:
        summary["batch_id"] = batch_id

    results = payload.get("results")
    if isinstance(results, list):
        success_count = sum(1 for item in results if isinstance(item, dict) and item.get("status") == "ok")
        error_count = sum(1 for item in results if isinstance(item, dict) and item.get("status") == "error")
    else:
        success_count = 0
        error_count = 0

    stats = payload.get("stats")
    total_images = None
    if isinstance(stats, Mapping):
        total_images = _coerce_nonnegative_int(stats.get("total_images"))
    if total_images is None and isinstance(results, list):
        total_images = len(results)

    if total_images is not None:
        summary["total_images"] = total_images
    summary["success_count"] = success_count
    summary["error_count"] = error_count
    summary["reviewable_outputs"] = success_count > 0
    summary["partial"] = success_count > 0 and error_count > 0

    return summary


def refresh_job_run_summary(job: ExecutionJob, output_dir: Path | None) -> Dict[str, Any]:
    if not job:
        return {}

    existing_summary = dict(job.run_summary) if isinstance(job.run_summary, dict) and job.run_summary else {}
    metadata = _artifacts._resolve_job_run_metadata(output_dir, max_bytes=1024 * 1024)
    summary: Dict[str, Any] = {}
    if metadata is not None and metadata.run_card_payload is not None:
        summary = _summarize_run_card_payload(metadata.run_card_payload)

    if not summary and metadata is not None and metadata.batch_manifest_payload is not None:
        summary = _summarize_batch_manifest_payload(metadata.batch_manifest_payload)

    if not summary and existing_summary:
        summary = existing_summary

    if summary:
        artifact_counts = _captioning_artifact_counts_from_job_artifacts(job.artifacts)
        existing_captioning_counts = summary.get("captioning_status")
        if isinstance(existing_captioning_counts, Mapping):
            artifact_counts = {
                "sidecar_count": max(
                    artifact_counts["sidecar_count"],
                    _coerce_nonnegative_int(existing_captioning_counts.get("sidecar_count")) or 0,
                ),
                "raw_count": max(
                    artifact_counts["raw_count"],
                    _coerce_nonnegative_int(existing_captioning_counts.get("raw_count")) or 0,
                ),
                "proxy_count": max(
                    artifact_counts["proxy_count"],
                    _coerce_nonnegative_int(existing_captioning_counts.get("proxy_count")) or 0,
                ),
            }
        raw_captioning_status = None
        if metadata is not None and metadata.run_card_payload is not None:
            raw_captioning_status = metadata.run_card_payload.get("captioning_status")
        if raw_captioning_status is None:
            raw_captioning_status = existing_summary.get("captioning_status")
        captioning_status = _normalize_fastvlm_run_status(
            raw_captioning_status,
            artifact_counts=artifact_counts,
        )
        if captioning_status is not None:
            summary["captioning_status"] = captioning_status

    job.run_summary = summary

    if job.state != "canceled" and summary.get("partial"):
        job.state = "partial"
        existing_code = ""
        if isinstance(job.error, dict):
            existing_code = str(job.error.get("code") or "").strip().upper()
        if existing_code in {"", "RUNNER_EXIT_NONZERO"}:
            total_images = summary.get("total_images")
            success_count = summary.get("success_count")
            error_count = summary.get("error_count")
            detail_text = "outputs remain reviewable"
            if (
                isinstance(total_images, int)
                and isinstance(success_count, int)
                and isinstance(error_count, int)
                and total_images > 0
            ):
                detail_text = f"{error_count}/{total_images} images failed; " f"{success_count} outputs remain reviewable"
            job.error = execution_error(
                "RUNNER_PARTIAL_FAILURE",
                detail_text,
                {
                    "exit_code": job.exit_code,
                    "total_images": total_images,
                    "success_count": success_count,
                    "error_count": error_count,
                },
            )

    return summary
