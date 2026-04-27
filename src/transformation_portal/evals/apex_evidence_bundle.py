"""APEX canonical evidence bundle helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformation_portal.evals.apex_metrics import (
    METRIC_STATUS_DIMENSION_MISMATCH,
    METRIC_STATUS_INVALID_INPUT,
    METRIC_STATUS_UNSUPPORTED_BIT_DEPTH,
    METRIC_STATUSES,
)
from transformation_portal.ingest.canonical_json import dump_json

APEX_EVIDENCE_BUNDLE_VERSION = "apex_evidence_bundle.v1"
APEX_METRIC_CONTRACT_VERSION = "apex_metrics.v1"
APEX_MATERIALS_PIXEL_OPS_EMPTY = "APEX_MATERIALS_PIXEL_OPS_EMPTY"
APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE = "APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE"
_PROMOTION_BLOCKING_METRIC_STATUSES = frozenset(
    {
        METRIC_STATUS_INVALID_INPUT,
        METRIC_STATUS_UNSUPPORTED_BIT_DEPTH,
        METRIC_STATUS_DIMENSION_MISMATCH,
    }
)


def parse_candidate_evidence(values: Iterable[str]) -> dict[str, dict[str, Path]]:
    """Parse ``candidate:asset_id=path`` evidence JSON mappings."""
    parsed: dict[str, dict[str, Path]] = {}
    for value in values:
        candidate_part, sep, evidence_path = value.partition("=")
        if sep != "=":
            raise ValueError(f"Invalid candidate evidence {value!r}; expected candidate:asset_id=path")
        candidate, sep, asset_id = candidate_part.partition(":")
        if sep != ":" or not candidate or not asset_id:
            raise ValueError(f"Invalid candidate evidence {value!r}; expected candidate:asset_id=path")
        parsed.setdefault(candidate, {})[asset_id] = Path(evidence_path)
    return parsed


def _load_evidence(path: Path, *, repo_root: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else repo_root / path
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Candidate evidence must be a JSON object: {path}")
    return payload


def _materials_status(candidate: str, evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {
            "status": "missing_evidence",
            "apex_pixel_ops_authority": False,
        }
    authority = dict(evidence.get("confidence_authority") or {})
    raw_authorized = bool(authority.get("raw_clip_similarity_authorized_pixel_ops"))
    applied_ops_count = int(evidence.get("applied_ops_count") or 0)
    blocked_reason_counts = dict(evidence.get("blocked_reason_counts") or {})

    # Soft-passthrough: the orchestrator records this when masks exist and every
    # implemented op was blocked solely by below_confidence_threshold, emitting
    # the output without pixel ops. Producers of the per-candidate evidence
    # mirror that signal here as ``passthrough_status`` so promotion isn't blocked
    # by an ``applied_ops_count == 0`` that the orchestrator already accepted.
    raw_passthrough = evidence.get("passthrough_status")
    passthrough_status: dict[str, Any] | None = None
    if isinstance(raw_passthrough, Mapping):
        passthrough_status = dict(raw_passthrough)
    passthrough_active = (
        passthrough_status is not None
        and str(passthrough_status.get("code") or "") == APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE
    )

    status = "ok"
    failure_code = None
    if (
        candidate == "materials_v3"
        and bool(evidence.get("materials_v3_enabled"))
        and bool(evidence.get("pixel_ops_enabled"))
        and bool(evidence.get("masks_exist"))
        and bool(evidence.get("implemented_ops_exist"))
        and applied_ops_count == 0
        and not passthrough_active
    ):
        status = "failed"
        failure_code = APEX_MATERIALS_PIXEL_OPS_EMPTY
    return {
        "status": status,
        "materials_v3_enabled": bool(evidence.get("materials_v3_enabled")),
        "pixel_ops_enabled": bool(evidence.get("pixel_ops_enabled")),
        "masks_exist": bool(evidence.get("masks_exist")),
        "implemented_ops_exist": bool(evidence.get("implemented_ops_exist")),
        "applied_ops_count": applied_ops_count,
        "blocked_reason_counts": blocked_reason_counts,
        "confidence_authority": authority,
        "apex_pixel_ops_authority": not raw_authorized and status == "ok",
        "failure_code": failure_code,
        "passthrough_status": passthrough_status,
    }


def _metrics_valid(candidate: Mapping[str, Any]) -> bool:
    status = candidate.get("status")
    if status in {
        "missing_candidate_output",
        "missing_candidate",
        "missing_reference",
        "source_not_ready",
        "shape_mismatch",
        "dimension_mismatch",
        "invalid_candidate_dimensions",
        "unreadable_image",
        "unreadable_reference",
        "unreadable_candidate",
        "unsupported_reference_bit_depth",
        "unsupported_candidate_bit_depth",
        "metrics_not_computed",
    }:
        return False
    if status != "ok":
        return False
    if candidate.get("metric_contract") != APEX_METRIC_CONTRACT_VERSION:
        return False
    if candidate.get("metrics_authoritative") is not True:
        return False
    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    if not metrics:
        return False
    for value in metrics.values():
        if not isinstance(value, Mapping):
            return False
        metric_status = value.get("status")
        if metric_status not in METRIC_STATUSES:
            return False
        if metric_status in _PROMOTION_BLOCKING_METRIC_STATUSES:
            return False
    return True


def evaluate_apex_promotion_eligibility(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate run-level APEX promotion eligibility from an evidence bundle."""
    run = dict(report.get("run") or {})
    cases = list(report.get("cases") or [])
    scope_ids = set(run.get("run_scope_asset_ids") or [])
    blocked: list[str] = []
    if run.get("synthetic_data") is True:
        blocked.append("synthetic_data")

    canonical_cases = [
        case
        for case in cases
        if case.get("canonical_scoring_eligible") is True and (not scope_ids or case.get("asset_id") in scope_ids)
    ]
    if not canonical_cases:
        blocked.append("zero_canonical_eligible_assets")

    for case in canonical_cases:
        if case.get("candidate_output", {}).get("status") != "present":
            blocked.append("missing_candidate_output")
        if case.get("metrics_status") != "valid":
            blocked.append("invalid_metrics")
        materials = dict(case.get("materials_v3") or {})
        if case.get("candidate_id") == "materials_v3" and materials.get("status") == "missing_evidence":
            blocked.append("missing_materials_v3_evidence")
        if materials.get("failure_code") == APEX_MATERIALS_PIXEL_OPS_EMPTY:
            blocked.append(APEX_MATERIALS_PIXEL_OPS_EMPTY)
        authority = dict(materials.get("confidence_authority") or {})
        if authority.get("raw_clip_similarity_authorized_pixel_ops") is True:
            blocked.append("raw_clip_similarity_authorized_pixel_ops")

    unique_blocked = sorted(set(blocked))
    return {
        "promotion_verdict": "eligible" if not unique_blocked else "blocked",
        "promotion_blocked_reasons": unique_blocked,
    }


def build_apex_evidence_bundle(
    apex_report: Mapping[str, Any],
    *,
    output_dir: Path | str,
    candidate_evidence: Mapping[str, Mapping[str, Path]] | None = None,
    run_scope_asset_ids: Iterable[str] | None = None,
    synthetic_data: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build and persist an APEX evidence bundle from an eval report."""
    candidate_evidence = candidate_evidence or {}
    report_repo_root = (
        apex_report.get("evalset", {}).get("repo_root") if isinstance(apex_report.get("evalset"), Mapping) else None
    )
    repo = repo_root or (Path(str(report_repo_root)) if report_repo_root else Path.cwd())
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        report_path_value = apex_report.get("report_path")
        report_parent = Path(str(report_path_value)).parent if report_path_value else None
        output_root = (report_parent if report_parent and report_parent.is_absolute() else repo) / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    scope = list(run_scope_asset_ids or [])
    for asset in apex_report.get("assets", []):
        candidates = asset.get("candidates") or [{"candidate": "none", "status": "not_evaluated", "metrics": {}}]
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate") or "none")
            evidence_path = candidate_evidence.get(candidate_id, {}).get(str(asset.get("asset_id")))
            evidence_payload = _load_evidence(evidence_path, repo_root=repo) if evidence_path is not None else None
            materials = _materials_status(candidate_id, evidence_payload)
            candidate_output = candidate.get("candidate_output")
            candidate_output_path = None
            if isinstance(candidate_output, Mapping):
                candidate_output_path = candidate_output.get("path")
            candidate_output_path = candidate_output_path or candidate.get("output_path")
            candidate_status = candidate.get("status")
            candidate_output_status = (
                "present"
                if candidate_output_path and candidate_status not in {"missing_candidate", "missing_candidate_output"}
                else "missing"
            )
            metrics_valid = _metrics_valid(candidate)
            case_verdict = (
                "pass"
                if candidate_output_status == "present" and metrics_valid and materials.get("status") != "failed"
                else "fail"
            )
            cases.append(
                {
                    "asset_id": asset.get("asset_id"),
                    "candidate_id": candidate_id,
                    "canonical_scoring_eligible": bool(asset.get("canonical_scoring_eligible")),
                    "reference": {
                        "path": asset.get("reference_path"),
                        "bit_depth": asset.get("reference_bit_depth"),
                        "format": asset.get("reference_format"),
                    },
                    "model_input": {
                        "allow_downsampled_model_inference": asset.get("allow_downsampled_model_inference"),
                    },
                    "evaluation_target": {
                        "path": candidate.get("evaluation_target_path") or asset.get("reference_path"),
                        "evaluate_at_native_resolution": asset.get("evaluate_at_native_resolution"),
                        "preserve_16bit_intermediates": asset.get("preserve_16bit_intermediates"),
                        "reference_resolution": candidate.get("reference_resolution"),
                    },
                    "candidate_output": {
                        "status": candidate_output_status,
                        "path": candidate_output_path,
                    },
                    "materials_v3": materials,
                    "mask_evidence": candidate.get("mask_evidence") or {"status": "not_supplied"},
                    "depth": {},
                    "metrics": candidate.get("metrics") or {},
                    "metrics_status": "valid" if metrics_valid else "invalid",
                    "case_verdict": case_verdict,
                }
            )

    report_path = output_root / "evidence_bundle.json"
    bundle: dict[str, Any] = {
        "schema_version": APEX_EVIDENCE_BUNDLE_VERSION,
        "report_path": str(report_path),
        "run": {
            "quality_tier": "apex",
            "synthetic_data": bool(synthetic_data),
            "run_scope_asset_ids": scope,
        },
        "corpus": {
            "dataset_id": apex_report.get("evalset", {}).get("evalset_id"),
            "canonical_scoring_eligible_count": apex_report.get("evalset", {}).get("canonical_scoring_eligible_count"),
        },
        "cases": cases,
    }
    bundle.update(evaluate_apex_promotion_eligibility(bundle))
    with report_path.open("w", encoding="utf-8") as handle:
        dump_json(bundle, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return bundle
