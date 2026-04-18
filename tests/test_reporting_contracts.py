"""Unit tests for shared reporting contract helpers."""

from __future__ import annotations

import pytest

from transformation_portal.reporting.contracts import (
    build_capability_report,
    build_orchestrator_result_capability_report,
    build_quality_gate_report,
    build_stage_report,
    derive_stage_report_map,
)

pytestmark = pytest.mark.unit


def test_build_quality_gate_report_projects_expected_shape() -> None:
    report = build_quality_gate_report(
        {
            "passed": False,
            "failure_codes": ["APEX_DEPTH_PLATEAU"],
            "warnings": ["APEX_DEPTH_GRADIENT_LOW"],
            "metrics": {"upper_iqr": 0.0},
            "thresholds": {"upper_iqr_min": 1e-4},
            "shape_context": {"native_shape": [64, 64]},
            "demoted_failure_codes": ["APEX_DEPTH_SATURATION_LOW"],
        }
    )

    assert report == {
        "kind": "apex_depth",
        "passed": False,
        "failure_codes": ["APEX_DEPTH_PLATEAU"],
        "warnings": ["APEX_DEPTH_GRADIENT_LOW"],
        "details": {
            "metrics": {"upper_iqr": 0.0},
            "thresholds": {"upper_iqr_min": 1e-4},
            "shape_context": {"native_shape": [64, 64]},
            "demoted_failure_codes": ["APEX_DEPTH_SATURATION_LOW"],
        },
    }


def test_build_orchestrator_result_capability_report_uses_selected_attempt_and_fallback_reason() -> None:
    report = build_orchestrator_result_capability_report(
        {
            "status": "ok",
            "backend": "depth_pro",
            "fallback_used": True,
            "selected_attempt_index": 1,
            "attempts": [
                {"backend": "da3", "status": "failed", "error_code": "BACKEND_RUNTIME_ERROR"},
                {"backend": "depth_pro", "status": "success", "model_id": "apple/ml-depth-pro", "model_revision": "rev-123"},
            ],
        },
        requested_backend="da3",
        resolution_reason="Fallback from da3 to depth_pro after startup failure",
    )

    assert report == {
        "requested_backend": "da3",
        "executed_backend": "depth_pro",
        "availability_state": "fallback_executed",
        "reason": "Fallback from da3 to depth_pro after startup failure",
        "synthetic_output": False,
        "stub_mode": False,
        "fallback_executed": True,
        "model_repo_id": "apple/ml-depth-pro",
        "model_revision": "rev-123",
        "asset_bundle_version": None,
    }


def test_stage_report_helpers_preserve_last_stage_entry() -> None:
    capability = build_capability_report(
        requested_backend="depth_anything_v2",
        executed_backend="synthetic",
        availability_state="synthetic_opt_in",
        synthetic_output=True,
    )
    reports = [
        build_stage_report(stage="depth", status="skipped"),
        build_stage_report(stage="depth", status="completed", capability=capability),
    ]

    report_map = derive_stage_report_map(reports)

    assert report_map["depth"]["status"] == "completed"
    assert report_map["depth"]["capability"]["synthetic_output"] is True
