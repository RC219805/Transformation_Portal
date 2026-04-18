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


def test_build_orchestrator_result_capability_report_keeps_semantic_gate_failure_available() -> None:
    report = build_orchestrator_result_capability_report(
        {
            "status": "error",
            "backend": "da3",
            "fallback_used": False,
            "error_code": "APEX_DEPTH_SATURATION_LOW",
            "quality_gate": {
                "passed": False,
                "failure_codes": ["APEX_DEPTH_SATURATION_LOW"],
                "warnings": [],
                "metrics": {"saturation_low_fraction": 0.031},
                "thresholds": {"saturation_low_fraction_max": 0.02},
                "shape_context": {"native_shape": [64, 64]},
            },
            "attempts": [
                {
                    "backend": "da3",
                    "status": "failed",
                    "failure_kind": "semantic",
                    "error_code": "APEX_DEPTH_SATURATION_LOW",
                    "error_message": "APEX depth validity gate failed: APEX_DEPTH_SATURATION_LOW",
                    "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
                    "model_revision": "rev-semantic",
                }
            ],
        },
        requested_backend="da3",
    )

    assert report == {
        "requested_backend": "da3",
        "executed_backend": "da3",
        "availability_state": "available",
        "reason": None,
        "synthetic_output": False,
        "stub_mode": False,
        "fallback_executed": False,
        "model_repo_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "model_revision": "rev-semantic",
        "asset_bundle_version": None,
    }


def test_build_orchestrator_result_capability_report_preserves_fallback_for_semantic_failure() -> None:
    report = build_orchestrator_result_capability_report(
        {
            "status": "error",
            "backend": "depth_pro",
            "fallback_used": True,
            "error_code": "APEX_DEPTH_PLATEAU",
            "quality_gate": {
                "passed": False,
                "failure_codes": ["APEX_DEPTH_PLATEAU"],
                "warnings": [],
                "metrics": {"upper_iqr": 0.0},
                "thresholds": {"upper_iqr_min": 1e-4},
                "shape_context": {"native_shape": [64, 64]},
            },
            "attempts": [
                {
                    "backend": "da3",
                    "status": "failed",
                    "failure_kind": "operational",
                    "error_code": "BACKEND_RUNTIME_ERROR",
                    "error_message": "da3 runtime crashed",
                    "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
                    "model_revision": "rev-da3",
                },
                {
                    "backend": "depth_pro",
                    "status": "failed",
                    "failure_kind": "semantic",
                    "error_code": "APEX_DEPTH_PLATEAU",
                    "error_message": "APEX depth validity gate failed: APEX_DEPTH_PLATEAU",
                    "model_id": "apple/ml-depth-pro",
                    "model_revision": "rev-depth-pro",
                },
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
        "model_revision": "rev-depth-pro",
        "asset_bundle_version": None,
    }


def test_build_orchestrator_result_capability_report_marks_operational_failure_failed() -> None:
    report = build_orchestrator_result_capability_report(
        {
            "status": "error",
            "backend": "da3",
            "fallback_used": False,
            "error_code": "BACKEND_RUNTIME_ERROR",
            "error": "backend failed",
            "attempts": [
                {
                    "backend": "da3",
                    "status": "failed",
                    "failure_kind": "operational",
                    "error_code": "BACKEND_RUNTIME_ERROR",
                    "error_message": "backend failed",
                    "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
                }
            ],
        },
        requested_backend="da3",
    )

    assert report == {
        "requested_backend": "da3",
        "executed_backend": "da3",
        "availability_state": "failed",
        "reason": "backend failed",
        "synthetic_output": False,
        "stub_mode": False,
        "fallback_executed": False,
        "model_repo_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "model_revision": None,
        "asset_bundle_version": None,
    }


def test_build_orchestrator_result_capability_report_marks_skipped_rows_skipped() -> None:
    report = build_orchestrator_result_capability_report(
        {
            "status": "skipped",
            "reason": "stage disabled",
        },
        requested_backend="depth_pro",
    )

    assert report == {
        "requested_backend": "depth_pro",
        "executed_backend": None,
        "availability_state": "skipped",
        "reason": "stage disabled",
        "synthetic_output": False,
        "stub_mode": False,
        "fallback_executed": False,
        "model_repo_id": None,
        "model_revision": None,
        "asset_bundle_version": None,
    }


def test_build_orchestrator_result_capability_report_marks_malformed_rows_unknown() -> None:
    report = build_orchestrator_result_capability_report({}, requested_backend="depth_pro")

    assert report == {
        "requested_backend": "depth_pro",
        "executed_backend": None,
        "availability_state": "unknown",
        "reason": None,
        "synthetic_output": False,
        "stub_mode": False,
        "fallback_executed": False,
        "model_repo_id": None,
        "model_revision": None,
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
