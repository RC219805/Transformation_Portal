"""Test coverage for EnhanceOrchestrator pure/static helper surfaces.

Phase 2 Coverage: targets the deterministic helper methods on
EnhanceOrchestrator (shape/path/int coercion, run-card summary builders,
backend-summary aggregation) plus the module-level dependency-status probe.
These paths are decision-heavy but side-effect free, so they are exercised
directly rather than through the full enhance_image dispatch.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from transformation_portal.lux_depth_v3.orchestrator import (
    EnhanceOrchestrator,
    _log_dependency_status,
)

pytestmark = pytest.mark.unit


def _make_mock_registry() -> Mock:
    backend = Mock()
    backend.name = "da3"
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None

    registry = Mock()
    registry.get_backend.return_value = backend
    return registry


def _create_orchestrator(tmp_path: Path, **config_kwargs: Any) -> EnhanceOrchestrator:
    """Build an orchestrator with a mocked depth backend registry."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    defaults: Dict[str, Any] = {
        "depth_backend": "da3",
        "depth_device": "cpu",
        "enable_v2": False,
        "enable_materials_v3": False,
    }
    defaults.update(config_kwargs)
    config = EnhanceConfig(**defaults)

    with patch(
        "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
        return_value=_make_mock_registry(),
    ):
        return EnhanceOrchestrator(config, tmp_path)


class TestLogDependencyStatus:
    """Tests for the module-level _log_dependency_status probe."""

    def test_reports_all_known_dependency_keys(self) -> None:
        status = _log_dependency_status()

        for key in ("torch", "transformers", "coremltools", "scikit-image", "numba", "hf_token"):
            assert key in status
        assert isinstance(status["torch_transformers_compatible"], bool)

    def test_missing_distributions_report_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _always_missing(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", _always_missing)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        status = _log_dependency_status()

        assert status["torch"] is False
        assert status["transformers"] is False
        assert status["coremltools"] is False
        assert status["scikit-image"] is False
        assert status["numba"] is False
        assert status["hf_token"] is False
        # With neither torch nor transformers, the runtime issue probe is satisfied.
        assert status["torch_transformers_compatible"] is True

    def test_present_distributions_report_versions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
        monkeypatch.setenv("HF_TOKEN", "secret-token")

        status = _log_dependency_status()

        assert status["torch"] is True
        assert status["torch_version"] == "9.9.9"
        assert status["transformers_version"] == "9.9.9"
        assert status["hf_token"] is True


class TestShapeList:
    """Tests for EnhanceOrchestrator._shape_list."""

    def test_normalizes_two_element_shape(self) -> None:
        assert EnhanceOrchestrator._shape_list((480, 640)) == [480, 640]

    def test_truncates_longer_shapes_to_first_two(self) -> None:
        assert EnhanceOrchestrator._shape_list([480, 640, 3]) == [480, 640]

    def test_rejects_short_or_non_sequence(self) -> None:
        assert EnhanceOrchestrator._shape_list([480]) is None
        assert EnhanceOrchestrator._shape_list("480x640") is None
        assert EnhanceOrchestrator._shape_list(None) is None

    def test_rejects_non_numeric_entries(self) -> None:
        assert EnhanceOrchestrator._shape_list(["a", "b"]) is None


class TestCoerceNonnegativeInt:
    """Tests for EnhanceOrchestrator._coerce_nonnegative_int."""

    def test_coerces_valid_values(self) -> None:
        assert EnhanceOrchestrator._coerce_nonnegative_int(5) == 5
        assert EnhanceOrchestrator._coerce_nonnegative_int("7") == 7
        assert EnhanceOrchestrator._coerce_nonnegative_int(0) == 0

    def test_rejects_bool(self) -> None:
        assert EnhanceOrchestrator._coerce_nonnegative_int(True) is None

    def test_rejects_negative(self) -> None:
        assert EnhanceOrchestrator._coerce_nonnegative_int(-3) is None

    def test_rejects_non_numeric(self) -> None:
        assert EnhanceOrchestrator._coerce_nonnegative_int("abc") is None
        assert EnhanceOrchestrator._coerce_nonnegative_int(None) is None


class TestResultImageKey:
    """Tests for EnhanceOrchestrator._result_image_key."""

    def test_resolves_absolute_path(self, tmp_path: Path) -> None:
        target = tmp_path / "image.png"
        assert EnhanceOrchestrator._result_image_key(str(target)) == str(target.resolve())

    def test_rejects_empty_or_non_string(self) -> None:
        assert EnhanceOrchestrator._result_image_key("") is None
        assert EnhanceOrchestrator._result_image_key(None) is None
        assert EnhanceOrchestrator._result_image_key(123) is None


class TestInputRootRelativePath:
    """Tests for EnhanceOrchestrator._input_root_relative_path."""

    def test_renders_relative_to_input_root(self, tmp_path: Path) -> None:
        root = tmp_path / "inputs"
        root.mkdir()
        rel = EnhanceOrchestrator._input_root_relative_path(str(root / "a" / "b.png"), input_root=root)
        assert rel == "a/b.png"

    def test_absolute_path_outside_root_falls_back_to_name(self, tmp_path: Path) -> None:
        rel = EnhanceOrchestrator._input_root_relative_path("/elsewhere/deep/file.png", input_root=tmp_path)
        assert rel == "file.png"

    def test_relative_path_without_root_is_posix(self) -> None:
        rel = EnhanceOrchestrator._input_root_relative_path("nested/file.png", input_root=None)
        assert rel == "nested/file.png"


class TestSelectedSuccessfulAttempt:
    """Tests for EnhanceOrchestrator._selected_successful_attempt."""

    def test_returns_none_without_attempts(self) -> None:
        assert EnhanceOrchestrator._selected_successful_attempt({}) is None
        assert EnhanceOrchestrator._selected_successful_attempt({"attempts": []}) is None

    def test_uses_selected_index_when_successful(self) -> None:
        result = {
            "attempts": [
                {"status": "failed", "backend": "depth_pro"},
                {"status": "success", "backend": "da3"},
            ],
            "selected_attempt_index": 1,
        }
        assert EnhanceOrchestrator._selected_successful_attempt(result)["backend"] == "da3"

    def test_falls_back_to_first_success_when_index_invalid(self) -> None:
        result = {
            "attempts": [
                {"status": "failed", "backend": "depth_pro"},
                {"status": "success", "backend": "da3"},
            ],
            "selected_attempt_index": 0,  # points at a failed attempt
        }
        assert EnhanceOrchestrator._selected_successful_attempt(result)["backend"] == "da3"

    def test_returns_none_when_no_attempt_succeeded(self) -> None:
        result = {"attempts": [{"status": "failed"}, {"status": "failed"}]}
        assert EnhanceOrchestrator._selected_successful_attempt(result) is None


class TestPixelOpsBlockedReasons:
    """Tests for EnhanceOrchestrator._pixel_ops_blocked_reasons."""

    def test_empty_when_no_blocked_list(self) -> None:
        assert EnhanceOrchestrator._pixel_ops_blocked_reasons({}) == {}
        assert EnhanceOrchestrator._pixel_ops_blocked_reasons({"blocked": "nope"}) == {}

    def test_histograms_blocked_by_reasons(self) -> None:
        pixel_ops = {
            "blocked": [
                {"blocked_by": ["policy", "shape"]},
                {"blocked_by": ["policy"]},
                {"reason": "manual"},
                {},  # no reason -> "unknown"
                "not-a-dict",  # skipped
            ]
        }
        histogram = EnhanceOrchestrator._pixel_ops_blocked_reasons(pixel_ops)
        assert histogram == {"policy": 2, "shape": 1, "manual": 1, "unknown": 1}


class TestExtractRunCardSegmentationMetadata:
    """Tests for EnhanceOrchestrator._extract_run_card_segmentation_metadata."""

    def test_returns_none_for_non_dict_or_missing_metadata(self) -> None:
        assert EnhanceOrchestrator._extract_run_card_segmentation_metadata("x") is None
        assert EnhanceOrchestrator._extract_run_card_segmentation_metadata({}) is None
        assert (
            EnhanceOrchestrator._extract_run_card_segmentation_metadata(
                {"materials_v3_metadata": {"segmentation_metadata": "bad"}}
            )
            is None
        )

    def test_backfills_counts_from_result(self) -> None:
        result = {
            "materials_v3_metadata": {"segmentation_metadata": {"backend": "sam2"}},
            "material_masks": {"wood": 1, "metal": 2},
            "materials_v3_pixel_ops": {
                "applied": [{"op": "a"}],
                "blocked": [{"op": "b"}, {"op": "c"}],
                "passthrough_status": {"code": "OK"},
            },
        }
        meta = EnhanceOrchestrator._extract_run_card_segmentation_metadata(result)

        assert meta["mask_count"] == 2
        assert meta["pixel_ops_applied_count"] == 1
        assert meta["pixel_ops_blocked_count"] == 2
        assert meta["pixel_ops_passthrough"] == {"code": "OK"}

    def test_blocked_count_derived_from_reason_histogram(self) -> None:
        result = {
            "materials_v3_metadata": {"segmentation_metadata": {"backend": "sam2"}},
            "materials_v3_pixel_ops": {
                "blocked": [{"blocked_by": ["policy"]}, {"blocked_by": ["policy", "shape"]}],
            },
        }
        meta = EnhanceOrchestrator._extract_run_card_segmentation_metadata(result)
        assert meta["pixel_ops_blocked_count"] == 2


class TestBuildRunCardMaterialsSummary:
    """Tests for EnhanceOrchestrator._build_run_card_materials_summary."""

    def test_summary_with_masks_and_applied_ops(self) -> None:
        summary = EnhanceOrchestrator._build_run_card_materials_summary(
            {
                "mask_count": 3,
                "pixel_ops_applied_count": 2,
                "pixel_ops_blocked_count": 1,
                "pixel_ops_passthrough": {"code": " GATE_OK "},
            }
        )
        assert summary["masks_generated"] is True
        assert summary["mask_count"] == 3
        assert summary["pixel_ops_applied"] is True
        assert summary["pixel_ops_applied_count"] == 2
        assert summary["blocked_count"] == 1
        assert summary["passthrough_code"] == "GATE_OK"

    def test_summary_defaults_when_fields_absent(self) -> None:
        summary = EnhanceOrchestrator._build_run_card_materials_summary({})
        assert summary["masks_generated"] is False
        assert summary["mask_count"] == 0
        assert summary["pixel_ops_applied"] is False
        assert summary["blocked_count"] == 0
        assert summary["passthrough_code"] is None

    def test_applied_count_derived_from_passthrough_details(self) -> None:
        summary = EnhanceOrchestrator._build_run_card_materials_summary(
            {
                "pixel_ops_passthrough": {"details": {"applied_ops_count": 4}},
            }
        )
        assert summary["pixel_ops_applied_count"] == 4

    def test_blocked_count_summed_from_passthrough_reasons(self) -> None:
        summary = EnhanceOrchestrator._build_run_card_materials_summary(
            {
                "pixel_ops_passthrough": {"details": {"blocked_reasons": {"policy": 2, "shape": 1}}},
            }
        )
        assert summary["blocked_count"] == 3


class TestSegmentationPerformanceWarnings:
    """Tests for _build_run_card_segmentation_performance_warnings."""

    def test_empty_for_non_sam2_backend(self) -> None:
        warnings = EnhanceOrchestrator._build_run_card_segmentation_performance_warnings(
            {"backend": "efficient_sam"},
            runtime_s=1.0,
            materials_summary={},
        )
        assert warnings == []

    def test_empty_without_timing_data(self) -> None:
        warnings = EnhanceOrchestrator._build_run_card_segmentation_performance_warnings(
            {"backend": "sam2"},
            runtime_s=1.0,
            materials_summary={},
        )
        assert warnings == []

    def test_warns_when_sam2_dominates_without_pixel_ops(self) -> None:
        warnings = EnhanceOrchestrator._build_run_card_segmentation_performance_warnings(
            {"backend": "sam2", "timing_ms": {"backend_segment": 980.0}},
            runtime_s=1.0,  # 1000 ms total -> 98% share
            materials_summary={"mask_count": 4, "pixel_ops_applied_count": 0},
        )
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "advisory"
        assert warnings[0]["details"]["mask_count"] == 4

    def test_no_warning_when_pixel_ops_applied(self) -> None:
        warnings = EnhanceOrchestrator._build_run_card_segmentation_performance_warnings(
            {"backend": "sam2", "timing_ms": {"backend_segment": 980.0}},
            runtime_s=1.0,
            materials_summary={"mask_count": 4, "pixel_ops_applied_count": 2},
        )
        assert warnings == []

    def test_empty_when_runtime_or_segment_timing_missing(self) -> None:
        # backend_segment key absent from timing_ms
        assert (
            EnhanceOrchestrator._build_run_card_segmentation_performance_warnings(
                {"backend": "sam2", "timing_ms": {"other": 1.0}},
                runtime_s=1.0,
                materials_summary={},
            )
            == []
        )
        # runtime_s is None
        assert (
            EnhanceOrchestrator._build_run_card_segmentation_performance_warnings(
                {"backend": "sam2", "timing_ms": {"backend_segment": 980.0}},
                runtime_s=None,
                materials_summary={},
            )
            == []
        )

    def test_empty_when_timing_values_are_non_numeric(self) -> None:
        warnings = EnhanceOrchestrator._build_run_card_segmentation_performance_warnings(
            {"backend": "sam2", "timing_ms": {"backend_segment": "not-a-number"}},
            runtime_s=1.0,
            materials_summary={},
        )
        assert warnings == []


class TestComputeBackendSummary:
    """Tests for EnhanceOrchestrator._compute_backend_summary."""

    def test_empty_results_yield_baseline_summary(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(tmp_path)
        summary = orch._compute_backend_summary([])

        assert summary["primary_backend"] is None
        assert summary["final_backends_used"] == []
        assert summary["fallback_images"] == 0

    def test_preferred_backend_leads_used_list(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(tmp_path)
        orch._backend_metadata.resolved_backend = "da3"
        results = [
            {"status": "ok", "backend": "synthetic", "attempts": [{"status": "success"}]},
            {"status": "ok", "backend": "da3", "attempts": [{"status": "success"}]},
        ]
        summary = orch._compute_backend_summary(results)

        assert summary["final_backends_used"][0] == "da3"
        assert "synthetic" in summary["final_backends_used"]

    def test_counts_semantic_and_operational_fallbacks(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(tmp_path)
        orch._backend_metadata.resolved_backend = "depth_pro"
        results = [
            {
                "status": "ok",
                "backend": "da3",
                "fallback_used": True,
                "attempts": [
                    {"status": "failed", "failure_kind": "semantic"},
                    {"status": "success"},
                ],
            },
            {
                "status": "ok",
                "backend": "da3",
                "attempts": [
                    {"status": "failed", "failure_kind": "operational"},
                    {"status": "success"},
                ],
            },
        ]
        summary = orch._compute_backend_summary(results)

        assert summary["fallback_images"] == 2
        assert summary["semantic_fallback_images"] == 1
        assert summary["operational_fallback_images"] == 1


class TestRequestedBackendFulfillmentDefect:
    """Tests for EnhanceOrchestrator._requested_backend_fulfillment_defect."""

    def test_none_when_requested_backend_not_depth_pro(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(tmp_path)
        summary = {"requested_backend": "da3", "primary_backend": "da3", "fallback_images": 0}
        assert orch._requested_backend_fulfillment_defect([], summary) is None

    def test_none_when_primary_matches_requested(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(tmp_path)
        results = [{"status": "ok"}]
        summary = {
            "requested_backend": "depth_pro",
            "primary_backend": "depth_pro",
            "fallback_images": 1,
        }
        assert orch._requested_backend_fulfillment_defect(results, summary) is None

    def test_reports_defect_when_depth_pro_fully_fell_back(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(tmp_path)
        results = [
            {
                "status": "ok",
                "attempts": [
                    {"status": "failed", "backend": "depth_pro", "error_message": "license missing"},
                    {"status": "success", "backend": "da3"},
                ],
            }
        ]
        summary = {
            "requested_backend": "depth_pro",
            "primary_backend": "da3",
            "fallback_images": 1,
        }
        message = orch._requested_backend_fulfillment_defect(results, summary)

        assert message is not None
        assert "depth_pro" in message
        assert "license missing" in message

    def test_falls_back_to_startup_reason_when_no_attempt_detail(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(tmp_path)
        orch._backend_metadata.resolution_reason = "depth_pro license not accepted"
        results = [{"status": "ok", "attempts": [{"status": "success", "backend": "da3"}]}]
        summary = {
            "requested_backend": "depth_pro",
            "primary_backend": "da3",
            "fallback_images": 1,
        }
        message = orch._requested_backend_fulfillment_defect(results, summary)

        assert message is not None
        assert "depth_pro license not accepted" in message


class TestBuildRunCardSegmentationStatus:
    """Tests for EnhanceOrchestrator._build_run_card_segmentation_status."""

    def test_not_requested_when_materials_v3_disabled(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(tmp_path, enable_materials_v3=False)
        status = orch._build_run_card_segmentation_status(None)

        assert status["status"] == "not_requested"
        assert status["enabled"] is False
        assert status["reason"] == "materials_v3_disabled"

    def test_reports_failure_code_from_result_info(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(
            tmp_path,
            enable_materials_v3=True,
            enable_material_segmentation=True,
        )
        status = orch._build_run_card_segmentation_status(
            None,
            result_failure_info={"error_code": "SEGMENTATION_BACKEND_UNAVAILABLE"},
        )

        assert status["status"] == "failed"
        assert status["failure_code"] == "SEGMENTATION_BACKEND_UNAVAILABLE"
        assert status["errors"] == ["SEGMENTATION_BACKEND_UNAVAILABLE"]

    def test_missing_evidence_placeholder_without_metadata(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(
            tmp_path,
            enable_materials_v3=True,
            enable_material_segmentation=True,
        )
        status = orch._build_run_card_segmentation_status(None)

        assert status["status"] in ("missing_evidence", "not_recorded")
        assert status["enabled"] is True

    def test_summarizes_metadata_when_present(self, tmp_path: Path) -> None:
        orch = _create_orchestrator(
            tmp_path,
            enable_materials_v3=True,
            enable_material_segmentation=True,
        )
        status = orch._build_run_card_segmentation_status(
            {"status": "ok", "backend": "sam2", "mask_count": 3},
        )

        assert status["status"] == "ok"
        assert status["enabled"] is True
        assert status["backend"] == "sam2"
        assert status["materials_summary"]["mask_count"] == 3
