"""Direct unit tests for ``collect_run_card_backend_semantic_errors``.

The wrapper ``validate_run_card_backend_semantics`` is exercised by
``test_run_card_validator.py`` (it raises ``RuntimeError`` with a
concatenated message). These tests target the underlying collector
directly so we can assert on:

* Multi-error accumulation (the wrapper concatenates and obscures count)
* Type-error branches not triggered by the higher-level tests
* Wrapper-semantics fields handled in isolation (logical_backend vs
  resolved_engine missing/present/non-string)
* Early-return behavior when backend metadata is absent
"""

from __future__ import annotations

import pytest

from transformation_portal.lux_depth_v3.validators.run_card_backend_semantics import (
    collect_run_card_backend_semantic_errors,
)

pytestmark = [pytest.mark.unit]


def _base_payload(**overrides):
    payload = {
        "backend_selection": {"requested": "da3", "resolved": "da3"},
        "backend_summary": {
            "requested_backend": "da3",
            "primary_backend": "da3",
            "final_backends_used": ["da3"],
            "fallback_images": 0,
        },
        "success_count": 5,
        "total_images": 5,
        "error_count": 0,
    }
    for key, value in overrides.items():
        payload[key] = value
    return payload


class TestEarlyReturn:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"backend_selection": {"resolved": "da3"}},
            {"backend_summary": {"final_backends_used": ["da3"]}},
            {"backend_selection": "not-a-dict", "backend_summary": {}},
            {"backend_selection": {}, "backend_summary": "not-a-dict"},
        ],
    )
    def test_returns_empty_when_metadata_incomplete(self, payload):
        assert collect_run_card_backend_semantic_errors(payload) == []

    def test_empty_final_backends_with_zero_success_is_clean(self):
        payload = _base_payload()
        payload["success_count"] = 0
        payload["backend_summary"]["final_backends_used"] = []
        assert collect_run_card_backend_semantic_errors(payload) == []

    def test_missing_success_count_treated_as_zero(self):
        payload = _base_payload()
        del payload["success_count"]
        payload["backend_summary"]["final_backends_used"] = []
        assert collect_run_card_backend_semantic_errors(payload) == []


class TestStructuralErrors:
    def test_non_list_final_backends_returns_single_error(self):
        payload = _base_payload()
        payload["backend_summary"]["final_backends_used"] = "da3"
        errors = collect_run_card_backend_semantic_errors(payload)
        assert errors == ["backend_summary.final_backends_used must be an array"]

    def test_empty_list_with_success_returns_single_error(self):
        payload = _base_payload()
        payload["backend_summary"]["final_backends_used"] = []
        errors = collect_run_card_backend_semantic_errors(payload)
        assert errors == ["backend_summary.final_backends_used must be non-empty when success_count > 0"]

    @pytest.mark.parametrize("primary", [None, "", 42, ["da3"]])
    def test_invalid_primary_short_circuits(self, primary):
        payload = _base_payload()
        payload["backend_summary"]["final_backends_used"] = [primary]
        errors = collect_run_card_backend_semantic_errors(payload)
        assert errors == ["backend_summary.final_backends_used[0] must be a non-empty string"]

    @pytest.mark.parametrize("resolved", [None, "", 42, ["da3"]])
    def test_invalid_resolved_short_circuits(self, resolved):
        payload = _base_payload()
        payload["backend_selection"]["resolved"] = resolved
        errors = collect_run_card_backend_semantic_errors(payload)
        assert errors == ["backend_selection.resolved must be a non-empty string"]


class TestMultiErrorAccumulation:
    def test_primary_and_resolved_both_mismatch_yield_two_errors(self):
        payload = _base_payload()
        payload["backend_selection"]["resolved"] = "depth_pro"
        payload["backend_summary"]["primary_backend"] = "da2"
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("primary_backend must equal" in e for e in errors)
        assert any("resolved must match" in e for e in errors)
        assert len(errors) == 2


class TestDepthProFallbackPolicy:
    def test_full_fallback_under_run_failure_is_tolerated(self):
        payload = _base_payload(
            success_count=2,
            total_images=3,
            error_count=1,
        )
        payload["backend_selection"]["requested"] = "depth_pro"
        payload["backend_selection"]["resolved"] = "da3"
        payload["backend_summary"]["requested_backend"] = "depth_pro"
        payload["backend_summary"]["primary_backend"] = "da3"
        payload["backend_summary"]["final_backends_used"] = ["da3"]
        payload["backend_summary"]["fallback_images"] = 2
        assert collect_run_card_backend_semantic_errors(payload) == []

    def test_full_fallback_with_clean_run_is_rejected(self):
        payload = _base_payload(success_count=2, total_images=2, error_count=0)
        payload["backend_selection"]["requested"] = "depth_pro"
        payload["backend_selection"]["resolved"] = "da3"
        payload["backend_summary"]["requested_backend"] = "depth_pro"
        payload["backend_summary"]["primary_backend"] = "da3"
        payload["backend_summary"]["final_backends_used"] = ["da3"]
        payload["backend_summary"]["fallback_images"] = 2
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("'depth_pro' was not honored" in e for e in errors)

    def test_partial_fallback_is_tolerated(self):
        payload = _base_payload(success_count=4, total_images=4, error_count=0)
        payload["backend_selection"]["requested"] = "depth_pro"
        payload["backend_selection"]["resolved"] = "depth_pro"
        payload["backend_summary"]["requested_backend"] = "depth_pro"
        payload["backend_summary"]["primary_backend"] = "depth_pro"
        payload["backend_summary"]["final_backends_used"] = ["depth_pro"]
        payload["backend_summary"]["fallback_images"] = 1
        assert collect_run_card_backend_semantic_errors(payload) == []

    def test_requested_falls_back_via_summary_only(self):
        # Wrapper test covers requested via backend_selection; this exercises
        # the alternate lookup path where only backend_summary declares it.
        payload = _base_payload(success_count=2, total_images=2, error_count=0)
        payload["backend_selection"].pop("requested", None)
        payload["backend_selection"]["resolved"] = "da3"
        payload["backend_summary"]["requested_backend"] = "depth_pro"
        payload["backend_summary"]["primary_backend"] = "da3"
        payload["backend_summary"]["final_backends_used"] = ["da3"]
        payload["backend_summary"]["fallback_images"] = 2
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("'depth_pro' was not honored" in e for e in errors)


class TestWrapperSemantics:
    def _wrapper_payload(self, **selection_overrides):
        payload = _base_payload()
        payload["backend_selection"].update(
            {
                "logical_backend": "ensemble",
                "resolved_engine": "da3",
            }
        )
        payload["backend_selection"].update(selection_overrides)
        return payload

    def test_no_wrapper_fields_means_no_wrapper_validation(self):
        payload = _base_payload()
        # Neither logical_backend nor resolved_engine declared.
        assert collect_run_card_backend_semantic_errors(payload) == []

    def test_logical_only_requires_resolved_engine(self):
        payload = _base_payload()
        payload["backend_selection"]["logical_backend"] = "ensemble"
        # resolved_engine intentionally absent
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("resolved_engine must be a non-empty string" in e for e in errors)

    def test_resolved_engine_only_requires_logical_backend(self):
        payload = _base_payload()
        payload["backend_selection"]["resolved_engine"] = "da3"
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("logical_backend must be a non-empty string" in e for e in errors)

    @pytest.mark.parametrize("logical", [None, "", 42, []])
    def test_invalid_logical_is_rejected(self, logical):
        payload = self._wrapper_payload(logical_backend=logical)
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("logical_backend must be a non-empty string" in e for e in errors)

    def test_logical_equal_to_resolved_engine_is_rejected(self):
        payload = self._wrapper_payload(logical_backend="da3")
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("logical_backend and backend_selection.resolved_engine must differ" in e for e in errors)

    def test_resolved_engine_must_match_primary(self):
        payload = self._wrapper_payload(resolved_engine="depth_pro")
        # final_backends_used[0] is "da3"; resolved_engine declares "depth_pro".
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("resolved_engine must match backend_summary.final_backends_used[0]" in e for e in errors)

    def test_wrapper_with_nonzero_fallback_is_rejected(self):
        payload = self._wrapper_payload()
        payload["backend_summary"]["fallback_images"] = 2
        errors = collect_run_card_backend_semantic_errors(payload)
        assert any("wrapper semantics are only valid when backend_summary.fallback_images == 0" in e for e in errors)

    def test_well_formed_wrapper_passes(self):
        payload = self._wrapper_payload()
        assert collect_run_card_backend_semantic_errors(payload) == []
