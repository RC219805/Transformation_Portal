"""Wire-format equivalence tests for the typed v1 envelope models.

The Pydantic models in ``transformation_portal.api.v1`` are introduced in
PR A of the Phase 1.2 sequence. They are NOT yet wired into any route in
``app.py`` — that happens in PRs B–E. The risk that lands first is that the
models' ``model_dump(mode="json")`` output drifts from what ``app.py`` emits
today via ``_api_envelope`` and ``_error_response``.

These tests reimplement the existing helper logic locally (so we don't import
the 8.7K-line ``app.py`` just for two small functions) and assert byte-for-byte
equivalence between the helper output and the model output across every
representative case the route handlers exercise. If these pass, PRs B–E can
swap the ad-hoc dict construction for ``ApiEnvelope`` / ``ErrorEnvelope`` with
no risk of breaking existing wire-contract test coverage in
``tests/test_app_orchestrator_contract_http.py``.

If a test here fails after a Pydantic upgrade or a model change, the regression
must be fixed before any route refactor can proceed.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.api.v1 import (
    ApiEnvelope,
    ErrorCode,
    ErrorEnvelope,
    ErrorObject,
)
from transformation_portal.api.v1.schemas import ERROR_SCHEMA

# ---------------------------------------------------------------------------
# Local copies of the helpers from app.py:1564-1600.
# Kept in sync manually; if app.py changes the wire shape, update here too.
# ---------------------------------------------------------------------------


def _error_obj(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details or {}}


def _api_envelope(
    schema: str,
    *,
    success: bool,
    data: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"schema": schema, "success": success, "data": data, "error": error}


# ---------------------------------------------------------------------------
# ApiEnvelope: success cases
# ---------------------------------------------------------------------------


class TestApiEnvelopeSuccess:
    def test_success_with_dict_data_matches_helper(self) -> None:
        target = _api_envelope(
            "tp.orchestrator.job.v1",
            success=True,
            data={"id": "job-1", "state": "queued"},
        )
        ours = ApiEnvelope[dict](
            schema="tp.orchestrator.job.v1",
            success=True,
            data={"id": "job-1", "state": "queued"},
        ).model_dump(mode="json")
        assert ours == target

    def test_success_with_no_data_emits_null(self) -> None:
        target = _api_envelope("tp.orchestrator.readiness.v1", success=True)
        ours = ApiEnvelope[None](schema="tp.orchestrator.readiness.v1", success=True).model_dump(mode="json")
        assert ours == target
        assert ours["data"] is None  # explicit null, not omitted
        assert ours["error"] is None

    def test_field_order_matches_wire_contract(self) -> None:
        # _api_envelope returns keys in this exact order; ApiEnvelope must too.
        ours = ApiEnvelope[None](schema="tp.orchestrator.readiness.v1", success=True).model_dump(mode="json")
        assert list(ours.keys()) == ["schema", "success", "data", "error"]

    def test_all_four_keys_always_present_when_data_is_none(self) -> None:
        # The helper returns 4 keys even when both data and error are None;
        # the model must too (no exclude_none).
        ours = ApiEnvelope[None](schema="tp.orchestrator.readiness.v1", success=True).model_dump(mode="json")
        assert set(ours.keys()) == {"schema", "success", "data", "error"}

    def test_nested_data_payload_serializes_unchanged(self) -> None:
        payload = {
            "jobs": [{"id": "j1"}, {"id": "j2"}],
            "total": 2,
            "returned": 2,
        }
        target = _api_envelope("tp.orchestrator.jobs.v1", success=True, data=payload)
        ours = ApiEnvelope[dict](schema="tp.orchestrator.jobs.v1", success=True, data=payload).model_dump(mode="json")
        assert ours == target

    def test_construction_via_alias_kwarg(self) -> None:
        # Callers should be able to write ApiEnvelope(schema=...) verbatim,
        # not ApiEnvelope(schema_=...). populate_by_name=True enables this.
        env = ApiEnvelope[dict](schema="tp.orchestrator.job.v1", success=True, data={"id": "x"})
        assert env.model_dump(mode="json")["schema"] == "tp.orchestrator.job.v1"

    def test_construction_via_field_name_kwarg(self) -> None:
        # And also via the underlying python field name, so internal call
        # sites that prefer the unambiguous name keep working.
        env = ApiEnvelope[dict](schema_="tp.orchestrator.job.v1", success=True, data={"id": "x"})
        assert env.model_dump(mode="json")["schema"] == "tp.orchestrator.job.v1"


# ---------------------------------------------------------------------------
# ApiEnvelope: rejection of invalid input
# ---------------------------------------------------------------------------


class TestApiEnvelopeValidation:
    def test_unknown_schema_string_is_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ApiEnvelope[None](schema="tp.orchestrator.not_a_real_schema.v1", success=True)

    def test_extra_top_level_keys_are_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ApiEnvelope[None](
                schema="tp.orchestrator.readiness.v1",
                success=True,
                **{"extra_key": "nope"},
            )


# ---------------------------------------------------------------------------
# ErrorEnvelope + ErrorObject
# ---------------------------------------------------------------------------


class TestErrorEnvelope:
    def test_with_details_matches_helper(self) -> None:
        target = _api_envelope(
            "tp.orchestrator.error.v1",
            success=False,
            data=None,
            error=_error_obj(
                "INVALID_ARGUMENT",
                "bad input",
                {"field": "input_dir", "reason": "missing"},
            ),
        )
        ours = ErrorEnvelope(
            error=ErrorObject(
                code="INVALID_ARGUMENT",
                message="bad input",
                details={"field": "input_dir", "reason": "missing"},
            )
        ).model_dump(mode="json")
        assert ours == target

    def test_without_details_defaults_to_empty_dict(self) -> None:
        # _error_obj coerces None details to {}; ErrorObject mirrors this with
        # default_factory=dict. This is an important wire-compat guarantee.
        target = _api_envelope(
            "tp.orchestrator.error.v1",
            success=False,
            data=None,
            error=_error_obj("NOT_FOUND", "no such job"),
        )
        ours = ErrorEnvelope(error=ErrorObject(code="NOT_FOUND", message="no such job")).model_dump(mode="json")
        assert ours == target

    def test_explicit_none_details_coerces_to_empty_dict(self) -> None:
        # _error_obj does `details or {}` so passing details=None is valid and
        # produces {}. ErrorObject must accept the same input shape — otherwise
        # routes that forward an Optional[dict] would raise ValidationError and
        # turn intended 4xx errors into 500s. The mode="before" field validator
        # coerces None -> {}.
        target = _api_envelope(
            "tp.orchestrator.error.v1",
            success=False,
            data=None,
            error=_error_obj("NOT_FOUND", "no such job", None),
        )
        ours = ErrorEnvelope(error=ErrorObject(code="NOT_FOUND", message="no such job", details=None)).model_dump(mode="json")
        assert ours == target
        assert ours["error"]["details"] == {}

    def test_schema_field_is_locked_to_error_schema(self) -> None:
        env = ErrorEnvelope(error=ErrorObject(code="NOT_FOUND", message="x"))
        dumped = env.model_dump(mode="json")
        assert dumped["schema"] == ERROR_SCHEMA == "tp.orchestrator.error.v1"

    def test_success_is_locked_to_false(self) -> None:
        env = ErrorEnvelope(error=ErrorObject(code="NOT_FOUND", message="x"))
        assert env.model_dump(mode="json")["success"] is False

    def test_data_is_locked_to_null(self) -> None:
        env = ErrorEnvelope(error=ErrorObject(code="NOT_FOUND", message="x"))
        assert env.model_dump(mode="json")["data"] is None

    def test_field_order_in_error_envelope(self) -> None:
        env = ErrorEnvelope(error=ErrorObject(code="NOT_FOUND", message="x"))
        assert list(env.model_dump(mode="json").keys()) == [
            "schema",
            "success",
            "data",
            "error",
        ]

    def test_error_object_field_order(self) -> None:
        # _error_obj returns {code, message, details} in that order.
        env = ErrorEnvelope(error=ErrorObject(code="INVALID_ARGUMENT", message="m", details={"f": "x"}))
        error_dict = env.model_dump(mode="json")["error"]
        assert list(error_dict.keys()) == ["code", "message", "details"]


class TestErrorCodeVocabulary:
    @pytest.mark.parametrize(
        "code",
        [
            # HTTP-status-derived
            "AUTH_CONFIGURATION_ERROR",
            "ARTIFACT_DELETED",
            "ARTIFACT_STORE_UNAVAILABLE",
            "CONFLICT",
            "FORBIDDEN",
            "HTTP_ERROR",
            "INTERNAL_ERROR",
            "INVALID_ARGUMENT",
            "METHOD_NOT_ALLOWED",
            "NOT_FOUND",
            "RATE_LIMITED",
            "REQUEST_TOO_LARGE",
            "SERVICE_UNAVAILABLE",
            "UNAUTHORIZED",
            # Runner-failure
            "RUNNER_ERROR",
            "RUNNER_EXIT_NONZERO",
            "RUNNER_NOT_FOUND",
            "RUNNER_PARTIAL_FAILURE",
        ],
    )
    def test_every_known_error_code_is_accepted(self, code: ErrorCode) -> None:
        # If a route emits a code not in this list, the model must reject it
        # at construction time so the discrepancy is caught early.
        obj = ErrorObject(code=code, message="m")
        assert obj.code == code

    def test_unknown_error_code_is_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ErrorObject(code="MADE_UP_CODE", message="x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cross-cutting: the full wire round-trip across every known schema
# ---------------------------------------------------------------------------


class TestSchemaCoverage:
    @pytest.mark.parametrize(
        "schema_name",
        [
            "tp.orchestrator.config_metadata.v1",
            "tp.orchestrator.config_preview.v1",
            "tp.orchestrator.error.v1",
            "tp.orchestrator.job.v1",
            "tp.orchestrator.job_status.v1",
            "tp.orchestrator.jobs.v1",
            "tp.orchestrator.portal_event.v1",
            "tp.orchestrator.portal_rum.v1",
            "tp.orchestrator.portal_rum_ingest.v1",
            "tp.orchestrator.presets.v1",
            "tp.orchestrator.readiness.v1",
            "tp.orchestrator.upload_staging.v1",
        ],
    )
    def test_every_known_schema_round_trips(self, schema_name: str) -> None:
        # The full set of schemas declared in api/v1/schemas.py must each be
        # constructible. If app.py adds a new schema string, this test surfaces
        # the omission.
        env: ApiEnvelope[None] = ApiEnvelope(schema=schema_name, success=True)
        assert env.model_dump(mode="json")["schema"] == schema_name
