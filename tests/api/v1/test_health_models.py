"""Unit tests for the health/readiness response models (Phase 1.2 PR B).

These tests verify the typed models in
``transformation_portal.api.v1.health`` accept and reject the wire shapes
that ``app.py``'s ``/healthz``, ``/ready``, and ``/v1/readiness`` handlers
produce. Wire-shape regressions are caught here BEFORE they reach FastAPI's
response validation in CI's contract tests.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from transformation_portal.api.v1 import (
    ApiEnvelope,
    HealthzResponse,
    ReadinessData,
    ReadinessEnvelope,
    ReadinessServer,
    ReadyResponse,
)

pytestmark = pytest.mark.unit


def _lux_depth_pipeline_payload() -> dict[str, Any]:
    return {
        "status": "ready",
        "canonical_command": "lux-depth-v3",
        "missing_prerequisites": [],
        "runner_details": {
            "type": "python_module",
            "available": True,
            "module": "transformation_portal.lux_depth_v3",
            "command": ["/usr/bin/python", "-m", "transformation_portal.lux_depth_v3"],
            "python_executable": "/usr/bin/python",
        },
        "notes": ["Base readiness covers runner invocation, path safety, and orchestrator preflight."],
        "canary_status": "ready",
    }


def _artifact_store_payload() -> dict[str, Any]:
    return {"backend": "local", "configured": True, "prefix": "", "signed_urls": False}


def _archive_gate_pipeline_payload(
    pipeline: str,
    *,
    status: str,
    missing_prerequisites: list[dict[str, Any]],
) -> dict[str, Any]:
    command_by_pipeline = {
        "archive-gate-a": "fixity-scan",
        "archive-gate-b": "bag-build",
        "archive-gate-c": "mets-export",
    }
    note_by_pipeline = {
        "archive-gate-a": "Canonical archive-gate-a dispatch expects fixity-scan with an existing archive index.",
        "archive-gate-b": "Canonical dispatch for this archive stage requires a prior rights-manifest artifact.",
        "archive-gate-c": "Canonical dispatch for this archive stage requires a prior rights-manifest artifact.",
    }
    command = command_by_pipeline[pipeline]
    return {
        "status": status,
        "canonical_command": command,
        "missing_prerequisites": missing_prerequisites,
        "runner_details": {
            "type": "python_script",
            "available": True,
            "script_path": "/repo/tools/archive_governance.py",
            "python_executable": "/usr/bin/python",
            "command": ["/usr/bin/python", "/repo/tools/archive_governance.py", "--json", command],
        },
        "notes": [note_by_pipeline[pipeline]],
    }


def _archive_index_required_issue() -> dict[str, Any]:
    return {
        "reason": "archive_index_required",
        "severity": "degraded",
        "message": "An existing archive index is required at dispatch time.",
        "field": "archive_index",
    }


def _rights_manifest_required_issue() -> dict[str, Any]:
    return {
        "reason": "rights_manifest_required",
        "severity": "blocked",
        "message": "A rights-manifest JSONL artifact from a prior archive stage is required.",
        "field": "manifest_jsonl",
    }


class TestHealthzResponse:
    """Mirrors ``healthz()``: ``JSONResponse({"ok": True, "time": _now()})``."""

    def test_minimal_valid_input(self) -> None:
        h = HealthzResponse(ok=True, time=1234.5)
        assert h.model_dump(mode="json") == {"ok": True, "time": 1234.5}

    def test_field_order_matches_handler(self) -> None:
        h = HealthzResponse(ok=True, time=1.0)
        assert list(h.model_dump(mode="json").keys()) == ["ok", "time"]

    def test_extra_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HealthzResponse(ok=True, time=1.0, extra_field="nope")  # type: ignore[call-arg]

    def test_int_time_is_coerced_to_float(self) -> None:
        # _now() returns a float, but accept ints too — JSON conflates the two.
        h = HealthzResponse(ok=True, time=1234)  # type: ignore[arg-type]
        assert h.time == 1234.0


class TestReadyResponse:
    """Mirrors ``ready()``: minimal vs verbose shapes."""

    def test_minimal_shape_when_verbose_false(self) -> None:
        # When TP_READY_VERBOSE=false, the handler returns ok/time/version plus
        # the always-present artifact_store block. The route is wired with
        # response_model_exclude_none=True so verbose-only fields stay absent.
        r = ReadyResponse(ok=True, time=1.0, version="0.1.0", artifact_store=_artifact_store_payload())
        dumped = r.model_dump(mode="json", exclude_none=True)
        assert dumped == {
            "ok": True,
            "time": 1.0,
            "version": "0.1.0",
            "artifact_store": _artifact_store_payload(),
        }

    def test_verbose_shape_includes_cli_jobs_security(self) -> None:
        r = ReadyResponse(
            ok=True,
            time=1.0,
            version="0.1.0",
            artifact_store=_artifact_store_payload(),
            cli={"lux-depth-v3": True, "archive-governance": True, "python": "3.11.15"},
            jobs={"active": 0, "total": 5},
            security={
                "api_key_enforced_for_jobs": True,
                "rate_limit_per_minute": 60,
                "max_concurrent_jobs": 4,
                "max_request_bytes": 65536,
                "trusted_hosts_enabled": True,
                "trust_x_forwarded_for": False,
                "trusted_proxy_ips_count": 0,
                "allowed_input_roots_count": 1,
                "allowed_output_roots_count": 1,
                "allow_sse_query_api_key": False,
                "docs_enabled": False,
            },
        )
        dumped = r.model_dump(mode="json")
        assert dumped["ok"] is True
        assert dumped["cli"] == {"lux-depth-v3": True, "archive-governance": True, "python": "3.11.15"}
        assert dumped["jobs"]["active"] == 0
        assert dumped["security"]["api_key_enforced_for_jobs"] is True

    def test_extra_top_level_keys_pass_through(self) -> None:
        # extra="allow" — the security dict churns and we don't want a strict
        # model to reject a new field added to the handler.
        r = ReadyResponse(
            ok=True,
            time=1.0,
            version="0.1.0",
            artifact_store=_artifact_store_payload(),
            future_field="ok",
        )
        # extra fields are accepted; they appear in dump
        assert r.model_dump(mode="json")["future_field"] == "ok"

    def test_unknown_keys_inside_security_are_tolerated(self) -> None:
        # The security sub-dict is `dict[str, Any]`, so any new feature flag
        # rolls through without a model bump.
        r = ReadyResponse(
            ok=True,
            time=1.0,
            version="0.1.0",
            artifact_store=_artifact_store_payload(),
            security={"api_key_enforced_for_jobs": True, "future_flag_2027": "rolled out"},
        )
        assert r.security == {"api_key_enforced_for_jobs": True, "future_flag_2027": "rolled out"}


class TestReadinessServer:
    def test_required_fields(self) -> None:
        s = ReadinessServer(time=1.0, version="0.1.0", auth_mode="direct_debug", backend_live=True)
        assert s.model_dump(mode="json") == {
            "time": 1.0,
            "version": "0.1.0",
            "auth_mode": "direct_debug",
            "backend_live": True,
        }

    def test_extra_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReadinessServer(
                time=1.0,
                version="0.1.0",
                auth_mode="direct_debug",
                backend_live=True,
                x="y",  # type: ignore[call-arg]
            )


class TestReadinessData:
    def test_envelope_payload_shape(self) -> None:
        data = ReadinessData(
            server=ReadinessServer(time=1.0, version="0.1.0", auth_mode="direct_debug", backend_live=True),
            pipelines={
                "lux-depth-v3": _lux_depth_pipeline_payload(),
                "archive-gate-a": _archive_gate_pipeline_payload(
                    "archive-gate-a",
                    status="degraded",
                    missing_prerequisites=[_archive_index_required_issue()],
                ),
            },
        )
        dumped = data.model_dump(mode="json")
        assert set(dumped.keys()) == {"server", "pipelines"}
        assert dumped["pipelines"]["lux-depth-v3"]["status"] == "ready"
        assert dumped["pipelines"]["lux-depth-v3"]["canonical_command"] == "lux-depth-v3"
        assert dumped["pipelines"]["archive-gate-a"]["status"] == "degraded"
        assert dumped["pipelines"]["archive-gate-a"]["missing_prerequisites"][0]["reason"] == "archive_index_required"


class TestReadinessEnvelope:
    """``ApiEnvelope[ReadinessData]`` — the wire shape of GET /v1/readiness."""

    def test_full_round_trip_matches_handler_shape(self) -> None:
        # Construct the envelope the way readiness() does through _api_envelope.
        env = ReadinessEnvelope(
            schema="tp.orchestrator.readiness.v1",
            success=True,
            data=ReadinessData(
                server=ReadinessServer(
                    time=1.0,
                    version="0.1.0",
                    auth_mode="direct_debug",
                    backend_live=True,
                ),
                pipelines={
                    "lux-depth-v3": _lux_depth_pipeline_payload(),
                    "archive-gate-a": _archive_gate_pipeline_payload(
                        "archive-gate-a",
                        status="degraded",
                        missing_prerequisites=[_archive_index_required_issue()],
                    ),
                    "archive-gate-b": _archive_gate_pipeline_payload(
                        "archive-gate-b",
                        status="blocked",
                        missing_prerequisites=[_rights_manifest_required_issue()],
                    ),
                    "archive-gate-c": _archive_gate_pipeline_payload(
                        "archive-gate-c",
                        status="blocked",
                        missing_prerequisites=[_rights_manifest_required_issue()],
                    ),
                },
            ),
        )
        dumped = env.model_dump(mode="json")
        # Top-level envelope keys in the standard order
        assert list(dumped.keys()) == ["schema", "success", "data", "error"]
        assert dumped["schema"] == "tp.orchestrator.readiness.v1"
        assert dumped["success"] is True
        assert dumped["error"] is None
        # Payload shape
        assert dumped["data"]["server"]["backend_live"] is True
        archive_gate_b = dumped["data"]["pipelines"]["archive-gate-b"]
        assert archive_gate_b["status"] == "blocked"
        assert archive_gate_b["missing_prerequisites"][0]["reason"] == "rights_manifest_required"

    def test_alias_is_apienvelope_specialization(self) -> None:
        # ReadinessEnvelope is just sugar for ApiEnvelope[ReadinessData];
        # the underlying generic specialization matters for FastAPI's OpenAPI
        # generation, but parameterized generic object identity is not stable
        # across all Python/Pydantic versions.
        # NOTE: by design, the alias does NOT lock the schema field to
        # "tp.orchestrator.readiness.v1" — any valid SchemaName is accepted at
        # the model layer, and the route handler is responsible for setting the
        # right schema string. This matches the alias pattern used elsewhere in
        # api/v1/ (only ErrorEnvelope locks its schema, because there's a single
        # canonical error schema). If a future refactor wants per-route schema
        # locking, that's a subclassing exercise, not an alias change.
        metadata = ReadinessEnvelope.__pydantic_generic_metadata__
        assert issubclass(ReadinessEnvelope, ApiEnvelope)
        assert metadata["origin"] is ApiEnvelope
        assert metadata["args"] == (ReadinessData,)

    def test_unrecognised_schema_string_rejected_at_literal_level(self) -> None:
        # The SchemaName Literal does still reject unknown strings — the
        # alias just doesn't pin the field to a single value.
        with pytest.raises(ValidationError):
            ReadinessEnvelope(
                schema="tp.orchestrator.not_a_real_schema.v1",
                success=True,
                data=ReadinessData(
                    server=ReadinessServer(time=1.0, version="0.1.0", auth_mode="direct_debug", backend_live=True),
                    pipelines={},
                ),
            )
