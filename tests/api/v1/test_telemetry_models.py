"""Unit tests for the telemetry response models (Phase 1.2 PR E).

These tests verify the typed models in
``transformation_portal.api.v1.telemetry`` accept and reject the wire shapes
that ``app.py``'s telemetry handlers produce. They complement (do not replace)
the end-to-end route tests in ``tests/test_app_orchestrator_contract_http.py``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from transformation_portal.api.v1 import (
    PortalEventData,
    PortalEventEnvelope,
    PortalRumIngestData,
    PortalRumIngestEnvelope,
)

# ---------------------------------------------------------------------------
# PortalEventData — payload for tp.orchestrator.portal_event.v1
# ---------------------------------------------------------------------------


class TestPortalEventData:
    def test_accepted_with_event_validates(self) -> None:
        record = {
            "schema": "tp.orchestrator.portal_event.v1",
            "timestamp": 1700000000,
            "event_type": "config_exported",
            "pipeline": "lux-depth-v3",
            "surface": "effective_config",
            "field": "reconstruction_tier",
            "metadata": {"mode": "auto"},
            "reasons": ["preview_ready"],
        }
        data = PortalEventData(accepted=True, event=record)
        dumped = data.model_dump(mode="json")
        assert dumped["accepted"] is True
        assert dumped["event"]["event_type"] == "config_exported"

    def test_extra_keys_are_preserved(self) -> None:
        data = PortalEventData(accepted=True, event={"x": 1}, future_field="y")
        dumped = data.model_dump(mode="json")
        assert dumped["future_field"] == "y"

    def test_accepted_literal_true_rejects_false(self) -> None:
        # portal_events never emits accepted=False; Literal[True] enforces this.
        with pytest.raises(ValidationError):
            PortalEventData(accepted=False, event={"x": 1})  # type: ignore[arg-type]

    def test_accepted_is_required(self) -> None:
        with pytest.raises(ValidationError):
            PortalEventData(event={"x": 1})  # type: ignore[call-arg]

    def test_event_is_required(self) -> None:
        with pytest.raises(ValidationError):
            PortalEventData(accepted=True)  # type: ignore[call-arg]


class TestPortalEventEnvelope:
    def test_envelope_round_trip(self) -> None:
        payload = PortalEventEnvelope(
            **{
                "schema": "tp.orchestrator.portal_event.v1",
                "success": True,
                "data": {"accepted": True, "event": {"event_type": "config_exported"}},
                "error": None,
            }
        )
        dumped = payload.model_dump(mode="json")
        assert dumped["schema"] == "tp.orchestrator.portal_event.v1"
        assert dumped["success"] is True
        assert dumped["data"]["accepted"] is True

    def test_envelope_validates_portal_event_data_shape(self) -> None:
        payload = PortalEventEnvelope(
            **{
                "schema": "tp.orchestrator.portal_event.v1",
                "success": True,
                "data": {"accepted": True, "event": {"event_type": "config_exported"}},
                "error": None,
            }
        )
        assert payload.data.accepted is True
        assert payload.data.event is not None
        assert payload.data.event["event_type"] == "config_exported"

        with pytest.raises(ValidationError):
            PortalEventEnvelope(
                **{
                    "schema": "tp.orchestrator.portal_event.v1",
                    "success": True,
                    "data": {"event": {"event_type": "config_exported"}},
                    "error": None,
                }
            )


# ---------------------------------------------------------------------------
# PortalRumIngestData — payload for tp.orchestrator.portal_rum_ingest.v1
# ---------------------------------------------------------------------------


class TestPortalRumIngestData:
    def test_disabled_path_validates(self) -> None:
        # Mirrors app.py portal_rum handler when RUM is disabled:
        # emits {"accepted": False, "disabled": True} — "event" key absent.
        data = PortalRumIngestData(accepted=False, disabled=True)
        dumped = data.model_dump(mode="json", exclude_none=True)
        assert dumped == {"accepted": False, "disabled": True}

    def test_accepted_path_with_event_validates(self) -> None:
        record = {
            "schema": "tp.orchestrator.portal_rum.v1",
            "timestamp": 1700000000,
            "event_type": "page_view",
            "route": "/portal",
            "view": "portal",
            "metric": "",
            "value": 0.0,
            "unit": "count",
            "metadata": {},
            "trace_id": "abc123",
            "cohort_bucket": 42,
            "auth_mode": "direct",
        }
        # Mirrors app.py portal_rum handler on the accepted path:
        # emits {"accepted": True, "event": record} — "disabled" key absent.
        data = PortalRumIngestData(accepted=True, event=record)
        dumped = data.model_dump(mode="json", exclude_none=True)
        assert dumped["accepted"] is True
        assert "disabled" not in dumped
        assert dumped["event"]["event_type"] == "page_view"

    def test_extra_keys_are_preserved(self) -> None:
        data = PortalRumIngestData(accepted=True, future_metric="p99")
        dumped = data.model_dump(mode="json")
        assert dumped["future_metric"] == "p99"

    def test_accepted_is_required(self) -> None:
        with pytest.raises(ValidationError):
            PortalRumIngestData(disabled=True)  # type: ignore[call-arg]


class TestPortalRumIngestEnvelope:
    def test_envelope_round_trip_disabled(self) -> None:
        payload = PortalRumIngestEnvelope(
            **{
                "schema": "tp.orchestrator.portal_rum_ingest.v1",
                "success": True,
                "data": {"accepted": False, "disabled": True},
                "error": None,
            }
        )
        dumped = payload.model_dump(mode="json")
        assert dumped["schema"] == "tp.orchestrator.portal_rum_ingest.v1"
        assert dumped["data"]["accepted"] is False
        assert dumped["data"]["disabled"] is True

    def test_envelope_round_trip_accepted(self) -> None:
        payload = PortalRumIngestEnvelope(
            **{
                "schema": "tp.orchestrator.portal_rum_ingest.v1",
                "success": True,
                "data": {
                    "accepted": True,
                    "event": {"event_type": "timing", "value": 1.23},
                },
                "error": None,
            }
        )
        dumped = payload.model_dump(mode="json")
        assert dumped["data"]["event"]["value"] == 1.23

    def test_envelope_validates_portal_rum_ingest_data_shape(self) -> None:
        payload = PortalRumIngestEnvelope(
            **{
                "schema": "tp.orchestrator.portal_rum_ingest.v1",
                "success": True,
                "data": {"accepted": True, "event": {"event_type": "timing", "value": 1.23}},
                "error": None,
            }
        )
        assert payload.data.accepted is True
        assert payload.data.event is not None
        assert payload.data.event["value"] == 1.23

        with pytest.raises(ValidationError):
            PortalRumIngestEnvelope(
                **{
                    "schema": "tp.orchestrator.portal_rum_ingest.v1",
                    "success": True,
                    "data": {"event": {"event_type": "timing"}},
                    "error": None,
                }
            )
