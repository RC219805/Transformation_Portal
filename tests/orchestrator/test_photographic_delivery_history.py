"""Committed delivery discovery is bounded, redacted and independent of image IO."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from transformation_portal.api.v1.photographic_review import DeliveryRequest
from transformation_portal.orchestrator.photographic_delivery_history import applied_derivatives, delivery_intent, history_page
from transformation_portal.orchestrator.photographic_review_access import ReviewContext, ReviewInputError
from transformation_portal.orchestrator.photographic_review_store import ReviewConflict, empty_state
from transformation_portal.photographic_delivery import ExportPreset

pytestmark = pytest.mark.unit
SHA = "a" * 64


def _context():
    return ReviewContext("job", "tenant", {"username": "reviewer"}, [], AsyncMock(), AsyncMock())


def _delivery(index=0):
    approval = {"artifact_path": "final.tif", "sha256": SHA, "status": "approved"}
    return {
        "id": f"delivery-{index}",
        "created_at": float(index),
        "actor": {"username": "reviewer", "private_path": "/private/secret"},
        "approvals": [approval],
        "approved_sha256": [SHA],
        "files": [
            {
                "filename": f"delivery-{index}.zip",
                "content_type": "application/zip",
                "sha256": SHA,
                "size_bytes": 20,
                "storage_path": "/private/secret",
                "private_original_root": "/secret",
            }
        ],
        "inventory": {"private": "/private/secret"},
        "receipt": {"source": "/private/secret"},
        "unrecognized": {"secret": "/private/secret"},
    }


def _state(count):
    return {
        **empty_state(),
        "state_version": 70,
        "decisions": [{"artifact_path": "final.tif", "sha256": SHA, "status": "approved"}],
        "deliveries": [_delivery(index) for index in range(count)],
    }


@pytest.mark.parametrize("count", [0, 1, 20, 21, 64])
def test_history_fixed_prefix_covers_every_committed_delivery_once_with_concurrent_append(count):
    state, context = _state(count), _context()
    page = history_page(context, state)
    ids = [row["id"] for row in page["items"]]
    cursor = page["next_cursor"]
    if count < 64:
        state["deliveries"].append(_delivery(count))
    while cursor:
        page = history_page(context, state, cursor=cursor)
        ids.extend(row["id"] for row in page["items"])
        cursor = page["next_cursor"]
    assert ids == [f"delivery-{index}" for index in reversed(range(count))]
    assert len(ids) == len(set(ids))
    assert context.materialize.await_count == context.publish.await_count == 0


def test_history_explicit_dto_redacts_private_fields_and_distinguishes_recorded_approval():
    state = _state(1)
    row = history_page(_context(), state)["items"][0]
    assert "/private/secret" not in str(row) and "storage_path" not in str(row)
    assert row["request_fingerprint_sha256"] is row["intent"] is row["approval_version"] is None
    assert row["sources"][0]["artifact_path"] == "final.tif"
    assert row["applied_derivatives"] is row["applied_derivatives_schema"] is None
    assert row["approval_at_creation"] == "recorded"
    assert row["current_recorded_approval"] == "approved"
    assert row["source_validation"] == row["file_validation"] == "not_checked"
    state["decisions"][0]["status"] = "needs_revision"
    changed = history_page(_context(), state)["items"][0]
    assert changed["approval_at_creation"] == "recorded" and changed["current_recorded_approval"] == "withdrawn"


@pytest.mark.parametrize("limit", [0, 65, True, -1])
def test_history_rejects_invalid_limits(limit):
    with pytest.raises(ReviewInputError):
        history_page(_context(), _state(1), limit=limit)


@pytest.mark.asyncio
async def test_history_pagination_does_not_raise_existing_delivery_creation_cap():
    from transformation_portal.orchestrator.photographic_review_service import PhotographicReviewService

    state = _state(64)
    context = replace(
        _context(), references=[{"artifact_path": "final.tif", "sha256": SHA, "scene_id": "scene", "label": "Room"}]
    )
    service = PhotographicReviewService()
    service.state = AsyncMock(return_value=state)
    payload = DeliveryRequest(property_name="House", expected_version=70, presets=["website"], approved_sha256=[SHA])
    with pytest.raises(ReviewInputError, match="bounded to 64"):
        await service.delivery(context, payload)
    context.materialize.assert_not_awaited()


def test_history_cursor_is_bound_to_scope_filter_and_unchanged_prefix():
    context, state = _context(), _state(21)
    cursor = history_page(context, state)["next_cursor"]
    for other in (
        replace(context, job_id="other"),
        replace(context, tenant_id="other"),
        replace(context, actor={"username": "other"}),
    ):
        with pytest.raises(ReviewInputError):
            history_page(other, state, cursor=cursor)
    with pytest.raises(ReviewInputError):
        history_page(context, state, cursor=cursor, request_correlation_id="different")
    state["deliveries"][0]["id"] = "replaced"
    with pytest.raises(ReviewConflict):
        history_page(context, state, cursor=cursor)


@pytest.mark.parametrize("cursor", ["!", "x" * 1025, "W10", "bnVsbA", "e30"])
def test_history_rejects_malformed_cursor(cursor):
    with pytest.raises(ReviewInputError):
        history_page(_context(), _state(1), cursor=cursor)


def test_history_correlation_filter_does_not_pick_an_arbitrary_duplicate():
    state = _state(3)
    for row in state["deliveries"][:2]:
        row["request_correlation_id"] = "attempt"
    rows = history_page(_context(), state, request_correlation_id="attempt")["items"]
    assert [row["id"] for row in rows] == ["delivery-1", "delivery-0"]


def test_history_redacts_future_intent_fields_without_rewriting_storage():
    state, context = _state(1), _context()
    payload = DeliveryRequest(property_name="House", expected_version=3, presets=["website"], approved_sha256=[SHA])
    intent, digest = delivery_intent(
        context,
        payload,
        [{"artifact_path": "final.tif", "sha256": SHA, "scene_id": "scene"}],
        [ExportPreset.from_payload("website")],
    )
    intent["private_future_field"] = "/secret"
    intent["sources"][0]["private_future_field"] = "/secret"
    intent["presets"][0]["private_future_field"] = "/secret"
    state["deliveries"][0].update(
        intent=intent, request_fingerprint_schema="tp.photo.delivery_intent.v1", request_fingerprint_sha256=digest
    )
    before = deepcopy(state)
    assert "/secret" not in str(history_page(context, state))
    assert state == before
    state["deliveries"][0]["request_fingerprint_schema"] = "tp.photo.delivery_intent.v2"
    row = history_page(context, state)["items"][0]
    assert row["intent"] is row["request_fingerprint_schema"] is row["request_fingerprint_sha256"] is None


def test_server_intent_expands_defaults_and_binds_auth_scope_version_order_labels_and_presentation():
    context = _context()
    payload = DeliveryRequest(property_name="House", expected_version=3, presets=["website"], approved_sha256=[SHA])
    refs = [
        {"artifact_path": "final.tif", "sha256": SHA, "scene_id": "scene", "label": "Room"},
        {"artifact_path": "other.tif", "sha256": "b" * 64, "scene_id": "other", "label": "Other"},
    ]
    presets = [ExportPreset.from_payload("website"), ExportPreset.from_payload("print")]
    public, expected = delivery_intent(context, payload, refs, presets)
    assert public["sources"][0]["source_identity_sha256"] == SHA
    assert public["sources"][0]["presentation_policy_sha256"] is None
    assert "scope" not in public and "actor" not in public
    equivalent = ExportPreset.from_payload({"name": "website", "jpeg_quality": 92})
    assert delivery_intent(context, payload, refs, [equivalent, presets[1]])[1] == expected
    assert (
        delivery_intent(context, payload.model_copy(update={"request_correlation_id": "transport-only"}), refs, presets)[1]
        == expected
    )
    assert delivery_intent(context, payload.model_copy(update={"expected_version": 4}), refs, presets)[1] != expected
    assert (
        delivery_intent(context, payload.model_copy(update={"scene_labels": {"scene": "Changed"}}), refs, presets)[1]
        != expected
    )
    for changes in ({"job_id": "other"}, {"tenant_id": "other"}, {"actor": {"username": "other"}}):
        assert delivery_intent(replace(context, **changes), payload, refs, presets)[1] != expected
    assert delivery_intent(context, payload, refs[::-1], presets)[1] != expected
    assert delivery_intent(context, payload, refs, presets[::-1])[1] != expected
    for key in ("sha256", "source_identity_sha256", "presentation_policy_sha256"):
        changed = deepcopy(refs)
        changed[0][key] = "c" * 64
        assert delivery_intent(context, payload, changed, presets)[1] != expected


@pytest.mark.parametrize("preserve", [False, True])
def test_applied_derivative_summary_records_actual_render_separately_from_requested_preset(preserve):
    recipe = ExportPreset.from_payload("website").to_payload()
    row = {
        "source_master_sha256": SHA,
        "source_identity_sha256": SHA,
        "sha256": "b" * 64,
        "presentation_policy_sha256": "c" * 64,
        "preset": "website",
        "width": 64,
        "height": 32,
        "bit_depth": 8,
        "color_domain": "display_encoded_srgb",
        "recipe": recipe,
        "render": {
            "schema": "tp.photo.presentation_render.v1" if preserve else "tp.lux.sdr_render_receipt.v1",
            "recipe": {"schema": "tp.lux.sdr_render.v1", "mode": "soft_srgb", "shoulder": 0.85},
            "private_path": "/secret",
        },
    }
    if preserve:
        row["render"].update(operation="preserve_recorded_display", presentation_policy_sha256="c" * 64)
    summary = applied_derivatives({"files": [row]})[0]
    assert summary["recipe"]["render"] == "perceptual_srgb"
    assert summary["render"]["mode"] == "soft_srgb"
    assert summary["operation"] == ("preserve_recorded_display" if preserve else "render_sdr")
    assert summary["presentation_policy_sha256"] == "c" * 64
    assert "/secret" not in str(summary)
