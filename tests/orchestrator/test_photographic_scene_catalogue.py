"""Lazy scene navigation preserves source authority and bounded selected IO."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import tifffile
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.responses import JSONResponse

from transformation_portal.api.routes.photographic_review import create_photographic_review_router
from transformation_portal.api.v1.photographic_review import (
    DeliveryHistoryIntent,
    DeliveryRequest,
    LightingGroupsRequest,
    ReviewDecisionRequest,
    RevisionRequest,
    SceneDetail,
    ScenePage,
)
from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.orchestrator.photographic_admission import ReviewAdmissionConfig, current_review_lease
from transformation_portal.orchestrator.photographic_review_access import (
    ReviewAccessError,
    ReviewContext,
    ReviewInputError,
    ReviewRuntime,
)
from transformation_portal.orchestrator.photographic_review_service import PhotographicReviewService
from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore, empty_state

pytestmark = [pytest.mark.unit]


@pytest.fixture
def scene_case(monkeypatch):
    from transformation_portal.orchestrator import photographic_review_service as services
    from transformation_portal.orchestrator import photographic_review_store as stores

    blobs = {}
    for index in range(50):
        encoded = io.BytesIO()
        tifffile.imwrite(
            encoded,
            np.full((6, 8, 3), 12000 + index, dtype=np.uint16),
            photometric="rgb",
            metadata=None,
            extratags=[(34675, "B", len(output_srgb_icc()), output_srgb_icc(), False)],
        )
        blobs[f"input-{index:04}"] = encoded.getvalue()
    refs = [
        {
            "artifact_path": f"v6/input-{index:04}/delivery.tif",
            "sha256": hashlib.sha256(blobs[f"input-{index:04}"]).hexdigest(),
            "source_identity_sha256": hashlib.sha256(blobs[f"input-{index:04}"]).hexdigest(),
            "size_bytes": len(blobs[f"input-{index:04}"]),
            "scene_id": f"input-{index:04}",
            "label": f"Room {49-index:02}",
            "role": "final",
            "source_name": "delivery.tif",
            "input_color": "auto",
            "kind": "indexed",
            "private_storage_key": "never-public",
        }
        for index in range(50)
    ]
    reads, selections = [], []
    controls = SimpleNamespace(revoked=False, deleted=False, binding="a" * 64)

    async def materialize(ref, destination, maximum_bytes):
        reads.append(ref["scene_id"])
        data = blobs[ref["scene_id"]]
        digest = hashlib.sha256(data).hexdigest()
        assert maximum_bytes >= len(data)
        destination.write_bytes(data)
        return {"sha256": digest, "size_bytes": len(data)}

    async def revalidate():
        if controls.revoked:
            raise ReviewAccessError("Access revoked", status=403, code="FORBIDDEN")
        if controls.deleted:
            raise ReviewAccessError("Artifacts deleted", status=410, code="ARTIFACT_DELETED")

    async def discover(selected):
        selections.append(set(selected))
        return [ref for ref in refs if ref["scene_id"] in selected]

    context = ReviewContext(
        "job-scenes",
        "tenant-scenes",
        {"username": "reviewer", "assurance": "authenticated"},
        [],
        materialize,
        AsyncMock(),
        revalidate=revalidate,
        discover_references=AsyncMock(side_effect=AssertionError("Do not eagerly discover the catalogue")),
        binding_identity=controls.binding,
        discover_scene_index=AsyncMock(side_effect=lambda: list(refs)),
        discover_scene_references=discover,
    )
    store = MemoryReviewStore()
    monkeypatch.setattr(stores, "get_review_store", lambda: store)
    monkeypatch.setattr(services, "get_review_store", lambda: store)
    return SimpleNamespace(context=context, store=store, refs=refs, reads=reads, selections=selections, controls=controls)


@pytest.mark.asyncio
async def test_fifty_scenes_page_without_any_decode_or_materialization(scene_case, monkeypatch):
    from transformation_portal.orchestrator import photographic_review_service as services

    case = scene_case
    monkeypatch.setattr(services, "load_image", AsyncMock(side_effect=AssertionError("Scene page must not decode")))
    service = PhotographicReviewService()
    seen, cursor, identity = [], None, None
    while True:
        page = await service.scenes(case.context, cursor=cursor)
        ScenePage.model_validate(page)
        assert page["schema"] == "tp.photo.scene_page.v1" and page["total_scenes"] == 50
        assert len(page["scenes"]) <= 8
        assert all(row["source_validation"] == "not_checked" and row["variant_count"] == 1 for row in page["scenes"])
        assert "never-public" not in json.dumps(page) and "private_storage_key" not in json.dumps(page)
        assert identity in {None, page["catalogue_identity_sha256"]}
        identity = page["catalogue_identity_sha256"]
        seen.extend(row["scene_id"] for row in page["scenes"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == [f"input-{index:04}" for index in range(50)]
    assert case.reads == case.selections == []
    case.context.discover_references.assert_not_awaited()
    await service.close()


@pytest.mark.asyncio
async def test_fifty_equal_labels_page_with_stable_scene_id_ties_outside_admission(scene_case, monkeypatch):
    case = scene_case
    for ref in case.refs:
        ref["label"] = "Same room"
    service = PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    monkeypatch.setattr(service.admission, "acquire", AsyncMock(side_effect=AssertionError("List must not acquire scarce IO")))
    seen, cursor = [], None
    while True:
        page = await service.scenes(case.context, sort="label", limit=16, cursor=cursor)
        seen.extend(row["scene_id"] for row in page["scenes"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == [f"input-{index:04}" for index in range(50)]
    assert not case.reads and not case.selections
    await service.close()


@pytest.mark.asyncio
async def test_sixty_four_scene_navigation_bound_is_explicit_and_does_not_decode(scene_case):
    case = scene_case
    for index in range(50, 65):
        case.refs.append(
            {**case.refs[0], "artifact_path": f"v6/input-{index:04}/delivery.tif", "scene_id": f"input-{index:04}"}
        )
    overflow = case.refs.pop()
    service = PhotographicReviewService()
    page = await service.scenes(case.context, limit=16)
    assert page["total_scenes"] == 64
    ScenePage.model_validate(page)
    case.refs.append(overflow)
    with pytest.raises(ReviewInputError, match="64 scenes"):
        await service.scenes(case.context)
    assert not case.reads
    await service.close()


@pytest.mark.asyncio
async def test_public_variant_paths_resolve_exact_original_selector_without_disclosing_private_bindings(scene_case):
    case = scene_case
    original_path = "original/input-0031.tif"
    case.refs.append(
        {
            **case.refs[31],
            "artifact_path": original_path,
            "role": "original",
            "source_name": "original.tif",
            "private_original_root": "/hidden/tenant/input",
            "private_original_path": "private/source.tif",
        }
    )
    service = PhotographicReviewService()
    page, cursor, selected = None, None, None
    while selected is None:
        page = await service.scenes(case.context, cursor=cursor)
        ScenePage.model_validate(page)
        selected = next((row for row in page["scenes"] if original_path in row["variant_paths"]), None)
        cursor = page["next_cursor"]
        assert selected is not None or cursor is not None
        serialized = json.dumps(page)
        assert "/hidden/tenant" not in serialized and "private/source.tif" not in serialized
        assert "private_original" not in serialized and "never-public" not in serialized
    assert selected["scene_id"] == "input-0031" and not case.reads
    detail = await service.execute(case.context, service.scene_detail, selected["scene_id"], page["catalogue_identity_sha256"])
    assert any(
        variant["artifact_path"] == original_path and variant["role"] == "original"
        for variant in detail["scenes"][0]["variants"]
    )
    assert set(case.reads) == {"input-0031"}
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_selected_scene_detail_and_decision_only_discover_selected_scene(scene_case, enabled):
    case = scene_case
    service = PhotographicReviewService(ReviewAdmissionConfig(enabled=enabled))
    page = await service.scenes(case.context)
    detail = await service.execute(
        service_context := case.context, service.scene_detail, "input-0049", page["catalogue_identity_sha256"]
    )
    SceneDetail.model_validate(detail)
    assert [scene["scene_id"] for scene in detail["scenes"]] == ["input-0049"]
    assert case.reads == ["input-0049"] and case.selections == [{"input-0049"}]
    ref = case.refs[-1]
    result = await service.execute(
        service_context,
        service.decision,
        ReviewDecisionRequest(artifact_path=ref["artifact_path"], sha256=ref["sha256"], status="approved", expected_version=0),
    )
    assert result["state_version"] == 1 and case.reads == ["input-0049", "input-0049"]
    assert case.selections == [{"input-0049"}, {"input-0049"}]
    with pytest.raises(ReviewAccessError) as stale:
        await service.execute(case.context, service.scene_detail, "input-0049", page["catalogue_identity_sha256"])
    assert stale.value.code == "CATALOGUE_CHANGED" and len(case.reads) == 2
    approved = await service.scenes(case.context, status="approved")
    assert approved["total_filtered"] == 1 and approved["scenes"][0]["scene_id"] == "input-0049"
    assert approved["scenes"][0]["recorded_decision"]["status"] == "approved"
    assert approved["scenes"][0]["source_validation"] == "not_checked"
    assert current_review_lease.get() is None and service.admission.snapshot()["active"] == 0
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,status", [("revoked", 403), ("deleted", 410)])
async def test_warmed_scene_inspection_never_bypasses_later_permission_or_deletion(scene_case, failure, status):
    case = scene_case
    service = PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    page = await service.scenes(case.context)
    await service.execute(case.context, service.scene_detail, "input-0000", page["catalogue_identity_sha256"])
    ref = case.refs[0]
    assert service._inspection_costs
    setattr(case.controls, failure, True)
    with pytest.raises(ReviewAccessError) as caught:
        await service.execute(case.context, service.render, ref["artifact_path"], ref["sha256"])
    assert caught.value.status == status and case.reads == ["input-0000"]
    assert service.admission.snapshot()["active"] == 0
    await service.close()


@pytest.mark.asyncio
async def test_render_revision_groups_and_delivery_discover_only_explicitly_selected_sources(scene_case):
    case = scene_case
    service = PhotographicReviewService()
    left, right = case.refs[:2]
    await service.execute(case.context, service.render, left["artifact_path"], left["sha256"])
    assert case.selections[-1] == {left["scene_id"]}
    for version, ref in enumerate((left, right)):
        await service.execute(
            case.context,
            service.decision,
            ReviewDecisionRequest(
                artifact_path=ref["artifact_path"], sha256=ref["sha256"], status="approved", expected_version=version
            ),
        )
    await service.execute(
        case.context,
        service.groups,
        LightingGroupsRequest.model_validate(
            {
                "expected_version": 2,
                "groups": [
                    {"name": "Selected", "reference_sha256": right["sha256"], "members": [left["sha256"], right["sha256"]]}
                ],
            }
        ),
    )
    assert case.selections[-1] == {left["scene_id"], right["scene_id"]}
    await service.execute(
        case.context,
        service.revision,
        RevisionRequest(
            artifact_path=left["artifact_path"],
            sha256=left["sha256"],
            scene_id=left["scene_id"],
            expected_version=3,
            recipe={"exposure_stops": -0.1},
            reference_sha256=right["sha256"],
            preview_only=True,
        ),
    )
    assert case.selections[-1] == {left["scene_id"], right["scene_id"]}
    delivery = await service.execute(
        case.context,
        service.delivery,
        DeliveryRequest.model_validate(
            {
                "expected_version": 3,
                "property_name": "Selected scenes",
                "presets": ["website"],
                "approved_sources": [
                    {"artifact_path": ref["artifact_path"], "sha256": ref["sha256"]} for ref in (left, right)
                ],
            }
        ),
    )
    assert delivery["state_version"] == 4
    assert case.selections[-1] == {left["scene_id"], right["scene_id"]}
    assert set(case.reads) == {left["scene_id"], right["scene_id"]}
    case.context.discover_references.assert_not_awaited()
    await service.close()


@pytest.mark.asyncio
async def test_mutation_projection_never_treats_native_inventory_hash_as_composite_authority(scene_case):
    case = scene_case
    native, selected = case.refs[:2]
    state = empty_state()
    state["decisions"] = [{"artifact_path": native["artifact_path"], "sha256": native["sha256"], "status": "approved"}]
    case.store.states[case.context.job_id, case.context.tenant_id] = state
    index = [{**native, "role": "working_master"}, *case.refs[1:]]
    context = replace(
        case.context,
        discover_scene_index=AsyncMock(return_value=index),
        discover_scene_references=AsyncMock(
            side_effect=lambda scenes: [ref for ref in case.refs[1:] if ref["scene_id"] in scenes]
        ),
    )
    service = PhotographicReviewService()
    try:
        result = await service.execute(
            context,
            service.decision,
            ReviewDecisionRequest(
                artifact_path=selected["artifact_path"], sha256=selected["sha256"], status="approved", expected_version=0
            ),
        )
        assert [row["artifact_path"] for row in result["decisions"]] == [selected["artifact_path"]]
        assert case.reads == [selected["scene_id"]]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_mutation_projection_keeps_selected_fresh_legacy_identity_without_promoting_unknown_other_sources(scene_case):
    case = scene_case
    selected = case.refs[0]
    index = [{**ref, "sha256": None, "source_identity_sha256": None} for ref in case.refs]
    context = replace(case.context, discover_scene_index=AsyncMock(return_value=index))
    service = PhotographicReviewService()
    try:
        result = await service.execute(
            context,
            service.decision,
            ReviewDecisionRequest(
                artifact_path=selected["artifact_path"], sha256=selected["sha256"], status="approved", expected_version=0
            ),
        )
        assert [row["artifact_path"] for row in result["decisions"]] == [selected["artifact_path"]]
        assert case.selections == [{selected["scene_id"]}] and case.reads == [selected["scene_id"]]
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["decision", "groups", "revision"])
@pytest.mark.parametrize("stale", ["changed", "removed"])
async def test_selected_mutations_preserve_all_current_decisions_without_reading_unselected_images(
    scene_case, operation, stale
):
    case = scene_case
    service = PhotographicReviewService()
    left, selected, outdated = case.refs[:3]
    try:
        for version, ref in enumerate((left, selected, outdated)):
            await service.execute(
                case.context,
                service.decision,
                ReviewDecisionRequest(
                    artifact_path=ref["artifact_path"], sha256=ref["sha256"], status="approved", expected_version=version
                ),
            )
        if stale == "changed":
            outdated["sha256"] = outdated["source_identity_sha256"] = "f" * 64
        else:
            case.refs.remove(outdated)
        case.reads.clear()
        case.selections.clear()
        if operation == "decision":
            payload = ReviewDecisionRequest(
                artifact_path=selected["artifact_path"], sha256=selected["sha256"], status="approved", expected_version=3
            )
        elif operation == "groups":
            payload = LightingGroupsRequest.model_validate(
                {
                    "expected_version": 3,
                    "groups": [{"name": "Selected", "reference_sha256": selected["sha256"], "members": [selected["sha256"]]}],
                }
            )
        else:
            payload = RevisionRequest(
                artifact_path=selected["artifact_path"],
                sha256=selected["sha256"],
                scene_id=selected["scene_id"],
                expected_version=3,
                recipe={"exposure_stops": -0.1},
                preview_only=False,
            )
        result = await service.execute(case.context, getattr(service, operation), payload)
        assert result["state_version"] == 4
        assert {row["artifact_path"] for row in result["decisions"]} == {left["artifact_path"], selected["artifact_path"]}
        assert len((await case.store.get(case.context.job_id, case.context.tenant_id))["decisions"]) == 3
        assert case.selections == [{selected["scene_id"]}]
        assert set(case.reads) == {selected["scene_id"]}
        case.context.discover_references.assert_not_awaited()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_saved_native_scene_hints_cannot_discover_scenes_outside_current_authorized_index(scene_case):
    case = scene_case
    state = empty_state()
    members = ["c" * 64, "d" * 64]
    state["cohesion"] = {
        "schema": "tp.photo.cohesion_diagnostics.v1",
        "groups": [
            {
                "images": [
                    {"source_identity_sha256": members[0], "scene": "input-0000"},
                    {"source_identity_sha256": members[1], "scene": "outside-authorized-index"},
                ]
            }
        ],
    }
    case.store.states[case.context.job_id, case.context.tenant_id] = state
    payload = LightingGroupsRequest.model_validate(
        {"expected_version": 0, "groups": [{"name": "Saved", "reference_sha256": members[0], "members": members}]}
    )
    service = PhotographicReviewService()
    resolved = await service._discover_selected(case.context, "groups", (payload,))
    assert case.selections == [{"input-0000"}]
    assert [row["scene_id"] for row in resolved.references] == ["input-0000"]
    assert not case.reads
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["actor", "tenant", "job", "inventory", "state", "filter", "sort", "limit"])
async def test_cursor_is_bound_to_scope_inventory_review_state_and_query(scene_case, change):
    case = scene_case
    service = PhotographicReviewService()
    first = await service.scenes(case.context)
    context, kwargs = case.context, {}
    if change == "actor":
        context = replace(context, actor={"username": "other"})
    elif change == "tenant":
        context = replace(context, tenant_id="other")
    elif change == "job":
        context = replace(context, job_id="other")
    elif change == "inventory":
        case.refs[0]["sha256"] = "b" * 64
    elif change == "state":
        await case.store.mutate(context.job_id, context.tenant_id, 0, lambda state: state, {"kind": "test"})
    else:
        kwargs[{"filter": "status", "sort": "sort", "limit": "limit"}[change]] = {
            "filter": "unreviewed",
            "sort": "label",
            "limit": 9,
        }[change]
    with pytest.raises(ReviewAccessError) as caught:
        await service.scenes(context, cursor=first["next_cursor"], **kwargs)
    assert caught.value.code == "CATALOGUE_CHANGED" and caught.value.status == 409
    assert not case.reads
    await service.close()


@pytest.mark.asyncio
async def test_cursor_tampering_malformed_input_and_service_restart_fail_closed(scene_case):
    service = PhotographicReviewService()
    page = await service.scenes(scene_case.context)
    cursor = page["next_cursor"]
    body, signature = cursor.split(".")
    changed = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    changed["offset"] = 40
    forged = base64.urlsafe_b64encode(json.dumps(changed).encode()).decode().rstrip("=") + "." + signature
    for value in ["not-a-cursor", "", "x" * 1025, forged, body + "." + "0" * 64]:
        with pytest.raises(ReviewInputError):
            await service.scenes(scene_case.context, cursor=value)
    restarted = PhotographicReviewService()
    with pytest.raises(ReviewAccessError) as caught:
        await restarted.scenes(scene_case.context, cursor=cursor)
    assert caught.value.code == "CATALOGUE_CHANGED"
    await service.close()
    await restarted.close()


@pytest.mark.asyncio
async def test_label_sort_filter_and_revision_reset_recorded_approval(scene_case):
    case = scene_case
    ref = case.refs[0]
    state = empty_state()
    state["decisions"] = [{"artifact_path": ref["artifact_path"], "sha256": ref["sha256"], "status": "approved"}]
    case.store.states[case.context.job_id, case.context.tenant_id] = state
    service = PhotographicReviewService()
    first = await service.scenes(case.context, sort="label")
    assert first["scenes"][0]["scene_id"] == "input-0049"
    approved = await service.scenes(case.context, status="approved")
    assert approved["total_filtered"] == 1
    state["revisions"] = [{"variant": {**ref, "artifact_path": "revisions/new/master.tif", "sha256": "c" * 64}}]
    revised_page = await service.scenes(case.context)
    assert "revisions/new/master.tif" in revised_page["scenes"][0]["variant_paths"]
    assert (await service.scenes(case.context, status="approved"))["total_filtered"] == 0
    assert (await service.scenes(case.context, status="unreviewed"))["total_filtered"] == 50
    with pytest.raises(ReviewAccessError) as changed:
        await service.scenes(case.context, sort="label", cursor=first["next_cursor"])
    assert changed.value.code == "CATALOGUE_CHANGED"
    await service.close()


@pytest.mark.asyncio
async def test_revocation_and_mutation_during_page_prevent_response(scene_case):
    case = scene_case
    service = PhotographicReviewService()
    original_index = case.context.discover_scene_index

    async def revoke():
        case.controls.revoked = True
        return await original_index()

    with pytest.raises(ReviewAccessError) as caught:
        await service.scenes(replace(case.context, discover_scene_index=revoke))
    assert caught.value.status == 403
    case.controls.revoked = False

    async def mutate():
        await case.store.mutate(case.context.job_id, case.context.tenant_id, 0, lambda state: state, {"kind": "test"})
        return await original_index()

    with pytest.raises(ReviewAccessError) as caught:
        await service.scenes(replace(case.context, discover_scene_index=mutate))
    assert caught.value.code == "CATALOGUE_CHANGED" and not case.reads
    await service.close()


@pytest.mark.asyncio
async def test_legacy_catalogue_explicitly_requires_paging_before_any_image_decode(scene_case):
    case = scene_case
    service = PhotographicReviewService()
    with pytest.raises(ReviewAccessError) as caught:
        await service.catalogue(replace(case.context, references=case.refs))
    assert caught.value.code == "PAGED_CATALOGUE_REQUIRED" and case.reads == []
    legacy = await service.catalogue(replace(case.context, references=case.refs[:1]))
    assert legacy["scenes"][0]["scene_id"] == "input-0000" and "schema" not in legacy
    await service.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_paged_http_aliases_authorization_validation_and_conflicts(scene_case, version):
    case = scene_case

    def error(status, **kwargs):
        return JSONResponse({"error": kwargs}, status_code=status)

    async def authorize(request, job_id):
        if request.headers.get("x-api-key") != "test-key":
            return error(401, code="UNAUTHORIZED")
        return case.context

    app = FastAPI()
    app.include_router(create_photographic_review_router(ReviewRuntime(authorize, error, authorize_catalogue=authorize)))
    base = f"/{version}/jobs/job-scenes/photographic-review"
    with TestClient(app) as client:
        assert client.get(base + "/scenes").status_code == 401
        assert (
            client.get(base + "/scene", params={"scene_id": "input-0000", "catalogue_identity_sha256": "a" * 64}).status_code
            == 401
        )
        client.headers["x-api-key"] = "test-key"
        page = client.get(base + "/scenes")
        assert page.status_code == 200 and page.headers["cache-control"] == "no-store"
        data = page.json()["data"]
        assert client.get(base + "/scenes", params={"limit": 17}).status_code == 422
        assert client.get(base + "/scenes", params={"status": "selected"}).status_code == 422
        assert client.get(base + "/scenes", params={"cursor": "invalid"}).status_code == 400
        detail = client.get(
            base + "/scene", params={"scene_id": "input-0000", "catalogue_identity_sha256": data["catalogue_identity_sha256"]}
        )
        assert detail.status_code == 200 and detail.json()["data"]["schema"] == "tp.photo.scene_detail.v1"
        stale = client.get(base + "/scene", params={"scene_id": "input-0000", "catalogue_identity_sha256": "0" * 64})
        assert stale.status_code == 409 and stale.json()["error"]["code"] == "CATALOGUE_CHANGED"


def test_scenes_requires_explicit_cheap_adapter_without_calling_eager_authorization():
    eager = AsyncMock(side_effect=AssertionError("Do not eagerly discover source descriptors for a scene page"))

    def error(status, **kwargs):
        return JSONResponse({"error": kwargs}, status_code=status)

    app = FastAPI()
    app.include_router(create_photographic_review_router(ReviewRuntime(eager, error)))
    with TestClient(app) as client:
        response = client.get("/v1/jobs/job-scenes/photographic-review/scenes")
        assert response.status_code == 503 and response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    eager.assert_not_awaited()


def test_scene_related_payload_bounds_align_to_sixty_four_without_raising_other_limits():
    hashes = [f"{index:064x}" for index in range(64)]
    groups = {"expected_version": 0, "groups": [{"name": "All", "reference_sha256": hashes[0], "members": hashes}]}
    assert len(LightingGroupsRequest.model_validate(groups).groups[0].members) == 64
    sources = [{"artifact_path": f"v6/input-{index:04}/delivery.tif", "sha256": digest} for index, digest in enumerate(hashes)]
    payload = {
        "expected_version": 0,
        "property_name": "Property",
        "presets": ["web"],
        "approved_sources": sources,
        "scene_labels": {str(index): str(index) for index in range(64)},
    }
    assert len(DeliveryRequest.model_validate(payload).approved_sources) == 64
    sources.append({"artifact_path": "overflow.tif", "sha256": "f" * 64})
    with pytest.raises(ValidationError):
        DeliveryRequest.model_validate(payload)
    assert DeliveryHistoryIntent.model_json_schema()["properties"]["sources"]["minItems"] == 1
    assert DeliveryHistoryIntent.model_json_schema()["properties"]["sources"]["maxItems"] == 64
