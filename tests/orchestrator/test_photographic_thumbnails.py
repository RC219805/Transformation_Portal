"""Fixed catalogue thumbnails preserve source authority and inspection capacity."""

from __future__ import annotations

import asyncio
import io
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from transformation_portal.api.routes.photographic_review import create_photographic_review_router
from transformation_portal.api.v1.photographic_review import ScenePage
from transformation_portal.orchestrator import photographic_admission as admission
from transformation_portal.orchestrator import photographic_review_service as review
from transformation_portal.orchestrator.photographic_admission import (
    GIB,
    ReviewAdmissionConfig,
    ReviewOverload,
    current_review_lease,
    operation_cost,
)
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewRuntime
from transformation_portal.orchestrator.photographic_scene_catalogue import thumbnail_source

from .test_photographic_scene_catalogue import scene_case  # noqa: F401 - shared verified TIFF fixture

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def thumbnail_case(scene_case, monkeypatch):
    monkeypatch.setenv("TP_PHOTO_REVIEW_THUMBNAILS_ENABLED", "1")
    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    service = review.PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    for ref in scene_case.refs:
        # This synthetic indexed source has known absence of recorded policy.
        ref["presentation_policy_sha256"] = None
    page = await service.scenes(scene_case.context)
    try:
        yield SimpleNamespace(**vars(scene_case), service=service, page=page, identity=page["catalogue_identity_sha256"])
    finally:
        await service.close()


async def read_thumbnail(case, context=None, identity=None, **kwargs):
    return await case.service.execute(
        context or case.context, case.service.thumbnail, "input-0000", identity or case.identity, **kwargs
    )


async def test_thumbnail_uses_one_scene_verified_png_and_metadata_stays_lazy(thumbnail_case):
    case = thumbnail_case
    ScenePage.model_validate(case.page)
    assert not case.reads and not case.selections
    hint = case.page["scenes"][0]["thumbnail"]
    assert hint["url"].endswith("/thumbnail") and hint["sha256"] == case.refs[0]["sha256"]
    response = await read_thumbnail(case)
    assert case.reads == ["input-0000"] and case.selections == [{"input-0000"}]
    assert response.media_type == "image/png" and len(response.body) <= 1024**2
    with Image.open(io.BytesIO(response.body)) as image:
        assert image.size == (8, 6) and image.info["icc_profile"]
    headers = response.headers
    assert headers["cache-control"] == "no-store" and headers["x-content-type-options"] == "nosniff"
    assert headers["x-tp-source-sha256"] == headers["x-tp-source-identity-sha256"] == hint["sha256"]
    assert headers["x-tp-catalogue-identity-sha256"] == case.identity
    assert unquote(headers["x-tp-artifact-path"]) == hint["artifact_path"]
    assert unquote(headers["x-tp-scene-id"]) == "input-0000"
    assert headers["x-tp-native-width"] == "8" and headers["x-tp-native-height"] == "6"
    assert headers["x-tp-thumbnail-max-edge"] == "256" and headers["x-tp-thumbnail-max-bytes"] == "1048576"
    assert headers["x-tp-thumbnail-renderer"] == "tp.photo.thumbnail.v1"
    assert headers["x-tp-presentation-view"] == "clip_srgb" and headers["x-tp-presentation-policy-sha256"] == "none"
    assert case.service.admission.snapshot()["active"] == 0
    case.context.publish.assert_not_called()


async def test_explicit_thumbnail_cost_preserves_full_render_vector_and_bulk_lane(thumbnail_case):
    case = thumbnail_case
    state = await case.service.state(case.context)
    thumbnail = case.service._resource_cost(case.context, state, "thumbnail", ("input-0000", case.identity), None)
    render = case.service._resource_cost(case.context, state, "render", ("image", "a" * 64), None)
    for name in ("memory_bytes", "scratch_bytes", "decode_slots", "io_slots", "max_pixels", "filesystems"):
        assert getattr(thumbnail, name) == getattr(render, name)
    assert not thumbnail.interactive and render.interactive
    with pytest.raises(ValueError, match="Unknown photographic operation"):
        operation_cost("thumbnail-small-untrusted", case.service.admission.config)


async def test_thumbnail_and_native_bulk_leave_inspection_capacity(thumbnail_case):
    controller = thumbnail_case.service.admission
    native = operation_cost(
        "revision", controller.config, [{"width": 8, "height": 6, "alpha": False}], native_materials_regions=1
    )
    active = await controller.acquire("tenant-a", native)
    waiting = asyncio.create_task(controller.acquire("tenant-b", operation_cost("thumbnail", controller.config)))
    interactive = None
    try:
        await asyncio.sleep(0)
        assert not waiting.done()
        interactive = await controller.acquire("tenant-a", operation_cost("render", controller.config))
        assert not waiting.done() and controller.snapshot()["active"] == 2
        assert controller.snapshot()["memory_reserved_bytes"] <= controller.config.memory_bytes
        await interactive.release()
        interactive = None
        await active.release()
        thumbnail = await asyncio.wait_for(waiting, 2)
        await thumbnail.release()
        assert controller.snapshot()["active"] == controller.snapshot()["queued"] == 0
    finally:
        if interactive is not None:
            await interactive.release()
        await active.release()
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_disabled_flag_and_disabled_admission_never_read(thumbnail_case, monkeypatch):
    case = thumbnail_case
    monkeypatch.delenv("TP_PHOTO_REVIEW_THUMBNAILS_ENABLED")
    with pytest.raises(ReviewOverload, match="capacity") as error:
        await read_thumbnail(case)
    assert error.value.reason == "thumbnail_feature_disabled"
    monkeypatch.setenv("TP_PHOTO_REVIEW_THUMBNAILS_ENABLED", "1")
    service = review.PhotographicReviewService(ReviewAdmissionConfig(enabled=False))
    try:
        with pytest.raises(ReviewOverload) as error:
            await service.execute(case.context, service.thumbnail, "input-0000", case.identity)
        assert error.value.reason == "thumbnail_admission_required"
    finally:
        await service.close()
    assert not case.reads and not case.selections


async def test_stale_catalogue_blocks_before_source_discovery(thumbnail_case):
    case = thumbnail_case
    case.refs[0]["sha256"] = "f" * 64
    with pytest.raises(ReviewAccessError) as error:
        await read_thumbnail(case)
    assert error.value.code == "CATALOGUE_CHANGED"
    assert not case.reads and not case.selections


@pytest.mark.parametrize("change", ["catalogue", "authorization"])
async def test_render_completion_rechecks_catalogue_and_authorization(thumbnail_case, monkeypatch, change):
    case = thumbnail_case
    original = review._display_with_geometry

    def display(*args):
        result = original(*args)
        if change == "catalogue":
            case.refs[0]["label"] = "changed"
        else:
            case.controls.revoked = True
        return result

    monkeypatch.setattr(review, "_display_with_geometry", display)
    with pytest.raises(ReviewAccessError) as error:
        await read_thumbnail(case)
    assert error.value.code == ("CATALOGUE_CHANGED" if change == "catalogue" else "FORBIDDEN")
    assert case.service.admission.snapshot()["active"] == 0


async def test_recorded_view_without_policy_fails_closed(thumbnail_case):
    with pytest.raises(ReviewAccessError) as error:
        await read_thumbnail(thumbnail_case, view="recorded")
    assert error.value.code == "PRESENTATION_UNAVAILABLE"


async def test_encoded_response_bound_is_enforced(thumbnail_case, monkeypatch):
    monkeypatch.setattr(review, "_display_with_geometry", lambda *_: (b"x" * (1024**2 + 1), 256, 256, 8000, 5000))
    with pytest.raises(ReviewOverload) as error:
        await read_thumbnail(thumbnail_case)
    assert error.value.status == 413 and error.value.reason == "thumbnail_response_limit"


async def test_fixed_output_geometry_keeps_original_dimensions():
    import hashlib

    from transformation_portal.photographic_delivery import DeliveryImage

    buffer = io.BytesIO()
    with Image.new("RGB", (1024, 512), "white") as source:
        source.save(buffer, format="PNG")
    raw = buffer.getvalue()
    image = DeliveryImage("large.png", hashlib.sha256(raw).hexdigest(), "scene", source_bytes=raw, input_color="srgb")
    encoded, width, height, native_width, native_height = review._display_with_geometry(image, None, 256)
    assert (width, height, native_width, native_height) == (256, 128, 1024, 512)
    with Image.open(io.BytesIO(encoded)) as thumbnail:
        assert thumbnail.size == (256, 128)


async def test_direct_thumbnail_call_cannot_bypass_resource_admission(thumbnail_case):
    case = thumbnail_case
    with pytest.raises(ReviewOverload) as error:
        await case.service.thumbnail(case.context, "input-0000", case.identity)
    assert error.value.reason == "thumbnail_reservation_required"
    assert not case.reads


async def test_completed_thumbnails_are_not_cached_on_server(thumbnail_case):
    case = thumbnail_case
    await read_thumbnail(case)
    await read_thumbnail(case)
    assert case.reads == ["input-0000", "input-0000"]
    assert case.service.admission.snapshot()["coalesced"] == 0


async def test_variant_selection_is_server_owned_and_deterministic():
    original = {"scene_id": "scene", "artifact_path": "input.png", "role": "original"}
    native = {"scene_id": "scene", "artifact_path": "master.npy", "role": "working_master"}
    first = {"scene_id": "scene", "artifact_path": "v1.tif", "role": "final"}
    latest = {"scene_id": "scene", "artifact_path": "v2.tif", "role": "final"}
    related = {"scene_id": "scene", "artifact_path": "depth.tif", "related_only": True}
    revised = {"scene_id": "scene", "artifact_path": "revision.tif", "role": "final"}
    refs = [original, native, first, latest, related]
    assert thumbnail_source(refs, {}, "scene") == latest
    assert thumbnail_source(refs, {"revisions": [{"variant": revised}]}, "scene") == revised
    assert thumbnail_source([original, native], {}, "scene") == native
    assert thumbnail_source([original], {}, "scene") == original
    assert thumbnail_source([related], {}, "scene") is None


async def test_missing_source_identity_is_private_admitted_read(thumbnail_case):
    case = thumbnail_case
    index = [{**ref, "source_identity_sha256": None} for ref in case.refs]
    case.context = replace(case.context, discover_scene_index=AsyncMock(side_effect=lambda: list(index)))
    page = await case.service.scenes(case.context)
    assert page["scenes"][0]["thumbnail"]["source_identity_sha256"] is None
    assert await case.service._thumbnail_key(case.context, ("input-0000", page["catalogue_identity_sha256"]), {}) is None
    response = await read_thumbnail(case, identity=page["catalogue_identity_sha256"])
    assert response.status_code == 200 and case.reads == ["input-0000"]
    assert case.service.admission.snapshot()["coalesced"] == 0


async def test_unknown_presentation_identity_is_a_private_admitted_read(thumbnail_case):
    case = thumbnail_case
    for ref in case.refs:
        ref.pop("presentation_policy_sha256", None)
    page = await case.service.scenes(case.context)
    assert await case.service._thumbnail_key(case.context, ("input-0000", page["catalogue_identity_sha256"]), {}) is None
    response = await read_thumbnail(case, identity=page["catalogue_identity_sha256"])
    assert response.status_code == 200 and case.service.admission.snapshot()["coalesced"] == 0


async def test_actor_and_variant_identity_are_in_coalescing_key(thumbnail_case):
    case = thumbnail_case
    key = await case.service._thumbnail_key(case.context, ("input-0000", case.identity), {})
    assert key is not None
    for actor in (
        {**case.context.actor, "username": "another"},
        {**case.context.actor, "accessEmail": "another@example.test"},
    ):
        context = replace(case.context, actor=actor)
        page = await case.service.scenes(context)
        changed = await case.service._thumbnail_key(context, ("input-0000", page["catalogue_identity_sha256"]), {})
        assert changed != key
    assert await case.service._thumbnail_key(case.context, ("input-0000", case.identity), {"view": "recorded"}) != key
    case.refs[0]["presentation_policy_sha256"] = "d" * 64
    page = await case.service.scenes(case.context)
    assert await case.service._thumbnail_key(case.context, ("input-0000", page["catalogue_identity_sha256"]), {}) != key


async def test_shared_thumbnail_cancellation_keeps_worker_owned(thumbnail_case, monkeypatch):
    case = thumbnail_case
    started, release = threading.Event(), threading.Event()
    calls = []

    def display(*_):
        lease = current_review_lease.get()
        calls.append(lease)
        assert lease is not None and not lease.cost.interactive
        started.set()
        assert release.wait(5)
        assert not lease.released
        return b"pixels", 8, 6, 8, 6

    monkeypatch.setattr(review, "_display_with_geometry", display)
    leader = asyncio.create_task(read_thumbnail(case))
    follower = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        follower = asyncio.create_task(read_thumbnail(case))
        for _ in range(20):
            if case.service.admission.snapshot()["coalesced"]:
                break
            await asyncio.sleep(0)
        assert case.service.admission.snapshot()["coalesced"] == 1
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        assert not calls[0].released and not follower.done()
        release.set()
        assert (await follower).body == b"pixels" and len(calls) == 1
        assert case.service.admission.snapshot()["active"] == 0
    finally:
        release.set()
        await asyncio.gather(leader, *([follower] if follower is not None else []), return_exceptions=True)


async def test_each_shared_subscriber_rechecks_its_own_authorization(thumbnail_case, monkeypatch):
    case = thumbnail_case
    started, release = threading.Event(), threading.Event()
    controls = {"revoked": False}

    async def follower_authorization():
        if controls["revoked"]:
            raise ReviewAccessError("Session revoked", status=403, code="FORBIDDEN")

    follower_context = replace(case.context, revalidate=follower_authorization)

    def display(*_):
        started.set()
        assert release.wait(5)
        return b"pixels", 8, 6, 8, 6

    monkeypatch.setattr(review, "_display_with_geometry", display)
    leader = asyncio.create_task(read_thumbnail(case))
    follower = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        follower = asyncio.create_task(read_thumbnail(case, context=follower_context))
        for _ in range(20):
            if case.service.admission.snapshot()["coalesced"]:
                break
            await asyncio.sleep(0)
        assert case.service.admission.snapshot()["coalesced"] == 1
        controls["revoked"] = True
        release.set()
        assert (await leader).body == b"pixels"
        with pytest.raises(ReviewAccessError) as error:
            await follower
        assert error.value.status == 403
    finally:
        release.set()
        await asyncio.gather(leader, *([follower] if follower is not None else []), return_exceptions=True)


async def test_revoked_leader_does_not_authorize_or_poison_valid_follower(thumbnail_case, monkeypatch):
    case = thumbnail_case
    started, release = threading.Event(), threading.Event()
    follower_context = replace(case.context, revalidate=AsyncMock())
    leases = []

    def display(*_):
        lease = current_review_lease.get()
        if leases:
            assert leases[0].released
        leases.append(lease)
        started.set()
        assert release.wait(5)
        return b"pixels", 8, 6, 8, 6

    monkeypatch.setattr(review, "_display_with_geometry", display)
    leader = asyncio.create_task(read_thumbnail(case))
    follower = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        follower = asyncio.create_task(read_thumbnail(case, context=follower_context))
        for _ in range(20):
            if case.service.admission.snapshot()["coalesced"]:
                break
            await asyncio.sleep(0)
        assert case.service.admission.snapshot()["coalesced"] == 1
        case.controls.revoked = True
        release.set()
        with pytest.raises(ReviewAccessError):
            await leader
        assert (await follower).body == b"pixels" and len(leases) == 2
    finally:
        release.set()
        await asyncio.gather(leader, *([follower] if follower is not None else []), return_exceptions=True)


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_http_thumbnail_aliases_and_typed_errors(thumbnail_case, monkeypatch, version):
    case = thumbnail_case
    monkeypatch.setenv("TP_PHOTO_REVIEW_ENABLED", "1")
    app = FastAPI()

    def error_response(status, *, code, message, details, **_):
        return review.JSONResponse(
            {"success": False, "error": {"code": code, "message": message, "details": details}}, status_code=status
        )

    authorize = AsyncMock(return_value=case.context)
    app.include_router(
        create_photographic_review_router(
            ReviewRuntime(authorize, error_response, authorize_admission=authorize, authorize_catalogue=authorize)
        )
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            base = f"/{version}/jobs/job-scenes/photographic-review"
            response = await client.get(
                base + "/thumbnail", params={"scene_id": "input-0000", "catalogue_identity_sha256": case.identity}
            )
            assert response.status_code == 200 and response.headers["content-type"] == "image/png"
            response = await client.get(
                base + "/thumbnail", params={"scene_id": "input-0000", "catalogue_identity_sha256": "0" * 64}
            )
            assert response.status_code == 409 and response.json()["error"]["code"] == "CATALOGUE_CHANGED"
            monkeypatch.setenv("TP_PHOTO_REVIEW_THUMBNAILS_ENABLED", "0")
            response = await client.get(
                base + "/thumbnail", params={"scene_id": "input-0000", "catalogue_identity_sha256": case.identity}
            )
            assert (
                response.status_code == 503 and response.json()["error"]["details"]["reason"] == "thumbnail_feature_disabled"
            )
