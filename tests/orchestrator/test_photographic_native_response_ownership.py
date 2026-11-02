"""Native HTTP responses own memory and request tickets through transport exit."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image

from transformation_portal.api.routes import photographic_review as routes
from transformation_portal.api.v1.photographic_review import RevisionRequest
from transformation_portal.orchestrator import photographic_admission as admission
from transformation_portal.orchestrator import photographic_review_service as review
from transformation_portal.orchestrator.photographic_admission import GIB, ReviewAdmissionConfig, ReviewCost, ReviewLease
from transformation_portal.orchestrator.photographic_review_access import ReviewContext, ReviewRuntime
from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
SCOPE = {"type": "http", "method": "POST", "headers": [], "extensions": {}}


@pytest.fixture(name="native_case")
async def _native_case(monkeypatch):
    monkeypatch.setenv("TP_PHOTO_REVIEW_NATIVE_MATERIALS_ENABLED", "1")
    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    encoded = io.BytesIO()
    with Image.new("RGB", (8, 6), (120, 150, 180)) as image:
        image.save(encoded, format="PNG")
    raw = encoded.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    ref = {
        "artifact_path": "source/final.png",
        "source_name": "final.png",
        "sha256": sha,
        "source_identity_sha256": sha,
        "size_bytes": len(raw),
        "scene_id": "scene",
        "input_color": "srgb",
        "kind": "indexed",
        "role": "final",
    }
    store = MemoryReviewStore()
    monkeypatch.setattr(review, "get_review_store", lambda: store)
    service = review.PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    assets, cas = {}, []

    async def materialize(selected, path, maximum):
        assert selected["sha256"] == sha and len(raw) <= maximum
        path.write_bytes(raw)
        return {"sha256": sha, "size_bytes": len(raw)}

    async def publish(storage, source, _mime):
        assert storage not in assets
        assets[storage] = source.read_bytes()

    async def discard(paths):
        for path in paths:
            assets.pop(path, None)

    context = ReviewContext(
        "native-job",
        "tenant",
        {"username": "operator"},
        [ref],
        materialize,
        publish,
        discard,
        AsyncMock(),
        binding_identity="b" * 64,
    )
    service._remember_inspection(context, ref, {"width": 8, "height": 6, "alpha": False})
    original_mutate = store.mutate

    async def mutate(*args):
        lease = admission.current_review_lease.get()
        cas.append((lease.native_response_owned, lease.cost.memory_bytes, lease.cost.decode_slots, lease.cost.io_slots))
        return await original_mutate(*args)

    monkeypatch.setattr(store, "mutate", mutate)
    payload = RevisionRequest(
        expected_version=0,
        artifact_path=ref["artifact_path"],
        sha256=sha,
        source_identity_sha256=sha,
        scene_id="scene",
        recipe={
            "schema": "tp.photo.materials_recipe.v1",
            "input_stage": "display_referred",
            "regions": [{"id": "stone", "label": "stone", "region": [0, 0, 1, 1], "strength": 0.03}],
        },
    )
    case = SimpleNamespace(
        service=service, context=context, payload=payload, assets=assets, cas=cas, owners=[], responses=[], tasks=[]
    )

    async def execute(selected=None):
        owner = await service.execute(context, service.revision, selected or payload, _response_owned=True)
        case.owners.append(owner)
        return owner

    async def response(owner):
        value = await review.owned_envelope(owner)
        case.responses.append(value)
        return value

    case.execute, case.response = execute, response
    try:
        yield case
    finally:
        for task in case.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*case.tasks, return_exceptions=True)
        for response in case.responses:
            await response.release()
        for owner in case.owners:
            await owner.release()
        await service.close()


@pytest.mark.parametrize("preview_only", [True, False])
async def test_native_response_holds_complete_memory_and_ticket_while_same_tenant_inspection_progresses(
    native_case, preview_only
):
    case = native_case
    owner = await case.execute(case.payload.model_copy(update={"preview_only": preview_only}))
    assert isinstance(owner, review._OwnedNativeRevisionResult)
    expected = review.envelope(dict(owner)).body
    lease = owner.admission_lease
    reserved = lease.cost.memory_bytes
    assert lease.native_response_owned and reserved > 0
    assert lease.cost.decode_slots == lease.cost.io_slots == 0
    if not preview_only:
        assert case.cas == [(True, reserved, 0, 0)]
    response = await case.response(owner)
    entered, finish = asyncio.Event(), asyncio.Event()
    messages = []

    async def send(message):
        messages.append(dict(message))
        if message["type"] == "http.response.body":
            entered.set()
            await finish.wait()

    task = asyncio.create_task(response(SCOPE, AsyncMock(), send))
    case.tasks.append(task)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        snapshot = case.service.admission.snapshot()
        assert snapshot["memory_reserved_bytes"] == reserved and snapshot["outstanding"] == 1
        inspection = await case.service.admission.acquire("tenant", ReviewCost(6 * GIB, 0))
        try:
            assert case.service.admission.snapshot()["memory_reserved_bytes"] == reserved + 6 * GIB
        finally:
            await inspection.release()
        following = asyncio.create_task(case.service.admission.acquire("tenant", ReviewCost(6 * GIB, 0, interactive=False)))
        try:
            await asyncio.sleep(0)
            assert not following.done(), "a native HTTP tail lost its complete bulk memory reservation"
            assert case.service.admission.snapshot()["queued"] == 1
        finally:
            following.cancel()
            await asyncio.gather(following, return_exceptions=True)
    finally:
        finish.set()
    await task
    assert messages[-1]["body"] == expected
    assert response.headers["cache-control"] == "no-store"
    assert response.body == b""
    assert case.service.admission.snapshot()["active"] == case.service.admission.snapshot()["outstanding"] == 0


@pytest.mark.parametrize("preview_only", [True, False])
async def test_native_direct_call_keeps_plain_dict_and_original_release_contract(native_case, preview_only):
    case = native_case
    result = await case.service.execute(
        case.context, case.service.revision, case.payload.model_copy(update={"preview_only": preview_only})
    )
    assert isinstance(result, dict) and not isinstance(result, review._OwnedNativeRevisionResult)
    assert result["preview_data_url"].startswith("data:image/png;base64,")
    if not preview_only:
        assert case.cas == [(False, 0, 0, 0)]
    assert case.service.admission.snapshot()["active"] == case.service.admission.snapshot()["outstanding"] == 0


@pytest.mark.parametrize("phase", ["payload", "render"])
async def test_failed_json_construction_drops_closed_serializer_buffers_before_releasing(native_case, monkeypatch, phase):
    case = native_case
    owner = await case.execute()
    buffers, releases = [], []
    original = ReviewLease.release

    def fail(*_args):
        temporary = np.zeros(1024, np.uint8)
        buffers.append(weakref.ref(temporary))
        raise ValueError("controlled JSON construction failure")

    async def release(lease):
        assert owner == {} and buffers and all(ref() is None for ref in buffers)
        releases.append(lease)
        await original(lease)

    monkeypatch.setattr(ReviewLease, "release", release)
    if phase == "payload":
        monkeypatch.setattr(review, "_envelope_payload", fail)
    else:
        monkeypatch.setattr(review.JSONResponse, "render", fail)
    with pytest.raises(ValueError, match="JSON construction"):
        await review.owned_envelope(owner)
    assert len(releases) == 1
    assert case.service.admission.snapshot()["active"] == case.service.admission.snapshot()["outstanding"] == 0


@pytest.mark.parametrize("failure_type", ["http.response.start", "http.response.body"])
async def test_failed_send_clears_response_and_sender_buffers_before_release(native_case, monkeypatch, failure_type):
    case = native_case
    response = await case.response(await case.execute())
    original = ReviewLease.release
    buffers = []

    async def send(message):
        if message["type"] == failure_type:
            temporary = np.zeros(1024, np.uint8)
            buffers.append(weakref.ref(temporary))
            raise OSError("controlled native send failure")

    async def release(lease):
        assert response.body == b"" and buffers and all(ref() is None for ref in buffers)
        await original(lease)

    monkeypatch.setattr(ReviewLease, "release", release)
    with pytest.raises(OSError, match="native send"):
        await response(SCOPE, AsyncMock(), send)
    assert case.service.admission.snapshot()["active"] == case.service.admission.snapshot()["outstanding"] == 0


async def test_repeated_cancellation_retains_native_response_until_sender_is_quiescent(native_case):
    case = native_case
    response = await case.response(await case.execute())
    entered, finish = asyncio.Event(), asyncio.Event()
    cancellations = [asyncio.Event(), asyncio.Event()]

    async def send(message):
        if message["type"] != "http.response.body":
            return
        entered.set()
        count = 0
        while not finish.is_set():
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancellations[min(count, 1)].set()
                count += 1
        if count:
            raise asyncio.CancelledError

    task = asyncio.create_task(response(SCOPE, AsyncMock(), send))
    case.tasks.append(task)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for observed in cancellations:
            task.cancel()
            await asyncio.wait_for(observed.wait(), 2)
            assert not task.done()
            assert case.service.admission.snapshot()["memory_reserved_bytes"] > 0
            assert case.service.admission.snapshot()["outstanding"] == 1
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert case.service.admission.snapshot()["active"] == case.service.admission.snapshot()["outstanding"] == 0


async def test_repeated_cancellation_drains_native_response_release(native_case, monkeypatch):
    case = native_case
    response = await case.response(await case.execute())
    entered, finish = asyncio.Event(), asyncio.Event()
    original = case.service.admission._release

    async def hold_release(lease):
        entered.set()
        await finish.wait()
        await original(lease)

    monkeypatch.setattr(case.service.admission, "_release", hold_release)
    task = asyncio.create_task(response(SCOPE, AsyncMock(), AsyncMock()))
    case.tasks.append(task)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and case.service.admission.snapshot()["outstanding"] == 1
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert case.service.admission.snapshot()["active"] == case.service.admission.snapshot()["outstanding"] == 0


async def test_queued_native_request_drops_planning_projection_before_acquiring(native_case, monkeypatch):
    case = native_case

    class Projection(dict):
        pass

    projections = []
    original_state = case.service.state

    async def state(context):
        value = Projection(await original_state(context))
        projections.append(weakref.ref(value))
        return value

    original_acquire = case.service.admission.acquire
    entered = asyncio.Event()

    async def acquire(tenant, cost):
        assert projections and projections[0]() is None
        entered.set()
        return await original_acquire(tenant, cost)

    blocking = await original_acquire("other", ReviewCost(6 * GIB, 0, interactive=False))
    monkeypatch.setattr(case.service, "state", state)
    monkeypatch.setattr(case.service.admission, "acquire", acquire)
    task = asyncio.create_task(case.execute())
    case.tasks.append(task)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not task.done() and case.service.admission.snapshot()["queued"] == 1
    finally:
        await blocking.release()
    await task


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_native_http_envelope_and_routes_stay_stable_with_owned_response(native_case, monkeypatch, version):
    case = native_case
    monkeypatch.setattr(routes, "PhotographicReviewService", lambda: case.service)

    async def authorize(_request, job):
        assert job == case.context.job_id
        return case.context

    def error(status, **values):
        return review.JSONResponse({"success": False, "data": None, "error": values}, status_code=status)

    app = FastAPI()
    app.include_router(
        routes.create_photographic_review_router(ReviewRuntime(authorize, error, authorize_admission=authorize))
    )
    original = routes.owned_envelope
    observed = []

    async def envelope(data):
        observed.append(isinstance(data, review._OwnedNativeRevisionResult))
        assert case.service.admission.snapshot()["outstanding"] == 1
        return await original(data)

    monkeypatch.setattr(routes, "owned_envelope", envelope)
    expected = await case.service.execute(case.context, case.service.revision, case.payload)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/{version}/jobs/{case.context.job_id}/photographic-review/revisions", json=case.payload.model_dump(mode="json")
        )
    assert response.status_code == 200 and observed == [True]
    assert response.content == review.envelope(expected).body
    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.content)["data"]["state_version"] == 0
    assert case.service.admission.snapshot()["active"] == case.service.admission.snapshot()["outstanding"] == 0
