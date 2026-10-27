"""Process-local fairness, bounded ownership, and independent render subscribers."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from transformation_portal.orchestrator.photographic_admission import (
    GIB,
    PhotographicAdmissionController,
    ReviewAdmissionConfig,
    ReviewCost,
    ReviewOverload,
    current_review_lease,
    operation_cost,
)
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewContext
from transformation_portal.orchestrator.photographic_review_service import (
    PhotographicReviewService,
    _owned_thread,
    _OwnedDownloadResponse,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def bulk_cost(controller):
    return replace(
        operation_cost("delivery", controller.config, [{"width": 6708, "height": 4472, "alpha": False}]),
        scratch_bytes=62 * GIB,
    )


def inspection_cost(controller):
    return replace(operation_cost("render", controller.config), scratch_bytes=2 * GIB)


def configuration(**changes):
    return replace(ReviewAdmissionConfig(enabled=True), **changes)


async def tick():
    for _ in range(4):
        await asyncio.sleep(0)


async def test_pilot_defaults_are_explicit_opt_in_and_process_local():
    config = ReviewAdmissionConfig.from_environment({})
    assert not config.enabled
    assert (config.memory_bytes, config.interactive_memory_bytes, config.bulk_memory_bytes) == (12 * GIB, 4 * GIB, 8 * GIB)
    assert (config.io_slots, config.interactive_io_slots, config.noninteractive_slots) == (2, 1, 1)
    assert (config.scratch_bytes, config.interactive_scratch_bytes, config.scratch_floor_bytes) == (64 * GIB, 2 * GIB, 2 * GIB)
    assert PhotographicAdmissionController(config).snapshot()["scope"] == "process"
    assert operation_cost("render", config).memory_bytes > 4 * GIB
    assert PhotographicAdmissionController(config).interactive_reserve == 6 * GIB
    assert operation_cost("delivery", config).memory_bytes > 6 * GIB


@pytest.mark.parametrize(
    "values",
    [
        {"TP_PHOTO_REVIEW_ENABLED": "true"},
        {"TP_PHOTO_REVIEW_IO_SLOTS": "-1"},
        {"TP_PHOTO_REVIEW_MEMORY_BYTES": "1"},
        {"TP_PHOTO_REVIEW_QUEUED": "32"},
        {"TP_PHOTO_REVIEW_INTERACTIVE_IO_SLOTS": "2"},
        {"TP_PHOTO_REVIEW_TENANT_SLOTS": "1"},
    ],
)
async def test_invalid_configuration_fails_closed(values):
    with pytest.raises(ValueError):
        ReviewAdmissionConfig.from_environment(values)


async def test_export_reserves_capacity_for_same_tenant_inspection():
    controller = PhotographicAdmissionController(configuration())
    export = await controller.acquire("tenant-a", bulk_cost(controller))
    queued_export = asyncio.create_task(controller.acquire("tenant-a", bulk_cost(controller)))
    inspection = await controller.acquire("tenant-a", inspection_cost(controller))
    await tick()
    assert not queued_export.done()
    assert controller.snapshot()["memory_reserved_bytes"] <= 12 * GIB
    assert controller.snapshot()["scratch_reserved_bytes"] == 64 * GIB
    assert controller.snapshot()["active"] == 2
    await inspection.release()
    assert not queued_export.done()
    await export.release()
    second = await queued_export
    await second.release()
    assert controller.snapshot()["active"] == 0


async def test_download_serializes_with_bulk_but_not_inspection():
    controller = PhotographicAdmissionController(configuration())
    download = await controller.acquire("tenant-a", operation_cost("download", controller.config))
    export = asyncio.create_task(controller.acquire("tenant-b", bulk_cost(controller)))
    inspection = await controller.acquire("tenant-a", inspection_cost(controller))
    await tick()
    assert not export.done()
    await inspection.release()
    await download.release()
    await (await export).release()


async def test_fifo_per_tenant_and_round_robin_across_waiting_tenants():
    controller = PhotographicAdmissionController(configuration())
    full = ReviewCost(12 * GIB, 1, decode_slots=2)
    blocker = await controller.acquire("blocker", full)
    order = []

    async def work(tenant, name):
        lease = await controller.acquire(tenant, full)
        order.append(name)
        await lease.release()

    tasks = [
        asyncio.create_task(work(tenant, name))
        for tenant, name in [
            ("a", "a1"),
            ("a", "a2"),
            ("b", "b1"),
            ("b", "b2"),
        ]
    ]
    await tick()
    await blocker.release()
    await asyncio.gather(*tasks)
    assert order == ["a1", "b1", "a2", "b2"]


async def test_queue_caps_and_cancel_before_admission_release_all_accounting():
    controller = PhotographicAdmissionController(configuration(tenant_queued=1))
    full = ReviewCost(12 * GIB, 1, decode_slots=2)
    blocker = await controller.acquire("blocker", full)
    waiting = asyncio.create_task(controller.acquire("a", full))
    await tick()
    with pytest.raises(ReviewOverload) as caught:
        await controller.acquire("a", full)
    assert caught.value.status == 429
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert controller.snapshot()["queued"] == 0
    await blocker.release()


async def test_queue_deadline_is_typed_and_releases_waiter():
    controller = PhotographicAdmissionController(configuration(interactive_wait_seconds=1))
    full = ReviewCost(12 * GIB, 1, decode_slots=2)
    blocker = await controller.acquire("blocker", full)
    with pytest.raises(ReviewOverload, match="retry") as caught:
        await controller.acquire("waiting", full)
    assert caught.value.reason == "admission_deadline"
    assert controller.snapshot()["queued"] == 0
    await blocker.release()


async def test_outstanding_cap_charges_coalesced_subscribers_too():
    controller = PhotographicAdmissionController(configuration(tenant_outstanding=2, tenant_queued=1))
    async with controller.request("a"), controller.request("a"):
        with pytest.raises(ReviewOverload) as caught:
            async with controller.request("a"):
                pytest.fail("tenant cap bypassed")
        assert caught.value.status == 429
    assert controller.snapshot()["outstanding"] == 0


async def test_permanent_oversize_does_not_queue_or_allocate():
    controller = PhotographicAdmissionController(configuration())
    with pytest.raises(ReviewOverload) as caught:
        await controller.acquire("a", ReviewCost(9 * GIB, 1, interactive=False))
    assert caught.value.status == 413 and not caught.value.retriable
    assert controller.snapshot()["queued"] == controller.snapshot()["active"] == 0


async def test_physical_disk_headroom_is_distinct_from_scratch_accounting(monkeypatch):
    from transformation_portal.orchestrator import photographic_admission as module

    controller = PhotographicAdmissionController(configuration())
    lease = await controller.acquire("a", ReviewCost(1, 6 * GIB, interactive=False))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * GIB - 1))
    with pytest.raises(ReviewOverload) as caught:
        lease.check_scratch(6 * GIB)
    assert caught.value.reason == "scratch_unavailable"  # Six + two interactive + two floor.
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * GIB))
    lease.check_scratch(6 * GIB)
    with pytest.raises(ReviewOverload) as caught:
        lease.check_scratch(63 * GIB)
    assert caught.value.status == 413
    await lease.release()


async def test_last_coalesced_interest_cancels_queued_work():
    controller = PhotographicAdmissionController(configuration())
    blocker = await controller.acquire("blocker", ReviewCost(12 * GIB, 1, decode_slots=2))
    entered = False

    async def operation():
        nonlocal entered
        lease = await controller.acquire("a", inspection_cost(controller))
        try:
            entered = True
        finally:
            await lease.release()

    subscriber = asyncio.create_task(controller.coalesce(("a", "source"), operation))
    await tick()
    assert controller.snapshot()["queued"] == 1
    subscriber.cancel()
    with pytest.raises(asyncio.CancelledError):
        await subscriber
    await tick()
    assert not entered and controller.snapshot()["queued"] == 0
    await blocker.release()


def context(*, actor="a", revalidate=None, discover=None, identity="binding"):
    return ReviewContext(
        job_id="job",
        tenant_id="tenant",
        actor={"username": actor},
        references=[],
        materialize=AsyncMock(),
        publish=AsyncMock(),
        revalidate=revalidate,
        discover_references=discover,
        binding_identity=identity,
    )


async def test_render_coalesces_but_revalidates_each_subscriber_and_retains_running_worker(monkeypatch):
    from transformation_portal.orchestrator import photographic_admission as admission

    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    service = PhotographicReviewService(configuration())
    monkeypatch.setattr(service, "state", AsyncMock(return_value={"state_version": 1}))
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []
    discover = AsyncMock(return_value=[])
    first_auth, second_auth = AsyncMock(), AsyncMock()

    async def render(_context, *_args, **_kwargs):
        def work():
            calls.append("render")
            started.set()
            try:
                assert release.wait(5)
                return b"pixels"
            finally:
                finished.set()

        return await _owned_thread(work)

    first = asyncio.create_task(service.execute(context(revalidate=first_auth, discover=discover), render, "path", "sha"))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        second = asyncio.create_task(
            service.execute(context(actor="b", revalidate=second_auth, discover=discover), render, "path", "sha")
        )
        await tick()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert service.admission.snapshot()["active"] == 1
        assert not finished.is_set()
        release.set()
        assert await second == b"pixels"
        assert calls == ["render"] and discover.await_count == 1
        assert second_auth.await_count == 2
        assert service.admission.snapshot()["active"] == service.admission.snapshot()["outstanding"] == 0
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)


async def test_coalesced_result_never_bypasses_a_subscribers_revoked_auth(monkeypatch):
    from transformation_portal.orchestrator import photographic_admission as admission

    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    service = PhotographicReviewService(configuration())
    monkeypatch.setattr(service, "state", AsyncMock(return_value={"state_version": 1}))
    running, release = asyncio.Event(), asyncio.Event()
    denied = False

    async def revoked_auth():
        if denied:
            raise ReviewAccessError("revoked", status=403)

    async def render(_context, *_args):
        running.set()
        await release.wait()
        return b"pixels"

    first = asyncio.create_task(service.execute(context(), render, "source", "sha"))
    await running.wait()
    second = asyncio.create_task(service.execute(context(actor="b", revalidate=revoked_auth), render, "source", "sha"))
    await tick()
    denied = True
    release.set()
    assert await first == b"pixels"
    with pytest.raises(ReviewAccessError, match="revoked"):
        await second


async def test_failed_owned_worker_and_repeat_cancellation_hold_lease_until_quiescent(monkeypatch):
    from transformation_portal.orchestrator import photographic_admission as admission

    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    service = PhotographicReviewService(configuration(memory_bytes=14 * GIB))
    started, release = threading.Event(), threading.Event()

    async def delivery(_context):
        def worker():
            started.set()
            assert release.wait(5)
            raise OSError("controlled worker failure")

        return await _owned_thread(worker)

    task = asyncio.create_task(service.execute(context(), delivery))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await tick()
        task.cancel()
        await tick()
        assert not task.done() and service.admission.snapshot()["active"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.admission.snapshot()["active"] == service.admission.snapshot()["outstanding"] == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_download_response_owns_io_and_scratch_until_failed_send_cleanup(monkeypatch, tmp_path):
    from transformation_portal.orchestrator import photographic_admission as admission

    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    service = PhotographicReviewService(configuration())
    directory = tmp_path / "download"
    directory.mkdir()
    source = directory / "artifact"
    source.write_bytes(b"verified")

    async def download(_context):
        assert current_review_lease.get() is not None
        return _OwnedDownloadResponse(
            source, directory=directory, media_type="application/zip", filename="test.zip", headers={}
        )

    response = await service.execute(context(), download)
    assert service.admission.snapshot()["active"] == 1 and source.exists()
    assert service.admission.snapshot()["outstanding"] == 1

    async def failed_send(_message):
        raise OSError("controlled transport failure")

    with pytest.raises(OSError, match="transport"):
        await response({"type": "http", "method": "GET", "headers": [], "extensions": {}}, AsyncMock(), failed_send)
    assert not directory.exists() and service.admission.snapshot()["active"] == 0
    assert service.admission.snapshot()["outstanding"] == 0


async def test_scratch_sampling_survives_owned_download_directory_cleanup(monkeypatch, tmp_path):
    from pathlib import Path

    from transformation_portal.orchestrator import photographic_admission as admission
    from transformation_portal.orchestrator import photographic_review_service as service_module

    controller = PhotographicAdmissionController(configuration())
    lease = await controller.acquire(
        "a", ReviewCost(1, GIB, decode_slots=0, interactive=False, filesystems=((tmp_path, GIB),))
    )
    inspection = await controller.acquire("b", ReviewCost(1, GIB, filesystems=((tmp_path, GIB),)))
    directory = tmp_path / "download"
    directory.mkdir()
    source = directory / "artifact"
    source.write_bytes(b"verified")
    lease.track_path(directory)
    response = _OwnedDownloadResponse(
        source, directory=directory, media_type="application/zip", filename="test.zip", headers={}
    )
    response.admission_lease = lease
    cleanup_started, enumeration_started, removed = threading.Event(), threading.Event(), threading.Event()
    original_rmtree, original_iterdir = service_module.shutil.rmtree, Path.iterdir

    def cleanup(path, *args, **kwargs):
        if Path(path) == directory:
            cleanup_started.set()
            assert enumeration_started.wait(5)
            try:
                return original_rmtree(path, *args, **kwargs)
            finally:
                removed.set()
        return original_rmtree(path, *args, **kwargs)

    def enumerate_directory(path):
        if path == directory:
            # remaining() has already observed this directory via lstat.
            # Its response worker removes it before enumeration can begin.
            enumeration_started.set()
            assert removed.wait(5)
        return original_iterdir(path)

    monkeypatch.setattr(service_module.shutil, "rmtree", cleanup)
    monkeypatch.setattr(Path, "iterdir", enumerate_directory)
    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))

    async def failed_send(_message):
        raise OSError("controlled transport failure")

    task = asyncio.create_task(
        response({"type": "http", "method": "GET", "headers": [], "extensions": {}}, AsyncMock(), failed_send)
    )
    try:
        assert await asyncio.to_thread(cleanup_started.wait, 2)
        inspection.check_scratch(0)
        assert lease.remaining(tmp_path.stat().st_dev) == GIB  # Vanished allocations grant no credit.
        with pytest.raises(OSError, match="transport"):
            await task
        assert lease.released and not directory.exists()
    finally:
        enumeration_started.set()
        await asyncio.gather(task, return_exceptions=True)
        await lease.release()
        await inspection.release()


async def test_measured_inspection_headroom_takes_precedence_over_minimum():
    controller = PhotographicAdmissionController(configuration())
    assert controller.interactive_reserve == 6 * GIB
    # The class maximum is not a promise to fit every job beside inspection.
    with pytest.raises(ReviewOverload) as caught:
        await controller.acquire("a", ReviewCost(8 * GIB, 1, interactive=False))
    assert caught.value.status == 413
    cost = bulk_cost(controller)
    assert cost.memory_bytes + controller.interactive_reserve <= controller.config.memory_bytes
    with pytest.raises(ReviewOverload):
        await controller.acquire("a", operation_cost("revision", controller.config))


async def test_cost_profiles_use_geometry_and_failed_alpha_export_is_not_zero_cost():
    config = configuration()
    small = operation_cost("delivery", config, [{"width": 96, "height": 64, "alpha": False}])
    large = operation_cost("delivery", config)
    assert small.memory_bytes < large.memory_bytes
    assert small.max_pixels == 96 * 64
    assert operation_cost("revision", config).memory_bytes > config.bulk_memory_bytes
    assert operation_cost("render", config, [{"width": 1, "height": 1, "alpha": "unknown"}]).max_pixels == 40_000_000


async def test_admitted_header_geometry_is_checked_before_decode(monkeypatch, tmp_path):
    from transformation_portal.orchestrator import photographic_review_service as service_module

    path = tmp_path / "image"
    ref = {"source_name": "final.png", "sha256": "a" * 64}
    materialize = AsyncMock(return_value={"sha256": "a" * 64})
    case = replace(context(), materialize=materialize)
    monkeypatch.setattr(service_module, "_check_geometry", lambda _: (100, 100))
    controller = PhotographicAdmissionController(configuration())
    lease = await controller.acquire("a", ReviewCost(1, 1, max_pixels=100))
    token = current_review_lease.set(lease)
    try:
        with pytest.raises(ReviewOverload) as caught:
            await service_module.spool_image(case, ref, path)
        assert caught.value.reason == "admitted_geometry_changed"
    finally:
        current_review_lease.reset(token)
        await lease.release()


async def test_shutdown_rejects_waiters_and_drains_active_ownership_before_pool_close():
    controller = PhotographicAdmissionController(configuration())
    owner = await controller.acquire("a", ReviewCost(12 * GIB, 1, decode_slots=2))
    waiting = asyncio.create_task(controller.acquire("b", ReviewCost(1, 1)))
    await tick()
    closing = asyncio.create_task(controller.close())
    await tick()
    with pytest.raises(ReviewOverload) as caught:
        await waiting
    assert caught.value.reason == "service_stopping"
    assert not closing.done()
    await owner.release()
    await closing
    assert controller.snapshot()["active"] == controller.snapshot()["queued"] == 0
    with pytest.raises(ReviewOverload):
        await controller.acquire("c", ReviewCost(1, 1))


async def test_http_overload_preserves_envelope_and_retry_header_without_discovery(monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from starlette.responses import JSONResponse

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator import photographic_admission as admission
    from transformation_portal.orchestrator.photographic_review_access import ReviewRuntime

    monkeypatch.setenv("TP_PHOTO_REVIEW_ENABLED", "1")
    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    discover = AsyncMock(return_value=[])
    case = context(discover=discover)
    authorize = AsyncMock(return_value=case)
    error = lambda status, **kwargs: JSONResponse({"success": False, "error": kwargs}, status_code=status)
    app = FastAPI()
    app.include_router(create_photographic_review_router(ReviewRuntime(authorize, error, authorize_admission=authorize)))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/v2/jobs/job/photographic-review/render", params={"artifact_path": "final.tif", "sha256": "a" * 64}
        )
    assert response.status_code == 503 and response.headers["Retry-After"] == "1"
    assert response.json()["error"]["code"] == "RESOURCE_EXHAUSTED"
    assert response.json()["error"]["details"] == {"job_id": "job", "reason": "scratch_unavailable", "scope": "process"}
    assert response.json()["error"]["retriable"] is True
    discover.assert_not_awaited()


async def test_http_permanent_cost_limit_has_no_retry_header_or_publication(monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from starlette.responses import JSONResponse

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator.photographic_review_access import ReviewRuntime

    monkeypatch.setenv("TP_PHOTO_REVIEW_ENABLED", "1")
    case = context(discover=AsyncMock())
    authorize = AsyncMock(return_value=case)
    error = lambda status, **kwargs: JSONResponse({"success": False, "error": kwargs}, status_code=status)
    app = FastAPI()
    app.include_router(create_photographic_review_router(ReviewRuntime(authorize, error, authorize_admission=authorize)))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/jobs/job/photographic-review/revisions",
            json={
                "expected_version": 0,
                "artifact_path": "source.tif",
                "sha256": "a" * 64,
                "scene_id": "input-0000",
                "recipe": {},
                "preview_only": True,
            },
        )
    assert response.status_code == 413 and "Retry-After" not in response.headers
    assert response.json()["error"]["code"] == "RESOURCE_EXHAUSTED"
    assert response.json()["error"]["retriable"] is False
    case.discover_references.assert_not_awaited()
    case.publish.assert_not_awaited()


async def test_http_enabled_admission_requires_trusted_lightweight_adapter(monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from starlette.responses import JSONResponse

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator.photographic_review_access import ReviewRuntime

    monkeypatch.setenv("TP_PHOTO_REVIEW_ENABLED", "1")
    eager = AsyncMock(return_value=context())
    error = lambda status, **kwargs: JSONResponse({"success": False, "error": kwargs}, status_code=status)
    app = FastAPI()
    app.include_router(create_photographic_review_router(ReviewRuntime(eager, error)))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/jobs/job/photographic-review")
    assert response.status_code == 503
    eager.assert_not_awaited()


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_http_history_uses_history_auth_outside_photographic_admission(monkeypatch, version):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from starlette.responses import JSONResponse

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator.photographic_review_access import ReviewRuntime

    monkeypatch.setenv("TP_PHOTO_REVIEW_ENABLED", "1")
    eager = AsyncMock(side_effect=AssertionError("History must use its lightweight authorization seam"))
    history = AsyncMock(return_value=context())
    execute = AsyncMock(side_effect=AssertionError("History must bypass photographic resource admission"))
    monkeypatch.setattr(PhotographicReviewService, "execute", execute)
    error = lambda status, **kwargs: JSONResponse({"success": False, "error": kwargs}, status_code=status)
    app = FastAPI()
    # History remains available even when no photographic admission adapter is
    # installed; the history adapter still owns authentication and revalidation.
    app.include_router(create_photographic_review_router(ReviewRuntime(eager, error, authorize_history=history)))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/{version}/jobs/job/photographic-review/deliveries")
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    history.assert_awaited_once()
    eager.assert_not_awaited()
    execute.assert_not_awaited()


async def test_cost_reduction_requires_matching_composite_identity():
    state = {
        "decisions": [
            {
                "artifact_path": "master.npy",
                "sha256": "a" * 64,
                "source_identity_sha256": "b" * 64,
                "inspection": {"width": 100, "height": 100, "alpha": False},
            }
        ]
    }
    assert (
        PhotographicReviewService._cost_inspections(
            state,
            "render",
            ("master.npy", "a" * 64),
            {"source_identity_sha256": "c" * 64},
        )
        is None
    )
    assert PhotographicReviewService._cost_inspections(
        state,
        "render",
        ("master.npy", "a" * 64),
        {"source_identity_sha256": "b" * 64},
    ) == [state["decisions"][0]["inspection"]]


async def test_catalogue_cost_metadata_requires_exact_tenant_job_binding_and_source():
    from transformation_portal.api.v1.photographic_review import RevisionRequest

    service = PhotographicReviewService(configuration())
    case = context()
    ref = {"artifact_path": "master.npy", "sha256": "a" * 64, "source_identity_sha256": "b" * 64}
    inspection = {"width": 6708, "height": 4472, "alpha": False, "private_bytes": b"must not be retained"}
    service._remember_inspection(case, ref, inspection)
    payload = RevisionRequest(
        artifact_path=ref["artifact_path"],
        sha256=ref["sha256"],
        source_identity_sha256=ref["source_identity_sha256"],
        scene_id="input-0000",
        expected_version=0,
        recipe={},
    )
    state = {"state_version": 0, "decisions": []}
    planned = service._admission_inspections(case, state, "revision", (payload,), {})
    assert planned == [{"width": 6708, "height": 4472, "alpha": False}]
    assert service._resource_cost(case, state, "revision", (payload,), planned).memory_bytes < 6 * GIB
    assert state["decisions"] == []
    for changed in [replace(case, tenant_id="other"), replace(case, job_id="other"), replace(case, binding_identity="other")]:
        assert service._admission_inspections(changed, state, "revision", (payload,), {}) is None
    for changed in [
        payload.model_copy(update={"sha256": "c" * 64}),
        payload.model_copy(update={"source_identity_sha256": "c" * 64}),
    ]:
        assert service._admission_inspections(case, state, "revision", (changed,), {}) is None
    await service.close()
    assert not service._inspection_costs


async def test_catalogue_cost_metadata_is_bounded_lru_and_disabled_with_legacy_lane():
    service = PhotographicReviewService(configuration())
    metadata = {"width": 100, "height": 100, "alpha": False}
    case = context()
    for index in range(256):
        service._remember_inspection(case, {"artifact_path": str(index), "sha256": f"{index:064x}"}, metadata)
    assert len(service._inspection_costs) == 256
    assert service._admission_inspections(case, {"decisions": []}, "render", ("0", "0" * 64), {}) == [metadata]
    service._remember_inspection(case, {"artifact_path": "256", "sha256": f"{256:064x}"}, metadata)
    assert len(service._inspection_costs) == 256
    assert any(key[3] == "0" for key in service._inspection_costs)
    assert not any(key[3] == "1" for key in service._inspection_costs)
    for tenant in range(1, 5):
        for index in range(256):
            service._remember_inspection(
                replace(case, tenant_id=str(tenant)), {"artifact_path": str(index), "sha256": f"{index:064x}"}, metadata
            )
    assert len(service._inspection_costs) == 1024
    assert all(key[0] != case.tenant_id for key in service._inspection_costs)
    legacy = PhotographicReviewService(ReviewAdmissionConfig())
    legacy._remember_inspection(case, {"artifact_path": "source", "sha256": "a" * 64}, metadata)
    assert not legacy._inspection_costs
    await service.close()
    await legacy.close()


async def test_untrusted_auth_work_is_bounded_without_claiming_a_tenant_or_io_slot():
    controller = PhotographicAdmissionController(configuration(outstanding=3, tenant_outstanding=2, queued=2, tenant_queued=1))
    async with controller.authorization(), controller.authorization(), controller.request("a"):
        assert controller.snapshot()["outstanding"] == 3
        assert controller.snapshot()["active"] == 0
        with pytest.raises(ReviewOverload) as caught:
            async with controller.authorization():
                pytest.fail("unbounded pre-authorization")
        assert caught.value.status == 503
    assert controller.snapshot()["outstanding"] == 0


async def test_scratch_checks_other_remaining_commitments_without_double_counting_allocated_bytes(monkeypatch, tmp_path):
    import tempfile
    from pathlib import Path

    from transformation_portal.orchestrator import photographic_admission as module

    root = Path(tempfile.gettempdir()).resolve()
    controller = PhotographicAdmissionController(configuration())
    bulk = await controller.acquire("a", ReviewCost(1, 6 * GIB, interactive=False, filesystems=((root, 6 * GIB),)))
    inspection = await controller.acquire("b", ReviewCost(1, 2 * GIB, filesystems=((root, 2 * GIB),)))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * GIB - 1))
    with pytest.raises(ReviewOverload):
        inspection.check_scratch(0)  # Six other + two own + two filesystem floor.
    owned = tmp_path / "owned"
    owned.write_bytes(b"x" * 4096)
    inspection.track_path(owned)
    allocated = owned.stat().st_blocks * 512
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * GIB - allocated))
    inspection.check_scratch(0)
    await inspection.release()
    await bulk.release()


async def test_local_publication_filesystem_is_checked_independently(monkeypatch, tmp_path):
    import tempfile
    from pathlib import Path

    from transformation_portal.orchestrator import photographic_admission as module

    root = Path(tempfile.gettempdir()).resolve()
    output = tmp_path / "publication"
    output.mkdir()
    original_stat = Path.stat

    def filesystem_stat(path, *args, **kwargs):
        info = original_stat(path, *args, **kwargs)
        return SimpleNamespace(st_dev=999999) if path == output else info

    monkeypatch.setattr(Path, "stat", filesystem_stat)
    controller = PhotographicAdmissionController(configuration())
    lease = await controller.acquire("a", ReviewCost(1, 2 * GIB, interactive=False, filesystems=((root, GIB), (output, GIB))))
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * GIB if path == root else 3 * GIB - 1)
    )
    with pytest.raises(ReviewOverload) as caught:
        lease.check_scratch(0)
    assert caught.value.reason == "scratch_unavailable"
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    lease.check_scratch(0)
    await lease.release()


async def test_cas_releases_expensive_capacity_but_retains_owned_scratch(monkeypatch, tmp_path):
    from transformation_portal.api.v1.photographic_review import DeliveryRequest
    from transformation_portal.orchestrator import photographic_admission as module
    from transformation_portal.orchestrator import photographic_review_service as service_module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    service = PhotographicReviewService(configuration())
    state = {
        "state_version": 0,
        "decisions": [
            {
                "artifact_path": "final.tif",
                "sha256": "a" * 64,
                "status": "approved",
                "inspection": {"width": 96, "height": 64, "alpha": False},
            }
        ],
    }
    following = None

    async def mutate(*_args):
        owner = current_review_lease.get()
        assert owner is not None and not owner.released
        assert owner.cost.memory_bytes == owner.cost.io_slots == owner.cost.decode_slots == 0
        assert owner.cost.scratch_bytes > 0
        await tick()
        assert following.done(), "CAS retained the noninteractive I/O slot"
        return state

    monkeypatch.setattr(
        service_module, "get_review_store", lambda: SimpleNamespace(get=AsyncMock(return_value=state), mutate=mutate)
    )

    async def delivery(case, _payload):
        nonlocal following
        owned = tmp_path / "owned-publication"
        owned.write_bytes(b"x" * 4096)
        current_review_lease.get().track_path(owned)
        following = asyncio.create_task(service.admission.acquire("other", ReviewCost(1, 1, interactive=False)))
        await tick()
        assert not following.done()
        return await service.mutate(case, 0, lambda value: value, "delivery", {})

    await service.execute(
        context(),
        delivery,
        DeliveryRequest(expected_version=0, property_name="Test", presets=["website"], approved_sha256=["a" * 64]),
    )
    await (await following).release()
    assert service.admission.snapshot()["active"] == service.admission.snapshot()["outstanding"] == 0


@pytest.mark.parametrize("phase", ["cas", "rollback"])
async def test_retained_publication_scratch_preserves_inspection_headroom(monkeypatch, phase):
    from transformation_portal.orchestrator import photographic_review_service as module

    service = PhotographicReviewService(configuration())
    controller = service.admission
    owner = await controller.acquire("a", ReviewCost(1, 2 * GIB, interactive=False))
    # Model an already allocated 2 GiB private publication without allocating
    # that amount of disk in the synthetic contract suite.
    monkeypatch.setattr(owner, "remaining", lambda _device: 0)
    entered, finish = asyncio.Event(), asyncio.Event()
    state = {"state_version": 0}

    async def hold_commit_or_cleanup(*_args):
        assert owner.cost.io_slots == owner.cost.memory_bytes == owner.cost.decode_slots == 0
        assert owner.cost.scratch_bytes == 2 * GIB and not owner.released
        entered.set()
        await finish.wait()
        return state

    monkeypatch.setattr(module, "get_review_store", lambda: SimpleNamespace(mutate=hold_commit_or_cleanup))
    case = replace(context(), discard=hold_commit_or_cleanup)

    async def publish():
        token = current_review_lease.set(owner)
        try:
            async with module.publication(case) as paths:
                paths.append("photographic-review/owned/file")
                if phase == "rollback":
                    raise OSError("controlled publication failure")
                await service.mutate(case, 0, lambda value: value, "delivery", {})
        finally:
            current_review_lease.reset(token)
            await owner.release()

    publication = asyncio.create_task(publish())
    oversized = None
    inspection = bounded = None
    try:
        await entered.wait()
        oversized = asyncio.create_task(controller.acquire("large", ReviewCost(1, 62 * GIB, interactive=False)))
        await tick()
        assert not oversized.done(), "retained publication scratch consumed protected inspection capacity"
        inspection = await controller.acquire("a", ReviewCost(1, 2 * GIB))
        bounded = await controller.acquire("bounded", ReviewCost(1, 60 * GIB, interactive=False))
        assert controller.snapshot()["scratch_reserved_bytes"] == 64 * GIB
        assert owner.cost.io_slots == 0 and bounded.cost.io_slots == inspection.cost.io_slots == 1
        await inspection.release()
        await bounded.release()
        finish.set()
        if phase == "rollback":
            with pytest.raises(OSError, match="controlled publication"):
                await publication
        else:
            await publication
        following = await oversized
        await following.release()
    finally:
        finish.set()
        await asyncio.gather(publication, return_exceptions=True)
        for lease in (inspection, bounded):
            if lease is not None:
                await lease.release()
        if oversized is not None:
            if not oversized.done():
                oversized.cancel()
            result = (await asyncio.gather(oversized, return_exceptions=True))[0]
            if not isinstance(result, BaseException):
                await result.release()
        await service.close()


async def test_publication_rollback_releases_expensive_capacity_before_cleanup():
    from transformation_portal.orchestrator.photographic_review_service import publication

    controller = PhotographicAdmissionController(configuration())
    lease = await controller.acquire("a", ReviewCost(1, 1, interactive=False))

    async def discard(paths):
        assert paths == ["photographic-review/owned/file"]
        assert lease.cost.io_slots == lease.cost.decode_slots == lease.cost.memory_bytes == 0
        assert not lease.released

    case = replace(context(), discard=discard)
    token = current_review_lease.set(lease)
    try:
        with pytest.raises(OSError, match="controlled"):
            async with publication(case) as paths:
                paths.append("photographic-review/owned/file")
                raise OSError("controlled failure")
    finally:
        current_review_lease.reset(token)
        await lease.release()


async def test_repeated_cancellation_during_discard_drains_cleanup_before_releasing_scratch(tmp_path):
    from transformation_portal.orchestrator.photographic_review_service import publication

    controller = PhotographicAdmissionController(configuration())
    lease = await controller.acquire("a", ReviewCost(1, GIB, interactive=False))
    path = tmp_path / "owned-publication"
    path.write_bytes(b"x" * 4096)
    lease.track_path(path)
    entered, cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def discard(_paths):
        assert lease.cost.io_slots == 0 and not lease.released
        cleanup_started.set()
        await release_cleanup.wait()
        path.unlink()

    case = replace(context(), discard=discard)

    async def run():
        token = current_review_lease.set(lease)
        try:
            async with publication(case) as paths:
                paths.append("photographic-review/owned/file")
                entered.set()
                await asyncio.Event().wait()
        finally:
            current_review_lease.reset(token)
            await lease.release()

    task = asyncio.create_task(run())
    try:
        await entered.wait()
        task.cancel()
        await cleanup_started.wait()
        task.cancel()
        await tick()
        task.cancel()
        await tick()
        assert not task.done() and not lease.released and path.exists()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not path.exists() and lease.released
    finally:
        release_cleanup.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("scope", ["tenant", "process"])
async def test_full_bulk_waiting_queue_cannot_reject_immediately_eligible_inspection(scope):
    config = (
        configuration() if scope == "tenant" else configuration(outstanding=4, tenant_outstanding=2, queued=2, tenant_queued=1)
    )
    controller = PhotographicAdmissionController(config)
    owner_tenant = "a"
    owner_ticket = await controller.begin_request(owner_tenant)
    owner = await controller.acquire(owner_tenant, ReviewCost(1, 1, interactive=False))
    tenants = ["a"] * 6 if scope == "tenant" else ["b", "c"]
    tickets = [await controller.begin_request(tenant) for tenant in tenants]
    waiters = [asyncio.create_task(controller.acquire(tenant, ReviewCost(1, 1, interactive=False))) for tenant in tenants]
    inspection_ticket = None
    inspection = None
    try:
        await tick()
        assert controller.snapshot()["queued"] == (config.tenant_queued if scope == "tenant" else config.queued)
        inspector = "a" if scope == "tenant" else "d"
        inspection_ticket = await controller.begin_request(inspector)
        inspection = await controller.acquire(inspector, ReviewCost(1, 1))
        assert not inspection.released
        assert controller.snapshot()["active"] == 2
        assert controller.snapshot()["queued"] == len(waiters)
    finally:
        if inspection is not None:
            await inspection.release()
        if inspection_ticket is not None:
            await inspection_ticket.release()
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        for ticket in tickets:
            await ticket.release()
        await owner.release()
        await owner_ticket.release()
    assert controller.snapshot()["outstanding"] == controller.snapshot()["active"] == controller.snapshot()["queued"] == 0
