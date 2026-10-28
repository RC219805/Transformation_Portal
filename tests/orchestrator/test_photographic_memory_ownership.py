"""Failed photographic work keeps its reservation until retained buffers die."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import traceback
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from transformation_portal.api.v1.photographic_review import RevisionRequest
from transformation_portal.orchestrator import photographic_admission as admission
from transformation_portal.orchestrator import photographic_review_service as review
from transformation_portal.orchestrator.photographic_admission import GIB, ReviewAdmissionConfig, ReviewLease
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewContext, ReviewInputError

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def revision_case(monkeypatch):
    raw = io.BytesIO()
    with Image.new("RGB", (8, 6), "white") as image:
        image.save(raw, format="PNG")
    source = raw.getvalue()
    sha = hashlib.sha256(source).hexdigest()
    ref = {
        "artifact_path": "source/final.png",
        "source_name": "final.png",
        "scene_id": "scene",
        "sha256": sha,
        "input_color": "srgb",
    }
    service = review.PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    state = {
        "state_version": 0,
        "groups": [],
        "revisions": [],
        "deliveries": [],
        "decisions": [{**ref, "status": "selected", "inspection": {"width": 8, "height": 6, "alpha": False}}],
    }
    monkeypatch.setattr(service, "state", AsyncMock(return_value=state))
    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    case = SimpleNamespace(service=service, buffers=[], images=[], tasks=[])

    async def materialize(selected, path, maximum):
        assert selected["sha256"] == sha and len(source) <= maximum
        path.write_bytes(source)
        return {"sha256": sha, "size_bytes": len(source)}

    case.context = ReviewContext(
        job_id="job",
        tenant_id="tenant",
        actor={"username": "operator"},
        references=[ref],
        materialize=materialize,
        publish=AsyncMock(side_effect=OSError("controlled publication failure")),
        revalidate=AsyncMock(),
    )
    case.payload = RevisionRequest(
        artifact_path=ref["artifact_path"],
        sha256=sha,
        source_identity_sha256=sha,
        scene_id="scene",
        expected_version=0,
        recipe={"exposure_stops": 0},
        preview_only=False,
    )

    def create(image, *_args, **_kwargs):
        values = np.zeros(8 * 1024**2, dtype=np.uint8)
        case.buffers.append(weakref.ref(values))
        case.images.append(weakref.ref(image))
        return {
            "source_bytes": values,
            "preview_bytes": source,
            "receipt": {"revision_sha256": hashlib.sha256(values).hexdigest()},
        }

    monkeypatch.setattr("transformation_portal.photographic_delivery.create_revision", create)
    case.start = lambda: asyncio.create_task(service.execute(case.context, service.revision, case.payload))
    try:
        yield case
    finally:
        await asyncio.gather(*case.tasks, return_exceptions=True)
        await service.close()


async def test_failed_publication_retains_memory_while_revision_result_is_still_live(revision_case, monkeypatch):
    case = revision_case
    started, finish = asyncio.Event(), asyncio.Event()
    original = ReviewLease.release_expensive

    async def observe(lease, **options):
        await original(lease, **options)
        started.set()
        await finish.wait()

    monkeypatch.setattr(ReviewLease, "release_expensive", observe)
    task = case.start()
    case.tasks.append(task)
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert case.buffers and all(value() is not None for value in case.buffers)
        snapshot = case.service.admission.snapshot()
        assert snapshot["memory_reserved_bytes"] > 0, "live revision bytes lost their memory reservation during cleanup"
    finally:
        finish.set()
    with pytest.raises(OSError, match="controlled publication failure"):
        await task


async def test_successful_preview_drops_captured_method_buffers_before_lease_release(revision_case, monkeypatch):
    case = revision_case
    case.payload = case.payload.model_copy(update={"preview_only": True})
    original = ReviewLease.release

    async def release(lease):
        try:
            assert case.buffers and all(value() is None for value in case.buffers)
            assert all(value() is None for value in case.images)
        finally:
            await original(lease)

    monkeypatch.setattr(ReviewLease, "release", release)
    response = await case.start()
    assert response["state_version"] == 0 and response["preview_data_url"].startswith("data:image/png;base64,")
    case.context.publish.assert_not_awaited()


@pytest.mark.parametrize("failure", ["publication", "worker_cause", "worker_context"])
async def test_failed_frames_release_real_buffers_before_final_lease(revision_case, monkeypatch, failure):
    case = revision_case
    releases = []
    original = ReviewLease.release

    async def release(lease):
        try:
            if case.buffers:
                assert all(value() is None for value in case.buffers), "failed worker/publication tracebacks retained arrays"
                assert all(value() is None for value in case.images), "failed revision frame retained its source snapshot"
                releases.append(lease)
        finally:
            await original(lease)

    monkeypatch.setattr(ReviewLease, "release", release)
    if failure != "publication":

        def fail(image, *_args, **_kwargs):
            values = np.zeros(8 * 1024**2, dtype=np.uint8)
            case.buffers.append(weakref.ref(values))
            case.images.append(weakref.ref(image))
            try:
                raise RuntimeError("inner worker failure")
            except RuntimeError as error:
                if failure == "worker_cause":
                    raise ReviewInputError("public revision diagnostic") from error
                raise ReviewInputError("public revision diagnostic")

        monkeypatch.setattr("transformation_portal.photographic_delivery.create_revision", fail)
    task = case.start()
    case.tasks.append(task)
    expected = OSError if failure == "publication" else ReviewInputError
    message = "controlled publication failure" if failure == "publication" else "public revision diagnostic"
    with pytest.raises(expected, match=message) as caught:
        await task
    assert case.buffers and releases
    assert caught.value.__traceback__ is not None
    if failure != "publication":
        chain = caught.value.__cause__ if failure == "worker_cause" else caught.value.__context__
        assert isinstance(chain, RuntimeError) and str(chain) == "inner worker failure"
        assert any(frame.name == "fail" for frame in traceback.extract_tb(chain.__traceback__))
    assert case.service.admission.snapshot()["memory_reserved_bytes"] == 0


@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_worker_drops_its_buffers_before_final_release(revision_case, monkeypatch, worker_fails):
    case = revision_case
    started, finish = threading.Event(), threading.Event()
    original = ReviewLease.release

    async def release(lease):
        try:
            assert case.buffers and all(value() is None for value in case.buffers)
            assert all(value() is None for value in case.images)
        finally:
            await original(lease)

    def fail(image, *_args, **_kwargs):
        values = np.zeros(8 * 1024**2, dtype=np.uint8)
        case.buffers.append(weakref.ref(values))
        case.images.append(weakref.ref(image))
        started.set()
        assert finish.wait(5)
        if worker_fails:
            raise RuntimeError("worker failed while request was cancelled")
        return {"source_bytes": values, "preview_bytes": b"unused"}

    monkeypatch.setattr(ReviewLease, "release", release)
    monkeypatch.setattr("transformation_portal.photographic_delivery.create_revision", fail)
    task = case.start()
    case.tasks.append(task)
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert case.buffers[0]() is not None
        assert case.service.admission.snapshot()["memory_reserved_bytes"] > 0
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert case.service.admission.snapshot()["memory_reserved_bytes"] == 0


@pytest.fixture
async def catalogue_case(revision_case, monkeypatch):
    case = revision_case
    raw = io.BytesIO()
    with Image.new("RGB", (8, 6), "white") as image:
        image.save(raw, format="PNG")
    # A valid encoded PNG with a substantial immutable source snapshot. Track
    # its DeliveryImage owner rather than relying on reservation counters alone.
    source = raw.getvalue() + bytes(8 * 1024**2)
    sha = hashlib.sha256(source).hexdigest()
    ref = {**case.context.references[0], "role": "final", "sha256": sha}

    async def materialize(selected, path, maximum):
        assert selected["sha256"] == sha and len(source) <= maximum
        path.write_bytes(source)
        return {"sha256": sha, "size_bytes": len(source)}

    case.context = replace(case.context, references=[ref], materialize=materialize)
    case.service.state.return_value["decisions"] = []
    load_image = review.load_image

    async def track(context, selected):
        snapshot = await load_image(context, selected)
        assert len(snapshot.source_bytes) > 8 * 1024**2
        case.images.append(weakref.ref(snapshot))
        return snapshot

    monkeypatch.setattr(review, "load_image", track)
    case.start = lambda: asyncio.create_task(case.service.execute(case.context, case.service.catalogue))
    return case


async def test_failed_catalogue_metadata_drops_snapshot_before_per_image_lease_release(catalogue_case, monkeypatch):
    case = catalogue_case
    releases = []
    original = ReviewLease.release

    async def release(lease):
        try:
            if case.images:
                assert all(value() is None for value in case.images), "catalogue retained its source after its image turn"
                assert all(value() is None for value in case.buffers), "metadata worker retained its failed allocation"
                releases.append(lease)
        finally:
            await original(lease)

    def fail(image):
        values = np.zeros(8 * 1024**2, dtype=np.uint8)
        case.buffers.append(weakref.ref(values))
        raise ReviewInputError("controlled catalogue metadata failure")

    monkeypatch.setattr(ReviewLease, "release", release)
    monkeypatch.setattr(review, "_metadata", fail)
    task = case.start()
    case.tasks.append(task)
    with pytest.raises(ReviewInputError, match="controlled catalogue metadata failure"):
        await task
    assert releases and case.images and case.buffers
    assert case.service.admission.snapshot()["memory_reserved_bytes"] == 0


@pytest.mark.parametrize("worker_fails", [False, True])
async def test_repeatedly_cancelled_catalogue_drops_snapshot_before_per_image_lease_release(
    catalogue_case, monkeypatch, worker_fails
):
    case = catalogue_case
    started, finish = threading.Event(), threading.Event()
    releases = []
    original = ReviewLease.release

    async def release(lease):
        try:
            if case.images:
                assert all(value() is None for value in case.images), "cancelled catalogue retained its source image"
                assert all(value() is None for value in case.buffers), "cancelled metadata worker retained its allocation"
                releases.append(lease)
        finally:
            await original(lease)

    def metadata(image):
        values = np.zeros(8 * 1024**2, dtype=np.uint8)
        case.buffers.append(weakref.ref(values))
        started.set()
        assert finish.wait(5)
        if worker_fails:
            raise ReviewInputError("catalogue metadata failed while cancelled")
        return {"width": 8, "height": 6, "alpha": False}

    monkeypatch.setattr(ReviewLease, "release", release)
    monkeypatch.setattr(review, "_metadata", metadata)
    task = case.start()
    case.tasks.append(task)
    try:
        assert await asyncio.to_thread(started.wait, 2)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert all(value() is not None for value in case.images)
        assert all(value() is not None for value in case.buffers)
        assert case.service.admission.snapshot()["memory_reserved_bytes"] > 0
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert releases and case.images and case.buffers
    assert case.service.admission.snapshot()["memory_reserved_bytes"] == 0


@pytest.fixture
async def companion_case(revision_case, monkeypatch):
    from transformation_portal.photographic_delivery import NativeWorkingMaster

    case = revision_case
    raw = io.BytesIO()
    np.save(raw, np.zeros((700, 1024, 3), dtype=np.float32), allow_pickle=False)
    source, descriptor = raw.getvalue(), b"{}"
    sha, descriptor_sha = (hashlib.sha256(value).hexdigest() for value in (source, descriptor))
    native = NativeWorkingMaster(descriptor_sha256=descriptor_sha, descriptor_bytes=descriptor)
    ref = {
        **case.context.references[0],
        "artifact_path": "source/master.npy",
        "source_name": "master.npy",
        "role": "working_master",
        "sha256": sha,
        "source_identity_sha256": native.identity(sha),
        "native": {"descriptor": {"source_name": "photograph.json", "sha256": descriptor_sha}},
    }

    async def materialize(selected, path, maximum):
        data = source if selected["sha256"] == sha else descriptor
        assert len(data) <= maximum
        path.write_bytes(data)
        return {"sha256": selected["sha256"], "size_bytes": len(data)}

    case.context = replace(case.context, references=[ref], materialize=materialize)
    case.service.state.return_value["decisions"] = []
    case.load_frames = []
    load_image, spool_image = review.load_image, review.spool_image

    def track_load(context, selected):
        operation = load_image(context, selected)
        case.load_frames.append(operation.cr_frame)
        return operation

    async def track_spool(*args):
        image = await spool_image(*args)
        case.images.append(weakref.ref(image))
        return image

    monkeypatch.setattr(review, "load_image", track_load)
    monkeypatch.setattr(review, "spool_image", track_spool)
    return case


@pytest.mark.parametrize("operation", ["catalogue", "render"])
@pytest.mark.parametrize("outcome", ["failure", "cancel_success", "cancel_failure"])
async def test_native_companion_read_drops_load_snapshot_before_lease_release(companion_case, monkeypatch, operation, outcome):
    from pathlib import Path

    case = companion_case
    started, finish = threading.Event(), threading.Event()
    releases = []
    original_release, original_read = ReviewLease.release, Path.read_bytes

    async def release(lease):
        try:
            if case.images:
                assert case.load_frames
                for frame in case.load_frames:
                    assert "data" not in frame.f_locals, "closed load_image retained the native source snapshot"
                    assert "values" not in frame.f_locals, "closed load_image retained companion snapshots"
                assert all(value() is None for value in case.images), "closed load_image retained its path-backed source"
                assert all(value() is None for value in case.buffers), "companion worker retained its allocation"
                releases.append(lease)
        finally:
            await original_release(lease)

    def read(path):
        if path.name != "photograph.json":
            return original_read(path)
        values = np.zeros(8 * 1024**2, dtype=np.uint8)
        case.buffers.append(weakref.ref(values))
        started.set()
        assert finish.wait(5)
        if outcome != "cancel_success":
            raise OSError("controlled native companion read failure")
        return original_read(path)

    monkeypatch.setattr(ReviewLease, "release", release)
    monkeypatch.setattr(Path, "read_bytes", read)
    if operation == "catalogue":
        invocation = case.service.execute(case.context, case.service.catalogue)
    else:
        ref = case.context.references[0]
        invocation = case.service.execute(
            case.context,
            case.service.render,
            ref["artifact_path"],
            ref["sha256"],
            source_identity_sha256=ref["source_identity_sha256"],
        )
    task = asyncio.create_task(invocation)
    case.tasks.append(task)
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert all(value() is not None for value in case.images)
        assert case.service.admission.snapshot()["memory_reserved_bytes"] > 0
        if outcome != "failure":
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done()
            assert case.service.admission.snapshot()["memory_reserved_bytes"] > 0
    finally:
        finish.set()
    error = OSError if outcome == "failure" else asyncio.CancelledError
    with pytest.raises(error):
        await task
    assert releases and case.images and case.buffers
    assert case.service.admission.snapshot()["memory_reserved_bytes"] == 0


@pytest.fixture
async def private_render_case(catalogue_case, monkeypatch):
    case = catalogue_case
    case.display_started, case.display_finish = threading.Event(), threading.Event()
    case.render_frames, case.releases = [], []
    render_response, original_release = case.service._render_response, ReviewLease.release

    def track(*args, **kwargs):
        operation = render_response(*args, **kwargs)
        case.render_frames.append(operation.cr_frame)
        return operation

    async def release(lease):
        try:
            if case.images:
                for frame in case.render_frames:
                    assert "image" not in frame.f_locals, "nested render retained its source snapshot"
                    assert "raw" not in frame.f_locals, "nested render retained its encoded display"
                assert all(value() is None for value in case.images), "nested render retained its source owner"
                assert all(value() is None for value in case.buffers), "render worker retained its failed allocation"
                assert case.service.admission.snapshot()["memory_reserved_bytes"] >= lease.cost.memory_bytes
                case.releases.append(lease)
        finally:
            await original_release(lease)

    monkeypatch.setattr(case.service, "_render_response", track)
    monkeypatch.setattr(ReviewLease, "release", release)
    case.outcome = "failure"

    def display(image, *_args, **_kwargs):
        values = np.zeros(8 * 1024**2, dtype=np.uint8)
        case.buffers.append(weakref.ref(values))
        case.display_started.set()
        assert case.display_finish.wait(5)
        if case.outcome not in {"cancel_success", "response_failure"}:
            raise ReviewInputError("controlled private display failure")
        return (memoryview(values) if case.outcome == "response_failure" else b"unused"), 8, 6

    monkeypatch.setattr(review, "_display", display)

    def start(context=None):
        context = case.context if context is None else context
        ref = context.references[0]
        task = asyncio.create_task(
            case.service.execute(
                context, case.service.render, ref["artifact_path"], ref["sha256"], source_identity_sha256=ref["sha256"]
            )
        )
        case.tasks.append(task)
        return task

    case.start = start
    try:
        yield case
    finally:
        case.display_finish.set()


async def settle_private_render(case, task, outcome):
    try:
        assert await asyncio.to_thread(case.display_started.wait, 2)
        assert case.images and all(value() is not None for value in case.images)
        assert case.service.admission.snapshot()["memory_reserved_bytes"] > 0
        if outcome.startswith("cancel"):
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done()
            assert case.service.admission.snapshot()["memory_reserved_bytes"] > 0
    finally:
        case.display_finish.set()
    expected = asyncio.CancelledError if outcome.startswith("cancel") else ReviewInputError
    message = (
        None
        if outcome.startswith("cancel")
        else ("controlled display response failure" if outcome == "response_failure" else "controlled private display failure")
    )
    with pytest.raises(expected, match=message) as caught:
        await task
    if not outcome.startswith("cancel"):
        owner = "fail_response" if outcome == "response_failure" else "display"
        assert any(frame.name == owner for frame in traceback.extract_tb(caught.value.__traceback__))
    assert case.releases and case.buffers
    assert case.service.admission.snapshot()["memory_reserved_bytes"] == 0


@pytest.mark.parametrize("outcome", ["failure", "cancel_success", "cancel_failure"])
async def test_private_render_drops_nested_source_before_lease_release(private_render_case, outcome):
    case = private_render_case
    case.outcome = outcome
    await settle_private_render(case, case.start(), outcome)


async def test_failed_render_response_drops_nested_encoded_output_before_lease_release(private_render_case, monkeypatch):
    case = private_render_case
    case.outcome = "response_failure"

    def fail_response(*_args, **_kwargs):
        raise ReviewInputError("controlled display response failure")

    monkeypatch.setattr(review, "Response", fail_response)
    await settle_private_render(case, case.start(), case.outcome)


@pytest.mark.parametrize("outcome", ["response_failure", "failure", "cancel_success", "cancel_failure"])
async def test_completed_executor_work_item_cannot_retain_buffers_after_refund(private_render_case, monkeypatch, outcome):
    from concurrent.futures import thread

    case = private_render_case
    case.outcome = outcome
    completed, finish = threading.Event(), threading.Event()
    run = thread._WorkItem.run

    def retain_completed_item(item):
        run(item)
        if threading.current_thread().name.startswith("tp-photo") and case.display_started.is_set():
            completed.set()
            assert finish.wait(5)

    monkeypatch.setattr(thread._WorkItem, "run", retain_completed_item)
    if outcome == "response_failure":

        def fail_response(*_args, **_kwargs):
            raise ReviewInputError("controlled display response failure")

        monkeypatch.setattr(review, "Response", fail_response)
    try:
        await settle_private_render(case, case.start(), outcome)
        assert await asyncio.to_thread(completed.wait, 2), "the executor item must remain alive after its Future completes"
        assert not finish.is_set()
    finally:
        finish.set()


async def test_direct_queued_worker_drains_repeated_cancellation_before_dropping_inputs(monkeypatch):
    controller = admission.PhotographicAdmissionController(ReviewAdmissionConfig(enabled=True))
    started = [threading.Event(), threading.Event()]
    finish, invoked = threading.Event(), threading.Event()
    submitted = asyncio.Event()

    def occupy(index):
        started[index].set()
        assert finish.wait(5)

    blockers = [asyncio.create_task(controller.run_blocking(occupy, index)) for index in range(2)]
    queued = None
    try:
        for event in started:
            assert await asyncio.to_thread(event.wait, 2)
        pool = controller._executor
        assert pool is not None
        submit = pool.submit

        def observe_submission(*args, **kwargs):
            future = submit(*args, **kwargs)
            submitted.set()
            return future

        monkeypatch.setattr(pool, "submit", observe_submission)
        values = np.zeros(8 * 1024**2, dtype=np.uint8)
        source = weakref.ref(values)

        def operation(value):
            invoked.set()
            assert value is source()
            return value

        queued = asyncio.create_task(controller.run_blocking(operation, values))
        del values
        await asyncio.wait_for(submitted.wait(), 2)
        for _ in range(3):
            queued.cancel()
            await asyncio.sleep(0)
        assert not queued.done() and not invoked.is_set()
        assert source() is not None, "queued work must keep its input until the real worker consumes it"
        finish.set()
        await asyncio.gather(*blockers)
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert invoked.is_set(), "cancellation must not abandon a queued owned worker"
        assert source() is None, "discarded worker output must die before cancellation returns"
    finally:
        finish.set()
        await asyncio.gather(*blockers, *(() if queued is None else (queued,)), return_exceptions=True)
        await controller.close()


@pytest.mark.parametrize("auth_status", [401, 403])
@pytest.mark.parametrize("outcome", ["failure", "cancel_success", "cancel_failure"])
async def test_private_render_fallback_drops_nested_source_before_lease_release(private_render_case, auth_status, outcome):
    case = private_render_case
    case.outcome = outcome
    started, finish = asyncio.Event(), asyncio.Event()
    original_materialize = case.context.materialize
    revoked = set()

    def context(actor):
        async def revalidate():
            if actor in revoked:
                raise ReviewAccessError("Controlled actor revocation", status=auth_status)

        async def materialize(*args):
            if actor == "leader":
                started.set()
                await finish.wait()
                await revalidate()
            return await original_materialize(*args)

        return replace(
            case.context,
            actor={"username": actor},
            binding_identity="same-private-source-binding",
            materialize=materialize,
            revalidate=revalidate,
        )

    leader = case.start(context("leader"))
    try:
        await asyncio.wait_for(started.wait(), 2)
        follower = case.start(context("follower"))
        for _ in range(20):
            if case.service.admission.snapshot()["coalesced"] == 1:
                break
            await asyncio.sleep(0)
        assert case.service.admission.snapshot()["coalesced"] == 1
        revoked.add("leader")
    finally:
        finish.set()
    with pytest.raises(ReviewAccessError) as caught:
        await leader
    assert caught.value.status == auth_status
    await settle_private_render(case, follower, outcome)
    assert case.service.admission.snapshot()["admitted"] == 2


async def test_cancelled_service_close_clears_inspection_cache_after_active_owner_drains(revision_case):
    case = revision_case
    context = replace(case.context, binding_identity="bounded-inspection-binding")
    case.service._remember_inspection(context, context.references[0], {"width": 8, "height": 6, "alpha": False})
    assert len(case.service._inspection_costs) == 1
    owner = await case.service.admission.acquire(context.tenant_id, admission.ReviewCost(1, 1))
    closing = asyncio.create_task(case.service.close())
    try:
        for _ in range(20):
            if case.service.admission._closed:
                break
            await asyncio.sleep(0)
        assert case.service.admission._closed
        for _ in range(3):
            closing.cancel()
            await asyncio.sleep(0)
        assert not closing.done() and case.service._inspection_costs
        assert case.service.admission.snapshot()["active"] == 1
        await owner.release()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert case.service.admission.snapshot()["active"] == 0
        assert not case.service._inspection_costs, "cancelled completed shutdown retained inspection metadata"
        await case.service.close()
        assert not case.service._inspection_costs
    finally:
        await owner.release()
        await asyncio.gather(closing, return_exceptions=True)
