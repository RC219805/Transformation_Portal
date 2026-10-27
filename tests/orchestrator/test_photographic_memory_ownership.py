"""Failed photographic work keeps its reservation until retained buffers die."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import traceback
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from transformation_portal.api.v1.photographic_review import RevisionRequest
from transformation_portal.orchestrator import photographic_admission as admission
from transformation_portal.orchestrator import photographic_review_service as review
from transformation_portal.orchestrator.photographic_admission import GIB, ReviewAdmissionConfig, ReviewLease
from transformation_portal.orchestrator.photographic_review_access import ReviewContext, ReviewInputError

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
