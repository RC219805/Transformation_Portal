"""Cancellation cannot release scratch paths or admit overlapping image work."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from transformation_portal.api.v1.photographic_review import DeliveryRequest
from transformation_portal.orchestrator import photographic_review_service as service_module
from transformation_portal.orchestrator.photographic_review_access import ReviewContext
from transformation_portal.orchestrator.photographic_review_store import empty_state

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_cancelled_delivery_retains_semaphore_and_scratch_until_export_thread_finishes(monkeypatch, tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), (100, 120, 140)).save(buffer, format="PNG")
    source = buffer.getvalue()
    digest = hashlib.sha256(source).hexdigest()
    ref = {
        "artifact_path": "input-0000/delivery.png",
        "sha256": digest,
        "size_bytes": len(source),
        "source_name": "delivery.png",
        "scene_id": "input-0000",
        "role": "final",
        "input_color": "srgb",
    }
    decision = {
        "artifact_path": ref["artifact_path"],
        "sha256": digest,
        "source_identity_sha256": digest,
        "status": "approved",
        "inspection": {"width": 12, "height": 8},
    }
    state = {**empty_state(), "state_version": 1, "decisions": [decision]}
    store = SimpleNamespace(get=AsyncMock(return_value=state), mutate=AsyncMock())
    monkeypatch.setattr(service_module, "get_review_store", lambda: store)
    temporary_directory = service_module.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        service_module.tempfile,
        "TemporaryDirectory",
        lambda **kwargs: temporary_directory(dir=tmp_path, **kwargs),
    )

    async def materialize(_ref, destination, _limit):
        destination.write_bytes(source)
        return {"sha256": digest, "size_bytes": len(source)}

    publish = AsyncMock()
    context = ReviewContext(
        job_id="cancellation-test",
        tenant_id="local",
        actor={"username": "operator"},
        references=[ref],
        materialize=materialize,
        publish=publish,
    )
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    paths = {}

    def blocked_export(images, output_dir, **_kwargs):
        paths.update(source=images[0].source_path, products=output_dir)
        started.set()
        try:
            if not release.wait(timeout=5):
                raise AssertionError("test did not release the controlled export thread")
            # A cancelled caller must not remove these inputs or this parent
            # while the owned worker can still read and create output files.
            assert images[0].source_path.is_file()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "late-product.txt").write_text("controlled synthetic product")
            return {}
        finally:
            finished.set()

    from transformation_portal import photographic_delivery

    monkeypatch.setattr(photographic_delivery, "export_delivery", blocked_export)
    service = service_module.PhotographicReviewService()
    payload = DeliveryRequest(
        expected_version=1,
        property_name="Synthetic cancellation",
        presets=["retouch_print"],
        approved_sha256=[digest],
    )
    delivery = asyncio.create_task(service.delivery(context, payload))
    second_admitted = asyncio.Event()

    async def second_operation():
        async with service.expensive:
            second_admitted.set()

    second = None
    try:
        assert await asyncio.to_thread(started.wait, 2), "export thread did not start"
        delivery.cancel()
        await asyncio.sleep(0)
        second = asyncio.create_task(second_operation())
        await asyncio.sleep(0)
        assert not delivery.done(), "cancellation released the export's owned resources before its worker stopped"
        assert not second_admitted.is_set(), "a second operation entered while cancelled work still owned image resources"
        assert paths["source"].is_file()
        delivery.cancel()  # Repeated transport cancellation must keep the same ownership boundary.
        await asyncio.sleep(0)
        assert not delivery.done()
        assert not second_admitted.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await delivery
        await second
        assert finished.is_set()
        assert not paths["products"].parent.exists()
        assert list(tmp_path.iterdir()) == []
        publish.assert_not_awaited()
        store.mutate.assert_not_awaited()
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)
        await asyncio.gather(*([delivery, second] if second is not None else [delivery]), return_exceptions=True)
