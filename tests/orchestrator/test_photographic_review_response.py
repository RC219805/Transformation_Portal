"""Verified download scratch is owned through every FileResponse exit path."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from transformation_portal.orchestrator import photographic_review_service as service_module
from transformation_portal.orchestrator.photographic_review_access import ReviewContext
from transformation_portal.orchestrator.photographic_review_store import empty_state

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _download(monkeypatch, tmp_path):
    body = b"synthetic verified delivery content"
    digest = hashlib.sha256(body).hexdigest()
    source = {"artifact_path": "input-0000/delivery.tif", "sha256": "a" * 64}
    approval = {**source, "status": "approved"}
    file = {
        "filename": "synthetic-approved.zip",
        "sha256": digest,
        "size_bytes": len(body),
        "content_type": "application/zip",
    }
    state = {
        **empty_state(),
        "decisions": [approval],
        "deliveries": [{"id": "bundle", "files": [file], "approvals": [approval], "approved_sha256": [source["sha256"]]}],
    }
    monkeypatch.setattr(service_module, "get_review_store", lambda: SimpleNamespace(get=AsyncMock(return_value=state)))
    mkdtemp = service_module.tempfile.mkdtemp
    monkeypatch.setattr(service_module.tempfile, "mkdtemp", lambda **kwargs: mkdtemp(dir=tmp_path, **kwargs))

    async def materialize(ref, destination, _maximum):
        assert ref == file
        destination.write_bytes(body)
        return {"sha256": digest, "size_bytes": len(body)}

    context = ReviewContext(
        job_id="download-test",
        tenant_id="local",
        actor={"username": "operator"},
        references=[source],
        materialize=materialize,
        publish=AsyncMock(),
    )
    response = await service_module.PhotographicReviewService().download(context, "bundle", file["filename"])
    return response, body, digest


def _scope(range_header=None):
    return {"type": "http", "method": "GET", "headers": [(b"range", range_header)] if range_header else []}


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def _track_cleanup(monkeypatch, directory, *, started=None, release=None):
    calls = []
    rmtree = service_module.shutil.rmtree

    def cleanup(path):
        calls.append(Path(path))
        assert Path(path) == directory, "cleanup must only own this response's scratch directory"
        if started is not None:
            started.set()
        if release is not None:
            assert release.wait(timeout=5), "test did not release controlled download cleanup"
        rmtree(path)

    monkeypatch.setattr(service_module.shutil, "rmtree", cleanup)
    return calls


@pytest.mark.parametrize(
    "range_header,status",
    [(None, 200), (b"bytes=3-9", 206), (b"bytes=999999999-", 416), (b"bytes=broken", 400)],
)
async def test_download_cleans_once_on_normal_range_and_range_error_responses(monkeypatch, tmp_path, range_header, status):
    response, body, digest = await _download(monkeypatch, tmp_path)
    directory = Path(response.path).parent
    retained = tmp_path / "another-download.zip"
    retained.write_bytes(b"preserve unrelated delivery")
    calls = _track_cleanup(monkeypatch, directory)
    messages = []

    async def send(message):
        messages.append(message)

    await response(_scope(range_header), _receive, send)

    assert messages[0]["status"] == status
    content = b"".join(message.get("body", b"") for message in messages[1:])
    if status in {200, 206}:
        headers = dict(messages[0]["headers"])
        assert content == (body if status == 200 else body[3:10])
        assert headers[b"content-type"] == b"application/zip"
        assert headers[b"cache-control"] == b"no-store"
        assert headers[b"x-tp-artifact-sha256"] == digest.encode()
        assert b"synthetic-approved.zip" in headers[b"content-disposition"]
        assert headers[b"accept-ranges"] == b"bytes"
        if status == 206:
            assert headers[b"content-range"] == f"bytes 3-9/{len(body)}".encode()
    assert calls == [directory]
    assert not directory.exists()
    assert retained.read_bytes() == b"preserve unrelated delivery"


async def test_download_cleans_once_when_response_send_fails(monkeypatch, tmp_path):
    response, _, _ = await _download(monkeypatch, tmp_path)
    directory = Path(response.path).parent
    calls = _track_cleanup(monkeypatch, directory)

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("controlled disconnected sender")

    with pytest.raises(OSError, match="controlled disconnected sender"):
        await response(_scope(), _receive, send)

    assert calls == [directory]
    assert not directory.exists()


async def test_cancelled_download_drains_cleanup_across_repeated_cancellation(monkeypatch, tmp_path):
    response, _, _ = await _download(monkeypatch, tmp_path)
    directory = Path(response.path).parent
    sending = asyncio.Event()
    cleanup_started, cleanup_release = threading.Event(), threading.Event()
    calls = _track_cleanup(monkeypatch, directory, started=cleanup_started, release=cleanup_release)

    async def send(message):
        if message["type"] == "http.response.body":
            sending.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(response(_scope(), _receive, send))
    try:
        await asyncio.wait_for(sending.wait(), timeout=2)
        task.cancel()
        assert await asyncio.to_thread(cleanup_started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "repeated cancellation abandoned the response's scratch cleanup"
        assert directory.is_dir()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == [directory]
        assert not directory.exists()
    finally:
        cleanup_release.set()
        await asyncio.gather(task, return_exceptions=True)
