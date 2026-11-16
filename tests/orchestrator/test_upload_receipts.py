"""Source-bound upload recovery contracts without service substitution."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from transformation_portal.ingest.upload_recovery import (
    MARKER_FILENAME,
    UploadBatchLock,
    initialize_batch,
    seal_completion,
    verify_completion,
)
from transformation_portal.ingest.upload_staging import IncomingUpload, cleanup_expired_batches, stage_upload_batch
from transformation_portal.orchestrator.photographic_admission import drain_owned
from transformation_portal.orchestrator.upload_receipts import UploadReservation, upload_owner, validate_upload_request_id

pytestmark = pytest.mark.unit


def reservation(root: Path, *, owner: str = "a" * 64) -> UploadReservation:
    batch = "upload_" + uuid.uuid4().hex
    return UploadReservation(
        batch, "tenant_a", owner, str(uuid.uuid4()), uuid.uuid4().hex, str(root / batch), "staging", 1.0, 10.0
    )


def sealed(root: Path):
    item = reservation(root)
    initialize_batch(item)
    result = stage_upload_batch(
        upload_root=root,
        uploads=[IncomingUpload("nested/test.txt", io.BytesIO(b"original"))],
        now=item.created_at,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    seal_completion(item, result)
    return item, result


def test_upload_owner_is_composite_and_role_independent():
    actor = {"accessEmail": "Shared@example.test", "username": "Alice", "role": "operator"}
    assert upload_owner("tenant_a", actor) == upload_owner("tenant_a", {**actor, "role": "admin"})
    assert upload_owner("tenant_a", actor) != upload_owner("tenant_a", {**actor, "username": "bob"})
    assert upload_owner("tenant_a", actor) != upload_owner("tenant_b", actor)
    with pytest.raises(ValueError):
        upload_owner("tenant_a", {"username": "alice"})


@pytest.mark.parametrize(
    "bad", ["", "../upload", "A" * 100, str(uuid.uuid4()).upper(), "00000000-0000-0000-0000-000000000000/"]
)
def test_request_id_is_bounded_canonical_uuid(bad):
    with pytest.raises(ValueError):
        validate_upload_request_id(bad)


def test_completion_seal_binds_request_generation_and_all_retained_source_bytes(tmp_path):
    item, result = sealed(tmp_path)
    digest, data = verify_completion(item)
    assert len(digest) == 64
    assert data == result.to_response_data()
    for wrong in [
        replace(item, request_id=str(uuid.uuid4())),
        replace(item, generation=uuid.uuid4().hex),
        replace(item, owner_sha256="b" * 64),
    ]:
        with pytest.raises(ValueError):
            verify_completion(wrong)
    (result.input_dir / "nested/test.txt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="source changed"):
        verify_completion(item)


def test_receipt_without_completion_seal_cannot_recover_ready(tmp_path):
    item = reservation(tmp_path)
    initialize_batch(item)
    stage_upload_batch(
        upload_root=tmp_path,
        uploads=[IncomingUpload("test.txt", io.BytesIO(b"body"))],
        now=1.0,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    with pytest.raises(FileNotFoundError):
        verify_completion(item)


def test_legacy_cleanup_never_deletes_ledger_marked_ready_or_partial_batch(tmp_path):
    item, _ = sealed(tmp_path)
    partial = reservation(tmp_path)
    initialize_batch(partial)
    assert (Path(partial.batch_root) / MARKER_FILENAME).is_file()
    assert cleanup_expired_batches(tmp_path, now=1e12, ttl_seconds=1, retained_input_dirs=[]) == []
    assert Path(item.batch_root).exists() and Path(partial.batch_root).exists()


def test_external_batch_lock_is_exclusive_and_survives_batch_removal(tmp_path):
    item = reservation(tmp_path)
    first, second = UploadBatchLock(tmp_path, item.batch_id), UploadBatchLock(tmp_path, item.batch_id)
    try:
        assert first.acquire()
        assert not second.acquire()
        assert not Path(item.batch_root).exists()
        first.close()
        assert second.acquire()
    finally:
        first.close()
        second.close()


def test_symlinked_lock_directory_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".upload-locks").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        UploadBatchLock(tmp_path, reservation(tmp_path).batch_id)


@pytest.mark.asyncio
async def test_cancelled_owner_waits_for_thread_before_stream_or_lock_release(tmp_path):
    entered, finish = threading.Event(), threading.Event()
    stream = io.BytesIO(b"body")
    item = reservation(tmp_path)
    lock = UploadBatchLock(tmp_path, item.batch_id)
    assert lock.acquire()

    def work():
        entered.set()
        assert finish.wait(timeout=5)
        assert not stream.closed
        assert stream.read() == b"body"

    async def owner():
        try:
            await drain_owned(asyncio.to_thread(work))
        finally:
            stream.close()
            lock.close()

    task = asyncio.create_task(owner())
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not stream.closed
    contender = UploadBatchLock(tmp_path, item.batch_id)
    try:
        assert not contender.acquire()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        assert contender.acquire()
    finally:
        finish.set()
        contender.close()


@pytest.mark.asyncio
async def test_receipt_writes_fail_closed_without_flag_or_managed_tenant(monkeypatch):
    from starlette.requests import Request

    import app

    request = Request({"type": "http", "method": "POST", "path": "/v1/uploads/staging", "headers": []})
    monkeypatch.delenv("TP_PORTAL_UPLOAD_RECEIPTS_ENABLED", raising=False)
    assert (await app._upload_receipt_context(request, write=True))[-1].status_code == 503
    monkeypatch.setenv("TP_PORTAL_UPLOAD_RECEIPTS_ENABLED", "1")

    async def no_managed_tenant(*args, **kwargs):
        return None, None

    monkeypatch.setattr(app, "_pilot_tenant_from_request", no_managed_tenant)
    assert (await app._upload_receipt_context(request, write=True))[-1].status_code == 403


@pytest.mark.asyncio
async def test_exact_receipt_read_remains_api_key_protected(monkeypatch):
    import httpx

    import app

    monkeypatch.setattr(app, "API_KEY_SECRET", "upload-test-key")
    monkeypatch.setattr(app, "ENFORCE_JOB_API_KEY", True)
    app.RATE_LIMIT_BUCKETS.clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://testserver") as client:
        response = await client.get("/v1/uploads/staging/receipts/" + str(uuid.uuid4()))
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_durable_seal_digest_rejects_rewritten_source_and_resealed_inventory(tmp_path):
    item, result = sealed(tmp_path)
    original_digest, _ = verify_completion(item)
    frozen = replace(item, receipt_sha256=original_digest)
    source = result.input_dir / "nested/test.txt"
    source.write_bytes(b"modified")
    baseline = json.loads(result.baseline_manifest_path.read_bytes())
    baseline["records"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    baseline["records"][0]["size_bytes"] = len(source.read_bytes())
    result.baseline_manifest_path.write_text(json.dumps(baseline))
    # A coherent rewrite of on-disk metadata still cannot change the database
    # completion identity pinned before the ready response.
    seal_completion(item, result)
    with pytest.raises(ValueError, match="durable seal digest"):
        verify_completion(frozen)


@pytest.mark.parametrize("status", ["pending", "unknown", "expired"])
def test_nonready_observations_cannot_include_upload_paths(tmp_path, status):
    from pydantic import ValidationError

    from transformation_portal.api.v1.uploads import UploadReceiptData

    item, result = sealed(tmp_path)
    with pytest.raises(ValidationError):
        UploadReceiptData(request_id=item.request_id, status=status, receipt=result.to_response_data())
    with pytest.raises(ValidationError):
        UploadReceiptData(request_id=item.request_id, status="ready", receipt=None)
