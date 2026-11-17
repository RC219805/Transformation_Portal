"""Source-bound upload recovery contracts without service substitution."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import threading
import uuid
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformation_portal.core.archive_execution_plan import ARCHIVE_OPERATIONS
from transformation_portal.ingest.upload_recovery import (
    MARKER_FILENAME,
    UploadBatchLock,
    initialize_batch,
    seal_completion,
    verify_completion,
)
from transformation_portal.ingest.upload_staging import (
    IncomingUpload,
    UploadStagingError,
    cleanup_expired_batches,
    stage_upload_batch,
)
from transformation_portal.orchestrator.photographic_admission import drain_owned
from transformation_portal.orchestrator.storage.base import JobRecord
from transformation_portal.orchestrator.upload_receipts import (
    UploadOwnershipLost,
    UploadReservation,
    bind_upload_references,
    upload_owner,
    upload_source_paths,
    validate_upload_request_id,
)

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


def partial_multipart(boundary: bytes) -> bytes:
    return (
        b"--" + boundary + b'\r\nContent-Disposition: form-data; name="files"; filename="one.txt"'
        b"\r\nContent-Type: application/octet-stream\r\n\r\nfirst\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="files"; filename="two.txt"'
        b"\r\nContent-Type: application/octet-stream\r\n\r\npartial"
    )


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
    with pytest.raises(ValueError, match="incomplete"):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_read", "during_verification"])
async def test_expired_receipt_read_never_returns_recovered_paths(tmp_path, monkeypatch, phase):
    from starlette.requests import Request

    import app
    from transformation_portal.ingest import upload_recovery
    from transformation_portal.orchestrator.upload_receipts import UploadReceiptExpired

    item = reservation(tmp_path) if phase == "before_read" else sealed(tmp_path)[0]
    clock = [11.0 if phase == "before_read" else 9.0]
    completions = []

    class Store:
        async def get(self, *_args):
            return item

        async def complete(self, *args):
            assert phase == "during_verification" and clock[0] >= item.expires_at
            completions.append(args)
            raise UploadReceiptExpired("staged upload expired")

    async def context(*_args, **_kwargs):
        return SimpleNamespace(tenant_id=item.tenant_id), item.owner_sha256, Store(), None

    async def authorized(*_args, **_kwargs):
        return None

    original_verify = upload_recovery.verify_completion

    def verify(*args):
        assert phase == "during_verification", "expired staging without a seal must not inspect the filesystem"
        result = original_verify(*args)
        clock[0] = 11.0
        return result

    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_resolved_portal_upload_root", lambda: tmp_path)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", authorized)
    monkeypatch.setattr(app, "_now", lambda: clock[0])
    monkeypatch.setattr(upload_recovery, "verify_completion", verify)
    request = Request({"type": "http", "method": "GET", "path": "/v1/uploads/staging/receipts/" + item.request_id})
    response = await app.get_portal_upload_receipt(request, item.request_id)
    assert response.status_code == 200
    assert json.loads(response.body)["data"] == {"request_id": item.request_id, "status": "expired", "receipt": None}
    assert response.headers["cache-control"] == "no-store"
    assert len(completions) == (phase == "during_verification")
    assert item.state == "staging"
    if phase == "before_read":
        assert not Path(item.batch_root).exists()
    else:
        assert Path(item.batch_root).is_dir()


@pytest.mark.asyncio
async def test_upload_completion_past_ttl_is_typed_without_replaying_or_deleting_body(tmp_path, monkeypatch):
    from starlette.requests import Request

    import app
    from transformation_portal.ingest import upload_recovery
    from transformation_portal.orchestrator.upload_receipts import UploadReceiptExpired

    clock = [1.0]
    stream = io.BytesIO(b"upload bytes")
    parsed = app._ParsedPortalUploadPayload([IncomingUpload("test.txt", stream)], None)
    calls = []

    class Store:
        current = None

        async def reserve(self, item):
            self.current = item
            calls.append("reserve")

        async def get(self, *_args):
            return self.current

        async def complete(self, item, *_args):
            assert clock[0] >= item.expires_at
            calls.append("complete")
            raise UploadReceiptExpired("staged upload expired")

    store = Store()

    async def context(*_args, **_kwargs):
        return SimpleNamespace(tenant_id="tenant_a"), "a" * 64, store, None

    async def authorized(*_args, **_kwargs):
        return None

    async def parse(*_args):
        calls.append("parse")
        return parsed

    original_verify = upload_recovery.verify_completion

    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        clock[0] = 12.0
        return result

    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", authorized)
    monkeypatch.setattr(app, "_parse_portal_upload_multipart", parse)
    monkeypatch.setattr(app, "PORTAL_UPLOAD_TTL_SECONDS", 10)
    monkeypatch.setattr(app, "_now", lambda: clock[0])
    monkeypatch.setattr(upload_recovery, "verify_completion", verify)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/uploads/staging",
            "headers": [(b"x-tp-upload-request-id", str(uuid.uuid4()).encode())],
        }
    )
    response = await app._stage_correlated_portal_upload(request, tmp_path)
    assert response.status_code == 409
    body = json.loads(response.body)
    assert body["error"]["code"] == "CONFLICT"
    assert body["error"]["details"]["reason"] == "upload_expired"
    assert body["data"] is None
    assert calls == ["reserve", "parse", "complete"]
    assert stream.closed and store.current.state == "staging"
    assert Path(store.current.batch_root).is_dir()
    lock = UploadBatchLock(tmp_path, store.current.batch_id)
    try:
        assert lock.acquire(), "the completed worker must release its lock without deleting the fenced batch"
    finally:
        lock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["body_limit", "direct_http", "provider", "mixed_group"])
async def test_correlated_chunked_upload_preserves_http_limit_and_resource_cleanup(tmp_path, monkeypatch, failure):
    import httpx

    import app

    calls = []
    streams = []

    class Store:
        current = None

        async def reserve(self, item):
            calls.append("reserve")
            if failure == "provider":
                raise RuntimeError("receipt provider unavailable")
            if failure == "mixed_group":
                raise ExceptionGroup(
                    "multiple request failures",
                    [app.HTTPException(status_code=413), RuntimeError("receipt provider unavailable")],
                )
            self.current = item

        async def get(self, *_args):
            return self.current

        async def complete(self, *_args):
            pytest.fail("an oversized body must not publish ready")

    store = Store()

    async def context(request, **_kwargs):
        assert "content-length" not in request.headers
        return SimpleNamespace(tenant_id="tenant_a"), "a" * 64, store, None

    async def authorized(*_args, **_kwargs):
        return None

    original_headers = app._parse_portal_upload_part_headers

    def track_headers(*args):
        part = original_headers(*args)
        if part.file_stream is not None:
            streams.append(part.file_stream)
        return part

    def unexpected_staging(*_args, **_kwargs):
        pytest.fail("the failed request must not start a staging worker")

    async def direct_http_rejection(*_args):
        raise app.HTTPException(status_code=413, detail="request body rejected")

    boundary = b"tp-chunked-limit"
    first = partial_multipart(boundary)

    async def chunks():
        yield first
        yield b"x" * 64
        yield b"\r\n--" + boundary + b"--\r\n"

    monkeypatch.setattr(app, "API_KEY_SECRET", "upload-test-key")
    monkeypatch.setattr(app, "ENFORCE_JOB_API_KEY", True)
    monkeypatch.setattr(app, "RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(app, "MAX_UPLOAD_REQUEST_BYTES", len(first) + 16)
    monkeypatch.setattr(app, "_portal_staged_uploads_enabled", lambda _actor: True)
    monkeypatch.setattr(app, "_resolved_portal_upload_root", lambda: tmp_path)
    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", authorized)
    monkeypatch.setattr(app, "_parse_portal_upload_part_headers", track_headers)
    monkeypatch.setattr(app, "stage_upload_batch", unexpected_staging)
    if failure == "direct_http":
        monkeypatch.setattr(app, "_parse_portal_upload_multipart", direct_http_rejection)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/uploads/staging",
            content=chunks(),
            headers={
                "x-api-key": "upload-test-key",
                "X-TP-Upload-Request-ID": str(uuid.uuid4()),
                "content-type": "multipart/form-data; boundary=" + boundary.decode(),
            },
        )
    expected_status = 413 if failure in {"body_limit", "direct_http"} else 503
    assert response.status_code == expected_status
    assert response.json()["schema"] == "tp.orchestrator.error.v1"
    assert response.json()["error"]["code"] == ("REQUEST_TOO_LARGE" if expected_status == 413 else "SERVICE_UNAVAILABLE")
    assert response.json()["data"] is None
    assert calls == ["reserve"]
    if failure in {"provider", "mixed_group"}:
        assert streams == [] and store.current is None
    else:
        assert len(streams) == (2 if failure == "body_limit" else 0)
        assert all(stream.closed for stream in streams)
        assert store.current.state == "staging" and store.current.receipt_sha256 is None
        assert Path(store.current.batch_root).is_dir()
        lock = UploadBatchLock(tmp_path, store.current.batch_id)
        try:
            assert lock.acquire(), "parser failure must release the drained request's stable batch lock"
        finally:
            lock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("correlated", [False, True])
async def test_cancelled_multipart_request_closes_custom_spools_before_staging(tmp_path, monkeypatch, correlated):
    from starlette.requests import Request

    import app

    streams = []
    waiting = asyncio.Event()
    blocked = asyncio.Event()
    received = False

    class Store:
        current = None

        async def reserve(self, item):
            self.current = item

        async def get(self, *_args):
            return self.current

        async def complete(self, *_args):
            pytest.fail("a cancelled parser must not publish ready")

    store = Store()

    async def context(*_args, **_kwargs):
        return SimpleNamespace(tenant_id="tenant_a"), "a" * 64, store, None

    async def authorized(*_args, **_kwargs):
        return None

    original_headers = app._parse_portal_upload_part_headers

    def track_headers(*args):
        part = original_headers(*args)
        if part.file_stream is not None:
            streams.append(part.file_stream)
        return part

    def unexpected_staging(*_args, **_kwargs):
        pytest.fail("cancellation before multipart completion must not start a worker")

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": partial_multipart(b"cancel-test"), "more_body": True}
        waiting.set()
        await blocked.wait()
        pytest.fail("the upload request should be cancelled while awaiting its body")

    monkeypatch.setattr(app, "_portal_staged_uploads_enabled", lambda _actor: True)
    monkeypatch.setattr(app, "_resolved_portal_upload_root", lambda: tmp_path)
    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", authorized)
    monkeypatch.setattr(app, "_parse_portal_upload_part_headers", track_headers)
    monkeypatch.setattr(app, "stage_upload_batch", unexpected_staging)
    headers = [(b"content-type", b"multipart/form-data; boundary=cancel-test")]
    if correlated:
        headers.append((b"x-tp-upload-request-id", str(uuid.uuid4()).encode()))
    request = Request({"type": "http", "method": "POST", "path": "/v1/uploads/staging", "headers": headers}, receive)
    task = asyncio.create_task(app.stage_portal_uploads(request))
    try:
        await asyncio.wait_for(waiting.wait(), 5)
        assert len(streams) == 2 and all(not stream.closed for stream in streams)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await request.close()
        assert all(stream.closed for stream in streams)
        if correlated:
            assert store.current.state == "staging" and store.current.receipt_sha256 is None
            assert Path(store.current.batch_root).is_dir()
            lock = UploadBatchLock(tmp_path, store.current.batch_id)
            try:
                assert lock.acquire(), "request cancellation must release its drained batch lock"
            finally:
                lock.close()
        else:
            assert store.current is None
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        for stream in streams:
            stream.close()


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


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_file",
        "empty_directory",
        "extra_symlink",
        "linked_ancestor",
        "linked_source",
        "linked_marker",
        "linked_seal",
        "hardlinked_source",
        "extra_root",
        "extra_metadata",
    ],
)
def test_completion_requires_exact_independent_inventory_at_every_level(tmp_path, mutation):
    from transformation_portal.ingest.upload_recovery import SEAL_FILENAME

    item, result = sealed(tmp_path)
    source = result.input_dir / "nested/test.txt"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"original")
    if mutation == "extra_file":
        (result.input_dir / "unlisted.txt").write_bytes(b"not sealed")
    elif mutation == "empty_directory":
        (result.input_dir / "unlisted").mkdir()
    elif mutation == "extra_symlink":
        (result.input_dir / "unlisted.txt").symlink_to(outside)
    elif mutation == "linked_ancestor":
        (result.input_dir / "nested").rename(result.input_dir / "actual")
        (result.input_dir / "nested").symlink_to("actual", target_is_directory=True)
    elif mutation == "linked_source":
        source.unlink()
        source.symlink_to(outside)
    elif mutation == "hardlinked_source":
        source.unlink()
        os.link(outside, source)
    elif mutation in {"linked_marker", "linked_seal"}:
        path = Path(item.batch_root) / MARKER_FILENAME if mutation == "linked_marker" else result.portal_dir / SEAL_FILENAME
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif mutation == "extra_root":
        (Path(item.batch_root) / "tmp").mkdir()
    else:
        (result.portal_dir / "unlisted.json").write_bytes(b"{}")
    with pytest.raises((ValueError, OSError)):
        verify_completion(item)


def test_recovery_uses_sealed_count_and_existing_configured_file_limit(tmp_path):
    item = reservation(tmp_path)
    initialize_batch(item)
    count = 4097  # A supported configured writer limit, not a new default.
    result = stage_upload_batch(
        upload_root=tmp_path,
        uploads=[IncomingUpload(f"file-{index:04d}.txt", io.BytesIO(b"x")) for index in range(count)],
        now=1.0,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    seal_completion(item, result)
    assert verify_completion(item, max_files=count)[1]["summary"]["file_count"] == count
    assert verify_completion(item)[1] == result.to_response_data()  # GET survives later config reduction.
    with pytest.raises(ValueError, match="inventory"):
        verify_completion(item, max_files=256)


@pytest.mark.parametrize("location", ["input", "metadata", "root"])
def test_completion_rejects_inventory_added_during_source_verification(tmp_path, monkeypatch, location):
    from transformation_portal.ingest import upload_recovery

    item, result = sealed(tmp_path)
    original = upload_recovery._source_digest

    def digest(*args):
        value = original(*args)
        directory = {"input": result.input_dir, "metadata": result.portal_dir, "root": result.batch_root}[location]
        (directory / "unlisted.txt").write_bytes(b"changed during verification")
        return value

    monkeypatch.setattr(upload_recovery, "_source_digest", digest)
    with pytest.raises(ValueError, match="directory changed"):
        verify_completion(item)


def test_metadata_capacity_is_checked_before_a_completion_can_be_sealed(tmp_path):
    from transformation_portal.ingest.upload_recovery import MAX_RECOVERY_METADATA_BYTES

    item, result = sealed(tmp_path)
    with result.capture_metadata_path.open("wb") as stream:
        stream.truncate(MAX_RECOVERY_METADATA_BYTES + 1)
    with pytest.raises(UploadStagingError) as error:
        seal_completion(item, result)
    assert error.value.status_code == 413 and error.value.reason == "upload_recovery_metadata_limit"


def test_completion_seal_capacity_is_a_typed_rejection_without_a_published_seal(tmp_path):
    from transformation_portal.ingest.upload_recovery import MAX_RECOVERY_METADATA_BYTES, SEAL_FILENAME

    item, result = sealed(tmp_path)
    seal_path = result.portal_dir / SEAL_FILENAME
    seal_path.unlink()
    oversized = replace(result, warnings=("x" * MAX_RECOVERY_METADATA_BYTES,))
    assert all(path.stat().st_size < MAX_RECOVERY_METADATA_BYTES for path in result.portal_dir.iterdir())
    with pytest.raises(UploadStagingError) as error:
        seal_completion(item, oversized)
    assert error.value.status_code == 413 and error.value.reason == "upload_recovery_metadata_limit"
    assert not seal_path.exists()


@pytest.mark.asyncio
async def test_empty_upload_read_set_never_touches_filesystem_or_database(monkeypatch):
    import tempfile

    from transformation_portal.orchestrator import upload_receipts

    def unexpected(*args, **kwargs):
        pytest.fail("an empty upload read set must not inspect filesystem paths")

    class Session:
        async def scalar(self, statement):
            pytest.fail("an empty upload read set must not query receipt rows")

        async def scalars(self, statement):
            pytest.fail("an empty upload read set must not query receipt rows")

    monkeypatch.setattr(upload_receipts, "Path", unexpected)
    monkeypatch.setattr(tempfile, "gettempdir", unexpected)
    await bind_upload_references(Session(), JobRecord(id="job", created_at=1.0, request={}), "")


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_id", [None, ""])
async def test_trusted_direct_admission_cannot_omit_tenant_scope_for_a_tracked_batch(tmp_path, monkeypatch, tenant_id):
    monkeypatch.setenv("TP_PORTAL_UPLOAD_ROOT", str(tmp_path))
    record = JobRecord(id="job", created_at=1.0, request={"args": {"input_dir": str(tmp_path / "upload_tracked" / "input")}})

    class Session:
        def __init__(self):
            self.references = []

        async def scalar(self, statement):
            return 2.0

        async def scalars(self, statement):
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(
                        batch_id="upload_tracked",
                        batch_root=str(tmp_path / "upload_tracked"),
                        tenant_id="tenant_a",
                        state="ready",
                        expires_at=10.0,
                    )
                ]
            )

        def add(self, row):
            self.references.append(row)

    session = Session()
    with pytest.raises(UploadOwnershipLost, match="outside this tenant"):
        await bind_upload_references(session, record, tenant_id)
    assert not session.references


@pytest.mark.asyncio
async def test_reservation_conflict_target_only_covers_the_scoped_request_identity(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager

    from sqlalchemy.dialects import postgresql

    from transformation_portal.orchestrator.upload_receipts import PostgresUploadReceiptStore

    item = reservation(tmp_path)
    statements = []

    class Session:
        @asynccontextmanager
        async def begin(self):
            yield self

        async def scalar(self, statement):
            statements.append(str(statement.compile(dialect=postgresql.dialect())))
            return item.batch_id

    @asynccontextmanager
    async def session():
        yield Session()

    store = PostgresUploadReceiptStore(database_url="postgresql+asyncpg://unused/unused_test")
    monkeypatch.setattr(store, "_session", session)
    await store.reserve(item)
    assert len(statements) == 1
    assert "ON CONFLICT (tenant_id, owner_sha256, request_id) DO NOTHING" in statements[0]


@pytest.mark.parametrize("pipeline", ["lux-depth", "lux-depth-v5", "lux-depth-v6"])
def test_photographic_manifests_retain_their_batches_without_output_paths(tmp_path, pipeline):
    args = {name: str(tmp_path / name) for name in ("input_dir", "companions_manifest", "materials_manifest", "output_dir")}
    paths = upload_source_paths({"pipeline": pipeline, "args": args})
    assert set(paths) == {Path(args[name]) for name in ("input_dir", "companions_manifest", "materials_manifest")}


def test_read_selector_contract_tracks_current_frontdoor_path_aliases(tmp_path):
    import app
    from transformation_portal.orchestrator import upload_receipts

    assert upload_receipts._ARCHIVE_DEFAULT_COMMANDS == app.ARCHIVE_GATE_DEFAULT_COMMANDS
    for canonical, aliases, scope in app.PATH_FIELD_SPECS:
        if scope != app.PATH_SCOPE_INPUT:
            continue
        if canonical in {"policy_yaml", "rights_jsonl"}:
            command = "rights-apply" if canonical == "policy_yaml" else "manifest-build"
            request = {"pipeline": "archive-gate-a", "args": {"archive_command": command}}
        elif canonical in {"companions_manifest", "materials_manifest"}:
            request = {"pipeline": "lux-depth-v5", "args": {}}
        elif canonical == "archive_root":
            request = {"pipeline": "archive-gate-a", "args": {}}
        else:
            request = {"pipeline": "lux-depth-v3", "args": {}}
        for alias in aliases:
            request["args"] = {**request["args"], alias: str(tmp_path / alias)}
            assert tmp_path / alias in upload_source_paths(request), alias


@pytest.mark.parametrize("command", tuple(ARCHIVE_OPERATIONS))
def test_archive_selector_roles_include_every_read_alias_and_exclude_outputs(tmp_path, command):
    from transformation_portal.core.archive_execution_plan import ARCHIVE_OPERATIONS

    operation = ARCHIVE_OPERATIONS[command]
    args = {"archive_command": command, "input_dir": str(tmp_path / "input")}
    expected = {tmp_path / "input"}
    for name in operation.files + operation.directories + operation.outputs:
        parts = name.split("_")
        alias = parts[0] + "".join(part.title() for part in parts[1:])
        args[alias] = str(tmp_path / name)
        if name not in operation.outputs:
            expected.add(tmp_path / name)
    assert set(upload_source_paths({"pipeline": operation.pipeline, "args": args})) == expected


@pytest.mark.asyncio
async def test_all_distinct_input_alias_batches_are_locked_once_in_stable_order(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_PORTAL_UPLOAD_ROOT", str(tmp_path))
    first, second = tmp_path / "upload_aaa" / "input", tmp_path / "upload_zzz" / "input"
    record = JobRecord(
        id="job",
        created_at=1.0,
        request={"args": {"inputDir": str(second), "archiveRoot": str(first / "nested")}},
        effective_request={"args": {"input_dir": str(first), "archive_root": str(second / "nested")}},
    )
    original_resolve = Path.resolve
    resolved_roots = []

    def resolve(path, *args, **kwargs):
        if path == tmp_path:
            resolved_roots.append(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)

    class Session:
        def __init__(self):
            self.locked, self.references = [], []
            self.selection = None

        async def scalar(self, statement):
            assert self.locked == ["upload_aaa", "upload_zzz"], "sample expiry only after every matching row is locked"
            return 2.0

        async def scalars(self, statement):
            self.selection = str(statement)
            self.locked.extend(["upload_aaa", "upload_zzz"])
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(
                        batch_id=batch, batch_root=str(tmp_path / batch), tenant_id="tenant_a", state="ready", expires_at=10.0
                    )
                    for batch in self.locked
                ]
            )

        def add(self, row):
            self.references.append(row)

    session = Session()
    await bind_upload_references(session, record, "tenant_a")
    assert session.locked == ["upload_aaa", "upload_zzz"]
    assert [(row.batch_id, row.job_id) for row in session.references] == [("upload_aaa", "job"), ("upload_zzz", "job")]
    assert len(upload_source_paths(record.request, record.effective_request)) == 4
    assert not resolved_roots, "admission must use absolute ledger identities, independent of the configured root"
    assert "ORDER BY upload_receipts.batch_id" in session.selection and "FOR UPDATE" in session.selection


def test_retained_scan_preserves_all_aliases_even_when_effective_request_exists(tmp_path):
    import app

    retained = set()
    paths = [tmp_path / name for name in ("input-snake", "input-camel", "archive-snake", "archive-camel")]
    job = SimpleNamespace(
        request={"args": {"inputDir": str(paths[1]), "archiveRoot": str(paths[3])}},
        effective_request={"args": {"input_dir": str(paths[0]), "archive_root": str(paths[2])}},
    )
    app._add_staged_input_dir(retained, job)
    assert retained == {str(path.resolve()) for path in paths}


def test_legacy_nested_archive_selector_keeps_its_entire_staged_batch(tmp_path):
    result = stage_upload_batch(
        upload_root=tmp_path, uploads=[IncomingUpload("nested/source.txt", io.BytesIO(b"body"))], now=1.0
    )
    assert cleanup_expired_batches(tmp_path, now=1e12, ttl_seconds=1, retained_input_dirs=[result.input_dir / "nested"]) == []
    assert result.input_dir.exists()


@pytest.mark.parametrize("retained", ["root", "ancestor", "sibling"])
def test_retained_directory_overlap_protects_legacy_batches_without_pinning_siblings(tmp_path, retained):
    result = stage_upload_batch(upload_root=tmp_path, uploads=[IncomingUpload("source.txt", io.BytesIO(b"body"))], now=1.0)
    selector = {"root": tmp_path, "ancestor": tmp_path.parent, "sibling": tmp_path / "unrelated"}[retained]
    removed = cleanup_expired_batches(tmp_path, now=1e12, ttl_seconds=1, retained_input_dirs=[selector])
    assert removed == ([result.batch_id] if retained == "sibling" else [])
    assert result.input_dir.exists() is (retained != "sibling")


@pytest.mark.parametrize("lock_kind", ["regular", "dangling_symlink"])
def test_legacy_cleanup_never_claims_a_markerless_batch_with_permanent_ledger_lock_identity(tmp_path, lock_kind):
    from transformation_portal.ingest.upload_recovery import SEAL_FILENAME

    item, result = sealed(tmp_path)
    (result.batch_root / MARKER_FILENAME).unlink()
    (result.portal_dir / SEAL_FILENAME).unlink()
    if lock_kind == "regular":
        lock = UploadBatchLock(tmp_path, item.batch_id)
        lock.close()
    else:
        directory = tmp_path / ".upload-locks"
        directory.mkdir()
        (directory / (item.batch_id + ".lock")).symlink_to(tmp_path / "missing-lock-target")
    assert cleanup_expired_batches(tmp_path, now=1e12, ttl_seconds=1, retained_input_dirs=[]) == []
    assert result.input_dir.exists()


@pytest.mark.parametrize("state", ["ready", "staging"])
def test_markerless_deletion_requires_a_fenced_deleting_ledger_state(tmp_path, state):
    from transformation_portal.ingest.upload_recovery import delete_owned_batch

    item, result = sealed(tmp_path)
    digest, data = verify_completion(item)
    (result.batch_root / MARKER_FILENAME).unlink()
    item = replace(
        item,
        state=state,
        receipt_sha256=digest if state == "ready" else None,
        response_data=data if state == "ready" else None,
    )
    lock = UploadBatchLock(tmp_path, item.batch_id)
    assert lock.acquire()
    try:
        with pytest.raises(ValueError):
            delete_owned_batch(item, tmp_path)
        assert result.input_dir.exists()
    finally:
        lock.close()


@pytest.mark.parametrize("mutation", ["mismatched_marker", "symlink_root", "wrong_root"])
def test_deleting_retry_never_weakens_root_or_existing_marker_ownership(tmp_path, mutation):
    from transformation_portal.ingest import upload_recovery

    item, result = sealed(tmp_path)
    digest, data = verify_completion(item)
    item = replace(item, state="deleting", receipt_sha256=digest, response_data=data)
    if mutation == "mismatched_marker":
        marker = result.batch_root / MARKER_FILENAME
        payload = json.loads(marker.read_bytes())
        marker.write_bytes(upload_recovery.canonicalize_json({**payload, "owner_sha256": "b" * 64}))
    elif mutation == "symlink_root":
        outside = tmp_path / "outside"
        result.batch_root.rename(outside)
        result.batch_root.symlink_to(outside, target_is_directory=True)
    lock = UploadBatchLock(tmp_path, item.batch_id)
    assert lock.acquire()
    try:
        with pytest.raises(ValueError):
            upload_recovery.delete_owned_batch(item, tmp_path / "wrong" if mutation == "wrong_root" else tmp_path)
        assert result.input_dir.exists()
    finally:
        lock.close()


def test_verification_rejects_an_earlier_file_rewritten_while_a_later_file_is_hashed(tmp_path, monkeypatch):
    from transformation_portal.ingest import upload_recovery

    item = reservation(tmp_path)
    initialize_batch(item)
    result = stage_upload_batch(
        upload_root=tmp_path,
        uploads=[IncomingUpload("a.txt", io.BytesIO(b"original")), IncomingUpload("b.txt", io.BytesIO(b"original"))],
        now=item.created_at,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    seal_completion(item, result)
    original = upload_recovery._source_digest
    verified = []

    def digest(directory_fd, name, size):
        value = original(directory_fd, name, size)
        verified.append(name)
        if len(verified) == 2:
            (result.input_dir / verified[0]).write_bytes(b"modified")
        return value

    monkeypatch.setattr(upload_recovery, "_source_digest", digest)
    with pytest.raises(ValueError, match="changed"):
        verify_completion(item)
    assert len(verified) == 2


def test_source_identities_are_rechecked_at_the_end_of_complete_verification(tmp_path, monkeypatch):
    from transformation_portal.ingest import upload_recovery

    item, result = sealed(tmp_path)
    original = upload_recovery._verify_input_inventory

    def inventory(*args, **kwargs):
        value = original(*args, **kwargs)
        (result.input_dir / "nested/test.txt").write_bytes(b"modified")
        return value

    monkeypatch.setattr(upload_recovery, "_verify_input_inventory", inventory)
    with pytest.raises(ValueError, match="changed"):
        verify_completion(item)


@pytest.mark.parametrize("metadata", ["marker", "seal", "baseline", "capture", "receipt"])
def test_metadata_identities_survive_until_the_end_of_source_verification(tmp_path, monkeypatch, metadata):
    from transformation_portal.ingest import upload_recovery

    item, result = sealed(tmp_path)
    paths = {
        "marker": result.batch_root / MARKER_FILENAME,
        "seal": result.portal_dir / upload_recovery.SEAL_FILENAME,
        "baseline": result.baseline_manifest_path,
        "capture": result.capture_metadata_path,
        "receipt": result.upload_receipt_path,
    }
    target = paths[metadata]
    original = upload_recovery._source_digest

    def digest(*args, **kwargs):
        value = original(*args, **kwargs)
        target.write_bytes(target.read_bytes() + b" ")
        return value

    monkeypatch.setattr(upload_recovery, "_source_digest", digest)
    with pytest.raises(ValueError, match="changed"):
        verify_completion(item)


def test_verified_nested_directory_cannot_be_replaced_by_a_link_during_later_hashing(tmp_path, monkeypatch):
    from transformation_portal.ingest import upload_recovery

    item = reservation(tmp_path)
    initialize_batch(item)
    result = stage_upload_batch(
        upload_root=tmp_path,
        uploads=[
            IncomingUpload("one/deep/a.txt", io.BytesIO(b"original")),
            IncomingUpload("two/deep/b.txt", io.BytesIO(b"original")),
        ],
        now=item.created_at,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    seal_completion(item, result)
    original = upload_recovery._source_digest
    verified = []

    def digest(directory_fd, name, size):
        value = original(directory_fd, name, size)
        verified.append(name)
        if len(verified) == 2:
            earlier = result.input_dir / ("one" if verified[0] == "a.txt" else "two") / "deep"
            outside = tmp_path / "retained-directory"
            earlier.rename(outside)
            earlier.symlink_to(outside, target_is_directory=True)
        return value

    monkeypatch.setattr(upload_recovery, "_source_digest", digest)
    with pytest.raises(ValueError, match="changed"):
        verify_completion(item)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["dangling_symlink", "hardlink", "acquire_oserror", "acquire_valueerror"])
async def test_one_unavailable_cleanup_lock_does_not_starve_later_batches(tmp_path, monkeypatch, caplog, failure):
    from transformation_portal.ingest import upload_recovery

    bad, bad_result = sealed(tmp_path)
    good, good_result = sealed(tmp_path)
    lock_path = tmp_path / ".upload-locks" / (bad.batch_id + ".lock")
    if failure == "dangling_symlink":
        lock_path.parent.mkdir()
        lock_path.symlink_to(tmp_path / "missing-lock")
    else:
        lock = UploadBatchLock(tmp_path, bad.batch_id)
        lock.close()
        if failure == "hardlink":
            os.link(lock_path, tmp_path / "second-lock-name")
        else:
            inode = lock_path.stat().st_ino
            acquire = UploadBatchLock.acquire

            def failing_acquire(lock, *args, **kwargs):
                if os.fstat(lock.fd).st_ino == inode:
                    raise (OSError if failure == "acquire_oserror" else ValueError)("lock unavailable")
                return acquire(lock, *args, **kwargs)

            monkeypatch.setattr(UploadBatchLock, "acquire", failing_acquire)

    class Store:
        def __init__(self):
            self.claimed, self.finished = [], []

        async def cleanup_candidates(self, now):
            return [bad, good]

        async def claim_deletion(self, expected, now, retained):
            self.claimed.append(expected.batch_id)
            return replace(expected, state="deleting", generation=uuid.uuid4().hex)

        async def finish_deletion(self, expected):
            self.finished.append(expected.batch_id)

    store = Store()
    assert await upload_recovery.cleanup_correlated_batches(store, tmp_path, 20.0, set()) == [good.batch_id]
    assert store.claimed == store.finished == [good.batch_id]
    assert bad_result.input_dir.exists() and not good_result.batch_root.exists()
    assert bad.batch_id in caplog.text
