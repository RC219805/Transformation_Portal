"""Real Postgres correlation, restart and upload ownership fencing evidence.

Use a dedicated migrated TP_UPLOAD_RECEIPT_TEST_DATABASE_URL ending in _test.
No Redis or model runtime is needed for this ledger lane.
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import delete, select, text

from transformation_portal.ingest import upload_recovery
from transformation_portal.ingest.upload_recovery import (
    UploadBatchLock,
    cleanup_correlated_batches,
    initialize_batch,
    seal_completion,
)
from transformation_portal.ingest.upload_staging import IncomingUpload, stage_upload_batch
from transformation_portal.orchestrator.models import JobModel, UploadJobReferenceModel
from transformation_portal.orchestrator.storage.base import JobRecord
from transformation_portal.orchestrator.storage.postgres import _SharedEngine
from transformation_portal.orchestrator.upload_receipts import (
    PostgresUploadReceiptStore,
    UploadOwnershipLost,
    UploadReceiptConflict,
    UploadReservation,
    bind_upload_references,
    upload_owner,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def store(tmp_path, monkeypatch):
    url = os.getenv("TP_UPLOAD_RECEIPT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("dedicated migrated Postgres upload receipt database required")
    if not url.rsplit("/", 1)[-1].endswith("_test"):
        pytest.fail("refusing to alter a database without _test suffix")
    item = PostgresUploadReceiptStore(database_url=url)
    async with item._session() as session:
        async with session.begin():
            await session.execute(
                text(
                    "TRUNCATE upload_job_references, upload_receipts, jobs, job_events, job_event_sequences, admission_capacity, dispatch_attempts, dispatch_plans, operational_records, operational_outbox, committed_generations RESTART IDENTITY CASCADE"
                )
            )
    monkeypatch.setenv("TP_PORTAL_UPLOAD_ROOT", str(tmp_path))
    yield item
    await _SharedEngine.dispose(url)


def reservation(root, *, request_id=None, owner="a" * 64, expiry=10.0):
    batch = "upload_" + uuid.uuid4().hex
    return UploadReservation(
        batch, "tenant_a", owner, request_id or str(uuid.uuid4()), uuid.uuid4().hex, str(root / batch), "staging", 1.0, expiry
    )


async def sealed(store, root, **kwargs):
    item = reservation(root, **kwargs)
    await store.reserve(item)
    initialize_batch(item)
    result = stage_upload_batch(
        upload_root=root,
        uploads=[IncomingUpload("source.txt", io.BytesIO(b"body"))],
        now=1.0,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    digest = seal_completion(item, result)
    return item, result, digest


async def test_concurrent_duplicate_attempts_reserve_one_batch_and_never_replay(store, tmp_path):
    request_id = str(uuid.uuid4())
    attempts = [reservation(tmp_path, request_id=request_id) for _ in range(2)]
    results = await asyncio.gather(*(store.reserve(item) for item in attempts), return_exceptions=True)
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, UploadReceiptConflict) for result in results) == 1
    await store.reserve(reservation(tmp_path, request_id=request_id, owner="b" * 64))


async def test_committed_reservation_is_visible_in_a_fresh_python_process(store, tmp_path):
    item = reservation(tmp_path)
    await store.reserve(item)
    command = """import asyncio, sys
from transformation_portal.orchestrator.upload_receipts import PostgresUploadReceiptStore
async def check():
 store=PostgresUploadReceiptStore(database_url=sys.argv[1])
 row=await store.get(sys.argv[2],sys.argv[3],sys.argv[4])
 print(row.batch_id if row else 'missing')
asyncio.run(check())
"""
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", command, store._database_url, item.tenant_id, item.owner_sha256, item.request_id],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == item.batch_id


async def test_live_filesystem_owner_blocks_ttl_even_after_deadline(store, tmp_path):
    item = reservation(tmp_path)
    lock = UploadBatchLock(tmp_path, item.batch_id)
    assert lock.acquire()
    await store.reserve(item)
    initialize_batch(item)
    try:
        assert await cleanup_correlated_batches(store, tmp_path, 20.0, set()) == []
        assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "staging"
    finally:
        lock.close()
    assert await cleanup_correlated_batches(store, tmp_path, 20.0, set()) == [item.batch_id]
    assert not Path(item.batch_root).exists()
    with pytest.raises(UploadOwnershipLost):
        await store.complete(item, "c" * 64, {})


async def test_failed_delete_stays_fenced_and_existing_cleanup_retries(store, tmp_path, monkeypatch):
    item, result, digest = await sealed(store, tmp_path)
    original = upload_recovery.delete_owned_batch
    monkeypatch.setattr(upload_recovery, "delete_owned_batch", lambda *args: (_ for _ in ()).throw(PermissionError("busy")))
    assert await cleanup_correlated_batches(store, tmp_path, 20.0, set()) == []
    current = await store.get(item.tenant_id, item.owner_sha256, item.request_id)
    assert current.state == "deleting" and current.generation != item.generation
    with pytest.raises(UploadOwnershipLost):
        await store.complete(item, digest, result.to_response_data())
    monkeypatch.setattr(upload_recovery, "delete_owned_batch", original)
    assert await cleanup_correlated_batches(store, tmp_path, 20.0, set()) == [item.batch_id]
    assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "expired"


async def test_retained_jobs_pin_inputs_and_keyless_reference_shares_expiry_lock(store, tmp_path):
    item, result, digest = await sealed(store, tmp_path)
    await store.complete(item, digest, result.to_response_data())
    job_id = "job_upload_" + uuid.uuid4().hex[:10]
    record = JobRecord(id=job_id, created_at=1.0, request={}, effective_request={"args": {"input_dir": str(result.input_dir)}})
    async with store._session() as session:
        async with session.begin():
            session.add(
                JobModel(
                    id=job_id,
                    created_at=1.0,
                    state="succeeded",
                    progress=100,
                    request={},
                    effective_request={},
                    logs_tail=[],
                    artifacts={},
                    run_summary={},
                )
            )
            await session.flush()
            await bind_upload_references(session, record, item.tenant_id, 2.0)
    assert await cleanup_correlated_batches(store, tmp_path, 20.0, set()) == []
    assert result.input_dir.exists()
    async with store._session() as session:
        async with session.begin():
            await session.execute(delete(JobModel).where(JobModel.id == job_id))
    assert await cleanup_correlated_batches(store, tmp_path, 20.0, set()) == [item.batch_id]
    with pytest.raises(UploadOwnershipLost):
        async with store._session() as session:
            async with session.begin():
                await bind_upload_references(session, record, item.tenant_id, 2.0)


async def test_get_recovers_seal_database_gap_but_rejects_other_owner_and_changed_bytes(store, tmp_path, monkeypatch):
    import app

    actor = {"accessEmail": "shared@example.test", "username": "alice", "role": "operator"}
    owner = upload_owner("tenant_a", actor)
    item, result, _ = await sealed(store, tmp_path, owner=owner)
    assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "staging"
    monkeypatch.setattr(app, "_upload_receipt_store", lambda: store)
    monkeypatch.setattr(app, "_verified_frontdoor_actor", lambda request: dict(actor))
    monkeypatch.setattr(app, "_resolved_portal_upload_root", lambda: tmp_path)
    monkeypatch.setattr(app, "_now", lambda: 2.0)
    monkeypatch.setattr(app, "API_KEY_SECRET", "upload-test-key")
    monkeypatch.setattr(app, "ENFORCE_JOB_API_KEY", True)
    app.RATE_LIMIT_BUCKETS.clear()

    async def tenant(request, **kwargs):
        return SimpleNamespace(tenant_id="tenant_a"), None

    async def paths(*args, **kwargs):
        return None

    monkeypatch.setattr(app, "_pilot_tenant_from_request", tenant)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", paths)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app.app), base_url="http://testserver", headers={"x-api-key": "upload-test-key"}
    ) as client:
        url = "/v1/uploads/staging/receipts/" + item.request_id
        response = await client.get(url)
        assert response.status_code == 200, response.text
        assert response.json()["data"]["status"] == "ready"
        assert response.headers["cache-control"] == "no-store"
        monkeypatch.setattr(app, "_now", lambda: 20.0)
        expired = await client.get(url)
        assert expired.json()["data"]["status"] == "expired"
        assert expired.json()["data"]["receipt"] is None
        assert result.input_dir.exists()  # Lookup never performs TTL deletion.
        monkeypatch.setattr(app, "_now", lambda: 2.0)
        actor["username"] = "bob"
        assert (await client.get(url)).json()["data"]["status"] == "unknown"
        actor["username"] = "alice"
        (result.input_dir / "source.txt").write_bytes(b"else")
        broken = await client.get(url)
        assert broken.json()["data"]["status"] == "unknown"
        assert broken.json()["data"]["receipt"] is None
        monkeypatch.setenv("TP_PORTAL_UPLOAD_RECEIPTS_ENABLED", "0")
        # Rollback keeps exact GET reads; write enablement is independent.
        assert (await client.get(url)).status_code == 200


async def test_correlated_post_reserves_before_parsing_and_duplicate_never_reparses(store, tmp_path, monkeypatch):
    from starlette.requests import Request

    import app

    request_id = str(uuid.uuid4())
    actor = {"accessEmail": "shared@example.test", "username": "alice"}
    owner = upload_owner("tenant_a", actor)
    parsed_count = 0
    streams = []

    async def context(request, **kwargs):
        return SimpleNamespace(tenant_id="tenant_a"), owner, store, None

    async def paths(*args, **kwargs):
        return None

    async def parse(request):
        nonlocal parsed_count
        row = await store.get("tenant_a", owner, request_id)
        assert row is not None and row.state == "staging"
        parsed_count += 1
        stream = io.BytesIO(b"body")
        streams.append(stream)
        return app._ParsedPortalUploadPayload([IncomingUpload("source.txt", stream)], None)

    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", paths)
    monkeypatch.setattr(app, "_parse_portal_upload_multipart", parse)

    def request():
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/uploads/staging",
                "headers": [(b"x-tp-upload-request-id", request_id.encode())],
            }
        )

    first = await app._stage_correlated_portal_upload(request(), tmp_path)
    assert first.status_code == 200, first.body
    assert streams[0].closed
    second = await app._stage_correlated_portal_upload(request(), tmp_path)
    assert second.status_code == 409, second.body
    assert parsed_count == 1
    assert (await store.get("tenant_a", owner, request_id)).state == "ready"


async def test_cancelled_http_owner_finishes_staging_and_ready_commit_before_closing_stream(store, tmp_path, monkeypatch):
    import threading

    from starlette.requests import Request

    import app

    entered, finish = threading.Event(), threading.Event()
    request_id, owner = str(uuid.uuid4()), "a" * 64
    stream = io.BytesIO(b"body")

    async def context(request, **kwargs):
        return SimpleNamespace(tenant_id="tenant_a"), owner, store, None

    async def paths(*args, **kwargs):
        return None

    async def parse(request):
        return app._ParsedPortalUploadPayload([IncomingUpload("source.txt", stream)], None)

    original = app.stage_upload_batch

    def work(**kwargs):
        entered.set()
        assert finish.wait(timeout=5)
        assert not stream.closed
        return original(**kwargs)

    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", paths)
    monkeypatch.setattr(app, "_parse_portal_upload_multipart", parse)
    monkeypatch.setattr(app, "stage_upload_batch", work)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/uploads/staging",
            "headers": [(b"x-tp-upload-request-id", request_id.encode())],
        }
    )
    task = asyncio.create_task(app._stage_correlated_portal_upload(request, tmp_path))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not stream.closed and not task.done()
    item = await store.get("tenant_a", owner, request_id)
    contender = UploadBatchLock(tmp_path, item.batch_id)
    try:
        assert not contender.acquire()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        assert contender.acquire()
        assert (await store.get("tenant_a", owner, request_id)).state == "ready"
        assert upload_recovery.verify_completion(item)[1]["input_dir"] == str(Path(item.batch_root) / "input")
    finally:
        finish.set()
        contender.close()


async def test_distributed_admission_and_legacy_postgres_creation_pin_tracked_uploads(store, tmp_path):
    import time

    from tests.core.test_execution_plan import _valid_payload
    from transformation_portal.core.execution_plan import CanonicalExecutionPlan
    from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
    from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository

    item, result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600)
    await store.complete(item, digest, result.to_response_data())
    plan = CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode()
    jobs = []
    for distributed in [True, False]:
        job_id = "job_upload_pin_" + uuid.uuid4().hex[:10]
        jobs.append(job_id)
        record = JobRecord(
            id=job_id,
            created_at=time.time(),
            request={"pipeline": "lux-depth-v3"},
            effective_request={
                "pipeline": "lux-depth-v3",
                "tenant_id": "tenant_a",
                "args": {"input_dir": str(result.input_dir)},
            },
        )
        if distributed:
            await PostgresOperationalRecordStore(database_url=store._database_url).admit(
                record,
                plan,
                tenant_id="tenant_a",
                output_root=str(tmp_path / "output"),
                requested_output_root=str(tmp_path / "output"),
                global_limit=8,
                tenant_limit=8,
            )
        else:
            await PostgresJobRepository(database_url=store._database_url).create(record)
    async with store._session() as session:
        references = list(
            (
                await session.scalars(
                    select(UploadJobReferenceModel.job_id).where(UploadJobReferenceModel.batch_id == item.batch_id)
                )
            ).all()
        )
    assert sorted(references) == sorted(jobs)


async def test_bounded_cleanup_cursor_does_not_starve_later_expired_batches(store, tmp_path):
    items = [reservation(tmp_path) for _ in range(3)]
    for item in items:
        await store.reserve(item)
    checked = []
    for now in [20.0, 21.0, 22.0]:
        candidates = await store.cleanup_candidates(now, limit=1)
        assert len(candidates) == 1
        checked.append(candidates[0].batch_id)
    assert set(checked) == {item.batch_id for item in items}


async def test_cleanup_recovers_crash_between_directory_creation_and_initial_marker(store, tmp_path):
    item = reservation(tmp_path)
    await store.reserve(item)
    root = Path(item.batch_root)
    root.mkdir()
    (root / ".initial-marker.tmp").write_bytes(b"interrupted")
    assert await cleanup_correlated_batches(store, tmp_path, 20.0, set()) == [item.batch_id]
    assert not root.exists()
    assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "expired"
