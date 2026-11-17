"""Real Postgres correlation, restart and upload ownership fencing evidence.

Use a dedicated migrated TP_UPLOAD_RECEIPT_TEST_DATABASE_URL ending in _test.
No Redis or model runtime is needed for this ledger lane.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, IntegrityError

from transformation_portal.ingest import upload_recovery
from transformation_portal.ingest.upload_recovery import (
    UploadBatchLock,
    cleanup_correlated_batches,
    initialize_batch,
    seal_completion,
)
from transformation_portal.ingest.upload_staging import IncomingUpload, stage_upload_batch
from transformation_portal.orchestrator.models import JobModel, UploadJobReferenceModel, UploadReceiptModel
from transformation_portal.orchestrator.storage.base import JobRecord
from transformation_portal.orchestrator.storage.postgres import _SharedEngine
from transformation_portal.orchestrator.upload_receipts import (
    PostgresUploadReceiptStore,
    UploadOwnershipLost,
    UploadReceiptConflict,
    UploadReservation,
    bind_upload_references,
    upload_owner,
    upload_source_paths,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _validate_upload_test_database_url(url):
    try:
        database = make_url(url).database
    except (ArgumentError, TypeError) as error:
        raise RuntimeError("refusing to alter a database without _test suffix") from error
    if not isinstance(database, str) or not database.endswith("_test"):
        raise RuntimeError("refusing to alter a database without _test suffix")


async def _clear_upload_tables(session):
    # The URL alone cannot prove where a routing proxy opened this transaction.
    database = await session.scalar(text("SELECT current_database()"))
    if not isinstance(database, str) or not database.endswith("_test"):
        raise RuntimeError("refusing to alter a database without _test suffix")
    await session.execute(
        text(
            "TRUNCATE upload_job_references, upload_receipts, jobs, job_events, job_event_sequences, admission_capacity, dispatch_attempts, dispatch_plans, operational_records, operational_outbox, committed_generations RESTART IDENTITY CASCADE"
        )
    )


@pytest.fixture
async def store(tmp_path, monkeypatch):
    url = os.getenv("TP_UPLOAD_RECEIPT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("dedicated migrated Postgres upload receipt database required")
    _validate_upload_test_database_url(url)
    item = PostgresUploadReceiptStore(database_url=url)
    try:
        async with item._session() as session:
            async with session.begin():
                await _clear_upload_tables(session)
        monkeypatch.setenv("TP_PORTAL_UPLOAD_ROOT", str(tmp_path))
        yield item
    finally:
        await _SharedEngine.dispose(url)


def reservation(root, *, request_id=None, owner="a" * 64, expiry=None, batch_id=None):
    batch = batch_id or "upload_" + uuid.uuid4().hex
    now = time.time()
    return UploadReservation(
        batch,
        "tenant_a",
        owner,
        request_id or str(uuid.uuid4()),
        uuid.uuid4().hex,
        str(root / batch),
        "staging",
        now,
        now + 3600 if expiry is None else expiry,
    )


async def sealed(store, root, *, filename="source.txt", **kwargs):
    item = reservation(root, **kwargs)
    await store.reserve(item)
    initialize_batch(item)
    result = stage_upload_batch(
        upload_root=root,
        uploads=[IncomingUpload(filename, io.BytesIO(b"body"))],
        now=item.created_at,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    digest = seal_completion(item, result)
    return item, result, digest


async def _expire_receipt(store, item):
    async with store._session() as session:
        async with session.begin():
            row = await session.get(UploadReceiptModel, item.batch_id, with_for_update=True)
            row.expires_at = float(await session.scalar(text("SELECT extract(epoch FROM clock_timestamp())"))) - 1


async def _expire_while_operation_waits_on_upload_row(store, item, operation):
    """Observe a real Postgres lock wait, then cross TTL before releasing it."""
    task = None
    try:
        async with store._session() as session:
            async with session.begin():
                row = await session.get(UploadReceiptModel, item.batch_id, with_for_update=True)
                task = asyncio.create_task(operation())

                async def wait_for_lock():
                    while True:
                        await session.execute(text("SELECT pg_stat_clear_snapshot()"))
                        blocked = await session.scalar(
                            text(
                                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                                "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                                "AND wait_event_type = 'Lock' AND query LIKE '%upload_receipts%')"
                            )
                        )
                        if blocked:
                            return
                        if task.done():
                            await task
                            pytest.fail("upload operation did not wait for its row lock")
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_for_lock(), 5)
                row.expires_at = float(await session.scalar(text("SELECT extract(epoch FROM clock_timestamp())")))
        return await asyncio.wait_for(task, 5)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_concurrent_duplicate_attempts_reserve_one_batch_and_never_replay(store, tmp_path):
    request_id = str(uuid.uuid4())
    attempts = [reservation(tmp_path, request_id=request_id) for _ in range(2)]
    results = await asyncio.gather(*(store.reserve(item) for item in attempts), return_exceptions=True)
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, UploadReceiptConflict) for result in results) == 1
    await store.reserve(reservation(tmp_path, request_id=request_id, owner="b" * 64))


async def test_unrelated_batch_primary_key_collision_is_not_a_request_identity_conflict(store, tmp_path):
    first = reservation(tmp_path)
    second = reservation(tmp_path, batch_id=first.batch_id)
    await store.reserve(first)
    with pytest.raises(IntegrityError) as error:
        await store.reserve(second)
    assert error.value.orig.sqlstate == "23505"
    assert await store.get(first.tenant_id, first.owner_sha256, first.request_id) == first
    assert await store.get(second.tenant_id, second.owner_sha256, second.request_id) is None
    with pytest.raises(UploadReceiptConflict):
        await store.reserve(reservation(tmp_path, request_id=first.request_id))


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
        assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == []
        assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "staging"
    finally:
        lock.close()
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == [item.batch_id]
    assert not Path(item.batch_root).exists()
    with pytest.raises(UploadOwnershipLost):
        await store.complete(item, "c" * 64, {})


async def test_failed_delete_stays_fenced_and_existing_cleanup_retries(store, tmp_path, monkeypatch):
    item, result, digest = await sealed(store, tmp_path)
    original = upload_recovery.delete_owned_batch
    monkeypatch.setattr(upload_recovery, "delete_owned_batch", lambda *args: (_ for _ in ()).throw(PermissionError("busy")))
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == []
    current = await store.get(item.tenant_id, item.owner_sha256, item.request_id)
    assert current.state == "deleting" and current.generation != item.generation
    with pytest.raises(UploadOwnershipLost):
        await store.complete(item, digest, result.to_response_data())
    monkeypatch.setattr(upload_recovery, "delete_owned_batch", original)
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == [item.batch_id]
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
            await bind_upload_references(session, record, item.tenant_id)
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == []
    assert result.input_dir.exists()
    async with store._session() as session:
        async with session.begin():
            await session.execute(delete(JobModel).where(JobModel.id == job_id))
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == [item.batch_id]
    with pytest.raises(UploadOwnershipLost):
        async with store._session() as session:
            async with session.begin():
                await bind_upload_references(session, record, item.tenant_id)


async def test_all_input_and_archive_aliases_pin_distinct_batches_atomically(store, tmp_path):
    first, first_result, first_digest = await sealed(store, tmp_path, filename="nested/source.txt")
    second, second_result, second_digest = await sealed(store, tmp_path)
    await store.complete(first, first_digest, first_result.to_response_data())
    await store.complete(second, second_digest, second_result.to_response_data())
    job_id = "job_multi_upload_" + uuid.uuid4().hex[:10]
    record = JobRecord(
        id=job_id,
        created_at=1.0,
        request={"args": {"archiveRoot": str(second_result.input_dir)}},
        effective_request={
            "args": {
                "input_dir": str(first_result.input_dir / "nested"),
                "inputDir": str(first_result.input_dir),
                "archive_root": str(second_result.input_dir),
            }
        },
    )
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
            await bind_upload_references(session, record, first.tenant_id)
    async with store._session() as session:
        references = (
            await session.scalars(select(UploadJobReferenceModel).where(UploadJobReferenceModel.job_id == job_id))
        ).all()
    assert {row.batch_id for row in references} == {first.batch_id, second.batch_id}
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == []
    assert first_result.input_dir.exists() and second_result.input_dir.exists()


async def test_expired_secondary_archive_batch_rolls_back_every_new_reference(store, tmp_path):
    first, first_result, first_digest = await sealed(store, tmp_path, batch_id="upload_aaa_valid")
    second, second_result, second_digest = await sealed(store, tmp_path, batch_id="upload_zzz_expired")
    await store.complete(first, first_digest, first_result.to_response_data())
    await store.complete(second, second_digest, second_result.to_response_data())
    await _expire_receipt(store, second)
    job_id = "job_expired_archive_" + uuid.uuid4().hex[:10]
    record = JobRecord(
        id=job_id,
        created_at=1.0,
        request={},
        effective_request={"args": {"input_dir": str(first_result.input_dir), "archiveRoot": str(second_result.input_dir)}},
    )
    with pytest.raises(UploadOwnershipLost):
        async with store._session() as session:
            async with session.begin():
                session.add(
                    JobModel(
                        id=job_id,
                        created_at=1.0,
                        state="queued",
                        progress=0,
                        request={},
                        effective_request={},
                        logs_tail=[],
                        artifacts={},
                        run_summary={},
                    )
                )
                await session.flush()
                await bind_upload_references(session, record, first.tenant_id)
    async with store._session() as session:
        assert await session.get(JobModel, job_id) is None
        assert not (
            await session.scalars(select(UploadJobReferenceModel).where(UploadJobReferenceModel.job_id == job_id))
        ).all()


async def test_retained_nested_archive_path_protects_ledger_batch_without_new_reference(store, tmp_path):
    item, result, digest = await sealed(store, tmp_path, filename="nested/source.txt")
    await store.complete(item, digest, result.to_response_data())
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, {str(result.input_dir / "nested")}) == []
    assert result.input_dir.exists()


async def test_manifest_in_tracked_batch_metadata_is_pinned_by_legacy_postgres_creation(store, tmp_path):
    import time

    from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository

    item, result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600)
    await store.complete(item, digest, result.to_response_data())
    job_id = "job_metadata_read_" + uuid.uuid4().hex[:10]
    request = {
        "pipeline": "lux-depth-v5",
        "tenant_id": "tenant_a",
        "args": {"companions_manifest": str(result.baseline_manifest_path)},
    }
    await PostgresJobRepository(database_url=store._database_url).create(
        JobRecord(id=job_id, created_at=time.time(), request=request, effective_request=request)
    )
    async with store._session() as session:
        assert (
            await session.scalars(select(UploadJobReferenceModel.batch_id).where(UploadJobReferenceModel.job_id == job_id))
        ).all() == [item.batch_id]


@pytest.mark.parametrize("tamper,expected_status", [("extra_input", 503), ("oversize_metadata", 413), ("oversize_seal", 413)])
async def test_post_never_commits_ready_before_exact_recoverable_verification(
    store, tmp_path, monkeypatch, tamper, expected_status
):
    from starlette.requests import Request

    import app

    request_id, owner = str(uuid.uuid4()), "a" * 64
    stream = io.BytesIO(b"body")

    async def context(request, **kwargs):
        return SimpleNamespace(tenant_id="tenant_a"), owner, store, None

    async def paths(*args, **kwargs):
        return None

    async def parse(request):
        return app._ParsedPortalUploadPayload([IncomingUpload("source.txt", stream)], None)

    original = app.stage_upload_batch

    def stage(**kwargs):
        result = original(**kwargs)
        if tamper == "extra_input":
            (result.input_dir / "extra.txt").write_bytes(b"unsealed")
        elif tamper == "oversize_metadata":
            with result.capture_metadata_path.open("wb") as output:
                output.truncate(upload_recovery.MAX_RECOVERY_METADATA_BYTES + 1)
        else:
            return replace(result, warnings=("x" * upload_recovery.MAX_RECOVERY_METADATA_BYTES,))
        return result

    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", paths)
    monkeypatch.setattr(app, "_parse_portal_upload_multipart", parse)
    monkeypatch.setattr(app, "stage_upload_batch", stage)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/uploads/staging",
            "headers": [(b"x-tp-upload-request-id", request_id.encode())],
        }
    )
    response = await app._stage_correlated_portal_upload(request, tmp_path)
    assert response.status_code == expected_status
    row = await store.get("tenant_a", owner, request_id)
    assert row.state == "staging" and row.receipt_sha256 is None and row.response_data is None
    if expected_status == 413:
        assert json.loads(response.body)["error"]["details"]["reason"] == "upload_recovery_metadata_limit"
        assert not (Path(row.batch_root) / "_portal" / upload_recovery.SEAL_FILENAME).exists()
    assert stream.closed


async def test_get_recovers_seal_database_gap_but_rejects_other_owner_and_changed_bytes(store, tmp_path, monkeypatch):
    import app

    actor = {"accessEmail": "shared@example.test", "username": "alice", "role": "operator"}
    owner = upload_owner("tenant_a", actor)
    item, result, _ = await sealed(store, tmp_path, owner=owner)
    assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "staging"
    monkeypatch.setattr(app, "_upload_receipt_store", lambda: store)
    monkeypatch.setattr(app, "_verified_frontdoor_actor", lambda request: dict(actor))
    monkeypatch.setattr(app, "_resolved_portal_upload_root", lambda: tmp_path)
    monkeypatch.setattr(app, "_now", time.time)
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
        monkeypatch.setattr(app, "_now", lambda: item.expires_at + 1)
        expired = await client.get(url)
        assert expired.json()["data"]["status"] == "expired"
        assert expired.json()["data"]["receipt"] is None
        assert result.input_dir.exists()  # Lookup never performs TTL deletion.
        monkeypatch.setattr(app, "_now", time.time)
        actor["username"] = "bob"
        assert (await client.get(url)).json()["data"]["status"] == "unknown"
        actor["username"] = "alice"
        extra = result.input_dir / "unlisted.txt"
        extra.write_bytes(b"unsealed")
        broken = await client.get(url)
        assert broken.json()["data"]["status"] == "unknown"
        assert broken.json()["data"]["receipt"] is None
        extra.unlink()
        assert (await client.get(url)).json()["data"]["status"] == "ready"
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


@pytest.mark.parametrize("distributed", [True, False])
async def test_tracked_upload_without_tenant_scope_rolls_back_all_admission_effects(store, tmp_path, distributed):
    import time

    from tests.core.test_execution_plan import _valid_payload
    from transformation_portal.core.execution_plan import CanonicalExecutionPlan
    from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
    from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository

    item, result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600)
    await store.complete(item, digest, result.to_response_data())
    job_id = "job_unscoped_" + uuid.uuid4().hex[:10]
    record = JobRecord(
        id=job_id,
        created_at=time.time(),
        request={"pipeline": "lux-depth-v3"},
        effective_request={"pipeline": "lux-depth-v3", "tenant_id": "", "args": {"input_dir": str(result.input_dir)}},
    )
    expected_error = ValueError if distributed else UploadOwnershipLost
    expected_message = "invalid locator tenant_id" if distributed else "outside this tenant"
    with pytest.raises(expected_error, match=expected_message):
        if distributed:
            plan = CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode()
            await PostgresOperationalRecordStore(database_url=store._database_url).admit(
                record,
                plan,
                tenant_id="",
                output_root=str(tmp_path / "output"),
                requested_output_root=str(tmp_path / "output"),
                global_limit=8,
                tenant_limit=8,
            )
        else:
            await PostgresJobRepository(database_url=store._database_url).create(record)
    async with store._session() as session:
        for table in (
            "jobs",
            "upload_job_references",
            "admission_capacity",
            "dispatch_plans",
            "dispatch_attempts",
            "operational_records",
            "operational_outbox",
        ):
            assert await session.scalar(text(f"SELECT count(*) FROM {table}")) == 0, table
    assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "ready"
    assert result.input_dir.exists()


async def test_legacy_keyless_unscoped_job_still_accepts_a_preledger_staged_batch(store, tmp_path):
    from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository

    result = stage_upload_batch(upload_root=tmp_path, uploads=[IncomingUpload("source.txt", io.BytesIO(b"body"))], now=1.0)
    record = JobRecord(
        id="job_preledger_" + uuid.uuid4().hex[:10],
        created_at=1.0,
        request={"pipeline": "lux-depth-v3"},
        effective_request={"pipeline": "lux-depth-v3", "args": {"input_dir": str(result.input_dir)}},
    )
    await PostgresJobRepository(database_url=store._database_url).create(record)
    async with store._session() as session:
        assert await session.get(JobModel, record.id) is not None
        assert await session.scalar(text("SELECT count(*) FROM upload_receipts")) == 0
        assert await session.scalar(text("SELECT count(*) FROM upload_job_references")) == 0


async def test_archive_derived_read_default_pins_batch_in_both_postgres_admission_paths(store, tmp_path, monkeypatch):
    import time

    import app
    from transformation_portal.orchestrator.execution_dispatch import archive_command_read_paths, prepare_dispatch_plan
    from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
    from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository

    item, result, digest = await sealed(store, tmp_path, filename="hash_manifest.csv.gz", expiry=time.time() + 3600)
    await store.complete(item, digest, result.to_response_data())
    source = tmp_path / "untracked-input"
    source.mkdir()
    for name in ("ALLOWED_INPUT_ROOTS", "ALLOWED_OUTPUT_ROOTS", "ALLOWED_PATH_ROOTS"):
        monkeypatch.setattr(app, name, [tmp_path])
    args = {"input_dir": str(source), "output_dir": str(result.input_dir), "archive_command": "fixity-verify"}
    request = {"pipeline": "archive-gate-a", "args": args, "tenant_id": "tenant_a"}
    # The request contains no hash_manifest selector. The server command's
    # existing default points into another tracked batch, which must be pinned.
    assert not upload_source_paths(request) or all(
        not path.is_relative_to(result.batch_root) for path in upload_source_paths(request)
    )
    command = app._archive_gate_argv("archive-gate-a", args, str(source), str(result.input_dir))
    reads = archive_command_read_paths("archive-gate-a", command)
    assert str(result.input_dir / "hash_manifest.csv.gz") in reads
    plan = prepare_dispatch_plan(request, trusted_argv=command)
    jobs = []
    for distributed in [True, False]:
        job_id = "job_default_archive_" + uuid.uuid4().hex[:10]
        record = JobRecord(id=job_id, created_at=time.time(), request=request, effective_request=request)
        if distributed:
            jobs.append(job_id)
            await PostgresOperationalRecordStore(database_url=store._database_url).admit(
                record,
                plan,
                tenant_id="tenant_a",
                output_root=str(result.input_dir),
                requested_output_root=str(result.input_dir),
                global_limit=8,
                tenant_limit=8,
            )
        else:

            async def preview(*args, **kwargs):
                return {"pipeline": "archive-gate-a", "execution_args": request["args"], "readiness": {}, "field_errors": []}

            async def no_error(*args, **kwargs):
                return None

            async def no_active(*args, **kwargs):
                return 0

            monkeypatch.setattr(app, "_build_config_preview_threaded", preview)
            monkeypatch.setattr(app, "_pilot_enforce_pipeline", no_error)
            monkeypatch.setattr(app, "_enforce_job_readiness_preflight", lambda *args: None)
            monkeypatch.setattr(app, "_enforce_dispatch_value_preflight", lambda *args: None)
            monkeypatch.setattr(
                app, "_enforce_dispatch_filesystem_preflight", lambda *args: SimpleNamespace(output_dir=result.input_dir)
            )
            monkeypatch.setattr(app, "_pilot_enforce_dispatch_filesystem_tenant", lambda *args, **kwargs: None)
            monkeypatch.setattr(app, "_uses_dispatch_authority", lambda: False)
            monkeypatch.setattr(app, "_job_repository", lambda: PostgresJobRepository(database_url=store._database_url))
            monkeypatch.setattr(app, "_active_job_count", no_active)
            monkeypatch.setattr(app, "_pilot_active_job_count_for_tenant", no_active)
            monkeypatch.setattr(app, "_cleanup_expired_jobs", no_error)
            monkeypatch.setattr(app, "_record_pilot_audit", no_error)
            monkeypatch.setattr(app, "_materialize_dispatch_output_dir", lambda *args: None)
            monkeypatch.setattr(app, "_cache_runtime_job", lambda *args: None)
            monkeypatch.setattr(app, "_dispatch_job", no_error)
            response = await app._create_job(
                {"pipeline": "archive-gate-a", "args": args}, pilot_tenant=SimpleNamespace(tenant_id="tenant_a")
            )
            assert response.status_code == 200, response.body
            jobs.append(json.loads(response.body)["data"]["id"])
    async with store._session() as session:
        references = (
            await session.scalars(
                select(UploadJobReferenceModel.job_id).where(UploadJobReferenceModel.batch_id == item.batch_id)
            )
        ).all()
    assert set(references) == set(jobs)


async def test_bounded_cleanup_cursor_does_not_starve_later_expired_batches(store, tmp_path):
    items = [reservation(tmp_path) for _ in range(3)]
    for item in items:
        await store.reserve(item)
    checked = []
    expired = max(item.expires_at for item in items) + 1
    for now in [expired, expired + 1, expired + 2]:
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
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == [item.batch_id]
    assert not root.exists()
    assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "expired"


async def _create_upload_read_job(store, root, source, *, tenant_id="tenant_a", distributed=False):
    import time

    from tests.core.test_execution_plan import _valid_payload
    from transformation_portal.core.execution_plan import CanonicalExecutionPlan
    from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
    from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository

    request = {"pipeline": "lux-depth-v3", "tenant_id": tenant_id, "args": {"input_dir": str(source)}}
    record = JobRecord(
        id="job_overlap_" + uuid.uuid4().hex[:10], created_at=time.time(), request=request, effective_request=request
    )
    if distributed:
        plan = CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode()
        await PostgresOperationalRecordStore(database_url=store._database_url).admit(
            record,
            plan,
            tenant_id=tenant_id,
            output_root=str(root / "output"),
            requested_output_root=str(root / "output"),
            global_limit=8,
            tenant_limit=8,
        )
    else:
        await PostgresJobRepository(database_url=store._database_url).create(record)
    return record


@pytest.mark.parametrize("distributed", [False, True])
async def test_relative_upload_root_uses_reserved_absolute_identity_after_cwd_changes(
    store, tmp_path, monkeypatch, distributed
):
    import time

    import app

    repo, cwd = tmp_path / "repo", tmp_path / "cwd"
    repo.mkdir()
    cwd.mkdir()
    root = repo / "relative-uploads"
    root.mkdir()
    monkeypatch.setattr(app, "REPO_ROOT", repo)
    monkeypatch.setattr(app, "ALLOWED_INPUT_ROOTS", [repo])
    monkeypatch.setattr(app, "PORTAL_UPLOAD_ROOT", Path("relative-uploads"))
    monkeypatch.setenv("TP_PORTAL_UPLOAD_ROOT", "relative-uploads")
    monkeypatch.chdir(cwd)
    assert app._resolved_portal_upload_root() == root
    item, result, digest = await sealed(store, root, expiry=time.time() + 3600)
    await store.complete(item, digest, result.to_response_data())
    record = await _create_upload_read_job(store, tmp_path, result.input_dir, distributed=distributed)
    async with store._session() as session:
        assert (
            await session.scalars(select(UploadJobReferenceModel.batch_id).where(UploadJobReferenceModel.job_id == record.id))
        ).all() == [item.batch_id]
    assert await cleanup_correlated_batches(store, root, time.time() + 7200, set()) == []
    assert result.input_dir.exists()


@pytest.mark.parametrize("ancestor", [False, True])
@pytest.mark.parametrize("distributed", [False, True])
async def test_recursive_root_and_ancestor_reads_pin_every_live_batch(store, tmp_path, ancestor, distributed):
    import time

    first, first_result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600, batch_id="upload_aaa")
    await store.complete(first, digest, first_result.to_response_data())
    second, second_result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600, batch_id="upload_zzz")
    await store.complete(second, digest, second_result.to_response_data())
    record = await _create_upload_read_job(store, tmp_path, tmp_path.parent if ancestor else tmp_path, distributed=distributed)
    async with store._session() as session:
        references = (
            await session.scalars(select(UploadJobReferenceModel.batch_id).where(UploadJobReferenceModel.job_id == record.id))
        ).all()
    assert set(references) == {first.batch_id, second.batch_id}
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == []


@pytest.mark.parametrize("invalid", ["foreign", "staging", "deleting", "missing_tenant"])
async def test_recursive_read_rejects_any_unavailable_or_unscoped_batch_atomically(store, tmp_path, invalid):
    import time

    first, result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600, batch_id="upload_aaa_valid")
    await store.complete(first, digest, result.to_response_data())
    second, result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600, batch_id="upload_zzz_invalid")
    if invalid != "staging":
        await store.complete(second, digest, result.to_response_data())
    if invalid == "foreign":
        async with store._session() as session:
            async with session.begin():
                (await session.get(UploadReceiptModel, second.batch_id)).tenant_id = "tenant_b"
    elif invalid == "deleting":
        assert await store.claim_deletion(second, time.time() + 7200, set()) is not None
    with pytest.raises(UploadOwnershipLost, match="outside this tenant"):
        await _create_upload_read_job(store, tmp_path, tmp_path, tenant_id="" if invalid == "missing_tenant" else "tenant_a")
    async with store._session() as session:
        for table in (
            "jobs",
            "upload_job_references",
            "admission_capacity",
            "dispatch_attempts",
            "operational_records",
            "operational_outbox",
        ):
            assert await session.scalar(text(f"SELECT count(*) FROM {table}")) == 0, table


async def test_ancestor_read_ignores_deleted_tombstone_but_explicit_expired_descendant_still_rejects(store, tmp_path):
    import time

    retired, old_result, digest = await sealed(store, tmp_path)
    await store.complete(retired, digest, old_result.to_response_data())
    await _expire_receipt(store, retired)
    assert await cleanup_correlated_batches(store, tmp_path, time.time(), set()) == [retired.batch_id]
    assert not old_result.batch_root.exists()
    live, result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600)
    await store.complete(live, digest, result.to_response_data())
    record = await _create_upload_read_job(store, tmp_path, tmp_path)
    async with store._session() as session:
        assert (
            await session.scalars(select(UploadJobReferenceModel.batch_id).where(UploadJobReferenceModel.job_id == record.id))
        ).all() == [live.batch_id]
    with pytest.raises(UploadOwnershipLost):
        await _create_upload_read_job(store, tmp_path, old_result.input_dir)
    async with store._session() as session:
        assert await session.scalar(text("SELECT count(*) FROM jobs")) == 1


@pytest.mark.parametrize("name,sibling_name", [("literal_%", "literal_ab"), ("uploads", "uploads-other")])
async def test_ledger_overlap_is_segment_and_literal_safe_for_ancestor_selectors(store, tmp_path, name, sibling_name):
    import time

    root, sibling = tmp_path / name, tmp_path / sibling_name
    root.mkdir()
    sibling.mkdir()
    item, result, digest = await sealed(store, root, expiry=time.time() + 3600)
    await store.complete(item, digest, result.to_response_data())
    foreign, other_result, digest = await sealed(store, sibling, expiry=time.time() + 3600)
    await store.complete(foreign, digest, other_result.to_response_data())
    async with store._session() as session:
        async with session.begin():
            (await session.get(UploadReceiptModel, foreign.batch_id)).tenant_id = "tenant_b"
    record = await _create_upload_read_job(store, tmp_path, root)
    async with store._session() as session:
        assert (
            await session.scalars(select(UploadJobReferenceModel.batch_id).where(UploadJobReferenceModel.job_id == record.id))
        ).all() == [item.batch_id]


@pytest.mark.parametrize("retained", ["root", "ancestor", "sibling"])
async def test_retained_directory_overlap_protects_ledger_batches_without_pinning_siblings(store, tmp_path, retained):
    item, result, digest = await sealed(store, tmp_path)
    await store.complete(item, digest, result.to_response_data())
    selector = {"root": tmp_path, "ancestor": tmp_path.parent, "sibling": tmp_path / "unrelated"}[retained]
    removed = await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, {str(selector)})
    assert removed == ([item.batch_id] if retained == "sibling" else [])
    assert result.input_dir.exists() is (retained != "sibling")


async def test_partial_completed_batch_deletion_retries_after_marker_and_seal_are_removed(store, tmp_path, monkeypatch):
    item, result, digest = await sealed(store, tmp_path)
    await store.complete(item, digest, result.to_response_data())
    original = upload_recovery.shutil.rmtree
    interrupted = []

    def partial(path, *args, **kwargs):
        if Path(path) == result.batch_root and not interrupted:
            interrupted.append(True)
            (result.batch_root / upload_recovery.MARKER_FILENAME).unlink()
            (result.portal_dir / upload_recovery.SEAL_FILENAME).unlink()
            raise PermissionError("interrupted after ownership-file removal")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(upload_recovery.shutil, "rmtree", partial)
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7200, set()) == []
    fenced = await store.get(item.tenant_id, item.owner_sha256, item.request_id)
    assert fenced.state == "deleting" and fenced.generation != item.generation
    assert result.input_dir.exists()
    with pytest.raises(UploadOwnershipLost):
        await store.complete(item, digest, result.to_response_data())
    with pytest.raises(UploadOwnershipLost):
        await _create_upload_read_job(store, tmp_path, result.input_dir)
    assert await cleanup_correlated_batches(store, tmp_path, time.time() + 7201, set()) == [item.batch_id]
    assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "expired"
    assert not result.batch_root.exists()


@pytest.mark.parametrize("distributed", [False, True])
async def test_filesystem_root_ancestor_read_pins_the_live_ledger_batch(store, tmp_path, distributed):
    import time

    item, result, digest = await sealed(store, tmp_path, expiry=time.time() + 3600)
    await store.complete(item, digest, result.to_response_data())
    record = await _create_upload_read_job(store, tmp_path, Path(tmp_path.anchor), distributed=distributed)
    async with store._session() as session:
        assert (
            await session.scalars(select(UploadJobReferenceModel.batch_id).where(UploadJobReferenceModel.job_id == record.id))
        ).all() == [item.batch_id]


@pytest.mark.parametrize("state", ["staging", "ready"])
async def test_completion_checks_current_expiry_after_waiting_for_its_row_lock(store, tmp_path, state):
    from transformation_portal.orchestrator.upload_receipts import UploadReceiptExpired

    item, result, digest = await sealed(store, tmp_path)
    if state == "ready":
        await store.complete(item, digest, result.to_response_data())
    before = await store.get(item.tenant_id, item.owner_sha256, item.request_id)
    with pytest.raises(UploadReceiptExpired):
        await _expire_while_operation_waits_on_upload_row(
            store, item, lambda: store.complete(item, digest, result.to_response_data())
        )
    after = await store.get(item.tenant_id, item.owner_sha256, item.request_id)
    assert after.state == before.state
    assert after.generation == before.generation
    assert after.receipt_sha256 == before.receipt_sha256 and after.response_data == before.response_data


@pytest.mark.parametrize("distributed", [False, True])
async def test_admission_checks_current_expiry_after_waiting_for_all_upload_row_locks(store, tmp_path, distributed):
    item, result, digest = await sealed(store, tmp_path)
    await store.complete(item, digest, result.to_response_data())
    with pytest.raises(UploadOwnershipLost, match="expired"):
        await _expire_while_operation_waits_on_upload_row(
            store,
            item,
            lambda: _create_upload_read_job(store, tmp_path, result.input_dir, distributed=distributed),
        )
    async with store._session() as session:
        for table in (
            "jobs",
            "upload_job_references",
            "admission_capacity",
            "dispatch_plans",
            "dispatch_attempts",
            "operational_records",
            "operational_outbox",
        ):
            assert await session.scalar(text(f"SELECT count(*) FROM {table}")) == 0, table
    assert (await store.get(item.tenant_id, item.owner_sha256, item.request_id)).state == "ready"
    assert result.input_dir.exists()


async def test_accepted_submission_replay_survives_upload_expiry_without_new_admission(store, tmp_path):
    from tests.core.test_execution_plan import _valid_payload
    from transformation_portal.core.execution_plan import CanonicalExecutionPlan
    from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
    from transformation_portal.orchestrator.submissions import SubmissionIdentity, submission_intent

    item, result, digest = await sealed(store, tmp_path)
    await store.complete(item, digest, result.to_response_data())
    request = {"pipeline": "lux-depth-v3", "tenant_id": item.tenant_id, "args": {"input_dir": str(result.input_dir)}}
    record = JobRecord(
        id="job_replay_" + uuid.uuid4().hex[:10], created_at=time.time(), request=request, effective_request=request
    )
    submission = SubmissionIdentity(str(uuid.uuid4()), "a" * 64, submission_intent(request))
    operational = PostgresOperationalRecordStore(database_url=store._database_url)
    plan = CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode()
    options = {
        "tenant_id": item.tenant_id,
        "output_root": str(tmp_path / "output"),
        "requested_output_root": str(tmp_path / "output"),
        "global_limit": 1,
        "tenant_limit": 1,
        "submission": submission,
    }
    accepted = await operational.admit(record, plan, **options)
    await _expire_receipt(store, item)
    replay = await operational.admit(replace(record, id="job_not_created"), plan, **options)
    assert replay == accepted
    async with store._session() as session:
        for table in ("jobs", "upload_job_references", "dispatch_attempts", "operational_outbox"):
            assert await session.scalar(text(f"SELECT count(*) FROM {table}")) == 1, table
