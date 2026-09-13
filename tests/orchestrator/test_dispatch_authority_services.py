"""Real PostgreSQL/Redis failure-mode evidence for distributed authority.

Set TP_DISPATCH_TEST_DATABASE_URL to a dedicated migrated database and
TP_DISPATCH_TEST_REDIS_URL. These tests truncate only the dedicated database.
"""

from __future__ import annotations

import asyncio
import hashlib
import multiprocessing
import os
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from tests.core.test_execution_plan import _valid_payload
from transformation_portal.core.execution_plan import CanonicalExecutionPlan
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator
from transformation_portal.orchestrator.queue.base import JobEnqueueRequest, JobLease, LeaseNotHeldError, QueueBrokerError
from transformation_portal.orchestrator.queue.locator import RedisLocatorQueueBroker
from transformation_portal.orchestrator.storage.base import JobRecord
from transformation_portal.orchestrator.storage.operational import (
    AdmissionRejected,
    DispatchAuthorityLost,
    PostgresOperationalRecordStore,
)
from transformation_portal.orchestrator.storage.postgres import _SharedEngine

pytestmark = pytest.mark.integration
DB_ENV = "TP_DISPATCH_TEST_DATABASE_URL"
REDIS_ENV = "TP_DISPATCH_TEST_REDIS_URL"


def _plan() -> bytes:
    return CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode("utf-8")


def _record(job_id: str, tenant: str = "tenant_a") -> JobRecord:
    return JobRecord(
        id=job_id,
        created_at=time.time(),
        request={"pipeline": "lux-depth-v3"},
        effective_request={"pipeline": "lux-depth-v3", "tenant_id": tenant, "args": {}},
    )


@pytest.fixture
def service_urls(tmp_path, monkeypatch):
    database_url = os.getenv(DB_ENV)
    redis_url = os.getenv(REDIS_ENV)
    if not database_url or not redis_url:
        pytest.skip("dedicated migrated Postgres and Redis are required")
    database_name = database_url.rsplit("/", 1)[-1]
    if not (database_name.endswith("_fencing") or database_name.endswith("_test")):
        pytest.fail("refusing to truncate a database without the _fencing or _test suffix")
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution-storage"))

    async def clear():
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        engine = create_async_engine(database_url, poolclass=NullPool)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "TRUNCATE jobs, job_events, job_event_sequences, admission_capacity, dispatch_attempts, dispatch_plans, operational_records, operational_outbox, committed_generations RESTART IDENTITY CASCADE"
                    )
                )
        finally:
            await engine.dispose()

    asyncio.run(clear())
    yield database_url, redis_url
    asyncio.run(clear())


async def _admit(store, job_id, root, *, tenant="tenant_a", global_limit=4, tenant_limit=2):
    return await store.admit(
        _record(job_id, tenant),
        _plan(),
        tenant_id=tenant,
        output_root=str(root / ".tp-attempts" / (job_id + "-" + uuid.uuid4().hex)),
        requested_output_root=str(root),
        global_limit=global_limit,
        tenant_limit=tenant_limit,
    )


def _admit_process(database_url, root, tenant, ready, start, result):
    ready.put(True)
    start.wait(15)

    async def run():
        store = PostgresOperationalRecordStore(database_url=database_url)
        try:
            locator = await _admit(store, "job_" + uuid.uuid4().hex, Path(root), tenant=tenant, global_limit=3, tenant_limit=2)
            result.put(("accepted", locator.tenant_id))
        except AdmissionRejected as exc:
            result.put(("rejected", exc.scope))
        except Exception as exc:
            result.put(("error", repr(exc)))
        finally:
            await _SharedEngine.dispose(database_url)

    asyncio.run(run())


def test_independent_processes_cannot_oversubscribe_global_or_tenant_capacity(service_urls, tmp_path):
    database_url, _ = service_urls
    context = multiprocessing.get_context("spawn")
    ready, result, start = context.Queue(), context.Queue(), context.Event()
    processes = [
        context.Process(target=_admit_process, args=(database_url, str(tmp_path), f"tenant_{i % 2}", ready, start, result))
        for i in range(8)
    ]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            ready.get(timeout=30)
        start.set()
        outcomes = [result.get(timeout=30) for _ in processes]
        assert not [outcome for outcome in outcomes if outcome[0] == "error"], outcomes
        accepted = [tenant for status, tenant in outcomes if status == "accepted"]
        assert len(accepted) == 3
        assert max(accepted.count(tenant) for tenant in set(accepted)) <= 2
    finally:
        start.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for channel in (ready, result):
            channel.close()
            channel.join_thread()


@pytest.mark.asyncio
async def test_duplicate_locator_claims_only_once_and_tombstone_releases_once(service_urls, tmp_path):
    database_url, redis_url = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=f"test-{uuid.uuid4().hex}:dispatch:v1:")
    try:
        locator = await _admit(store, "job_duplicate", tmp_path, global_limit=1, tenant_limit=1)
        await broker.enqueue(locator)
        await broker.enqueue(locator)  # outbox retry is idempotent
        lease = await broker.acquire_lease("worker-a", lease_seconds=10)
        assert lease is not None and lease.request == locator
        claim = await store.claim_dispatch(locator, "worker-a", lease_seconds=10)
        with pytest.raises(DispatchAuthorityLost):
            await store.claim_dispatch(locator, "worker-b", lease_seconds=10)
        await store.finish_dispatch(claim, state="succeeded", exit_code=0)
        with pytest.raises(DispatchAuthorityLost):
            await store.finish_dispatch(claim, state="succeeded", exit_code=0)
        await broker.release_lease("worker-a", locator.job_id)
        await broker.enqueue(locator)  # duplicated delivery after terminal release
        with pytest.raises(DispatchAuthorityLost):
            await store.claim_dispatch(locator, "worker-b", lease_seconds=10)
        await _admit(store, "job_after_terminal", tmp_path, global_limit=1, tenant_limit=1)
        engine, _ = await _SharedEngine.get(database_url)
        async with engine.connect() as connection:
            assert (
                await connection.scalar(
                    text("SELECT count(*) FROM job_events WHERE job_id='job_duplicate' AND event_type='done'")
                )
                == 1
            )
            assert await connection.scalar(text("SELECT active FROM admission_capacity WHERE scope='global'")) == 1
    finally:
        await broker.reset()
        await broker.close()
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_expired_lease_cannot_renew_publish_or_requeue_and_recovery_releases_slot(service_urls, tmp_path):
    database_url, redis_url = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=f"test-{uuid.uuid4().hex}:dispatch:v1:")
    try:
        locator = await _admit(store, "job_stale", tmp_path, global_limit=1, tenant_limit=1)
        await broker.enqueue(locator)
        await broker.acquire_lease("worker-a", lease_seconds=0.12)
        claim = await store.claim_dispatch(locator, "worker-a", lease_seconds=0.12)
        await asyncio.sleep(0.16)
        with pytest.raises(LeaseNotHeldError):
            await broker.extend_lease("worker-a", locator.job_id, lease_seconds=10)
        with pytest.raises(DispatchAuthorityLost):
            await store.renew_dispatch(claim, lease_seconds=10)
        with pytest.raises(DispatchAuthorityLost):
            await store.finish_dispatch(claim, state="succeeded", exit_code=0)
        assert await broker.reclaim_expired_leases(now=await broker.server_time()) == [locator.job_id]
        assert await broker.acquire_lease("worker-b", lease_seconds=10) is None
        assert await store.expire_dispatches() == [locator.job_id]
        assert await store.expire_dispatches() == []
        with pytest.raises(DispatchAuthorityLost):
            await store.claim_dispatch(locator, "worker-b", lease_seconds=10)
        await _admit(store, "job_after_expiry", tmp_path, global_limit=1, tenant_limit=1)
    finally:
        await broker.reset()
        await broker.close()
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_retryable_executor_claim_survives_until_database_expiry(service_urls, tmp_path, monkeypatch):
    from transformation_portal.orchestrator import worker as worker_module
    from transformation_portal.orchestrator.dispatch import current_dispatch_fence
    from transformation_portal.orchestrator.worker import RetryableExecutorUnavailable, WorkerConfig, WorkerRunner

    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    released = []
    try:
        locator = await _admit(store, "job_retryable_start", tmp_path, global_limit=1, tenant_limit=1)

        class Broker:
            async def acquire_lease(self, worker_id, *, lease_seconds):
                return JobLease(locator.job_id, worker_id, time.monotonic() + lease_seconds, locator)

            async def release_lease(self, worker_id, job_id):
                released.append(job_id)

        async def unavailable_executor(request, cancellation):
            assert request == locator
            assert current_dispatch_fence() is not None
            raise RetryableExecutorUnavailable("temporary job hydration failure")

        monkeypatch.setattr(worker_module, "get_operational_record_store", lambda: store)
        runner = WorkerRunner(
            broker=Broker(),
            config=WorkerConfig(worker_id="worker-retryable", lease_seconds=2.0, heartbeat_interval_seconds=10.0),
            executor=unavailable_executor,
        )
        assert await runner.step() is True
        assert released == []
        assert current_dispatch_fence() is None
        engine, _ = await _SharedEngine.get(database_url)
        async with engine.connect() as connection:
            row = (
                await connection.execute(
                    text("SELECT state, generation_id FROM dispatch_attempts WHERE job_id=:job_id"),
                    {"job_id": locator.job_id},
                )
            ).one()
            assert row.state == "running"
            assert row.generation_id is None
            assert await connection.scalar(text("SELECT active FROM admission_capacity WHERE scope='global'")) == 1
            assert await connection.scalar(text("SELECT count(*) FROM job_events WHERE event_type='done'")) == 0
        assert await store.expire_dispatches() == []
        await asyncio.sleep(2.1)
        assert await store.expire_dispatches() == [locator.job_id]
        assert await store.expire_dispatches() == []
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT active FROM admission_capacity WHERE scope='global'")) == 0
            assert (
                await connection.scalar(
                    text("SELECT state FROM dispatch_attempts WHERE job_id=:job_id"), {"job_id": locator.job_id}
                )
                == "worker_lost"
            )
            assert await connection.scalar(text("SELECT count(*) FROM job_events WHERE event_type='done'")) == 1
        with pytest.raises(DispatchAuthorityLost):
            await store.claim_dispatch(locator, "worker-late", lease_seconds=10)
        assert await store.committed_manifest(locator.job_id, locator.tenant_id) is None
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_generation_visibility_requires_current_claim_and_preserves_immutable_bytes(service_urls, tmp_path):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    artifacts = LocalArtifactStore(root_dir=tmp_path / "artifacts")
    try:
        locator = await _admit(store, "job_generation", tmp_path)
        claim = await store.claim_dispatch(locator, "worker-a", lease_seconds=10)
        output = Path(claim.output_root)
        output.mkdir(parents=True)
        source = output / "result.txt"
        source.write_bytes(b"verified generation")
        publisher = GenerationPublisher(artifact_store=artifacts, record_store=store)
        wrong = DispatchFence(
            locator, "worker-b", claim.lease_epoch, claim.lease_valid_until, claim.output_root, claim.requested_output_root
        )
        with pytest.raises(DispatchAuthorityLost):
            await publisher.publish(
                wrong, {"result.txt": source}, state="succeeded", exit_code=0, artifacts={}, run_summary={}
            )
        assert await store.committed_manifest(locator.job_id, locator.tenant_id) is None
        await publisher.publish(claim, {"result.txt": source}, state="succeeded", exit_code=0, artifacts={}, run_summary={})
        manifest = await store.committed_manifest(locator.job_id, locator.tenant_id)
        assert manifest["files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert await store.committed_manifest(locator.job_id, "other_tenant") is None
        stored_path = manifest["files"][0]["storage_path"]
        source.write_bytes(b"replacement")
        from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError

        with pytest.raises(ArtifactStoreError):
            await artifacts.write_immutable_file(locator.job_id, stored_path, source)
        stream = await artifacts.open_bytes(locator.job_id, stored_path)
        assert b"".join([chunk async for chunk in stream]) == b"verified generation"
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_locator_queue_rejects_executable_and_unknown_fields(service_urls):
    _, redis_url = service_urls
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=f"test-{uuid.uuid4().hex}:dispatch:v1:")
    try:
        with pytest.raises(QueueBrokerError):
            await broker.enqueue(JobEnqueueRequest(job_id="job_raw", argv=["sh", "-c", "echo unsafe"]))
        locator = DispatchLocator("job_x", "attempt", "dispatch", "a" * 64, "tenant")
        with pytest.raises(ValueError):
            DispatchLocator.from_json(locator.to_json()[:-1] + ',"argv":[]}')
    finally:
        await broker.close()


def _claim_and_wait_process(database_url, redis_url, prefix, result):
    async def run():
        store = PostgresOperationalRecordStore(database_url=database_url)
        broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=prefix)
        lease = await broker.acquire_lease("crash-worker", lease_seconds=0.5)
        claim = await store.claim_dispatch(lease.request, "crash-worker", lease_seconds=0.5)
        result.put(claim)
        await asyncio.Event().wait()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_killed_worker_process_leaves_no_reacquirable_dispatch(service_urls, tmp_path):
    database_url, redis_url = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    prefix = f"test-{uuid.uuid4().hex}:dispatch:v1:"
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=prefix)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_claim_and_wait_process, args=(database_url, redis_url, prefix, result))
    try:
        locator = await _admit(store, "job_crash", tmp_path, global_limit=1, tenant_limit=1)
        await broker.enqueue(locator)
        process.start()
        claim = await asyncio.to_thread(result.get, True, 15)
        process.kill()
        await asyncio.to_thread(process.join, 5)
        assert process.exitcode is not None and process.exitcode != 0
        await asyncio.sleep(0.55)
        assert await store.expire_dispatches() == [locator.job_id]
        assert await broker.reclaim_expired_leases(now=await broker.server_time()) == [locator.job_id]
        assert await broker.acquire_lease("replacement-worker", lease_seconds=10) is None
        with pytest.raises(DispatchAuthorityLost):
            await store.finish_dispatch(claim, state="succeeded", exit_code=0)
        await _admit(store, "job_replacement", tmp_path, global_limit=1, tenant_limit=1)
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        result.close()
        result.join_thread()
        await broker.reset()
        await broker.close()
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_minio_generation_is_conditional_digest_verified_and_revocable(service_urls, tmp_path):
    endpoint = os.getenv("TP_DISPATCH_TEST_S3_ENDPOINT")
    bucket = os.getenv("TP_DISPATCH_TEST_S3_BUCKET")
    if not endpoint or not bucket:
        pytest.skip("isolated S3-compatible service required")
    import boto3

    from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

    database_url, _ = service_urls
    client = boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1")
    artifacts = S3ArtifactStore(bucket=bucket, prefix=f"dispatch-test-{uuid.uuid4().hex}", client=client)
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator = await _admit(store, "job_s3", tmp_path)
        claim = await store.claim_dispatch(locator, "s3-worker", lease_seconds=10)
        output = Path(claim.output_root)
        output.mkdir(parents=True)
        source = output / "artifact.txt"
        source.write_bytes(b"real MinIO immutable generation")
        await GenerationPublisher(artifact_store=artifacts, record_store=store).publish(
            claim, {"artifact.txt": source}, state="succeeded", exit_code=0, artifacts={}, run_summary={}
        )
        manifest = await store.committed_manifest(locator.job_id, locator.tenant_id)
        item = manifest["files"][0]
        source.write_bytes(b"stale replacement")
        with pytest.raises(ArtifactStoreError):
            await artifacts.write_immutable_file(locator.job_id, item["storage_path"], source)
        stream = await artifacts.open_bytes(locator.job_id, item["storage_path"])
        assert b"".join([chunk async for chunk in stream]) == b"real MinIO immutable generation"
        await store.revoke_generation(locator.job_id, locator.tenant_id, reason="explicit_delete")
        assert await store.committed_manifest(locator.job_id, locator.tenant_id) is None
        with pytest.raises(DispatchAuthorityLost):
            await store.finish_dispatch(claim, state="succeeded", exit_code=0)
    finally:
        try:
            await artifacts.reset()
        finally:
            client.close()
            await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_database_rejects_immutable_identity_and_unfenced_projection_mutation(service_urls, tmp_path):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator = await _admit(store, "job_guard", tmp_path)
        engine, _ = await _SharedEngine.get(database_url)
        from sqlalchemy.exc import DBAPIError

        for statement in (
            "UPDATE dispatch_attempts SET plan_digest = repeat('a',64) WHERE job_id='job_guard'",
            "UPDATE jobs SET state='succeeded' WHERE id='job_guard'",
            "UPDATE dispatch_plans SET canonical_bytes=decode('00','hex')",
            "DELETE FROM dispatch_attempts WHERE job_id='job_guard'",
        ):
            with pytest.raises(DBAPIError):
                async with engine.begin() as connection:
                    await connection.execute(text(statement))
        assert await store.get_locator(locator.job_id) == locator
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_http_admission_independent_worker_and_committed_artifact_route(service_urls, tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    import httpx

    import app as application
    from transformation_portal.orchestrator.artifact_store import reset_singleton as reset_artifacts
    from transformation_portal.orchestrator.queue import reset_singleton as reset_queue
    from transformation_portal.orchestrator.storage import reset_singletons

    database_url, redis_url = service_urls
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "fixtures" / "archive_small"
    output = tmp_path / "requested-output"
    artifact_root = tmp_path / "artifact-store"
    prefix = "http-test-" + uuid.uuid4().hex
    environment = {
        "TP_ORCHESTRATOR_STATE_BACKEND": "postgres",
        "TP_ORCHESTRATOR_QUEUE_BACKEND": "redis",
        "TP_DATABASE_URL": database_url,
        "TP_REDIS_URL": redis_url,
        "TP_REDIS_KEY_PREFIX": prefix,
        "TP_PILOT_MAX_ACTIVE_JOBS_PER_TENANT": "2",
        "TP_ARTIFACT_STORE": "local",
        "TP_ARTIFACT_LOCAL_ROOT": str(artifact_root),
        "TP_ALLOWED_INPUT_ROOTS": f"{fixture},{tmp_path}",
        "TP_ALLOWED_OUTPUT_ROOTS": str(tmp_path),
        "TP_WORKER_LEASE_SECONDS": "10",
        "TP_WORKER_HEARTBEAT_SECONDS": "1",
        "PYTHONPATH": f"{repo_root / 'src'}:{repo_root}",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(application, "PILOT_CONTROL_PLANE_ENABLED", False)
    monkeypatch.setattr(application, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 2)
    monkeypatch.setattr(application, "ALLOWED_INPUT_ROOTS", [fixture, tmp_path])
    monkeypatch.setattr(application, "ALLOWED_OUTPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(application, "ALLOWED_PATH_ROOTS", [fixture, tmp_path])
    monkeypatch.setattr(application, "API_KEY_SECRET", "test-dispatch-api-key")
    monkeypatch.setattr(application, "JOBS", {})
    monkeypatch.setattr(application, "EVENT_SUBSCRIBERS", {})
    reset_singletons()
    reset_queue()
    reset_artifacts()
    worker = None
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=prefix + ":dispatch:v1:")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application.app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/jobs",
                headers={"x-api-key": "test-dispatch-api-key"},
                json={
                    "pipeline": "archive-gate-a",
                    "args": {
                        "input_dir": str(fixture / "archive_root"),
                        "output_dir": str(output),
                        "archive_command": "fixity-scan",
                        "archive_index": str(fixture / "archive_index_normalized.csv.gz"),
                        "validate_schemas": False,
                    },
                },
            )
            assert response.status_code == 200, response.text
            job_id = response.json()["data"]["id"]
            assert not output.exists(), "admission must not create outputs"
            # Separate interpreter, broker connections, repository pool and process runtime.
            with (tmp_path / "worker.log").open("wb") as log:
                worker = subprocess.Popen(
                    [sys.executable, "-m", "transformation_portal.orchestrator.worker_process"],
                    cwd=repo_root,
                    env={**os.environ, **environment},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    status = await client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-dispatch-api-key"})
                    assert status.status_code == 200, status.text
                    if status.json()["data"]["state"] not in {"queued", "running"}:
                        break
                    await asyncio.sleep(0.1)
            assert status.json()["data"]["state"] == "succeeded", (status.text, (tmp_path / "worker.log").read_text())
            store = PostgresOperationalRecordStore(database_url=database_url)
            manifest = await store.committed_manifest(job_id, "default")
            assert manifest and manifest["files"]
            relative = manifest["files"][0]["path"]
            download = await client.get(
                f"/v1/jobs/{job_id}/artifacts/{relative}", headers={"x-api-key": "test-dispatch-api-key"}
            )
            assert download.status_code == 200, download.text
            assert hashlib.sha256(download.content).hexdigest() == manifest["files"][0]["sha256"]
            hidden = await client.get(
                f"/v1/jobs/{job_id}/artifacts/{manifest['files'][0]['storage_path']}",
                headers={"x-api-key": "test-dispatch-api-key"},
            )
            assert hidden.status_code == 404
    finally:
        if worker is not None:
            worker.terminate()
            try:
                await asyncio.to_thread(worker.wait, 5)
            except subprocess.TimeoutExpired:
                worker.kill()
                await asyncio.to_thread(worker.wait, 5)
        await broker.reset()
        await broker.close()
        from transformation_portal.orchestrator.queue import get_queue_broker

        await get_queue_broker().close()
        await _SharedEngine.dispose(database_url)
        reset_singletons()
        reset_queue()
        reset_artifacts()


@pytest.mark.asyncio
async def test_terminal_generation_pointer_cannot_be_cleared_by_a_fresh_unprivileged_session(service_urls, tmp_path):
    from sqlalchemy.exc import DBAPIError
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    fresh = create_async_engine(database_url, poolclass=NullPool)
    try:
        locator = await _admit(store, "job_terminal_guard", tmp_path)
        claim = await store.claim_dispatch(locator, "worker", lease_seconds=10)
        await store.finish_dispatch(claim, state="succeeded", exit_code=0)
        async with fresh.connect() as connection:
            assert await connection.scalar(text("SELECT current_setting('tp.operational_write', true)")) is None
        with pytest.raises(DBAPIError):
            async with fresh.begin() as connection:
                await connection.execute(
                    text(
                        "UPDATE dispatch_attempts SET generation_id=NULL, manifest_digest=NULL WHERE job_id='job_terminal_guard'"
                    )
                )
    finally:
        await fresh.dispose()
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_committed_cancel_and_artifact_delete_survive_redis_delivery_outage(service_urls, tmp_path, monkeypatch):
    import app as application
    from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository

    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    repository = PostgresJobRepository(database_url=database_url)
    artifacts = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    monkeypatch.setattr(application, "_uses_dispatch_authority", lambda: True)
    monkeypatch.setattr(application, "get_operational_record_store", lambda: store)
    monkeypatch.setattr(application, "_job_repository", lambda: repository)
    monkeypatch.setattr(application, "_artifact_store", lambda: artifacts)

    def unavailable_broker():
        raise ConnectionError("Redis is unavailable after commit")

    monkeypatch.setattr(application, "get_queue_broker", unavailable_broker)
    try:
        cancel_locator = await _admit(store, "job_cancel_outage", tmp_path)
        job = application._job_from_record(await repository.get(cancel_locator.job_id))
        await application._request_cancel(job)
        assert (await repository.get(job.id)).state == "canceled"
        assert job.state == "canceled"
        locator = await _admit(store, "job_delete_outage", tmp_path)
        claim = await store.claim_dispatch(locator, "worker", lease_seconds=10)
        output = Path(claim.output_root)
        output.mkdir(parents=True)
        source = output / "result.txt"
        source.write_bytes(b"delete this committed generation")
        await GenerationPublisher(artifact_store=artifacts, record_store=store).publish(
            claim, {"result.txt": source}, state="succeeded", exit_code=0, artifacts={}, run_summary={}
        )
        job = application._job_from_record(await repository.get(locator.job_id))
        deleted = await application._delete_job_artifacts_for_job(
            job, reason="explicit_delete", fail_closed_on_repository=True
        )
        assert deleted == 2  # artifact plus immutable manifest
        assert await store.committed_manifest(locator.job_id, locator.tenant_id) is None
        assert job.artifacts["lifecycle"]["deletion_status"] == "deleted"
        assert await store.pending_outbox(), "delivery remains retryable after Redis recovers"
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_bytes_staged_after_claim_expiry_never_gain_reader_visibility(service_urls, tmp_path):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    artifacts = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    try:
        locator = await _admit(store, "job_expired_publication", tmp_path)
        claim = await store.claim_dispatch(locator, "worker", lease_seconds=0.1)
        output = Path(claim.output_root)
        output.mkdir(parents=True)
        source = output / "result.txt"
        source.write_bytes(b"finished after the lease expired")
        await asyncio.sleep(0.15)
        with pytest.raises(DispatchAuthorityLost):
            await GenerationPublisher(artifact_store=artifacts, record_store=store).publish(
                claim, {"result.txt": source}, state="succeeded", exit_code=0, artifacts={}, run_summary={}
            )
        assert list((artifacts.root_dir / locator.job_id).rglob("result.txt")), "staging itself is not a commit"
        assert await store.committed_manifest(locator.job_id, locator.tenant_id) is None
        await store.expire_dispatches()
        with pytest.raises(DispatchAuthorityLost):
            await store.claim_dispatch(locator, "replacement", lease_seconds=10)
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_cleanup_after_expired_publication_removes_only_invisible_staging_and_private_output(service_urls, tmp_path):
    from transformation_portal.orchestrator.artifact_store.generation_cleanup import cleanup_terminal_generation

    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    artifacts = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    try:
        locator = await _admit(store, "job_cleanup_stale", tmp_path)
        claim = await store.claim_dispatch(locator, "worker", lease_seconds=0.1)
        output = Path(claim.output_root)
        output.mkdir(parents=True)
        source = output / "result.txt"
        source.write_bytes(b"uncommitted generation")
        protected = tmp_path / "existing-user-output.txt"
        protected.write_bytes(b"preserve")
        await asyncio.sleep(0.15)
        with pytest.raises(DispatchAuthorityLost):
            await GenerationPublisher(artifact_store=artifacts, record_store=store).publish(
                claim, {"result.txt": source}, state="succeeded", exit_code=0, artifacts={}, run_summary={}
            )
        await store.expire_dispatches()
        assert locator.job_id in await store.generation_cleanup_candidates()
        await cleanup_terminal_generation(record_store=store, artifact_store=artifacts, job_id=locator.job_id)
        assert not output.exists()
        assert not (artifacts.root_dir / locator.job_id).exists()
        assert protected.read_bytes() == b"preserve"
        assert locator.job_id not in await store.generation_cleanup_candidates()
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_cleanup_preserves_generation_when_commit_acknowledgement_is_lost(service_urls, tmp_path, monkeypatch):
    from transformation_portal.orchestrator.artifact_store.generation_cleanup import cleanup_terminal_generation

    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    artifacts = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    original_commit = store.commit_generation

    async def commit_then_lose_ack(
        fence, *, generation_id, manifest_bytes, state, exit_code, artifacts, run_summary, error=None
    ):
        await original_commit(
            fence,
            generation_id=generation_id,
            manifest_bytes=manifest_bytes,
            state=state,
            exit_code=exit_code,
            artifacts=artifacts,
            run_summary=run_summary,
            error=error,
        )
        raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(store, "commit_generation", commit_then_lose_ack)
    try:
        locator = await _admit(store, "job_cleanup_committed", tmp_path)
        claim = await store.claim_dispatch(locator, "worker", lease_seconds=10)
        output = Path(claim.output_root)
        output.mkdir(parents=True)
        source = output / "result.txt"
        source.write_bytes(b"committed despite lost acknowledgement")
        with pytest.raises(ConnectionError):
            await GenerationPublisher(artifact_store=artifacts, record_store=store).publish(
                claim, {"result.txt": source}, state="succeeded", exit_code=0, artifacts={}, run_summary={}
            )
        await cleanup_terminal_generation(record_store=store, artifact_store=artifacts, job_id=locator.job_id)
        assert not output.exists()
        manifest = await store.committed_manifest(locator.job_id, locator.tenant_id)
        stream = await artifacts.open_bytes(locator.job_id, manifest["files"][0]["storage_path"])
        assert b"".join([chunk async for chunk in stream]) == b"committed despite lost acknowledgement"
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_fence_is_rechecked_after_waiting_for_projection_row_lock(service_urls, tmp_path):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator = await _admit(store, "job_lock_expired", tmp_path)
        claim = await store.claim_dispatch(locator, "worker", lease_seconds=0.1)
        engine, _ = await _SharedEngine.get(database_url)
        async with engine.begin() as blocker:
            await blocker.execute(text("SELECT id FROM jobs WHERE id='job_lock_expired' FOR UPDATE"))
            pending = asyncio.create_task(store.finish_dispatch(claim, state="succeeded", exit_code=0))
            await asyncio.sleep(0.15)
        with pytest.raises(DispatchAuthorityLost):
            await pending
        assert await store.expire_dispatches() == [locator.job_id]
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_periodic_scrub_removes_late_recreated_output_and_preserves_committed_generation(
    service_urls, tmp_path, monkeypatch
):
    from transformation_portal.orchestrator.artifact_store.generation_cleanup import cleanup_terminal_generation
    from transformation_portal.orchestrator.execution_workspace import create_execution_workspace

    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution-storage"))
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    artifacts = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    try:
        locator = await _admit(store, "job_periodic_scrub", tmp_path)
        claim = await store.claim_dispatch(locator, "worker", lease_seconds=10)
        output = Path(claim.output_root)
        output.mkdir(parents=True)
        source = output / "result.txt"
        source.write_bytes(b"committed original")
        private = create_execution_workspace(output, require_configured=True)
        (private / "original-native-output").write_bytes(b"private original")
        await GenerationPublisher(artifact_store=artifacts, record_store=store).publish(
            claim, {"result.txt": source}, state="succeeded", exit_code=0, artifacts={}, run_summary={}
        )
        await cleanup_terminal_generation(record_store=store, artifact_store=artifacts, job_id=locator.job_id)
        assert not output.exists()
        assert not private.exists()
        output.mkdir()
        (output / "late.txt").write_bytes(b"late orphan child write")
        private = create_execution_workspace(output)
        (private / "late-native-output").write_bytes(b"late private write")
        assert locator.job_id not in await store.generation_cleanup_candidates()
        engine, _ = await _SharedEngine.get(database_url)
        async with engine.begin() as connection:
            await connection.execute(text("SELECT set_config('tp.operational_write', 'on', true)"))
            await connection.execute(
                text("UPDATE dispatch_attempts SET cleanup_checked_at=cleanup_checked_at-3601 WHERE job_id=:job_id"),
                {"job_id": locator.job_id},
            )
        assert locator.job_id in await store.generation_cleanup_candidates()
        await cleanup_terminal_generation(record_store=store, artifact_store=artifacts, job_id=locator.job_id)
        assert not output.exists()
        assert not private.exists()
        manifest = await store.committed_manifest(locator.job_id, locator.tenant_id)
        stream = await artifacts.open_bytes(locator.job_id, manifest["files"][0]["storage_path"])
        assert b"".join([chunk async for chunk in stream]) == b"committed original"
        async with engine.connect() as connection:
            clean_records = await connection.scalar(
                text("SELECT count(*) FROM operational_records WHERE job_id=:job_id AND kind='staging_cleaned'"),
                {"job_id": locator.job_id},
            )
        assert clean_records == 1  # periodic checking must not grow immutable evidence hourly
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.asyncio
async def test_cleanup_reservations_are_bounded_and_a_failed_first_check_cannot_starve_later_jobs(service_urls, tmp_path):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        for index in range(3):
            locator = await _admit(store, f"job_fair_{index}", tmp_path)
            claim = await store.claim_dispatch(locator, "worker", lease_seconds=10)
            await store.finish_dispatch(claim, state="failed", exit_code=1)
        first = await store.generation_cleanup_candidates(limit=1)
        assert len(first) == 1
        # No cleanup/ack happens for first: emulate an unavailable filesystem.
        second, third = await asyncio.gather(
            store.generation_cleanup_candidates(limit=1), store.generation_cleanup_candidates(limit=1)
        )
        assert len(second) == len(third) == 1
        assert len(set(first + second + third)) == 3
        assert await store.generation_cleanup_candidates() == []
    finally:
        await _SharedEngine.dispose(database_url)
