"""Real Postgres proof for immutable photography bindings and migration safety."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from tests.orchestrator.test_dispatch_authority_services import service_urls  # noqa: F401 - shared isolated database
from tests.orchestrator.test_dispatch_authority_services import _admit, _record
from tests.orchestrator.test_operational_storage_offline import _photography_authority
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost, PostgresOperationalRecordStore
from transformation_portal.orchestrator.storage.postgres import _SharedEngine

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _admit_photography(store, root: Path):
    plan, bindings = _photography_authority(root)
    record = _record("job_photography")
    record.request["pipeline"] = record.effective_request["pipeline"] = "lux-depth-v5"
    locator = await store.admit(
        record,
        plan,
        tenant_id="tenant_a",
        output_root=str(root / "attempt"),
        requested_output_root=str(root / "requested"),
        global_limit=4,
        tenant_limit=2,
        execution_bindings=bindings,
    )
    return locator, plan, bindings


def _migration(connection, direction):
    module = importlib.import_module("migrations.versions.0007_photography_bindings")
    with Operations.context(MigrationContext.configure(connection)):
        getattr(module, direction)()


async def test_photography_bindings_are_stored_audited_and_immutable(service_urls, tmp_path):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator, plan, bindings = await _admit_photography(store, tmp_path)
        assert await store.fetch_plan(locator) == plan
        assert await store.fetch_execution_bindings(locator) == bindings
        assert locator.plan_digest == hashlib.sha256(plan).hexdigest()
        engine, _ = await _SharedEngine.get(database_url)
        async with engine.connect() as connection:
            row = (
                await connection.execute(
                    text("SELECT execution_bindings, execution_bindings_digest FROM dispatch_attempts WHERE job_id=:job"),
                    {"job": locator.job_id},
                )
            ).one()
            assert row.execution_bindings == bindings
            assert row.execution_bindings_digest == hashlib.sha256(bindings).hexdigest()
            evidence = await connection.scalar(
                text("SELECT canonical_bytes FROM operational_records WHERE job_id=:job AND kind='admitted'"),
                {"job": locator.job_id},
            )
            assert json.loads(evidence)["execution_bindings_digest"] == row.execution_bindings_digest
        for update in (
            "execution_bindings = execution_bindings || decode('20', 'hex')",
            "execution_bindings_digest = repeat('f', 64)",
            "execution_bindings = NULL, execution_bindings_digest = NULL",
        ):
            with pytest.raises(DBAPIError, match="immutable dispatch identity"):
                async with engine.begin() as connection:
                    # Operational projection permission never grants identity mutation.
                    await connection.execute(text("SELECT set_config('tp.operational_write', 'on', true)"))
                    await connection.execute(
                        text(f"UPDATE dispatch_attempts SET {update} WHERE job_id=:job"), {"job": locator.job_id}
                    )
        assert await store.fetch_execution_bindings(locator) == bindings
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.parametrize(
    "bindings,digest",
    [(None, "a" * 64), (b"{}", None), (b"", "a" * 64), (b"x" * 65537, "a" * 64), (b"{}", "A" * 64)],
)
async def test_database_rejects_unpaired_or_unbounded_bindings_on_insert(service_urls, tmp_path, bindings, digest):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator = await _admit(store, "job_source", tmp_path)
        engine, _ = await _SharedEngine.get(database_url)
        with pytest.raises(DBAPIError, match="ck_dispatch_execution_bindings"):
            async with engine.begin() as connection:
                await connection.execute(
                    text("""
                        INSERT INTO dispatch_attempts (
                            job_id, attempt_id, dispatch_id, tenant_id, plan_digest,
                            execution_bindings, execution_bindings_digest,
                            api_version, output_root, requested_output_root, state, lease_epoch, admitted_at
                        ) SELECT
                            'job_invalid', 'attempt_invalid', 'dispatch_invalid', tenant_id, plan_digest,
                            :bindings, :digest,
                            api_version, output_root, requested_output_root, state, lease_epoch, admitted_at
                        FROM dispatch_attempts WHERE job_id = :job
                        """),
                    {"bindings": bindings, "digest": digest, "job": locator.job_id},
                )
        assert await store.fetch_execution_bindings(locator) is None
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.parametrize("terminal", [False, True])
async def test_migration_refuses_to_remove_admitted_photography_authority(service_urls, tmp_path, terminal):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator, plan, bindings = await _admit_photography(store, tmp_path)
        if terminal:
            fence = await store.claim_dispatch(locator, "worker", lease_seconds=10)
            await store.finish_dispatch(fence, state="failed", exit_code=1)
        engine, _ = await _SharedEngine.get(database_url)
        with pytest.raises(DBAPIError, match="cannot remove immutable photography execution bindings"):
            async with engine.begin() as connection:
                await connection.run_sync(_migration, "downgrade")
        assert await store.fetch_plan(locator) == plan
        assert await store.fetch_execution_bindings(locator) == bindings
    finally:
        await _SharedEngine.dispose(database_url)


async def test_binding_migration_downgrade_and_upgrade_preserve_legacy_attempts(service_urls, tmp_path):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator = await _admit(store, "job_legacy", tmp_path)
        plan = await store.fetch_plan(locator)
        engine, _ = await _SharedEngine.get(database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_migration, "downgrade")
            assert await connection.scalar(text("""
                        SELECT count(*) FROM information_schema.columns
                        WHERE table_schema = current_schema() AND table_name = 'dispatch_attempts'
                        AND column_name IN ('execution_bindings', 'execution_bindings_digest')
                        """)) == 0
            await connection.run_sync(_migration, "upgrade")
        assert await store.get_locator(locator.job_id) == locator
        assert await store.fetch_plan(locator) == plan
        assert await store.fetch_execution_bindings(locator) is None
    finally:
        await _SharedEngine.dispose(database_url)


async def test_downgrade_waits_for_inflight_admission_and_then_refuses_authority_loss(service_urls, tmp_path, monkeypatch):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    pending_writes, allow_commit, downgrade_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_session = store._session
    downgrade_backend = []
    tasks = []

    @asynccontextmanager
    async def session_with_held_commit():
        async with original_session() as session:
            original_begin = session.begin

            @asynccontextmanager
            async def held_transaction():
                async with original_begin():
                    yield
                    # Exercise the real admission writes while holding their
                    # transaction open, rather than synthesizing a committed row.
                    await session.flush()
                    pending_writes.set()
                    await allow_commit.wait()

            session.begin = held_transaction
            yield session

    monkeypatch.setattr(store, "_session", session_with_held_commit)
    try:
        engine, _ = await _SharedEngine.get(database_url)
        admission = asyncio.create_task(_admit_photography(store, tmp_path))
        tasks.append(admission)
        await asyncio.wait_for(pending_writes.wait(), timeout=10)

        async def downgrade():
            async with engine.begin() as connection:
                downgrade_backend.append(await connection.scalar(text("SELECT pg_backend_pid()")))
                downgrade_started.set()
                await connection.run_sync(_migration, "downgrade")

        migration = asyncio.create_task(downgrade())
        tasks.append(migration)
        await asyncio.wait_for(downgrade_started.wait(), timeout=10)
        async with engine.connect() as connection:
            for _ in range(100):
                wait_type = await connection.scalar(
                    text("SELECT wait_event_type FROM pg_stat_activity WHERE pid=:pid"),
                    {"pid": downgrade_backend[0]},
                )
                if wait_type == "Lock":
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Downgrade did not wait for the admitted attempt's transaction lock")
        assert not migration.done()
        allow_commit.set()
        locator, plan, bindings = await asyncio.wait_for(admission, timeout=10)
        with pytest.raises(DBAPIError, match="cannot remove immutable photography execution bindings"):
            await asyncio.wait_for(migration, timeout=10)
        assert await store.fetch_plan(locator) == plan
        assert await store.fetch_execution_bindings(locator) == bindings
    finally:
        allow_commit.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _SharedEngine.dispose(database_url)


async def test_prelaunch_authority_check_does_not_extend_live_lease_or_append_evidence(service_urls, tmp_path):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator = await _admit(store, "job_prelaunch", tmp_path)
        fence = await store.claim_dispatch(locator, "worker", lease_seconds=30)
        await store.assert_dispatch_authority(fence)
        engine, _ = await _SharedEngine.get(database_url)
        async with engine.connect() as connection:
            assert (
                await connection.scalar(
                    text("SELECT lease_valid_until FROM dispatch_attempts WHERE job_id=:job"), {"job": locator.job_id}
                )
                == fence.lease_valid_until
            )
            assert (
                await connection.scalar(
                    text("SELECT count(*) FROM operational_records WHERE job_id=:job"), {"job": locator.job_id}
                )
                == 2
            )  # Admission and claim only.
            assert (
                await connection.scalar(text("SELECT count(*) FROM job_events WHERE job_id=:job"), {"job": locator.job_id})
                == 0
            )
    finally:
        await _SharedEngine.dispose(database_url)


@pytest.mark.parametrize("revocation", ["canceled", "cancel_requested", "expired", "holder", "epoch", "locator"])
async def test_prelaunch_authority_check_rejects_cancellation_expiry_and_stale_fences(service_urls, tmp_path, revocation):
    database_url, _ = service_urls
    store = PostgresOperationalRecordStore(database_url=database_url)
    try:
        locator = await _admit(store, "job_prelaunch_revoked", tmp_path)
        fence = await store.claim_dispatch(locator, "worker", lease_seconds=30)
        if revocation == "canceled":
            assert await store.cancel_dispatch(locator.job_id, locator.tenant_id)
        elif revocation in {"expired", "cancel_requested"}:
            engine, _ = await _SharedEngine.get(database_url)
            async with engine.begin() as connection:
                if revocation == "expired":
                    await connection.execute(
                        text(
                            "UPDATE dispatch_attempts SET lease_valid_until=extract(epoch FROM clock_timestamp()) - 1 WHERE job_id=:job"
                        ),
                        {"job": locator.job_id},
                    )
                else:
                    await connection.execute(text("SELECT set_config('tp.operational_write', 'on', true)"))
                    await connection.execute(
                        text("UPDATE jobs SET cancel_requested=true WHERE id=:job"), {"job": locator.job_id}
                    )
        elif revocation == "holder":
            fence = replace(fence, holder="other-worker")
        elif revocation == "epoch":
            fence = replace(fence, lease_epoch=fence.lease_epoch + 1)
        else:
            fence = replace(fence, locator=replace(locator, dispatch_id="other-dispatch"))
        with pytest.raises(DispatchAuthorityLost):
            await store.assert_dispatch_authority(fence)
    finally:
        await _SharedEngine.dispose(database_url)
