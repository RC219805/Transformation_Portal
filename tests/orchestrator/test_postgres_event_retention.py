"""Postgres replay bounds and cursor integrity across processes/transactions."""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from transformation_portal.orchestrator.models import JobEventModel, JobEventSequenceModel
from transformation_portal.orchestrator.storage.base import JobRecord, RepositoryError
from transformation_portal.orchestrator.storage.postgres import (
    PostgresJobEventStore,
    PostgresJobRepository,
    append_event_in_session,
    event_retention_per_job,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def postgres_events() -> AsyncIterator[tuple[PostgresJobRepository, PostgresJobEventStore]]:
    url = os.getenv("TP_TEST_POSTGRES_URL", "").strip()
    if not url:
        pytest.skip("TP_TEST_POSTGRES_URL is not set; Postgres replay contracts require a live service")
    repo = PostgresJobRepository(database_url=url)
    store = PostgresJobEventStore(database_url=url, per_job_cap=3)
    await repo.reset()
    try:
        yield repo, store
    finally:
        await repo.reset()
        await repo.close()
        await store.close()


@pytest.mark.parametrize("value", ["0", "-1", "", "bad", "1.5", "True", "١"])
async def test_event_retention_config_rejects_nonpositive_or_invalid_caps(monkeypatch, value):
    monkeypatch.setenv("TP_ORCHESTRATOR_EVENT_RETENTION_PER_JOB", value)
    with pytest.raises(RepositoryError):
        event_retention_per_job()


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
async def test_event_store_rejects_invalid_explicit_cap(cap):
    with pytest.raises(RepositoryError):
        PostgresJobEventStore(database_url="postgresql+asyncpg://host/db", per_job_cap=cap)


async def test_concurrent_writers_allocate_once_and_retain_latest(postgres_events):
    _, store = postgres_events
    stores = [PostgresJobEventStore(database_url=store._database_url, per_job_cap=3) for _ in range(5)]
    appended = await asyncio.gather(
        *[stores[i % len(stores)].append("orphan-concurrent", "progress", {"i": i}, created_at=float(i)) for i in range(50)]
    )
    assert sorted(event.seq for event in appended) == list(range(1, 51))
    replay = [event async for event in store.events_since("orphan-concurrent", after_seq=1)]
    assert [event.seq for event in replay] == [48, 49, 50]
    assert [event.seq for event in await _replay(store, "orphan-concurrent", 49)] == [50]


async def _replay(store, job_id, after_seq=0):
    return [event async for event in store.events_since(job_id, after_seq=after_seq)]


async def test_cursor_survives_complete_history_prune_and_new_store(postgres_events):
    _, store = postgres_events
    for i in range(10):
        await store.append("job-pruned", "log", {"line": i}, created_at=float(i))
    assert [event.seq for event in await _replay(store, "job-pruned")] == [8, 9, 10]
    async with store._session() as session:
        async with session.begin():
            await session.execute(delete(JobEventModel).where(JobEventModel.job_id == "job-pruned"))
    restarted = PostgresJobEventStore(database_url=store._database_url, per_job_cap=3)
    assert (await restarted.append("job-pruned", "done", {}, created_at=20)).seq == 11
    assert (await restarted.append("other-job", "state", {}, created_at=20)).seq == 1


async def test_transaction_rollback_restores_counter_events_and_pruning(postgres_events):
    _, store = postgres_events
    first = await store.append("job-rollback", "state", {"state": "running"}, created_at=1)
    with pytest.raises(RuntimeError, match="rollback"):
        async with store._session() as session:
            async with session.begin():
                event = await append_event_in_session(session, "job-rollback", "done", {}, 2, per_job_cap=1)
                assert event.seq == 2
                raise RuntimeError("rollback")
    assert [event.seq for event in await _replay(store, "job-rollback")] == [first.seq]
    assert (await store.append("job-rollback", "done", {}, created_at=3)).seq == 2


async def test_legacy_history_seeds_counter_without_reusing_cursor(postgres_events):
    _, store = postgres_events
    async with store._session() as session:
        async with session.begin():
            session.add(JobEventModel(job_id="job-legacy", seq=123, event_type="state", payload={}, created_at=1))
    assert (await store.append("job-legacy", "done", {}, created_at=2)).seq == 124
    assert [event.seq for event in await _replay(store, "job-legacy")] == [123, 124]


async def test_job_retention_cleans_counter_and_replay_rows(postgres_events):
    repo, store = postgres_events
    await repo.create(JobRecord(id="job-expired", created_at=1, state="succeeded", finished_at=2))
    await store.append("job-expired", "done", {}, created_at=2)
    assert await repo.cleanup_expired(20, 5) == ["job-expired"]
    async with store._session() as session:
        assert (
            await session.execute(select(JobEventSequenceModel).where(JobEventSequenceModel.job_id == "job-expired"))
        ).scalar_one_or_none() is None
    assert await _replay(store, "job-expired") == []


async def test_durable_dispatch_authority_survives_legacy_cleanup_and_restart_sweep(postgres_events):
    from transformation_portal.orchestrator.models import DispatchAttemptModel, DispatchPlanModel

    repo, store = postgres_events
    await repo.create(JobRecord(id="job-authority", created_at=1, state="running", finished_at=2))
    async with store._session() as session:
        async with session.begin():
            session.add(DispatchPlanModel(digest="a" * 64, canonical_bytes=b"{}"))
            await session.flush()
            session.add(
                DispatchAttemptModel(
                    job_id="job-authority",
                    attempt_id="attempt-a",
                    dispatch_id="dispatch-a",
                    tenant_id="tenant_a",
                    plan_digest="a" * 64,
                    api_version="v1",
                    output_root="/tmp/attempt-a",
                    requested_output_root="/tmp/out",
                    state="running",
                    admitted_at=1,
                    lease_epoch=1,
                )
            )
    with pytest.raises(RepositoryError, match="active durable dispatch"):
        await repo.delete("job-authority")
    assert await repo.cleanup_expired(20, 5) == []
    assert await repo.sweep_orphaned() == []
    assert (await repo.get("job-authority")).state == "running"


async def test_postgres_sse_observes_other_host_committed_terminal_once(postgres_events, monkeypatch):
    from types import SimpleNamespace

    import app as orchestrator_app

    repo, store = postgres_events
    await repo.create(JobRecord(id="job-live", created_at=1, state="running"))
    monkeypatch.setattr(orchestrator_app, "_job_repository", lambda: repo)
    job = orchestrator_app.Job(id="job-live", created_at=1, state="running")

    async def connected():
        return False

    request = SimpleNamespace(is_disconnected=connected)
    stream = orchestrator_app._committed_job_event_stream(request, job, store, None)
    assert "event: state" in await anext(stream)
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.05)
    assert not pending.done()
    # Independent writer has no reference to the API host's in-memory queue.
    other_host = PostgresJobEventStore(database_url=store._database_url, per_job_cap=3)
    done = await other_host.append("job-live", "done", {"state": "succeeded", "committed": True}, created_at=2)
    chunk = await asyncio.wait_for(pending, 2.0)
    assert f"id: {done.seq}" in chunk
    assert '"committed":true' in chunk
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_postgres_sse_gap_marker_and_cursor_deduplication(postgres_events):
    from types import SimpleNamespace

    import app as orchestrator_app

    _, store = postgres_events
    for i in range(1, 6):
        await store.append("job-gap", "done" if i == 5 else "log", {"i": i}, created_at=float(i))

    async def connected():
        return False

    job = orchestrator_app.Job(id="job-gap", created_at=1, state="succeeded", done_published_at=5)
    chunks = [
        chunk
        async for chunk in orchestrator_app._committed_job_event_stream(
            SimpleNamespace(is_disconnected=connected), job, store, 1
        )
    ]
    assert chunks[0] == ": replay-gap requested_after=1 first_available=3\n\n"
    assert [chunk.splitlines()[0] for chunk in chunks[1:]] == ["id: 3", "id: 4", "id: 5"]
    resumed = [
        chunk
        async for chunk in orchestrator_app._committed_job_event_stream(
            SimpleNamespace(is_disconnected=connected), job, store, 4
        )
    ]
    assert len(resumed) == 1 and "id: 5" in resumed[0]


async def test_committed_sse_retries_store_failure_without_synthetic_done():
    from types import SimpleNamespace

    import app as orchestrator_app
    from transformation_portal.orchestrator.storage.base import JobEvent

    class RecoveringStore:
        calls = 0

        async def replay_batch(self, _job_id, *, after_seq):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary event store outage")
            return [JobEvent(job_id="job-retry", seq=1, event_type="done", payload={"committed": True}, created_at=1)]

    async def connected():
        return False

    job = orchestrator_app.Job(id="job-retry", created_at=1, state="succeeded", done_published_at=1)
    stream = orchestrator_app._committed_job_event_stream(
        SimpleNamespace(is_disconnected=connected), job, RecoveringStore(), 0
    )
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.05)
    assert not pending.done()
    assert '"committed":true' in await asyncio.wait_for(pending, 2)
    await stream.aclose()


async def test_postgres_sse_closes_after_terminal_event_is_pruned(postgres_events, monkeypatch):
    from types import SimpleNamespace

    import app as orchestrator_app

    repo, store = postgres_events
    job_id = "job-terminal-pruned"
    await repo.create(JobRecord(id=job_id, created_at=1, state="running"))
    monkeypatch.setattr(orchestrator_app, "_job_repository", lambda: repo)

    async def connected():
        return False

    job = orchestrator_app.Job(id=job_id, created_at=1, state="running")
    stream = orchestrator_app._committed_job_event_stream(SimpleNamespace(is_disconnected=connected), job, store, None)
    assert '"state":"running"' in await anext(stream)
    # Another host commits completion, then later events evict done under cap3.
    await repo.update(job_id, state="succeeded", finished_at=2, done_published_at=2)
    await store.append(job_id, "done", {"state": "succeeded"}, created_at=2)
    for seq in range(3):
        await store.append(job_id, "artifact_deleted", {"i": seq}, created_at=3 + seq)

    async def drain():
        return [chunk async for chunk in stream]

    chunks = await asyncio.wait_for(drain(), 2.5)
    assert chunks[0] == ": replay-gap requested_after=0 first_available=2\n\n"
    assert [chunk.splitlines()[0] for chunk in chunks[1:]] == ["id: 2", "id: 3", "id: 4"]
    assert all("event: done" not in chunk for chunk in chunks)
    assert job.done_published_at is None  # The original local snapshot never changed.


async def test_postgres_sse_closes_after_retention_removes_job(postgres_events, monkeypatch):
    from types import SimpleNamespace

    import app as orchestrator_app

    repo, store = postgres_events
    job_id = "job-retained-away"
    await repo.create(JobRecord(id=job_id, created_at=1, state="running"))
    monkeypatch.setattr(orchestrator_app, "_job_repository", lambda: repo)

    async def connected():
        return False

    stream = orchestrator_app._committed_job_event_stream(
        SimpleNamespace(is_disconnected=connected), orchestrator_app.Job(id=job_id, created_at=1, state="running"), store, None
    )
    assert '"state":"running"' in await anext(stream)
    await repo.update(job_id, state="succeeded", finished_at=2, done_published_at=2)
    await store.append(job_id, "done", {"state": "succeeded"}, created_at=2)
    assert await repo.cleanup_expired(20, 5) == [job_id]

    async def drain():
        return [chunk async for chunk in stream]

    assert await asyncio.wait_for(drain(), 1.5) == []


async def test_committed_sse_drains_done_committed_during_terminal_refresh(monkeypatch):
    from types import SimpleNamespace

    import app as orchestrator_app
    from transformation_portal.orchestrator.storage.base import JobEvent

    job_id = "job-terminal-race"

    class RacingStore:
        terminal = False
        cursors = []

        async def replay_batch(self, _job_id, *, after_seq):
            self.cursors.append(after_seq)
            if not self.terminal:
                return []
            return [JobEvent(job_id=job_id, seq=1, event_type="done", payload={"committed": True}, created_at=2)]

        async def get(self, _job_id):
            # The terminal transaction commits after the empty event read.
            self.terminal = True
            return JobRecord(id=job_id, created_at=1, state="succeeded", finished_at=2, done_published_at=2)

    async def connected():
        return False

    store = RacingStore()
    monkeypatch.setattr(orchestrator_app, "_job_repository", lambda: store)
    stream = orchestrator_app._committed_job_event_stream(
        SimpleNamespace(is_disconnected=connected), orchestrator_app.Job(id=job_id, created_at=1, state="running"), store, 0
    )
    chunk = await asyncio.wait_for(anext(stream), 1.5)
    assert "event: done" in chunk and '"committed":true' in chunk
    assert store.cursors == [0, 0]
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.parametrize("read_failures", [0, 1])
async def test_committed_sse_terminal_refresh_uses_repository_and_retries_errors(monkeypatch, read_failures):
    from types import SimpleNamespace

    import app as orchestrator_app

    job_id = "job-terminal-refresh"

    class TerminalRepository:
        calls = 0

        async def get(self, _job_id):
            self.calls += 1
            if self.calls <= read_failures:
                raise RepositoryError("temporary repository outage")
            return JobRecord(id=job_id, created_at=1, state="canceled", finished_at=2, done_published_at=2)

    class EmptyStore:
        calls = 0

        async def replay_batch(self, _job_id, *, after_seq):
            self.calls += 1
            return []

    async def connected():
        return False

    repo, store = TerminalRepository(), EmptyStore()
    monkeypatch.setattr(orchestrator_app, "_job_repository", lambda: repo)
    stream = orchestrator_app._committed_job_event_stream(
        SimpleNamespace(is_disconnected=connected), orchestrator_app.Job(id=job_id, created_at=1, state="running"), store, 4
    )

    async def drain():
        return [chunk async for chunk in stream]

    assert await asyncio.wait_for(drain(), 2.5) == []
    assert repo.calls == read_failures + 1
    assert store.calls == read_failures + 2  # Always replay once after terminal confirmation.
