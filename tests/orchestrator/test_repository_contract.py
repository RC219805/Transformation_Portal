"""Contract tests shared across every ``JobRepository`` backend.

The fixtures in ``conftest.py`` parametrize over registered backends; the
contract assertions here run identically against each. Memory is always
included; Postgres activates when ``TP_TEST_POSTGRES_URL`` is set.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Tuple

import pytest

from transformation_portal.orchestrator import (
    JobEventStore,
    JobNotFoundError,
    JobRecord,
    JobRepository,
    RepositoryError,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

# Convenience type alias so tests read cleanly.
RepoAndEvents = Tuple[JobRepository, JobEventStore]


def _new_record(job_id: str = "job-1", *, created_at: float = 1.0) -> JobRecord:
    return JobRecord(id=job_id, created_at=created_at)


class _AppendLogOnlyRepository(JobRepository):
    """Minimal concrete repository used to exercise the base append_logs default."""

    def __init__(self) -> None:
        self.appended: list[tuple[str, str, int]] = []

    async def create(self, record: JobRecord) -> None:
        return None

    async def get(self, job_id: str) -> JobRecord | None:
        return None

    async def list(self, *, limit: int | None = None) -> tuple[list[JobRecord], int]:
        return [], 0

    async def update(self, job_id: str, **fields: Any) -> JobRecord:
        raise JobNotFoundError(job_id)

    async def append_log(self, job_id: str, line: str, *, tail_limit: int) -> None:
        self.appended.append((job_id, line, tail_limit))

    async def set_artifacts(
        self,
        job_id: str,
        artifacts: dict[str, Any],
        artifact_lookup: dict[str, Path],
    ) -> None:
        return None

    async def delete(self, job_id: str) -> None:
        return None

    async def cleanup_expired(self, now: float, retention_seconds: float) -> list[str]:
        return []

    async def count_active(self) -> int:
        return 0

    async def sweep_orphaned(
        self,
        *,
        live_job_ids: list[str] | None = None,
        reason_code: str = "worker_lost_on_restart",
        now: float | None = None,
    ) -> list[str]:
        return []

    async def reset(self) -> None:
        self.appended.clear()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_create_then_get_round_trip(repository_and_events: RepoAndEvents) -> None:
    repo, _ = repository_and_events
    record = _new_record("job-A")
    record.request = {"pipeline": "demo"}
    await repo.create(record)

    fetched = await repo.get("job-A")
    assert fetched is not None
    assert fetched.id == "job-A"
    assert fetched.created_at == 1.0
    assert fetched.state == "queued"
    assert fetched.request == {"pipeline": "demo"}


async def test_repository_isolates_nested_mutable_record_fields(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    record = _new_record("job-nested-copy")
    record.request = {"args": {"quality": "premium"}}
    record.effective_request = {"args": {"resolved_backend": "da3"}}
    record.artifacts = {"items": [{"relative_path": "report.json"}]}
    record.run_summary = {"counts": {"succeeded": 1}}
    record.error = {"details": {"code": "original"}}

    await repo.create(record)
    record.request["args"]["quality"] = "mutated"
    record.effective_request["args"]["resolved_backend"] = "mutated"
    record.artifacts["items"][0]["relative_path"] = "mutated.json"
    record.run_summary["counts"]["succeeded"] = 99
    record.error["details"]["code"] = "mutated"

    fetched = await repo.get("job-nested-copy")
    assert fetched is not None
    assert fetched.request == {"args": {"quality": "premium"}}
    assert fetched.effective_request == {"args": {"resolved_backend": "da3"}}
    assert fetched.artifacts == {"items": [{"relative_path": "report.json"}]}
    assert fetched.run_summary == {"counts": {"succeeded": 1}}
    assert fetched.error == {"details": {"code": "original"}}

    fetched.request["args"]["quality"] = "fetched-mutated"
    fetched.effective_request["args"]["resolved_backend"] = "fetched-mutated"
    fetched.artifacts["items"][0]["relative_path"] = "fetched-mutated.json"
    fetched.run_summary["counts"]["succeeded"] = 100
    assert fetched.error is not None
    fetched.error["details"]["code"] = "fetched-mutated"

    refetched = await repo.get("job-nested-copy")
    assert refetched is not None
    assert refetched.request == {"args": {"quality": "premium"}}
    assert refetched.effective_request == {"args": {"resolved_backend": "da3"}}
    assert refetched.artifacts == {"items": [{"relative_path": "report.json"}]}
    assert refetched.run_summary == {"counts": {"succeeded": 1}}
    assert refetched.error == {"details": {"code": "original"}}


async def test_get_missing_returns_none(repository_and_events: RepoAndEvents) -> None:
    repo, _ = repository_and_events
    assert await repo.get("nope") is None


async def test_create_duplicate_raises(repository_and_events: RepoAndEvents) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-dup"))
    with pytest.raises(RepositoryError):
        await repo.create(_new_record("job-dup"))


async def test_list_orders_by_created_at_desc(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("old", created_at=10.0))
    await repo.create(_new_record("newer", created_at=20.0))
    await repo.create(_new_record("newest", created_at=30.0))

    records, total = await repo.list()
    assert total == 3
    assert [r.id for r in records] == ["newest", "newer", "old"]


async def test_list_respects_limit(repository_and_events: RepoAndEvents) -> None:
    repo, _ = repository_and_events
    for i in range(5):
        await repo.create(_new_record(f"job-{i}", created_at=float(i)))
    records, total = await repo.list(limit=2)
    assert total == 5
    assert len(records) == 2


# ---------------------------------------------------------------------------
# State transitions and partial updates
# ---------------------------------------------------------------------------


async def test_update_patches_named_fields(repository_and_events: RepoAndEvents) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-up"))
    refreshed = await repo.update("job-up", state="running", started_at=5.0)
    assert refreshed.state == "running"
    assert refreshed.started_at == 5.0
    fetched = await repo.get("job-up")
    assert fetched is not None
    assert fetched.state == "running"


async def test_update_without_fields_returns_isolated_current_record(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    record = _new_record("job-update-noop-copy")
    record.request = {"args": {"quality": "premium"}}
    record.run_summary = {"counts": {"succeeded": 1}}
    await repo.create(record)

    refreshed = await repo.update("job-update-noop-copy")

    assert refreshed.request == {"args": {"quality": "premium"}}
    assert refreshed.run_summary == {"counts": {"succeeded": 1}}

    refreshed.request["args"]["quality"] = "mutated"
    refreshed.run_summary["counts"]["succeeded"] = 99

    refetched = await repo.get("job-update-noop-copy")
    assert refetched is not None
    assert refetched.request == {"args": {"quality": "premium"}}
    assert refetched.run_summary == {"counts": {"succeeded": 1}}


async def test_update_isolates_nested_mutable_fields(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-update-nested-copy"))
    request = {"args": {"quality": "premium"}}
    effective_request = {"args": {"resolved_backend": "da3"}}
    artifacts = {"items": [{"relative_path": "report.json"}]}
    run_summary = {"counts": {"succeeded": 1}}
    error = {"details": {"code": "original"}}
    logs_tail = ["line-1"]

    refreshed = await repo.update(
        "job-update-nested-copy",
        request=request,
        effective_request=effective_request,
        artifacts=artifacts,
        run_summary=run_summary,
        error=error,
        logs_tail=logs_tail,
    )
    request["args"]["quality"] = "mutated"
    effective_request["args"]["resolved_backend"] = "mutated"
    artifacts["items"][0]["relative_path"] = "mutated.json"
    run_summary["counts"]["succeeded"] = 99
    error["details"]["code"] = "mutated"
    logs_tail.append("mutated")
    refreshed.request["args"]["quality"] = "refreshed-mutated"
    refreshed.effective_request["args"]["resolved_backend"] = "refreshed-mutated"
    refreshed.artifacts["items"][0]["relative_path"] = "refreshed-mutated.json"
    refreshed.run_summary["counts"]["succeeded"] = 100
    assert refreshed.error is not None
    refreshed.error["details"]["code"] = "refreshed-mutated"
    refreshed.logs_tail.append("refreshed-mutated")

    refetched = await repo.get("job-update-nested-copy")

    assert refetched is not None
    assert refetched.request == {"args": {"quality": "premium"}}
    assert refetched.effective_request == {"args": {"resolved_backend": "da3"}}
    assert refetched.artifacts == {"items": [{"relative_path": "report.json"}]}
    assert refetched.run_summary == {"counts": {"succeeded": 1}}
    assert refetched.error == {"details": {"code": "original"}}
    assert refetched.logs_tail == ["line-1"]


async def test_update_missing_raises_job_not_found(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    with pytest.raises(JobNotFoundError):
        await repo.update("ghost", state="running")


async def test_update_without_fields_missing_raises_job_not_found(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    with pytest.raises(JobNotFoundError):
        await repo.update("ghost-noop")


async def test_update_rejects_unknown_field(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-bad"))
    with pytest.raises(RepositoryError):
        await repo.update("job-bad", bogus="value")  # type: ignore[arg-type]


async def test_update_rejects_id_and_created_at(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-locked"))
    with pytest.raises(RepositoryError):
        await repo.update("job-locked", id="other")
    with pytest.raises(RepositoryError):
        await repo.update("job-locked", created_at=999.0)


async def test_update_rejects_artifact_lookup_mutation(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-artifact-lookup-locked"))

    with pytest.raises(RepositoryError):
        await repo.update(
            "job-artifact-lookup-locked",
            artifact_lookup={"out/report.json": Path("/abs/out/report.json")},
        )


# ---------------------------------------------------------------------------
# Log tail
# ---------------------------------------------------------------------------


async def test_append_log_rotates_tail(repository_and_events: RepoAndEvents) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-log"))
    for i in range(10):
        await repo.append_log("job-log", f"line-{i}", tail_limit=4)
    fetched = await repo.get("job-log")
    assert fetched is not None
    assert fetched.logs_tail == ["line-6", "line-7", "line-8", "line-9"]


async def test_append_logs_rotates_tail(repository_and_events: RepoAndEvents) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-log-batch"))
    await repo.append_logs("job-log-batch", [f"line-{i}" for i in range(10)], tail_limit=4)
    fetched = await repo.get("job-log-batch")
    assert fetched is not None
    assert fetched.logs_tail == ["line-6", "line-7", "line-8", "line-9"]


async def test_default_append_logs_delegates_to_append_log_for_compatibility() -> None:
    repo = _AppendLogOnlyRepository()

    await repo.append_logs("job-default-append", ["line-1", "line-2"], tail_limit=4)

    assert repo.appended == [
        ("job-default-append", "line-1", 4),
        ("job-default-append", "line-2", 4),
    ]


@pytest.mark.parametrize("bad_tail_limit", [0, -1])
async def test_default_append_logs_rejects_non_positive_tail_limit_before_delegation(
    bad_tail_limit: int,
) -> None:
    repo = _AppendLogOnlyRepository()

    with pytest.raises(RepositoryError):
        await repo.append_logs("job-default-append", ["line"], tail_limit=bad_tail_limit)

    assert repo.appended == []


async def test_append_logs_under_limit_preserves_all_lines(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-log-batch-under-limit"))

    await repo.append_logs("job-log-batch-under-limit", ["line-1", "line-2"], tail_limit=4)

    fetched = await repo.get("job-log-batch-under-limit")
    assert fetched is not None
    assert fetched.logs_tail == ["line-1", "line-2"]


async def test_append_logs_rejects_zero_or_negative_limit(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-log-batch-limit"))

    with pytest.raises(RepositoryError):
        await repo.append_logs("job-log-batch-limit", ["x"], tail_limit=0)
    with pytest.raises(RepositoryError):
        await repo.append_logs("job-log-batch-limit", ["x"], tail_limit=-1)


async def test_append_logs_empty_batch_is_noop(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-log-empty-batch"))
    await repo.append_log("job-log-empty-batch", "existing", tail_limit=4)

    await repo.append_logs("job-log-empty-batch", [], tail_limit=4)

    fetched = await repo.get("job-log-empty-batch")
    assert fetched is not None
    assert fetched.logs_tail == ["existing"]


async def test_append_logs_missing_job_raises_job_not_found(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events

    with pytest.raises(JobNotFoundError):
        await repo.append_logs("missing-log-batch", ["line"], tail_limit=4)


@pytest.mark.parametrize("bad_tail_limit", [0, -1])
async def test_append_log_rejects_zero_or_negative_limit(
    repository_and_events: RepoAndEvents,
    bad_tail_limit: int,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-log0"))
    with pytest.raises(RepositoryError):
        await repo.append_log("job-log0", "x", tail_limit=bad_tail_limit)


async def test_append_log_missing_job_raises_job_not_found(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events

    with pytest.raises(JobNotFoundError):
        await repo.append_log("missing-log", "line", tail_limit=4)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


async def test_set_artifacts_replaces_index_and_lookup(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-art"))
    await repo.set_artifacts(
        "job-art",
        artifacts={"items": [{"path": "out/x.png"}]},
        artifact_lookup={"out/x.png": Path("/abs/out/x.png")},
    )
    fetched = await repo.get("job-art")
    assert fetched is not None
    assert fetched.artifacts == {"items": [{"path": "out/x.png"}]}
    assert fetched.artifact_lookup == {"out/x.png": Path("/abs/out/x.png")}


async def test_set_artifacts_isolates_nested_mutable_artifacts(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-artifact-nested-copy"))
    artifacts = {"items": [{"path": "out/x.png", "metadata": {"kind": "preview"}}]}

    await repo.set_artifacts(
        "job-artifact-nested-copy",
        artifacts=artifacts,
        artifact_lookup={"out/x.png": Path("/abs/out/x.png")},
    )
    artifacts["items"][0]["metadata"]["kind"] = "mutated"

    fetched = await repo.get("job-artifact-nested-copy")
    assert fetched is not None
    fetched.artifacts["items"][0]["metadata"]["kind"] = "fetched-mutated"

    refetched = await repo.get("job-artifact-nested-copy")
    assert refetched is not None
    assert refetched.artifacts == {"items": [{"path": "out/x.png", "metadata": {"kind": "preview"}}]}
    assert refetched.artifact_lookup == {"out/x.png": Path("/abs/out/x.png")}


async def test_set_artifacts_missing_job_raises_job_not_found(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events

    with pytest.raises(JobNotFoundError):
        await repo.set_artifacts(
            "missing-artifacts",
            artifacts={"items": []},
            artifact_lookup={},
        )


# ---------------------------------------------------------------------------
# Cleanup, count, delete
# ---------------------------------------------------------------------------


async def test_cleanup_expired_removes_finished_old_jobs(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("fresh", created_at=190.0))
    await repo.update("fresh", finished_at=190.0, state="succeeded")
    await repo.create(_new_record("stale", created_at=50.0))
    await repo.update("stale", finished_at=50.0, state="succeeded")

    removed = await repo.cleanup_expired(now=200.0, retention_seconds=100.0)
    assert set(removed) == {"stale"}
    assert await repo.get("stale") is None
    assert await repo.get("fresh") is not None


async def test_count_active_counts_only_queued_and_running(
    repository_and_events: RepoAndEvents,
) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("q"))  # queued by default
    await repo.create(_new_record("r"))
    await repo.update("r", state="running")
    await repo.create(_new_record("ok"))
    await repo.update("ok", state="succeeded", finished_at=1.0)
    assert await repo.count_active() == 2


async def test_delete_is_idempotent(repository_and_events: RepoAndEvents) -> None:
    repo, _ = repository_and_events
    await repo.create(_new_record("job-del"))
    await repo.delete("job-del")
    assert await repo.get("job-del") is None
    # Deleting again must not raise.
    await repo.delete("job-del")
    await repo.delete("never-existed")


# ---------------------------------------------------------------------------
# Restart sweeper (used by Layer 1.C)
# ---------------------------------------------------------------------------


async def test_sweep_orphaned_marks_active_jobs_worker_lost(
    repository_and_events: RepoAndEvents,
) -> None:
    """Phase 2.D: sweep produces ``state=worker_lost`` (distinct from
    ``failed``) and a ``retriable=True`` flag on the error envelope.

    ``failed`` is reserved for executor-level errors (RUNNER_EXIT_NONZERO,
    RUNNER_NOT_FOUND, etc.) where the work itself is broken;
    ``worker_lost`` signals the worker died with the lease held and the
    job can safely be re-admitted.
    """
    repo, _ = repository_and_events
    await repo.create(_new_record("alive"))
    await repo.update("alive", state="running", started_at=1.0)
    await repo.create(_new_record("dead"))
    await repo.update("dead", state="running", started_at=2.0)
    await repo.create(_new_record("queued"))
    await repo.create(_new_record("done"))
    await repo.update("done", state="succeeded", finished_at=3.0)

    swept = await repo.sweep_orphaned(live_job_ids=["alive"], now=10.0)
    assert set(swept) == {"dead", "queued"}

    fetched = await repo.get("dead")
    assert fetched is not None
    assert fetched.state == "worker_lost"
    assert fetched.finished_at == 10.0
    assert fetched.done_published_at == 10.0
    assert fetched.error == {
        "code": "worker_lost_on_restart",
        "message": "Process did not survive backend restart.",
        "retriable": True,
    }

    fetched_alive = await repo.get("alive")
    assert fetched_alive is not None
    assert fetched_alive.state == "running"

    fetched_done = await repo.get("done")
    assert fetched_done is not None
    assert fetched_done.state == "succeeded"


# ---------------------------------------------------------------------------
# Concurrent updates
# ---------------------------------------------------------------------------


async def test_concurrent_append_log_does_not_lose_writes(
    repository_and_events: RepoAndEvents,
) -> None:
    """All concurrently-appended log lines must end up in ``logs_tail``.

    Replaces a weaker progress-race test that could pass even if every
    update was silently dropped (default ``progress`` is 0). ``append_log``
    has an accumulating contract: N concurrent appends with a generous
    ``tail_limit`` must produce exactly N entries.
    """
    repo, _ = repository_and_events
    await repo.create(_new_record("job-conc"))

    async def write(i: int) -> None:
        await repo.append_log("job-conc", f"line-{i}", tail_limit=1000)

    await asyncio.gather(*[write(i) for i in range(50)])
    fetched = await repo.get("job-conc")
    assert fetched is not None
    assert len(fetched.logs_tail) == 50
    # Ordering across the race is intentionally unspecified, but every
    # input line must be present exactly once - that is what proves no
    # writes were lost.
    assert set(fetched.logs_tail) == {f"line-{i}" for i in range(50)}


async def test_concurrent_updates_yield_a_consistent_final_value(
    repository_and_events: RepoAndEvents,
) -> None:
    """A race of ``update(progress=N)`` calls must end on one of the N values.

    This is the weaker "last-writer-wins ends consistently" property:
    we don't assert which value wins, but we do assert the final value
    is exactly one of the inputs (i.e., the row was not corrupted and
    a write actually landed). The stronger no-lost-writes property is
    covered by ``test_concurrent_append_log_does_not_lose_writes``.
    """
    repo, _ = repository_and_events
    await repo.create(_new_record("job-final"))

    async def bump(progress: int) -> None:
        await repo.update("job-final", progress=progress)

    inputs = list(range(1, 51))
    await asyncio.gather(*[bump(p) for p in inputs])
    fetched = await repo.get("job-final")
    assert fetched is not None
    assert fetched.progress in inputs


# ---------------------------------------------------------------------------
# Event store
# ---------------------------------------------------------------------------


async def test_event_store_appends_monotonic_seq(
    repository_and_events: RepoAndEvents,
) -> None:
    _, events = repository_and_events
    e1 = await events.append("job-e", "state", {"state": "queued"}, created_at=1.0)
    e2 = await events.append("job-e", "state", {"state": "running"}, created_at=2.0)
    e3 = await events.append("job-e", "log", {"line": "hello"}, created_at=3.0)
    assert [e1.seq, e2.seq, e3.seq] == [1, 2, 3]


async def test_event_store_events_since_filters_by_seq(
    repository_and_events: RepoAndEvents,
) -> None:
    _, events = repository_and_events
    await events.append("job-f", "a", {"i": 0}, created_at=1.0)
    await events.append("job-f", "a", {"i": 1}, created_at=2.0)
    await events.append("job-f", "a", {"i": 2}, created_at=3.0)
    collected = [e async for e in events.events_since("job-f", after_seq=1)]
    assert [e.payload["i"] for e in collected] == [1, 2]


async def test_event_store_payload_is_stable_after_append(
    repository_and_events: RepoAndEvents,
) -> None:
    _, events = repository_and_events
    payload = {"state": "running", "nested": {"progress": 10}}

    appended = await events.append("job-payload-copy", "state", payload, created_at=1.0)
    payload["nested"]["progress"] = 88
    payload["state"] = "mutated"
    payload["nested"] = {"progress": 99}

    replay = [event async for event in events.events_since("job-payload-copy")]

    assert appended.payload == {"state": "running", "nested": {"progress": 10}}
    assert [event.payload for event in replay] == [{"state": "running", "nested": {"progress": 10}}]

    appended.payload["nested"]["progress"] = 77
    replay[0].payload["nested"]["progress"] = 66
    replay_again = [event async for event in events.events_since("job-payload-copy")]

    assert [event.payload for event in replay_again] == [{"state": "running", "nested": {"progress": 10}}]


async def test_event_store_replay_orders_by_assigned_seq_not_timestamp(
    repository_and_events: RepoAndEvents,
) -> None:
    _, events = repository_and_events
    await events.append("job-event-order", "latest-time", {"order": 1}, created_at=30.0)
    await events.append("job-event-order", "earliest-time", {"order": 2}, created_at=10.0)
    await events.append("job-event-order", "middle-time", {"order": 3}, created_at=20.0)

    replay = [event async for event in events.events_since("job-event-order")]

    assert [(event.seq, event.event_type, event.created_at) for event in replay] == [
        (1, "latest-time", 30.0),
        (2, "earliest-time", 10.0),
        (3, "middle-time", 20.0),
    ]


async def test_event_store_isolates_jobs(
    repository_and_events: RepoAndEvents,
) -> None:
    _, events = repository_and_events
    await events.append("job-x", "a", {}, created_at=1.0)
    await events.append("job-y", "a", {}, created_at=2.0)
    x_events = [e async for e in events.events_since("job-x")]
    y_events = [e async for e in events.events_since("job-y")]
    assert len(x_events) == 1
    assert len(y_events) == 1
    assert x_events[0].seq == 1
    assert y_events[0].seq == 1


async def test_event_store_accepts_arbitrary_job_id(
    repository_and_events: RepoAndEvents,
) -> None:
    """The event store contract does not require a parent job row.

    Memory and Postgres backends must both accept appends for any
    job_id, including ids that were never created in the repository.
    This pins down the contract because the Postgres impl deliberately
    omits a SQL FK on ``job_events.job_id``.
    """
    repo, events = repository_and_events
    # No `await repo.create(...)` for "phantom-job".
    appended = await events.append("phantom-job", "state", {"state": "queued"}, created_at=1.0)
    assert appended.seq == 1
    replay = [e async for e in events.events_since("phantom-job")]
    assert [e.payload for e in replay] == [{"state": "queued"}]


async def test_delete_removes_job_from_repository(
    repository_and_events: RepoAndEvents,
) -> None:
    """``repo.delete`` must remove the job from the repository on every backend.

    This is the cross-backend slice of the contract. The Postgres-only
    cascade-of-events semantic is pinned in
    ``test_postgres_delete_job_cascades_events`` below.
    """
    repo, events = repository_and_events
    await repo.create(_new_record("job-cascade"))
    await events.append("job-cascade", "state", {"v": 1}, created_at=1.0)
    await events.append("job-cascade", "state", {"v": 2}, created_at=2.0)

    await repo.delete("job-cascade")
    assert await repo.get("job-cascade") is None


async def test_postgres_delete_job_cascades_events(
    repository_and_events: RepoAndEvents,
) -> None:
    """Postgres backend only: ``repo.delete`` cascades event deletion.

    The memory backend's event store is independently keyed and does
    not cascade; that behavior is intentional and is covered by the
    cross-backend ``test_delete_removes_job_from_repository``. This
    test pins the Postgres-specific cascade that ``PostgresJobRepository
    .delete`` does manually (the SQL FK on ``job_events.job_id`` was
    deliberately dropped; see models.py).
    """
    repo, events = repository_and_events
    if not _is_postgres_backend(repo):
        pytest.skip("postgres-only contract: cascade-on-delete in events table")

    await repo.create(_new_record("job-pg-cascade"))
    await events.append("job-pg-cascade", "state", {"v": 1}, created_at=1.0)
    await events.append("job-pg-cascade", "state", {"v": 2}, created_at=2.0)

    await repo.delete("job-pg-cascade")
    remaining = [e async for e in events.events_since("job-pg-cascade")]
    assert remaining == []


def _is_postgres_backend(repo: JobRepository) -> bool:
    """Return True iff ``repo`` is the Postgres backend.

    Uses ``isinstance`` rather than ``type(...).__name__`` so renames or
    subclasses don't silently change behavior. The import is guarded
    because ``sqlalchemy`` may not be installed in the offline lane.
    """
    try:
        from transformation_portal.orchestrator.storage.postgres import (
            PostgresJobRepository,
        )
    except ImportError:
        return False
    return isinstance(repo, PostgresJobRepository)


# ---------------------------------------------------------------------------
# Memory backend - boundary-condition unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_cap", [0, -1, -100])
def test_memory_event_store_rejects_non_positive_per_job_cap(bad_cap: int) -> None:
    """Constructing a memory event store with a non-positive cap must fail loud.

    A ``deque(maxlen=0)`` would silently drop every appended event while
    still incrementing the seq counter, which would break SSE replay in
    an extremely non-obvious way. The constructor refuses up front so the
    contract violation is caught at construction time, not at first read.
    """
    from transformation_portal.orchestrator.storage.memory import MemoryJobEventStore

    with pytest.raises(RepositoryError):
        MemoryJobEventStore(per_job_cap=bad_cap)


async def test_memory_event_store_cap_retains_latest_events_with_monotonic_seq() -> None:
    """Bounded replay keeps newest events without resetting durable seq ids."""
    from transformation_portal.orchestrator.storage.memory import MemoryJobEventStore

    events = MemoryJobEventStore(per_job_cap=2)

    first = await events.append("job-cap", "state", {"step": "queued"}, created_at=1.0)
    second = await events.append("job-cap", "progress", {"step": "running"}, created_at=2.0)
    third = await events.append("job-cap", "done", {"step": "succeeded"}, created_at=3.0)

    assert [first.seq, second.seq, third.seq] == [1, 2, 3]

    replay_all = [event async for event in events.events_since("job-cap")]
    assert [(event.seq, event.event_type, event.payload["step"]) for event in replay_all] == [
        (2, "progress", "running"),
        (3, "done", "succeeded"),
    ]

    replay_after_oldest_retained = [event async for event in events.events_since("job-cap", after_seq=2)]
    assert [(event.seq, event.event_type) for event in replay_after_oldest_retained] == [(3, "done")]


async def test_memory_event_store_reset_clears_events_and_restarts_seq() -> None:
    """Reset must fully clear event history for hermetic test/runtime restarts."""
    from transformation_portal.orchestrator.storage.memory import MemoryJobEventStore

    events = MemoryJobEventStore()
    await events.append("job-reset", "state", {"state": "queued"}, created_at=1.0)

    await events.reset()
    replay_after_reset = [event async for event in events.events_since("job-reset")]
    next_event = await events.append("job-reset", "state", {"state": "queued-again"}, created_at=2.0)

    assert replay_after_reset == []
    assert next_event.seq == 1
