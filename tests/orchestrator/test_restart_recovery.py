"""Phase 1.C - tests for the restart-recovery sweeper.

The sweeper itself runs against a ``JobRepository`` so the same
contract assertions apply to memory and Postgres. The Postgres branch
auto-skips when ``TP_TEST_POSTGRES_URL`` is unset.
"""

from __future__ import annotations

from typing import Tuple

import pytest

from transformation_portal.orchestrator import (
    JobEventStore,
    JobRecord,
    JobRepository,
)
from transformation_portal.orchestrator.recovery import (
    WORKER_LOST_REASON_CODE,
    sweep_orphaned_jobs,
)
from transformation_portal.orchestrator.runtime_handles import RuntimeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

RepoAndEvents = Tuple[JobRepository, JobEventStore]


def _record(job_id: str, *, created_at: float = 1.0) -> JobRecord:
    return JobRecord(id=job_id, created_at=created_at)


async def test_sweep_marks_running_and_queued_jobs_failed(
    repository_and_events: RepoAndEvents,
) -> None:
    """At startup with no live workers, every active job must be marked failed."""
    repo, _ = repository_and_events
    await repo.create(_record("queued-1"))
    await repo.create(_record("queued-2"))
    await repo.create(_record("running-1"))
    await repo.update("running-1", state="running", started_at=2.0)
    await repo.create(_record("running-2"))
    await repo.update("running-2", state="running", started_at=3.0)
    # A pre-existing terminal job must not be touched.
    await repo.create(_record("done-1"))
    await repo.update("done-1", state="succeeded", finished_at=4.0, done_published_at=4.0)

    empty_registry = RuntimeRegistry()
    swept = await sweep_orphaned_jobs(repo, runtime_registry=empty_registry)

    assert set(swept) == {"queued-1", "queued-2", "running-1", "running-2"}

    for jid in swept:
        rec = await repo.get(jid)
        assert rec is not None
        assert rec.state == "failed"
        assert rec.finished_at is not None
        assert rec.done_published_at is not None
        assert rec.last_event_at is not None
        assert rec.error == {
            "code": WORKER_LOST_REASON_CODE,
            "message": "Process did not survive backend restart.",
        }

    done = await repo.get("done-1")
    assert done is not None
    assert done.state == "succeeded"
    assert done.error is None


async def test_sweep_excludes_jobs_with_live_workers(
    repository_and_events: RepoAndEvents,
) -> None:
    """Jobs whose ids are in ``runtime_registry.live_job_ids`` must survive."""
    repo, _ = repository_and_events
    await repo.create(_record("alive"))
    await repo.update("alive", state="running", started_at=1.0)
    await repo.create(_record("dead"))
    await repo.update("dead", state="running", started_at=2.0)

    class _FakeRegistry:
        def live_job_ids(self) -> list[str]:
            return ["alive"]

    swept = await sweep_orphaned_jobs(repo, runtime_registry=_FakeRegistry())  # type: ignore[arg-type]

    assert swept == ["dead"]
    alive = await repo.get("alive")
    assert alive is not None and alive.state == "running"


async def test_sweep_is_noop_when_no_active_jobs(
    repository_and_events: RepoAndEvents,
) -> None:
    """Sweeping a repository with no queued/running rows returns an empty list."""
    repo, _ = repository_and_events
    await repo.create(_record("done-only"))
    await repo.update("done-only", state="succeeded", finished_at=1.0)

    swept = await sweep_orphaned_jobs(repo, runtime_registry=RuntimeRegistry())
    assert swept == []


async def test_sweep_is_idempotent(
    repository_and_events: RepoAndEvents,
) -> None:
    """A second sweep right after the first must touch nothing."""
    repo, _ = repository_and_events
    await repo.create(_record("orphan"))
    await repo.update("orphan", state="running", started_at=1.0)

    first = await sweep_orphaned_jobs(repo, runtime_registry=RuntimeRegistry())
    assert first == ["orphan"]

    second = await sweep_orphaned_jobs(repo, runtime_registry=RuntimeRegistry())
    assert second == []

    rec = await repo.get("orphan")
    assert rec is not None and rec.state == "failed"


async def test_sweep_uses_default_registry_when_none_passed(
    repository_and_events: RepoAndEvents,
) -> None:
    """The production code path - no explicit registry - must work too.

    The default registry (``get_runtime_registry``) is the process
    singleton; tests cannot easily reset it without affecting other
    tests, so we keep this as a smoke test that asserts the call shape
    and that a freshly-created job not present in the default registry
    is treated as orphaned.
    """
    repo, _ = repository_and_events
    await repo.create(_record("default-registry-orphan"))
    await repo.update("default-registry-orphan", state="queued")

    swept = await sweep_orphaned_jobs(repo)
    # The default singleton may contain unrelated entries from earlier
    # tests, but our just-created job_id is not in it.
    assert "default-registry-orphan" in swept
