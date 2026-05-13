"""Pessimistic restart recovery for the orchestrator.

Phase 1.C wires the existing ``JobRepository.sweep_orphaned`` into the
FastAPI lifespan: on every startup, any job that the repository still
records as ``queued`` or ``running`` but that no live worker in this
process is executing is marked ``worker_lost`` (Phase 2.D state)
with an explicit ``worker_lost_on_restart`` error carrying
``retriable=True``. SSE clients reconnecting after a restart see a
terminal ``done`` event rather than hanging on a state that can no
longer make progress, and operator tooling can distinguish the
recovered jobs from executor-level failures.

Memory backend: the repository is per-process so it is empty at startup
and the sweeper is a deterministic no-op. Postgres backend (Phase 1.B):
the repository persists across restarts and the sweeper does the real
work.

The runtime registry (``runtime_handles``) tracks live ``proc`` /
``terminate_task`` entries within the current process. At startup that
registry is empty, which is exactly the contract the sweeper expects:
any active row with no live worker is by definition orphaned.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol

from transformation_portal.orchestrator.runtime_handles import get_runtime_registry
from transformation_portal.orchestrator.storage.base import JobRepository

logger = logging.getLogger(__name__)

WORKER_LOST_REASON_CODE = "worker_lost_on_restart"

# How many swept ids to include in the WARNING-level log line. The full
# list is still emitted at DEBUG so operators can opt in via log level
# without paying the cost (or polluting the warning channel) on every
# restart of a Postgres backend that's been offline for a while.
_SWEPT_LOG_SAMPLE_LIMIT = 20


class LiveJobIdsProvider(Protocol):
    """Minimal contract the sweeper needs from a runtime registry.

    Anything with ``live_job_ids() -> list[str]`` satisfies it. Today
    that's ``RuntimeRegistry``; Phase 2's Redis-lease-backed registry
    will satisfy the same Protocol without the sweeper changing.
    """

    def live_job_ids(self) -> List[str]: ...


async def sweep_orphaned_jobs(
    repository: JobRepository,
    *,
    runtime_registry: Optional[LiveJobIdsProvider] = None,
    reason_code: str = WORKER_LOST_REASON_CODE,
) -> List[str]:
    """Mark any active job without a live worker as ``worker_lost``.

    Returns the list of swept ids so callers (or tests) can log /
    metric the recovery without re-querying. The error payload is
    fixed by the contract:

        {"code": "worker_lost_on_restart",
         "message": "Process did not survive backend restart.",
         "retriable": True}

    ``state`` is ``worker_lost`` (Phase 2.D — distinct from ``failed``
    so callers can distinguish executor-level failures from worker
    death), and ``last_event_at`` / ``finished_at`` /
    ``done_published_at`` are all set to the same timestamp so SSE
    late-clients see a terminal ``done`` event.

    The optional ``runtime_registry`` lets callers pass an alternative
    registry (test isolation). In production, the default singleton is
    used: at process startup it is empty by construction, so every
    active row in the repository is treated as orphaned.
    """
    registry = runtime_registry if runtime_registry is not None else get_runtime_registry()
    live = registry.live_job_ids()

    swept = await repository.sweep_orphaned(live_job_ids=live, reason_code=reason_code)

    if swept:
        sorted_ids = sorted(swept)
        sample = sorted_ids[:_SWEPT_LOG_SAMPLE_LIMIT]
        logger.warning(
            "orchestrator restart recovery marked %d orphaned job(s) as worker_lost " "(sample of up to %d: %s)",
            len(swept),
            _SWEPT_LOG_SAMPLE_LIMIT,
            sample,
        )
        # Full list goes to DEBUG so operators can opt in via log level
        # without flooding the warning channel.
        logger.debug(
            "orchestrator restart recovery: full swept id list (%d): %s",
            len(swept),
            sorted_ids,
        )
    else:
        logger.info(
            "orchestrator restart recovery: no orphaned jobs (live=%d)",
            len(live),
        )

    return swept


__all__ = ["WORKER_LOST_REASON_CODE", "sweep_orphaned_jobs"]
