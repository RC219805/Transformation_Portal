"""Phase 2.A - shared contract tests for every ``QueueBroker`` backend.

Memory backend is always included; the Redis backend (Phase 2.B)
will activate when ``TP_TEST_REDIS_URL`` is set, mirroring the
Phase 1.B Postgres pattern.

Coverage:

- enqueue / acquire round-trip + FIFO order
- duplicate enqueue rejection (admission-collision detection)
- lease deadlines pin in-flight ownership
- heartbeat-extension contract (only the holder can extend)
- pre-lease cancellation drops the queue entry
- in-flight cancellation surfaces via ``LeaseStatus.cancelled``
- lease-expiry sweeper re-queues abandoned jobs
- worker runner end-to-end through the broker (1 step,
  placeholder executor) using the in-memory backend.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio

from transformation_portal.orchestrator.queue import (
    JobEnqueueRequest,
    LeaseStatus,
    QueueBroker,
    QueueBrokerError,
    reset_singleton,
)
from transformation_portal.orchestrator.queue.base import LeaseNotHeldError
from transformation_portal.orchestrator.queue.memory import MemoryQueueBroker
from transformation_portal.orchestrator.worker import (
    WorkerConfig,
    WorkerRunner,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_REDIS_URL_ENV = "TP_TEST_REDIS_URL"


def _available_backends() -> list[str]:
    backends = ["memory"]
    if os.getenv(_REDIS_URL_ENV, "").strip():
        backends.append("redis")
    return backends


@pytest.fixture(params=_available_backends())
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest_asyncio.fixture
async def broker(backend: str, request: pytest.FixtureRequest) -> AsyncIterator[QueueBroker]:
    """Yield a freshly-reset ``QueueBroker`` for the parameterized backend.

    Redis instances use a per-test ``key_prefix`` so parallel test
    runs (pytest-xdist) and shared-tenant Redis deployments never
    collide and never touch keys outside the broker's namespace.
    """
    reset_singleton()
    if backend == "memory":
        instance: QueueBroker = MemoryQueueBroker()
    elif backend == "redis":
        try:
            from transformation_portal.orchestrator.queue.redis import RedisQueueBroker
        except ImportError:
            pytest.skip("redis backend module not available; expected in Phase 2.B")
        # Per-test key prefix isolates parallel runs and survives shared Redis.
        prefix = f"tp:test:{uuid.uuid4().hex[:12]}:"
        instance = RedisQueueBroker(
            redis_url=os.environ[_REDIS_URL_ENV],
            key_prefix=prefix,
        )
        await instance.reset()
    else:
        raise RuntimeError(f"unknown backend {backend!r}")
    try:
        yield instance
    finally:
        await instance.reset()
        await instance.close()
        reset_singleton()


def _request(job_id: str, *, argv: list[str] | None = None) -> JobEnqueueRequest:
    return JobEnqueueRequest(
        job_id=job_id,
        argv=argv if argv is not None else ["lux-depth-v3", "--input", "x", "--output", "y"],
        api_version="v1",
    )


# ---------------------------------------------------------------------------
# Enqueue + acquire
# ---------------------------------------------------------------------------


async def test_enqueue_then_acquire_round_trip(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-A"))
    assert await broker.queued_job_ids() == ["job-A"]

    lease = await broker.acquire_lease("worker-1", lease_seconds=10.0)
    assert lease is not None
    assert lease.job_id == "job-A"
    assert lease.worker_id == "worker-1"
    assert lease.deadline > 0
    assert lease.request.argv == ["lux-depth-v3", "--input", "x", "--output", "y"]

    assert await broker.queued_job_ids() == []
    assert await broker.leased_job_ids() == ["job-A"]


async def test_acquire_returns_none_on_empty_queue(broker: QueueBroker) -> None:
    assert await broker.acquire_lease("worker-1", lease_seconds=10.0) is None


async def test_enqueue_preserves_fifo_across_workers(broker: QueueBroker) -> None:
    for jid in ("job-1", "job-2", "job-3"):
        await broker.enqueue(_request(jid))

    leased = []
    for worker in ("worker-A", "worker-B", "worker-C"):
        lease = await broker.acquire_lease(worker, lease_seconds=10.0)
        assert lease is not None
        leased.append(lease.job_id)

    assert leased == ["job-1", "job-2", "job-3"]


async def test_duplicate_enqueue_raises(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-dup"))
    with pytest.raises(QueueBrokerError):
        await broker.enqueue(_request("job-dup"))


async def test_acquire_lease_rejects_non_positive_lease(broker: QueueBroker) -> None:
    with pytest.raises(QueueBrokerError):
        await broker.acquire_lease("worker", lease_seconds=0)
    with pytest.raises(QueueBrokerError):
        await broker.acquire_lease("worker", lease_seconds=-1)


# ---------------------------------------------------------------------------
# Heartbeat / extend
# ---------------------------------------------------------------------------


async def test_extend_lease_keeps_lease_active(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-hb"))
    lease = await broker.acquire_lease("worker-1", lease_seconds=10.0)
    assert lease is not None
    initial_deadline = lease.deadline
    # Sleep a tiny bit so monotonic time advances measurably.
    await asyncio.sleep(0.01)
    status = await broker.extend_lease("worker-1", "job-hb", lease_seconds=20.0)
    assert status is LeaseStatus.active
    # Lease moved further into the future.
    fresh = await broker.leased_job_ids()
    assert fresh == ["job-hb"]
    # initial_deadline + (20 - elapsed) should be greater than initial_deadline.
    # We can't reach inside without a getter; assert via reclaim no-op.
    reclaimed = await broker.reclaim_expired_leases(now=initial_deadline + 1.0)
    assert reclaimed == [], "lease should not have expired after the extension"


async def test_extend_lease_rejects_non_holder(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-other"))
    lease = await broker.acquire_lease("worker-A", lease_seconds=10.0)
    assert lease is not None
    with pytest.raises(LeaseNotHeldError):
        await broker.extend_lease("worker-B", "job-other", lease_seconds=10.0)


async def test_extend_lease_rejects_unknown_job(broker: QueueBroker) -> None:
    with pytest.raises(LeaseNotHeldError):
        await broker.extend_lease("worker-1", "ghost", lease_seconds=10.0)


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


async def test_release_lease_clears_in_flight_state(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-rel"))
    await broker.acquire_lease("worker-1", lease_seconds=10.0)
    await broker.release_lease("worker-1", "job-rel")
    assert await broker.leased_job_ids() == []
    assert await broker.queued_job_ids() == []
    # And a fresh enqueue with the same id is now allowed.
    await broker.enqueue(_request("job-rel"))


async def test_release_lease_is_idempotent(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-rel2"))
    await broker.acquire_lease("worker-1", lease_seconds=10.0)
    await broker.release_lease("worker-1", "job-rel2")
    # Releasing again must not raise.
    await broker.release_lease("worker-1", "job-rel2")
    # Releasing a job we never owned must not raise either.
    await broker.release_lease("worker-1", "never-existed")


# ---------------------------------------------------------------------------
# Lease expiry / reclaim
# ---------------------------------------------------------------------------


async def test_reclaim_expired_leases_requeues_abandoned_jobs(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-exp"))
    lease = await broker.acquire_lease("worker-A", lease_seconds=10.0)
    assert lease is not None

    # Pretend the worker died: sweep with a future "now" past the lease.
    reclaimed = await broker.reclaim_expired_leases(now=lease.deadline + 5.0)
    assert reclaimed == ["job-exp"]

    assert await broker.leased_job_ids() == []
    # The reclaimed job is back at the head of the queue.
    again = await broker.acquire_lease("worker-B", lease_seconds=10.0)
    assert again is not None
    assert again.job_id == "job-exp"


async def test_reclaim_no_op_when_no_leases(broker: QueueBroker) -> None:
    assert await broker.reclaim_expired_leases(now=time.monotonic() + 1000.0) == []


async def test_server_time_shares_domain_with_lease_deadline(broker: QueueBroker) -> None:
    """``server_time()`` and ``JobLease.deadline`` must share one clock.

    Production sweepers (Phase 2.D) pass ``await broker.server_time()``
    to ``reclaim_expired_leases`` and compare it against the deadlines
    written by ``acquire_lease``. Verify the two values live in the
    same time domain *without* sleeping: bracket the acquire call
    between two ``server_time()`` reads, then prove the resulting
    deadline falls inside ``[t_before + lease_seconds,
    t_after + lease_seconds]``. That window can only hold if both
    clocks measure the same thing.
    """
    lease_seconds = 30.0
    t_before = await broker.server_time()
    await broker.enqueue(_request("job-clock-domain"))
    lease = await broker.acquire_lease("worker-clock", lease_seconds=lease_seconds)
    assert lease is not None
    t_after = await broker.server_time()

    # server_time is non-decreasing across the call.
    assert t_after >= t_before
    # And the lease's deadline lives in the same time domain — it must be
    # bracketed by [t_before + lease, t_after + lease]. Any other clock
    # would put deadline outside this window.
    assert t_before + lease_seconds <= lease.deadline <= t_after + lease_seconds

    # A sweep at "now == deadline - epsilon" must NOT reclaim the lease
    # (deadline strictly in the future). A sweep at "now == deadline"
    # must reclaim it (the broker uses <= for expiry).
    not_yet = await broker.reclaim_expired_leases(now=lease.deadline - 1.0)
    assert "job-clock-domain" not in not_yet
    just_expired = await broker.reclaim_expired_leases(now=lease.deadline)
    assert "job-clock-domain" in just_expired


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_pre_lease_drops_queue_entry(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-cancel-pre"))
    assert await broker.queued_job_ids() == ["job-cancel-pre"]

    assert await broker.cancel("job-cancel-pre") is True
    # The cancelled job_id must be gone from the queue immediately,
    # not deferred to the next acquire. queued_job_ids reflects the
    # post-cancel reality.
    assert await broker.queued_job_ids() == []
    # The next acquire returns None (queue is empty).
    assert await broker.acquire_lease("worker-1", lease_seconds=10.0) is None
    # And a fresh enqueue with the same id is allowed without first
    # waiting for a worker to acquire (the slot was freed at cancel
    # time).
    await broker.enqueue(_request("job-cancel-pre"))
    second = await broker.acquire_lease("worker-1", lease_seconds=10.0)
    assert second is not None and second.job_id == "job-cancel-pre"


async def test_cancel_inflight_surfaces_via_extend_lease(broker: QueueBroker) -> None:
    await broker.enqueue(_request("job-cancel-inflight"))
    lease = await broker.acquire_lease("worker-1", lease_seconds=10.0)
    assert lease is not None

    assert await broker.cancel("job-cancel-inflight") is True
    status = await broker.extend_lease("worker-1", "job-cancel-inflight", lease_seconds=10.0)
    assert status is LeaseStatus.cancelled

    # The worker still holds the lease until it releases.
    assert await broker.leased_job_ids() == ["job-cancel-inflight"]
    await broker.release_lease("worker-1", "job-cancel-inflight")
    assert await broker.leased_job_ids() == []


async def test_cancel_unknown_job_returns_false(broker: QueueBroker) -> None:
    assert await broker.cancel("nope") is False


# ---------------------------------------------------------------------------
# Worker runner integration (memory broker only).
# ---------------------------------------------------------------------------


async def test_worker_runner_step_processes_one_job(broker: QueueBroker) -> None:
    """End-to-end: enqueue -> WorkerRunner.step -> lease released."""
    await broker.enqueue(_request("job-worker-1"))
    config = WorkerConfig(
        worker_id="worker-end-to-end",
        lease_seconds=2.0,
        heartbeat_interval_seconds=5.0,  # never fires inside the short test
        poll_interval_seconds=0.05,
    )
    runner = WorkerRunner(broker=broker, config=config)
    did_work = await runner.step()
    assert did_work is True
    assert await broker.leased_job_ids() == []
    assert await broker.queued_job_ids() == []


async def test_worker_runner_step_returns_false_on_empty_queue(broker: QueueBroker) -> None:
    config = WorkerConfig(worker_id="worker-empty")
    runner = WorkerRunner(broker=broker, config=config)
    assert await runner.step() is False
