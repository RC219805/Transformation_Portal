"""Unit coverage for the orchestrator ``WorkerRunner`` and supervisor loop.

The existing broker-dispatch suite drives the worker through the full FastAPI
orchestrator. These tests isolate ``worker.py`` against a hand-rolled fake
broker so the executor error branches, the heartbeat-loop cancellation paths,
the supervisor backoff loop, broker disposal, and the CLI entry point are all
exercised deterministically (no app, no network, sub-second).
"""

from __future__ import annotations

import asyncio
import signal
import time

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.orchestrator import worker as worker_module
from transformation_portal.orchestrator.queue.base import (
    JobEnqueueRequest,
    JobLease,
    LeaseNotHeldError,
    LeaseStatus,
)
from transformation_portal.orchestrator.worker import (
    CancelledByOrchestrator,
    RetryableExecutorUnavailable,
    WorkerConfig,
    WorkerRunner,
    _config_from_env,
    _default_executor,
    main,
    monotonic_now,
    run_worker_forever,
)


def _request(job_id: str = "job-1") -> JobEnqueueRequest:
    return JobEnqueueRequest(job_id=job_id, argv=["enhance", "--in", "x"])


def _lease(worker_id: str = "w", job_id: str = "job-1") -> JobLease:
    return JobLease(job_id=job_id, worker_id=worker_id, deadline=0.0, request=_request(job_id))


def _config(**overrides) -> WorkerConfig:
    base = dict(
        worker_id="w",
        lease_seconds=30.0,
        heartbeat_interval_seconds=100.0,  # never fires during fast executors
        poll_interval_seconds=0.01,
        max_poll_backoff_seconds=0.05,
    )
    base.update(overrides)
    return WorkerConfig(**base)


class FakeBroker:
    """Minimal duck-typed QueueBroker for driving WorkerRunner.step."""

    def __init__(self, *, leases=None, extend_results=None):
        # ``leases`` is consumed one acquire() at a time; None means empty queue.
        self._leases = list(leases or [])
        self._extend_results = list(extend_results or [])
        self.released: list[str] = []
        self.closed = False
        self.acquire_calls = 0
        self.on_acquire = None  # optional callback(acquire_calls)

    async def acquire_lease(self, worker_id, *, lease_seconds):
        self.acquire_calls += 1
        if self.on_acquire is not None:
            self.on_acquire(self.acquire_calls)
        if self._leases:
            return self._leases.pop(0)
        return None

    async def extend_lease(self, worker_id, job_id, *, lease_seconds):
        result = self._extend_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def release_lease(self, worker_id, job_id):
        self.released.append(job_id)

    async def close(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# _default_executor
# --------------------------------------------------------------------------- #


async def test_default_executor_returns_zero_without_cancellation() -> None:
    assert await _default_executor(_request(), asyncio.Event()) == 0


async def test_default_executor_raises_when_cancelled() -> None:
    event = asyncio.Event()
    event.set()
    with pytest.raises(CancelledByOrchestrator):
        await _default_executor(_request(), event)


# --------------------------------------------------------------------------- #
# WorkerRunner.step
# --------------------------------------------------------------------------- #


async def test_step_returns_false_when_queue_empty() -> None:
    runner = WorkerRunner(broker=FakeBroker(leases=[]), config=_config())
    assert await runner.step() is False


async def test_step_runs_executor_and_releases_lease() -> None:
    broker = FakeBroker(leases=[_lease()])

    async def ok_executor(request, cancel):
        return 0

    runner = WorkerRunner(broker=broker, config=_config(), executor=ok_executor)
    assert await runner.step() is True
    assert broker.released == ["job-1"]


async def test_step_handles_cancelled_by_orchestrator() -> None:
    broker = FakeBroker(leases=[_lease()])

    async def cancel_executor(request, cancel):
        raise CancelledByOrchestrator()

    runner = WorkerRunner(broker=broker, config=_config(), executor=cancel_executor)
    assert await runner.step() is True
    assert broker.released == ["job-1"]  # lease still released on cancel


async def test_step_leaves_lease_unreleased_on_retryable_unavailable() -> None:
    broker = FakeBroker(leases=[_lease()])

    async def unavailable_executor(request, cancel):
        raise RetryableExecutorUnavailable()

    runner = WorkerRunner(broker=broker, config=_config(), executor=unavailable_executor)
    assert await runner.step() is True
    # The lease is intentionally left for the broker to reclaim/requeue.
    assert broker.released == []


async def test_retryable_canonical_executor_retains_claim_without_terminal_publication(monkeypatch):
    from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator, current_dispatch_fence

    locator = DispatchLocator("job_unavailable", "attempt", "dispatch", "a" * 64, "tenant")
    fence = DispatchFence(locator, "w", 1, 0.0, "/output", "/requested")
    terminal_calls = []
    heartbeat_started = asyncio.Event()
    heartbeat_stopped = asyncio.Event()

    class Store:
        async def claim_dispatch(self, *args, **kwargs):
            return fence

        async def finish_dispatch(self, observed, **kwargs):
            terminal_calls.append((observed, kwargs))

    async def heartbeat(*args, **kwargs):
        heartbeat_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            heartbeat_stopped.set()

    async def executor(request, cancellation):
        assert request == locator
        assert current_dispatch_fence() == fence
        await heartbeat_started.wait()
        raise RetryableExecutorUnavailable("durable job hydration unavailable")

    broker = FakeBroker(leases=[JobLease(locator.job_id, "w", 0.0, locator)])
    monkeypatch.setattr(worker_module, "get_operational_record_store", lambda: Store())
    runner = WorkerRunner(broker=broker, config=_config(), executor=executor)
    monkeypatch.setattr(runner, "_heartbeat_loop", heartbeat)
    assert await runner.step() is True
    assert terminal_calls == []  # DB-clock expiry owns the eventual worker_lost outcome.
    assert broker.released == []
    assert heartbeat_stopped.is_set()
    assert current_dispatch_fence() is None


async def test_step_releases_lease_on_generic_executor_error() -> None:
    broker = FakeBroker(leases=[_lease()])

    async def boom_executor(request, cancel):
        raise RuntimeError("job blew up")

    runner = WorkerRunner(broker=broker, config=_config(), executor=boom_executor)
    assert await runner.step() is True
    assert broker.released == ["job-1"]


async def test_step_cancels_executor_when_heartbeat_io_fails() -> None:
    broker = FakeBroker(leases=[_lease()], extend_results=[ConnectionError("broker unavailable")])
    observations = []

    async def executor(request, cancel):
        observations.append("started")
        await asyncio.wait_for(cancel.wait(), timeout=1.0)
        observations.append("cancelled")
        raise CancelledByOrchestrator()

    runner = WorkerRunner(broker=broker, config=_config(heartbeat_interval_seconds=0.0), executor=executor)
    assert await runner.step() is True
    assert observations == ["started", "cancelled"]
    assert broker.released == ["job-1"]
    assert await runner.step() is False


async def test_step_contains_release_failure_after_executor_stops() -> None:
    class ReleaseFailingBroker(FakeBroker):
        async def release_lease(self, worker_id, job_id):
            await super().release_lease(worker_id, job_id)
            raise ConnectionError("broker unavailable")

    broker = ReleaseFailingBroker(leases=[_lease()])
    executions = []

    async def executor(request, cancel):
        executions.append(request.job_id)
        return 0

    runner = WorkerRunner(broker=broker, config=_config(), executor=executor)
    assert await runner.step() is True
    assert await runner.step() is False
    assert executions == ["job-1"]
    assert broker.released == ["job-1"]


# --------------------------------------------------------------------------- #
# WorkerRunner._heartbeat_loop
# --------------------------------------------------------------------------- #


async def test_heartbeat_signals_cancellation_on_lost_lease() -> None:
    broker = FakeBroker(extend_results=[LeaseNotHeldError("w", "job-1")])
    runner = WorkerRunner(broker=broker, config=_config(heartbeat_interval_seconds=0.0))
    event = asyncio.Event()

    await runner._heartbeat_loop("job-1", event)
    assert event.is_set()


async def test_heartbeat_signals_cancellation_on_broker_cancel() -> None:
    # First extension is active (loop continues), second reports cancellation.
    broker = FakeBroker(extend_results=[LeaseStatus.active, LeaseStatus.cancelled])
    runner = WorkerRunner(broker=broker, config=_config(heartbeat_interval_seconds=0.0))
    event = asyncio.Event()

    await runner._heartbeat_loop("job-1", event)
    assert event.is_set()


@pytest.mark.parametrize("failure", [ConnectionError("disconnected"), TimeoutError("timed out")])
async def test_heartbeat_signals_cancellation_on_broker_io_failure(failure) -> None:
    broker = FakeBroker(extend_results=[failure])
    runner = WorkerRunner(broker=broker, config=_config(heartbeat_interval_seconds=0.0))
    event = asyncio.Event()

    await runner._heartbeat_loop("job-1", event)
    assert event.is_set()


@pytest.mark.parametrize("boundary", ["acquire", "heartbeat", "release"])
async def test_broker_task_cancellation_propagates(boundary) -> None:
    class CancelledBroker(FakeBroker):
        async def acquire_lease(self, worker_id, *, lease_seconds):
            if boundary == "acquire":
                raise asyncio.CancelledError()
            return await super().acquire_lease(worker_id, lease_seconds=lease_seconds)

        async def extend_lease(self, *args, **kwargs):
            raise asyncio.CancelledError()

        async def release_lease(self, *args, **kwargs):
            raise asyncio.CancelledError()

    broker = CancelledBroker(leases=[_lease()])
    runner = WorkerRunner(broker=broker, config=_config(heartbeat_interval_seconds=0.0))
    with pytest.raises(asyncio.CancelledError):
        if boundary == "heartbeat":
            await runner._heartbeat_loop("job-1", asyncio.Event())
        else:
            await runner.step()


# --------------------------------------------------------------------------- #
# run_worker_forever supervisor loop
# --------------------------------------------------------------------------- #


async def test_run_worker_forever_resets_backoff_then_backs_off_then_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    calls = {"n": 0}

    # Drive the supervisor loop directly via a controlled step() so all three
    # branches run deterministically regardless of the event loop:
    #   1) did_work=True  -> reset backoff + continue
    #   2) did_work=False -> empty-queue backoff wait that times out
    #   3) did_work=False -> request stop, loop exits
    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            pass

        async def step(self) -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                return True
            if calls["n"] == 2:
                return False  # stop not set yet -> wait_for times out, backoff grows
            stop_event.set()
            return False

    monkeypatch.setattr(worker_module, "WorkerRunner", FakeRunner)
    broker = FakeBroker(leases=[])

    await run_worker_forever(
        broker=broker,
        config=_config(poll_interval_seconds=0.001, max_poll_backoff_seconds=0.01),
        stop_event=stop_event,
    )
    assert calls["n"] == 3
    # Caller-supplied broker is NOT closed by the loop.
    assert broker.closed is False


async def test_supervisor_backs_off_on_acquire_outage_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecoveringBroker(FakeBroker):
        async def acquire_lease(self, worker_id, *, lease_seconds):
            if self.acquire_calls < 2:
                self.acquire_calls += 1
                raise ConnectionError("broker unavailable")
            return await super().acquire_lease(worker_id, lease_seconds=lease_seconds)

    broker = RecoveringBroker(leases=[_lease()])
    stop_event = asyncio.Event()
    executions = []
    backoffs = []
    original_wait_for = asyncio.wait_for

    async def record_wait_for(awaitable, timeout):
        backoffs.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(worker_module.asyncio, "wait_for", record_wait_for)

    async def executor(request, cancel):
        executions.append(request.job_id)
        stop_event.set()
        return 0

    await run_worker_forever(
        broker=broker,
        config=_config(poll_interval_seconds=0.001, max_poll_backoff_seconds=0.002),
        executor=executor,
        stop_event=stop_event,
    )

    assert backoffs == [0.001, 0.002]
    assert broker.acquire_calls == 3
    assert executions == ["job-1"]
    assert broker.released == ["job-1"]


async def test_run_worker_forever_closes_constructed_broker_and_swallows_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosingBroker(FakeBroker):
        async def close(self):
            await super().close()
            raise RuntimeError("close failed")

    constructed = ClosingBroker(leases=[])
    monkeypatch.setattr(worker_module, "get_queue_broker", lambda: constructed)

    stop_event = asyncio.Event()
    stop_event.set()  # exit immediately; exercise the finally/close path

    # broker=None → loop constructs its own and must dispose it on exit even if
    # close() raises. config=None → also exercises _config_from_env().
    await run_worker_forever(stop_event=stop_event)
    assert constructed.closed is True


# --------------------------------------------------------------------------- #
# Config + CLI helpers
# --------------------------------------------------------------------------- #


def test_config_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TP_WORKER_ID", "worker-fixed")
    monkeypatch.setenv("TP_WORKER_LEASE_SECONDS", "12")
    monkeypatch.setenv("TP_WORKER_HEARTBEAT_SECONDS", "3")
    monkeypatch.setenv("TP_WORKER_POLL_SECONDS", "0.5")
    monkeypatch.setenv("TP_WORKER_MAX_BACKOFF_SECONDS", "7")

    config = _config_from_env()
    assert config.worker_id == "worker-fixed"
    assert config.lease_seconds == 12.0
    assert config.heartbeat_interval_seconds == 3.0
    assert config.poll_interval_seconds == 0.5
    assert config.max_poll_backoff_seconds == 7.0


def test_config_from_env_generates_worker_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TP_WORKER_ID", raising=False)
    config = _config_from_env()
    assert config.worker_id.startswith("worker_")


def test_main_registers_signal_handlers_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    handlers: dict[int, object] = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))

    ran = {"called": False}

    async def fake_forever(*, stop_event):
        ran["called"] = True

    monkeypatch.setattr(worker_module, "run_worker_forever", fake_forever)

    # Stub asyncio.run so its internal Runner does not install its own SIGINT
    # handler through the patched signal.signal (which would otherwise clobber
    # main()'s _request_stop with asyncio's KeyboardInterrupt-raising handler).
    def fake_run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(worker_module.asyncio, "run", fake_run)

    main()

    assert ran["called"] is True
    assert signal.SIGINT in handlers and signal.SIGTERM in handlers
    # The installed handler must run without error (sets the local stop event).
    handlers[signal.SIGINT](signal.SIGINT, None)


def test_monotonic_now_returns_float() -> None:
    assert isinstance(monotonic_now(), float)


@pytest.mark.asyncio
async def test_rejected_canonical_dispatch_release_outage_does_not_stop_worker(monkeypatch):
    from transformation_portal.orchestrator.dispatch import DispatchLocator
    from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

    locator = DispatchLocator("job_rejected", "attempt", "dispatch", "a" * 64, "tenant")
    broker = FakeBroker(leases=[JobLease(locator.job_id, "w", 10, locator)])

    async def fail_release(*args):
        raise ConnectionError("Redis release unavailable")

    class LostStore:
        async def claim_dispatch(self, *args, **kwargs):
            raise DispatchAuthorityLost("tombstoned")

    broker.release_lease = fail_release
    monkeypatch.setattr(worker_module, "get_operational_record_store", lambda: LostStore())
    runner = WorkerRunner(broker=broker, config=_config())
    assert await runner.step() is True
    assert await runner.step() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", ["redis", "postgres"])
@pytest.mark.parametrize("resists_cancellation", [False, True])
async def test_stalled_heartbeat_cancels_executor_by_lease_deadline(monkeypatch, stalled, resists_cancellation):
    from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator

    locator = DispatchLocator("job_stalled", "attempt", "dispatch", "a" * 64, "tenant")
    fence = DispatchFence(locator, "w", 1, 0.0, "/output", "/requested")
    broker = FakeBroker(leases=[JobLease(locator.job_id, "w", 0.0, locator)])
    never = asyncio.Event()
    cancellation_observed = asyncio.Event()

    async def stalled_network():
        try:
            await never.wait()
        except asyncio.CancelledError:
            if not resists_cancellation:
                raise
            await never.wait()

    class Store:
        async def claim_dispatch(self, *args, **kwargs):
            return fence

        async def renew_dispatch(self, *args, **kwargs):
            if stalled == "postgres":
                await stalled_network()

        async def finish_dispatch(self, *args, **kwargs):
            return None

    async def renew_broker(*args, **kwargs):
        if stalled == "redis":
            await stalled_network()
        return LeaseStatus.active

    async def executor(request, cancellation):
        await cancellation.wait()
        cancellation_observed.set()
        return 0

    broker.extend_lease = renew_broker
    monkeypatch.setattr(worker_module, "get_operational_record_store", lambda: Store())
    runner = WorkerRunner(
        broker=broker, config=_config(lease_seconds=0.15, heartbeat_interval_seconds=0.01), executor=executor
    )
    started = asyncio.get_running_loop().time()
    await asyncio.wait_for(runner.step(), timeout=0.6)
    assert cancellation_observed.is_set()
    assert asyncio.get_running_loop().time() - started < 0.5
    await asyncio.sleep(0)
    if resists_cancellation:
        assert runner._pending_network
        calls = broker.acquire_calls
        assert await runner.step() is False
        assert broker.acquire_calls == calls
        never.set()
        for _ in range(5):
            await asyncio.sleep(0)
    assert not runner._pending_network


async def test_placeholder_executor_cannot_consume_canonical_dispatch() -> None:
    from transformation_portal.orchestrator.dispatch import DispatchLocator

    locator = DispatchLocator("job_canonical", "attempt", "dispatch", "a" * 64, "tenant")
    with pytest.raises(RuntimeError, match="app-owned trusted executor"):
        await _default_executor(locator, asyncio.Event())


@pytest.mark.parametrize("constructed", [False, True])
async def test_default_supervisor_rejects_locator_queue_and_closes_only_owned_broker(monkeypatch, constructed):
    from transformation_portal.orchestrator.queue.locator import RedisLocatorQueueBroker

    broker = RedisLocatorQueueBroker(redis_url="redis://unused.invalid/0")
    closed = []

    async def close():
        closed.append(True)

    async def forbidden_acquire(*args, **kwargs):
        pytest.fail("placeholder executor must never acquire canonical work")

    monkeypatch.setattr(broker, "close", close)
    monkeypatch.setattr(broker, "acquire_lease", forbidden_acquire)
    monkeypatch.setattr(worker_module, "get_queue_broker", lambda: broker)
    with pytest.raises(RuntimeError, match="app-owned canonical executor"):
        await run_worker_forever(broker=None if constructed else broker, config=_config())
    assert closed == ([True] if constructed else [])


@pytest.mark.parametrize("failure", ["database_outage", "expired_broker_response"])
async def test_unconfirmed_claim_never_executes_or_releases_broker_lease(monkeypatch, failure):
    from transformation_portal.orchestrator.dispatch import DispatchLocator, current_dispatch_fence

    locator = DispatchLocator("job_unconfirmed", "attempt", "dispatch", "a" * 64, "tenant")

    class Broker(FakeBroker):
        async def acquire_lease(self, worker_id, *, lease_seconds):
            lease = await super().acquire_lease(worker_id, lease_seconds=lease_seconds)
            if failure == "expired_broker_response":
                # Model a response already received but delayed by a blocked
                # event loop beyond the conservative request-start deadline.
                time.sleep(0.02)
            return lease

    claims = []

    class Store:
        async def claim_dispatch(self, *args, **kwargs):
            claims.append(True)
            raise ConnectionError("Postgres unavailable")

    async def forbidden_executor(*args):
        pytest.fail("unconfirmed authority reached executor")

    broker = Broker(leases=[JobLease(locator.job_id, "w", 0.0, locator)])
    monkeypatch.setattr(worker_module, "get_operational_record_store", lambda: Store())
    runner = WorkerRunner(broker=broker, config=_config(lease_seconds=0.01), executor=forbidden_executor)
    assert await runner.step() is False
    assert claims == ([True] if failure == "database_outage" else [])
    assert broker.released == []
    assert current_dispatch_fence() is None


@pytest.mark.parametrize("outcome", ["already_committed", "database_outage"])
async def test_executor_completion_retains_uncertain_authority_and_clears_context(monkeypatch, outcome):
    from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator, current_dispatch_fence
    from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

    locator = DispatchLocator("job_finished", "attempt", "dispatch", "a" * 64, "tenant")
    fence = DispatchFence(locator, "w", 1, 0.0, "/output", "/requested")
    terminal_calls = []

    class Store:
        async def claim_dispatch(self, *args, **kwargs):
            return fence

        async def finish_dispatch(self, observed, **kwargs):
            assert observed == fence
            terminal_calls.append(kwargs)
            if outcome == "already_committed":
                raise DispatchAuthorityLost("executor already committed its terminal outcome")
            raise ConnectionError("Postgres unavailable")

    async def executor(request, cancellation):
        assert current_dispatch_fence() == fence
        assert request == locator
        return 0

    broker = FakeBroker(leases=[JobLease(locator.job_id, "w", 0.0, locator)])
    monkeypatch.setattr(worker_module, "get_operational_record_store", lambda: Store())
    runner = WorkerRunner(broker=broker, config=_config(), executor=executor)
    assert await runner.step() is True
    assert len(terminal_calls) == 1
    assert terminal_calls[0]["error"]["code"] == "RUNNER_ERROR"
    assert broker.released == ([locator.job_id] if outcome == "already_committed" else [])
    assert current_dispatch_fence() is None


async def test_expired_heartbeat_deadline_cancels_without_attempting_renewal():
    class Broker(FakeBroker):
        async def extend_lease(self, *args, **kwargs):
            pytest.fail("expired authority cannot be renewed")

    runner = WorkerRunner(broker=Broker(), config=_config())
    cancellation = asyncio.Event()
    await runner._heartbeat_loop("expired", cancellation, lease_deadline=asyncio.get_running_loop().time() - 1.0)
    assert cancellation.is_set()
