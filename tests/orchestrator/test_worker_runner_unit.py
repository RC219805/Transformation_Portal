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

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.orchestrator import worker as worker_module
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
from transformation_portal.orchestrator.queue.base import (
    JobEnqueueRequest,
    JobLease,
    LeaseNotHeldError,
    LeaseStatus,
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


async def test_step_releases_lease_on_generic_executor_error() -> None:
    broker = FakeBroker(leases=[_lease()])

    async def boom_executor(request, cancel):
        raise RuntimeError("job blew up")

    runner = WorkerRunner(broker=broker, config=_config(), executor=boom_executor)
    assert await runner.step() is True
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
