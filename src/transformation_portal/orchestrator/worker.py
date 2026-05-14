"""Worker runner that consumes jobs from the ``QueueBroker``.

Phase 2.A ships the consumer skeleton. The worker is **not yet
wired** as the actual job executor: ``app.py``'s
``asyncio.create_subprocess_exec`` call still happens in-band
(orchestrator process) and the real cut-over to "orchestrator
enqueues, worker consumes" is the Phase 2.C deliverable.

This module exists in 2.A so the contract is reviewable end-to-end:
the worker loop, the heartbeat cadence, the cancellation handling,
and the lease release are all in one place. Phase 2.C will replace
the ``_default_executor`` placeholder with the real subprocess
dispatch (the same code path ``_run_job`` uses today, factored out
of ``app.py``).

Layout:

- ``WorkerRunner`` — the consumer. Polls the broker for a lease,
  delegates execution to a pluggable async ``executor`` callable,
  heartbeats while the executor runs, releases the lease on
  completion, and handles ``LeaseStatus.cancelled`` by signalling
  the executor.
- ``run_worker_forever`` — the supervisor loop. Runs ``WorkerRunner
  .step`` repeatedly with backoff when the queue is empty.
- ``main`` — the CLI entry point (``python -m
  transformation_portal.orchestrator.worker``); reads
  ``TP_WORKER_*`` env vars and spawns ``run_worker_forever``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from transformation_portal.orchestrator.queue import (
    LeaseStatus,
    QueueBroker,
    get_queue_broker,
)
from transformation_portal.orchestrator.queue.base import (
    JobEnqueueRequest,
    LeaseNotHeldError,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Tunables for the worker loop.

    ``worker_id`` is required so ``acquire_lease`` / ``extend_lease``
    can identify the lease holder; the timing knobs are optional and
    env-overridable via ``_config_from_env``. Production callers
    typically build the config via ``_config_from_env``, which
    generates a ``worker_id`` of the form ``worker_<8 hex chars>``
    when ``TP_WORKER_ID`` is unset.
    """

    worker_id: str
    lease_seconds: float = 30.0
    heartbeat_interval_seconds: float = 10.0
    poll_interval_seconds: float = 0.25
    max_poll_backoff_seconds: float = 5.0


class CancelledByOrchestrator(Exception):
    """Raised inside the executor when the broker reports cancellation.

    The default executor (placeholder) catches this and shuts down
    cleanly; Phase 2.C's real subprocess executor will translate
    this into a SIGTERM-then-SIGKILL of the dispatch subprocess and
    will mark the job ``canceled`` via the JobRepository.
    """


class RetryableExecutorUnavailable(Exception):
    """Raised when the executor could not safely hydrate or start a job.

    The worker leaves the lease unreleased so the broker can reclaim and
    requeue it after the lease timeout instead of dropping the dispatch
    payload while durable job state is unavailable.
    """


# Signature: ``executor(request, cancellation_event) -> int`` where the
# return value is an exit code (0 = succeeded, nonzero = failed) that the
# caller will translate into a JobRepository state update. Both arguments
# are positional so the type alias matches the ``WorkerRunner.step`` /
# ``_default_executor`` call shape exactly. ``cancellation_event`` is set
# by the heartbeat loop when the broker reports ``LeaseStatus.cancelled``;
# the executor must observe it and exit promptly so the lease can be
# released.
JobExecutor = Callable[[JobEnqueueRequest, asyncio.Event], Awaitable[int]]


async def _default_executor(
    request: JobEnqueueRequest,
    cancellation_event: asyncio.Event,
) -> int:
    """Phase 2.A placeholder executor.

    Logs that a job was received, sleeps briefly to simulate work,
    and exits 0. Phase 2.C replaces this with the real subprocess
    dispatch carved out of ``app.py:_run_job``.
    """
    logger.info(
        "phase2a placeholder executor processing job_id=%s argv=%s",
        request.job_id,
        request.argv,
    )
    # Cooperative cancel: break out of the simulated work as soon as the
    # heartbeat signals cancellation, just like the real executor will.
    try:
        await asyncio.wait_for(cancellation_event.wait(), timeout=0.1)
    except asyncio.TimeoutError:
        pass
    if cancellation_event.is_set():
        raise CancelledByOrchestrator()
    return 0


class WorkerRunner:
    """One worker's main loop. Re-entrant across leases; one job at a time."""

    def __init__(
        self,
        *,
        broker: QueueBroker,
        config: WorkerConfig,
        executor: JobExecutor = _default_executor,
    ) -> None:
        self._broker = broker
        self._config = config
        self._executor = executor

    async def step(self) -> bool:
        """Process at most one job. Returns ``True`` if work was done.

        Callers loop on this; a ``False`` return is the signal to
        back off and poll again.
        """
        lease = await self._broker.acquire_lease(
            self._config.worker_id,
            lease_seconds=self._config.lease_seconds,
        )
        if lease is None:
            return False

        cancellation_event = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(lease.job_id, cancellation_event))
        release_lease = True
        try:
            try:
                exit_code = await self._executor(lease.request, cancellation_event)
                logger.info(
                    "worker %s finished job %s with exit_code=%s",
                    self._config.worker_id,
                    lease.job_id,
                    exit_code,
                )
            except CancelledByOrchestrator:
                logger.info(
                    "worker %s observed cancellation for job %s",
                    self._config.worker_id,
                    lease.job_id,
                )
            except RetryableExecutorUnavailable:
                release_lease = False
                logger.exception(
                    "worker %s executor could not safely start job %s; leaving lease for reclaim",
                    self._config.worker_id,
                    lease.job_id,
                )
            except Exception:  # noqa: BLE001 - executor errors are job-level
                logger.exception(
                    "worker %s executor raised for job %s",
                    self._config.worker_id,
                    lease.job_id,
                )
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, LeaseNotHeldError):
                pass
            if release_lease:
                await self._broker.release_lease(self._config.worker_id, lease.job_id)
        return True

    async def _heartbeat_loop(
        self,
        job_id: str,
        cancellation_event: asyncio.Event,
    ) -> None:
        while True:
            await asyncio.sleep(self._config.heartbeat_interval_seconds)
            try:
                status = await self._broker.extend_lease(
                    self._config.worker_id,
                    job_id,
                    lease_seconds=self._config.lease_seconds,
                )
            except LeaseNotHeldError:
                # Our lease was reclaimed by the broker (likely because
                # this worker fell behind on its heartbeat). Signal
                # cancellation so the executor stops; the orchestrator
                # will see worker_lost via the broker's reclaim sweep.
                logger.warning(
                    "worker %s lost lease on job %s; signalling cancellation",
                    self._config.worker_id,
                    job_id,
                )
                cancellation_event.set()
                return
            if status is LeaseStatus.cancelled:
                logger.info(
                    "worker %s received cancellation for job %s",
                    self._config.worker_id,
                    job_id,
                )
                cancellation_event.set()
                return


async def run_worker_forever(
    *,
    broker: Optional[QueueBroker] = None,
    config: Optional[WorkerConfig] = None,
    executor: JobExecutor = _default_executor,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Supervisor loop with exponential backoff when the queue is empty.

    ``stop_event`` lets tests / signal handlers ask the loop to exit
    cleanly between jobs. When this function constructs the broker
    itself (caller passed ``broker=None``), it also disposes of it
    via ``await broker.close()`` on exit so the Phase 2.B Redis
    backend doesn't leak network connections on SIGINT/SIGTERM.
    Brokers passed in by the caller are left to the caller's
    lifecycle.
    """
    broker_was_constructed = broker is None
    broker = broker if broker is not None else get_queue_broker()
    config = config if config is not None else _config_from_env()
    runner = WorkerRunner(broker=broker, config=config, executor=executor)
    stop_event = stop_event if stop_event is not None else asyncio.Event()

    backoff = config.poll_interval_seconds
    logger.info(
        "worker %s starting (lease=%ss, hb=%ss)", config.worker_id, config.lease_seconds, config.heartbeat_interval_seconds
    )
    try:
        while not stop_event.is_set():
            did_work = await runner.step()
            if did_work:
                backoff = config.poll_interval_seconds
                continue
            # Empty queue - exponential backoff capped at max_poll_backoff_seconds.
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, config.max_poll_backoff_seconds)
    finally:
        if broker_was_constructed:
            try:
                await broker.close()
            except Exception:  # noqa: BLE001 - never block shutdown on close
                logger.exception("worker %s broker close failed", config.worker_id)
        logger.info("worker %s stopping", config.worker_id)


def _config_from_env() -> WorkerConfig:
    return WorkerConfig(
        worker_id=os.getenv("TP_WORKER_ID", f"worker_{uuid.uuid4().hex[:8]}"),
        lease_seconds=float(os.getenv("TP_WORKER_LEASE_SECONDS", "30")),
        heartbeat_interval_seconds=float(os.getenv("TP_WORKER_HEARTBEAT_SECONDS", "10")),
        poll_interval_seconds=float(os.getenv("TP_WORKER_POLL_SECONDS", "0.25")),
        max_poll_backoff_seconds=float(os.getenv("TP_WORKER_MAX_BACKOFF_SECONDS", "5.0")),
    )


def main() -> None:
    """``python -m transformation_portal.orchestrator.worker`` entry point."""
    logging.basicConfig(
        level=os.getenv("TP_WORKER_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stop_event = asyncio.Event()

    def _request_stop(_signum: int, _frame: object) -> None:
        logger.info("received signal; requesting worker stop")
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    asyncio.run(run_worker_forever(stop_event=stop_event))


if __name__ == "__main__":  # pragma: no cover - executed via `python -m ...`
    main()


# Time helper for tests that want to pin "now" for the broker sweeper.
def monotonic_now() -> float:
    return time.monotonic()


__all__ = [
    "CancelledByOrchestrator",
    "JobExecutor",
    "RetryableExecutorUnavailable",
    "WorkerConfig",
    "WorkerRunner",
    "main",
    "monotonic_now",
    "run_worker_forever",
]
