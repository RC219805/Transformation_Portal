"""Worker runner that consumes jobs from the ``QueueBroker``.

The FastAPI lifespan uses ``WorkerRunner`` with the app-owned
``_orchestrator_job_executor`` for the default in-process worker pool.
Multi-host deployments can run the same executor through
``python -m transformation_portal.orchestrator.worker_process`` while
the backend keeps broker admission as the single execution seam.

The ``_default_executor`` remains a lightweight test/CLI fallback for
this module's generic ``python -m transformation_portal.orchestrator.worker``
entrypoint; production execution should use the app-wired executor.

Layout:

- ``WorkerRunner`` — the consumer. Polls the broker for a lease,
  delegates execution to a pluggable async ``executor`` callable,
  heartbeats while the executor runs, releases the lease on
  completion, and handles ``LeaseStatus.cancelled`` by signalling
  the executor.
- ``run_worker_forever`` — the supervisor loop. Runs ``WorkerRunner
  .step`` repeatedly with backoff when the queue is empty.
- ``main`` — the generic CLI entry point (``python -m
  transformation_portal.orchestrator.worker``); reads ``TP_WORKER_*``
  env vars and spawns ``run_worker_forever`` with the default executor.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator, _dispatch_fence
from transformation_portal.orchestrator.queue import (
    LeaseStatus,
    QueueBroker,
    get_queue_broker,
)
from transformation_portal.orchestrator.queue.base import (
    JobEnqueueRequest,
    LeaseNotHeldError,
)
from transformation_portal.orchestrator.storage import get_operational_record_store
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

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
JobExecutor = Callable[[JobEnqueueRequest | DispatchLocator, asyncio.Event], Awaitable[int]]


async def _default_executor(
    request: JobEnqueueRequest | DispatchLocator,
    cancellation_event: asyncio.Event,
) -> int:
    """Lightweight fallback executor for this generic runner module.

    Production workers pass the app-owned orchestrator executor instead.
    """
    if isinstance(request, DispatchLocator):
        raise RuntimeError("canonical dispatch requires the app-owned trusted executor")
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
        self._pending_network: set[asyncio.Future[Any]] = set()

    async def _network(self, operation: Awaitable[Any], *, timeout: float) -> Any:
        """Bound waiting without waiting for an unresponsive socket's cancellation."""
        task = asyncio.ensure_future(operation)
        self._pending_network.add(task)

        def completed(future: asyncio.Future[Any]) -> None:
            self._pending_network.discard(future)
            if not future.cancelled():
                future.exception()  # consume a late failure after the deadline

        task.add_done_callback(completed)
        try:
            done, _ = await asyncio.wait({task}, timeout=max(0.001, timeout))
            if not done:
                task.cancel()
                raise TimeoutError("worker authority operation exceeded its lease safety deadline")
            return task.result()
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def step(self) -> bool:
        """Process at most one job. Returns ``True`` if work was done.

        Callers loop on this; a ``False`` return is the signal to
        back off and poll again, including when the broker is unavailable.
        """
        if self._pending_network:
            return False
        lease_deadline = asyncio.get_running_loop().time() + self._config.lease_seconds
        try:
            lease = await self._network(
                self._broker.acquire_lease(
                    self._config.worker_id,
                    lease_seconds=self._config.lease_seconds,
                ),
                timeout=min(5.0, self._config.lease_seconds),
            )
        except Exception:  # noqa: BLE001 - broker IO must not stop the supervisor
            logger.exception("worker %s could not acquire a lease; backing off", self._config.worker_id)
            return False
        if lease is None:
            return False

        fence = None
        if isinstance(lease.request, DispatchLocator):
            try:
                remaining = lease_deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("broker lease expired before claim")
                fence = await self._network(
                    get_operational_record_store().claim_dispatch(
                        lease.request, self._config.worker_id, lease_seconds=self._config.lease_seconds
                    ),
                    timeout=min(5.0, remaining),
                )
            except DispatchAuthorityLost:
                try:
                    await self._network(self._broker.release_lease(self._config.worker_id, lease.job_id), timeout=5.0)
                except Exception:
                    logger.exception("rejected dispatch release failed; leaving broker item for reclaim")
                return True
            except Exception:
                logger.exception("claim unavailable; retaining broker lease for fail-closed recovery")
                return False
        token = _dispatch_fence.set(fence)
        cancellation_event = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(lease.job_id, cancellation_event, fence, lease_deadline))
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
            if fence is not None:
                try:
                    # An executor must commit its terminal outcome. A return or
                    # crash without a commit consumes this attempt once only.
                    await self._network(
                        get_operational_record_store().finish_dispatch(
                            fence,
                            state="failed",
                            error={"code": "RUNNER_ERROR", "message": "Executor exited without committed publication."},
                        ),
                        timeout=min(5.0, self._config.lease_seconds),
                    )
                except DispatchAuthorityLost:
                    pass
                except Exception:
                    release_lease = False
                    logger.exception("terminal authority unavailable; leaving claim for DB-clock expiry")
            _dispatch_fence.reset(token)
            if release_lease:
                try:
                    await self._network(self._broker.release_lease(self._config.worker_id, lease.job_id), timeout=5.0)
                except Exception:  # noqa: BLE001 - an uncertain release is left for reclaim
                    logger.exception(
                        "worker %s could not release lease for job %s; leaving lease for reclaim",
                        self._config.worker_id,
                        lease.job_id,
                    )
        return True

    async def _heartbeat_loop(
        self,
        job_id: str,
        cancellation_event: asyncio.Event,
        fence: Optional[DispatchFence] = None,
        lease_deadline: Optional[float] = None,
    ) -> None:
        clock = asyncio.get_running_loop().time
        deadline = lease_deadline if lease_deadline is not None else clock() + self._config.lease_seconds
        while True:
            await asyncio.sleep(min(self._config.heartbeat_interval_seconds, max(0.0, deadline - clock())))
            renewal_started = clock()

            async def renew() -> LeaseStatus:
                status = await self._broker.extend_lease(
                    self._config.worker_id, job_id, lease_seconds=self._config.lease_seconds
                )
                if status is LeaseStatus.active and fence is not None:
                    # The DB fence renews only after confirmed broker authority.
                    await get_operational_record_store().renew_dispatch(fence, lease_seconds=self._config.lease_seconds)
                return status

            try:
                remaining = deadline - renewal_started
                if remaining <= 0:
                    raise TimeoutError("worker lease expired without a confirmed renewal")
                status = await self._network(renew(), timeout=remaining)
            except Exception:
                # This deadline is independent of TCP/socket cancellation: a
                # paused server cannot keep a child alive past its last lease.
                cancellation_event.set()
                logger.exception(
                    "worker %s lost or could not confirm lease for %s; signalling cancellation", self._config.worker_id, job_id
                )
                return
            if status is LeaseStatus.cancelled:
                cancellation_event.set()
                return
            # Anchor to request start, conservatively including roundtrip time.
            deadline = renewal_started + self._config.lease_seconds


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
    if executor is _default_executor:
        from transformation_portal.orchestrator.queue.locator import RedisLocatorQueueBroker

        if isinstance(broker, RedisLocatorQueueBroker):
            if broker_was_constructed:
                await broker.close()
            raise RuntimeError("Redis locator workers require the app-owned canonical executor")
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
