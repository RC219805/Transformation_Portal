"""External orchestrator worker process entrypoint.

This module runs the same executor that the FastAPI lifespan uses for the
default in-process worker pool, but in a standalone process for multi-host
paid-pilot deployments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from transformation_portal.orchestrator.execution_runtime import create_managed_execution_service
from transformation_portal.orchestrator.worker import run_worker_forever

logger = logging.getLogger(__name__)


async def run_external_worker(*, stop_event: asyncio.Event | None = None) -> None:
    """Consume immutable dispatches through the shared execution service."""
    service = create_managed_execution_service()
    await run_worker_forever(executor=service.execute, stop_event=stop_event)


def main() -> None:
    """``python -m transformation_portal.orchestrator.worker_process`` entry point."""
    logging.basicConfig(
        level=os.getenv("TP_WORKER_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stop_event = asyncio.Event()

    def _request_stop(_signum: int, _frame: object) -> None:
        logger.info("received signal; requesting external worker drain")
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    asyncio.run(run_external_worker(stop_event=stop_event))


if __name__ == "__main__":  # pragma: no cover - executed via `python -m ...`
    main()


__all__ = ["main", "run_external_worker"]
