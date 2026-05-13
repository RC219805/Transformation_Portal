"""Queue-broker factory keyed off ``TP_ORCHESTRATOR_QUEUE_BACKEND``.

Phase 2.A shipped the Protocol + the in-process ``memory`` backend.
Phase 2.B adds the ``redis`` backend behind ``TP_REDIS_URL`` so a
fleet of workers can consume from a single durable queue.

Supported backends:

- ``memory`` (default) — single-process FIFO + lease table.
  Restart loses state.
- ``redis`` — Phase 2.B; requires ``TP_REDIS_URL``. Survives
  orchestrator/worker restarts; lease deadlines pinned to the Redis
  server clock so multi-host workers share a single source of truth.
"""

from __future__ import annotations

import os
from typing import Optional

from transformation_portal.orchestrator.queue.base import (
    JobEnqueueRequest,
    JobLease,
    LeaseNotHeldError,
    LeaseStatus,
    QueueBroker,
    QueueBrokerError,
)

_BACKEND_ENV = "TP_ORCHESTRATOR_QUEUE_BACKEND"
_REDIS_URL_ENV = "TP_REDIS_URL"

_broker: Optional[QueueBroker] = None


def _selected_backend() -> str:
    return os.getenv(_BACKEND_ENV, "memory").strip().lower() or "memory"


def get_queue_broker() -> QueueBroker:
    """Return the singleton broker, constructing it on first use."""
    global _broker
    if _broker is not None:
        return _broker

    backend = _selected_backend()
    if backend == "memory":
        from transformation_portal.orchestrator.queue.memory import MemoryQueueBroker

        _broker = MemoryQueueBroker()
        return _broker

    if backend == "redis":
        try:
            from transformation_portal.orchestrator.queue.redis import RedisQueueBroker
        except ImportError as exc:
            raise RuntimeError(
                f"{_BACKEND_ENV}=redis requires the redis-py async client. "
                "Reinstall the base runtime dependencies (`make install-core`) "
                "to pick up `redis>=5`."
            ) from exc

        redis_url = os.getenv(_REDIS_URL_ENV, "").strip()
        if not redis_url:
            raise RuntimeError(f"{_BACKEND_ENV}=redis requires {_REDIS_URL_ENV} to be set " "(e.g. redis://localhost:6379/0).")

        _broker = RedisQueueBroker(redis_url=redis_url)
        return _broker

    raise RuntimeError(f"Unsupported {_BACKEND_ENV}={backend!r}; expected 'memory' or 'redis'.")


def reset_singleton() -> None:
    """Drop the cached singleton. Tests call this between cases."""
    global _broker
    _broker = None


__all__ = [
    "JobEnqueueRequest",
    "JobLease",
    "LeaseNotHeldError",
    "LeaseStatus",
    "QueueBroker",
    "QueueBrokerError",
    "get_queue_broker",
    "reset_singleton",
]
