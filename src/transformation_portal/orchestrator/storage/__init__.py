"""Storage backend factory keyed off ``TP_ORCHESTRATOR_STATE_BACKEND``.

The factory caches singleton repository / event-store instances. Tests can
call ``reset_singletons()`` (and ``await store.reset()`` on the returned
instances) to start clean.

Supported backends:

- ``memory`` (default) — in-process; behavior-identical to the legacy
  ``app.py:JOBS`` dict.
- ``postgres`` — added in Phase 1 Layer 1.B; requires ``TP_DATABASE_URL``.
"""

from __future__ import annotations

import os
from typing import Optional

from transformation_portal.orchestrator.storage.base import (
    JobEventStore,
    JobRepository,
)

_BACKEND_ENV = "TP_ORCHESTRATOR_STATE_BACKEND"
_DATABASE_URL_ENV = "TP_DATABASE_URL"

_repository: Optional[JobRepository] = None
_event_store: Optional[JobEventStore] = None


def _selected_backend() -> str:
    return os.getenv(_BACKEND_ENV, "memory").strip().lower() or "memory"


def get_job_repository() -> JobRepository:
    """Return the singleton repository, constructing it on first use."""
    global _repository
    if _repository is not None:
        return _repository

    backend = _selected_backend()
    if backend == "memory":
        from transformation_portal.orchestrator.storage.memory import (
            MemoryJobRepository,
        )

        _repository = MemoryJobRepository()
        return _repository

    if backend == "postgres":
        try:
            from transformation_portal.orchestrator.storage.postgres import (
                PostgresJobRepository,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"{_BACKEND_ENV}=postgres requires sqlalchemy[asyncio] and "
                "asyncpg, which Phase 1.B will add to requirements/base.in. "
                "Until that PR lands, only the memory backend is available."
            ) from exc

        database_url = os.getenv(_DATABASE_URL_ENV, "").strip()
        if not database_url:
            raise RuntimeError(
                f"{_BACKEND_ENV}=postgres requires {_DATABASE_URL_ENV} to "
                "be set (e.g. postgresql+asyncpg://user:pw@host:5432/db)."
            )

        _repository = PostgresJobRepository(database_url=database_url)
        return _repository

    raise RuntimeError(f"Unsupported {_BACKEND_ENV}={backend!r}; expected 'memory' or 'postgres'.")


def get_job_event_store() -> JobEventStore:
    """Return the singleton event store, constructing it on first use."""
    global _event_store
    if _event_store is not None:
        return _event_store

    backend = _selected_backend()
    if backend == "memory":
        from transformation_portal.orchestrator.storage.memory import (
            MemoryJobEventStore,
        )

        _event_store = MemoryJobEventStore()
        return _event_store

    if backend == "postgres":
        try:
            from transformation_portal.orchestrator.storage.postgres import (
                PostgresJobEventStore,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"{_BACKEND_ENV}=postgres requires sqlalchemy[asyncio] and "
                "asyncpg, which Phase 1.B will add to requirements/base.in. "
                "Until that PR lands, only the memory backend is available."
            ) from exc

        database_url = os.getenv(_DATABASE_URL_ENV, "").strip()
        if not database_url:
            raise RuntimeError(
                f"{_BACKEND_ENV}=postgres requires {_DATABASE_URL_ENV} to "
                "be set (e.g. postgresql+asyncpg://user:pw@host:5432/db)."
            )

        _event_store = PostgresJobEventStore(database_url=database_url)
        return _event_store

    raise RuntimeError(f"Unsupported {_BACKEND_ENV}={backend!r}; expected 'memory' or 'postgres'.")


def reset_singletons() -> None:
    """Drop the cached singletons. Tests call this between cases."""
    global _repository, _event_store
    _repository = None
    _event_store = None


__all__ = ["get_job_event_store", "get_job_repository", "reset_singletons"]
