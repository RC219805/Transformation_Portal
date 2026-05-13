"""Backend-parametrized fixtures for orchestrator storage contract tests.

The same set of contract assertions runs against every registered backend.
The Postgres branch only activates when ``TP_TEST_POSTGRES_URL`` is set, so
``make test-fast`` (offline lane) remains green by exercising memory only.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Tuple

import pytest
import pytest_asyncio

from transformation_portal.orchestrator import (
    JobEventStore,
    JobRepository,
    reset_singletons,
)
from transformation_portal.orchestrator.storage.memory import (
    MemoryJobEventStore,
    MemoryJobRepository,
)

_POSTGRES_URL_ENV = "TP_TEST_POSTGRES_URL"


def _available_backends() -> list[str]:
    backends = ["memory"]
    if os.getenv(_POSTGRES_URL_ENV, "").strip():
        backends.append("postgres")
    return backends


@pytest.fixture(params=_available_backends())
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest_asyncio.fixture
async def repository_and_events(
    backend: str,
) -> AsyncIterator[Tuple[JobRepository, JobEventStore]]:
    """Yield ``(repository, event_store)`` for the parameterized backend.

    Each test gets a freshly-reset pair. Postgres requires
    ``TP_TEST_POSTGRES_URL=postgresql+asyncpg://...``; missing var skips
    the postgres branch entirely (it is not auto-registered).
    """
    reset_singletons()
    if backend == "memory":
        repo: JobRepository = MemoryJobRepository()
        events: JobEventStore = MemoryJobEventStore()
    elif backend == "postgres":
        try:
            from transformation_portal.orchestrator.storage.postgres import (
                PostgresJobEventStore,
                PostgresJobRepository,
            )
        except ImportError:
            pytest.skip("sqlalchemy[asyncio] not installed; skipping postgres")
        url = os.environ[_POSTGRES_URL_ENV]
        repo = PostgresJobRepository(database_url=url)
        events = PostgresJobEventStore(database_url=url)
        await repo.reset()
        await events.reset()
    else:
        raise RuntimeError(f"unknown backend {backend!r}")

    try:
        yield repo, events
    finally:
        await repo.reset()
        await events.reset()
        await repo.close()
        await events.close()
        reset_singletons()
