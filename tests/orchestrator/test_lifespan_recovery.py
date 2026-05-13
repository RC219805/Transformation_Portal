"""End-to-end test for the FastAPI lifespan restart-recovery wiring.

The unit tests in ``test_restart_recovery.py`` exercise
``sweep_orphaned_jobs`` directly. This file proves the wiring inside
``app.py``'s ``_orchestrator_lifespan`` actually invokes the sweep on
startup and stashes the result so operators can observe it via
``app.state.restart_recovery_swept``.

Memory backend only - the Postgres backend is verified by the
parametrized contract tests when ``TP_TEST_POSTGRES_URL`` is set. We
deliberately do not seed orphans here because cross-event-loop reuse
of the memory backend's per-job ``asyncio.Lock`` instances between an
``asyncio.run(...)`` setup loop and the ``TestClient`` lifespan loop
is fragile across Python versions. The wiring contract being pinned
is "the lifespan calls the sweeper and exposes its result"; the
sweeper's own behavior is fully covered elsewhere.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as orchestrator_app
from transformation_portal.orchestrator import reset_singletons
from transformation_portal.orchestrator.runtime_handles import reset_runtime_registry

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _reset_orchestrator_singletons() -> None:
    reset_singletons()
    reset_runtime_registry()
    yield
    reset_singletons()
    reset_runtime_registry()


def test_orchestrator_lifespan_runs_restart_recovery_sweep() -> None:
    """The lifespan must call ``sweep_orphaned_jobs`` and stash its result."""
    with TestClient(orchestrator_app.app):
        # The lifespan startup completed without raising. The swept list
        # is exposed on app.state so operators can log / metric it.
        swept = getattr(orchestrator_app.app.state, "restart_recovery_swept", None)
        assert swept is not None, (
            "lifespan did not set app.state.restart_recovery_swept; " "sweep_orphaned_jobs may not have been wired"
        )
        assert isinstance(swept, list)


def test_orchestrator_lifespan_recovery_handles_repository_failures() -> None:
    """If the sweeper raises, startup must continue and stash an empty list.

    Phase 1.C contract: restart recovery is best-effort - a transient
    Postgres outage at boot must not block the orchestrator from
    serving requests. The exception is logged but not propagated.
    """
    import transformation_portal.orchestrator.recovery as recovery_module
    from transformation_portal.orchestrator.storage import base as storage_base

    original = recovery_module.sweep_orphaned_jobs

    async def _failing_sweep(*_args, **_kwargs):
        raise storage_base.RepositoryError("simulated transient failure")

    recovery_module.sweep_orphaned_jobs = _failing_sweep
    # app.py imported sweep_orphaned_jobs at module-load, so monkeypatch
    # the symbol it actually calls too.
    original_in_app = orchestrator_app.sweep_orphaned_jobs
    orchestrator_app.sweep_orphaned_jobs = _failing_sweep
    try:
        with TestClient(orchestrator_app.app):
            swept = getattr(orchestrator_app.app.state, "restart_recovery_swept", None)
            assert swept == [], (
                "failed sweep must leave restart_recovery_swept as an empty list " "rather than propagating the exception"
            )
    finally:
        recovery_module.sweep_orphaned_jobs = original
        orchestrator_app.sweep_orphaned_jobs = original_in_app
