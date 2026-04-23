#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract tests for the FastAPI lifespan migration.

Pins the outcomes of item 11 of the portal backend hardening pass:

* The app is constructed with a ``lifespan`` context manager.
* No ``on_event`` handlers are registered on the router (the migration
  replaces them; a regression would re-populate ``on_startup`` /
  ``on_shutdown``).
* Entering the lifespan context spawns the cleanup task; exiting it
  cancels the task and clears the handle from ``app.state``.

Relocated from ``tests/test_portal_backend_hardening.py`` so each
contract has a single failure surface.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")


def test_lifespan_context_registered() -> None:
    assert orchestrator_app.app.router.lifespan_context is not None


def test_no_on_event_handlers_registered() -> None:
    # Re-introducing an ``@app.on_event("startup")`` or ``@app.on_event("shutdown")``
    # decorator would repopulate these router lists. Pinning them to empty
    # is the assertion the mis-named original test was meant to make.
    assert orchestrator_app.app.router.on_startup == []
    assert orchestrator_app.app.router.on_shutdown == []


def test_lifespan_creates_and_cancels_cleanup_task() -> None:
    async def _drive() -> tuple[bool, bool]:
        async with orchestrator_app.app.router.lifespan_context(orchestrator_app.app):
            cleanup_task = orchestrator_app.app.state.cleanup_task
            started = cleanup_task is not None and not cleanup_task.done()
        finished = orchestrator_app.app.state.cleanup_task is None
        return started, finished

    started, finished = asyncio.run(_drive())
    assert started
    assert finished
