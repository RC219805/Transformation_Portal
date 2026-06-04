"""Contracts for the external orchestrator worker process entrypoint."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from transformation_portal.orchestrator import worker_process

pytestmark = pytest.mark.unit


def test_external_worker_uses_app_owned_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_executor(_request: Any, _cancel: asyncio.Event) -> int:
        return 0

    async def fake_run_worker_forever(
        *,
        executor: Any,
        stop_event: asyncio.Event | None = None,
        **_kwargs: Any,
    ) -> None:
        captured["executor"] = executor
        captured["stop_event"] = stop_event

    monkeypatch.setattr(
        worker_process.importlib,
        "import_module",
        lambda name: SimpleNamespace(_orchestrator_job_executor=fake_executor) if name == "app" else None,
    )
    monkeypatch.setattr(worker_process, "run_worker_forever", fake_run_worker_forever)
    stop_event = asyncio.Event()

    asyncio.run(worker_process.run_external_worker(stop_event=stop_event))

    assert captured == {"executor": fake_executor, "stop_event": stop_event}
