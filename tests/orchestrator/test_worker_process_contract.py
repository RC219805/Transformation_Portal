"""Contracts for the external orchestrator worker process entrypoint."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from transformation_portal.orchestrator import worker_process

pytestmark = pytest.mark.unit


def test_external_worker_uses_shared_service_executor(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(worker_process, "create_managed_execution_service", lambda: SimpleNamespace(execute=fake_executor))
    monkeypatch.setattr(worker_process, "run_worker_forever", fake_run_worker_forever)
    stop_event = asyncio.Event()

    asyncio.run(worker_process.run_external_worker(stop_event=stop_event))

    assert captured == {"executor": fake_executor, "stop_event": stop_event}


def test_external_worker_constructs_without_importing_http_application() -> None:
    program = """
import importlib.abc
import sys
class DenyHttpApplication(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *_args):
        if fullname == 'app' or fullname.startswith('fastapi'):
            raise AssertionError('worker imported HTTP application: ' + fullname)
sys.meta_path.insert(0, DenyHttpApplication())
from transformation_portal.orchestrator.execution_runtime import create_managed_execution_service
from transformation_portal.orchestrator.worker_process import run_external_worker
service = create_managed_execution_service()
assert callable(service.execute)
assert 'app' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        timeout=20,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr.decode()
