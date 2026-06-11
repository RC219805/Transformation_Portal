"""Behavioral coverage for ``dashboard.execution_api``.

Drives the execution router via ``TestClient`` against an injected
``ExecutionManager`` (set with ``set_manager``). Run state is created directly
through ``prepare_run`` so the run/list/cancel endpoints are exercised without
spinning real background tasks; the run endpoint's launch is stubbed for the
success path and forced to fail for the 500 path. Also covers ``broadcast``
fan-out, the manager accessor, and the WebSocket ping/pong.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import execution_api
from transformation_portal.dashboard.execution_manager import ExecutionManager, RunStatus


@pytest.fixture(autouse=True)
def reset_clients():
    execution_api._websocket_clients.clear()
    yield
    execution_api._websocket_clients.clear()


@pytest.fixture
def manager() -> ExecutionManager:
    mgr = ExecutionManager()
    execution_api.set_manager(mgr)
    return mgr


@pytest.fixture
def client(manager: ExecutionManager) -> TestClient:
    app = FastAPI()
    app.include_router(execution_api.create_execution_router())
    return TestClient(app)


def _pipeline() -> Dict[str, Any]:
    return {"nodes": [{"id": "a", "type": "passthrough"}], "edges": []}


# --------------------------------------------------------------------------- #
# manager accessor + broadcast
# --------------------------------------------------------------------------- #


def test_get_manager_lazily_creates_then_set_overrides() -> None:
    execution_api.set_manager(None)  # type: ignore[arg-type]
    created = execution_api.get_manager()
    assert isinstance(created, ExecutionManager)
    assert execution_api.get_manager() is created  # stable

    replacement = ExecutionManager()
    execution_api.set_manager(replacement)
    assert execution_api.get_manager() is replacement


async def test_broadcast_prunes_disconnected_clients() -> None:
    sent: List[Dict[str, Any]] = []

    class _Good:
        async def send_json(self, msg):
            sent.append(msg)

    class _Bad:
        async def send_json(self, msg):
            raise RuntimeError("closed")

    good, bad = _Good(), _Bad()
    execution_api._websocket_clients.extend([good, bad])
    await execution_api.broadcast({"type": "tick"})

    assert sent == [{"type": "tick"}]
    assert bad not in execution_api._websocket_clients
    assert good in execution_api._websocket_clients


# --------------------------------------------------------------------------- #
# /run
# --------------------------------------------------------------------------- #


def test_run_accepts_and_returns_run_id(client: TestClient, manager: ExecutionManager, monkeypatch) -> None:
    class _FakeTask:
        def get_name(self) -> str:
            return "fake-task"

    monkeypatch.setattr(manager, "start_pipeline_background", lambda *a, **k: _FakeTask())

    resp = client.post("/api/exec/run", json=_pipeline())
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert isinstance(body["run_id"], str) and body["run_id"]


def test_run_launch_failure_returns_500(client: TestClient, manager: ExecutionManager, monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("cannot start")

    monkeypatch.setattr(manager, "start_pipeline_background", _boom)
    assert client.post("/api/exec/run", json=_pipeline()).status_code == 500


# --------------------------------------------------------------------------- #
# /runs + /runs/{id} + cancel
# --------------------------------------------------------------------------- #


def test_list_runs(client: TestClient, manager: ExecutionManager) -> None:
    manager.prepare_run("r1", _pipeline())
    runs = client.get("/api/exec/runs").json()["runs"]
    assert any(r["run_id"] == "r1" for r in runs)


def test_get_run_detail_and_404(client: TestClient, manager: ExecutionManager) -> None:
    manager.prepare_run("r1", _pipeline())
    body = client.get("/api/exec/runs/r1").json()
    assert body["run_id"] == "r1"
    assert body["status"] == RunStatus.PENDING.value
    assert "a" in body["nodes"]

    assert client.get("/api/exec/runs/ghost").status_code == 404


def test_cancel_pending_run_returns_202(client: TestClient, manager: ExecutionManager) -> None:
    manager.prepare_run("r1", _pipeline())
    resp = client.post("/api/exec/runs/r1/cancel")
    assert resp.status_code == 202
    assert resp.json()["status"] == "cancelling"


def test_cancel_terminal_run_returns_200(client: TestClient, manager: ExecutionManager) -> None:
    manager.prepare_run("r1", _pipeline())
    manager.get_run_state("r1").status = RunStatus.COMPLETE
    resp = client.post("/api/exec/runs/r1/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "complete"


def test_cancel_unknown_run_404(client: TestClient) -> None:
    assert client.post("/api/exec/runs/ghost/cancel").status_code == 404


# --------------------------------------------------------------------------- #
# UI + WebSocket
# --------------------------------------------------------------------------- #


def test_execution_ui_serves_html(client: TestClient) -> None:
    resp = client.get("/api/exec/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_websocket_ping_pong(client: TestClient) -> None:
    with client.websocket_connect("/api/exec/ws") as ws:
        ws.send_text("ping")
        assert ws.receive_text() == "pong"
    assert len(execution_api._websocket_clients) == 0
