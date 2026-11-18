"""Behavioral coverage for ``dashboard.execution_api``.

Drives the execution router via ``TestClient`` against an injected
``ExecutionManager`` (set with ``set_manager``). Run state is created directly
through ``prepare_run`` so the run/list/cancel endpoints are exercised without
spinning real background tasks; the run endpoint's launch is stubbed for the
success path and forced to fail for the 500 path. Also covers ``broadcast``
fan-out, the manager accessor, and the WebSocket ping/pong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.responses import JSONResponse
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


class _StrictClient:
    """WebSocket stand-in that serialises like Starlette's strict JSON path."""

    def __init__(self) -> None:
        self.sent: List[Any] = []

    async def send_json(self, msg):
        # JSONResponse.render uses allow_nan=False and raises TypeError on
        # objects json cannot encode, which is what a real send would hit.
        self.sent.append(JSONResponse(content=msg).body)


def _unserialisable_outputs() -> Dict[str, Any]:
    return {
        "depth": np.zeros((2, 3), dtype=np.float32),
        "file": Path("/tmp/depth.npy"),
        "loss": float("nan"),
        "scale": np.float32(1.5),
    }


async def test_broadcast_unserialisable_payload_keeps_client_subscribed() -> None:
    # D2: a node output such as an ndarray used to raise TypeError inside
    # send_json, which broadcast treated as a disconnect, so every client was
    # silently unsubscribed while its socket stayed open.
    client = _StrictClient()
    execution_api._websocket_clients.append(client)

    await execution_api.broadcast({"type": "node_complete", "outputs": _unserialisable_outputs()})
    await execution_api.broadcast({"type": "run_complete"})

    assert client in execution_api._websocket_clients
    assert len(client.sent) == 2


async def test_broadcast_unexpected_send_error_keeps_client() -> None:
    class _Flaky:
        async def send_json(self, msg):
            raise ValueError("not a disconnect")

    flaky = _Flaky()
    execution_api._websocket_clients.append(flaky)
    await execution_api.broadcast({"type": "tick"})
    assert flaky in execution_api._websocket_clients


def test_to_json_safe_summarises_unserialisable_values() -> None:
    safe = execution_api.to_json_safe(_unserialisable_outputs())
    assert safe == {
        "depth": {"type": "ndarray", "shape": [2, 3], "dtype": "float32"},
        "file": "/tmp/depth.npy",
        "loss": None,
        "scale": 1.5,
    }


def test_get_run_with_unserialisable_outputs_returns_200(client: TestClient, manager: ExecutionManager) -> None:
    # D2: raw node outputs used to go straight into JSONResponse -> 500.
    manager.prepare_run("r1", _pipeline())
    state = manager.get_run_state("r1")
    state.nodes["a"].outputs = _unserialisable_outputs()
    state.results = {"a": _unserialisable_outputs()}

    resp = client.get("/api/exec/runs/r1")
    assert resp.status_code == 200
    outputs = resp.json()["nodes"]["a"]["outputs"]
    assert outputs["depth"] == {"type": "ndarray", "shape": [2, 3], "dtype": "float32"}
    assert outputs["loss"] is None
    assert resp.json()["results"]["a"]["file"] == "/tmp/depth.npy"


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
