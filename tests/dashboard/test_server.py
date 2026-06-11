"""Behavioral coverage for ``dashboard.server``.

Covers the dashboard app factory and its event plumbing offline:

- ``DashboardEvent.to_dict`` shape
- ``broadcast_event`` history append + trim, with the no-event-loop path
- ``_broadcast_to_clients`` fan-out and disconnected-client pruning
- ``create_app`` routes: index HTML, ``/api/status``, ``/api/history`` limit,
  ``/api/event`` POST, and the ``/ws`` WebSocket ping/pong + history-on-connect
- ``DashboardServer`` construction + ``stop`` and the FastAPI guards
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.unit

from fastapi.testclient import TestClient

from transformation_portal.dashboard import server


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module-global client/history lists around each test."""
    server._connected_clients.clear()
    server._event_history.clear()
    original_max = server._max_history
    yield
    server._connected_clients.clear()
    server._event_history.clear()
    server._max_history = original_max


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.create_app())


# --------------------------------------------------------------------------- #
# DashboardEvent + broadcast_event
# --------------------------------------------------------------------------- #


def test_dashboard_event_to_dict() -> None:
    evt = server.DashboardEvent(event_type="eval", data={"score": 1}, source="node")
    d = evt.to_dict()
    assert d == {"type": "eval", "data": {"score": 1}, "timestamp": evt.timestamp, "source": "node"}


def test_broadcast_event_appends_history() -> None:
    server.broadcast_event("eval", {"score": 0.9}, source="q")
    assert len(server._event_history) == 1
    assert server._event_history[0].event_type == "eval"


def test_broadcast_event_trims_history() -> None:
    server._max_history = 3
    for i in range(5):
        server.broadcast_event("eval", {"i": i})
    assert len(server._event_history) == 3
    # Oldest entries dropped; the most recent survive.
    assert [e.data["i"] for e in server._event_history] == [2, 3, 4]


async def test_broadcast_to_clients_prunes_disconnected() -> None:
    sent: List[str] = []

    class _Good:
        async def send_text(self, msg: str) -> None:
            sent.append(msg)

    class _Bad:
        async def send_text(self, msg: str) -> None:
            raise RuntimeError("closed")

    good, bad = _Good(), _Bad()
    server._connected_clients.extend([good, bad])

    await server._broadcast_to_clients(server.DashboardEvent(event_type="x", data={}))

    assert len(sent) == 1  # good client received
    assert bad not in server._connected_clients  # failing client pruned
    assert good in server._connected_clients


async def test_broadcast_to_clients_noop_without_clients() -> None:
    # No clients -> returns immediately without error.
    await server._broadcast_to_clients(server.DashboardEvent(event_type="x", data={}))


# --------------------------------------------------------------------------- #
# HTTP routes
# --------------------------------------------------------------------------- #


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_status_endpoint(client: TestClient) -> None:
    body = client.get("/api/status").json()
    assert body["status"] == "running"
    assert body["connected_clients"] == 0
    assert body["event_history_size"] == 0


def test_history_endpoint_respects_limit(client: TestClient) -> None:
    for i in range(5):
        server.broadcast_event("eval", {"i": i})
    body = client.get("/api/history", params={"limit": 2}).json()
    assert body["count"] == 2
    assert [e["data"]["i"] for e in body["events"]] == [3, 4]


def test_post_event_endpoint(client: TestClient) -> None:
    resp = client.post("/api/event", params={"event_type": "progress"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["event"]["type"] == "progress"
    assert len(server._event_history) == 1


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #


def test_websocket_ping_pong_and_history(client: TestClient) -> None:
    # Seed one history event; it is replayed on connect.
    server.broadcast_event("eval", {"score": 1})

    with client.websocket_connect("/ws") as ws:
        first = ws.receive_text()  # history replay
        assert "eval" in first
        ws.send_text("ping")
        assert ws.receive_text() == "pong"
    # After the context exits the client is pruned from the connection list.
    assert len(server._connected_clients) == 0


# --------------------------------------------------------------------------- #
# DashboardServer + guards
# --------------------------------------------------------------------------- #


def test_dashboard_server_construct_and_stop() -> None:
    srv = server.DashboardServer()
    assert srv.app is not None
    srv._server_thread = object()  # type: ignore[assignment]
    srv.stop()
    assert srv._should_stop is True
    assert srv._server_thread is None


def test_create_app_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        server.create_app()


def test_dashboard_server_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        server.DashboardServer()
