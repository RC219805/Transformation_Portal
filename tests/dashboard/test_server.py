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

import asyncio
import threading
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


def test_uvicorn_auto_selects_the_pinned_websocket_protocol() -> None:
    config = server.uvicorn.Config(server.create_app(), ws="auto", lifespan="off")
    config.load()

    protocol = config.ws_protocol_class
    assert protocol is not None
    assert protocol.__module__ == "uvicorn.protocols.websockets.websockets_sansio_impl"
    assert protocol.__name__ == "WebSocketsSansIOProtocol"


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


# --------------------------------------------------------------------------- #
# D5: bounded history, bounded limit, no innerHTML sink for event fields
# --------------------------------------------------------------------------- #


def test_post_event_respects_history_cap(client: TestClient) -> None:
    server._max_history = 3
    for i in range(6):
        assert client.post("/api/event", params={"event_type": "eval"}, json={"i": i}).status_code == 200
    assert len(server._event_history) == 3
    assert [e.data["i"] for e in server._event_history] == [3, 4, 5]


def test_history_cap_shared_by_post_and_broadcast(client: TestClient) -> None:
    server._max_history = 4
    for i in range(5):
        client.post("/api/event", params={"event_type": "eval"}, json={"i": i})
    server.broadcast_event("eval", {"i": 99})
    assert len(server._event_history) == 4
    assert server._event_history[-1].data == {"i": 99}


@pytest.mark.parametrize("limit", [0, -1, server._max_history + 1])
def test_history_rejects_out_of_range_limit(client: TestClient, limit: int) -> None:
    for i in range(3):
        server.broadcast_event("eval", {"i": i})
    resp = client.get("/api/history", params={"limit": limit})
    assert resp.status_code == 422


def test_index_html_renders_event_fields_as_text() -> None:
    html = server._get_index_html()
    # Event type/source/data are client-supplied via POST /api/event.
    assert "innerHTML" not in html
    assert "sourceEl.textContent = String(data.source);" in html
    assert "dataEl.textContent = JSON.stringify(data.data, null, 2);" in html


# --------------------------------------------------------------------------- #
# D6: broadcast_event from a non-serving thread reaches live clients
# --------------------------------------------------------------------------- #


class _ThreadRecordingClient:
    """WebSocket stand-in that records which thread delivered each message."""

    def __init__(self) -> None:
        self.messages: List[str] = []
        self.threads: List[int] = []
        self.received = threading.Event()

    async def send_text(self, msg: str) -> None:
        self.messages.append(msg)
        self.threads.append(threading.get_ident())
        self.received.set()


@pytest.fixture
def serving_loop(monkeypatch: pytest.MonkeyPatch):
    """A running event loop on its own thread, registered as the serving loop."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    idents: Dict[str, int] = {}

    def _run() -> None:
        asyncio.set_event_loop(loop)
        idents["thread"] = threading.get_ident()
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert ready.wait(5)
    monkeypatch.setattr(server, "_serving_loop", loop, raising=False)
    yield loop, idents["thread"]
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def test_broadcast_event_from_worker_thread_reaches_clients(serving_loop: Any) -> None:
    _loop, loop_thread = serving_loop
    ws = _ThreadRecordingClient()
    server._connected_clients.append(ws)

    worker = threading.Thread(target=server.broadcast_event, args=("eval", {"score": 0.1}), kwargs={"source": "worker"})
    worker.start()
    worker.join(5)

    assert ws.received.wait(5), "worker-thread broadcast never reached the connected client"
    assert '"source": "worker"' in ws.messages[0]
    assert ws.threads == [loop_thread]


def test_broadcast_event_from_main_thread_runs_on_serving_loop(serving_loop: Any) -> None:
    _loop, loop_thread = serving_loop
    ws = _ThreadRecordingClient()
    server._connected_clients.append(ws)

    server.broadcast_event("eval", {"score": 0.9}, source="main")

    assert ws.received.wait(5)
    # Delivered on the loop that owns the client, never on an ad-hoc loop of the caller.
    assert ws.threads == [loop_thread]


def test_broadcast_event_without_serving_loop_keeps_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_serving_loop", None, raising=False)
    ws = _ThreadRecordingClient()
    server._connected_clients.append(ws)
    server.broadcast_event("eval", {"score": 1})
    assert len(server._event_history) == 1
    assert ws.messages == []


def test_websocket_connect_captures_serving_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_serving_loop", None, raising=False)
    with TestClient(server.create_app()) as tc:
        with tc.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            assert ws.receive_text() == "pong"
            assert server._serving_loop is not None
