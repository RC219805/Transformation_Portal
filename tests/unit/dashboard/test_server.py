"""Unit tests for the dashboard FastAPI server module.

Covers the DashboardEvent dataclass, the connected-client broadcast helpers,
broadcast_event history management, the create_app endpoints (including the
/ws WebSocket), and the DashboardServer convenience wrapper.
"""

from __future__ import annotations

from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from transformation_portal.dashboard import server
from transformation_portal.dashboard.server import (
    DashboardEvent,
    DashboardServer,
    broadcast_event,
    create_app,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_state() -> Generator[None, None, None]:
    """Reset the module-level client/history globals between tests."""
    server._connected_clients.clear()
    server._event_history.clear()
    original_max = server._max_history
    yield
    server._connected_clients.clear()
    server._event_history.clear()
    server._max_history = original_max


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as tc:
        yield tc


class _FakeClient:
    """A WebSocket-like client that records sent text."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[str] = []

    async def send_text(self, message: str) -> None:
        if self.fail:
            raise RuntimeError("client gone")
        self.sent.append(message)


class TestDashboardEvent:
    """Tests for the DashboardEvent dataclass."""

    def test_to_dict_shape(self) -> None:
        event = DashboardEvent(event_type="eval", data={"score": 0.9}, source="quality")

        payload = event.to_dict()
        assert payload["type"] == "eval"
        assert payload["data"] == {"score": 0.9}
        assert payload["source"] == "quality"
        assert isinstance(payload["timestamp"], str)

    def test_timestamp_defaults_and_source_optional(self) -> None:
        event = DashboardEvent(event_type="progress", data={})
        assert event.source is None
        assert event.timestamp


class TestBroadcastToClients:
    """Tests for the async _broadcast_to_clients helper."""

    @pytest.mark.asyncio
    async def test_no_clients_is_noop(self) -> None:
        await server._broadcast_to_clients(DashboardEvent("eval", {}))  # must not raise

    @pytest.mark.asyncio
    async def test_sends_to_connected_clients(self) -> None:
        good = _FakeClient()
        server._connected_clients.append(good)

        await server._broadcast_to_clients(DashboardEvent("eval", {"score": 1}))

        assert len(good.sent) == 1

    @pytest.mark.asyncio
    async def test_removes_disconnected_clients(self) -> None:
        good = _FakeClient()
        broken = _FakeClient(fail=True)
        server._connected_clients.extend([good, broken])

        await server._broadcast_to_clients(DashboardEvent("eval", {}))

        assert good in server._connected_clients
        assert broken not in server._connected_clients


class TestBroadcastEvent:
    """Tests for the synchronous broadcast_event wrapper."""

    def test_appends_to_history(self) -> None:
        broadcast_event("eval", {"score": 0.5}, source="node-a")

        assert len(server._event_history) == 1
        assert server._event_history[0].event_type == "eval"
        assert server._event_history[0].source == "node-a"

    def test_history_is_trimmed_to_max(self) -> None:
        server._max_history = 2

        broadcast_event("a", {})
        broadcast_event("b", {})
        broadcast_event("c", {})

        assert len(server._event_history) == 2
        assert [e.event_type for e in server._event_history] == ["b", "c"]


class TestCreateAppEndpoints:
    """Tests for the create_app HTTP endpoints."""

    def test_index_serves_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "APEX Evaluation Dashboard" in response.text

    def test_status_endpoint(self, client: TestClient) -> None:
        body = client.get("/api/status").json()
        assert body["status"] == "running"
        assert body["connected_clients"] == 0
        assert body["event_history_size"] == 0

    def test_history_endpoint_returns_recent_events(self, client: TestClient) -> None:
        broadcast_event("eval", {"score": 1})

        body = client.get("/api/history").json()
        assert body["count"] == 1
        assert body["events"][0]["type"] == "eval"

    def test_history_endpoint_respects_limit(self, client: TestClient) -> None:
        for i in range(5):
            broadcast_event(f"e{i}", {})

        body = client.get("/api/history?limit=2").json()
        assert body["count"] == 2

    def test_post_event_endpoint(self, client: TestClient) -> None:
        response = client.post("/api/event", params={"event_type": "progress"}, json={"pct": 50})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["event"]["type"] == "progress"
        assert len(server._event_history) == 1


class TestWebSocketEndpoint:
    """Tests for the /ws WebSocket endpoint."""

    def test_ping_pong_and_client_tracking(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            assert len(server._connected_clients) == 1
            ws.send_text("ping")
            assert ws.receive_text() == "pong"

        # Disconnecting removes the client in the endpoint's finally block.
        assert len(server._connected_clients) == 0

    def test_sends_recent_history_on_connect(self, client: TestClient) -> None:
        broadcast_event("eval", {"score": 0.7})

        with client.websocket_connect("/ws") as ws:
            first = ws.receive_json()
            assert first["type"] == "eval"


class TestIndexHtmlHelper:
    """Tests for the static index HTML helper."""

    def test_index_html_is_self_contained(self) -> None:
        html = server._get_index_html()
        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")


class TestDashboardServer:
    """Tests for the DashboardServer convenience wrapper."""

    def test_init_builds_app(self) -> None:
        dash = DashboardServer()
        assert dash.app is not None
        assert dash._server_thread is None

    def test_start_invokes_uvicorn_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []

        def _fake_run(app: Any, *, host: str, port: int) -> None:
            calls.append({"app": app, "host": host, "port": port})

        monkeypatch.setattr(server.uvicorn, "run", _fake_run)
        dash = DashboardServer()

        dash.start(host="127.0.0.1", port=9123)

        assert calls == [{"app": dash.app, "host": "127.0.0.1", "port": 9123}]

    def test_start_background_runs_in_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def _fake_run(app: Any, *, host: str, port: int) -> None:
            calls.append(port)

        monkeypatch.setattr(server.uvicorn, "run", _fake_run)
        dash = DashboardServer()

        dash.start_background(port=9124)
        assert dash._server_thread is not None
        dash._server_thread.join(timeout=2)

        assert calls == [9124]

    def test_stop_clears_thread_reference(self) -> None:
        dash = DashboardServer()
        dash._server_thread = object()  # type: ignore[assignment]

        dash.stop()

        assert dash._should_stop is True
        assert dash._server_thread is None
