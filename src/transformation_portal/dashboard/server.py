"""FastAPI server for real-time evaluation dashboard.

This module provides a FastAPI application with WebSocket support
for streaming evaluation results and pipeline status in real-time.

Usage:
    # Start server
    uvicorn transformation_portal.dashboard.server:app --reload

    # Or programmatically
    server = DashboardServer()
    server.start(host="0.0.0.0", port=8000)

Features:
    - WebSocket streaming for real-time updates
    - REST API for status and history
    - Static file serving for frontend
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    import uvicorn
    from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None


@dataclass
class DashboardEvent:
    """Event for dashboard streaming.

    Attributes:
        event_type: Type of event (eval, progress, error, etc.)
        data: Event payload
        timestamp: ISO timestamp
        source: Source identifier (node ID, pipeline ID, etc.)
    """

    event_type: str
    data: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
        }


# Global state for WebSocket connections
_max_history = 1000
_connected_clients: list[Any] = []
# Bounded on every append path (see ``_record_event``); ``maxlen`` is a hard backstop.
_event_history: deque[DashboardEvent] = deque(maxlen=_max_history)
_history_lock = threading.Lock()
# Event loop that owns the connected WebSocket clients (the serving loop). Captured when
# a client connects or an event is posted, so producers on other threads can hand
# broadcasts to it with ``run_coroutine_threadsafe`` instead of silently dropping them.
_serving_loop: Optional[asyncio.AbstractEventLoop] = None
# Strong references to in-flight broadcast tasks so they are not garbage-collected.
_pending_broadcasts: set[asyncio.Task[None]] = set()


def _record_event(event: DashboardEvent) -> None:
    """Append ``event`` to the history, keeping at most ``_max_history`` entries."""
    with _history_lock:
        _event_history.append(event)
        while len(_event_history) > max(_max_history, 0):
            _event_history.popleft()


def _recent_events(limit: int) -> list[DashboardEvent]:
    """Return a snapshot of the ``limit`` most recent events (oldest first)."""
    with _history_lock:
        snapshot = list(_event_history)
    return snapshot[-limit:] if limit > 0 else []


def _capture_serving_loop() -> None:
    """Remember the running loop as the one that owns the WebSocket clients."""
    global _serving_loop
    _serving_loop = asyncio.get_running_loop()


def _schedule_broadcast(event: DashboardEvent) -> None:
    """Deliver ``event`` to live clients on the serving loop, from any thread."""
    loop = _serving_loop
    if loop is None or loop.is_closed():
        # No client has ever connected to a running server: nothing to deliver.
        return
    try:
        running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        task = loop.create_task(_broadcast_to_clients(event))
        _pending_broadcasts.add(task)
        task.add_done_callback(_pending_broadcasts.discard)
    elif loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast_to_clients(event), loop)
    else:
        logger.debug("Dashboard serving loop is not running; event %s kept in history only", event.event_type)


async def _broadcast_to_clients(event: DashboardEvent) -> None:
    """Broadcast event to all connected WebSocket clients."""
    if not _connected_clients:
        return

    message = json.dumps(event.to_dict())
    disconnected = []

    for client in _connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.append(client)

    # Remove disconnected clients
    for client in disconnected:
        if client in _connected_clients:
            _connected_clients.remove(client)


def broadcast_event(
    event_type: str,
    data: dict[str, Any],
    source: Optional[str] = None,
) -> None:
    """Broadcast an event to all dashboard clients.

    This is a synchronous wrapper that queues the event for async broadcast. It is
    safe to call from any thread: the broadcast is scheduled on the loop that serves
    the connected WebSocket clients.

    Args:
        event_type: Type of event
        data: Event data
        source: Event source identifier

    Example:
        >>> broadcast_event("eval", {"score": 0.85, "passes": True}, source="quality_node")
    """
    event = DashboardEvent(
        event_type=event_type,
        data=data,
        source=source,
    )

    _record_event(event)
    _schedule_broadcast(event)


def create_app() -> "FastAPI":
    """Create the FastAPI application.

    Returns:
        Configured FastAPI app

    Raises:
        ImportError: If FastAPI is not installed
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for the dashboard. " "Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="APEX Evaluation Dashboard",
        description="Real-time evaluation streaming and monitoring",
        version="1.0.0",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the dashboard frontend."""
        return _get_index_html()

    @app.get("/api/status")
    async def status():
        """Get dashboard status."""
        return JSONResponse(
            {
                "status": "running",
                "connected_clients": len(_connected_clients),
                "event_history_size": len(_event_history),
            }
        )

    @app.get("/api/history")
    async def history(limit: int = Query(100, ge=1, le=_max_history)):
        """Get recent event history."""
        events = _recent_events(limit)
        return JSONResponse(
            {
                "count": len(events),
                "events": [e.to_dict() for e in events],
            }
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time streaming."""
        await websocket.accept()
        _capture_serving_loop()
        _connected_clients.append(websocket)
        logger.info("WebSocket client connected (%d total)", len(_connected_clients))

        try:
            # Send recent history on connect
            for event in _recent_events(10):
                await websocket.send_text(json.dumps(event.to_dict()))

            # Keep connection alive
            while True:
                try:
                    # Wait for messages (ping/pong)
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0,
                    )
                    # Echo back pings
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "heartbeat",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    )
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("WebSocket error: %s", exc)
        finally:
            if websocket in _connected_clients:
                _connected_clients.remove(websocket)
            logger.info("WebSocket client disconnected (%d total)", len(_connected_clients))

    @app.post("/api/event")
    async def post_event(event_type: str, data: dict = None, source: str = None):
        """Post an event to the dashboard."""
        event = DashboardEvent(
            event_type=event_type,
            data=data or {},
            source=source,
        )
        _capture_serving_loop()
        _record_event(event)
        await _broadcast_to_clients(event)
        return JSONResponse({"status": "ok", "event": event.to_dict()})

    return app


def _get_index_html() -> str:
    """Get the dashboard frontend HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APEX Evaluation Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 1rem 2rem;
            border-bottom: 1px solid #0f3460;
        }
        .header h1 { font-size: 1.5rem; }
        .header .status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            margin-left: 1rem;
        }
        .status.connected { background: #00d25b; color: #000; }
        .status.disconnected { background: #ff5252; }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        .card {
            background: #16213e;
            border-radius: 0.5rem;
            padding: 1.5rem;
            border: 1px solid #0f3460;
        }
        .card h2 {
            font-size: 1rem;
            color: #94a3b8;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
        .metric {
            background: #1a1a2e;
            padding: 1rem;
            border-radius: 0.25rem;
            text-align: center;
        }
        .metric .value { font-size: 2rem; font-weight: bold; color: #e94560; }
        .metric .label { font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; }
        .events {
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.8rem;
        }
        .event {
            padding: 0.75rem;
            border-bottom: 1px solid #0f3460;
            animation: fadeIn 0.3s ease;
        }
        .event:last-child { border-bottom: none; }
        .event .time { color: #94a3b8; font-size: 0.7rem; }
        .event .type {
            display: inline-block;
            padding: 0.1rem 0.4rem;
            border-radius: 0.2rem;
            font-size: 0.7rem;
            margin: 0 0.5rem;
        }
        .type.eval { background: #00d25b; color: #000; }
        .type.progress { background: #0d6efd; }
        .type.error { background: #ff5252; }
        .type.heartbeat { background: #333; }
        .event .data { color: #94a3b8; margin-top: 0.25rem; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>APEX Evaluation Dashboard</h1>
        <span id="status" class="status disconnected">Disconnected</span>
    </div>
    <div class="container">
        <div class="grid">
            <div class="card">
                <h2>Latest Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <div id="score" class="value">--</div>
                        <div class="label">Score</div>
                    </div>
                    <div class="metric">
                        <div id="psnr" class="value">--</div>
                        <div class="label">PSNR</div>
                    </div>
                    <div class="metric">
                        <div id="events-count" class="value">0</div>
                        <div class="label">Events</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>Event Stream</h2>
                <div id="events" class="events"></div>
            </div>
        </div>
    </div>
    <script>
        const eventsEl = document.getElementById('events');
        const statusEl = document.getElementById('status');
        const scoreEl = document.getElementById('score');
        const psnrEl = document.getElementById('psnr');
        const countEl = document.getElementById('events-count');
        let eventCount = 0;
        let ws;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                statusEl.textContent = 'Connected';
                statusEl.className = 'status connected';
            };

            ws.onclose = () => {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'status disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                addEvent(data);
                updateMetrics(data);
            };
        }

        function addEvent(data) {
            eventCount++;
            countEl.textContent = eventCount;

            const eventEl = document.createElement('div');
            eventEl.className = 'event';

            // Event fields are client-supplied (POST /api/event): build the DOM with
            // textContent only, never by parsing them as markup.
            const time = new Date(data.timestamp).toLocaleTimeString();
            const typeText = String(data.type ?? '');

            const timeEl = document.createElement('span');
            timeEl.className = 'time';
            timeEl.textContent = time;
            eventEl.appendChild(timeEl);

            const typeEl = document.createElement('span');
            typeEl.className = 'type';
            if (/^[a-z0-9_-]+$/i.test(typeText)) {
                typeEl.classList.add(typeText.toLowerCase());
            }
            typeEl.textContent = typeText;
            eventEl.appendChild(typeEl);

            if (data.source) {
                const sourceEl = document.createElement('span');
                sourceEl.className = 'source';
                sourceEl.textContent = String(data.source);
                eventEl.appendChild(sourceEl);
            }

            const dataEl = document.createElement('div');
            dataEl.className = 'data';
            dataEl.textContent = JSON.stringify(data.data, null, 2);
            eventEl.appendChild(dataEl);

            eventsEl.insertBefore(eventEl, eventsEl.firstChild);

            // Limit events shown
            while (eventsEl.children.length > 50) {
                eventsEl.removeChild(eventsEl.lastChild);
            }
        }

        function updateMetrics(data) {
            if (data.type === 'eval' && data.data) {
                if (data.data.score !== undefined) {
                    scoreEl.textContent = data.data.score.toFixed(2);
                }
                if (data.data.psnr !== undefined) {
                    psnrEl.textContent = data.data.psnr.toFixed(1);
                }
            }
        }

        // Heartbeat
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 25000);

        connect();
    </script>
</body>
</html>"""


class DashboardServer:
    """Convenience wrapper for running the dashboard server.

    Example:
        >>> server = DashboardServer()
        >>> server.start(port=8000)  # Blocks
        >>>
        >>> # Or in background
        >>> server.start_background(port=8000)
        >>> # ... do other things ...
        >>> server.stop()
    """

    def __init__(self) -> None:
        """Initialize dashboard server."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for the dashboard. " "Install with: pip install fastapi uvicorn")

        self.app = create_app()
        self._server_thread: Optional[threading.Thread] = None
        self._should_stop = False

    def start(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        """Start the server (blocking).

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        logger.info("Starting dashboard server at http://%s:%d", host, port)
        uvicorn.run(self.app, host=host, port=port)

    def start_background(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        """Start the server in a background thread.

        Args:
            host: Host to bind to
            port: Port to bind to
        """

        def run():
            uvicorn.run(self.app, host=host, port=port)

        self._server_thread = threading.Thread(target=run, daemon=True)
        self._server_thread.start()
        logger.info("Dashboard server started in background at http://%s:%d", host, port)

    def stop(self) -> None:
        """Stop the background server."""
        self._should_stop = True
        if self._server_thread:
            self._server_thread = None


# Create default app instance for uvicorn
app = create_app() if FASTAPI_AVAILABLE else None
