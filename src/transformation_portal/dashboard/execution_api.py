"""Pipeline execution API with WebSocket streaming.

This module provides FastAPI endpoints for pipeline execution
with real-time status updates via WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from transformation_portal.dashboard.execution_manager import ExecutionManager

logger = logging.getLogger(__name__)

# Global execution manager and WebSocket clients
_manager: Optional[ExecutionManager] = None
_websocket_clients: List[WebSocket] = []


def get_manager() -> ExecutionManager:
    """Get or create the global execution manager."""
    global _manager
    if _manager is None:
        _manager = ExecutionManager()
    return _manager


def set_manager(manager: ExecutionManager) -> None:
    """Set the global execution manager.

    Args:
        manager: ExecutionManager instance
    """
    global _manager
    _manager = manager


async def broadcast(msg: Dict[str, Any]) -> None:
    """Broadcast message to all connected WebSocket clients.

    Args:
        msg: Message to broadcast
    """
    disconnected = []
    for client in _websocket_clients:
        try:
            await client.send_json(msg)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        if client in _websocket_clients:
            _websocket_clients.remove(client)


def create_execution_router() -> APIRouter:
    """Create the execution API router.

    Returns:
        FastAPI APIRouter with execution endpoints
    """
    router = APIRouter(prefix="/api/exec", tags=["execution"])

    @router.websocket("/ws")
    async def execution_websocket(websocket: WebSocket) -> None:
        """WebSocket endpoint for execution status streaming."""
        await websocket.accept()
        _websocket_clients.append(websocket)
        logger.info("Execution WebSocket connected (%d clients)", len(_websocket_clients))

        try:
            while True:
                # Keep connection alive, handle pings
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "heartbeat"})
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Execution WebSocket error: %s", exc)
        finally:
            if websocket in _websocket_clients:
                _websocket_clients.remove(websocket)
            logger.info("Execution WebSocket disconnected (%d clients)", len(_websocket_clients))

    @router.post("/run", status_code=202)
    async def execute_pipeline(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
        """Execute a pipeline in the background.

        This endpoint returns immediately with a run_id.
        The actual execution happens asynchronously.
        Connect to /ws for real-time progress updates.

        Args:
            payload: Pipeline definition with nodes and edges

        Returns:
            202 Accepted with run_id for tracking
        """
        manager = get_manager()

        # Allocate run_id up front and start execution in background
        run_id = manager.allocate_run_id()
        try:
            task = manager.start_pipeline_background(run_id, payload, broadcast)
            logger.info("Started background pipeline execution: run_id=%s task=%s", run_id, task.get_name())
        except Exception:
            logger.exception("Failed to start pipeline execution: run_id=%s", run_id)
            raise HTTPException(status_code=500, detail="Failed to start pipeline")

        return JSONResponse(
            content={
                "status": "accepted",
                "run_id": run_id,
                "message": "Pipeline execution started. Connect to /api/exec/ws for live updates.",
            },
            status_code=202,
        )

    @router.get("/runs")
    async def list_runs() -> JSONResponse:
        """List active and recent runs."""
        manager = get_manager()
        return JSONResponse(
            {
                "runs": manager.get_active_runs(),
            }
        )

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        """Get run details."""
        manager = get_manager()
        run = manager.get_run_state(run_id)

        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return JSONResponse(
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "error": run.error,
                "cancel_requested": run.cancel_requested,
                "current_node_id": run.current_node_id,
                "nodes": {
                    node_id: {
                        "status": state.status.value,
                        "start_time": state.start_time,
                        "end_time": state.end_time,
                        "outputs": state.outputs,
                        "error": state.error,
                        "progress": state.progress,
                        "merkle_hash": state.merkle_hash,
                    }
                    for node_id, state in run.nodes.items()
                },
                "results": run.results,
            }
        )

    @router.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> JSONResponse:
        """Cancel a running pipeline.

        Semantics:
        - Active run: returns 202 with status "cancelling"
        - Already cancelled: returns 200 with status "cancelled"
        - Already complete/error: returns 200 with current status
        - Not found: returns 404
        """
        manager = get_manager()
        result_status = await manager.cancel_run(run_id, broadcast)

        if result_status is None:
            raise HTTPException(status_code=404, detail="Run not found")

        # Use 202 for in-progress cancellation, 200 for terminal states
        status_code = 202 if result_status == "cancelling" else 200

        return JSONResponse(
            {"status": result_status, "run_id": run_id},
            status_code=status_code,
        )

    @router.get("/", response_class=HTMLResponse)
    async def execution_ui() -> str:
        """Serve the execution monitoring UI."""
        return get_execution_ui_html()

    return router


def get_execution_ui_html() -> str:
    """Get the execution monitoring UI HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Execution</title>
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
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.25rem; }
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.75rem;
        }
        .status-badge.connected { background: #00d25b; color: #000; }
        .status-badge.disconnected { background: #ff5252; }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: 300px 1fr; gap: 2rem; }
        .card {
            background: #16213e;
            border-radius: 0.5rem;
            border: 1px solid #0f3460;
            overflow: hidden;
        }
        .card-header {
            padding: 1rem;
            background: #0f3460;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .runs-list { max-height: 400px; overflow-y: auto; }
        .run-item {
            padding: 1rem;
            border-bottom: 1px solid #0f3460;
            cursor: pointer;
        }
        .run-item:hover { background: rgba(233, 69, 96, 0.1); }
        .run-item.selected { background: rgba(233, 69, 96, 0.2); }
        .run-item h3 { font-size: 0.9rem; margin-bottom: 0.25rem; }
        .run-item p { font-size: 0.75rem; color: #94a3b8; }
        .status {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 0.2rem;
            font-size: 0.7rem;
            text-transform: uppercase;
        }
        .status.running { background: #ffc107; color: #000; }
        .status.complete { background: #00d25b; color: #000; }
        .status.error { background: #ff5252; }
        .status.pending { background: #94a3b8; color: #000; }
        .status.queued { background: #0d6efd; }
        .status.cancelling { background: #ff9800; color: #000; animation: pulse 1s infinite; }
        .status.cancelled { background: #9e9e9e; color: #000; }
        .status.skipped { background: #607d8b; color: #fff; }
        .nodes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
            padding: 1rem;
        }
        .node-card {
            background: #1a1a2e;
            border-radius: 0.5rem;
            padding: 1rem;
            border-left: 4px solid #94a3b8;
        }
        .node-card.running { border-left-color: #ffc107; animation: pulse 1s infinite; }
        .node-card.complete { border-left-color: #00d25b; }
        .node-card.error { border-left-color: #ff5252; }
        .node-card.skipped { border-left-color: #607d8b; opacity: 0.7; }
        .node-card h4 { font-size: 0.9rem; margin-bottom: 0.5rem; }
        .node-card .meta { font-size: 0.75rem; color: #94a3b8; }
        .node-card .outputs {
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid #0f3460;
            font-size: 0.75rem;
            font-family: monospace;
            max-height: 100px;
            overflow: auto;
        }
        .logs-panel {
            padding: 1rem;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.8rem;
            background: #0a0a15;
        }
        .log-entry {
            padding: 0.25rem 0;
            border-bottom: 1px solid #0f3460;
        }
        .log-entry .time { color: #94a3b8; }
        .log-entry .node { color: #e94560; }
        .log-entry .message { color: #eee; }
        .log-entry.error .message { color: #ff5252; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .empty { padding: 2rem; text-align: center; color: #94a3b8; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Pipeline Execution Monitor</h1>
        <span id="ws-status" class="status-badge disconnected">Disconnected</span>
    </div>
    <div class="container">
        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span>Runs</span>
                    <button onclick="loadRuns()" style="background:#0f3460;border:none;color:#eee;padding:0.25rem 0.5rem;border-radius:0.25rem;cursor:pointer;">Refresh</button>
                </div>
                <div id="runs" class="runs-list">
                    <div class="empty">No runs yet</div>
                </div>
            </div>
            <div>
                <div class="card" style="margin-bottom: 1rem;">
                    <div class="card-header">
                        <span>Nodes</span>
                        <div style="display:flex;align-items:center;gap:0.5rem;">
                            <span id="current-run">No run selected</span>
                            <button id="cancel-btn" onclick="cancelRun()" style="display:none;background:#ff5252;border:none;color:#fff;padding:0.25rem 0.75rem;border-radius:0.25rem;cursor:pointer;font-size:0.75rem;">Cancel</button>
                        </div>
                    </div>
                    <div id="nodes" class="nodes-grid">
                        <div class="empty">Select a run to view nodes</div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">Logs</div>
                    <div id="logs" class="logs-panel">
                        <div class="empty">Waiting for events...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let selectedRunId = null;
        let runs = {};
        let logs = [];

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/api/exec/ws`);

            ws.onopen = () => {
                document.getElementById('ws-status').textContent = 'Connected';
                document.getElementById('ws-status').className = 'status-badge connected';
            };

            ws.onclose = () => {
                document.getElementById('ws-status').textContent = 'Disconnected';
                document.getElementById('ws-status').className = 'status-badge disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            // Add to logs
            if (msg.type !== 'heartbeat') {
                addLog(msg);
            }

            switch (msg.type) {
                case 'run_started':
                    runs[msg.run_id] = {
                        run_id: msg.run_id,
                        status: 'running',
                        start_time: msg.timestamp,
                        nodes: {}
                    };
                    loadRuns();
                    if (!selectedRunId) selectRun(msg.run_id);
                    break;

                case 'node_start':
                    if (runs[msg.run_id]) {
                        runs[msg.run_id].nodes[msg.node] = {
                            status: 'running',
                            start_time: msg.timestamp
                        };
                        if (selectedRunId === msg.run_id) renderNodes();
                    }
                    break;

                case 'node_complete':
                    if (runs[msg.run_id]) {
                        runs[msg.run_id].nodes[msg.node] = {
                            status: 'complete',
                            outputs: msg.outputs,
                            merkle_hash: msg.merkle_hash
                        };
                        if (selectedRunId === msg.run_id) renderNodes();
                    }
                    break;

                case 'node_error':
                    if (runs[msg.run_id]) {
                        runs[msg.run_id].nodes[msg.node] = {
                            status: 'error',
                            error: msg.error
                        };
                        if (selectedRunId === msg.run_id) renderNodes();
                    }
                    break;

                case 'node_skipped':
                    if (runs[msg.run_id]) {
                        runs[msg.run_id].nodes[msg.node] = {
                            status: 'skipped',
                            reason: msg.reason
                        };
                        if (selectedRunId === msg.run_id) renderNodes();
                    }
                    break;

                case 'run_cancelling':
                    if (runs[msg.run_id]) {
                        runs[msg.run_id].status = 'cancelling';
                        loadRuns();
                        updateCancelButton();
                    }
                    break;

                case 'run_cancelled':
                    if (runs[msg.run_id]) {
                        runs[msg.run_id].status = 'cancelled';
                        loadRuns();
                        updateCancelButton();
                    }
                    break;

                case 'run_complete':
                    if (runs[msg.run_id]) {
                        runs[msg.run_id].status = 'complete';
                        runs[msg.run_id].results = msg.results;
                        loadRuns();
                        updateCancelButton();
                    }
                    break;

                case 'run_error':
                    if (runs[msg.run_id]) {
                        runs[msg.run_id].status = 'error';
                        runs[msg.run_id].error = msg.error;
                        loadRuns();
                        updateCancelButton();
                    }
                    break;
            }
        }

        function addLog(msg) {
            const time = new Date().toLocaleTimeString();
            logs.unshift({ time, ...msg });
            if (logs.length > 100) logs.pop();
            renderLogs();
        }

        function renderLogs() {
            const container = document.getElementById('logs');
            container.innerHTML = logs.map(log => `
                <div class="log-entry ${log.type === 'node_error' || log.type === 'run_error' ? 'error' : ''}">
                    <span class="time">[${log.time}]</span>
                    ${log.node ? `<span class="node">[${log.node}]</span>` : ''}
                    <span class="message">${log.type}${log.message ? ': ' + log.message : ''}${log.error ? ': ' + log.error : ''}</span>
                </div>
            `).join('');
        }

        async function loadRuns() {
            try {
                const res = await fetch('/api/exec/runs');
                const data = await res.json();

                data.runs.forEach(r => {
                    if (!runs[r.run_id]) {
                        runs[r.run_id] = r;
                    } else {
                        runs[r.run_id].status = r.status;
                    }
                });

                renderRuns();
            } catch (e) {
                console.error('Failed to load runs:', e);
            }
        }

        function renderRuns() {
            const container = document.getElementById('runs');
            const runList = Object.values(runs).sort((a, b) =>
                (b.start_time || '').localeCompare(a.start_time || '')
            );

            if (runList.length === 0) {
                container.innerHTML = '<div class="empty">No runs yet</div>';
                return;
            }

            container.innerHTML = runList.map(r => `
                <div class="run-item ${selectedRunId === r.run_id ? 'selected' : ''}"
                     onclick="selectRun('${r.run_id}')">
                    <h3>Run ${r.run_id}</h3>
                    <p>
                        <span class="status ${r.status}">${r.status}</span>
                        ${r.start_time ? new Date(r.start_time).toLocaleString() : ''}
                    </p>
                </div>
            `).join('');
        }

        async function selectRun(runId) {
            selectedRunId = runId;
            document.getElementById('current-run').textContent = `Run ${runId}`;
            renderRuns();

            // Fetch full run details
            try {
                const res = await fetch(`/api/exec/runs/${runId}`);
                const data = await res.json();
                runs[runId] = { ...runs[runId], ...data };
                renderNodes();
                updateCancelButton();
            } catch (e) {
                renderNodes();
                updateCancelButton();
            }
        }

        function updateCancelButton() {
            const btn = document.getElementById('cancel-btn');
            if (!selectedRunId || !runs[selectedRunId]) {
                btn.style.display = 'none';
                return;
            }
            const run = runs[selectedRunId];
            // Show button only for active runs (running, pending)
            if (run.status === 'running' || run.status === 'pending') {
                btn.style.display = 'inline-block';
                btn.disabled = false;
                btn.textContent = 'Cancel';
            } else if (run.status === 'cancelling') {
                btn.style.display = 'inline-block';
                btn.disabled = true;
                btn.textContent = 'Cancelling...';
            } else {
                btn.style.display = 'none';
            }
        }

        async function cancelRun() {
            if (!selectedRunId) return;
            const btn = document.getElementById('cancel-btn');
            btn.disabled = true;
            btn.textContent = 'Cancelling...';
            
            try {
                const res = await fetch(`/api/exec/runs/${selectedRunId}/cancel`, { method: 'POST' });
                const data = await res.json();
                if (runs[selectedRunId]) {
                    runs[selectedRunId].status = data.status;
                }
                loadRuns();
                updateCancelButton();
            } catch (e) {
                console.error('Failed to cancel run:', e);
                btn.disabled = false;
                btn.textContent = 'Cancel';
            }
        }

        function renderNodes() {
            const container = document.getElementById('nodes');
            const run = runs[selectedRunId];

            if (!run || !run.nodes || Object.keys(run.nodes).length === 0) {
                container.innerHTML = '<div class="empty">No nodes</div>';
                return;
            }

            container.innerHTML = Object.entries(run.nodes).map(([nodeId, node]) => `
                <div class="node-card ${node.status || 'pending'}">
                    <h4>${nodeId}</h4>
                    <div class="meta">
                        <span class="status ${node.status || 'pending'}">${node.status || 'pending'}</span>
                        ${node.merkle_hash ? `<br>Hash: ${node.merkle_hash.slice(0, 8)}...` : ''}
                    </div>
                    ${node.outputs ? `
                        <div class="outputs">
                            ${JSON.stringify(node.outputs, null, 2)}
                        </div>
                    ` : ''}
                    ${node.error ? `<div class="outputs" style="color:#ff5252">${node.error}</div>` : ''}
                </div>
            `).join('');
        }

        // Heartbeat
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 25000);

        // Check for run_id in URL parameters for auto-selection
        const urlParams = new URLSearchParams(window.location.search);
        const targetRunId = urlParams.get('run_id');

        connect();
        loadRuns().then(() => {
            // Auto-select the run from URL parameter after loading runs
            if (targetRunId) {
                selectRun(targetRunId);
            }
        });
    </script>
</body>
</html>"""
