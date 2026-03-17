"""Time travel and visual diff system for ML debugging.

This module provides:
- Node version history across runs (time travel)
- Image diff with interactive slider
- 3D model comparison with synchronized cameras
- Merkle lineage integration
- Semantic diff support (LLaVA-powered)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformation_portal.dashboard.node_state_store import NodeStateStore
    from transformation_portal.storage.merkle_dag import MerkleDAG

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import APIRouter, HTTPException, Query
    from fastapi.responses import JSONResponse, HTMLResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None


# Global references
_node_store: Optional["NodeStateStore"] = None  # type: ignore
_merkle_dag: Optional["MerkleDAG"] = None  # type: ignore


def set_time_travel_store(store: "NodeStateStore") -> None:  # type: ignore
    """Set the node state store for time travel."""
    global _node_store
    _node_store = store


def set_time_travel_merkle(dag: "MerkleDAG") -> None:  # type: ignore
    """Set the Merkle DAG for lineage."""
    global _merkle_dag
    _merkle_dag = dag


def create_time_travel_router() -> "APIRouter":
    """Create the time travel router.

    Returns:
        FastAPI APIRouter with time travel endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for time travel")

    router = APIRouter(prefix="/api/timetravel", tags=["timetravel"])

    @router.get("/nodes/{node_id}/history")
    async def get_node_history(node_id: str, limit: int = Query(default=50)):
        """Get version history for a node across all runs.

        Args:
            node_id: Node identifier
            limit: Maximum number of versions to return
        """
        if _node_store is None:
            return JSONResponse({"history": [], "error": "Store not configured"})

        history = []

        for run_id, run_state in _node_store.runs.items():
            if node_id in run_state.nodes:
                node = run_state.nodes[node_id]

                entry = {
                    "run_id": run_id,
                    "status": node.status,
                    "outputs": node.outputs,
                    "artifacts": node.artifacts,
                    "merkle_hash": node.merkle_hash,
                    "start_time": node.start_time,
                    "end_time": node.end_time,
                }

                # Add Merkle lineage if available
                if _merkle_dag and node.merkle_hash:
                    merkle_node = _merkle_dag.get_node(node.merkle_hash)
                    if merkle_node:
                        entry["lineage"] = {
                            "inputs": list(merkle_node.inputs),
                            "metadata": merkle_node.metadata,
                        }

                history.append(entry)

        # Sort by timestamp (most recent first)
        history.sort(
            key=lambda x: x.get("end_time") or x.get("start_time") or "",
            reverse=True,
        )

        return JSONResponse({
            "node_id": node_id,
            "total_versions": len(history),
            "history": history[:limit],
        })

    @router.get("/runs/{run_id}/snapshot")
    async def get_run_snapshot(run_id: str):
        """Get full snapshot of a run for comparison.

        Args:
            run_id: Run identifier
        """
        if _node_store is None:
            raise HTTPException(status_code=503, detail="Store not configured")

        run = _node_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return JSONResponse({
            "run_id": run.run_id,
            "status": run.status,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "nodes": {
                node_id: {
                    "status": node.status,
                    "outputs": node.outputs,
                    "artifacts": node.artifacts,
                    "merkle_hash": node.merkle_hash,
                }
                for node_id, node in run.nodes.items()
            },
        })

    @router.get("/compare")
    async def compare_artifacts(
        hash_a: str = Query(...),
        hash_b: str = Query(...),
    ):
        """Get comparison metadata for two artifacts.

        Args:
            hash_a: First artifact hash
            hash_b: Second artifact hash
        """
        return JSONResponse({
            "hash_a": hash_a,
            "hash_b": hash_b,
            "url_a": f"/api/preview/artifact/{hash_a}/raw",
            "url_b": f"/api/preview/artifact/{hash_b}/raw",
        })

    @router.get("/", response_class=HTMLResponse)
    async def time_travel_ui():
        """Serve the time travel UI."""
        return get_time_travel_html()

    @router.get("/diff", response_class=HTMLResponse)
    async def diff_viewer():
        """Serve the visual diff viewer."""
        return get_diff_viewer_html()

    return router


def get_time_travel_html() -> str:
    """Get the time travel UI HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Time Travel Debugger</title>
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
        .header-controls {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .header-controls input, .header-controls select {
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-size: 0.85rem;
        }
        .header-controls button {
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: 350px 1fr; gap: 2rem; }
        .card {
            background: #16213e;
            border-radius: 0.5rem;
            border: 1px solid #0f3460;
            overflow: hidden;
        }
        .card-header {
            background: #0f3460;
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .card-body { padding: 1rem; }
        .version-list { max-height: 500px; overflow-y: auto; }
        .version-item {
            padding: 0.75rem;
            border-bottom: 1px solid #0f3460;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .version-item:hover { background: rgba(233, 69, 96, 0.1); }
        .version-item.selected { background: rgba(233, 69, 96, 0.2); border-left: 3px solid #e94560; }
        .version-item .run-id { font-family: monospace; font-size: 0.8rem; }
        .version-item .time { font-size: 0.7rem; color: #94a3b8; }
        .version-item .status {
            padding: 0.15rem 0.4rem;
            border-radius: 0.2rem;
            font-size: 0.65rem;
            text-transform: uppercase;
        }
        .status.complete { background: #00d25b; color: #000; }
        .status.error { background: #ff5252; }
        .status.running { background: #ffc107; color: #000; }
        .compare-panel { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #0f3460; }
        .compare-panel h4 { font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; }
        .compare-slots { display: flex; gap: 0.5rem; margin-bottom: 0.75rem; }
        .compare-slot {
            flex: 1;
            background: #1a1a2e;
            padding: 0.5rem;
            border-radius: 0.25rem;
            text-align: center;
            font-size: 0.75rem;
            min-height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px dashed #0f3460;
        }
        .compare-slot.filled { border-style: solid; border-color: #e94560; }
        .compare-slot .label { color: #94a3b8; }
        .compare-btn {
            width: 100%;
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.5rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .compare-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .details-panel { }
        .detail-section { margin-bottom: 1rem; }
        .detail-section h4 {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }
        .detail-content {
            background: #1a1a2e;
            padding: 0.75rem;
            border-radius: 0.25rem;
            font-family: monospace;
            font-size: 0.75rem;
            max-height: 200px;
            overflow: auto;
        }
        .artifact-list { list-style: none; }
        .artifact-item {
            padding: 0.5rem;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .artifact-item:last-child { border-bottom: none; }
        .artifact-hash { font-family: monospace; font-size: 0.75rem; color: #94a3b8; }
        .artifact-actions button {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 0.25rem 0.5rem;
            border-radius: 0.2rem;
            cursor: pointer;
            font-size: 0.65rem;
            margin-left: 0.25rem;
        }
        .artifact-actions button:hover { background: #e94560; }
        .lineage { font-size: 0.75rem; }
        .lineage .hash { color: #e94560; font-family: monospace; }
        .empty { color: #94a3b8; font-style: italic; text-align: center; padding: 2rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🕐 Time Travel Debugger</h1>
        <div class="header-controls">
            <input type="text" id="node-input" placeholder="Enter node ID...">
            <button onclick="loadHistory()">Load History</button>
            <button onclick="window.location.href='/api/timetravel/diff'">Open Diff Viewer</button>
        </div>
    </div>
    <div class="container">
        <div class="grid">
            <div>
                <div class="card">
                    <div class="card-header">Version History</div>
                    <div id="version-list" class="version-list">
                        <div class="empty">Enter a node ID to load history</div>
                    </div>
                </div>
                <div class="card" style="margin-top: 1rem;">
                    <div class="card-header">Compare Versions</div>
                    <div class="card-body">
                        <div class="compare-slots">
                            <div class="compare-slot" id="slot-a">
                                <span class="label">Version A<br>(click to set)</span>
                            </div>
                            <div class="compare-slot" id="slot-b">
                                <span class="label">Version B<br>(click to set)</span>
                            </div>
                        </div>
                        <button class="compare-btn" id="compare-btn" onclick="openComparison()" disabled>
                            Compare Selected Versions
                        </button>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">Version Details</div>
                <div class="card-body details-panel" id="details-panel">
                    <div class="empty">Select a version to view details</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let history = [];
        let selectedVersion = null;
        let compareA = null;
        let compareB = null;
        let selectingSlot = null;

        async function loadHistory() {
            const nodeId = document.getElementById('node-input').value.trim();
            if (!nodeId) return;

            try {
                const res = await fetch(`/api/timetravel/nodes/${nodeId}/history`);
                const data = await res.json();
                history = data.history || [];
                renderVersionList();
            } catch (e) {
                alert('Failed to load history: ' + e.message);
            }
        }

        function renderVersionList() {
            const container = document.getElementById('version-list');

            if (history.length === 0) {
                container.innerHTML = '<div class="empty">No history found for this node</div>';
                return;
            }

            container.innerHTML = history.map((v, i) => `
                <div class="version-item ${selectedVersion === i ? 'selected' : ''}"
                     onclick="selectVersion(${i})"
                     ondblclick="setCompareSlot(${i})">
                    <div>
                        <div class="run-id">${v.run_id}</div>
                        <div class="time">${v.end_time ? new Date(v.end_time).toLocaleString() : 'In progress'}</div>
                    </div>
                    <span class="status ${v.status}">${v.status}</span>
                </div>
            `).join('');
        }

        function selectVersion(index) {
            selectedVersion = index;
            renderVersionList();
            renderDetails(history[index]);

            // If a slot is being selected, set it
            if (selectingSlot) {
                setCompareSlot(index);
            }
        }

        function renderDetails(version) {
            const panel = document.getElementById('details-panel');

            let html = `
                <div class="detail-section">
                    <h4>Run Info</h4>
                    <div class="detail-content">
                        <div>Run ID: ${version.run_id}</div>
                        <div>Status: ${version.status}</div>
                        <div>Start: ${version.start_time || 'N/A'}</div>
                        <div>End: ${version.end_time || 'N/A'}</div>
                        ${version.merkle_hash ? `<div>Merkle: ${version.merkle_hash.slice(0, 16)}...</div>` : ''}
                    </div>
                </div>
            `;

            // Outputs
            if (version.outputs && Object.keys(version.outputs).length > 0) {
                html += `
                    <div class="detail-section">
                        <h4>Outputs</h4>
                        <div class="detail-content">
                            <pre>${JSON.stringify(version.outputs, null, 2)}</pre>
                        </div>
                    </div>
                `;
            }

            // Artifacts
            if (version.artifacts && Object.keys(version.artifacts).length > 0) {
                html += `
                    <div class="detail-section">
                        <h4>Artifacts</h4>
                        <ul class="artifact-list">
                            ${Object.entries(version.artifacts).map(([name, hash]) => `
                                <li class="artifact-item">
                                    <div>
                                        <strong>${name}</strong><br>
                                        <span class="artifact-hash">${hash}</span>
                                    </div>
                                    <div class="artifact-actions">
                                        <button onclick="previewArtifact('${hash}')">Preview</button>
                                        <button onclick="openIn3D('${hash}')">3D View</button>
                                    </div>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                `;
            }

            // Lineage
            if (version.lineage) {
                html += `
                    <div class="detail-section">
                        <h4>Lineage</h4>
                        <div class="detail-content lineage">
                            <div>Inputs: ${version.lineage.inputs.map(h =>
                                `<span class="hash">${h.slice(0, 8)}...</span>`
                            ).join(', ') || 'None'}</div>
                        </div>
                    </div>
                `;
            }

            panel.innerHTML = html;
        }

        function setCompareSlot(index) {
            const version = history[index];

            if (!compareA) {
                compareA = version;
                document.getElementById('slot-a').innerHTML = `
                    <div>
                        <strong>Version A</strong><br>
                        <span style="font-size:0.7rem;">${version.run_id}</span>
                    </div>
                `;
                document.getElementById('slot-a').classList.add('filled');
            } else if (!compareB) {
                compareB = version;
                document.getElementById('slot-b').innerHTML = `
                    <div>
                        <strong>Version B</strong><br>
                        <span style="font-size:0.7rem;">${version.run_id}</span>
                    </div>
                `;
                document.getElementById('slot-b').classList.add('filled');
            }

            updateCompareButton();
        }

        function updateCompareButton() {
            const btn = document.getElementById('compare-btn');
            btn.disabled = !(compareA && compareB);
        }

        function clearCompare() {
            compareA = null;
            compareB = null;
            document.getElementById('slot-a').innerHTML = '<span class="label">Version A<br>(click to set)</span>';
            document.getElementById('slot-b').innerHTML = '<span class="label">Version B<br>(click to set)</span>';
            document.getElementById('slot-a').classList.remove('filled');
            document.getElementById('slot-b').classList.remove('filled');
            updateCompareButton();
        }

        function openComparison() {
            if (!compareA || !compareB) return;

            // Get first artifact from each
            const hashA = Object.values(compareA.artifacts || {})[0];
            const hashB = Object.values(compareB.artifacts || {})[0];

            if (!hashA || !hashB) {
                alert('Both versions need artifacts to compare');
                return;
            }

            window.open(`/api/timetravel/diff?hash_a=${hashA}&hash_b=${hashB}`, '_blank');
        }

        function previewArtifact(hash) {
            window.open(`/api/preview/?hash=${hash}`, '_blank');
        }

        function openIn3D(hash) {
            window.open(`/api/studio/?hash=${hash}`, '_blank');
        }

        // Slot click handlers
        document.getElementById('slot-a').addEventListener('click', () => {
            if (compareA) {
                compareA = null;
                document.getElementById('slot-a').innerHTML = '<span class="label">Version A<br>(click version)</span>';
                document.getElementById('slot-a').classList.remove('filled');
                updateCompareButton();
            }
        });

        document.getElementById('slot-b').addEventListener('click', () => {
            if (compareB) {
                compareB = null;
                document.getElementById('slot-b').innerHTML = '<span class="label">Version B<br>(click version)</span>';
                document.getElementById('slot-b').classList.remove('filled');
                updateCompareButton();
            }
        });
    </script>
</body>
</html>"""


def get_diff_viewer_html() -> str:
    """Get the visual diff viewer HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Diff Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/RGBELoader.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }
        .header {
            background: #16213e;
            padding: 0.75rem 1.5rem;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 50px;
        }
        .header h1 { font-size: 1rem; }
        .header-controls {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .header-controls input {
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.4rem 0.75rem;
            border-radius: 0.25rem;
            width: 250px;
            font-family: monospace;
            font-size: 0.8rem;
        }
        .header-controls button {
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.4rem 0.75rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .header-controls select {
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.4rem;
            border-radius: 0.25rem;
        }
        .main {
            height: calc(100vh - 50px);
            position: relative;
        }

        /* Image Diff Slider Mode */
        .image-diff {
            position: relative;
            width: 100%;
            height: 100%;
            overflow: hidden;
            display: none;
        }
        .image-diff.active { display: block; }
        .image-diff img {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        }
        .image-diff .img-a { z-index: 1; }
        .image-diff .img-b { z-index: 2; clip-path: inset(0 50% 0 0); }
        .slider-container {
            position: absolute;
            bottom: 2rem;
            left: 50%;
            transform: translateX(-50%);
            width: 80%;
            max-width: 600px;
            z-index: 10;
            background: rgba(22, 33, 62, 0.9);
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
        }
        .slider-container label {
            display: block;
            text-align: center;
            margin-bottom: 0.5rem;
            font-size: 0.8rem;
            color: #94a3b8;
        }
        .slider-container input[type="range"] {
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: #0f3460;
            border-radius: 4px;
        }
        .slider-container input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #e94560;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        .slider-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 0.5rem;
            font-size: 0.7rem;
            color: #94a3b8;
        }
        .divider-line {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 3px;
            background: #e94560;
            z-index: 5;
            pointer-events: none;
        }
        .divider-line::before {
            content: '◀ A | B ▶';
            position: absolute;
            top: 1rem;
            left: 50%;
            transform: translateX(-50%);
            background: #e94560;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.6rem;
            white-space: nowrap;
        }

        /* 3D Side-by-Side Mode */
        .mesh-diff {
            display: none;
            width: 100%;
            height: 100%;
        }
        .mesh-diff.active { display: flex; }
        .viewport {
            flex: 1;
            position: relative;
            border-right: 1px solid #0f3460;
        }
        .viewport:last-child { border-right: none; }
        .viewport-label {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(22, 33, 62, 0.9);
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
            z-index: 10;
        }
        .viewport canvas {
            width: 100% !important;
            height: 100% !important;
        }

        /* Side-by-Side Image Mode */
        .image-sidebyside {
            display: none;
            width: 100%;
            height: 100%;
        }
        .image-sidebyside.active { display: flex; }
        .image-panel {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border-right: 1px solid #0f3460;
            position: relative;
        }
        .image-panel:last-child { border-right: none; }
        .image-panel img {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }
        .image-panel .label {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(22, 33, 62, 0.9);
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
        }

        /* Sync indicator */
        .sync-indicator {
            position: fixed;
            bottom: 1rem;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(233, 69, 96, 0.9);
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            z-index: 100;
            display: none;
        }
        .sync-indicator.visible { display: block; }

        /* Loading */
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #0f3460;
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Visual Diff Viewer</h1>
        <div class="header-controls">
            <input type="text" id="hash-a" placeholder="Hash A (before)">
            <input type="text" id="hash-b" placeholder="Hash B (after)">
            <select id="diff-mode" onchange="switchMode(this.value)">
                <option value="slider">Image Slider</option>
                <option value="sidebyside">Side by Side (Image)</option>
                <option value="mesh">3D Comparison</option>
            </select>
            <button onclick="loadDiff()">Compare</button>
            <button onclick="swapInputs()">⇄ Swap</button>
        </div>
    </div>
    <div class="main">
        <!-- Image Slider Mode -->
        <div class="image-diff" id="image-diff">
            <img class="img-a" id="img-a" src="">
            <img class="img-b" id="img-b" src="">
            <div class="divider-line" id="divider-line"></div>
            <div class="slider-container">
                <label>Drag to compare: <span id="slider-value">50%</span></label>
                <input type="range" min="0" max="100" value="50" id="diff-slider" oninput="updateSlider(this.value)">
                <div class="slider-labels">
                    <span>◀ Before (A)</span>
                    <span>After (B) ▶</span>
                </div>
            </div>
        </div>

        <!-- Side by Side Image Mode -->
        <div class="image-sidebyside" id="image-sidebyside">
            <div class="image-panel">
                <span class="label">Before (A)</span>
                <img id="side-img-a" src="">
            </div>
            <div class="image-panel">
                <span class="label">After (B)</span>
                <img id="side-img-b" src="">
            </div>
        </div>

        <!-- 3D Mesh Mode -->
        <div class="mesh-diff" id="mesh-diff">
            <div class="viewport" id="viewport-a">
                <span class="viewport-label">Before (A)</span>
            </div>
            <div class="viewport" id="viewport-b">
                <span class="viewport-label">After (B)</span>
            </div>
        </div>

        <div class="sync-indicator" id="sync-indicator">🔗 Views Synchronized</div>
    </div>

    <script>
        let currentMode = 'slider';
        let viewerA = null;
        let viewerB = null;
        let syncEnabled = true;

        // Check URL params
        const params = new URLSearchParams(window.location.search);
        if (params.get('hash_a')) {
            document.getElementById('hash-a').value = params.get('hash_a');
        }
        if (params.get('hash_b')) {
            document.getElementById('hash-b').value = params.get('hash_b');
        }
        if (params.get('hash_a') && params.get('hash_b')) {
            setTimeout(loadDiff, 100);
        }

        function switchMode(mode) {
            currentMode = mode;
            document.getElementById('image-diff').classList.remove('active');
            document.getElementById('image-sidebyside').classList.remove('active');
            document.getElementById('mesh-diff').classList.remove('active');
            document.getElementById('sync-indicator').classList.remove('visible');

            if (mode === 'slider') {
                document.getElementById('image-diff').classList.add('active');
            } else if (mode === 'sidebyside') {
                document.getElementById('image-sidebyside').classList.add('active');
            } else if (mode === 'mesh') {
                document.getElementById('mesh-diff').classList.add('active');
                document.getElementById('sync-indicator').classList.add('visible');
            }
        }

        function loadDiff() {
            const hashA = document.getElementById('hash-a').value.trim();
            const hashB = document.getElementById('hash-b').value.trim();

            if (!hashA || !hashB) {
                alert('Please enter both artifact hashes');
                return;
            }

            const urlA = `/api/preview/artifact/${hashA}/raw`;
            const urlB = `/api/preview/artifact/${hashB}/raw`;

            if (currentMode === 'slider') {
                loadImageSlider(urlA, urlB);
            } else if (currentMode === 'sidebyside') {
                loadImageSideBySide(urlA, urlB);
            } else if (currentMode === 'mesh') {
                loadMeshComparison(urlA, urlB);
            }
        }

        function loadImageSlider(urlA, urlB) {
            document.getElementById('img-a').src = urlA;
            document.getElementById('img-b').src = urlB;
            updateSlider(50);
        }

        function loadImageSideBySide(urlA, urlB) {
            document.getElementById('side-img-a').src = urlA;
            document.getElementById('side-img-b').src = urlB;
        }

        function updateSlider(value) {
            const percent = parseInt(value);
            document.getElementById('slider-value').textContent = percent + '%';
            document.getElementById('img-b').style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
            document.getElementById('divider-line').style.left = `calc(50% - 45% + ${percent * 0.9}%)`;
        }

        function swapInputs() {
            const a = document.getElementById('hash-a');
            const b = document.getElementById('hash-b');
            const temp = a.value;
            a.value = b.value;
            b.value = temp;
            loadDiff();
        }

        // 3D Mesh Comparison
        function loadMeshComparison(urlA, urlB) {
            // Cleanup existing viewers
            if (viewerA) {
                document.getElementById('viewport-a').innerHTML = '<span class="viewport-label">Before (A)</span>';
            }
            if (viewerB) {
                document.getElementById('viewport-b').innerHTML = '<span class="viewport-label">After (B)</span>';
            }

            viewerA = createViewer('viewport-a', urlA);
            viewerB = createViewer('viewport-b', urlB);

            // Sync controls
            if (syncEnabled) {
                viewerA.controls.addEventListener('change', () => {
                    viewerB.camera.position.copy(viewerA.camera.position);
                    viewerB.camera.rotation.copy(viewerA.camera.rotation);
                    viewerB.controls.target.copy(viewerA.controls.target);
                });
            }

            animateMesh();
        }

        function createViewer(containerId, url) {
            const container = document.getElementById(containerId);
            const width = container.clientWidth;
            const height = container.clientHeight;

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            renderer.outputEncoding = THREE.sRGBEncoding;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            container.appendChild(renderer.domElement);

            const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
            camera.position.set(2, 1.5, 2);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            // Lighting
            const ambient = new THREE.HemisphereLight(0xffffff, 0x444444, 0.5);
            scene.add(ambient);
            const key = new THREE.DirectionalLight(0xffffff, 1.5);
            key.position.set(5, 10, 7.5);
            scene.add(key);

            // Grid
            const grid = new THREE.GridHelper(10, 20, 0x0f3460, 0x0f3460);
            scene.add(grid);

            // Environment
            const rgbeLoader = new THREE.RGBELoader();
            rgbeLoader.load(
                'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r158/examples/textures/equirectangular/royal_esplanade_1k.hdr',
                (texture) => {
                    texture.mapping = THREE.EquirectangularReflectionMapping;
                    scene.environment = texture;
                }
            );

            // Load model
            const loader = new THREE.GLTFLoader();
            loader.load(url, (gltf) => {
                const model = gltf.scene;

                // Center and scale
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                model.position.sub(center);
                const scale = 2 / Math.max(size.x, size.y, size.z);
                model.scale.setScalar(scale);

                scene.add(model);
            });

            return { scene, camera, renderer, controls };
        }

        function animateMesh() {
            requestAnimationFrame(animateMesh);

            if (viewerA) {
                viewerA.controls.update();
                viewerA.renderer.render(viewerA.scene, viewerA.camera);
            }
            if (viewerB) {
                viewerB.controls.update();
                viewerB.renderer.render(viewerB.scene, viewerB.camera);
            }
        }

        // Handle resize
        window.addEventListener('resize', () => {
            if (currentMode === 'mesh') {
                ['viewport-a', 'viewport-b'].forEach((id, i) => {
                    const container = document.getElementById(id);
                    const viewer = i === 0 ? viewerA : viewerB;
                    if (viewer) {
                        const width = container.clientWidth;
                        const height = container.clientHeight;
                        viewer.camera.aspect = width / height;
                        viewer.camera.updateProjectionMatrix();
                        viewer.renderer.setSize(width, height);
                    }
                });
            }
        });

        // Initialize default mode
        switchMode('slider');
    </script>
</body>
</html>"""
