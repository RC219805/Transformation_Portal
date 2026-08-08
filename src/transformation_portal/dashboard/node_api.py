"""Node inspection API for debugging and observability.

This module provides FastAPI endpoints for inspecting node execution
details including inputs, outputs, artifacts, and logs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformation_portal.storage.cas_store import ArtifactStore

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None

from transformation_portal.dashboard.node_state_store import get_store

# Global CAS reference
_global_cas: Optional["ArtifactStore"] = None  # type: ignore


def set_cas(cas: "ArtifactStore") -> None:  # type: ignore
    """Set the global CAS for artifact resolution.

    Args:
        cas: ArtifactStore instance
    """
    global _global_cas
    _global_cas = cas


def create_node_inspection_router() -> "APIRouter":
    """Create the node inspection router.

    Returns:
        FastAPI APIRouter with node inspection endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for node inspection API")

    router = APIRouter(prefix="/api/inspect", tags=["inspection"])

    @router.get("/runs/{run_id}/nodes/{node_id}")
    async def get_node_details(run_id: str, node_id: str):
        """Get detailed execution state for a node.

        Args:
            run_id: Run identifier
            node_id: Node identifier

        Returns:
            Full node state including inputs, outputs, artifacts, logs
        """
        store = get_store()
        node = store.get_node(run_id, node_id)

        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: run={run_id}, node={node_id}")

        return JSONResponse(node.to_dict())

    @router.get("/runs/{run_id}/nodes/{node_id}/inputs")
    async def get_node_inputs(run_id: str, node_id: str):
        """Get node inputs only."""
        store = get_store()
        node = store.get_node(run_id, node_id)

        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")

        return JSONResponse(
            {
                "node_id": node_id,
                "inputs": node.inputs,
            }
        )

    @router.get("/runs/{run_id}/nodes/{node_id}/outputs")
    async def get_node_outputs(run_id: str, node_id: str):
        """Get node outputs only."""
        store = get_store()
        node = store.get_node(run_id, node_id)

        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")

        return JSONResponse(
            {
                "node_id": node_id,
                "outputs": node.outputs,
            }
        )

    @router.get("/runs/{run_id}/nodes/{node_id}/artifacts")
    async def get_node_artifacts(run_id: str, node_id: str):
        """Get node artifacts with CAS metadata."""
        store = get_store()
        node = store.get_node(run_id, node_id)

        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")

        artifacts = []
        for name, hash in node.artifacts.items():
            artifact_info = {"name": name, "hash": hash}

            # Try to get CAS metadata
            if _global_cas:
                try:
                    obj = _global_cas.get_object(hash)
                except ValueError:
                    obj = None
                if obj:
                    artifact_info["size_bytes"] = obj.size_bytes
                    artifact_info["path"] = str(obj.path)
                    artifact_info["exists"] = obj.path.exists()

            artifacts.append(artifact_info)

        return JSONResponse(
            {
                "node_id": node_id,
                "artifacts": artifacts,
            }
        )

    @router.get("/runs/{run_id}/nodes/{node_id}/logs")
    async def get_node_logs(run_id: str, node_id: str, tail: int = 100):
        """Get node execution logs.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            tail: Number of recent log entries to return
        """
        store = get_store()
        node = store.get_node(run_id, node_id)

        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")

        return JSONResponse(
            {
                "node_id": node_id,
                "logs": node.logs[-tail:],
                "total_logs": len(node.logs),
            }
        )

    @router.get("/runs/{run_id}/summary")
    async def get_run_summary(run_id: str):
        """Get summary of all nodes in a run."""
        store = get_store()
        run = store.get_run(run_id)

        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return JSONResponse(
            {
                "run_id": run.run_id,
                "status": run.status,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "nodes": {
                    node_id: {
                        "status": node.status,
                        "has_inputs": bool(node.inputs),
                        "has_outputs": bool(node.outputs),
                        "artifact_count": len(node.artifacts),
                        "log_count": len(node.logs),
                        "error": node.error,
                    }
                    for node_id, node in run.nodes.items()
                },
            }
        )

    @router.get("/artifact/{hash}")
    async def get_artifact_info(hash: str):
        """Get artifact metadata by CAS hash.

        Args:
            hash: SHA-256 hash of the artifact
        """
        if _global_cas is None:
            raise HTTPException(status_code=503, detail="CAS not configured")

        try:
            obj = _global_cas.get_object(hash)
        except ValueError:
            raise HTTPException(status_code=404, detail="Artifact not found") from None
        if obj is None:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {hash}")

        return JSONResponse(
            {
                "hash": obj.sha256,
                "size_bytes": obj.size_bytes,
                "path": str(obj.path),
                "exists": obj.path.exists(),
            }
        )

    @router.get("/artifact/{hash}/preview")
    async def preview_artifact(hash: str, max_bytes: int = 4096):
        """Preview artifact contents.

        Args:
            hash: SHA-256 hash
            max_bytes: Maximum bytes to return
        """
        if _global_cas is None:
            raise HTTPException(status_code=503, detail="CAS not configured")

        try:
            obj = _global_cas.get_object(hash)
        except ValueError:
            raise HTTPException(status_code=404, detail="Artifact not found") from None
        if obj is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        try:
            with obj.path.open("rb") as f:
                data = f.read(max_bytes)

            # Detect content type
            content_type = "binary"
            preview = None

            # Check for image
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                content_type = "image/png"
            elif data[:2] == b"\xff\xd8":
                content_type = "image/jpeg"
            elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                content_type = "image/webp"
            # Check for JSON
            elif data[:1] in (b"{", b"["):
                try:
                    preview = data.decode("utf-8")
                    content_type = "application/json"
                except Exception:
                    pass
            # Check for text
            else:
                try:
                    preview = data.decode("utf-8")
                    content_type = "text/plain"
                except Exception:
                    preview = data.hex()[:500]

            return JSONResponse(
                {
                    "hash": hash,
                    "content_type": content_type,
                    "size_bytes": obj.size_bytes,
                    "preview": preview,
                    "truncated": obj.size_bytes > max_bytes,
                }
            )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/artifact/{hash}/download")
    async def download_artifact(hash: str):
        """Download artifact file.

        Args:
            hash: SHA-256 hash
        """
        if _global_cas is None:
            raise HTTPException(status_code=503, detail="CAS not configured")

        try:
            obj = _global_cas.get_object(hash)
        except ValueError:
            raise HTTPException(status_code=404, detail="Artifact not found") from None
        if obj is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        if not obj.path.exists():
            raise HTTPException(status_code=404, detail="Artifact file missing")

        return FileResponse(
            path=str(obj.path),
            filename=f"{hash[:16]}.bin",
            media_type="application/octet-stream",
        )

    @router.get("/", response_class=HTMLResponse)
    async def inspection_ui():
        """Serve the node inspection UI."""
        return get_inspection_ui_html()

    return router


def get_inspection_ui_html() -> str:
    """Get the node inspection UI HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Node Inspector</title>
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
        .header h1 { font-size: 1.25rem; }
        .container { padding: 2rem; max-width: 1200px; margin: 0 auto; }
        .selector {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .selector select, .selector input {
            background: #16213e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            min-width: 200px;
        }
        .selector button {
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .panel {
            background: #16213e;
            border-radius: 0.5rem;
            border: 1px solid #0f3460;
            overflow: hidden;
        }
        .panel-header {
            background: #0f3460;
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .panel-content {
            padding: 1rem;
            max-height: 300px;
            overflow: auto;
        }
        .panel-content pre {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.8rem;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .status {
            padding: 0.2rem 0.5rem;
            border-radius: 0.2rem;
            font-size: 0.7rem;
            text-transform: uppercase;
        }
        .status.complete { background: #00d25b; color: #000; }
        .status.running { background: #ffc107; color: #000; }
        .status.error { background: #ff5252; }
        .status.idle { background: #94a3b8; color: #000; }
        .artifact-list { list-style: none; }
        .artifact-item {
            padding: 0.5rem;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .artifact-item:last-child { border-bottom: none; }
        .artifact-hash {
            font-family: monospace;
            font-size: 0.8rem;
            color: #94a3b8;
        }
        .artifact-actions button {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 0.25rem 0.5rem;
            border-radius: 0.2rem;
            cursor: pointer;
            font-size: 0.7rem;
            margin-left: 0.25rem;
        }
        .artifact-actions button:hover { background: #e94560; }
        .log-entry {
            font-family: monospace;
            font-size: 0.75rem;
            padding: 0.25rem 0;
            border-bottom: 1px solid #0f3460;
            color: #94a3b8;
        }
        .log-entry:last-child { border-bottom: none; }
        .empty { color: #94a3b8; font-style: italic; }
        .preview-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        .preview-modal.active { display: flex; }
        .preview-content {
            background: #16213e;
            padding: 2rem;
            border-radius: 0.5rem;
            max-width: 90vw;
            max-height: 90vh;
            overflow: auto;
        }
        .preview-content img { max-width: 100%; max-height: 80vh; }
        .preview-content pre { max-width: 80vw; overflow: auto; }
        .close-btn {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: none;
            border: none;
            color: #eee;
            font-size: 2rem;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Node Inspector</h1>
    </div>
    <div class="container">
        <div class="selector">
            <input type="text" id="run-id" placeholder="Run ID">
            <input type="text" id="node-id" placeholder="Node ID">
            <button onclick="loadNode()">Inspect</button>
        </div>

        <div id="node-header" style="margin-bottom: 1rem; display: none;">
            <h2 id="node-title">Node: </h2>
            <span id="node-status" class="status">-</span>
        </div>

        <div class="panels">
            <div class="panel">
                <div class="panel-header">Inputs</div>
                <div class="panel-content">
                    <pre id="inputs-content" class="empty">No data</pre>
                </div>
            </div>
            <div class="panel">
                <div class="panel-header">Outputs</div>
                <div class="panel-content">
                    <pre id="outputs-content" class="empty">No data</pre>
                </div>
            </div>
            <div class="panel">
                <div class="panel-header">
                    <span>Artifacts</span>
                    <span id="artifact-count">0</span>
                </div>
                <div class="panel-content">
                    <ul id="artifacts-list" class="artifact-list">
                        <li class="empty">No artifacts</li>
                    </ul>
                </div>
            </div>
            <div class="panel">
                <div class="panel-header">
                    <span>Logs</span>
                    <span id="log-count">0</span>
                </div>
                <div class="panel-content">
                    <div id="logs-content" class="empty">No logs</div>
                </div>
            </div>
        </div>
    </div>

    <div id="preview-modal" class="preview-modal">
        <button class="close-btn" onclick="closePreview()">&times;</button>
        <div class="preview-content" id="preview-content"></div>
    </div>

    <script>
        async function loadNode() {
            const runId = document.getElementById('run-id').value;
            const nodeId = document.getElementById('node-id').value;

            if (!runId || !nodeId) {
                alert('Please enter Run ID and Node ID');
                return;
            }

            try {
                const res = await fetch(`/api/inspect/runs/${runId}/nodes/${nodeId}`);
                if (!res.ok) throw new Error('Node not found');
                const data = await res.json();
                renderNode(data);
            } catch (e) {
                alert('Failed to load node: ' + e.message);
            }
        }

        function renderNode(data) {
            document.getElementById('node-header').style.display = 'block';
            document.getElementById('node-title').textContent = `Node: ${data.node_id}`;
            document.getElementById('node-status').textContent = data.status;
            document.getElementById('node-status').className = `status ${data.status}`;

            // Inputs
            const inputs = document.getElementById('inputs-content');
            if (Object.keys(data.inputs).length > 0) {
                inputs.textContent = JSON.stringify(data.inputs, null, 2);
                inputs.className = '';
            } else {
                inputs.textContent = 'No inputs';
                inputs.className = 'empty';
            }

            // Outputs
            const outputs = document.getElementById('outputs-content');
            if (Object.keys(data.outputs).length > 0) {
                outputs.textContent = JSON.stringify(data.outputs, null, 2);
                outputs.className = '';
            } else {
                outputs.textContent = 'No outputs';
                outputs.className = 'empty';
            }

            // Artifacts
            document.getElementById('artifact-count').textContent = Object.keys(data.artifacts).length;
            const artifactsList = document.getElementById('artifacts-list');
            if (Object.keys(data.artifacts).length > 0) {
                artifactsList.innerHTML = Object.entries(data.artifacts).map(([name, hash]) => `
                    <li class="artifact-item">
                        <div>
                            <strong>${name}</strong><br>
                            <span class="artifact-hash">${hash}</span>
                        </div>
                        <div class="artifact-actions">
                            <button onclick="previewArtifact('${hash}')">Preview</button>
                            <button onclick="downloadArtifact('${hash}')">Download</button>
                        </div>
                    </li>
                `).join('');
            } else {
                artifactsList.innerHTML = '<li class="empty">No artifacts</li>';
            }

            // Logs
            document.getElementById('log-count').textContent = data.logs.length;
            const logsContent = document.getElementById('logs-content');
            if (data.logs.length > 0) {
                logsContent.innerHTML = data.logs.map(log =>
                    `<div class="log-entry">${log}</div>`
                ).join('');
            } else {
                logsContent.innerHTML = '<div class="empty">No logs</div>';
            }
        }

        async function previewArtifact(hash) {
            try {
                const res = await fetch(`/api/inspect/artifact/${hash}/preview`);
                const data = await res.json();

                const modal = document.getElementById('preview-modal');
                const content = document.getElementById('preview-content');

                if (data.content_type.startsWith('image/')) {
                    content.innerHTML = `<img src="/api/inspect/artifact/${hash}/download" alt="Preview">`;
                } else if (data.preview) {
                    content.innerHTML = `<pre>${escapeHtml(data.preview)}</pre>`;
                } else {
                    content.innerHTML = `<p>Cannot preview ${data.content_type}</p>`;
                }

                modal.classList.add('active');
            } catch (e) {
                alert('Failed to preview: ' + e.message);
            }
        }

        function downloadArtifact(hash) {
            window.open(`/api/inspect/artifact/${hash}/download`);
        }

        function closePreview() {
            document.getElementById('preview-modal').classList.remove('active');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Close modal on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closePreview();
        });
    </script>
</body>
</html>"""
