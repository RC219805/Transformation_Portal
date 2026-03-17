"""DAG editor API for interactive pipeline authoring.

This module provides FastAPI endpoints for creating, editing,
and saving pipeline definitions via a drag-and-drop UI.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import APIRouter, HTTPException, Body
    from fastapi.responses import JSONResponse, HTMLResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None


# Default pipelines directory
_pipelines_dir: Path = Path("pipelines")

# Safe filename pattern: alphanumeric, underscores, hyphens, dots (no path separators)
_SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _get_safe_pipeline_path(name: str) -> Path:
    """Construct a safe path for a pipeline JSON file.

    This prevents path traversal attacks by:
    1. Validating the name against a safe pattern
    2. Ensuring the resolved path stays within _pipelines_dir

    Args:
        name: Pipeline name (filename without extension)

    Returns:
        Safe resolved path to the pipeline JSON file

    Raises:
        HTTPException: If the name is invalid or would escape the directory
    """
    # Reject empty or whitespace-only names
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Invalid pipeline name: empty")

    # Reject names with path separators or traversal patterns
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid pipeline name: path traversal")

    # Validate against safe filename pattern
    if not _SAFE_NAME_PATTERN.match(name):
        raise HTTPException(status_code=400, detail="Invalid pipeline name: unsafe characters")

    # Construct path and resolve
    base_dir = _pipelines_dir.resolve()
    filepath = (base_dir / f"{name}.json").resolve()

    # Verify the path is within the base directory
    try:
        filepath.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline name: path traversal")

    return filepath


def set_pipelines_dir(path: Path) -> None:
    """Set the pipelines storage directory.

    Args:
        path: Directory path for pipeline JSON files
    """
    global _pipelines_dir
    _pipelines_dir = path
    _pipelines_dir.mkdir(parents=True, exist_ok=True)


class PipelineNode(BaseModel):
    """Node in a pipeline definition."""
    id: str
    type: str = "default"
    label: str
    position: dict[str, float]
    config: dict[str, Any] = {}


class PipelineEdge(BaseModel):
    """Edge connecting nodes in a pipeline."""
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None


class PipelineDefinition(BaseModel):
    """Complete pipeline definition."""
    name: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    metadata: dict[str, Any] = {}


def create_dag_editor_router() -> "APIRouter":
    """Create the DAG editor router.

    Returns:
        FastAPI APIRouter with pipeline editing endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for DAG editor")

    router = APIRouter(prefix="/api/editor", tags=["editor"])

    @router.get("/pipelines")
    async def list_pipelines():
        """List all saved pipelines."""
        _pipelines_dir.mkdir(parents=True, exist_ok=True)
        pipelines = []
        for p in _pipelines_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                pipelines.append({
                    "name": p.stem,
                    "filename": p.name,
                    "node_count": len(data.get("nodes", [])),
                    "edge_count": len(data.get("edges", [])),
                })
            except Exception as exc:
                logger.warning("Failed to read pipeline %s: %s", p, exc)
        return JSONResponse({"pipelines": pipelines})

    @router.get("/pipelines/{name}")
    async def get_pipeline(name: str):
        """Get a specific pipeline definition."""
        _pipelines_dir.mkdir(parents=True, exist_ok=True)
        filepath = _get_safe_pipeline_path(name)
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"Pipeline not found: {name}")

        try:
            data = json.loads(filepath.read_text())
            return JSONResponse(data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/pipelines/{name}")
    async def save_pipeline(name: str, payload: dict = Body(...)):
        """Save a pipeline definition."""
        _pipelines_dir.mkdir(parents=True, exist_ok=True)
        filepath = _get_safe_pipeline_path(name)

        # Add metadata
        payload["name"] = name

        try:
            filepath.write_text(json.dumps(payload, indent=2))
            logger.info("Saved pipeline: %s", name)
            return JSONResponse({"status": "ok", "name": name})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.delete("/pipelines/{name}")
    async def delete_pipeline(name: str):
        """Delete a pipeline definition."""
        filepath = _get_safe_pipeline_path(name)
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"Pipeline not found: {name}")

        try:
            filepath.unlink()
            logger.info("Deleted pipeline: %s", name)
            return JSONResponse({"status": "ok"})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/node-types")
    async def get_node_types():
        """Get available node types for the editor."""
        return JSONResponse({
            "node_types": [
                {
                    "type": "ingest",
                    "label": "Ingest",
                    "category": "input",
                    "inputs": [],
                    "outputs": ["image", "metadata"],
                },
                {
                    "type": "segment",
                    "label": "Segmentation (SAM2)",
                    "category": "processing",
                    "inputs": ["image"],
                    "outputs": ["masks", "scores"],
                },
                {
                    "type": "depth",
                    "label": "Depth Estimation",
                    "category": "processing",
                    "inputs": ["image"],
                    "outputs": ["depth_map"],
                },
                {
                    "type": "materials",
                    "label": "Material Reconstruction",
                    "category": "processing",
                    "inputs": ["image", "masks"],
                    "outputs": ["albedo", "normal", "roughness"],
                },
                {
                    "type": "quality",
                    "label": "Quality Assessment (LLaVA)",
                    "category": "evaluation",
                    "inputs": ["image"],
                    "outputs": ["score", "issues"],
                },
                {
                    "type": "export",
                    "label": "Export",
                    "category": "output",
                    "inputs": ["image", "metadata"],
                    "outputs": [],
                },
            ]
        })

    @router.get("/", response_class=HTMLResponse)
    async def editor_ui():
        """Serve the DAG editor UI."""
        return get_dag_editor_html()

    return router


def get_dag_editor_html() -> str:
    """Get the DAG editor frontend HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Editor</title>
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
        .toolbar {
            display: flex;
            gap: 0.5rem;
        }
        .toolbar button, .toolbar select {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .toolbar button:hover { background: #e94560; }
        .toolbar select { min-width: 150px; }
        .main {
            display: flex;
            height: calc(100vh - 50px);
        }
        .sidebar {
            width: 220px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            padding: 1rem;
            overflow-y: auto;
        }
        .sidebar h3 {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }
        .node-type {
            background: #1a1a2e;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 0.25rem;
            cursor: grab;
            border-left: 3px solid #e94560;
            font-size: 0.8rem;
        }
        .node-type:hover { background: #0f3460; }
        .node-type.input { border-left-color: #00d25b; }
        .node-type.processing { border-left-color: #0d6efd; }
        .node-type.evaluation { border-left-color: #ffc107; }
        .node-type.output { border-left-color: #e94560; }
        .canvas {
            flex: 1;
            position: relative;
            overflow: hidden;
        }
        #canvas-svg {
            width: 100%;
            height: 100%;
            background:
                linear-gradient(rgba(15, 52, 96, 0.3) 1px, transparent 1px),
                linear-gradient(90deg, rgba(15, 52, 96, 0.3) 1px, transparent 1px);
            background-size: 20px 20px;
        }
        .node {
            position: absolute;
            background: #16213e;
            border: 2px solid #0f3460;
            border-radius: 0.5rem;
            min-width: 150px;
            cursor: move;
            user-select: none;
        }
        .node.selected { border-color: #e94560; }
        .node-header {
            background: #0f3460;
            padding: 0.5rem 0.75rem;
            border-radius: 0.35rem 0.35rem 0 0;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .node-body { padding: 0.5rem 0.75rem; font-size: 0.75rem; color: #94a3b8; }
        .node-ports { display: flex; justify-content: space-between; padding: 0.25rem 0.5rem; }
        .port {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #0f3460;
            border: 2px solid #94a3b8;
            cursor: crosshair;
        }
        .port.input { background: #00d25b; }
        .port.output { background: #e94560; }
        .edge {
            stroke: #0f3460;
            stroke-width: 2;
            fill: none;
        }
        .edge.selected { stroke: #e94560; }
        .status-bar {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: #16213e;
            padding: 0.5rem 1rem;
            font-size: 0.75rem;
            color: #94a3b8;
            border-top: 1px solid #0f3460;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Pipeline Editor</h1>
        <div class="toolbar">
            <select id="pipeline-select">
                <option value="">New Pipeline</option>
            </select>
            <button onclick="savePipeline()">Save</button>
            <button onclick="loadPipeline()">Load</button>
            <button onclick="runPipeline()">Run</button>
            <button onclick="clearCanvas()">Clear</button>
        </div>
    </div>
    <div class="main">
        <div class="sidebar">
            <h3>Node Types</h3>
            <div id="node-types"></div>
        </div>
        <div class="canvas">
            <svg id="canvas-svg">
                <defs>
                    <marker id="arrowhead" viewBox="0 -5 10 10" refX="8" refY="0"
                            markerWidth="6" markerHeight="6" orient="auto">
                        <path d="M0,-5L10,0L0,5" fill="#0f3460"/>
                    </marker>
                </defs>
                <g id="edges"></g>
            </svg>
            <div id="nodes"></div>
            <div class="status-bar">
                <span id="status">Ready | Drag nodes from sidebar to canvas</span>
            </div>
        </div>
    </div>

    <script>
        let nodes = [];
        let edges = [];
        let selectedNode = null;
        let nodeIdCounter = 0;
        let dragNode = null;
        let dragOffset = { x: 0, y: 0 };
        let connecting = null;

        // Load node types
        async function loadNodeTypes() {
            const res = await fetch('/api/editor/node-types');
            const data = await res.json();
            const container = document.getElementById('node-types');

            data.node_types.forEach(nt => {
                const div = document.createElement('div');
                div.className = `node-type ${nt.category}`;
                div.textContent = nt.label;
                div.draggable = true;
                div.dataset.type = nt.type;
                div.dataset.label = nt.label;
                div.dataset.inputs = JSON.stringify(nt.inputs);
                div.dataset.outputs = JSON.stringify(nt.outputs);

                div.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('node-type', JSON.stringify(nt));
                });

                container.appendChild(div);
            });
        }

        // Load pipelines list
        async function loadPipelinesList() {
            const res = await fetch('/api/editor/pipelines');
            const data = await res.json();
            const select = document.getElementById('pipeline-select');

            // Clear existing options except first
            while (select.options.length > 1) select.remove(1);

            data.pipelines.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.name;
                opt.textContent = `${p.name} (${p.node_count} nodes)`;
                select.appendChild(opt);
            });
        }

        // Canvas drop handler
        document.querySelector('.canvas').addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        document.querySelector('.canvas').addEventListener('drop', (e) => {
            e.preventDefault();
            const ntData = e.dataTransfer.getData('node-type');
            if (!ntData) return;

            const nt = JSON.parse(ntData);
            const rect = e.target.getBoundingClientRect();
            const x = e.clientX - rect.left - 75;
            const y = e.clientY - rect.top - 30;

            addNode(nt, x, y);
        });

        function addNode(nodeType, x, y) {
            const id = `node_${++nodeIdCounter}`;
            const node = {
                id,
                type: nodeType.type,
                label: nodeType.label,
                position: { x, y },
                inputs: nodeType.inputs,
                outputs: nodeType.outputs,
            };
            nodes.push(node);
            renderNode(node);
            updateStatus(`Added node: ${node.label}`);
        }

        function renderNode(node) {
            const container = document.getElementById('nodes');
            const div = document.createElement('div');
            div.className = 'node';
            div.id = node.id;
            div.style.left = node.position.x + 'px';
            div.style.top = node.position.y + 'px';

            div.innerHTML = `
                <div class="node-header">${node.label}</div>
                <div class="node-body">ID: ${node.id}</div>
                <div class="node-ports">
                    <div class="ports-in">${node.inputs.map(i =>
                        `<div class="port input" title="${i}" data-port="${i}" data-dir="in"></div>`
                    ).join('')}</div>
                    <div class="ports-out">${node.outputs.map(o =>
                        `<div class="port output" title="${o}" data-port="${o}" data-dir="out"></div>`
                    ).join('')}</div>
                </div>
            `;

            // Drag handling
            div.addEventListener('mousedown', (e) => {
                if (e.target.classList.contains('port')) return;
                dragNode = node;
                dragOffset = {
                    x: e.clientX - node.position.x,
                    y: e.clientY - node.position.y
                };
                div.classList.add('selected');
            });

            // Port connection
            div.querySelectorAll('.port').forEach(port => {
                port.addEventListener('mousedown', (e) => {
                    e.stopPropagation();
                    if (port.dataset.dir === 'out') {
                        connecting = { nodeId: node.id, port: port.dataset.port };
                    }
                });

                port.addEventListener('mouseup', (e) => {
                    if (connecting && port.dataset.dir === 'in') {
                        addEdge(connecting.nodeId, node.id);
                        connecting = null;
                    }
                });
            });

            container.appendChild(div);
        }

        document.addEventListener('mousemove', (e) => {
            if (dragNode) {
                dragNode.position.x = e.clientX - dragOffset.x;
                dragNode.position.y = e.clientY - dragOffset.y;
                const el = document.getElementById(dragNode.id);
                el.style.left = dragNode.position.x + 'px';
                el.style.top = dragNode.position.y + 'px';
                renderEdges();
            }
        });

        document.addEventListener('mouseup', () => {
            if (dragNode) {
                document.getElementById(dragNode.id).classList.remove('selected');
                dragNode = null;
            }
            connecting = null;
        });

        function addEdge(sourceId, targetId) {
            if (sourceId === targetId) return;
            if (edges.some(e => e.source === sourceId && e.target === targetId)) return;

            edges.push({ id: `edge_${edges.length}`, source: sourceId, target: targetId });
            renderEdges();
            updateStatus(`Connected: ${sourceId} → ${targetId}`);
        }

        function renderEdges() {
            const g = document.getElementById('edges');
            g.innerHTML = '';

            edges.forEach(edge => {
                const sourceNode = nodes.find(n => n.id === edge.source);
                const targetNode = nodes.find(n => n.id === edge.target);
                if (!sourceNode || !targetNode) return;

                const x1 = sourceNode.position.x + 150;
                const y1 = sourceNode.position.y + 40;
                const x2 = targetNode.position.x;
                const y2 = targetNode.position.y + 40;

                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const midX = (x1 + x2) / 2;
                path.setAttribute('d', `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`);
                path.setAttribute('class', 'edge');
                path.setAttribute('marker-end', 'url(#arrowhead)');
                g.appendChild(path);
            });
        }

        async function savePipeline() {
            const name = prompt('Pipeline name:', 'my_pipeline');
            if (!name) return;

            await fetch(`/api/editor/pipelines/${name}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ nodes, edges })
            });

            updateStatus(`Saved pipeline: ${name}`);
            loadPipelinesList();
        }

        async function loadPipeline() {
            const name = document.getElementById('pipeline-select').value;
            if (!name) return;

            const res = await fetch(`/api/editor/pipelines/${name}`);
            const data = await res.json();

            clearCanvas();
            nodes = data.nodes || [];
            edges = data.edges || [];
            nodeIdCounter = nodes.length;

            nodes.forEach(n => renderNode(n));
            renderEdges();
            updateStatus(`Loaded pipeline: ${name}`);
        }

        function runPipeline() {
            updateStatus('Running pipeline... (not implemented)');
            // TODO: Submit to scheduler
        }

        function clearCanvas() {
            nodes = [];
            edges = [];
            document.getElementById('nodes').innerHTML = '';
            document.getElementById('edges').innerHTML = '';
            updateStatus('Canvas cleared');
        }

        function updateStatus(msg) {
            document.getElementById('status').textContent = msg;
        }

        // Initialize
        loadNodeTypes();
        loadPipelinesList();
    </script>
</body>
</html>"""
