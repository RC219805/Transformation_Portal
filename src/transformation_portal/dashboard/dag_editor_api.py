"""DAG editor API for interactive pipeline authoring.

This module provides FastAPI endpoints for creating, editing,
and saving pipeline definitions via a drag-and-drop UI.

Security: All filesystem operations go through FSGuard for
zero-trust file access with audit logging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from transformation_portal.core.security.fs_guard import (
    FSContext,
    FSPolicyError,
    get_fs_guard,
)
from transformation_portal.core.security.path_safety import (
    PathSafetyError,
    validate_safe_name,
)

logger = logging.getLogger(__name__)

# Default pipelines directory and FSGuard context
_pipelines_dir: Path = Path("pipelines")
_fs_context: Optional[FSContext] = None


def _get_fs_context() -> FSContext:
    """Get or create the FSContext for pipeline operations."""
    global _fs_context
    if _fs_context is None or _fs_context.base_dir != _pipelines_dir:
        _fs_context = FSContext(mode="user", base_dir=_pipelines_dir)
    return _fs_context


def _validate_pipeline_name(name: str) -> str:
    """Strict whitelist validation for pipeline names.

    Uses the centralized path_safety module for validation.
    Converts PathSafetyError to HTTPException for API responses.

    Args:
        name: Pipeline name to validate

    Returns:
        The validated name (unchanged if valid)

    Raises:
        HTTPException: If name fails validation
    """
    try:
        return validate_safe_name(name)
    except PathSafetyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pipeline name: {e}",
        )


def _get_safe_pipeline_path(name: str) -> Path:
    """Construct a safe pipeline path using FSGuard.

    Uses zero-trust FSGuard for CodeQL-compliant path construction.
    Validation happens BEFORE path construction.

    Args:
        name: Pipeline name (will be validated first)

    Returns:
        Path to the pipeline JSON file

    Raises:
        HTTPException: If name fails validation
    """
    fs = get_fs_guard()
    ctx = _get_fs_context()

    try:
        return fs.user_file(ctx, name, suffix=".json")
    except (PathSafetyError, FSPolicyError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pipeline name: {e}",
        )


def set_pipelines_dir(path: Path) -> None:
    """Set the pipelines storage directory.

    Args:
        path: Directory path for pipeline JSON files
    """
    global _pipelines_dir, _fs_context
    _pipelines_dir = path
    # Reset context so _get_fs_context() creates fresh one with new base_dir
    _fs_context = None
    # Use FSGuard for directory creation
    fs = get_fs_guard()
    fs.mkdir(_pipelines_dir)
    # Pre-initialize the context after mkdir to avoid repeated resets
    _fs_context = _get_fs_context()


class PipelineNode(BaseModel):
    """Node in a pipeline definition."""

    id: str
    type: str = "default"
    label: str
    position: dict[str, float]
    config: dict[str, Any] = Field(default_factory=dict)


class PipelineEdge(BaseModel):
    """Edge connecting nodes in a pipeline."""

    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None


class PipelineDefinition(BaseModel):
    """Complete pipeline definition.

    Note: The name field defaults to empty because save_pipeline() always
    overrides it with the name from the URL path for consistency.

    Backward Compatibility: The nodes and edges fields accept both typed
    objects (PipelineNode/PipelineEdge) and raw dicts for compatibility
    with existing saved pipelines.
    """

    name: str = ""
    nodes: list[PipelineNode] = Field(default_factory=list)
    edges: list[PipelineEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("nodes", mode="before")
    @classmethod
    def coerce_nodes(cls, v: list[Union[PipelineNode, dict[str, Any]]]) -> list[PipelineNode]:
        """Coerce raw dicts to PipelineNode for backward compatibility."""
        if not v:
            return []
        return [item if isinstance(item, PipelineNode) else PipelineNode(**item) for item in v]

    @field_validator("edges", mode="before")
    @classmethod
    def coerce_edges(cls, v: list[Union[PipelineEdge, dict[str, Any]]]) -> list[PipelineEdge]:
        """Coerce raw dicts to PipelineEdge for backward compatibility."""
        if not v:
            return []
        return [item if isinstance(item, PipelineEdge) else PipelineEdge(**item) for item in v]


def create_dag_editor_router() -> APIRouter:
    """Create the DAG editor router.

    Returns:
        FastAPI APIRouter with pipeline editing endpoints
    """
    router = APIRouter(prefix="/api/editor", tags=["editor"])
    fs = get_fs_guard()

    @router.get("/pipelines")
    async def list_pipelines() -> JSONResponse:
        """List all saved pipelines.

        Note: Uses FSGuard.list_dir for audit logging consistency.
        Pipeline names are derived from validated filenames only.
        """
        fs.mkdir(_pipelines_dir)
        pipelines = []
        # Use FSGuard for directory listing to maintain audit trail
        for p in fs.list_dir(_pipelines_dir):
            # Skip directories and non-JSON files
            if not p.is_file() or p.suffix != ".json":
                continue
            try:
                data = json.loads(fs.read_text(p))
                pipelines.append(
                    {
                        "name": p.stem,
                        "filename": p.name,
                        "node_count": len(data.get("nodes", [])),
                        "edge_count": len(data.get("edges", [])),
                    }
                )
            except Exception as exc:
                logger.warning("Failed to read pipeline %s: %s", p, exc)
        return JSONResponse({"pipelines": pipelines})

    @router.get("/pipelines/{name}")
    async def get_pipeline(name: str) -> JSONResponse:
        """Get a specific pipeline definition."""
        fs.mkdir(_pipelines_dir)
        filepath = _get_safe_pipeline_path(name)

        if not fs.exists(filepath):
            raise HTTPException(status_code=404, detail=f"Pipeline not found: {name}")

        try:
            data = json.loads(fs.read_text(filepath))
            return JSONResponse(data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/pipelines/{name}")
    async def save_pipeline(name: str, payload: PipelineDefinition) -> JSONResponse:
        """Save a pipeline definition.

        Args:
            name: Pipeline name (from URL path)
            payload: Pipeline definition (validated by Pydantic)
        """
        fs.mkdir(_pipelines_dir)
        filepath = _get_safe_pipeline_path(name)

        # Build save data from validated model, overriding name from URL
        save_data = payload.model_dump()
        save_data["name"] = name

        try:
            fs.write_text(filepath, json.dumps(save_data, indent=2))
            logger.info("Saved pipeline: %s", name)
            return JSONResponse({"status": "ok", "name": name})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.delete("/pipelines/{name}")
    async def delete_pipeline(name: str) -> JSONResponse:
        """Delete a pipeline definition."""
        filepath = _get_safe_pipeline_path(name)

        if not fs.exists(filepath):
            raise HTTPException(status_code=404, detail=f"Pipeline not found: {name}")

        try:
            fs.delete(filepath)
            logger.info("Deleted pipeline: %s", name)
            return JSONResponse({"status": "ok"})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/node-types")
    async def get_node_types() -> JSONResponse:
        """Get available node types for the editor."""
        return JSONResponse(
            {
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
            }
        )

    @router.get("/", response_class=HTMLResponse)
    async def editor_ui() -> str:
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

            try {
                const response = await fetch(`/api/editor/pipelines/${name}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ nodes, edges })
                });

                if (!response.ok) {
                    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                    const contentType = response.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        try {
                            const errorData = await response.json();
                            errorMessage = errorData.detail || errorMessage;
                        } catch (parseError) {
                            console.warn('Failed to parse JSON error:', parseError);
                        }
                    }
                    throw new Error(errorMessage);
                }

                updateStatus(`Saved pipeline: ${name}`);
                loadPipelinesList();
            } catch (error) {
                updateStatus(`Error saving pipeline: ${error.message}`);
                console.error('Pipeline save failed:', error);
            }
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

        async function runPipeline() {
            if (nodes.length === 0) {
                updateStatus('Error: No nodes in pipeline');
                return;
            }

            updateStatus('Submitting pipeline to scheduler...');

            try {
                // Generate a descriptive default name with timestamp
                const selectedName = document.getElementById('pipeline-select').value;
                const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
                const pipelineName = selectedName || `untitled-pipeline-${timestamp}`;

                const response = await fetch('/api/exec/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: pipelineName,
                        nodes: nodes,
                        edges: edges
                    })
                });

                if (!response.ok) {
                    // Handle both JSON and non-JSON error responses
                    let errorMessage;
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.detail || `HTTP ${response.status}`;
                    } catch (parseError) {
                        // Non-JSON response (e.g., 502 Bad Gateway HTML)
                        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                        console.warn('Non-JSON error response:', parseError);
                    }
                    throw new Error(errorMessage);
                }

                const data = await response.json();
                // Status message with run ID (monitor opens automatically)
                updateStatus(`Pipeline started: Run ID ${data.run_id}`);

                // Automatically open execution monitor with run_id for auto-selection
                window.open(`/api/exec/?run_id=${data.run_id}`, '_blank');
            } catch (error) {
                updateStatus(`Error: ${error.message}`);
                console.error('Pipeline execution failed:', error);
            }
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
