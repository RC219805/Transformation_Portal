"""DAG visualization API endpoints.

This module provides FastAPI endpoints for serving DAG structure
and Merkle lineage data for visualization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformation_portal.execution_graph.scheduler import PriorityDAGScheduler
    from transformation_portal.storage.merkle_dag import MerkleDAG

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None


# Global references (injected at runtime)
_global_merkle_dag: Optional["MerkleDAG"] = None  # type: ignore
_global_scheduler: Optional["PriorityDAGScheduler"] = None  # type: ignore


def set_merkle_dag(dag: "MerkleDAG") -> None:  # type: ignore
    """Set the global Merkle DAG for visualization.

    Args:
        dag: MerkleDAG instance to visualize
    """
    global _global_merkle_dag
    _global_merkle_dag = dag


def set_scheduler(scheduler: "PriorityDAGScheduler") -> None:  # type: ignore
    """Set the global scheduler for visualization.

    Args:
        scheduler: PriorityDAGScheduler instance
    """
    global _global_scheduler
    _global_scheduler = scheduler


def create_dag_router() -> "APIRouter":
    """Create the DAG visualization router.

    Returns:
        FastAPI APIRouter with DAG endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for DAG visualization")

    router = APIRouter(prefix="/api/dag", tags=["dag"])

    @router.get("/graph")
    async def get_dag_graph():
        """Get the execution DAG as nodes and edges for visualization."""
        if _global_scheduler is None:
            return JSONResponse({"nodes": [], "edges": [], "error": "No scheduler configured"})

        nodes = []
        edges = []

        for node_id, scheduled_node in _global_scheduler.nodes.items():
            # Get execution status if available
            status = "pending"
            outputs = {}
            if node_id in _global_scheduler.results:
                status = "completed"
                outputs = _global_scheduler.results.get(node_id, {})

            nodes.append({
                "id": node_id,
                "label": node_id,
                "priority": -scheduled_node.priority,  # Restore original priority
                "status": status,
                "resources": {
                    "gpu": scheduled_node.resources.gpu,
                    "gpu_memory_mb": scheduled_node.resources.gpu_memory_mb,
                },
                "outputs": outputs,
                "score": outputs.get("score") if isinstance(outputs, dict) else None,
            })

            for dep in scheduled_node.deps:
                edges.append({
                    "source": dep,
                    "target": node_id,
                })

        return JSONResponse({
            "nodes": nodes,
            "edges": edges,
            "execution_order": _global_scheduler.get_execution_order(),
        })

    @router.get("/merkle")
    async def get_merkle_graph():
        """Get the Merkle DAG as nodes and edges."""
        if _global_merkle_dag is None:
            return JSONResponse({"nodes": [], "edges": [], "error": "No Merkle DAG configured"})

        nodes = []
        edges = []

        for node_hash, node in _global_merkle_dag.nodes.items():
            nodes.append({
                "id": node_hash,
                "hash_short": node_hash[:8],
                "type": node.node_type,
                "metadata": node.metadata,
                "outputs": node.outputs,
                "timestamp": node.timestamp,
            })

            for input_hash in node.inputs:
                edges.append({
                    "source": input_hash,
                    "target": node_hash,
                })

        return JSONResponse({
            "nodes": nodes,
            "edges": edges,
            "summary": _global_merkle_dag.summary(),
        })

    @router.get("/merkle/{node_hash}")
    async def get_merkle_node(node_hash: str):
        """Get details for a specific Merkle node."""
        if _global_merkle_dag is None:
            raise HTTPException(status_code=404, detail="No Merkle DAG configured")

        node = _global_merkle_dag.get_node(node_hash)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_hash}")

        return JSONResponse({
            "hash": node.hash,
            "type": node.node_type,
            "inputs": list(node.inputs),
            "outputs": node.outputs,
            "metadata": node.metadata,
            "timestamp": node.timestamp,
        })

    @router.get("/merkle/{node_hash}/lineage")
    async def get_merkle_lineage(node_hash: str, max_depth: int = 10):
        """Get the lineage (ancestry) of a Merkle node."""
        if _global_merkle_dag is None:
            raise HTTPException(status_code=404, detail="No Merkle DAG configured")

        lineage = _global_merkle_dag.get_lineage(node_hash, max_depth=max_depth)
        if not lineage:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_hash}")

        return JSONResponse({
            "target": node_hash,
            "depth": len(lineage),
            "lineage": [
                {
                    "hash": n.hash,
                    "type": n.node_type,
                    "metadata": n.metadata,
                }
                for n in lineage
            ],
        })

    return router


def get_dag_visualization_html() -> str:
    """Get the DAG visualization frontend HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline DAG Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
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
        .controls button {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 0.5rem 1rem;
            margin-left: 0.5rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .controls button:hover { background: #e94560; }
        .main { display: flex; height: calc(100vh - 60px); }
        .sidebar {
            width: 300px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            padding: 1rem;
            overflow-y: auto;
        }
        .sidebar h3 {
            font-size: 0.875rem;
            color: #94a3b8;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }
        .node-info {
            background: #1a1a2e;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin-bottom: 0.5rem;
            font-size: 0.8rem;
        }
        .node-info .label { color: #94a3b8; }
        .node-info .value { color: #eee; word-break: break-all; }
        .graph-container { flex: 1; position: relative; }
        svg { width: 100%; height: 100%; }
        .node circle {
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
            transition: r 0.2s;
        }
        .node circle:hover { r: 15; }
        .node.completed circle { fill: #00d25b; }
        .node.pending circle { fill: #94a3b8; }
        .node.running circle { fill: #0d6efd; }
        .node.failed circle { fill: #ff5252; }
        .node.high-score circle { fill: #00d25b; }
        .node.low-score circle { fill: #ff5252; }
        .node text {
            font-size: 11px;
            fill: #eee;
            pointer-events: none;
        }
        .link {
            stroke: #0f3460;
            stroke-width: 2px;
            fill: none;
            marker-end: url(#arrowhead);
        }
        .link.lineage { stroke: #e94560; stroke-dasharray: 5,5; }
        .legend {
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            background: rgba(22, 33, 62, 0.9);
            padding: 0.75rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 0.25rem;
        }
        .legend-item span {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        .legend-item .completed { background: #00d25b; }
        .legend-item .pending { background: #94a3b8; }
        .legend-item .running { background: #0d6efd; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Pipeline DAG Visualization</h1>
        <div class="controls">
            <button onclick="loadDAG()">Refresh DAG</button>
            <button onclick="loadMerkle()">Show Lineage</button>
            <button onclick="resetView()">Reset View</button>
        </div>
    </div>
    <div class="main">
        <div class="sidebar">
            <h3>Selected Node</h3>
            <div id="node-details" class="node-info">
                <div class="label">Click a node to view details</div>
            </div>
            <h3>Execution Order</h3>
            <div id="exec-order" class="node-info">
                <div class="value">Loading...</div>
            </div>
            <h3>Statistics</h3>
            <div id="stats" class="node-info">
                <div class="value">Loading...</div>
            </div>
        </div>
        <div class="graph-container">
            <svg>
                <defs>
                    <marker id="arrowhead" viewBox="0 -5 10 10" refX="20" refY="0"
                            markerWidth="6" markerHeight="6" orient="auto">
                        <path d="M0,-5L10,0L0,5" fill="#0f3460"/>
                    </marker>
                </defs>
            </svg>
            <div class="legend">
                <div class="legend-item"><span class="completed"></span> Completed</div>
                <div class="legend-item"><span class="pending"></span> Pending</div>
                <div class="legend-item"><span class="running"></span> Running</div>
            </div>
        </div>
    </div>
    <script>
        let currentData = null;
        let simulation = null;

        async function loadDAG() {
            const res = await fetch('/api/dag/graph');
            const data = await res.json();
            currentData = data;
            renderGraph(data);
            updateSidebar(data);
        }

        async function loadMerkle() {
            const res = await fetch('/api/dag/merkle');
            const data = await res.json();
            renderMerkleOverlay(data);
        }

        function renderGraph(data) {
            const svg = d3.select('svg');
            svg.selectAll('*:not(defs)').remove();

            const width = svg.node().clientWidth;
            const height = svg.node().clientHeight;

            const g = svg.append('g');

            // Zoom
            svg.call(d3.zoom().on('zoom', (e) => g.attr('transform', e.transform)));

            simulation = d3.forceSimulation(data.nodes)
                .force('link', d3.forceLink(data.edges).id(d => d.id).distance(120))
                .force('charge', d3.forceManyBody().strength(-400))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('y', d3.forceY(height / 2).strength(0.1));

            const link = g.selectAll('.link')
                .data(data.edges)
                .enter().append('path')
                .attr('class', 'link');

            const node = g.selectAll('.node')
                .data(data.nodes)
                .enter().append('g')
                .attr('class', d => {
                    const scoreClass = d.score > 0.7 ? 'high-score' : d.score !== null ? 'low-score' : '';
                    return `node ${d.status} ${scoreClass}`;
                })
                .call(d3.drag()
                    .on('start', dragstart)
                    .on('drag', drag)
                    .on('end', dragend))
                .on('click', (e, d) => showNodeDetails(d));

            node.append('circle').attr('r', 12);
            node.append('text')
                .attr('dy', -18)
                .attr('text-anchor', 'middle')
                .text(d => d.label);

            simulation.on('tick', () => {
                link.attr('d', d => {
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    return `M${d.source.x},${d.source.y}L${d.target.x},${d.target.y}`;
                });
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });
        }

        function renderMerkleOverlay(data) {
            // Add Merkle lineage edges as dashed lines
            const g = d3.select('svg g');

            g.selectAll('.link.lineage').remove();

            g.selectAll('.link.lineage')
                .data(data.edges)
                .enter().append('line')
                .attr('class', 'link lineage')
                .attr('x1', d => findNodePos(d.source)?.x || 0)
                .attr('y1', d => findNodePos(d.source)?.y || 0)
                .attr('x2', d => findNodePos(d.target)?.x || 0)
                .attr('y2', d => findNodePos(d.target)?.y || 0);
        }

        function findNodePos(hash) {
            if (!currentData) return null;
            return currentData.nodes.find(n => n.id.startsWith(hash) || hash.startsWith(n.id));
        }

        function showNodeDetails(node) {
            const details = document.getElementById('node-details');
            let html = `
                <div><span class="label">ID:</span> <span class="value">${node.id}</span></div>
                <div><span class="label">Status:</span> <span class="value">${node.status}</span></div>
                <div><span class="label">Priority:</span> <span class="value">${node.priority}</span></div>
            `;
            if (node.score !== null) {
                const scoreVal = node.score.toFixed(3);
                html += `<div><span class="label">Score:</span> <span class="value">${scoreVal}</span></div>`;
            }
            if (node.resources?.gpu) {
                const memVal = node.resources.gpu_memory_mb;
                html += `<div><span class="label">GPU:</span> <span class="value">${memVal}MB</span></div>`;
            }
            details.innerHTML = html;
        }

        function updateSidebar(data) {
            document.getElementById('exec-order').innerHTML =
                `<div class="value">${data.execution_order?.join(' → ') || 'N/A'}</div>`;

            const completed = data.nodes.filter(n => n.status === 'completed').length;
            document.getElementById('stats').innerHTML = `
                <div><span class="label">Total:</span> <span class="value">${data.nodes.length}</span></div>
                <div><span class="label">Completed:</span> <span class="value">${completed}</span></div>
                <div><span class="label">Edges:</span> <span class="value">${data.edges.length}</span></div>
            `;
        }

        function resetView() {
            d3.select('svg g').attr('transform', null);
            if (simulation) simulation.alpha(1).restart();
        }

        function dragstart(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function drag(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragend(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        // Initial load
        loadDAG();
    </script>
</body>
</html>"""
