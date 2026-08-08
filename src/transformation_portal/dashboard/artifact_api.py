"""Artifact browser API endpoints.

This module provides FastAPI endpoints for browsing CAS artifacts
and exploring Merkle DAG lineage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformation_portal.storage.cas_store import ArtifactStore
    from transformation_portal.storage.merkle_dag import MerkleDAG

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import APIRouter, HTTPException, Query
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None


# Global references (injected at runtime)
_global_cas: Optional["ArtifactStore"] = None  # type: ignore
_global_merkle_dag: Optional["MerkleDAG"] = None  # type: ignore


def set_artifact_store(cas: "ArtifactStore") -> None:  # type: ignore
    """Set the global CAS for artifact browsing.

    Args:
        cas: ArtifactStore instance
    """
    global _global_cas
    _global_cas = cas


def set_merkle_dag(dag: "MerkleDAG") -> None:  # type: ignore
    """Set the global Merkle DAG for lineage browsing.

    Args:
        dag: MerkleDAG instance
    """
    global _global_merkle_dag
    _global_merkle_dag = dag


def create_artifact_router() -> "APIRouter":
    """Create the artifact browser router.

    Returns:
        FastAPI APIRouter with artifact endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for artifact browser")

    router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])

    @router.get("/")
    async def list_artifacts(
        prefix: Optional[str] = None,
        limit: int = Query(default=100, le=1000),
        offset: int = 0,
    ):
        """List artifacts in the CAS.

        Args:
            prefix: Optional hash prefix filter
            limit: Maximum number of results
            offset: Pagination offset
        """
        if _global_cas is None:
            return JSONResponse(
                {
                    "artifacts": [],
                    "total": 0,
                    "error": "No CAS configured",
                }
            )

        artifacts = []
        total = 0

        for prefix_dir in _global_cas.objects_dir.iterdir():
            if not prefix_dir.is_dir():
                continue

            for obj_path in prefix_dir.iterdir():
                obj_hash = obj_path.name

                # Apply prefix filter
                if prefix and not obj_hash.startswith(prefix):
                    continue

                total += 1

                # Apply pagination
                if total <= offset:
                    continue
                if len(artifacts) >= limit:
                    continue

                try:
                    stat = obj_path.stat()
                    artifacts.append(
                        {
                            "hash": obj_hash,
                            "hash_short": obj_hash[:8],
                            "size_bytes": stat.st_size,
                            "size_human": _human_size(stat.st_size),
                            "path": str(obj_path),
                        }
                    )
                except Exception as exc:
                    logger.warning("Failed to stat %s: %s", obj_path, exc)

        return JSONResponse(
            {
                "artifacts": artifacts,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @router.get("/{hash}")
    async def get_artifact(hash: str):
        """Get metadata for a specific artifact.

        Args:
            hash: SHA-256 hash of the artifact
        """
        if _global_cas is None:
            raise HTTPException(status_code=404, detail="No CAS configured")

        try:
            obj = _global_cas.get_object(hash)
        except ValueError:
            raise HTTPException(status_code=404, detail="Artifact not found") from None
        if obj is None:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {hash}")

        # Try to find related Merkle nodes
        related_nodes = []
        if _global_merkle_dag:
            for node_hash, node in _global_merkle_dag.nodes.items():
                if node.outputs.get("content_hash") == hash:
                    related_nodes.append(
                        {
                            "node_hash": node_hash,
                            "type": node.node_type,
                        }
                    )

        return JSONResponse(
            {
                "hash": obj.sha256,
                "size_bytes": obj.size_bytes,
                "size_human": _human_size(obj.size_bytes),
                "path": str(obj.path),
                "exists": obj.path.exists(),
                "related_merkle_nodes": related_nodes,
            }
        )

    @router.get("/{hash}/preview")
    async def preview_artifact(hash: str, max_bytes: int = 1024):
        """Get a preview of artifact contents.

        Args:
            hash: SHA-256 hash
            max_bytes: Maximum bytes to return
        """
        if _global_cas is None:
            raise HTTPException(status_code=404, detail="No CAS configured")

        try:
            obj = _global_cas.get_object(hash)
        except ValueError:
            raise HTTPException(status_code=404, detail="Artifact not found") from None
        if obj is None:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {hash}")

        # Try to read and decode
        try:
            with obj.path.open("rb") as f:
                data = f.read(max_bytes)

            # Try UTF-8 decode
            try:
                text = data.decode("utf-8")
                return JSONResponse(
                    {
                        "hash": hash,
                        "type": "text",
                        "preview": text,
                        "truncated": obj.size_bytes > max_bytes,
                    }
                )
            except UnicodeDecodeError:
                # Return hex preview for binary
                return JSONResponse(
                    {
                        "hash": hash,
                        "type": "binary",
                        "preview": data.hex()[:200],
                        "truncated": True,
                    }
                )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/stats/summary")
    async def get_stats():
        """Get CAS storage statistics."""
        if _global_cas is None:
            return JSONResponse({"error": "No CAS configured"})

        total_size = 0
        total_count = 0
        size_distribution = {"<1KB": 0, "1KB-1MB": 0, "1MB-100MB": 0, ">100MB": 0}

        for prefix_dir in _global_cas.objects_dir.iterdir():
            if not prefix_dir.is_dir():
                continue

            for obj_path in prefix_dir.iterdir():
                try:
                    size = obj_path.stat().st_size
                    total_size += size
                    total_count += 1

                    if size < 1024:
                        size_distribution["<1KB"] += 1
                    elif size < 1024 * 1024:
                        size_distribution["1KB-1MB"] += 1
                    elif size < 100 * 1024 * 1024:
                        size_distribution["1MB-100MB"] += 1
                    else:
                        size_distribution[">100MB"] += 1
                except Exception:
                    pass

        return JSONResponse(
            {
                "total_objects": total_count,
                "total_size_bytes": total_size,
                "total_size_human": _human_size(total_size),
                "size_distribution": size_distribution,
            }
        )

    return router


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_artifact_browser_html() -> str:
    """Get the artifact browser frontend HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Artifact Browser</title>
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
        .search {
            display: flex;
            gap: 0.5rem;
        }
        .search input {
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            width: 300px;
        }
        .search button {
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .stat-card {
            background: #16213e;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        .stat-card .value {
            font-size: 2rem;
            font-weight: bold;
            color: #e94560;
        }
        .stat-card .label {
            color: #94a3b8;
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }
        .artifacts-table {
            background: #16213e;
            border-radius: 0.5rem;
            overflow: hidden;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 1rem; text-align: left; }
        th {
            background: #0f3460;
            color: #94a3b8;
            font-size: 0.75rem;
            text-transform: uppercase;
        }
        tr { border-bottom: 1px solid #0f3460; }
        tr:hover { background: rgba(233, 69, 96, 0.1); }
        .hash {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.8rem;
        }
        .hash-full { display: none; }
        tr:hover .hash-full { display: inline; }
        tr:hover .hash-short { display: none; }
        .size { color: #94a3b8; }
        .actions button {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 0.75rem;
            margin-right: 0.25rem;
        }
        .actions button:hover { background: #e94560; }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: #16213e;
            padding: 2rem;
            border-radius: 0.5rem;
            max-width: 800px;
            max-height: 80vh;
            overflow: auto;
        }
        .modal-content h3 { margin-bottom: 1rem; }
        .modal-content pre {
            background: #1a1a2e;
            padding: 1rem;
            border-radius: 0.25rem;
            overflow-x: auto;
            font-size: 0.8rem;
        }
        .close-modal {
            float: right;
            background: none;
            border: none;
            color: #eee;
            font-size: 1.5rem;
            cursor: pointer;
        }
        .pagination {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        .pagination button {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .pagination button:disabled { opacity: 0.5; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Artifact Browser (CAS)</h1>
        <div class="search">
            <input type="text" id="search" placeholder="Search by hash prefix...">
            <button onclick="searchArtifacts()">Search</button>
            <button onclick="loadArtifacts()">Refresh</button>
        </div>
    </div>
    <div class="container">
        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="value" id="total-objects">-</div>
                <div class="label">Total Objects</div>
            </div>
            <div class="stat-card">
                <div class="value" id="total-size">-</div>
                <div class="label">Total Size</div>
            </div>
            <div class="stat-card">
                <div class="value" id="small-files">-</div>
                <div class="label">Files < 1MB</div>
            </div>
            <div class="stat-card">
                <div class="value" id="large-files">-</div>
                <div class="label">Files > 1MB</div>
            </div>
        </div>
        <div class="artifacts-table">
            <table>
                <thead>
                    <tr>
                        <th>Hash</th>
                        <th>Size</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="artifacts-body">
                </tbody>
            </table>
        </div>
        <div class="pagination">
            <button id="prev-btn" onclick="prevPage()" disabled>Previous</button>
            <span id="page-info">Page 1</span>
            <button id="next-btn" onclick="nextPage()">Next</button>
        </div>
    </div>

    <div id="modal" class="modal">
        <div class="modal-content">
            <button class="close-modal" onclick="closeModal()">&times;</button>
            <h3 id="modal-title">Artifact Details</h3>
            <pre id="modal-content"></pre>
        </div>
    </div>

    <script>
        let currentPage = 0;
        const pageSize = 50;
        let searchPrefix = '';

        async function loadStats() {
            const res = await fetch('/api/artifacts/stats/summary');
            const data = await res.json();

            document.getElementById('total-objects').textContent = data.total_objects || 0;
            document.getElementById('total-size').textContent = data.total_size_human || '0 B';

            const dist = data.size_distribution || {};
            document.getElementById('small-files').textContent =
                (dist['<1KB'] || 0) + (dist['1KB-1MB'] || 0);
            document.getElementById('large-files').textContent =
                (dist['1MB-100MB'] || 0) + (dist['>100MB'] || 0);
        }

        async function loadArtifacts() {
            const res = await fetch(
                `/api/artifacts/?limit=${pageSize}&offset=${currentPage * pageSize}` +
                (searchPrefix ? `&prefix=${searchPrefix}` : '')
            );
            const data = await res.json();

            const tbody = document.getElementById('artifacts-body');
            tbody.innerHTML = '';

            data.artifacts.forEach(a => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="hash">
                        <span class="hash-short">${a.hash_short}...</span>
                        <span class="hash-full">${a.hash}</span>
                    </td>
                    <td class="size">${a.size_human}</td>
                    <td class="actions">
                        <button onclick="viewDetails('${a.hash}')">Details</button>
                        <button onclick="viewPreview('${a.hash}')">Preview</button>
                    </td>
                `;
                tbody.appendChild(tr);
            });

            document.getElementById('page-info').textContent =
                `Page ${currentPage + 1} (${data.artifacts.length} of ${data.total})`;
            document.getElementById('prev-btn').disabled = currentPage === 0;
            document.getElementById('next-btn').disabled =
                (currentPage + 1) * pageSize >= data.total;
        }

        async function viewDetails(hash) {
            const res = await fetch(`/api/artifacts/${hash}`);
            const data = await res.json();

            document.getElementById('modal-title').textContent = `Artifact: ${hash.slice(0, 8)}...`;
            document.getElementById('modal-content').textContent = JSON.stringify(data, null, 2);
            document.getElementById('modal').classList.add('active');
        }

        async function viewPreview(hash) {
            const res = await fetch(`/api/artifacts/${hash}/preview`);
            const data = await res.json();

            document.getElementById('modal-title').textContent = `Preview: ${hash.slice(0, 8)}...`;
            document.getElementById('modal-content').textContent =
                data.type === 'text' ? data.preview : `[Binary: ${data.preview}]`;
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        function searchArtifacts() {
            searchPrefix = document.getElementById('search').value;
            currentPage = 0;
            loadArtifacts();
        }

        function prevPage() {
            if (currentPage > 0) {
                currentPage--;
                loadArtifacts();
            }
        }

        function nextPage() {
            currentPage++;
            loadArtifacts();
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });

        // Initial load
        loadStats();
        loadArtifacts();
    </script>
</body>
</html>"""
