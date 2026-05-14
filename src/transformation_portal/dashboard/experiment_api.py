"""Experiment tracking API with SQLite backend.

This module provides experiment and run tracking for ML pipelines,
storing configurations, metrics, and metadata in a SQLite database.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import APIRouter, Body, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None


# Database path
_db_path: Path = Path("experiments.db")


def set_db_path(path: Path) -> None:
    """Set the database file path.

    Args:
        path: Path to SQLite database file
    """
    global _db_path
    _db_path = path


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Get database connection context manager."""
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the database schema."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                name TEXT,
                status TEXT DEFAULT 'pending',
                config TEXT,
                params TEXT,
                metrics TEXT,
                artifacts TEXT,
                start_time TEXT,
                end_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
        """)
        conn.commit()
    logger.info("Database initialized: %s", _db_path)


@dataclass
class Experiment:
    """Experiment record."""

    id: int
    name: str
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class Run:
    """Run record."""

    id: int
    experiment_id: int
    name: Optional[str] = None
    status: str = "pending"
    config: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    created_at: Optional[str] = None


# ============================================================================
# Python API
# ============================================================================


def create_experiment(
    name: str,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> int:
    """Create a new experiment.

    Args:
        name: Experiment name (must be unique)
        description: Optional description
        tags: Optional list of tags

    Returns:
        Experiment ID
    """
    init_db()
    tags_json = json.dumps(tags or [])

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO experiments (name, description, tags) VALUES (?, ?, ?)",
            (name, description, tags_json),
        )
        conn.commit()
        return cursor.lastrowid


def get_experiment(experiment_id: int) -> Optional[Experiment]:
    """Get experiment by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()

        if row is None:
            return None

        return Experiment(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            tags=json.loads(row["tags"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def list_experiments() -> list[Experiment]:
    """List all experiments."""
    init_db()
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()

        return [
            Experiment(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                tags=json.loads(row["tags"] or "[]"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]


def create_run(
    experiment_id: int,
    name: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
) -> int:
    """Create a new run.

    Args:
        experiment_id: Parent experiment ID
        name: Optional run name
        config: Optional configuration dict
        params: Optional parameters dict

    Returns:
        Run ID
    """
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO runs
               (experiment_id, name, status, config, params, start_time)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (
                experiment_id,
                name,
                json.dumps(config or {}),
                json.dumps(params or {}),
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def log_metrics(run_id: int, metrics: dict[str, Any]) -> None:
    """Log metrics for a run.

    Args:
        run_id: Run ID
        metrics: Metrics dictionary
    """
    with get_db() as conn:
        # Get existing metrics
        row = conn.execute(
            "SELECT metrics FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

        existing = json.loads(row["metrics"] or "{}") if row else {}
        existing.update(metrics)

        conn.execute(
            "UPDATE runs SET metrics = ? WHERE id = ?",
            (json.dumps(existing), run_id),
        )
        conn.commit()


def complete_run(run_id: int, status: str = "completed") -> None:
    """Mark a run as completed.

    Args:
        run_id: Run ID
        status: Final status ("completed", "failed", "cancelled")
    """
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, end_time = ? WHERE id = ?",
            (status, now, run_id),
        )
        conn.commit()


def get_run(run_id: int) -> Optional[Run]:
    """Get run by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

        if row is None:
            return None

        return Run(
            id=row["id"],
            experiment_id=row["experiment_id"],
            name=row["name"],
            status=row["status"],
            config=json.loads(row["config"] or "{}"),
            params=json.loads(row["params"] or "{}"),
            metrics=json.loads(row["metrics"] or "{}"),
            artifacts=json.loads(row["artifacts"] or "[]"),
            start_time=row["start_time"],
            end_time=row["end_time"],
            created_at=row["created_at"],
        )


def list_runs(experiment_id: int) -> list[Run]:
    """List runs for an experiment."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY created_at DESC",
            (experiment_id,),
        ).fetchall()

        return [
            Run(
                id=row["id"],
                experiment_id=row["experiment_id"],
                name=row["name"],
                status=row["status"],
                config=json.loads(row["config"] or "{}"),
                params=json.loads(row["params"] or "{}"),
                metrics=json.loads(row["metrics"] or "{}"),
                artifacts=json.loads(row["artifacts"] or "[]"),
                start_time=row["start_time"],
                end_time=row["end_time"],
                created_at=row["created_at"],
            )
            for row in rows
        ]


# ============================================================================
# FastAPI Router
# ============================================================================


def create_experiment_router() -> "APIRouter":
    """Create the experiment tracking router.

    Returns:
        FastAPI APIRouter with experiment endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for experiment API")

    router = APIRouter(prefix="/api/experiments", tags=["experiments"])

    @router.on_event("startup")
    async def startup():
        init_db()

    @router.get("/")
    async def api_list_experiments():
        """List all experiments."""
        experiments = list_experiments()
        return JSONResponse(
            {
                "experiments": [
                    {
                        "id": e.id,
                        "name": e.name,
                        "description": e.description,
                        "tags": e.tags,
                        "created_at": e.created_at,
                    }
                    for e in experiments
                ]
            }
        )

    @router.post("/")
    async def api_create_experiment(
        name: str,
        description: Optional[str] = None,
        tags: list[str] = [],
    ):
        """Create a new experiment."""
        try:
            exp_id = create_experiment(name, description, tags)
            return JSONResponse({"id": exp_id, "name": name})
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail=f"Experiment '{name}' already exists")

    @router.get("/ui", response_class=HTMLResponse)
    async def experiments_ui():
        """Serve the experiments UI."""
        return get_experiments_html()

    @router.get("/{experiment_id}")
    async def api_get_experiment(experiment_id: int):
        """Get experiment details."""
        exp = get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        runs = list_runs(experiment_id)
        return JSONResponse(
            {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "tags": exp.tags,
                "created_at": exp.created_at,
                "run_count": len(runs),
                "runs": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "status": r.status,
                        "metrics": r.metrics,
                        "created_at": r.created_at,
                    }
                    for r in runs[:10]  # Last 10 runs
                ],
            }
        )

    @router.post("/{experiment_id}/runs")
    async def api_create_run(
        experiment_id: int,
        name: Optional[str] = None,
        config: dict = Body(default={}),
        params: dict = Body(default={}),
    ):
        """Create a new run."""
        exp = get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        run_id = create_run(experiment_id, name, config, params)
        return JSONResponse({"id": run_id, "experiment_id": experiment_id})

    @router.get("/{experiment_id}/runs")
    async def api_list_runs(experiment_id: int):
        """List runs for an experiment."""
        runs = list_runs(experiment_id)
        return JSONResponse(
            {
                "runs": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "status": r.status,
                        "metrics": r.metrics,
                        "start_time": r.start_time,
                        "end_time": r.end_time,
                    }
                    for r in runs
                ]
            }
        )

    @router.post("/runs/{run_id}/metrics")
    async def api_log_metrics(run_id: int, metrics: dict = Body(...)):
        """Log metrics for a run."""
        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        log_metrics(run_id, metrics)
        return JSONResponse({"status": "ok"})

    @router.post("/runs/{run_id}/complete")
    async def api_complete_run(run_id: int, status: str = "completed"):
        """Mark run as completed."""
        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        complete_run(run_id, status)
        return JSONResponse({"status": "ok"})

    return router


def get_experiments_html() -> str:
    """Get the experiments UI HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Tracking</title>
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
        .header button {
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .container { padding: 2rem; max-width: 1200px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: 350px 1fr; gap: 2rem; }
        .card {
            background: #16213e;
            border-radius: 0.5rem;
            border: 1px solid #0f3460;
        }
        .card-header {
            padding: 1rem;
            border-bottom: 1px solid #0f3460;
            font-weight: 500;
        }
        .experiment-list { max-height: 600px; overflow-y: auto; }
        .experiment-item {
            padding: 1rem;
            border-bottom: 1px solid #0f3460;
            cursor: pointer;
        }
        .experiment-item:hover { background: rgba(233, 69, 96, 0.1); }
        .experiment-item.selected { background: rgba(233, 69, 96, 0.2); border-left: 3px solid #e94560; }
        .experiment-item h3 { font-size: 0.9rem; margin-bottom: 0.25rem; }
        .experiment-item p { font-size: 0.75rem; color: #94a3b8; }
        .runs-table { width: 100%; }
        .runs-table th {
            text-align: left;
            padding: 0.75rem 1rem;
            background: #0f3460;
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #94a3b8;
        }
        .runs-table td { padding: 0.75rem 1rem; border-bottom: 1px solid #0f3460; }
        .status {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 0.2rem;
            font-size: 0.7rem;
        }
        .status.completed { background: #00d25b; color: #000; }
        .status.running { background: #0d6efd; }
        .status.failed { background: #ff5252; }
        .status.pending { background: #94a3b8; color: #000; }
        .metrics { font-family: monospace; font-size: 0.8rem; }
        .empty { padding: 2rem; text-align: center; color: #94a3b8; }
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
            min-width: 400px;
        }
        .modal-content h3 { margin-bottom: 1rem; }
        .modal-content input {
            width: 100%;
            padding: 0.75rem;
            margin-bottom: 1rem;
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            border-radius: 0.25rem;
        }
        .modal-content .buttons { display: flex; gap: 0.5rem; justify-content: flex-end; }
        .modal-content button {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .modal-content button.primary { background: #e94560; color: #fff; }
        .modal-content button.secondary { background: #0f3460; color: #eee; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Experiment Tracking</h1>
        <button onclick="showCreateModal()">New Experiment</button>
    </div>
    <div class="container">
        <div class="grid">
            <div class="card">
                <div class="card-header">Experiments</div>
                <div id="experiments" class="experiment-list">
                    <div class="empty">Loading...</div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">Runs</div>
                <div id="runs">
                    <div class="empty">Select an experiment</div>
                </div>
            </div>
        </div>
    </div>

    <div id="modal" class="modal">
        <div class="modal-content">
            <h3>Create Experiment</h3>
            <input type="text" id="exp-name" placeholder="Experiment name">
            <input type="text" id="exp-desc" placeholder="Description (optional)">
            <div class="buttons">
                <button class="secondary" onclick="closeModal()">Cancel</button>
                <button class="primary" onclick="createExperiment()">Create</button>
            </div>
        </div>
    </div>

    <script>
        let selectedExperiment = null;

        async function loadExperiments() {
            const res = await fetch('/api/experiments/');
            const data = await res.json();
            const container = document.getElementById('experiments');

            if (data.experiments.length === 0) {
                container.innerHTML = '<div class="empty">No experiments yet</div>';
                return;
            }

            container.innerHTML = data.experiments.map(e => `
                <div class="experiment-item ${selectedExperiment?.id === e.id ? 'selected' : ''}"
                     onclick="selectExperiment(${e.id})">
                    <h3>${e.name}</h3>
                    <p>${e.description || 'No description'}</p>
                </div>
            `).join('');
        }

        async function selectExperiment(id) {
            const res = await fetch(`/api/experiments/${id}`);
            selectedExperiment = await res.json();
            loadExperiments();
            renderRuns(selectedExperiment.runs || []);
        }

        function renderRuns(runs) {
            const container = document.getElementById('runs');

            if (runs.length === 0) {
                container.innerHTML = '<div class="empty">No runs yet</div>';
                return;
            }

            container.innerHTML = `
                <table class="runs-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Status</th>
                            <th>Metrics</th>
                            <th>Created</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${runs.map(r => `
                            <tr>
                                <td>${r.id}</td>
                                <td>${r.name || '-'}</td>
                                <td><span class="status ${r.status}">${r.status}</span></td>
                                <td class="metrics">${formatMetrics(r.metrics)}</td>
                                <td>${new Date(r.created_at).toLocaleString()}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }

        function formatMetrics(metrics) {
            if (!metrics || Object.keys(metrics).length === 0) return '-';
            return Object.entries(metrics)
                .slice(0, 3)
                .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(3) : v}`)
                .join(', ');
        }

        function showCreateModal() {
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        async function createExperiment() {
            const name = document.getElementById('exp-name').value;
            const desc = document.getElementById('exp-desc').value;

            if (!name) return;

            await fetch(`/api/experiments/?name=${encodeURIComponent(name)}&description=${encodeURIComponent(desc)}`, {
                method: 'POST'
            });

            closeModal();
            loadExperiments();
        }

        loadExperiments();
    </script>
</body>
</html>"""
