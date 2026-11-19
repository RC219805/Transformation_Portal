"""Dashboard API for autonomous optimization.

This module provides API endpoints for running and monitoring
autonomous pipeline optimization from the dashboard UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Optional FastAPI import. These names must be resolvable at module scope:
# `from __future__ import annotations` defers annotation evaluation, so
# FastAPI resolves endpoint annotations (e.g. ``BackgroundTasks``) against
# the module globals rather than any function-local import.
try:
    from fastapi import APIRouter, BackgroundTasks, HTTPException

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without FastAPI installed
    FASTAPI_AVAILABLE = False
    APIRouter = None
    BackgroundTasks = None
    HTTPException = None

# Upper bound on jobs kept in the in-memory registry. Finished jobs are evicted
# oldest-first once it is reached; when every retained job is still active a
# new job is refused instead of growing the registry without bound.
MAX_RETAINED_JOBS = 256

STATUS_STOPPED = "stopped"
TERMINAL_STATUSES = frozenset({"completed", "error", STATUS_STOPPED})


class OptimizationStopped(Exception):
    """Raised inside the optimizer callbacks once a stop was requested."""


class OptimizationCapacityError(RuntimeError):
    """Raised when the job registry is full of still-active jobs."""


@dataclass
class OptimizationJobState:
    """State of an optimization job."""

    job_id: str
    status: str = "pending"
    progress: float = 0.0
    current_iteration: int = 0
    max_iterations: int = 10
    current_score: float = 0.0
    best_score: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class OptimizationJobManager:
    """Manager for optimization jobs."""

    def __init__(self) -> None:
        self.jobs: dict[str, OptimizationJobState] = {}
        self._broadcast_fn: Callable | None = None

    def set_broadcast(self, fn: Callable) -> None:
        """Set broadcast function for WebSocket updates."""
        self._broadcast_fn = fn

    async def _broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast event to connected clients."""
        if self._broadcast_fn:
            await self._broadcast_fn(event)

    def _evict_finished_jobs(self) -> None:
        """Drop the oldest finished jobs until there is room for one more."""
        for existing_id in [jid for jid, j in self.jobs.items() if j.status in TERMINAL_STATUSES]:
            if len(self.jobs) < MAX_RETAINED_JOBS:
                return
            del self.jobs[existing_id]
        if len(self.jobs) >= MAX_RETAINED_JOBS:
            raise OptimizationCapacityError(f"too many active optimization jobs (limit {MAX_RETAINED_JOBS})")

    def create_job(self, job_id: str, max_iterations: int = 10) -> OptimizationJobState:
        """Create a new optimization job.

        Raises:
            OptimizationCapacityError: when ``MAX_RETAINED_JOBS`` active jobs
                are already registered.
        """
        if job_id not in self.jobs:
            self._evict_finished_jobs()
        job = OptimizationJobState(job_id=job_id, max_iterations=max_iterations)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> OptimizationJobState | None:
        """Get job state by ID."""
        return self.jobs.get(job_id)

    async def run_optimization(
        self,
        job_id: str,
        pipeline: dict[str, Any],
        run_fn: Callable,
        eval_fn: Callable,
        diff_fn: Callable,
    ) -> OptimizationJobState:
        """Run optimization job asynchronously.

        Args:
            job_id: Job identifier
            pipeline: Initial pipeline configuration
            run_fn: Pipeline runner function
            eval_fn: Evaluation function
            diff_fn: Diff generation function

        Returns:
            Final job state
        """
        job = self.jobs.get(job_id)
        if job is None:
            job = self.create_job(job_id)

        if job.status == STATUS_STOPPED:
            # Stopped before the background task got to run: do not start it.
            return job

        job.status = "running"
        await self._broadcast(
            {
                "type": "optimization_started",
                "job_id": job_id,
            }
        )

        try:
            from transformation_portal.evals.auto_opt_types import OptimizationConfig
            from transformation_portal.evals.auto_optimizer import AutoOptimizer

            config = OptimizationConfig(max_iterations=job.max_iterations)

            def _raise_if_stopped() -> None:
                # Read from the worker thread; the stop endpoint sets the flag.
                if job.status == STATUS_STOPPED:
                    raise OptimizationStopped(job_id)

            def stoppable_run(p: Any) -> Any:
                _raise_if_stopped()
                return run_fn(p)

            def stoppable_diff(p: Any, r: Any) -> Any:
                _raise_if_stopped()
                return diff_fn(p, r)

            optimizer = AutoOptimizer(
                run_fn=stoppable_run,
                eval_fn=eval_fn,
                diff_fn=stoppable_diff,
                config=config,
            )

            # Run with progress callbacks
            result = await self._run_with_progress(optimizer, pipeline, job)

            if job.status == STATUS_STOPPED:
                # A stop raced the final iteration: never report it as completed.
                await self._broadcast_stopped(job_id)
                return job

            job.status = "completed"
            job.best_score = result.best_score

            await self._broadcast(
                {
                    "type": "optimization_completed",
                    "job_id": job_id,
                    "best_score": result.best_score,
                    "iterations": result.iterations,
                }
            )

        except Exception as e:
            if job.status == STATUS_STOPPED:
                await self._broadcast_stopped(job_id)
                return job
            logger.error("Optimization failed: %s", e)
            job.status = "error"
            job.error = str(e)

            await self._broadcast(
                {
                    "type": "optimization_error",
                    "job_id": job_id,
                    "error": str(e),
                }
            )

        return job

    async def _broadcast_stopped(self, job_id: str) -> None:
        await self._broadcast({"type": "optimization_stopped", "job_id": job_id})

    async def _run_with_progress(
        self,
        optimizer: Any,
        pipeline: dict[str, Any],
        job: OptimizationJobState,
    ) -> Any:
        """Run optimizer with progress updates.

        Uses asyncio.to_thread to avoid blocking the event loop.
        """
        import asyncio

        # Run the synchronous optimizer.optimize() in a worker thread
        # to avoid blocking the event loop
        result = await asyncio.to_thread(optimizer.optimize, pipeline)

        # Update job state from result
        job.current_iteration = result.iterations
        job.best_score = result.best_score
        job.progress = 1.0
        job.history = [s.to_dict() for s in result.history]

        return result


# Global manager instance
optimization_manager = OptimizationJobManager()


def create_optimization_router():
    """Create FastAPI router for optimization endpoints.

    Returns:
        APIRouter with optimization endpoints
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, optimization API disabled")
        return None

    router = APIRouter(prefix="/optimize", tags=["optimization"])

    @router.post("/start")
    async def start_optimization(
        payload: dict,
        background_tasks: BackgroundTasks,
    ):
        """Start an optimization job.

        Request body:
            pipeline: Pipeline configuration
            max_iterations: Maximum iterations (default: 10)
        """
        import uuid

        job_id = str(uuid.uuid4())[:8]
        max_iters = payload.get("max_iterations", 10)

        try:
            job = optimization_manager.create_job(job_id, max_iters)
        except OptimizationCapacityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        # Mock functions for now - would be injected in real usage
        def mock_run(p):
            return {"score": 0.5, "metrics": {}}

        def mock_eval(r):
            return r.get("score", 0.0)

        def mock_diff(p, r):
            return {"changes": []}

        background_tasks.add_task(
            optimization_manager.run_optimization,
            job_id,
            payload.get("pipeline", {}),
            mock_run,
            mock_eval,
            mock_diff,
        )

        return {"job_id": job_id, "status": "started"}

    @router.get("/status/{job_id}")
    async def get_optimization_status(job_id: str):
        """Get status of an optimization job."""
        job = optimization_manager.get_job(job_id)

        if job is None:
            return {"error": "Job not found"}

        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "current_iteration": job.current_iteration,
            "max_iterations": job.max_iterations,
            "current_score": job.current_score,
            "best_score": job.best_score,
            "error": job.error,
        }

    @router.get("/history/{job_id}")
    async def get_optimization_history(job_id: str):
        """Get optimization history for a job."""
        job = optimization_manager.get_job(job_id)

        if job is None:
            return {"error": "Job not found"}

        return {
            "job_id": job.job_id,
            "history": job.history,
        }

    @router.post("/stop/{job_id}")
    async def stop_optimization(job_id: str):
        """Stop an optimization job."""
        job = optimization_manager.get_job(job_id)

        if job is None:
            return {"error": "Job not found"}

        if job.status in TERMINAL_STATUSES:
            # Already finished: a stop must not rewrite its terminal outcome.
            return {"job_id": job_id, "status": job.status}

        # The running optimizer checks this before every pipeline run / diff.
        job.status = STATUS_STOPPED

        return {"job_id": job_id, "status": "stopped"}

    @router.get("/jobs")
    async def list_optimization_jobs():
        """List all optimization jobs."""
        return {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "progress": j.progress,
                    "best_score": j.best_score,
                }
                for j in optimization_manager.jobs.values()
            ]
        }

    return router


# Export router if FastAPI available
try:
    optimization_router = create_optimization_router()
except Exception:
    optimization_router = None
