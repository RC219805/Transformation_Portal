"""Dashboard API for RL-based optimization.

This module provides API endpoints for running RL-based pipeline
optimization from the dashboard UI.

Optimization jobs use ``MockPipelineEnv`` and a fresh model. Loading a
pre-trained model is unsupported; the deprecated ``model_path`` request field
remains accepted and ignored for backward compatibility.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Bounds for the ``max_iterations`` request field. Out-of-range or non-integer
# values are rejected with 422 instead of reaching the training loop.
MIN_MAX_ITERATIONS = 1
MAX_MAX_ITERATIONS = 10_000

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


def _validate_max_iterations(value: Any) -> int:
    """Return ``value`` as a bounded iteration count or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_iterations must be an integer")
    if not MIN_MAX_ITERATIONS <= value <= MAX_MAX_ITERATIONS:
        raise ValueError(f"max_iterations must be between {MIN_MAX_ITERATIONS} and {MAX_MAX_ITERATIONS}")
    return value


def create_rl_api_router():
    """Create FastAPI router for RL optimization endpoints.

    Returns:
        APIRouter with RL endpoints, or None if FastAPI unavailable
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, RL API disabled")
        return None

    router = APIRouter(prefix="/rl", tags=["rl-optimization"])

    # Global state for running optimizations
    _running_jobs: dict[str, dict[str, Any]] = {}

    @router.post("/optimize")
    async def optimize_rl_api(payload: dict, background_tasks: BackgroundTasks):
        """Start a mock RL optimization job with a fresh model.

        Each job uses MockPipelineEnv; pre-trained model loading is unsupported.

        Request body:
            pipeline: Pipeline configuration
            max_iterations: Maximum training iterations (default: 50, integer
                between 1 and 10000; anything else is rejected with 422)
            model_path: Deprecated, unsupported field; accepted and ignored
                for backward compatibility
        """
        import uuid

        job_id = str(uuid.uuid4())[:8]
        pipeline = payload.get("pipeline", {})
        try:
            max_iters = _validate_max_iterations(payload.get("max_iterations", 50))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        _running_jobs[job_id] = {
            "status": "starting",
            "progress": 0.0,
            "best_score": 0.0,
            "iterations": 0,
        }

        def train_job() -> Any:
            """Run the blocking imports and training loop (worker thread)."""
            # Mock implementation - would use actual RL optimizer
            from transformation_portal.rl.action_space import enumerate_actions
            from transformation_portal.rl.env import MockPipelineEnv
            from transformation_portal.rl.model import create_model
            from transformation_portal.rl.optimize_rl import (
                RLOptimizationConfig,
                train_rl,
            )
            from transformation_portal.rl.state_encoder import get_state_dim
            from transformation_portal.rl.trainer import RLTrainer

            # Create mock environment for testing
            actions = enumerate_actions()
            env = MockPipelineEnv(actions)

            # Create model and trainer
            state_dim = get_state_dim()
            model = create_model(state_dim, len(actions))
            trainer = RLTrainer(model, actions)

            # Run optimization. The trained model is not persisted: jobs are
            # mock runs and must not write into the server's working directory.
            config = RLOptimizationConfig(max_iterations=max_iters, save_model=False)
            return train_rl(env, trainer, pipeline, config)

        async def run_optimization():
            try:
                _running_jobs[job_id]["status"] = "running"
                # Keep the event loop responsive while the job trains.
                result = await asyncio.to_thread(train_job)

                _running_jobs[job_id].update(
                    {
                        "status": "completed",
                        "progress": 1.0,
                        "best_score": result.best_score,
                        "iterations": result.iterations,
                        "result": result.to_dict(),
                    }
                )

            except Exception as e:
                logger.error("RL optimization failed: %s", e)
                _running_jobs[job_id].update(
                    {
                        "status": "error",
                        "error": str(e),
                    }
                )

        background_tasks.add_task(run_optimization)

        return {"job_id": job_id, "status": "started"}

    @router.get("/status/{job_id}")
    async def get_rl_status(job_id: str):
        """Get status of RL optimization job."""
        if job_id not in _running_jobs:
            return {"error": "Job not found"}

        return _running_jobs[job_id]

    @router.get("/actions")
    async def list_actions():
        """List all available RL actions."""
        from transformation_portal.rl.action_space import enumerate_actions

        actions = enumerate_actions()
        return {
            "count": len(actions),
            "actions": [
                {
                    "index": a.index,
                    "node": a.node,
                    "action_type": a.action_type,
                    "params": a.params,
                }
                for a in actions[:50]  # Limit to first 50 for readability
            ],
        }

    @router.get("/policy")
    async def get_policy_config():
        """Get current policy configuration."""
        from transformation_portal.rl.policy_guard import (
            BLOCKED_ACTIONS,
            RISKY_ACTIONS,
            SAFE_ACTIONS,
        )

        return {
            "safe_actions": list(SAFE_ACTIONS),
            "risky_actions": list(RISKY_ACTIONS),
            "blocked_actions": list(BLOCKED_ACTIONS),
        }

    return router


# Export router if FastAPI available
try:
    rl_api_router = create_rl_api_router()
except Exception:
    rl_api_router = None
