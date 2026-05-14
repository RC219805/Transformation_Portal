"""Unit tests for the dashboard autonomous-optimization API.

Covers the OptimizationJobState dataclass, the OptimizationJobManager
lifecycle (including broadcast fan-out and the success/error branches of
run_optimization with an injected fake optimizer), and the FastAPI router
endpoints.
"""

from __future__ import annotations

from typing import Any, Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import optimization_api
from transformation_portal.dashboard.optimization_api import (
    OptimizationJobManager,
    OptimizationJobState,
    create_optimization_router,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_manager() -> Generator[None, None, None]:
    """Clear the module-level optimization manager between tests."""
    optimization_api.optimization_manager.jobs.clear()
    optimization_api.optimization_manager._broadcast_fn = None
    yield
    optimization_api.optimization_manager.jobs.clear()
    optimization_api.optimization_manager._broadcast_fn = None


class _FakeResult:
    def __init__(self) -> None:
        self.best_score = 0.87
        self.iterations = 3
        self.history = [_FakeStep(), _FakeStep()]


class _FakeStep:
    def to_dict(self) -> dict[str, Any]:
        return {"score": 0.5}


class _FakeOptimizer:
    """Stand-in for evals.AutoOptimizer; records construction kwargs."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        _FakeOptimizer.last_kwargs = kwargs

    def optimize(self, pipeline: dict[str, Any]) -> _FakeResult:
        return _FakeResult()


class TestOptimizationJobState:
    """Tests for the OptimizationJobState dataclass."""

    def test_defaults(self) -> None:
        job = OptimizationJobState(job_id="job_1")

        assert job.job_id == "job_1"
        assert job.status == "pending"
        assert job.progress == 0.0
        assert job.current_iteration == 0
        assert job.max_iterations == 10
        assert job.current_score == 0.0
        assert job.best_score == 0.0
        assert job.history == []
        assert job.error is None


class TestJobManagerLifecycle:
    """Tests for create_job / get_job."""

    def test_create_job_registers_state(self) -> None:
        manager = OptimizationJobManager()

        job = manager.create_job("job_1", max_iterations=25)

        assert job.job_id == "job_1"
        assert job.max_iterations == 25
        assert manager.get_job("job_1") is job

    def test_get_job_returns_none_for_unknown(self) -> None:
        assert OptimizationJobManager().get_job("missing") is None


class TestBroadcast:
    """Tests for the broadcast fan-out helper."""

    @pytest.mark.asyncio
    async def test_broadcast_without_fn_is_noop(self) -> None:
        manager = OptimizationJobManager()

        await manager._broadcast({"type": "noop"})  # must not raise

    @pytest.mark.asyncio
    async def test_broadcast_invokes_registered_fn(self) -> None:
        manager = OptimizationJobManager()
        received: list[dict[str, Any]] = []

        async def _capture(event: dict[str, Any]) -> None:
            received.append(event)

        manager.set_broadcast(_capture)
        await manager._broadcast({"type": "ping"})

        assert received == [{"type": "ping"}]


class TestRunOptimization:
    """Tests for the run_optimization success and error branches."""

    @pytest.mark.asyncio
    async def test_success_branch_updates_job_and_broadcasts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transformation_portal.evals.auto_optimizer.AutoOptimizer",
            _FakeOptimizer,
        )
        manager = OptimizationJobManager()
        events: list[dict[str, Any]] = []

        async def _capture(event: dict[str, Any]) -> None:
            events.append(event)

        manager.set_broadcast(_capture)
        manager.create_job("job_1", max_iterations=3)

        job = await manager.run_optimization(
            "job_1",
            {"nodes": []},
            run_fn=lambda p: {"score": 0.5},
            eval_fn=lambda r: r["score"],
            diff_fn=lambda p, r: {"changes": []},
        )

        assert job.status == "completed"
        assert job.best_score == pytest.approx(0.87)
        assert job.current_iteration == 3
        assert job.progress == 1.0
        assert len(job.history) == 2
        event_types = [e["type"] for e in events]
        assert event_types == ["optimization_started", "optimization_completed"]

    @pytest.mark.asyncio
    async def test_creates_job_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transformation_portal.evals.auto_optimizer.AutoOptimizer",
            _FakeOptimizer,
        )
        manager = OptimizationJobManager()

        job = await manager.run_optimization(
            "auto_created",
            {},
            run_fn=lambda p: {},
            eval_fn=lambda r: 0.0,
            diff_fn=lambda p, r: {},
        )

        assert job.job_id == "auto_created"
        assert manager.get_job("auto_created") is job

    @pytest.mark.asyncio
    async def test_error_branch_records_failure_and_broadcasts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _explode(**kwargs: Any) -> None:
            raise RuntimeError("optimizer unavailable")

        monkeypatch.setattr(
            "transformation_portal.evals.auto_optimizer.AutoOptimizer",
            _explode,
        )
        manager = OptimizationJobManager()
        events: list[dict[str, Any]] = []

        async def _capture(event: dict[str, Any]) -> None:
            events.append(event)

        manager.set_broadcast(_capture)
        manager.create_job("job_1")

        job = await manager.run_optimization(
            "job_1",
            {},
            run_fn=lambda p: {},
            eval_fn=lambda r: 0.0,
            diff_fn=lambda p, r: {},
        )

        assert job.status == "error"
        assert "optimizer unavailable" in job.error
        assert events[-1]["type"] == "optimization_error"


class TestOptimizationRouter:
    """Tests for the FastAPI router endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        app = FastAPI()
        app.include_router(create_optimization_router())
        return TestClient(app)

    def test_start_returns_job_id(self, client: TestClient) -> None:
        response = client.post("/optimize/start", json={"pipeline": {}, "max_iterations": 2})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "started"
        assert isinstance(body["job_id"], str) and body["job_id"]

    def test_status_unknown_job_returns_error(self, client: TestClient) -> None:
        assert client.get("/optimize/status/missing").json() == {"error": "Job not found"}

    def test_status_returns_job_fields(self, client: TestClient) -> None:
        optimization_api.optimization_manager.create_job("job_1", max_iterations=4)

        body = client.get("/optimize/status/job_1").json()

        assert body["job_id"] == "job_1"
        assert body["status"] == "pending"
        assert body["max_iterations"] == 4
        assert body["progress"] == 0.0
        assert body["error"] is None

    def test_history_unknown_job_returns_error(self, client: TestClient) -> None:
        assert client.get("/optimize/history/missing").json() == {"error": "Job not found"}

    def test_history_returns_recorded_steps(self, client: TestClient) -> None:
        job = optimization_api.optimization_manager.create_job("job_1")
        job.history = [{"score": 0.1}, {"score": 0.2}]

        body = client.get("/optimize/history/job_1").json()

        assert body["job_id"] == "job_1"
        assert body["history"] == [{"score": 0.1}, {"score": 0.2}]

    def test_stop_unknown_job_returns_error(self, client: TestClient) -> None:
        assert client.post("/optimize/stop/missing").json() == {"error": "Job not found"}

    def test_stop_marks_job_stopped(self, client: TestClient) -> None:
        optimization_api.optimization_manager.create_job("job_1")

        body = client.post("/optimize/stop/job_1").json()

        assert body == {"job_id": "job_1", "status": "stopped"}
        assert optimization_api.optimization_manager.get_job("job_1").status == "stopped"

    def test_jobs_lists_all_registered_jobs(self, client: TestClient) -> None:
        optimization_api.optimization_manager.create_job("job_1")
        optimization_api.optimization_manager.create_job("job_2")

        body = client.get("/optimize/jobs").json()

        job_ids = {j["job_id"] for j in body["jobs"]}
        assert job_ids == {"job_1", "job_2"}
        for entry in body["jobs"]:
            assert set(entry) == {"job_id", "status", "progress", "best_score"}
