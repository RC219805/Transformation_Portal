"""Behavioral coverage for ``dashboard.optimization_api``.

Covers the in-memory ``OptimizationJobManager`` (job create/get, broadcast
wiring) and the optimization router's status/history/stop/list endpoints plus
the start endpoint with the background optimizer stubbed out (the real
optimizer pulls heavy eval dependencies and is out of scope here).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import optimization_api
from transformation_portal.dashboard.optimization_api import OptimizationJobManager


@pytest.fixture(autouse=True)
def clear_jobs():
    optimization_api.optimization_manager.jobs.clear()
    yield
    optimization_api.optimization_manager.jobs.clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(optimization_api.create_optimization_router())
    return TestClient(app)


# --------------------------------------------------------------------------- #
# OptimizationJobManager
# --------------------------------------------------------------------------- #


def test_create_and_get_job() -> None:
    mgr = OptimizationJobManager()
    job = mgr.create_job("j1", max_iterations=5)
    assert job.job_id == "j1"
    assert job.max_iterations == 5
    assert mgr.get_job("j1") is job
    assert mgr.get_job("absent") is None


async def test_broadcast_invokes_fn_when_set() -> None:
    received = []
    mgr = OptimizationJobManager()

    async def sink(evt):
        received.append(evt)

    mgr.set_broadcast(sink)
    await mgr._broadcast({"type": "x"})
    assert received[-1] == {"type": "x"}


async def test_broadcast_noop_without_fn() -> None:
    mgr = OptimizationJobManager()
    await mgr._broadcast({"type": "x"})  # must not raise


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def test_start_optimization_creates_job(client: TestClient, monkeypatch) -> None:
    # Stub the background optimizer so no heavy eval work runs.
    async def _stub_run(*args, **kwargs):
        return None

    monkeypatch.setattr(optimization_api.optimization_manager, "run_optimization", _stub_run)

    resp = client.post("/optimize/start", json={"max_iterations": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert body["job_id"] in optimization_api.optimization_manager.jobs


def test_status_found_and_not_found(client: TestClient) -> None:
    optimization_api.optimization_manager.create_job("known", max_iterations=2)
    ok = client.get("/optimize/status/known").json()
    assert ok["job_id"] == "known"
    assert ok["max_iterations"] == 2

    assert client.get("/optimize/status/ghost").json() == {"error": "Job not found"}


def test_history_found_and_not_found(client: TestClient) -> None:
    optimization_api.optimization_manager.create_job("known")
    assert client.get("/optimize/history/known").json() == {"job_id": "known", "history": []}
    assert client.get("/optimize/history/ghost").json() == {"error": "Job not found"}


def test_stop_found_and_not_found(client: TestClient) -> None:
    optimization_api.optimization_manager.create_job("known")
    body = client.post("/optimize/stop/known").json()
    assert body == {"job_id": "known", "status": "stopped"}
    assert optimization_api.optimization_manager.get_job("known").status == "stopped"

    assert client.post("/optimize/stop/ghost").json() == {"error": "Job not found"}


def test_list_jobs(client: TestClient) -> None:
    optimization_api.optimization_manager.create_job("a")
    optimization_api.optimization_manager.create_job("b")
    jobs = client.get("/optimize/jobs").json()["jobs"]
    assert {j["job_id"] for j in jobs} == {"a", "b"}


def test_create_router_returns_none_without_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimization_api, "FASTAPI_AVAILABLE", False)
    assert optimization_api.create_optimization_router() is None


# --------------------------------------------------------------------------- #
# Stop semantics and registry bounds (audit finding D12)
# --------------------------------------------------------------------------- #


def _patch_optimizer(monkeypatch: pytest.MonkeyPatch, optimizer_cls) -> None:
    monkeypatch.setattr("transformation_portal.evals.auto_optimizer.AutoOptimizer", optimizer_cls)


async def test_stop_before_worker_starts_is_not_overwritten(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Optimizer:
        def __init__(self, **kwargs):
            self.run_fn = kwargs["run_fn"]

        def optimize(self, pipeline):
            calls.append("optimize")
            raise AssertionError("a stopped job must not run")

    _patch_optimizer(monkeypatch, _Optimizer)
    mgr = OptimizationJobManager()
    mgr.create_job("j", max_iterations=3)
    mgr.get_job("j").status = "stopped"

    job = await mgr.run_optimization("j", {}, lambda p: {"score": 0.5}, lambda r: 0.5, lambda p, r: {})

    assert job.status == "stopped"
    assert job.error is None
    assert calls == []


async def test_stop_during_run_interrupts_the_optimizer_and_stays_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[int] = []
    events: list[dict] = []
    mgr = OptimizationJobManager()

    async def sink(evt):
        events.append(evt)

    mgr.set_broadcast(sink)

    class _Result:
        best_score = 0.9
        iterations = 100
        history: list = []

    class _Optimizer:
        def __init__(self, **kwargs):
            self.run_fn = kwargs["run_fn"]
            self.diff_fn = kwargs["diff_fn"]

        def optimize(self, pipeline):
            for i in range(100):
                if i == 2:
                    # The stop endpoint flips the status while the worker runs.
                    mgr.get_job("j").status = "stopped"
                self.diff_fn(pipeline, {})
                self.run_fn(pipeline)
                runs.append(i)
            return _Result()

    _patch_optimizer(monkeypatch, _Optimizer)
    mgr.create_job("j", max_iterations=100)

    job = await mgr.run_optimization("j", {}, lambda p: {"score": 0.5}, lambda r: 0.5, lambda p, r: {})

    assert job.status == "stopped"
    assert job.error is None
    assert runs == [0, 1]
    assert [e["type"] for e in events] == ["optimization_started", "optimization_stopped"]


async def test_stop_racing_the_last_iteration_is_not_reported_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = OptimizationJobManager()

    class _Result:
        best_score = 0.9
        iterations = 1
        history: list = []

    class _Optimizer:
        def __init__(self, **kwargs):
            pass

        def optimize(self, pipeline):
            mgr.get_job("j").status = "stopped"
            return _Result()

    _patch_optimizer(monkeypatch, _Optimizer)
    mgr.create_job("j")

    job = await mgr.run_optimization("j", {}, lambda p: {}, lambda r: 0.0, lambda p, r: {})

    assert job.status == "stopped"


def test_stop_does_not_rewrite_a_finished_job(client: TestClient) -> None:
    job = optimization_api.optimization_manager.create_job("done")
    job.status = "completed"

    body = client.post("/optimize/stop/done").json()

    assert body == {"job_id": "done", "status": "completed"}
    assert job.status == "completed"


def test_registry_evicts_finished_jobs_oldest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimization_api, "MAX_RETAINED_JOBS", 3)
    mgr = OptimizationJobManager()
    for jid in ("a", "b", "c"):
        mgr.create_job(jid).status = "completed"
    mgr.get_job("a").status = "running"

    mgr.create_job("d")

    assert list(mgr.jobs) == ["a", "c", "d"]


def test_registry_full_of_active_jobs_refuses_new_ones(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimization_api, "MAX_RETAINED_JOBS", 2)
    mgr = optimization_api.optimization_manager
    mgr.create_job("a").status = "running"
    mgr.create_job("b").status = "running"

    with pytest.raises(optimization_api.OptimizationCapacityError):
        mgr.create_job("c")

    resp = client.post("/optimize/start", json={"max_iterations": 1})
    assert resp.status_code == 503
    assert set(mgr.jobs) == {"a", "b"}
