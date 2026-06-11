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
