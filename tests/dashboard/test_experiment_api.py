"""Behavioral coverage for ``dashboard.experiment_api``.

The experiment-tracking module pairs a pure SQLite-backed Python API
(create/get/list experiments and runs, log metrics, complete runs) with a
FastAPI router over it. These tests point the module-global DB path at a
``tmp_path`` file and exercise both layers offline:

- Python API: experiment + run CRUD, metric merge-on-update, the
  missing-row metrics branch, and ``get_experiment`` miss
- Router: list/create (incl. duplicate-name ``IntegrityError`` -> 400),
  experiment detail + 404, run create/list + unknown-experiment 404,
  metric logging + unknown-run 404, run completion + 404, the HTML UI route,
  and the FastAPI-availability guard

The DB path is restored after each test so no ``experiments.db`` leaks into
the repo and tests stay isolated.
"""

from __future__ import annotations

import sqlite3

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import experiment_api


@pytest.fixture
def db(tmp_path):
    """Point the module DB at a tmp file with schema initialized."""
    original = experiment_api._db_path
    experiment_api.set_db_path(tmp_path / "exp.db")
    experiment_api.init_db()
    try:
        yield
    finally:
        experiment_api.set_db_path(original)


@pytest.fixture
def client(db):
    app = FastAPI()
    app.include_router(experiment_api.create_experiment_router())
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Python API
# --------------------------------------------------------------------------- #


def test_create_get_and_list_experiments(db) -> None:
    exp_id = experiment_api.create_experiment("alpha", description="first", tags=["t1", "t2"])
    assert isinstance(exp_id, int)

    exp = experiment_api.get_experiment(exp_id)
    assert exp is not None
    assert exp.name == "alpha"
    assert exp.description == "first"
    assert exp.tags == ["t1", "t2"]

    experiment_api.create_experiment("beta")
    names = {e.name for e in experiment_api.list_experiments()}
    assert names == {"alpha", "beta"}


def test_get_experiment_miss_returns_none(db) -> None:
    assert experiment_api.get_experiment(9999) is None


def test_run_lifecycle_create_metrics_complete(db) -> None:
    exp_id = experiment_api.create_experiment("exp")
    run_id = experiment_api.create_run(exp_id, name="r1", config={"lr": 0.1}, params={"seed": 7})

    run = experiment_api.get_run(run_id)
    assert run is not None
    assert run.status == "running"
    assert run.config == {"lr": 0.1}
    assert run.params == {"seed": 7}
    assert run.start_time is not None

    experiment_api.log_metrics(run_id, {"acc": 0.5})
    experiment_api.log_metrics(run_id, {"loss": 1.2})  # merges, not replaces
    assert experiment_api.get_run(run_id).metrics == {"acc": 0.5, "loss": 1.2}

    experiment_api.complete_run(run_id, status="failed")
    completed = experiment_api.get_run(run_id)
    assert completed.status == "failed"
    assert completed.end_time is not None


def test_list_runs_orders_and_scopes_by_experiment(db) -> None:
    exp_a = experiment_api.create_experiment("a")
    exp_b = experiment_api.create_experiment("b")
    experiment_api.create_run(exp_a, name="a1")
    experiment_api.create_run(exp_a, name="a2")
    experiment_api.create_run(exp_b, name="b1")

    a_runs = experiment_api.list_runs(exp_a)
    assert {r.name for r in a_runs} == {"a1", "a2"}
    assert [r.name for r in experiment_api.list_runs(exp_b)] == ["b1"]


def test_get_run_miss_returns_none(db) -> None:
    assert experiment_api.get_run(4242) is None


def test_log_metrics_on_missing_run_is_noop(db) -> None:
    # No row exists; the function must not raise (existing defaults to {}).
    experiment_api.log_metrics(123, {"x": 1})
    assert experiment_api.get_run(123) is None


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def test_api_list_experiments_empty_then_populated(client: TestClient) -> None:
    assert client.get("/api/experiments/").json() == {"experiments": []}

    client.post("/api/experiments/", params={"name": "alpha"})
    body = client.get("/api/experiments/").json()
    assert [e["name"] for e in body["experiments"]] == ["alpha"]


def test_api_create_experiment_duplicate_returns_400(client: TestClient) -> None:
    first = client.post("/api/experiments/", params={"name": "dup"})
    assert first.status_code == 200
    second = client.post("/api/experiments/", params={"name": "dup"})
    assert second.status_code == 400
    assert "already exists" in second.json()["detail"]


def test_api_get_experiment_detail_and_404(client: TestClient) -> None:
    exp_id = client.post("/api/experiments/", params={"name": "withruns"}).json()["id"]
    client.post(f"/api/experiments/{exp_id}/runs", json={"config": {}, "params": {}})

    detail = client.get(f"/api/experiments/{exp_id}").json()
    assert detail["name"] == "withruns"
    assert detail["run_count"] == 1
    assert len(detail["runs"]) == 1

    assert client.get("/api/experiments/9999").status_code == 404


def test_api_create_run_unknown_experiment_404(client: TestClient) -> None:
    resp = client.post("/api/experiments/777/runs", json={"config": {}, "params": {}})
    assert resp.status_code == 404


def test_api_list_runs(client: TestClient) -> None:
    exp_id = client.post("/api/experiments/", params={"name": "e"}).json()["id"]
    client.post(f"/api/experiments/{exp_id}/runs", json={})
    runs = client.get(f"/api/experiments/{exp_id}/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "running"


def test_api_log_metrics_success_and_404(client: TestClient) -> None:
    exp_id = client.post("/api/experiments/", params={"name": "m"}).json()["id"]
    run_id = client.post(f"/api/experiments/{exp_id}/runs", json={}).json()["id"]

    ok = client.post(f"/api/experiments/runs/{run_id}/metrics", json={"acc": 0.9})
    assert ok.status_code == 200
    assert experiment_api.get_run(run_id).metrics == {"acc": 0.9}

    assert client.post("/api/experiments/runs/999/metrics", json={"acc": 1}).status_code == 404


def test_api_complete_run_success_and_404(client: TestClient) -> None:
    exp_id = client.post("/api/experiments/", params={"name": "c"}).json()["id"]
    run_id = client.post(f"/api/experiments/{exp_id}/runs", json={}).json()["id"]

    ok = client.post(f"/api/experiments/runs/{run_id}/complete", params={"status": "completed"})
    assert ok.status_code == 200
    assert experiment_api.get_run(run_id).status == "completed"

    assert client.post("/api/experiments/runs/999/complete").status_code == 404


def test_router_startup_event_initializes_db(db) -> None:
    # Driving the app through its lifespan context fires the router startup
    # handler (init_db); the endpoint then responds without error.
    app = FastAPI()
    app.include_router(experiment_api.create_experiment_router())
    with TestClient(app) as ctx_client:
        assert ctx_client.get("/api/experiments/").status_code == 200


def test_experiments_ui_serves_html(client: TestClient) -> None:
    resp = client.get("/api/experiments/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<html" in resp.text.lower()


def test_set_db_path_mutates_global(tmp_path) -> None:
    original = experiment_api._db_path
    target = tmp_path / "custom.db"
    try:
        experiment_api.set_db_path(target)
        assert experiment_api._db_path == target
    finally:
        experiment_api.set_db_path(original)


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_api, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        experiment_api.create_experiment_router()
