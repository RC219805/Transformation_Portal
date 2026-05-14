"""Unit tests for the dashboard experiment tracking API.

Exercises the SQLite-backed Python API (experiments, runs, metrics) against
a per-test temporary database, the Experiment/Run dataclasses, and the
FastAPI router endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import experiment_api
from transformation_portal.dashboard.experiment_api import (
    Experiment,
    Run,
    complete_run,
    create_experiment,
    create_run,
    get_experiment,
    get_run,
    init_db,
    list_experiments,
    list_runs,
    log_metrics,
    set_db_path,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _temp_db(tmp_path: Path) -> Generator[Path, None, None]:
    """Point the module at an isolated SQLite database for each test."""
    original = experiment_api._db_path
    db_path = tmp_path / "experiments.db"
    set_db_path(db_path)
    yield db_path
    set_db_path(original)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(experiment_api.create_experiment_router())
    with TestClient(app) as tc:
        yield tc


class TestDataclasses:
    """Tests for the Experiment / Run dataclasses."""

    def test_experiment_defaults(self) -> None:
        exp = Experiment(id=1, name="exp")
        assert exp.description is None
        assert exp.tags == []

    def test_run_defaults(self) -> None:
        run = Run(id=1, experiment_id=2)
        assert run.status == "pending"
        assert run.config == {}
        assert run.params == {}
        assert run.metrics == {}
        assert run.artifacts == []


class TestInitDb:
    """Tests for schema initialization."""

    def test_init_db_creates_schema(self, _temp_db: Path) -> None:
        init_db()
        assert _temp_db.exists()
        # init_db is idempotent.
        init_db()


class TestExperimentApi:
    """Tests for the experiment Python API."""

    def test_create_and_get_experiment(self) -> None:
        exp_id = create_experiment("alpha", description="first", tags=["a", "b"])

        exp = get_experiment(exp_id)
        assert exp is not None
        assert exp.name == "alpha"
        assert exp.description == "first"
        assert exp.tags == ["a", "b"]

    def test_get_experiment_returns_none_for_unknown(self) -> None:
        init_db()
        assert get_experiment(99999) is None

    def test_list_experiments_returns_all(self) -> None:
        create_experiment("alpha")
        create_experiment("beta")

        names = {e.name for e in list_experiments()}
        assert names == {"alpha", "beta"}

    def test_list_experiments_empty_initially(self) -> None:
        assert list_experiments() == []

    def test_duplicate_name_raises_integrity_error(self) -> None:
        import sqlite3

        create_experiment("dup")
        with pytest.raises(sqlite3.IntegrityError):
            create_experiment("dup")


class TestRunApi:
    """Tests for the run Python API."""

    def test_create_and_get_run(self) -> None:
        exp_id = create_experiment("alpha")
        run_id = create_run(exp_id, name="run-1", config={"lr": 0.1}, params={"seed": 7})

        run = get_run(run_id)
        assert run is not None
        assert run.experiment_id == exp_id
        assert run.name == "run-1"
        assert run.status == "running"
        assert run.config == {"lr": 0.1}
        assert run.params == {"seed": 7}
        assert run.start_time is not None

    def test_get_run_returns_none_for_unknown(self) -> None:
        init_db()
        assert get_run(99999) is None

    def test_log_metrics_merges(self) -> None:
        exp_id = create_experiment("alpha")
        run_id = create_run(exp_id)

        log_metrics(run_id, {"loss": 0.5})
        log_metrics(run_id, {"accuracy": 0.9})

        run = get_run(run_id)
        assert run.metrics == {"loss": 0.5, "accuracy": 0.9}

    def test_complete_run_sets_status_and_end_time(self) -> None:
        exp_id = create_experiment("alpha")
        run_id = create_run(exp_id)

        complete_run(run_id, status="failed")

        run = get_run(run_id)
        assert run.status == "failed"
        assert run.end_time is not None

    def test_list_runs_scoped_to_experiment(self) -> None:
        exp_a = create_experiment("alpha")
        exp_b = create_experiment("beta")
        create_run(exp_a, name="a1")
        create_run(exp_a, name="a2")
        create_run(exp_b, name="b1")

        assert len(list_runs(exp_a)) == 2
        assert len(list_runs(exp_b)) == 1


class TestRouter:
    """Tests for the FastAPI router endpoints."""

    def test_list_experiments_empty(self, client: TestClient) -> None:
        body = client.get("/api/experiments/").json()
        assert body == {"experiments": []}

    def test_create_experiment_endpoint(self, client: TestClient) -> None:
        body = client.post("/api/experiments/", params={"name": "alpha"}).json()

        assert body["name"] == "alpha"
        assert isinstance(body["id"], int)

    def test_create_duplicate_experiment_returns_400(self, client: TestClient) -> None:
        client.post("/api/experiments/", params={"name": "dup"})

        response = client.post("/api/experiments/", params={"name": "dup"})
        assert response.status_code == 400

    def test_get_experiment_endpoint(self, client: TestClient) -> None:
        exp_id = client.post("/api/experiments/", params={"name": "alpha"}).json()["id"]

        body = client.get(f"/api/experiments/{exp_id}").json()
        assert body["name"] == "alpha"
        assert body["run_count"] == 0

    def test_get_experiment_404(self, client: TestClient) -> None:
        assert client.get("/api/experiments/99999").status_code == 404

    def test_create_run_endpoint(self, client: TestClient) -> None:
        exp_id = client.post("/api/experiments/", params={"name": "alpha"}).json()["id"]

        body = client.post(f"/api/experiments/{exp_id}/runs", params={"name": "run-1"}).json()
        assert body["experiment_id"] == exp_id
        assert isinstance(body["id"], int)

    def test_create_run_for_missing_experiment_404(self, client: TestClient) -> None:
        assert client.post("/api/experiments/99999/runs").status_code == 404

    def test_list_runs_endpoint(self, client: TestClient) -> None:
        exp_id = client.post("/api/experiments/", params={"name": "alpha"}).json()["id"]
        client.post(f"/api/experiments/{exp_id}/runs", params={"name": "run-1"})

        body = client.get(f"/api/experiments/{exp_id}/runs").json()
        assert len(body["runs"]) == 1
        assert body["runs"][0]["name"] == "run-1"

    def test_log_metrics_endpoint(self, client: TestClient) -> None:
        exp_id = client.post("/api/experiments/", params={"name": "alpha"}).json()["id"]
        run_id = client.post(f"/api/experiments/{exp_id}/runs").json()["id"]

        response = client.post(f"/api/experiments/runs/{run_id}/metrics", json={"loss": 0.25})
        assert response.json() == {"status": "ok"}

    def test_log_metrics_missing_run_404(self, client: TestClient) -> None:
        assert client.post("/api/experiments/runs/99999/metrics", json={"loss": 0.1}).status_code == 404

    def test_complete_run_endpoint(self, client: TestClient) -> None:
        exp_id = client.post("/api/experiments/", params={"name": "alpha"}).json()["id"]
        run_id = client.post(f"/api/experiments/{exp_id}/runs").json()["id"]

        response = client.post(f"/api/experiments/runs/{run_id}/complete")
        assert response.json() == {"status": "ok"}

    def test_complete_run_missing_404(self, client: TestClient) -> None:
        assert client.post("/api/experiments/runs/99999/complete").status_code == 404

    def test_ui_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/experiments/ui")
        assert response.status_code == 200
        assert "Experiment Tracking" in response.text


class TestHtmlHelper:
    """Tests for the static HTML helper."""

    def test_experiments_html_is_self_contained(self) -> None:
        html = experiment_api.get_experiments_html()

        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")
