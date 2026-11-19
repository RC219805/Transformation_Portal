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

import math
import sqlite3
import threading

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


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_log_metrics_rejects_non_finite_values_without_writing(db, bad: float) -> None:
    # D3: a NaN/Infinity metric used to be stored as a bare JSON literal that
    # JSONResponse (allow_nan=False) cannot render, poisoning every read
    # endpoint of the experiment. It is now rejected before anything is written.
    exp_id = experiment_api.create_experiment("nonfinite")
    run_id = experiment_api.create_run(exp_id)
    experiment_api.log_metrics(run_id, {"acc": 0.5})

    with pytest.raises(ValueError, match="JSON compliant"):
        experiment_api.log_metrics(run_id, {"loss": bad, "nested": {"x": 1.0}})

    assert experiment_api.get_run(run_id).metrics == {"acc": 0.5}


def test_log_metrics_concurrent_writer_does_not_lose_keys(db, monkeypatch: pytest.MonkeyPatch) -> None:
    # D4: the SELECT-merge-UPDATE in log_metrics was not one write
    # transaction, so a second writer interleaving between the read and the
    # write had its keys overwritten. Force exactly that interleaving: while
    # the first call is between its read and its write, a second thread logs
    # another key. With the write lock taken before the read, the second
    # writer waits and then merges on top of the first one.
    exp_id = experiment_api.create_experiment("race")
    run_id = experiment_api.create_run(exp_id)

    real_loads = experiment_api.json.loads
    errors: list[BaseException] = []
    other: list[threading.Thread] = []

    def other_writer() -> None:
        try:
            experiment_api.log_metrics(run_id, {"from_other": 2})
        except BaseException as exc:  # surfaced by the assertion below
            errors.append(exc)

    def interleaving_loads(*args, **kwargs):
        if not other and threading.current_thread() is threading.main_thread():
            thread = threading.Thread(target=other_writer)
            other.append(thread)
            thread.start()
            # Let the other writer run to completion if this call's read is
            # not protected by a write lock.
            thread.join(timeout=0.5)
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(experiment_api.json, "loads", interleaving_loads)
    try:
        experiment_api.log_metrics(run_id, {"from_first": 1})
        other[0].join(timeout=10)
    finally:
        monkeypatch.undo()

    assert not other[0].is_alive()
    assert errors == []
    assert experiment_api.get_run(run_id).metrics == {"from_first": 1, "from_other": 2}


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


def test_api_log_metrics_nan_is_422_and_reads_stay_200(client: TestClient) -> None:
    # D3: ``{"loss": NaN}`` is accepted by the JSON body parser; it used to be
    # stored and turn GET /{id} and GET /{id}/runs into permanent 500s.
    exp_id = client.post("/api/experiments/", params={"name": "nan"}).json()["id"]
    run_id = client.post(f"/api/experiments/{exp_id}/runs", json={}).json()["id"]

    resp = client.post(
        f"/api/experiments/runs/{run_id}/metrics",
        content=b'{"loss": NaN}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422

    assert client.get(f"/api/experiments/{exp_id}").status_code == 200
    runs = client.get(f"/api/experiments/{exp_id}/runs")
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["metrics"] == {}


def test_api_reads_sanitize_legacy_non_finite_metrics(client: TestClient) -> None:
    # D3: rows written before the rejection may already hold NaN/Infinity
    # literals; reads map them to null instead of failing with a 500.
    exp_id = client.post("/api/experiments/", params={"name": "legacy"}).json()["id"]
    run_id = client.post(f"/api/experiments/{exp_id}/runs", json={}).json()["id"]
    conn = sqlite3.connect(str(experiment_api._db_path))
    try:
        conn.execute(
            "UPDATE runs SET metrics = ? WHERE id = ?",
            ('{"loss": NaN, "hi": Infinity, "lo": -Infinity, "acc": 0.5}', run_id),
        )
        conn.commit()
    finally:
        conn.close()

    expected = {"loss": None, "hi": None, "lo": None, "acc": 0.5}
    detail = client.get(f"/api/experiments/{exp_id}")
    assert detail.status_code == 200
    assert detail.json()["runs"][0]["metrics"] == expected
    runs = client.get(f"/api/experiments/{exp_id}/runs")
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["metrics"] == expected

    # A later finite update merges on top of the sanitized values.
    ok = client.post(f"/api/experiments/runs/{run_id}/metrics", json={"acc": 0.75})
    assert ok.status_code == 200
    assert experiment_api.get_run(run_id).metrics == {**expected, "acc": 0.75}


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


@pytest.mark.parametrize(
    "raw_sink",
    ["<h3>${e.name}</h3>", "<p>${e.description", "<td>${r.name", "${formatMetrics(r.metrics)}</td>"],
)
def test_experiments_ui_escapes_client_supplied_strings(raw_sink: str) -> None:
    # Experiment names/descriptions, run names and metric keys are client-supplied (D5).
    html = experiment_api.get_experiments_html()
    assert "function escapeHtml(value)" in html
    assert raw_sink not in html


def test_set_db_path_mutates_global(tmp_path) -> None:
    original = experiment_api._db_path
    target = tmp_path / "custom.db"
    try:
        experiment_api.set_db_path(target)
        assert experiment_api._db_path == target
    finally:
        experiment_api.set_db_path(original)


def test_read_helpers_do_not_create_database(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # D13: reads against a not-yet-created DB must not leave ./experiments.db behind.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_api, "_db_path", experiment_api.Path("experiments.db"))
    assert experiment_api.list_experiments() == []
    assert experiment_api.get_experiment(1) is None
    assert experiment_api.get_run(1) is None
    assert experiment_api.list_runs(1) == []
    assert list(tmp_path.iterdir()) == []
    # Writes still create the database on demand.
    exp_id = experiment_api.create_experiment("alpha")
    assert (tmp_path / "experiments.db").is_file()
    assert [e.id for e in experiment_api.list_experiments()] == [exp_id]


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_api, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        experiment_api.create_experiment_router()
