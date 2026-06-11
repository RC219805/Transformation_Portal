"""Behavioral coverage for ``dashboard.dag_editor_api``.

Drives the pipeline-editor CRUD router via ``TestClient`` against a
``tmp_path`` pipelines directory (set with ``set_pipelines_dir``). Covers the
save/get/list/delete round-trip, pydantic node/edge coercion, the node-types
catalog, and the security-relevant name-validation rejection path.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import dag_editor_api


@pytest.fixture
def client(tmp_path):
    original = dag_editor_api._pipelines_dir
    dag_editor_api.set_pipelines_dir(tmp_path / "pipelines")
    app = FastAPI()
    app.include_router(dag_editor_api.create_dag_editor_router())
    try:
        yield TestClient(app)
    finally:
        dag_editor_api.set_pipelines_dir(original)


def _pipeline_payload() -> dict:
    return {
        "nodes": [{"id": "n1", "label": "Ingest", "type": "ingest", "position": {"x": 0.0, "y": 0.0}}],
        "edges": [{"id": "e1", "source": "n1", "target": "n2"}],
        "metadata": {"author": "test"},
    }


def test_save_get_list_delete_roundtrip(client: TestClient) -> None:
    # Save
    save = client.post("/api/editor/pipelines/demo", json=_pipeline_payload())
    assert save.status_code == 200
    assert save.json() == {"status": "ok", "name": "demo"}

    # Get returns the persisted definition with name overridden from the URL.
    got = client.get("/api/editor/pipelines/demo").json()
    assert got["name"] == "demo"
    assert len(got["nodes"]) == 1
    assert got["nodes"][0]["id"] == "n1"

    # List reflects the saved pipeline with computed counts.
    listing = client.get("/api/editor/pipelines").json()["pipelines"]
    entry = next(p for p in listing if p["name"] == "demo")
    assert entry["node_count"] == 1
    assert entry["edge_count"] == 1

    # Delete then 404 on subsequent fetch.
    assert client.delete("/api/editor/pipelines/demo").json() == {"status": "ok"}
    assert client.get("/api/editor/pipelines/demo").status_code == 404


def test_get_unknown_pipeline_404(client: TestClient) -> None:
    assert client.get("/api/editor/pipelines/nope").status_code == 404


def test_delete_unknown_pipeline_404(client: TestClient) -> None:
    assert client.delete("/api/editor/pipelines/nope").status_code == 404


def test_list_empty(client: TestClient) -> None:
    assert client.get("/api/editor/pipelines").json() == {"pipelines": []}


def test_node_types_catalog(client: TestClient) -> None:
    body = client.get("/api/editor/node-types").json()
    types = {nt["type"] for nt in body["node_types"]}
    assert "ingest" in types


def test_invalid_pipeline_name_rejected(client: TestClient) -> None:
    # A traversal-style name must be rejected by validation, not written.
    resp = client.get("/api/editor/pipelines/..%2F..%2Fetc")
    assert resp.status_code in (400, 404)  # 400 from validation; 404 if router declines the path


def test_save_rejects_invalid_name(client: TestClient) -> None:
    resp = client.post("/api/editor/pipelines/bad name!", json=_pipeline_payload())
    assert resp.status_code == 400


def test_save_coerces_empty_node_and_edge_lists(client: TestClient) -> None:
    resp = client.post("/api/editor/pipelines/empty", json={"nodes": [], "edges": []})
    assert resp.status_code == 200
    got = client.get("/api/editor/pipelines/empty").json()
    assert got["nodes"] == []
    assert got["edges"] == []
