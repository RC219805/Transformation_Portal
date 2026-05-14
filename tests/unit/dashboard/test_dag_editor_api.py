"""Unit tests for the dashboard DAG editor API.

Exercises the FSGuard-backed pipeline CRUD router against a temporary
pipelines directory, the name-validation / path helpers, set_pipelines_dir,
and the static HTML helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from transformation_portal.dashboard import dag_editor_api

pytestmark = pytest.mark.unit

_NODE = {"id": "n1", "type": "ingest", "label": "Node 1", "position": {"x": 1.0, "y": 2.0}}


@pytest.fixture(autouse=True)
def _temp_pipelines_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Point the module at an isolated pipelines directory for each test."""
    original_dir = dag_editor_api._pipelines_dir
    original_ctx = dag_editor_api._fs_context
    pipelines = tmp_path / "pipelines"
    dag_editor_api.set_pipelines_dir(pipelines)
    yield pipelines
    dag_editor_api._pipelines_dir = original_dir
    dag_editor_api._fs_context = original_ctx


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(dag_editor_api.create_dag_editor_router())
    return TestClient(app)


class TestSetPipelinesDirAndContext:
    """Tests for set_pipelines_dir and _get_fs_context."""

    def test_set_pipelines_dir_creates_directory(self, _temp_pipelines_dir: Path) -> None:
        assert _temp_pipelines_dir.exists()
        assert dag_editor_api._pipelines_dir == _temp_pipelines_dir

    def test_get_fs_context_is_cached(self) -> None:
        first = dag_editor_api._get_fs_context()
        second = dag_editor_api._get_fs_context()
        assert first is second

    def test_get_fs_context_rebuilds_when_dir_changes(self, tmp_path: Path) -> None:
        first = dag_editor_api._get_fs_context()
        dag_editor_api.set_pipelines_dir(tmp_path / "other")
        second = dag_editor_api._get_fs_context()
        assert first is not second


class TestNameValidation:
    """Tests for the name-validation / path helpers."""

    def test_validate_pipeline_name_accepts_safe_name(self) -> None:
        assert dag_editor_api._validate_pipeline_name("my_pipeline") == "my_pipeline"

    def test_validate_pipeline_name_rejects_unsafe_name(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            dag_editor_api._validate_pipeline_name("../escape")
        assert exc_info.value.status_code == 400

    def test_get_safe_pipeline_path_rejects_unsafe_name(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            dag_editor_api._get_safe_pipeline_path("../escape")
        assert exc_info.value.status_code == 400

    def test_get_safe_pipeline_path_builds_json_path(self) -> None:
        path = dag_editor_api._get_safe_pipeline_path("my_pipeline")
        assert path.suffix == ".json"
        assert path.stem == "my_pipeline"


class TestPipelineCrud:
    """Tests for the pipeline CRUD endpoints."""

    def test_list_pipelines_empty(self, client: TestClient) -> None:
        body = client.get("/api/editor/pipelines").json()
        assert body == {"pipelines": []}

    def test_save_and_get_pipeline(self, client: TestClient) -> None:
        payload = {"nodes": [_NODE], "edges": []}

        save = client.post("/api/editor/pipelines/demo", json=payload)
        assert save.status_code == 200
        assert save.json() == {"status": "ok", "name": "demo"}

        got = client.get("/api/editor/pipelines/demo").json()
        assert got["name"] == "demo"
        assert len(got["nodes"]) == 1
        assert got["nodes"][0]["id"] == "n1"

    def test_save_overrides_name_from_url(self, client: TestClient) -> None:
        client.post("/api/editor/pipelines/url_name", json={"name": "body_name", "nodes": [], "edges": []})

        got = client.get("/api/editor/pipelines/url_name").json()
        assert got["name"] == "url_name"

    def test_list_pipelines_includes_saved(self, client: TestClient) -> None:
        client.post("/api/editor/pipelines/demo", json={"nodes": [_NODE], "edges": []})

        body = client.get("/api/editor/pipelines").json()
        assert len(body["pipelines"]) == 1
        entry = body["pipelines"][0]
        assert entry["name"] == "demo"
        assert entry["filename"] == "demo.json"
        assert entry["node_count"] == 1
        assert entry["edge_count"] == 0

    def test_get_missing_pipeline_returns_404(self, client: TestClient) -> None:
        assert client.get("/api/editor/pipelines/missing").status_code == 404

    def test_delete_pipeline(self, client: TestClient) -> None:
        client.post("/api/editor/pipelines/demo", json={"nodes": [], "edges": []})

        delete = client.delete("/api/editor/pipelines/demo")
        assert delete.status_code == 200
        assert delete.json() == {"status": "ok"}
        assert client.get("/api/editor/pipelines/demo").status_code == 404

    def test_delete_missing_pipeline_returns_404(self, client: TestClient) -> None:
        assert client.delete("/api/editor/pipelines/missing").status_code == 404


class TestNodeTypesAndUi:
    """Tests for the node-types catalogue and the served UI."""

    def test_node_types_catalogue(self, client: TestClient) -> None:
        body = client.get("/api/editor/node-types").json()

        types = {nt["type"] for nt in body["node_types"]}
        assert {"ingest", "segment", "depth", "materials", "quality", "export"} == types

    def test_editor_ui_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/editor/")

        assert response.status_code == 200
        assert "Pipeline Editor" in response.text

    def test_html_helper_is_self_contained(self) -> None:
        html = dag_editor_api.get_dag_editor_html()

        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")
