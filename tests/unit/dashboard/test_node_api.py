"""Unit tests for the dashboard node inspection API.

Exercises the node/run inspection endpoints against the global node state
store, the CAS-backed artifact endpoints, and the static HTML helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import node_api
from transformation_portal.dashboard.node_state_store import NodeStateStore, set_store
from transformation_portal.storage.cas_store import ArtifactStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store() -> NodeStateStore:
    """Install a fresh global node state store for each test."""
    fresh = NodeStateStore()
    set_store(fresh)
    return fresh


@pytest.fixture(autouse=True)
def _reset_cas() -> Generator[None, None, None]:
    node_api.set_cas(None)
    yield
    node_api.set_cas(None)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(node_api.create_node_inspection_router())
    return TestClient(app)


@pytest.fixture
def populated_store(store: NodeStateStore) -> NodeStateStore:
    store.init_run("run_1", ["ingest"])
    store.set_status("run_1", "ingest", "complete")
    store.update_inputs("run_1", "ingest", {"image": "in.png"})
    store.update_outputs("run_1", "ingest", {"rgb": "data"})
    store.add_artifact("run_1", "ingest", "rgb.png", "hash_abc")
    store.add_log("run_1", "ingest", "loaded image")
    return store


class TestNodeDetails:
    """Tests for GET /api/inspect/runs/{run}/nodes/{node}."""

    def test_returns_404_for_unknown_node(self, client: TestClient, store: NodeStateStore) -> None:
        assert client.get("/api/inspect/runs/missing/nodes/missing").status_code == 404

    def test_returns_full_node_state(self, client: TestClient, populated_store: NodeStateStore) -> None:
        body = client.get("/api/inspect/runs/run_1/nodes/ingest").json()

        assert body["node_id"] == "ingest"
        assert body["status"] == "complete"
        assert body["inputs"] == {"image": "in.png"}
        assert body["outputs"] == {"rgb": "data"}

    def test_inputs_endpoint(self, client: TestClient, populated_store: NodeStateStore) -> None:
        body = client.get("/api/inspect/runs/run_1/nodes/ingest/inputs").json()
        assert body == {"node_id": "ingest", "inputs": {"image": "in.png"}}

    def test_inputs_endpoint_404(self, client: TestClient, store: NodeStateStore) -> None:
        assert client.get("/api/inspect/runs/x/nodes/y/inputs").status_code == 404

    def test_outputs_endpoint(self, client: TestClient, populated_store: NodeStateStore) -> None:
        body = client.get("/api/inspect/runs/run_1/nodes/ingest/outputs").json()
        assert body == {"node_id": "ingest", "outputs": {"rgb": "data"}}

    def test_outputs_endpoint_404(self, client: TestClient, store: NodeStateStore) -> None:
        assert client.get("/api/inspect/runs/x/nodes/y/outputs").status_code == 404

    def test_logs_endpoint(self, client: TestClient, populated_store: NodeStateStore) -> None:
        body = client.get("/api/inspect/runs/run_1/nodes/ingest/logs").json()

        assert body["node_id"] == "ingest"
        assert body["total_logs"] == 1
        assert body["logs"][0].endswith("loaded image")

    def test_logs_endpoint_tail(self, client: TestClient, store: NodeStateStore) -> None:
        store.init_run("run_1", ["ingest"])
        for i in range(5):
            store.add_log("run_1", "ingest", f"line {i}")

        body = client.get("/api/inspect/runs/run_1/nodes/ingest/logs?tail=2").json()

        assert body["total_logs"] == 5
        assert len(body["logs"]) == 2

    def test_logs_endpoint_404(self, client: TestClient, store: NodeStateStore) -> None:
        assert client.get("/api/inspect/runs/x/nodes/y/logs").status_code == 404


class TestNodeArtifacts:
    """Tests for GET /api/inspect/runs/{run}/nodes/{node}/artifacts."""

    def test_returns_404_for_unknown_node(self, client: TestClient, store: NodeStateStore) -> None:
        assert client.get("/api/inspect/runs/x/nodes/y/artifacts").status_code == 404

    def test_lists_artifacts_without_cas(self, client: TestClient, populated_store: NodeStateStore) -> None:
        body = client.get("/api/inspect/runs/run_1/nodes/ingest/artifacts").json()

        assert body["node_id"] == "ingest"
        assert body["artifacts"] == [{"name": "rgb.png", "hash": "hash_abc"}]

    def test_enriches_artifacts_with_cas_metadata(self, client: TestClient, store: NodeStateStore, tmp_path: Path) -> None:
        cas = ArtifactStore(tmp_path / "cas")
        obj = cas.add_bytes(b"rgb-bytes")
        node_api.set_cas(cas)

        store.init_run("run_1", ["ingest"])
        store.add_artifact("run_1", "ingest", "rgb.png", obj.sha256)

        body = client.get("/api/inspect/runs/run_1/nodes/ingest/artifacts").json()

        entry = body["artifacts"][0]
        assert entry["size_bytes"] == len(b"rgb-bytes")
        assert entry["exists"] is True

    def test_invalid_artifact_hash_remains_listable_with_cas(
        self,
        client: TestClient,
        populated_store: NodeStateStore,
        tmp_path: Path,
    ) -> None:
        node_api.set_cas(ArtifactStore(tmp_path / "cas"))

        body = client.get("/api/inspect/runs/run_1/nodes/ingest/artifacts").json()

        assert body["artifacts"] == [{"name": "rgb.png", "hash": "hash_abc"}]


class TestRunSummary:
    """Tests for GET /api/inspect/runs/{run}/summary."""

    def test_returns_404_for_unknown_run(self, client: TestClient, store: NodeStateStore) -> None:
        assert client.get("/api/inspect/runs/missing/summary").status_code == 404

    def test_summarizes_run_nodes(self, client: TestClient, populated_store: NodeStateStore) -> None:
        body = client.get("/api/inspect/runs/run_1/summary").json()

        assert body["run_id"] == "run_1"
        node = body["nodes"]["ingest"]
        assert node["status"] == "complete"
        assert node["has_inputs"] is True
        assert node["has_outputs"] is True
        assert node["artifact_count"] == 1
        assert node["log_count"] == 1


class TestArtifactEndpoints:
    """Tests for the CAS-backed /api/inspect/artifact/* endpoints."""

    def test_artifact_info_503_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/inspect/artifact/abc").status_code == 503

    def test_artifact_info_404_for_unknown_hash(self, client: TestClient, tmp_path: Path) -> None:
        node_api.set_cas(ArtifactStore(tmp_path / "cas"))

        assert client.get("/api/inspect/artifact/deadbeef").status_code == 404

    @pytest.mark.parametrize("suffix", ["", "/preview", "/download"])
    def test_artifact_endpoints_return_404_for_invalid_hash(
        self,
        client: TestClient,
        tmp_path: Path,
        suffix: str,
    ) -> None:
        node_api.set_cas(ArtifactStore(tmp_path / "cas"))

        assert client.get(f"/api/inspect/artifact/{'g' * 64}{suffix}").status_code == 404

    def test_artifact_info_returns_metadata(self, client: TestClient, tmp_path: Path) -> None:
        cas = ArtifactStore(tmp_path / "cas")
        obj = cas.add_bytes(b"the payload")
        node_api.set_cas(cas)

        body = client.get(f"/api/inspect/artifact/{obj.sha256}").json()

        assert body["hash"] == obj.sha256
        assert body["size_bytes"] == len(b"the payload")
        assert body["exists"] is True

    def test_artifact_preview_503_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/inspect/artifact/abc/preview").status_code == 503

    def test_artifact_preview_detects_json(self, client: TestClient, tmp_path: Path) -> None:
        cas = ArtifactStore(tmp_path / "cas")
        obj = cas.add_bytes(b'{"key": "value"}')
        node_api.set_cas(cas)

        body = client.get(f"/api/inspect/artifact/{obj.sha256}/preview").json()

        assert body["content_type"] == "application/json"
        assert body["preview"] == '{"key": "value"}'

    def test_artifact_preview_detects_png(self, client: TestClient, tmp_path: Path) -> None:
        cas = ArtifactStore(tmp_path / "cas")
        obj = cas.add_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        node_api.set_cas(cas)

        body = client.get(f"/api/inspect/artifact/{obj.sha256}/preview").json()

        assert body["content_type"] == "image/png"

    def test_artifact_download_503_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/inspect/artifact/abc/download").status_code == 503

    def test_artifact_download_streams_file(self, client: TestClient, tmp_path: Path) -> None:
        cas = ArtifactStore(tmp_path / "cas")
        obj = cas.add_bytes(b"downloadable bytes")
        node_api.set_cas(cas)

        response = client.get(f"/api/inspect/artifact/{obj.sha256}/download")

        assert response.status_code == 200
        assert response.content == b"downloadable bytes"


class TestInspectionUi:
    """Tests for the served HTML route and helper."""

    def test_ui_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/inspect/")

        assert response.status_code == 200
        assert "Node Inspector" in response.text

    def test_html_helper_is_self_contained(self) -> None:
        html = node_api.get_inspection_ui_html()

        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")
