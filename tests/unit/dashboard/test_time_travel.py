"""Unit tests for the dashboard time-travel / visual-diff API.

Covers the store/merkle setters, the router's empty-state fallbacks and the
node-state-store-backed paths (including Merkle lineage enrichment), and the
static HTML helpers.
"""

from __future__ import annotations

from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import time_travel
from transformation_portal.dashboard.node_state_store import NodeStateStore
from transformation_portal.storage.merkle_dag import MerkleDAG

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_globals() -> Generator[None, None, None]:
    """Ensure the module-level store/merkle globals do not leak across tests."""
    time_travel.set_time_travel_store(None)
    time_travel.set_time_travel_merkle(None)
    yield
    time_travel.set_time_travel_store(None)
    time_travel.set_time_travel_merkle(None)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(time_travel.create_time_travel_router())
    return TestClient(app)


def _store_with_history() -> NodeStateStore:
    """A store where node 'segment' appears in two runs."""
    store = NodeStateStore()
    store.init_run("run_1", ["segment"])
    store.set_status("run_1", "segment", "complete")
    store.update_outputs("run_1", "segment", {"mask": "m1"})
    store.add_artifact("run_1", "segment", "mask.png", "hash_1")

    store.init_run("run_2", ["segment"])
    store.set_status("run_2", "segment", "running")
    return store


class TestRouterFactory:
    """Tests for create_time_travel_router."""

    def test_router_has_timetravel_prefix(self) -> None:
        assert time_travel.create_time_travel_router().prefix == "/api/timetravel"

    def test_factory_raises_without_fastapi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(time_travel, "FASTAPI_AVAILABLE", False)

        with pytest.raises(ImportError, match="FastAPI is required"):
            time_travel.create_time_travel_router()


class TestGlobalSetters:
    """Tests for set_time_travel_store / set_time_travel_merkle."""

    def test_set_store_updates_global(self) -> None:
        store = NodeStateStore()
        time_travel.set_time_travel_store(store)
        assert time_travel._node_store is store

    def test_set_merkle_updates_global(self) -> None:
        dag = MerkleDAG()
        time_travel.set_time_travel_merkle(dag)
        assert time_travel._merkle_dag is dag


class TestNodeHistoryEndpoint:
    """Tests for GET /api/timetravel/nodes/{node_id}/history."""

    def test_returns_error_payload_without_store(self, client: TestClient) -> None:
        body = client.get("/api/timetravel/nodes/segment/history").json()

        assert body["history"] == []
        assert "error" in body

    def test_returns_versions_across_runs(self, client: TestClient) -> None:
        time_travel.set_time_travel_store(_store_with_history())

        body = client.get("/api/timetravel/nodes/segment/history").json()

        assert body["node_id"] == "segment"
        assert body["total_versions"] == 2
        run_ids = {entry["run_id"] for entry in body["history"]}
        assert run_ids == {"run_1", "run_2"}

    def test_empty_history_for_unknown_node(self, client: TestClient) -> None:
        time_travel.set_time_travel_store(_store_with_history())

        body = client.get("/api/timetravel/nodes/does-not-exist/history").json()

        assert body["total_versions"] == 0
        assert body["history"] == []

    def test_limit_caps_returned_versions(self, client: TestClient) -> None:
        time_travel.set_time_travel_store(_store_with_history())

        body = client.get("/api/timetravel/nodes/segment/history?limit=1").json()

        assert body["total_versions"] == 2
        assert len(body["history"]) == 1

    def test_includes_merkle_lineage_when_available(self, client: TestClient) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["segment"])
        store.set_status("run_1", "segment", "complete")

        dag = MerkleDAG()
        artifact_hash = dag.add_artifact(artifact_type="image", content_hash="c0ffee")
        comp_hash = dag.add_computation(
            node_id="segment",
            inputs=[artifact_hash],
            outputs={"mask": "m1"},
        )
        store.set_merkle_hash("run_1", "segment", comp_hash)

        time_travel.set_time_travel_store(store)
        time_travel.set_time_travel_merkle(dag)

        body = client.get("/api/timetravel/nodes/segment/history").json()

        entry = body["history"][0]
        assert "lineage" in entry
        assert artifact_hash in entry["lineage"]["inputs"]


class TestRunSnapshotEndpoint:
    """Tests for GET /api/timetravel/runs/{run_id}/snapshot."""

    def test_returns_503_without_store(self, client: TestClient) -> None:
        assert client.get("/api/timetravel/runs/run_1/snapshot").status_code == 503

    def test_returns_404_for_unknown_run(self, client: TestClient) -> None:
        time_travel.set_time_travel_store(NodeStateStore())

        assert client.get("/api/timetravel/runs/missing/snapshot").status_code == 404

    def test_returns_run_snapshot(self, client: TestClient) -> None:
        time_travel.set_time_travel_store(_store_with_history())

        body = client.get("/api/timetravel/runs/run_1/snapshot").json()

        assert body["run_id"] == "run_1"
        assert "segment" in body["nodes"]
        assert body["nodes"]["segment"]["status"] == "complete"
        assert body["nodes"]["segment"]["artifacts"] == {"mask.png": "hash_1"}


class TestCompareEndpoint:
    """Tests for GET /api/timetravel/compare."""

    def test_returns_comparison_metadata(self, client: TestClient) -> None:
        body = client.get(
            "/api/timetravel/compare",
            params={"hash_a": "aaa", "hash_b": "bbb"},
        ).json()

        assert body["hash_a"] == "aaa"
        assert body["hash_b"] == "bbb"
        assert body["url_a"].endswith("/aaa/raw")
        assert body["url_b"].endswith("/bbb/raw")

    def test_requires_both_hashes(self, client: TestClient) -> None:
        response = client.get("/api/timetravel/compare", params={"hash_a": "aaa"})

        assert response.status_code == 422


class TestHtmlRoutesAndHelpers:
    """Tests for the served HTML routes and their helpers."""

    def test_ui_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/timetravel/")

        assert response.status_code == 200
        assert "Time Travel Debugger" in response.text

    def test_diff_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/timetravel/diff")

        assert response.status_code == 200
        assert "Visual Diff Viewer" in response.text

    def test_time_travel_html_is_self_contained(self) -> None:
        html = time_travel.get_time_travel_html()

        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")

    def test_diff_viewer_html_is_self_contained(self) -> None:
        html = time_travel.get_diff_viewer_html()

        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")
