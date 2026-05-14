"""Unit tests for the dashboard DAG visualization API.

Exercises the router's empty-state fallbacks plus the populated paths backed
by a real PriorityDAGScheduler and MerkleDAG.
"""

from __future__ import annotations

from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import dag_api
from transformation_portal.execution_graph.scheduler import (
    PriorityDAGScheduler,
    ResourceRequirements,
)
from transformation_portal.storage.merkle_dag import MerkleDAG

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_globals() -> Generator[None, None, None]:
    """Ensure the module-level DAG/scheduler globals do not leak across tests."""
    dag_api.set_merkle_dag(None)
    dag_api.set_scheduler(None)
    yield
    dag_api.set_merkle_dag(None)
    dag_api.set_scheduler(None)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(dag_api.create_dag_router())
    return TestClient(app)


class _StubNode:
    """A no-op DAG node; the visualization endpoints never call run()."""

    def run(self, *args: object, **kwargs: object) -> dict:
        return {}


def _scheduler_with_two_nodes() -> PriorityDAGScheduler:
    scheduler = PriorityDAGScheduler()
    scheduler.add_node(
        "ingest",
        _StubNode(),
        priority=10,
        resources=ResourceRequirements(gpu=False, gpu_memory_mb=0),
    )
    scheduler.add_node(
        "segment",
        _StubNode(),
        deps=["ingest"],
        priority=5,
        resources=ResourceRequirements(gpu=True, gpu_memory_mb=2048),
    )
    return scheduler


class TestGlobalSetters:
    """Tests for set_merkle_dag / set_scheduler."""

    def test_set_scheduler_updates_global(self) -> None:
        scheduler = PriorityDAGScheduler()
        dag_api.set_scheduler(scheduler)
        assert dag_api._global_scheduler is scheduler

    def test_set_merkle_dag_updates_global(self) -> None:
        dag = MerkleDAG()
        dag_api.set_merkle_dag(dag)
        assert dag_api._global_merkle_dag is dag


class TestDagGraphEndpoint:
    """Tests for GET /api/dag/graph."""

    def test_returns_error_payload_without_scheduler(self, client: TestClient) -> None:
        response = client.get("/api/dag/graph")

        assert response.status_code == 200
        body = response.json()
        assert body["nodes"] == []
        assert body["edges"] == []
        assert "error" in body

    def test_returns_nodes_and_edges_with_scheduler(self, client: TestClient) -> None:
        dag_api.set_scheduler(_scheduler_with_two_nodes())

        body = client.get("/api/dag/graph").json()

        node_ids = {n["id"] for n in body["nodes"]}
        assert node_ids == {"ingest", "segment"}
        assert {"source": "ingest", "target": "segment"} in body["edges"]
        assert body["execution_order"] == ["ingest", "segment"]

        segment = next(n for n in body["nodes"] if n["id"] == "segment")
        assert segment["status"] == "pending"
        assert segment["priority"] == 5
        assert segment["resources"] == {"gpu": True, "gpu_memory_mb": 2048}
        assert segment["score"] is None

    def test_marks_completed_nodes_from_results(self, client: TestClient) -> None:
        scheduler = _scheduler_with_two_nodes()
        scheduler.results["ingest"] = {"score": 0.91}
        dag_api.set_scheduler(scheduler)

        body = client.get("/api/dag/graph").json()

        ingest = next(n for n in body["nodes"] if n["id"] == "ingest")
        assert ingest["status"] == "completed"
        assert ingest["outputs"] == {"score": 0.91}
        assert ingest["score"] == 0.91


class TestMerkleGraphEndpoint:
    """Tests for GET /api/dag/merkle."""

    def test_returns_error_payload_without_dag(self, client: TestClient) -> None:
        body = client.get("/api/dag/merkle").json()

        assert body["nodes"] == []
        assert body["edges"] == []
        assert "error" in body

    def test_returns_nodes_edges_and_summary_with_dag(self, client: TestClient) -> None:
        dag = MerkleDAG()
        artifact_hash = dag.add_artifact(artifact_type="image", content_hash="c0ffee")
        comp_hash = dag.add_computation(
            node_id="segment",
            inputs=[artifact_hash],
            outputs={"mask": "m1"},
        )
        dag_api.set_merkle_dag(dag)

        body = client.get("/api/dag/merkle").json()

        node_ids = {n["id"] for n in body["nodes"]}
        assert {artifact_hash, comp_hash} <= node_ids
        assert {"source": artifact_hash, "target": comp_hash} in body["edges"]
        assert "summary" in body


class TestMerkleNodeEndpoint:
    """Tests for GET /api/dag/merkle/{node_hash}."""

    def test_returns_404_without_dag(self, client: TestClient) -> None:
        assert client.get("/api/dag/merkle/abc").status_code == 404

    def test_returns_404_for_unknown_node(self, client: TestClient) -> None:
        dag_api.set_merkle_dag(MerkleDAG())

        assert client.get("/api/dag/merkle/does-not-exist").status_code == 404

    def test_returns_node_details(self, client: TestClient) -> None:
        dag = MerkleDAG()
        artifact_hash = dag.add_artifact(artifact_type="image", content_hash="c0ffee")
        comp_hash = dag.add_computation(
            node_id="segment",
            inputs=[artifact_hash],
            outputs={"mask": "m1"},
        )
        dag_api.set_merkle_dag(dag)

        body = client.get(f"/api/dag/merkle/{comp_hash}").json()

        assert body["hash"] == comp_hash
        assert artifact_hash in body["inputs"]
        assert body["outputs"] == {"mask": "m1"}


class TestMerkleLineageEndpoint:
    """Tests for GET /api/dag/merkle/{node_hash}/lineage."""

    def test_returns_404_without_dag(self, client: TestClient) -> None:
        assert client.get("/api/dag/merkle/abc/lineage").status_code == 404

    def test_returns_404_for_unknown_node(self, client: TestClient) -> None:
        dag_api.set_merkle_dag(MerkleDAG())

        assert client.get("/api/dag/merkle/missing/lineage").status_code == 404

    def test_returns_lineage_chain(self, client: TestClient) -> None:
        dag = MerkleDAG()
        artifact_hash = dag.add_artifact(artifact_type="image", content_hash="c0ffee")
        comp_hash = dag.add_computation(
            node_id="segment",
            inputs=[artifact_hash],
            outputs={"mask": "m1"},
        )
        dag_api.set_merkle_dag(dag)

        body = client.get(f"/api/dag/merkle/{comp_hash}/lineage").json()

        assert body["target"] == comp_hash
        assert body["depth"] >= 1
        lineage_hashes = {entry["hash"] for entry in body["lineage"]}
        assert comp_hash in lineage_hashes


class TestVisualizationHtml:
    """Tests for the static visualization HTML helper."""

    def test_html_is_self_contained_document(self) -> None:
        html = dag_api.get_dag_visualization_html()

        assert html.startswith("<!DOCTYPE html>")
        assert "/api/dag/graph" in html
        assert html.strip().endswith("</html>")
