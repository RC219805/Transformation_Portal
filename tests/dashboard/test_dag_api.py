"""Behavioral coverage for ``dashboard.dag_api``.

Drives the DAG/Merkle visualization router via ``TestClient`` against fake
scheduler and Merkle DAG stand-ins (injected through the module setters) so
both the configured and not-configured branches of every route are exercised
offline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import dag_api


class _FakeScheduler:
    def __init__(self) -> None:
        self.nodes: dict = {}
        self.results: dict = {}

    def add(self, node_id, *, priority=0, deps=(), gpu=False, gpu_memory_mb=0):
        self.nodes[node_id] = SimpleNamespace(
            priority=priority,
            deps=list(deps),
            resources=SimpleNamespace(gpu=gpu, gpu_memory_mb=gpu_memory_mb),
        )

    def get_execution_order(self):
        return list(self.nodes)


class _MerkleNode:
    def __init__(self, h, *, inputs=(), outputs=None, metadata=None, node_type="compute", timestamp="t0"):
        self.hash = h
        self.node_type = node_type
        self.inputs = list(inputs)
        self.outputs = outputs or {}
        self.metadata = metadata or {}
        self.timestamp = timestamp


class _FakeMerkle:
    def __init__(self) -> None:
        self.nodes: dict = {}

    def get_node(self, h):
        return self.nodes.get(h)

    def get_lineage(self, h, max_depth=10):
        node = self.nodes.get(h)
        return [node] if node else []

    def summary(self):
        return {"node_count": len(self.nodes)}


@pytest.fixture
def restore_globals():
    orig_s, orig_m = dag_api._global_scheduler, dag_api._global_merkle_dag
    yield
    dag_api.set_scheduler(orig_s)  # type: ignore[arg-type]
    dag_api.set_merkle_dag(orig_m)  # type: ignore[arg-type]


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(dag_api.create_dag_router())
    return TestClient(app)


# --------------------------------------------------------------------------- #
# /graph
# --------------------------------------------------------------------------- #


def test_graph_without_scheduler_returns_error(restore_globals) -> None:
    dag_api.set_scheduler(None)  # type: ignore[arg-type]
    body = _client().get("/api/dag/graph").json()
    assert body["nodes"] == [] and body["edges"] == []
    assert "error" in body


def test_graph_with_scheduler(restore_globals) -> None:
    sched = _FakeScheduler()
    sched.add("a", priority=-3, deps=[], gpu=False)
    sched.add("b", priority=-1, deps=["a"], gpu=True, gpu_memory_mb=2048)
    sched.results = {"a": {"score": 0.7}}
    dag_api.set_scheduler(sched)  # type: ignore[arg-type]

    body = _client().get("/api/dag/graph").json()
    ids = {n["id"]: n for n in body["nodes"]}
    assert ids["a"]["status"] == "completed"  # in results
    assert ids["a"]["priority"] == 3  # negated back to original
    assert ids["a"]["score"] == 0.7
    assert ids["b"]["status"] == "pending"
    assert ids["b"]["resources"]["gpu"] is True
    assert {"source": "a", "target": "b"} in body["edges"]
    assert body["execution_order"] == ["a", "b"]


# --------------------------------------------------------------------------- #
# /merkle
# --------------------------------------------------------------------------- #


def test_merkle_graph_without_dag_returns_error(restore_globals) -> None:
    dag_api.set_merkle_dag(None)  # type: ignore[arg-type]
    body = _client().get("/api/dag/merkle").json()
    assert body["nodes"] == [] and "error" in body


def test_merkle_graph_with_dag(restore_globals) -> None:
    merkle = _FakeMerkle()
    merkle.nodes["h1"] = _MerkleNode("h1")
    merkle.nodes["h2"] = _MerkleNode("h2", inputs=["h1"])
    dag_api.set_merkle_dag(merkle)  # type: ignore[arg-type]

    body = _client().get("/api/dag/merkle").json()
    assert {n["id"] for n in body["nodes"]} == {"h1", "h2"}
    assert {"source": "h1", "target": "h2"} in body["edges"]
    assert body["summary"] == {"node_count": 2}


def test_merkle_node_detail_and_404s(restore_globals) -> None:
    merkle = _FakeMerkle()
    merkle.nodes["h1"] = _MerkleNode("h1", inputs=["seed"], outputs={"x": 1})
    dag_api.set_merkle_dag(merkle)  # type: ignore[arg-type]
    client = _client()

    found = client.get("/api/dag/merkle/h1").json()
    assert found["hash"] == "h1"
    assert found["inputs"] == ["seed"]

    assert client.get("/api/dag/merkle/ghost").status_code == 404

    dag_api.set_merkle_dag(None)  # type: ignore[arg-type]
    assert _client().get("/api/dag/merkle/h1").status_code == 404


def test_merkle_lineage_and_404s(restore_globals) -> None:
    merkle = _FakeMerkle()
    merkle.nodes["h1"] = _MerkleNode("h1")
    dag_api.set_merkle_dag(merkle)  # type: ignore[arg-type]
    client = _client()

    body = client.get("/api/dag/merkle/h1/lineage").json()
    assert body["target"] == "h1"
    assert body["depth"] == 1

    assert client.get("/api/dag/merkle/ghost/lineage").status_code == 404  # empty lineage

    dag_api.set_merkle_dag(None)  # type: ignore[arg-type]
    assert _client().get("/api/dag/merkle/h1/lineage").status_code == 404


def test_setters_mutate_globals(restore_globals) -> None:
    s, m = _FakeScheduler(), _FakeMerkle()
    dag_api.set_scheduler(s)  # type: ignore[arg-type]
    dag_api.set_merkle_dag(m)  # type: ignore[arg-type]
    assert dag_api._global_scheduler is s
    assert dag_api._global_merkle_dag is m


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dag_api, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        dag_api.create_dag_router()
