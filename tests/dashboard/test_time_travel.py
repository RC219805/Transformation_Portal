"""Behavioral coverage for ``dashboard.time_travel``.

``time_travel.py`` exposes a FastAPI router for node version history, run
snapshots, and artifact comparison, wired to module-global ``NodeStateStore``
and ``MerkleDAG`` references via setters. These tests drive the router through
``TestClient`` against a real (in-memory) ``NodeStateStore`` and a fake Merkle
DAG so every route branch is exercised deterministically — no app, no network,
no ML:

- store-not-configured fallbacks (empty history / 503 snapshot)
- history aggregation across runs, recency sort, and ``limit`` truncation
- Merkle-lineage enrichment (hash present + node found / not found)
- run snapshot success and 404 for unknown runs
- artifact-compare URL shaping
- HTML UI endpoints + the FastAPI-availability guard
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import time_travel
from transformation_portal.dashboard.node_state_store import NodeStateStore


class _FakeMerkleNode:
    def __init__(self, inputs: List[str], metadata: Dict[str, Any]) -> None:
        self.inputs = inputs
        self.metadata = metadata


class _FakeMerkleDAG:
    """Minimal MerkleDAG stand-in exposing ``get_node``."""

    def __init__(self, nodes: Dict[str, _FakeMerkleNode]) -> None:
        self._nodes = nodes

    def get_node(self, node_hash: str) -> _FakeMerkleNode | None:
        return self._nodes.get(node_hash)


@pytest.fixture
def store() -> NodeStateStore:
    """A store with two runs sharing a node id, for history aggregation."""
    s = NodeStateStore()
    s.init_run("run_old", ["ingest", "segment"])
    s.set_status("run_old", "ingest", "complete")
    s.update_outputs("run_old", "ingest", {"rgb": "a"})
    s.set_merkle_hash("run_old", "ingest", "mhash-old")

    s.init_run("run_new", ["ingest"])
    s.set_status("run_new", "ingest", "complete")
    s.update_outputs("run_new", "ingest", {"rgb": "b"})
    return s


@pytest.fixture
def client(store: NodeStateStore):
    """TestClient with the router wired to a populated store (no merkle)."""
    # Reset module globals around each test to avoid cross-test bleed.
    original_store = time_travel._node_store
    original_merkle = time_travel._merkle_dag
    time_travel.set_time_travel_store(store)
    time_travel.set_time_travel_merkle(None)  # type: ignore[arg-type]

    app = FastAPI()
    app.include_router(time_travel.create_time_travel_router())
    try:
        yield TestClient(app)
    finally:
        time_travel.set_time_travel_store(original_store)  # type: ignore[arg-type]
        time_travel.set_time_travel_merkle(original_merkle)  # type: ignore[arg-type]


@pytest.fixture
def unconfigured_client():
    """TestClient with the store explicitly unconfigured (None)."""
    original_store = time_travel._node_store
    time_travel.set_time_travel_store(None)  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(time_travel.create_time_travel_router())
    try:
        yield TestClient(app)
    finally:
        time_travel.set_time_travel_store(original_store)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# node history
# --------------------------------------------------------------------------- #


def test_history_aggregates_across_runs_sorted_recent_first(client: TestClient) -> None:
    resp = client.get("/api/timetravel/nodes/ingest/history")
    assert resp.status_code == 200
    body = resp.json()

    assert body["node_id"] == "ingest"
    assert body["total_versions"] == 2
    run_ids = [entry["run_id"] for entry in body["history"]]
    # run_new completed after run_old, so it sorts first (reverse by end_time).
    assert run_ids == ["run_new", "run_old"]
    assert body["history"][0]["outputs"] == {"rgb": "b"}


def test_history_respects_limit(client: TestClient) -> None:
    resp = client.get("/api/timetravel/nodes/ingest/history", params={"limit": 1})
    body = resp.json()
    assert body["total_versions"] == 2  # total counts all matches
    assert len(body["history"]) == 1  # but only `limit` returned
    assert body["history"][0]["run_id"] == "run_new"


def test_history_for_unknown_node_is_empty(client: TestClient) -> None:
    resp = client.get("/api/timetravel/nodes/ghost/history")
    body = resp.json()
    assert body["total_versions"] == 0
    assert body["history"] == []


def test_history_without_store_returns_error_payload(unconfigured_client: TestClient) -> None:
    resp = unconfigured_client.get("/api/timetravel/nodes/ingest/history")
    assert resp.status_code == 200
    assert resp.json() == {"history": [], "error": "Store not configured"}


def test_history_enriches_with_merkle_lineage(store: NodeStateStore) -> None:
    merkle = _FakeMerkleDAG(
        {"mhash-old": _FakeMerkleNode(inputs=["dep-1"], metadata={"run_id": "run_old"})}
    )
    original_store = time_travel._node_store
    original_merkle = time_travel._merkle_dag
    time_travel.set_time_travel_store(store)
    time_travel.set_time_travel_merkle(merkle)  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(time_travel.create_time_travel_router())
    try:
        body = TestClient(app).get("/api/timetravel/nodes/ingest/history").json()
    finally:
        time_travel.set_time_travel_store(original_store)  # type: ignore[arg-type]
        time_travel.set_time_travel_merkle(original_merkle)  # type: ignore[arg-type]

    old_entry = next(e for e in body["history"] if e["run_id"] == "run_old")
    assert old_entry["lineage"] == {"inputs": ["dep-1"], "metadata": {"run_id": "run_old"}}
    # run_new's ingest node had no merkle_hash -> no lineage key.
    new_entry = next(e for e in body["history"] if e["run_id"] == "run_new")
    assert "lineage" not in new_entry


def test_history_skips_lineage_when_merkle_node_missing(store: NodeStateStore) -> None:
    merkle = _FakeMerkleDAG({})  # get_node always returns None
    original_store = time_travel._node_store
    original_merkle = time_travel._merkle_dag
    time_travel.set_time_travel_store(store)
    time_travel.set_time_travel_merkle(merkle)  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(time_travel.create_time_travel_router())
    try:
        body = TestClient(app).get("/api/timetravel/nodes/ingest/history").json()
    finally:
        time_travel.set_time_travel_store(original_store)  # type: ignore[arg-type]
        time_travel.set_time_travel_merkle(original_merkle)  # type: ignore[arg-type]

    assert all("lineage" not in entry for entry in body["history"])


# --------------------------------------------------------------------------- #
# run snapshot
# --------------------------------------------------------------------------- #


def test_run_snapshot_success(client: TestClient) -> None:
    resp = client.get("/api/timetravel/runs/run_old/snapshot")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run_old"
    assert set(body["nodes"]) == {"ingest", "segment"}
    assert body["nodes"]["ingest"]["merkle_hash"] == "mhash-old"


def test_run_snapshot_unknown_run_returns_404(client: TestClient) -> None:
    resp = client.get("/api/timetravel/runs/ghost/snapshot")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Run not found"


def test_run_snapshot_without_store_returns_503(unconfigured_client: TestClient) -> None:
    resp = unconfigured_client.get("/api/timetravel/runs/run_old/snapshot")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Store not configured"


# --------------------------------------------------------------------------- #
# compare + UI + guards
# --------------------------------------------------------------------------- #


def test_compare_shapes_preview_urls(client: TestClient) -> None:
    resp = client.get("/api/timetravel/compare", params={"hash_a": "aaa", "hash_b": "bbb"})
    assert resp.status_code == 200
    assert resp.json() == {
        "hash_a": "aaa",
        "hash_b": "bbb",
        "url_a": "/api/preview/artifact/aaa/raw",
        "url_b": "/api/preview/artifact/bbb/raw",
    }


def test_compare_requires_both_hashes(client: TestClient) -> None:
    resp = client.get("/api/timetravel/compare", params={"hash_a": "only"})
    assert resp.status_code == 422  # missing required query param


@pytest.mark.parametrize("path", ["/api/timetravel/", "/api/timetravel/diff"])
def test_html_endpoints_serve_markup(client: TestClient, path: str) -> None:
    resp = client.get(path)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<!DOCTYPE html>" in resp.text or "<html" in resp.text.lower()


def test_setters_mutate_module_globals() -> None:
    original_store = time_travel._node_store
    original_merkle = time_travel._merkle_dag
    sentinel_store = NodeStateStore()
    sentinel_merkle = _FakeMerkleDAG({})
    try:
        time_travel.set_time_travel_store(sentinel_store)
        time_travel.set_time_travel_merkle(sentinel_merkle)  # type: ignore[arg-type]
        assert time_travel._node_store is sentinel_store
        assert time_travel._merkle_dag is sentinel_merkle
    finally:
        time_travel.set_time_travel_store(original_store)  # type: ignore[arg-type]
        time_travel.set_time_travel_merkle(original_merkle)  # type: ignore[arg-type]


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time_travel, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        time_travel.create_time_travel_router()
