"""Unit tests for the dashboard artifact browser API.

Exercises the router against a real on-disk ArtifactStore (CAS) plus the
empty-state fallbacks, the _human_size helper, and the static HTML helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import artifact_api
from transformation_portal.storage.cas_store import ArtifactStore
from transformation_portal.storage.merkle_dag import MerkleDAG

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_globals() -> Generator[None, None, None]:
    """Ensure the module-level CAS/Merkle globals do not leak across tests."""
    artifact_api.set_artifact_store(None)
    artifact_api.set_merkle_dag(None)
    yield
    artifact_api.set_artifact_store(None)
    artifact_api.set_merkle_dag(None)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(artifact_api.create_artifact_router())
    return TestClient(app)


@pytest.fixture
def cas(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "cas")


class TestHumanSize:
    """Tests for the _human_size helper."""

    @pytest.mark.parametrize(
        "size,expected",
        [
            (0, "0.0 B"),
            (512, "512.0 B"),
            (1536, "1.5 KB"),
            (5 * 1024 * 1024, "5.0 MB"),
            (3 * 1024**3, "3.0 GB"),
        ],
    )
    def test_human_size_scales_units(self, size: int, expected: str) -> None:
        assert artifact_api._human_size(size) == expected


class TestListArtifacts:
    """Tests for GET /api/artifacts/."""

    def test_returns_error_payload_without_cas(self, client: TestClient) -> None:
        body = client.get("/api/artifacts/").json()

        assert body["artifacts"] == []
        assert body["total"] == 0
        assert "error" in body

    def test_lists_stored_objects(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"hello world")
        artifact_api.set_artifact_store(cas)

        body = client.get("/api/artifacts/").json()

        assert body["total"] == 1
        assert body["artifacts"][0]["hash"] == obj.sha256
        assert body["artifacts"][0]["size_bytes"] == 11
        assert body["artifacts"][0]["hash_short"] == obj.sha256[:8]

    def test_prefix_filter_excludes_non_matching(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"hello world")
        artifact_api.set_artifact_store(cas)

        matching = client.get(f"/api/artifacts/?prefix={obj.sha256[:4]}").json()
        assert matching["total"] == 1

        non_matching = client.get("/api/artifacts/?prefix=zzzzzz").json()
        assert non_matching["total"] == 0

    def test_limit_caps_results_but_not_total(self, client: TestClient, cas: ArtifactStore) -> None:
        for i in range(3):
            cas.add_bytes(f"object-{i}".encode())
        artifact_api.set_artifact_store(cas)

        body = client.get("/api/artifacts/?limit=2").json()

        assert body["total"] == 3
        assert len(body["artifacts"]) == 2


class TestGetArtifact:
    """Tests for GET /api/artifacts/{hash}."""

    def test_returns_404_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/artifacts/abc123").status_code == 404

    def test_returns_404_for_unknown_hash(self, client: TestClient, cas: ArtifactStore) -> None:
        artifact_api.set_artifact_store(cas)

        assert client.get("/api/artifacts/deadbeef").status_code == 404

    @pytest.mark.parametrize("suffix", ["", "/preview"])
    def test_returns_404_for_invalid_hash(self, client: TestClient, cas: ArtifactStore, suffix: str) -> None:
        artifact_api.set_artifact_store(cas)

        assert client.get(f"/api/artifacts/{'g' * 64}{suffix}").status_code == 404

    def test_returns_metadata_for_known_artifact(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"hello world")
        artifact_api.set_artifact_store(cas)

        body = client.get(f"/api/artifacts/{obj.sha256}").json()

        assert body["hash"] == obj.sha256
        assert body["size_bytes"] == 11
        assert body["exists"] is True
        assert body["related_merkle_nodes"] == []

    def test_includes_related_merkle_nodes(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"payload")
        dag = MerkleDAG()
        node_hash = dag.add_computation(
            node_id="produce",
            inputs=[],
            outputs={"content_hash": obj.sha256},
        )
        artifact_api.set_artifact_store(cas)
        artifact_api.set_merkle_dag(dag)

        body = client.get(f"/api/artifacts/{obj.sha256}").json()

        related = body["related_merkle_nodes"]
        assert len(related) == 1
        assert related[0]["node_hash"] == node_hash


class TestPreviewArtifact:
    """Tests for GET /api/artifacts/{hash}/preview."""

    def test_returns_404_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/artifacts/abc/preview").status_code == 404

    def test_text_preview(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"plain text content")
        artifact_api.set_artifact_store(cas)

        body = client.get(f"/api/artifacts/{obj.sha256}/preview").json()

        assert body["type"] == "text"
        assert body["preview"] == "plain text content"

    def test_binary_preview_returns_hex(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
        artifact_api.set_artifact_store(cas)

        body = client.get(f"/api/artifacts/{obj.sha256}/preview").json()

        assert body["type"] == "binary"
        assert body["truncated"] is True

    def test_truncation_flag_for_long_text(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"x" * 100)
        artifact_api.set_artifact_store(cas)

        body = client.get(f"/api/artifacts/{obj.sha256}/preview?max_bytes=10").json()

        assert body["truncated"] is True


class TestStatsSummary:
    """Tests for GET /api/artifacts/stats/summary."""

    def test_returns_error_without_cas(self, client: TestClient) -> None:
        assert "error" in client.get("/api/artifacts/stats/summary").json()

    def test_aggregates_counts_and_sizes(self, client: TestClient, cas: ArtifactStore) -> None:
        cas.add_bytes(b"small")
        cas.add_bytes(b"y" * 2048)
        artifact_api.set_artifact_store(cas)

        body = client.get("/api/artifacts/stats/summary").json()

        assert body["total_objects"] == 2
        assert body["total_size_bytes"] == 5 + 2048
        assert body["size_distribution"]["<1KB"] == 1
        assert body["size_distribution"]["1KB-1MB"] == 1


class TestHtmlHelper:
    """Tests for the static HTML helper."""

    def test_browser_html_is_self_contained(self) -> None:
        html = artifact_api.get_artifact_browser_html()

        assert html.startswith("<!DOCTYPE html>")
        assert "Artifact Browser" in html
        assert html.strip().endswith("</html>")
