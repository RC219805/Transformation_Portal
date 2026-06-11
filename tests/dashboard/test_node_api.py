"""Behavioral coverage for ``dashboard.node_api``.

The node-inspection router exposes per-node detail/inputs/outputs/artifacts/logs,
run summaries, and CAS-backed artifact info/preview/download. These tests drive
it via ``TestClient`` against a real in-memory ``NodeStateStore`` (injected with
``set_store``) and a fake ``ArtifactStore`` (injected with ``set_cas``), so every
branch is exercised offline:

- 404s when a node/run is absent on each detail endpoint
- artifact listing with and without CAS metadata
- log ``tail`` truncation
- artifact info/preview/download: CAS-not-configured 503, missing-object 404,
  missing-file 404, and the content-type sniffing matrix (PNG/JPEG/WEBP/JSON/
  text/binary-hex) plus the read-failure 500 path
- the HTML UI route and the FastAPI-availability guard
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import node_api
from transformation_portal.dashboard.node_state_store import NodeStateStore, set_store


@dataclass
class _FakeCASObject:
    sha256: str
    path: Path
    size_bytes: int


class _FakeCAS:
    """Minimal ArtifactStore stand-in exposing ``get_object``."""

    def __init__(self) -> None:
        self._objs: dict[str, _FakeCASObject] = {}

    def add(self, sha256: str, path: Path, *, size_bytes: Optional[int] = None) -> None:
        size = size_bytes if size_bytes is not None else (path.stat().st_size if path.exists() else 0)
        self._objs[sha256] = _FakeCASObject(sha256=sha256, path=path, size_bytes=size)

    def get_object(self, sha256: str) -> Optional[_FakeCASObject]:
        return self._objs.get(sha256)


@pytest.fixture
def populated_store() -> NodeStateStore:
    store = NodeStateStore()
    store.init_run("run_1", ["ingest"])
    store.set_status("run_1", "ingest", "complete")
    store.update_inputs("run_1", "ingest", {"image": "in.png"})
    store.update_outputs("run_1", "ingest", {"rgb": [1, 2, 3]})
    store.add_log("run_1", "ingest", "loaded")
    return store


@pytest.fixture
def client(populated_store: NodeStateStore):
    """TestClient wired to a populated store; CAS reset to None per test."""
    original_store = node_api.get_store()
    original_cas = node_api._global_cas
    set_store(populated_store)
    node_api.set_cas(None)  # type: ignore[arg-type]

    app = FastAPI()
    app.include_router(node_api.create_node_inspection_router())
    try:
        yield TestClient(app)
    finally:
        set_store(original_store)
        node_api.set_cas(original_cas)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# node detail endpoints
# --------------------------------------------------------------------------- #


def test_node_details_success(client: TestClient) -> None:
    resp = client.get("/api/inspect/runs/run_1/nodes/ingest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["node_id"] == "ingest"
    assert body["status"] == "complete"
    assert body["inputs"] == {"image": "in.png"}


@pytest.mark.parametrize(
    "suffix",
    ["", "/inputs", "/outputs", "/artifacts", "/logs"],
)
def test_node_endpoints_404_for_unknown_node(client: TestClient, suffix: str) -> None:
    resp = client.get(f"/api/inspect/runs/run_1/nodes/ghost{suffix}")
    assert resp.status_code == 404


def test_node_inputs_and_outputs(client: TestClient) -> None:
    inputs = client.get("/api/inspect/runs/run_1/nodes/ingest/inputs").json()
    assert inputs == {"node_id": "ingest", "inputs": {"image": "in.png"}}
    outputs = client.get("/api/inspect/runs/run_1/nodes/ingest/outputs").json()
    assert outputs == {"node_id": "ingest", "outputs": {"rgb": [1, 2, 3]}}


def test_node_logs_respects_tail(client: TestClient) -> None:
    store = node_api.get_store()
    for i in range(5):
        store.add_log("run_1", "ingest", f"line-{i}")
    body = client.get("/api/inspect/runs/run_1/nodes/ingest/logs", params={"tail": 2}).json()
    assert body["total_logs"] == 6  # 1 from fixture + 5
    assert len(body["logs"]) == 2
    assert body["logs"][-1].endswith("line-4")


# --------------------------------------------------------------------------- #
# artifacts (with / without CAS)
# --------------------------------------------------------------------------- #


def test_artifacts_without_cas_returns_bare_entries(client: TestClient) -> None:
    node_api.get_store().add_artifact("run_1", "ingest", "depth", "sha-d")
    body = client.get("/api/inspect/runs/run_1/nodes/ingest/artifacts").json()
    assert body["artifacts"] == [{"name": "depth", "hash": "sha-d"}]


def test_artifacts_with_cas_but_unknown_hash_stays_bare(client: TestClient) -> None:
    # CAS is configured, but the artifact hash is not in it -> get_object None,
    # so the entry is not enriched (exercises the `if obj:` false branch).
    node_api.set_cas(_FakeCAS())  # type: ignore[arg-type]
    node_api.get_store().add_artifact("run_1", "ingest", "depth", "sha-unknown")
    body = client.get("/api/inspect/runs/run_1/nodes/ingest/artifacts").json()
    assert body["artifacts"] == [{"name": "depth", "hash": "sha-unknown"}]


def test_artifacts_with_cas_enriches_metadata(client: TestClient, tmp_path: Path) -> None:
    artifact_file = tmp_path / "depth.bin"
    artifact_file.write_bytes(b"xyz")
    cas = _FakeCAS()
    cas.add("sha-d", artifact_file)
    node_api.set_cas(cas)  # type: ignore[arg-type]
    node_api.get_store().add_artifact("run_1", "ingest", "depth", "sha-d")

    entry = client.get("/api/inspect/runs/run_1/nodes/ingest/artifacts").json()["artifacts"][0]
    assert entry["name"] == "depth"
    assert entry["size_bytes"] == 3
    assert entry["exists"] is True
    assert entry["path"].endswith("depth.bin")


# --------------------------------------------------------------------------- #
# run summary
# --------------------------------------------------------------------------- #


def test_run_summary_success(client: TestClient) -> None:
    body = client.get("/api/inspect/runs/run_1/summary").json()
    assert body["run_id"] == "run_1"
    node = body["nodes"]["ingest"]
    assert node["has_inputs"] is True
    assert node["has_outputs"] is True
    assert node["log_count"] >= 1


def test_run_summary_404_for_unknown_run(client: TestClient) -> None:
    assert client.get("/api/inspect/runs/ghost/summary").status_code == 404


# --------------------------------------------------------------------------- #
# artifact info / preview / download
# --------------------------------------------------------------------------- #


def test_artifact_info_503_without_cas(client: TestClient) -> None:
    resp = client.get("/api/inspect/artifact/sha-x")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "CAS not configured"


def test_artifact_info_404_when_missing(client: TestClient) -> None:
    node_api.set_cas(_FakeCAS())  # type: ignore[arg-type]
    assert client.get("/api/inspect/artifact/absent").status_code == 404


def test_artifact_info_success(client: TestClient, tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    cas = _FakeCAS()
    cas.add("sha-a", f)
    node_api.set_cas(cas)  # type: ignore[arg-type]

    body = client.get("/api/inspect/artifact/sha-a").json()
    assert body["hash"] == "sha-a"
    assert body["size_bytes"] == 5
    assert body["exists"] is True


@pytest.mark.parametrize(
    ("data", "expected_type"),
    [
        (b"\x89PNG\r\n\x1a\n" + b"rest", "image/png"),
        (b"\xff\xd8\xff\xe0" + b"jfif", "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBPxxxx", "image/webp"),
        (b'{"k": 1}', "application/json"),
        (b"plain text body", "text/plain"),
        (b"\x00\x01\x02\xff\xfe", "binary"),
    ],
)
def test_artifact_preview_content_type_matrix(
    client: TestClient, tmp_path: Path, data: bytes, expected_type: str
) -> None:
    f = tmp_path / "blob"
    f.write_bytes(data)
    cas = _FakeCAS()
    cas.add("sha-blob", f)
    node_api.set_cas(cas)  # type: ignore[arg-type]

    body = client.get("/api/inspect/artifact/sha-blob/preview").json()
    assert body["content_type"] == expected_type
    assert body["truncated"] is False


def test_artifact_preview_json_leading_but_undecodable_falls_back_to_binary(
    client: TestClient, tmp_path: Path
) -> None:
    # Leading "{" routes into the JSON branch, but invalid UTF-8 makes the
    # decode raise -> the except: pass leaves it as binary with no preview.
    f = tmp_path / "bad.json"
    f.write_bytes(b"{\xff\xfe\x00")
    cas = _FakeCAS()
    cas.add("sha-badjson", f)
    node_api.set_cas(cas)  # type: ignore[arg-type]

    body = client.get("/api/inspect/artifact/sha-badjson/preview").json()
    assert body["content_type"] == "binary"
    assert body["preview"] is None


def test_artifact_preview_truncated_flag(client: TestClient, tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_bytes(b"a" * 100)
    cas = _FakeCAS()
    cas.add("sha-big", f)
    node_api.set_cas(cas)  # type: ignore[arg-type]

    body = client.get("/api/inspect/artifact/sha-big/preview", params={"max_bytes": 10}).json()
    assert body["truncated"] is True


def test_artifact_preview_503_and_404(client: TestClient) -> None:
    assert client.get("/api/inspect/artifact/x/preview").status_code == 503  # no CAS
    node_api.set_cas(_FakeCAS())  # type: ignore[arg-type]
    assert client.get("/api/inspect/artifact/absent/preview").status_code == 404


def test_artifact_preview_read_failure_returns_500(client: TestClient, tmp_path: Path) -> None:
    cas = _FakeCAS()
    # Object exists in CAS, but the backing file does not -> open() raises.
    cas.add("sha-missing", tmp_path / "does_not_exist.bin", size_bytes=10)
    node_api.set_cas(cas)  # type: ignore[arg-type]
    assert client.get("/api/inspect/artifact/sha-missing/preview").status_code == 500


def test_artifact_download_success(client: TestClient, tmp_path: Path) -> None:
    f = tmp_path / "dl.bin"
    f.write_bytes(b"payload")
    cas = _FakeCAS()
    cas.add("sha-dl", f)
    node_api.set_cas(cas)  # type: ignore[arg-type]

    resp = client.get("/api/inspect/artifact/sha-dl/download")
    assert resp.status_code == 200
    assert resp.content == b"payload"


def test_artifact_download_503_404_and_missing_file(client: TestClient, tmp_path: Path) -> None:
    assert client.get("/api/inspect/artifact/x/download").status_code == 503  # no CAS

    cas = _FakeCAS()
    node_api.set_cas(cas)  # type: ignore[arg-type]
    assert client.get("/api/inspect/artifact/absent/download").status_code == 404  # unknown hash

    cas.add("sha-gone", tmp_path / "gone.bin", size_bytes=5)  # object known, file absent
    assert client.get("/api/inspect/artifact/sha-gone/download").status_code == 404


# --------------------------------------------------------------------------- #
# UI route + guards
# --------------------------------------------------------------------------- #


def test_inspection_ui_serves_html(client: TestClient) -> None:
    resp = client.get("/api/inspect/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<html" in resp.text.lower()


def test_set_cas_mutates_global() -> None:
    original = node_api._global_cas
    sentinel = _FakeCAS()
    try:
        node_api.set_cas(sentinel)  # type: ignore[arg-type]
        assert node_api._global_cas is sentinel
    finally:
        node_api.set_cas(original)  # type: ignore[arg-type]


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_api, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        node_api.create_node_inspection_router()
