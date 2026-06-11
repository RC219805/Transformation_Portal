"""Behavioral coverage for ``dashboard.artifact_api``.

Drives the CAS artifact-browser router via ``TestClient`` against a fake
``ArtifactStore`` whose ``objects_dir`` is a real ``tmp_path`` tree, so the
filesystem-walking list/stats endpoints and the get/preview endpoints are
exercised offline. Also covers the ``_human_size`` helper directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import artifact_api


@dataclass
class _CASObject:
    sha256: str
    path: Path
    size_bytes: int


class _FakeCAS:
    def __init__(self, objects_dir: Path) -> None:
        self.objects_dir = objects_dir

    def _path_for(self, sha256: str) -> Path:
        return self.objects_dir / sha256[:2] / sha256

    def put(self, sha256: str, data: bytes) -> None:
        p = self._path_for(sha256)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_object(self, sha256: str) -> Optional[_CASObject]:
        p = self._path_for(sha256)
        if not p.exists():
            return None
        return _CASObject(sha256=sha256, path=p, size_bytes=p.stat().st_size)


class _MerkleNode:
    def __init__(self, node_type: str, outputs: dict) -> None:
        self.node_type = node_type
        self.outputs = outputs


class _FakeMerkle:
    def __init__(self) -> None:
        self.nodes: dict = {}


@pytest.fixture
def cas(tmp_path) -> _FakeCAS:
    return _FakeCAS(tmp_path / "objects")


@pytest.fixture
def restore_globals():
    orig_c, orig_m = artifact_api._global_cas, artifact_api._global_merkle_dag
    yield
    artifact_api.set_artifact_store(orig_c)  # type: ignore[arg-type]
    artifact_api.set_merkle_dag(orig_m)  # type: ignore[arg-type]


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(artifact_api.create_artifact_router())
    return TestClient(app)


# --------------------------------------------------------------------------- #
# _human_size helper
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("size", "expected"),
    [(512, "512.0 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB")],
)
def test_human_size(size: int, expected: str) -> None:
    assert artifact_api._human_size(size) == expected


def test_human_size_petabyte_overflow() -> None:
    assert artifact_api._human_size(1024**5).endswith("PB")


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #


def test_list_without_cas_returns_error(restore_globals) -> None:
    artifact_api.set_artifact_store(None)  # type: ignore[arg-type]
    body = _client().get("/api/artifacts/").json()
    assert body["artifacts"] == [] and body["total"] == 0 and "error" in body


def test_list_artifacts_with_prefix_and_pagination(restore_globals, cas: _FakeCAS) -> None:
    cas.put("aa" + "0" * 62, b"x")
    cas.put("aa" + "1" * 62, b"yy")
    cas.put("bb" + "2" * 62, b"zzz")
    artifact_api.set_artifact_store(cas)  # type: ignore[arg-type]
    client = _client()

    # Prefix filter keeps only the "aa..." objects.
    body = client.get("/api/artifacts/", params={"prefix": "aa"}).json()
    assert body["total"] == 2
    assert all(a["hash"].startswith("aa") for a in body["artifacts"])

    # Pagination: limit caps the returned slice, total still counts all.
    paged = client.get("/api/artifacts/", params={"prefix": "aa", "limit": 1}).json()
    assert paged["total"] == 2
    assert len(paged["artifacts"]) == 1


# --------------------------------------------------------------------------- #
# get
# --------------------------------------------------------------------------- #


def test_get_artifact_404_without_cas(restore_globals) -> None:
    artifact_api.set_artifact_store(None)  # type: ignore[arg-type]
    assert _client().get("/api/artifacts/somehash").status_code == 404


def test_get_artifact_404_when_missing(restore_globals, cas: _FakeCAS) -> None:
    artifact_api.set_artifact_store(cas)  # type: ignore[arg-type]
    assert _client().get("/api/artifacts/absent").status_code == 404


def test_get_artifact_with_related_merkle_nodes(restore_globals, cas: _FakeCAS) -> None:
    cas.put("h" * 64, b"data")
    merkle = _FakeMerkle()
    merkle.nodes["mn1"] = _MerkleNode("compute", {"content_hash": "h" * 64})
    merkle.nodes["mn2"] = _MerkleNode("compute", {"content_hash": "other"})
    artifact_api.set_artifact_store(cas)  # type: ignore[arg-type]
    artifact_api.set_merkle_dag(merkle)  # type: ignore[arg-type]

    body = _client().get(f"/api/artifacts/{'h' * 64}").json()
    assert body["size_bytes"] == 4
    assert [n["node_hash"] for n in body["related_merkle_nodes"]] == ["mn1"]


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #


def test_preview_text(restore_globals, cas: _FakeCAS) -> None:
    cas.put("t" * 64, b"hello world")
    artifact_api.set_artifact_store(cas)  # type: ignore[arg-type]
    body = _client().get(f"/api/artifacts/{'t' * 64}/preview").json()
    assert body["type"] == "text"
    assert body["preview"] == "hello world"


def test_preview_binary_returns_hex(restore_globals, cas: _FakeCAS) -> None:
    cas.put("b" * 64, b"\x00\x01\xff\xfe")
    artifact_api.set_artifact_store(cas)  # type: ignore[arg-type]
    body = _client().get(f"/api/artifacts/{'b' * 64}/preview").json()
    assert body["type"] == "binary"
    assert body["preview"] == "0001fffe"


def test_preview_404s_and_500(restore_globals, cas: _FakeCAS, tmp_path) -> None:
    artifact_api.set_artifact_store(None)  # type: ignore[arg-type]
    assert _client().get("/api/artifacts/x/preview").status_code == 404  # no CAS

    artifact_api.set_artifact_store(cas)  # type: ignore[arg-type]
    assert _client().get("/api/artifacts/absent/preview").status_code == 404  # unknown

    # Object known to CAS but the file vanished -> open() raises -> 500. We
    # build a CAS whose get_object returns an object for a non-existent path.
    class _GhostCAS(_FakeCAS):
        def get_object(self, sha256: str):
            return _CASObject(sha256=sha256, path=tmp_path / "gone.bin", size_bytes=10)

    artifact_api.set_artifact_store(_GhostCAS(tmp_path / "objects"))  # type: ignore[arg-type]
    assert _client().get("/api/artifacts/ghost/preview").status_code == 500


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def test_stats_without_cas_returns_error(restore_globals) -> None:
    artifact_api.set_artifact_store(None)  # type: ignore[arg-type]
    assert "error" in _client().get("/api/artifacts/stats/summary").json()


def test_stats_size_distribution(restore_globals, cas: _FakeCAS) -> None:
    cas.put("aa" + "0" * 62, b"x")  # <1KB
    cas.put("bb" + "1" * 62, b"y" * 2048)  # 1KB-1MB
    artifact_api.set_artifact_store(cas)  # type: ignore[arg-type]

    body = _client().get("/api/artifacts/stats/summary").json()
    assert body["total_objects"] == 2
    assert body["size_distribution"]["<1KB"] == 1
    assert body["size_distribution"]["1KB-1MB"] == 1


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_api, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        artifact_api.create_artifact_router()
