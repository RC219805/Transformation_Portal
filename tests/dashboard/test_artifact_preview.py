"""Behavioral coverage for ``dashboard.artifact_preview``.

Covers the pure ``detect_content_type`` (extension map, mimetypes fallback,
and magic-byte sniffing) plus the preview router (meta/raw/thumbnail/text)
driven via ``TestClient`` against a fake CAS whose objects are real
``tmp_path`` files. Thumbnail generation uses Pillow (a core dependency).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import artifact_preview
from transformation_portal.dashboard.artifact_preview import detect_content_type


class _FakeCAS:
    def __init__(self) -> None:
        self._paths: dict[str, Path] = {}

    def put(self, hash: str, path: Path) -> None:
        self._paths[hash] = path

    def get_object(self, hash: str):
        path = self._paths.get(hash)
        return SimpleNamespace(path=path) if path is not None else None


@pytest.fixture
def cas() -> _FakeCAS:
    return _FakeCAS()


@pytest.fixture
def client(cas: _FakeCAS):
    original = artifact_preview._global_cas
    artifact_preview.set_preview_cas(cas)  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(artifact_preview.create_preview_router())
    try:
        yield TestClient(app)
    finally:
        artifact_preview.set_preview_cas(original)  # type: ignore[arg-type]


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# detect_content_type
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.png", "image/png"),
        ("a.glb", "model/gltf-binary"),
        ("a.json", "application/json"),
        ("a.bin", "application/octet-stream"),
    ],
)
def test_detect_by_extension(name: str, expected: str) -> None:
    assert detect_content_type(Path(name)) == expected


def test_detect_by_mimetypes_fallback() -> None:
    # ".html" is not in MIME_EXTENSIONS, so mimetypes resolves it.
    assert detect_content_type(Path("page.html")) == "text/html"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBP", "image/webp"),
        (b"glTF\x02\x00", "model/gltf-binary"),
        (b'{"k":1}', "application/json"),
        (b"\x00\x01\x02\x03", "application/octet-stream"),
    ],
)
def test_detect_by_magic_bytes(data: bytes, expected: str) -> None:
    # Unknown extension forces the magic-byte path.
    assert detect_content_type(Path("blob.dat"), data) == expected


def test_detect_json_leading_but_undecodable_falls_back(tmp_path) -> None:
    assert detect_content_type(Path("blob.dat"), b"{\xff\xfe") == "application/octet-stream"


# --------------------------------------------------------------------------- #
# meta
# --------------------------------------------------------------------------- #


def test_meta_404_when_unknown(client: TestClient) -> None:
    assert client.get("/api/preview/artifact/ghost/meta").status_code == 404


def test_meta_404_when_file_missing(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    cas.put("h", tmp_path / "missing.png")
    assert client.get("/api/preview/artifact/h/meta").status_code == 404


def test_meta_for_image(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(_png_bytes())
    cas.put("img", f)
    body = client.get("/api/preview/artifact/img/meta").json()
    assert body["content_type"] == "image/png"
    assert body["is_image"] is True
    assert body["previewable"] is True
    assert body["is_3d"] is False


def test_meta_for_text(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    cas.put("txt", f)
    body = client.get("/api/preview/artifact/txt/meta").json()
    assert body["is_text"] is True


# --------------------------------------------------------------------------- #
# raw / thumbnail / text
# --------------------------------------------------------------------------- #


def test_raw_streams_file(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"payload")
    cas.put("raw", f)
    resp = client.get("/api/preview/artifact/raw/raw")
    assert resp.status_code == 200
    assert resp.content == b"payload"


def test_raw_404_when_unknown(client: TestClient) -> None:
    assert client.get("/api/preview/artifact/ghost/raw").status_code == 404


def test_thumbnail_success_for_image(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(_png_bytes())
    cas.put("img", f)
    resp = client.get("/api/preview/artifact/img/thumbnail", params={"size": 8})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_thumbnail_rejects_non_image(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    cas.put("txt", f)
    assert client.get("/api/preview/artifact/txt/thumbnail").status_code == 400


def test_thumbnail_corrupt_image_returns_500(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    # .png extension routes into the image branch, but the bytes are not a
    # valid image, so PIL raises -> 500.
    f = tmp_path / "broken.png"
    f.write_bytes(b"not really a png")
    cas.put("broken", f)
    assert client.get("/api/preview/artifact/broken/thumbnail").status_code == 500


def test_thumbnail_404_when_unknown(client: TestClient) -> None:
    assert client.get("/api/preview/artifact/ghost/thumbnail").status_code == 404


def test_text_success(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("line one")
    cas.put("txt", f)
    body = client.get("/api/preview/artifact/txt/text").json()
    assert body["content"] == "line one"
    assert body["truncated"] is False


def test_text_rejects_binary(client: TestClient, cas: _FakeCAS, tmp_path) -> None:
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\xff\xfe\x00\x01")
    cas.put("blob", f)
    assert client.get("/api/preview/artifact/blob/text").status_code == 400


def test_text_404_when_unknown(client: TestClient) -> None:
    assert client.get("/api/preview/artifact/ghost/text").status_code == 404


def test_preview_viewer_serves_html(client: TestClient) -> None:
    resp = client.get("/api/preview/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_preview, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        artifact_preview.create_preview_router()
