"""Unit tests for the dashboard artifact preview API.

Covers the detect_content_type helper (extension / mimetypes / magic-byte
paths), the preview router endpoints backed by a real CAS, and the static
HTML helper.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import artifact_preview
from transformation_portal.dashboard.artifact_preview import detect_content_type
from transformation_portal.storage.cas_store import ArtifactStore

pytestmark = pytest.mark.unit

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _reset_cas() -> Generator[None, None, None]:
    artifact_preview.set_preview_cas(None)
    yield
    artifact_preview.set_preview_cas(None)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(artifact_preview.create_preview_router())
    return TestClient(app)


@pytest.fixture
def cas(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "cas")


def _png_bytes() -> bytes:
    """A tiny valid PNG, encoded via Pillow."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestDetectContentType:
    """Tests for the detect_content_type helper."""

    def test_extension_match_wins(self) -> None:
        assert detect_content_type(Path("model.glb")) == "model/gltf-binary"
        assert detect_content_type(Path("data.json")) == "application/json"
        assert detect_content_type(Path("notes.md")) == "text/markdown"

    def test_mimetypes_fallback(self) -> None:
        # .html is not in MIME_EXTENSIONS but mimetypes knows it.
        assert detect_content_type(Path("page.html")) == "text/html"

    def test_magic_bytes_png(self) -> None:
        assert detect_content_type(Path("noext"), _PNG_MAGIC) == "image/png"

    def test_magic_bytes_jpeg(self) -> None:
        assert detect_content_type(Path("noext"), b"\xff\xd8\xff\xe0") == "image/jpeg"

    def test_magic_bytes_gif(self) -> None:
        assert detect_content_type(Path("noext"), b"GIF89a....") == "image/gif"

    def test_magic_bytes_webp(self) -> None:
        assert detect_content_type(Path("noext"), b"RIFF1234WEBP....") == "image/webp"

    def test_magic_bytes_glb(self) -> None:
        assert detect_content_type(Path("noext"), b"glTF....") == "model/gltf-binary"

    def test_magic_bytes_json(self) -> None:
        assert detect_content_type(Path("noext"), b'{"a": 1}') == "application/json"

    def test_unknown_defaults_to_octet_stream(self) -> None:
        assert detect_content_type(Path("noext"), b"\x00\x01\x02\x03") == "application/octet-stream"

    def test_unknown_without_data_defaults_to_octet_stream(self) -> None:
        assert detect_content_type(Path("noext")) == "application/octet-stream"


class TestMetaEndpoint:
    """Tests for GET /api/preview/artifact/{hash}/meta."""

    def test_returns_404_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/preview/artifact/abc/meta").status_code == 404

    def test_returns_404_for_unknown_hash(self, client: TestClient, cas: ArtifactStore) -> None:
        artifact_preview.set_preview_cas(cas)

        assert client.get("/api/preview/artifact/deadbeef/meta").status_code == 404

    @pytest.mark.parametrize("operation", ["meta", "raw", "thumbnail", "text"])
    def test_returns_404_for_invalid_hash(
        self,
        client: TestClient,
        cas: ArtifactStore,
        operation: str,
    ) -> None:
        artifact_preview.set_preview_cas(cas)

        assert client.get(f"/api/preview/artifact/{'g' * 64}/{operation}").status_code == 404

    def test_returns_image_metadata(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(_png_bytes())
        artifact_preview.set_preview_cas(cas)

        body = client.get(f"/api/preview/artifact/{obj.sha256}/meta").json()

        assert body["hash"] == obj.sha256
        assert body["content_type"] == "image/png"
        assert body["is_image"] is True
        assert body["is_3d"] is False
        assert body["previewable"] is True

    def test_returns_json_metadata(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b'{"key": "value"}')
        artifact_preview.set_preview_cas(cas)

        body = client.get(f"/api/preview/artifact/{obj.sha256}/meta").json()

        assert body["content_type"] == "application/json"
        assert body["is_text"] is True
        assert body["previewable"] is True


class TestRawEndpoint:
    """Tests for GET /api/preview/artifact/{hash}/raw."""

    def test_returns_404_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/preview/artifact/abc/raw").status_code == 404

    def test_streams_raw_content(self, client: TestClient, cas: ArtifactStore) -> None:
        payload = _png_bytes()
        obj = cas.add_bytes(payload)
        artifact_preview.set_preview_cas(cas)

        response = client.get(f"/api/preview/artifact/{obj.sha256}/raw")

        assert response.status_code == 200
        assert response.content == payload
        assert response.headers["content-type"] == "image/png"


class TestThumbnailEndpoint:
    """Tests for GET /api/preview/artifact/{hash}/thumbnail."""

    def test_returns_404_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/preview/artifact/abc/thumbnail").status_code == 404

    def test_rejects_non_image_with_400(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b'{"not": "an image"}')
        artifact_preview.set_preview_cas(cas)

        assert client.get(f"/api/preview/artifact/{obj.sha256}/thumbnail").status_code == 400

    def test_generates_png_thumbnail_for_image(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(_png_bytes())
        artifact_preview.set_preview_cas(cas)

        response = client.get(f"/api/preview/artifact/{obj.sha256}/thumbnail?size=4")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == _PNG_MAGIC


class TestTextEndpoint:
    """Tests for GET /api/preview/artifact/{hash}/text."""

    def test_returns_404_without_cas(self, client: TestClient) -> None:
        assert client.get("/api/preview/artifact/abc/text").status_code == 404

    def test_returns_text_content(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"hello text content")
        artifact_preview.set_preview_cas(cas)

        body = client.get(f"/api/preview/artifact/{obj.sha256}/text").json()

        assert body["content"] == "hello text content"
        assert body["truncated"] is False

    def test_truncation_flag(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"x" * 50)
        artifact_preview.set_preview_cas(cas)

        body = client.get(f"/api/preview/artifact/{obj.sha256}/text?max_chars=10").json()

        assert body["truncated"] is True
        assert len(body["content"]) == 10

    def test_rejects_non_utf8_with_400(self, client: TestClient, cas: ArtifactStore) -> None:
        obj = cas.add_bytes(b"\xff\xfe\x00\x01\x80")
        artifact_preview.set_preview_cas(cas)

        assert client.get(f"/api/preview/artifact/{obj.sha256}/text").status_code == 400


class TestViewerRouteAndHelper:
    """Tests for the served HTML route and helper."""

    def test_viewer_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/preview/")

        assert response.status_code == 200
        assert "Artifact Preview" in response.text

    def test_viewer_html_is_self_contained(self) -> None:
        html = artifact_preview.get_preview_viewer_html()

        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")
