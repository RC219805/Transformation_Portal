"""Behavioral coverage for ``dashboard.studio_inspector``.

``studio_inspector.py`` is almost entirely HTML-string generation behind a
two-route FastAPI router. These tests pin the router wiring, the
FastAPI-availability guard, and structural contracts on the generated markup
(well-formed document shell, the WebGL canvas/script the UI depends on) so an
accidental truncation or corruption of the large HTML blobs is caught. No app
bootstrap, network, or ML — just ``TestClient`` over the isolated router.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import studio_inspector
from transformation_portal.dashboard.studio_inspector import (
    create_studio_inspector_router,
    get_comparison_view_html,
    get_studio_inspector_html,
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(create_studio_inspector_router())
    return TestClient(app)


@pytest.mark.parametrize("path", ["/api/studio/", "/api/studio/compare"])
def test_router_serves_html_routes(client: TestClient, path: str) -> None:
    resp = client.get(path)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<!DOCTYPE html>" in resp.text or "<html" in resp.text.lower()


def test_studio_inspector_html_is_well_formed_document() -> None:
    html = get_studio_inspector_html()
    assert isinstance(html, str) and html.strip()
    lowered = html.lower()
    assert "<html" in lowered and "</html>" in lowered
    # The inspector is a WebGL/3D surface built on Three.js.
    assert "three" in lowered
    assert "renderer" in lowered
    assert "<script" in lowered


def test_comparison_view_html_is_well_formed_document() -> None:
    html = get_comparison_view_html()
    assert isinstance(html, str) and html.strip()
    lowered = html.lower()
    assert "<html" in lowered and "</html>" in lowered
    assert "<script" in lowered


def test_studio_and_comparison_html_are_distinct() -> None:
    # Guards against the two routes accidentally returning the same blob.
    assert get_studio_inspector_html() != get_comparison_view_html()


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio_inspector, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        create_studio_inspector_router()
