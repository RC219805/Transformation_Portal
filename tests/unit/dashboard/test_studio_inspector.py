"""Unit tests for the dashboard studio 3D inspector API.

studio_inspector.py is almost entirely static HTML; the Python surface is
the router factory plus two HTML helpers, all covered here.
"""

from __future__ import annotations

import re
from typing import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import artifact_preview, studio_inspector, time_travel

pytestmark = pytest.mark.unit


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(studio_inspector.create_studio_inspector_router())
    return TestClient(app)


class TestRouterFactory:
    """Tests for create_studio_inspector_router."""

    def test_router_has_studio_prefix(self) -> None:
        router = studio_inspector.create_studio_inspector_router()

        assert router.prefix == "/api/studio"

    def test_factory_raises_without_fastapi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(studio_inspector, "FASTAPI_AVAILABLE", False)

        with pytest.raises(ImportError, match="FastAPI is required"):
            studio_inspector.create_studio_inspector_router()


class TestRoutes:
    """Tests for the served HTML routes."""

    def test_inspector_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/studio/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Studio 3D Inspector" in response.text

    def test_comparison_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/studio/compare")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Model Comparison" in response.text


class TestHtmlHelpers:
    """Tests for the static HTML helpers."""

    def test_inspector_html_is_self_contained_document(self) -> None:
        html = studio_inspector.get_studio_inspector_html()

        assert html.startswith("<!DOCTYPE html>")
        assert "<title>Studio 3D Inspector</title>" in html
        assert html.strip().endswith("</html>")

    def test_comparison_html_is_self_contained_document(self) -> None:
        html = studio_inspector.get_comparison_view_html()

        assert html.startswith("<!DOCTYPE html>")
        assert "<title>Model Comparison</title>" in html
        assert html.strip().endswith("</html>")


# three r147 is the last npm release that still ships the UMD add-ons under
# examples/js/; r148 removed that directory (only examples/jsm/ remains), so
# add-on <script> tags for any later version 404 and THREE.OrbitControls etc.
# are undefined. Verified against the published npm tarballs, not in a browser.
_LAST_THREE_MINOR_WITH_EXAMPLES_JS = 147
_THREE_NPM_URL = re.compile(r"https://cdn\.jsdelivr\.net/npm/three@(\d+)\.(\d+)\.(\d+)/([^\"'\s]+)")

_THREE_PAGES: dict[str, Callable[[], str]] = {
    "studio_inspector": studio_inspector.get_studio_inspector_html,
    "studio_comparison": studio_inspector.get_comparison_view_html,
    "artifact_preview": artifact_preview.get_preview_viewer_html,
    "time_travel_diff": time_travel.get_diff_viewer_html,
}


class TestThreeJsPinning:
    """Regression for D9: core and add-on three.js URLs must resolve together."""

    @pytest.mark.parametrize("page", sorted(_THREE_PAGES))
    def test_core_and_addons_share_one_version_that_ships_examples_js(self, page: str) -> None:
        html = _THREE_PAGES[page]()
        matches = _THREE_NPM_URL.findall(html)
        paths = [m[3] for m in matches]

        assert "build/three.min.js" in paths
        addons = [p for p in paths if p.startswith("examples/")]
        assert addons, "page is expected to load at least one three.js add-on"
        assert all(p.startswith("examples/js/") for p in addons), addons

        versions = {(int(a), int(b), int(c)) for a, b, c, _ in matches}
        assert len(versions) == 1, versions
        major, minor, _patch = versions.pop()
        assert major == 0
        assert minor <= _LAST_THREE_MINOR_WITH_EXAMPLES_JS
