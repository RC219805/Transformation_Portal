"""Unit tests for the dashboard studio 3D inspector API.

studio_inspector.py is almost entirely static HTML; the Python surface is
the router factory plus two HTML helpers, all covered here.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import studio_inspector

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
