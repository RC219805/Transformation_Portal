"""Documentation reader, flag, escaping and strict-CSP compatibility regressions."""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_oauth2_redirect_html
from fastapi.testclient import TestClient
from starlette.responses import Response

from transformation_portal.api import docs

pytestmark = pytest.mark.unit


def _app(*, root_path: str = "") -> FastAPI:
    application = FastAPI(title="Example API", docs_url=None, redoc_url=None, root_path=root_path)

    @application.post("/example/{name}")
    async def example(name: str, payload: dict[str, str]) -> dict[str, str]:
        """An example operation description."""
        return {"name": name, **payload}

    return application


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_readers_present_live_operations_and_schema_definitions_without_inline_code(path):
    application = _app()
    application.include_router(docs.create_api_docs_router(application))
    with TestClient(application) as client:
        response = client.get(path)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert 'data-ui="api-docs-operation"' in response.text
    assert "POST" in response.text
    assert "/example/{name}" in response.text
    assert "An example operation description." in response.text
    assert "Parameters" in response.text
    assert "Request body" in response.text
    assert "application/json" in response.text
    assert "Responses" in response.text
    assert "HTTPValidationError" in response.text
    assert 'href="/openapi.json"' in response.text
    assert 'href="/docs/assets/api-docs.css"' in response.text
    assert "<details" in response.text
    assert "<script" not in response.text
    assert "<style" not in response.text
    assert "https://" not in response.text


def test_docs_routes_do_not_change_openapi_bytes_or_envelopes():
    application = _app()
    with TestClient(application) as client:
        before = client.get("/openapi.json").content
        application.include_router(docs.create_api_docs_router(application))
        application.openapi_schema = None
        after = client.get("/openapi.json").content
    assert after == before
    paths = json.loads(after)["paths"]
    assert list(paths) == ["/example/{name}"]


@pytest.mark.parametrize(
    ("name", "path"),
    [("swagger_ui_html", "/docs"), ("redoc_html", "/redoc"), ("swagger_ui_redirect", "/docs/oauth2-redirect")],
)
def test_docs_preserve_fastapi_route_names_for_reverse_lookup(name, path):
    application = _app()
    application.include_router(docs.create_api_docs_router(application))
    assert str(application.url_path_for(name)) == path


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_existing_docs_routes_preserve_head_with_empty_body_and_get_headers(path):
    application = _app()
    application.include_router(docs.create_api_docs_router(application))
    with TestClient(application) as client:
        head = client.head(path)
        get = client.get(path)
    assert head.status_code == get.status_code == 200
    assert head.content == b""
    assert head.headers["content-type"] == get.headers["content-type"]
    assert head.headers["content-length"] == get.headers["content-length"]
    assert head.headers["cache-control"] == get.headers["cache-control"] == "no-store"


def test_docs_links_respect_proxy_root_path_and_stylesheet_is_packaged():
    application = _app(root_path="/backend")
    application.include_router(docs.create_api_docs_router(application))
    with TestClient(application) as client:
        response = client.get("/docs")
        css = client.get("/docs/assets/api-docs.css")
        oauth = client.get("/docs/oauth2-redirect")
    assert 'href="/backend/openapi.json"' in response.text
    assert 'href="/backend/docs/assets/api-docs.css"' in response.text
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert b"summary:focus-visible" in css.content
    assert 'src="/backend/docs/assets/oauth2-redirect.js"' in oauth.text


@pytest.mark.parametrize("path", ["/docs/assets/unknown.css", "/docs/assets/docs.py", "/docs/assets/%2e%2e/docs.py"])
def test_docs_assets_have_only_exact_allowlisted_routes(path):
    application = _app()
    application.include_router(docs.create_api_docs_router(application))
    with TestClient(application) as client:
        assert client.get(path).status_code == 404


def test_openapi_derived_html_is_escaped_and_schema_anchors_handle_special_names():
    malicious = '<script>alert("escaped")</script>'
    name = 'schema / ~ " <name>'
    pointer = name.replace("~", "~0").replace("/", "~1")
    schema = {
        "paths": {
            malicious: {
                "parameters": [{"name": malicious}],
                "post": {
                    "summary": malicious,
                    "description": malicious,
                    "requestBody": {"$ref": "#/components/schemas/" + pointer},
                    "responses": {"200": {"description": malicious}},
                },
            }
        },
        "components": {"schemas": {name: {"description": malicious}}},
    }
    rendered = docs._render_docs(schema, title=malicious, openapi_url='/schema?"<', stylesheet_url='/styles?"<')
    assert malicious not in rendered
    assert html.escape(malicious) in rendered
    assert "<script" not in rendered
    assert 'href="/schema?&quot;&lt;"' in rendered
    assert 'href="/styles?&quot;&lt;"' in rendered
    schema_id = "schema-" + name.encode("utf-8").hex()
    assert f'id="{schema_id}"' in rendered
    assert f'href="#{schema_id}"' in rendered
    assert "Path-level parameters" in rendered


@pytest.mark.parametrize("opening_tag, closing_tag", [("script", "script"), ("SCRIPT", "SCRIPT"), ("ScRiPt", "sCrIpT")])
def test_oauth_compatibility_uses_exact_trusted_script_as_external_asset(monkeypatch, opening_tag, closing_tag):
    original = get_swagger_ui_oauth2_redirect_html().body.decode("utf-8")
    document = original.replace("<script>", f"<{opening_tag}>").replace("</script>", f"</{closing_tag}>")
    monkeypatch.setattr(docs, "get_swagger_ui_oauth2_redirect_html", lambda: Response(document))
    application = _app()
    application.include_router(docs.create_api_docs_router(application))
    with TestClient(application) as client:
        redirect = client.get("/docs/oauth2-redirect")
        script = client.get("/docs/assets/oauth2-redirect.js")
    original_script = re.search(r"<script>(.*?)</script>", document, re.DOTALL | re.IGNORECASE).group(1)
    assert script.content == original_script.encode("utf-8")
    assert script.headers["content-type"].startswith("application/javascript")
    assert redirect.status_code == 200
    assert 'src="/docs/assets/oauth2-redirect.js"' in redirect.text
    assert f"<{opening_tag}>" not in redirect.text
    assert "<style" not in redirect.text
    assert "window.opener" not in redirect.text


@pytest.mark.parametrize(
    "document",
    [
        "<html>No script</html>",
        "<script></script>",
        "<SCRIPT></SCRIPT>",
        "<ScRiPt>run()</sCrIpT >",
        '<ScRiPt>run()</sCrIpT foo="bar">',
        "<ScRiPt>run()</sCrIpT",
        "<script>run()</script foo><SCRIPT>more()</SCRIPT>",
        "<script>one()</script><script>two()</script>",
        "<SCRIPT>one()</SCRIPT><sCrIpT>two()</sCrIpT>",
        '<script src="external.js"></script>',
        '<SCRIPT SRC="external.js"></SCRIPT>',
        '<script>run()</script><SCRIPT SRC="external.js"></SCRIPT>',
        "<style>body{}</style><script>run()</script>",
        "<STYLE>body{}</STYLE><ScRiPt>run()</sCrIpT>",
        '<body onload="run()"><script>run()</script>',
        '<body oNlOaD="run()"><ScRiPt>run()</sCrIpT>',
        '<body StYlE="color:red"><SCRIPT>run()</SCRIPT>',
    ],
)
def test_oauth_helper_shape_drift_fails_closed(monkeypatch, document):
    monkeypatch.setattr(docs, "get_swagger_ui_oauth2_redirect_html", lambda: Response(document))
    with pytest.raises(RuntimeError, match="FastAPI OAuth redirect"):
        docs.create_api_docs_router(_app())


def test_docs_router_requires_an_enabled_schema():
    with pytest.raises(ValueError, match="enabled OpenAPI"):
        docs.create_api_docs_router(FastAPI(openapi_url=None))


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_root_app_docs_flag_and_strict_csp_in_fresh_process(enabled):
    repo_root = Path(__file__).resolve().parents[2]
    probe = """
import app
from fastapi.testclient import TestClient
with TestClient(app.app) as client:
    for path in ('/docs', '/redoc', '/docs/oauth2-redirect', '/openapi.json', '/docs/assets/api-docs.css', '/docs/assets/oauth2-redirect.js'):
        methods = ('GET', 'HEAD') if path in ('/docs', '/redoc', '/docs/oauth2-redirect') else ('GET',)
        for method in methods:
            response = client.request(method, path)
            assert response.status_code == EXPECTED_STATUS, (method, path, response.status_code)
            assert response.headers['content-security-policy'] == app.DEFAULT_CSP
            assert response.headers['x-content-type-options'] == 'nosniff'
            if method == 'HEAD':
                assert response.content == b''
""".replace("EXPECTED_STATUS", "200" if enabled == "1" else "404")
    environment = {
        **os.environ,
        "TP_ENABLE_API_DOCS": enabled,
        "TP_CSP": "",
        "TP_ORCHESTRATOR_STATE_BACKEND": "memory",
        "TP_ORCHESTRATOR_QUEUE_BACKEND": "memory",
        "TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED": "0",
        "TP_PILOT_CONTROL_PLANE_ENABLED": "0",
        "TP_ENABLE_TRUSTED_HOSTS": "0",
        "TP_RATE_LIMIT_PER_MINUTE": "1000",
    }
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=repo_root, env=environment, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
