"""Script-free OpenAPI readers and strict-CSP documentation compatibility routes."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_oauth2_redirect_html
from starlette.responses import Response

from transformation_portal.ingest.canonical_json import dumps_json

_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "options", "head", "trace")
_DOCS_CSS_PATH = "/docs/assets/api-docs.css"
_OAUTH_SCRIPT_PATH = "/docs/assets/oauth2-redirect.js"
_NO_STORE = {"Cache-Control": "no-store"}


def _schema_id(name: str) -> str:
    return "schema-" + name.encode("utf-8").hex()


def _schema_references(value: Any, schemas: Mapping[str, Any]) -> list[str]:
    references: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            reference = item.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
                name = reference.removeprefix("#/components/schemas/").replace("~1", "/").replace("~0", "~")
                if name in schemas:
                    references.add(name)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return sorted(references)


def _definition(label: str, value: Any, schemas: Mapping[str, Any]) -> str:
    encoded = html.escape(dumps_json(value, indent=2, ensure_ascii=False, sort_keys=True))
    links = ", ".join(
        f'<a href="#{html.escape(_schema_id(name))}">{html.escape(name)}</a>' for name in _schema_references(value, schemas)
    )
    references = f'<p class="schema-links">Referenced schemas: {links}</p>' if links else ""
    return f"<details><summary>{html.escape(label)}</summary>{references}<pre><code>{encoded}</code></pre></details>"


def _render_docs(schema: Mapping[str, Any], *, title: str, openapi_url: str, stylesheet_url: str) -> str:
    schemas = schema.get("components", {}).get("schemas", {})
    operations: list[str] = []
    for path, path_item in sorted(schema.get("paths", {}).items()):
        for method in _HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            description = html.escape(str(operation.get("description", "")))
            summary = html.escape(str(operation.get("summary", operation.get("operationId", ""))))
            path_parameters = (
                _definition("Path-level parameters", path_item["parameters"], schemas) if "parameters" in path_item else ""
            )
            request_body = (
                _definition("Request body", operation["requestBody"], schemas)
                if "requestBody" in operation
                else "<p>No request body is declared in this schema.</p>"
            )
            operations.append(
                '<details class="operation" data-ui="api-docs-operation">'
                f'<summary><span class="method">{method.upper()}</span> '
                f"<code>{html.escape(path)}</code> {summary}</summary>"
                f'<p class="description">{description}</p>{path_parameters}'
                f'{_definition("Parameters", operation.get("parameters", []), schemas)}'
                f'{request_body}{_definition("Responses", operation.get("responses", {}), schemas)}'
                "</details>"
            )
    schema_definitions = "".join(
        f'<section id="{html.escape(_schema_id(name))}" data-ui="api-docs-schema">'
        f"{_definition(name, definition, schemas)}</section>"
        for name, definition in sorted(schemas.items())
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)} — API Reference</title>"
        f'<link rel="stylesheet" href="{html.escape(stylesheet_url)}"></head><body><main>'
        f"<h1>{html.escape(title)}</h1><p>API reference</p>"
        f'<p><a data-ui="api-docs-schema-link" href="{html.escape(openapi_url)}">OpenAPI JSON schema</a> '
        '<a href="#schemas">Schema definitions</a></p>'
        f'<section aria-label="API operations">{"".join(operations)}</section>'
        f'<section id="schemas"><h2>Schema definitions</h2>{schema_definitions}</section>'
        "</main></body></html>"
    )


def _oauth_redirect_assets() -> tuple[str, bytes]:
    """Externalize exactly one trusted FastAPI callback script; reject helper drift."""
    document = get_swagger_ui_oauth2_redirect_html().body.decode("utf-8")
    scripts = list(re.finditer(r"<script>(.*?)</script>", document, re.DOTALL))
    if len(scripts) != 1 or not scripts[0].group(1).strip():
        raise RuntimeError("FastAPI OAuth redirect must contain exactly one inline script")
    script = scripts[0]
    shell = document[: script.start()] + "__OAUTH_SCRIPT__" + document[script.end() :]
    if re.search(r"<(?:script|style)\b|\s(?:style|on\w+)\s*=", shell, re.IGNORECASE):
        raise RuntimeError("FastAPI OAuth redirect contains unexpected inline content")
    return shell, script.group(1).encode("utf-8")


def create_api_docs_router(app: FastAPI) -> APIRouter:
    """Build docs routes; the caller gates registration with its existing docs flag."""
    if app.openapi_url is None:
        raise ValueError("API docs require an enabled OpenAPI schema URL")
    openapi_url = app.openapi_url
    stylesheet = files("transformation_portal.api").joinpath("docs_assets", "api-docs.css").read_bytes()
    oauth_shell, oauth_script = _oauth_redirect_assets()
    router = APIRouter()

    @router.api_route("/docs", methods=["GET", "HEAD"], name="swagger_ui_html", include_in_schema=False)
    @router.api_route("/redoc", methods=["GET", "HEAD"], name="redoc_html", include_in_schema=False)
    async def docs_reader(request: Request) -> Response:
        root_path = request.scope.get("root_path", "").rstrip("/")
        content = _render_docs(
            app.openapi(),
            title=app.title,
            openapi_url=root_path + openapi_url,
            stylesheet_url=root_path + _DOCS_CSS_PATH,
        )
        return Response(content, media_type="text/html", headers=_NO_STORE)

    @router.get(_DOCS_CSS_PATH, include_in_schema=False)
    async def docs_stylesheet() -> Response:
        return Response(stylesheet, media_type="text/css", headers=_NO_STORE)

    @router.api_route("/docs/oauth2-redirect", methods=["GET", "HEAD"], name="swagger_ui_redirect", include_in_schema=False)
    async def oauth_redirect(request: Request) -> Response:
        root_path = request.scope.get("root_path", "").rstrip("/")
        script_url = html.escape(root_path + _OAUTH_SCRIPT_PATH)
        content = oauth_shell.replace("__OAUTH_SCRIPT__", f'<script src="{script_url}" defer></script>')
        return Response(content, media_type="text/html", headers=_NO_STORE)

    @router.get(_OAUTH_SCRIPT_PATH, include_in_schema=False)
    async def oauth_redirect_script() -> Response:
        return Response(oauth_script, media_type="application/javascript", headers=_NO_STORE)

    return router
