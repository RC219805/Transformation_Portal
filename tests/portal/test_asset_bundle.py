"""Unit tests for portal asset bundle helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
from starlette.requests import Request

from transformation_portal.portal import asset_bundle

pytestmark = pytest.mark.unit


def _request(
    *,
    query_string: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/portal/assets/portal.css",
            "query_string": query_string,
            "headers": headers or [],
        }
    )


def test_load_portal_asset_manifest_rejects_paths_outside_assets_dir(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "portal-asset-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "assets": {
                    "portal.css": {
                        "repo_path": "../portal.html",
                        "media_type": "text/css; charset=utf-8",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(asset_bundle, "PORTAL_ASSET_MANIFEST_PATH", manifest_path)

    with pytest.raises(RuntimeError, match="points outside"):
        asset_bundle._load_portal_asset_manifest()


def test_portal_asset_manifest_exports_load_lazily_until_access(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asset_bundle, "PORTAL_ASSET_MANIFEST_PATH", tmp_path / "missing-manifest.json")
    asset_bundle._load_portal_asset_manifest_cached.cache_clear()

    assert isinstance(asset_bundle.PORTAL_ASSET_MANIFEST, Mapping)
    with pytest.raises(RuntimeError, match="Unable to load portal asset manifest"):
        len(asset_bundle.PORTAL_ASSET_MANIFEST)


def test_render_portal_template_rejects_missing_required_token() -> None:
    with pytest.raises(RuntimeError, match="missing required token __PORTAL_JS_URL__"):
        asset_bundle._render_portal_template(
            "<html></html>",
            {"__PORTAL_JS_URL__": "/portal/assets/portal.js?v=abc123"},
            template_name="portal.html",
        )


def test_render_portal_template_rejects_unresolved_portal_tokens() -> None:
    with pytest.raises(RuntimeError, match="unresolved tokens: __PORTAL_UNKNOWN_URL__"):
        asset_bundle._render_portal_template(
            "<html>__PORTAL_UNKNOWN_URL__</html>",
            {},
            template_name="portal.html",
        )


def test_portal_asset_bundle_emits_versioned_asset_urls() -> None:
    bundle = asset_bundle._get_portal_asset_bundle()

    assert bundle.urls["portal.css"].startswith("/portal/assets/portal.css?v=")
    assert bundle.urls["portal.js"].startswith("/portal/assets/portal.js?v=")
    assert bundle.urls["fonts/portal-sans.woff2"].startswith("/portal/assets/fonts/portal-sans.woff2?v=")
    assert bundle.urls["portal.css"] in bundle.html
    assert bundle.urls["fonts/portal-sans.woff2"] in bundle.css
    assert len(bundle.fingerprints["portal.css"]) == asset_bundle.PORTAL_ASSET_FINGERPRINT_LENGTH
    assert len(bundle.html_bytes) == len(bundle.html.encode("utf-8"))
    assert len(bundle.css_bytes) == len(bundle.css.encode("utf-8"))


def test_portal_asset_cache_control_preserves_legacy_unversioned_and_stale_requests() -> None:
    assert asset_bundle._portal_asset_cache_control("abc123", "") == asset_bundle.PORTAL_ASSET_CACHE_CONTROL
    assert asset_bundle._portal_asset_cache_control("abc123", "stale") == asset_bundle.PORTAL_ASSET_CACHE_CONTROL
    assert asset_bundle._portal_asset_cache_control("abc123", "abc123") == asset_bundle.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL


def test_requested_portal_asset_fingerprint_reads_version_query_param() -> None:
    request = _request(query_string=b"v=abc123")

    assert asset_bundle._requested_portal_asset_fingerprint(request) == "abc123"


def test_portal_asset_etag_matching_accepts_exact_match_and_wildcard() -> None:
    exact_request = _request(headers=[(b"if-none-match", b'"abc123"')])
    wildcard_request = _request(headers=[(b"if-none-match", b"*")])
    stale_request = _request(headers=[(b"if-none-match", b'"stale"')])

    assert asset_bundle._portal_asset_request_etag_matches(exact_request, '"abc123"') is True
    assert asset_bundle._portal_asset_request_etag_matches(wildcard_request, '"abc123"') is True
    assert asset_bundle._portal_asset_request_etag_matches(stale_request, '"abc123"') is False


def test_portal_asset_not_modified_response_sets_cache_headers() -> None:
    response = asset_bundle._portal_asset_not_modified_response(
        etag='"abc123"',
        cache_control=asset_bundle.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL,
    )

    assert response.status_code == 304
    assert response.headers["Cache-Control"] == asset_bundle.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert response.headers["ETag"] == '"abc123"'
