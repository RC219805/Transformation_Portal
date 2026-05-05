"""Portal asset bundle rendering, fingerprints, and cache semantics."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Tuple
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import Response

__all__ = [
    "PORTAL_ASSETS_DIR",
    "PORTAL_ASSETS_DIR_REAL",
    "PORTAL_ASSET_CACHE_CONTROL",
    "PORTAL_ASSET_FINGERPRINT_LENGTH",
    "PORTAL_ASSET_FINGERPRINT_PARAM",
    "PORTAL_ASSET_MANIFEST",
    "PORTAL_ASSET_MANIFEST_PATH",
    "PORTAL_ASSET_MEDIA_TYPES",
    "PORTAL_ASSET_PATHS",
    "PORTAL_CSS_TEMPLATE_PATH",
    "PORTAL_CSS_TEMPLATE_TOKENS",
    "PORTAL_DIRECT_FINGERPRINT_ASSET_NAMES",
    "PORTAL_HTML",
    "PORTAL_HTML_TEMPLATE_TOKENS",
    "PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL",
    "PortalAssetBundle",
    "PortalAssetSpec",
    "PortalRenderedTextAsset",
    "_build_portal_asset_bundle",
    "_build_portal_css_asset",
    "_build_portal_direct_asset_fingerprint",
    "_fingerprint_bytes",
    "_get_portal_asset_bundle",
    "_get_portal_css_asset",
    "_get_portal_direct_asset_fingerprint",
    "_load_portal_asset_manifest",
    "_portal_asset_cache_control",
    "_portal_asset_etag",
    "_portal_asset_not_modified_response",
    "_portal_asset_request_etag_matches",
    "_portal_asset_route_path",
    "_portal_asset_signature",
    "_portal_asset_versioned_url",
    "_portal_css_dependency_asset_names",
    "_portal_css_signature",
    "_portal_direct_asset_signature",
    "_portal_html_signature",
    "_render_portal_template",
    "_requested_portal_asset_fingerprint",
]


@dataclass(frozen=True)
class PortalAssetSpec:
    path: Path
    media_type: str


@dataclass(frozen=True)
class PortalAssetBundle:
    html: str
    html_bytes: bytes
    css: str
    css_bytes: bytes
    fingerprints: Dict[str, str]
    urls: Dict[str, str]


@dataclass(frozen=True)
class PortalRenderedTextAsset:
    text: str
    content_bytes: bytes
    fingerprint: str


REPO_ROOT = Path(__file__).resolve().parents[3]
PORTAL_HTML = REPO_ROOT / "portal.html"
PORTAL_ASSETS_DIR = REPO_ROOT / "public" / "portal-assets"
PORTAL_ASSETS_DIR_REAL = Path(os.path.realpath(PORTAL_ASSETS_DIR))
PORTAL_ASSET_MANIFEST_PATH = REPO_ROOT / "config" / "portal_asset_manifest.json"
PORTAL_ASSET_CACHE_CONTROL = "no-store"
PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"
PORTAL_ASSET_FINGERPRINT_PARAM = "v"
PORTAL_ASSET_FINGERPRINT_LENGTH = 12
PORTAL_CSS_TEMPLATE_PATH = PORTAL_ASSETS_DIR / "portal.css"
PORTAL_DIRECT_FINGERPRINT_ASSET_NAMES = (
    "portal.js",
    "portal-review.js",
    "portal-review.css",
    "shared-ui-tokens.css",
    "fonts/portal-sans.woff2",
    "fonts/portal-mono.woff2",
    "brand/dna-symbol-dark.svg",
    "brand/dna-symbol-light.svg",
)
PORTAL_CSS_TEMPLATE_TOKENS = {
    "__PORTAL_FONT_SANS_URL__": "fonts/portal-sans.woff2",
    "__PORTAL_FONT_MONO_URL__": "fonts/portal-mono.woff2",
}
PORTAL_HTML_TEMPLATE_TOKENS = {
    "__PORTAL_CSS_URL__": "portal.css",
    "__PORTAL_JS_URL__": "portal.js",
    "__PORTAL_REVIEW_JS_URL__": "portal-review.js",
    "__PORTAL_REVIEW_CSS_URL__": "portal-review.css",
    "__PORTAL_BRAND_LIGHT_URL__": "brand/dna-symbol-light.svg",
    "__PORTAL_BRAND_DARK_URL__": "brand/dna-symbol-dark.svg",
    "__PORTAL_FONT_SANS_URL__": "fonts/portal-sans.woff2",
}


def _load_portal_asset_manifest() -> Dict[str, PortalAssetSpec]:
    try:
        raw_manifest = json.loads(PORTAL_ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load portal asset manifest: {exc}") from exc

    assets = raw_manifest.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise RuntimeError("Portal asset manifest must define a non-empty 'assets' object")

    manifest: Dict[str, PortalAssetSpec] = {}
    for asset_name, entry in assets.items():
        if not isinstance(asset_name, str) or not asset_name.strip():
            raise RuntimeError("Portal asset manifest contains an invalid asset key")
        if not isinstance(entry, dict):
            raise RuntimeError(f"Portal asset manifest entry for {asset_name!r} must be an object")

        repo_path = str(entry.get("repo_path", "")).strip()
        media_type = str(entry.get("media_type", "")).strip()
        if not repo_path or not media_type:
            raise RuntimeError(f"Portal asset manifest entry for {asset_name!r} is incomplete")

        resolved_path = Path(os.path.realpath(REPO_ROOT / repo_path))
        try:
            resolved_path.relative_to(PORTAL_ASSETS_DIR_REAL)
        except ValueError as exc:
            raise RuntimeError(f"Portal asset manifest entry for {asset_name!r} points outside {PORTAL_ASSETS_DIR}") from exc

        manifest[asset_name] = PortalAssetSpec(path=resolved_path, media_type=media_type)

    return manifest


PORTAL_ASSET_MANIFEST = _load_portal_asset_manifest()
PORTAL_ASSET_PATHS = {name: asset.path for name, asset in PORTAL_ASSET_MANIFEST.items()}
PORTAL_ASSET_MEDIA_TYPES = {name: asset.media_type for name, asset in PORTAL_ASSET_MANIFEST.items()}


def _portal_asset_signature(path: Path) -> Tuple[str, int, int]:
    stat_result = path.stat()
    return str(path), stat_result.st_mtime_ns, stat_result.st_size


def _fingerprint_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:PORTAL_ASSET_FINGERPRINT_LENGTH]


def _portal_asset_route_path(asset_name: str) -> str:
    encoded_parts = [quote(part, safe="") for part in asset_name.split("/")]
    return "/portal/assets/" + "/".join(encoded_parts)


def _portal_asset_versioned_url(asset_name: str, fingerprint: str) -> str:
    return f"{_portal_asset_route_path(asset_name)}?{PORTAL_ASSET_FINGERPRINT_PARAM}={fingerprint}"


def _render_portal_template(template_text: str, replacements: Mapping[str, str], *, template_name: str) -> str:
    rendered = template_text
    for token, value in replacements.items():
        if token not in rendered:
            raise RuntimeError(f"{template_name} is missing required token {token}")
        rendered = rendered.replace(token, value)

    unresolved = sorted(set(re.findall(r"__PORTAL_[A-Z0-9_]+__", rendered)))
    if unresolved:
        raise RuntimeError(f"{template_name} has unresolved tokens: {', '.join(unresolved)}")
    return rendered


def _portal_direct_asset_signature(asset_name: str) -> Tuple[str, int, int]:
    return _portal_asset_signature(PORTAL_ASSET_PATHS[asset_name])


@lru_cache(maxsize=16)
def _build_portal_direct_asset_fingerprint(asset_name: str, _: Tuple[str, int, int]) -> str:
    return _fingerprint_bytes(PORTAL_ASSET_PATHS[asset_name].read_bytes())


def _get_portal_direct_asset_fingerprint(asset_name: str) -> str:
    return _build_portal_direct_asset_fingerprint(asset_name, _portal_direct_asset_signature(asset_name))


def _portal_css_dependency_asset_names() -> Tuple[str, ...]:
    return tuple(dict.fromkeys(PORTAL_CSS_TEMPLATE_TOKENS.values()))


def _portal_css_signature() -> Tuple[object, ...]:
    return (
        _portal_asset_signature(PORTAL_CSS_TEMPLATE_PATH),
        *(
            (asset_name, _get_portal_direct_asset_fingerprint(asset_name))
            for asset_name in _portal_css_dependency_asset_names()
        ),
    )


@lru_cache(maxsize=4)
def _build_portal_css_asset(_: Tuple[object, ...]) -> PortalRenderedTextAsset:
    direct_asset_fingerprints = {
        asset_name: _get_portal_direct_asset_fingerprint(asset_name)
        for asset_name in dict.fromkeys(PORTAL_CSS_TEMPLATE_TOKENS.values())
    }
    css_template = PORTAL_CSS_TEMPLATE_PATH.read_text(encoding="utf-8")
    css_render = _render_portal_template(
        css_template,
        {
            token: _portal_asset_versioned_url(asset_name, direct_asset_fingerprints[asset_name])
            for token, asset_name in PORTAL_CSS_TEMPLATE_TOKENS.items()
        },
        template_name="portal.css",
    )
    css_bytes = css_render.encode("utf-8")
    return PortalRenderedTextAsset(
        text=css_render,
        content_bytes=css_bytes,
        fingerprint=_fingerprint_bytes(css_bytes),
    )


def _get_portal_css_asset() -> PortalRenderedTextAsset:
    return _build_portal_css_asset(_portal_css_signature())


def _portal_html_signature() -> Tuple[object, ...]:
    css_asset = _get_portal_css_asset()
    return (
        _portal_asset_signature(PORTAL_HTML),
        ("portal.css", css_asset.fingerprint),
        *(
            (asset_name, _get_portal_direct_asset_fingerprint(asset_name))
            for asset_name in (
                "portal.js",
                "portal-review.js",
                "portal-review.css",
                "brand/dna-symbol-dark.svg",
                "brand/dna-symbol-light.svg",
                "fonts/portal-sans.woff2",
            )
        ),
    )


@lru_cache(maxsize=4)
def _build_portal_asset_bundle(_: Tuple[object, ...]) -> PortalAssetBundle:
    css_asset = _get_portal_css_asset()
    fingerprints = {
        asset_name: _get_portal_direct_asset_fingerprint(asset_name) for asset_name in PORTAL_DIRECT_FINGERPRINT_ASSET_NAMES
    }
    fingerprints["portal.css"] = css_asset.fingerprint
    urls = {
        asset_name: _portal_asset_versioned_url(asset_name, fingerprint) for asset_name, fingerprint in fingerprints.items()
    }
    html_template = PORTAL_HTML.read_text(encoding="utf-8")
    html_render = _render_portal_template(
        html_template,
        {token: urls[asset_name] for token, asset_name in PORTAL_HTML_TEMPLATE_TOKENS.items()},
        template_name="portal.html",
    )
    html_bytes = html_render.encode("utf-8")
    return PortalAssetBundle(
        html=html_render,
        html_bytes=html_bytes,
        css=css_asset.text,
        css_bytes=css_asset.content_bytes,
        fingerprints=fingerprints,
        urls=urls,
    )


def _get_portal_asset_bundle() -> PortalAssetBundle:
    return _build_portal_asset_bundle(_portal_html_signature())


def _requested_portal_asset_fingerprint(request: Request) -> str:
    return str(request.query_params.get(PORTAL_ASSET_FINGERPRINT_PARAM, "")).strip()


def _portal_asset_cache_control(current_fingerprint: str, requested_fingerprint: str) -> str:
    if requested_fingerprint and hmac.compare_digest(requested_fingerprint, current_fingerprint):
        return PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    return PORTAL_ASSET_CACHE_CONTROL


def _portal_asset_etag(fingerprint: str) -> str:
    return f'"{fingerprint}"'


def _portal_asset_request_etag_matches(request: Request, current_etag: str) -> bool:
    raw_if_none_match = str(request.headers.get("if-none-match") or "").strip()
    if not raw_if_none_match:
        return False
    for candidate in raw_if_none_match.split(","):
        normalized = candidate.strip()
        if normalized == "*":
            return True
        if normalized and hmac.compare_digest(normalized, current_etag):
            return True
    return False


def _portal_asset_not_modified_response(*, etag: str, cache_control: str) -> Response:
    return Response(
        status_code=304,
        headers={
            "Cache-Control": cache_control,
            "ETag": etag,
        },
    )
