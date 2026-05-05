#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for root FastAPI orchestrator app runtime behavior."""

from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import gzip
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.requests import Request as StarletteRequest

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")
portal_asset_bundle = importlib.import_module("transformation_portal.portal.asset_bundle")
upload_staging = importlib.import_module("transformation_portal.ingest.upload_staging")
PORTAL_HTML_PATH = Path(__file__).resolve().parents[1] / "portal.html"
PORTAL_ASSET_ROOT = PORTAL_HTML_PATH.parent / "public" / "portal-assets"
PORTAL_FRONTDOOR_ROOT = PORTAL_HTML_PATH.parent / "web" / "secure-landing"
PORTAL_INTERNAL_STATE_SOURCE_PATH = PORTAL_FRONTDOOR_ROOT / "portal-src" / "internal" / "state.js"
PORTAL_TEMPLATE_SOURCE_PATH = PORTAL_FRONTDOOR_ROOT / "portal-src" / "portal.template.js"
PORTAL_REVIEW_SURFACE_SOURCE_PATH = PORTAL_FRONTDOOR_ROOT / "portal-src" / "review-surface-deferred.js"
FRONTDOOR_BRAND_ROOT = PORTAL_HTML_PATH.parent / "web" / "secure-landing" / "public" / "brand"


class _FakeRequest:
    """Lightweight request stub for SSE generator tests."""

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _flag_value(argv: list[str], flag: str) -> str:
    idx = argv.index(flag)
    return argv[idx + 1]


def _write_archive_index(path: Path, relpaths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["origin_drive", "partition", "relpath"],
            lineterminator="\n",
        )
        writer.writeheader()
        for relpath in relpaths:
            writer.writerow(
                {
                    "origin_drive": "local",
                    "partition": "test",
                    "relpath": relpath,
                }
            )


@lru_cache(maxsize=1)
def _portal_html_content() -> str:
    return orchestrator_app._get_portal_asset_bundle().html


@lru_cache(maxsize=1)
def _portal_asset_urls_from_html() -> set[str]:
    return set(re.findall(r'["\'](/portal/assets/[^"\']+)["\']', _portal_html_content()))


@lru_cache(maxsize=1)
def _portal_asset_urls_from_css() -> set[str]:
    return set(re.findall(r'url\(["\']?(/portal/assets/[^)"\']+)["\']?\)', _portal_css_content()))


def _portal_asset_path(asset_url: str) -> Path:
    parsed = urlparse(asset_url)
    if not parsed.path.startswith("/portal/assets/"):
        raise AssertionError(f"unexpected portal asset url: {asset_url}")
    candidate = PORTAL_ASSET_ROOT / parsed.path.removeprefix("/portal/assets/")
    if not candidate.is_file():
        raise AssertionError(f"portal asset missing: {candidate}")
    return candidate


def _portal_asset_name(asset_url: str) -> str:
    parsed = urlparse(asset_url)
    if not parsed.path.startswith("/portal/assets/"):
        raise AssertionError(f"unexpected portal asset url: {asset_url}")
    return parsed.path.removeprefix("/portal/assets/")


def _portal_asset_version(asset_url: str) -> str:
    return parse_qs(urlparse(asset_url).query).get("v", [""])[0]


@lru_cache(maxsize=1)
def _portal_css_content() -> str:
    html = _portal_html_content()
    match = re.search(r'<link rel="stylesheet" href="(/portal/assets/[^"]+)"\s*/?>', html)
    if match is None:
        raise AssertionError("portal stylesheet link not found")
    return orchestrator_app._get_portal_asset_bundle().css


@lru_cache(maxsize=1)
def _portal_js_content() -> str:
    html = _portal_html_content()
    match = re.search(r'<script src="(/portal/assets/[^"]+)" defer></script>', html)
    if match is None:
        raise AssertionError("portal script tag not found")
    return _portal_asset_path(match.group(1)).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _portal_js_source_content() -> str:
    return PORTAL_TEMPLATE_SOURCE_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _portal_review_bundle_content() -> str:
    html = _portal_html_content()
    match = re.search(r'data-review-surface-js-url="(/portal/assets/[^"]+)"', html)
    if match is None:
        raise AssertionError("portal review surface asset url not found")
    return _portal_asset_path(match.group(1)).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _portal_review_source_content() -> str:
    return PORTAL_REVIEW_SURFACE_SOURCE_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _portal_internal_state_source_content() -> str:
    return PORTAL_INTERNAL_STATE_SOURCE_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _portal_bundle_content() -> str:
    return "\n".join((_portal_html_content(), _portal_css_content(), _portal_js_source_content()))


@lru_cache(maxsize=1)
def _portal_runtime_bundle_content() -> str:
    return "\n".join((_portal_html_content(), _portal_css_content(), _portal_js_content()))


def _extract_js_function_body(content: str, function_name: str) -> str:
    marker = f"function {function_name}("
    start = content.find(marker)
    if start < 0:
        raise AssertionError(f"{function_name} not found")
    brace_start = content.find("{", start)
    if brace_start < 0:
        raise AssertionError(f"{function_name} opening brace not found")

    depth = 0
    for idx in range(brace_start, len(content)):
        char = content[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[brace_start + 1 : idx]
    raise AssertionError(f"{function_name} closing brace not found")


def _extract_js_function_block(content: str, function_name: str) -> str:
    marker = f"function {function_name}("
    start = content.find(marker)
    if start < 0:
        raise AssertionError(f"{function_name} not found")
    next_candidates = [
        content.find("\nfunction ", start + len(marker)),
        content.find("\nasync function ", start + len(marker)),
        content.find("\n        function ", start + len(marker)),
        content.find("\n        async function ", start + len(marker)),
    ]
    next_marker = min((candidate for candidate in next_candidates if candidate >= 0), default=-1)
    if next_marker < 0:
        return content[start:]
    return content[start:next_marker]


def _extract_portal_canonical_lux_arg_keys(content: str) -> set[str]:
    body = _extract_js_function_body(content, "buildCanonicalLuxDepthArgs")
    args_match = re.search(r"const args = \{(.*?)\n\s*\};", body, flags=re.DOTALL)
    if args_match is None:
        raise AssertionError("buildCanonicalLuxDepthArgs args object not found")
    args_body = args_match.group(1)
    explicit_keys = set(re.findall(r"^\s*([a-z_][a-z0-9_]*)\s*:", args_body, flags=re.MULTILINE))
    shorthand_keys = set(re.findall(r"^\s*([a-z_][a-z0-9_]*)\s*,?\s*$", args_body, flags=re.MULTILINE))
    conditional_keys = set(re.findall(r"\bargs\.([a-z_][a-z0-9_]*)\s*=", body))
    return explicit_keys | shorthand_keys | conditional_keys


def _extract_portal_lux_cli_flags(content: str) -> set[str]:
    body = _extract_js_function_body(content, "renderCLI")
    lux_block = re.search(
        r"if \(payload\.pipeline === 'lux-depth-v3'\) \{(.*?)\n\s*\} else \{",
        body,
        flags=re.DOTALL,
    )
    if lux_block is None:
        raise AssertionError("renderCLI lux-depth-v3 block not found")
    return set(re.findall(r"--[a-z0-9-]+", lux_block.group(1)))


def _capture_lux_cli_config_from_args(cli_args: list[str]) -> object:
    from unittest.mock import MagicMock, patch

    from typer.testing import CliRunner

    from transformation_portal.lux_depth_v3.__main__ import app as lux_cli_app

    captured: Dict[str, object] = {}

    def _mock_orchestrator_init(config, output_root):
        del output_root
        captured["config"] = config
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_batch.return_value = [{"status": "ok"}]
        return mock_orchestrator

    with patch(
        "transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator",
        side_effect=_mock_orchestrator_init,
    ):
        result = CliRunner().invoke(lux_cli_app, cli_args)
    assert result.exit_code == 0, result.stdout
    assert "config" in captured
    return captured["config"]


def _run_card_fingerprint_from_config(config: object) -> Dict[str, object]:
    from transformation_portal.lux_depth_v3.config import ModelVariant
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    if getattr(config, "model_variant", None) is None:
        setattr(config, "model_variant", ModelVariant.METRIC_LARGE)

    orch = object.__new__(EnhanceOrchestrator)
    orch.config = config
    requested_backend = getattr(config, "depth_backend", None) or "da3"
    resolved_backend = (
        "da3"
        if str(requested_backend).strip().lower() in {"depth-anything-v3", "depth_anything_v3"}
        else str(requested_backend).strip().lower()
    )
    orch._backend_metadata = SimpleNamespace(
        requested_backend=str(requested_backend).strip().lower(),
        resolved_backend=resolved_backend,
        device=getattr(config, "depth_device", "cpu"),
    )
    orch._is_apex_tier = lambda: str(getattr(config, "quality_tier", "standard")).strip().lower() == "apex"
    return orch._build_run_card_config_fingerprint()


def _build_request(
    method: str,
    path: str,
    headers: Dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
    query_string: str = "",
) -> StarletteRequest:
    raw_headers = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query_string.encode("utf-8"),
        "headers": raw_headers,
        "client": (client_host, 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return StarletteRequest(scope, receive)


def _build_multipart_form_body(
    boundary: str,
    *,
    fields: list[tuple[str, str]] | None = None,
    files: list[tuple[str, str, bytes, str]] | None = None,
) -> bytes:
    payload = bytearray()
    for field_name, value in fields or []:
        payload.extend(f"--{boundary}\r\n".encode("utf-8"))
        payload.extend(f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode("utf-8"))
        payload.extend(value.encode("utf-8"))
        payload.extend(b"\r\n")
    for field_name, filename, content, content_type in files or []:
        payload.extend(f"--{boundary}\r\n".encode("utf-8"))
        payload.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        payload.extend(content)
        payload.extend(b"\r\n")
    payload.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(payload)


@pytest.fixture(autouse=True)
def _reset_global_state() -> None:
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    yield
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()


def test_argv_normalization_accepts_canonical_keys() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "preset": "premium",
            "quality_tier": "apex",
            "depth_backend": "da3",
            "depth_device": "cuda",
            "materials_v3": True,
            "pbr": True,
            "cache_depth": True,
            "emit_master16": True,
            "emit_upscaled16": True,
            "emit_marketing": False,
            "emit_report": True,
            "emit_run_card": True,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[:3] == [sys.executable, "-m", orchestrator_app.LUX_DEPTH_MODULE]
    assert _flag_value(argv, "--materials-v3") == "on"
    assert _flag_value(argv, "--cache-depth") == "on"
    assert _flag_value(argv, "--depth-backend") == "da3"
    assert _flag_value(argv, "--depth-device") == "cuda"
    assert _flag_value(argv, "--emit-report") == "on"
    assert _flag_value(argv, "--run-card-version") == "v1"


def test_argv_normalization_accepts_legacy_keys() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "quality_tier": "standard",
            "depth_backend": "depth_anything_v3",
            "depthDevice": "cpu",
            "materials": True,
            "cacheDepth": True,
            "pbr": False,
            "enableV2": True,
            "v2Preset": "default",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--materials-v3") == "on"
    assert _flag_value(argv, "--cache-depth") == "on"
    assert _flag_value(argv, "--depth-backend") == "da3"
    assert _flag_value(argv, "--depth-device") == "cpu"
    assert _flag_value(argv, "--enable-v2") == "on"
    assert _flag_value(argv, "--v2-preset") == "default"


def test_argv_normalization_honors_explicit_v2_disable() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "enable_v2": False,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--enable-v2") == "off"
    assert "--v2-preset" not in argv


def test_argv_normalization_depth_backend_is_case_insensitive() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "depth_backend": "Depth-Anything-V3",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)
    assert _flag_value(argv, "--depth-backend") == "da3"


def test_argv_normalization_trims_and_normalizes_string_values() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "preset": " premium ",
            "quality_tier": " APEX ",
            "depth_backend": " Depth_Pro ",
            "depth_device": "  cpu  ",
            "materials_v3": "on",
            "pbr": "false",
            "cache_depth": "off",
            "enable_v2": "off",
            "emit_master16": "true",
            "emit_upscaled16": "1",
            "emit_marketing": "0",
            "emit_report": "yes",
            "emit_run_card": "no",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--preset") == "premium"
    assert _flag_value(argv, "--quality-tier") == "apex"
    assert _flag_value(argv, "--depth-backend") == "depth_pro"
    assert _flag_value(argv, "--depth-device") == "cpu"
    assert _flag_value(argv, "--materials-v3") == "on"
    assert _flag_value(argv, "--pbr") == "off"
    assert _flag_value(argv, "--cache-depth") == "off"
    assert _flag_value(argv, "--enable-v2") == "off"
    assert "--v2-preset" not in argv
    assert _flag_value(argv, "--emit-marketing") == "off"
    assert _flag_value(argv, "--emit-report") == "on"
    assert _flag_value(argv, "--emit-run-card") == "off"
    assert _flag_value(argv, "--run-card-version") == "v1"


def test_argv_normalization_trims_pipeline_name() -> None:
    payload: Dict[str, object] = {
        "pipeline": " lux-depth-v3 ",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[:3] == [sys.executable, "-m", orchestrator_app.LUX_DEPTH_MODULE]
    assert _flag_value(argv, "--input-dir") == str((orchestrator_app.REPO_ROOT / "input_images").resolve())
    assert _flag_value(argv, "--output-dir") == str((orchestrator_app.REPO_ROOT / "output").resolve())


def test_portal_cli_template_excludes_unsupported_lux_flags() -> None:
    content = _portal_bundle_content()
    assert "--emit-manifest" not in content
    assert "--emit-provenance" not in content
    assert "--enable-segmentation" in content
    assert "--segmentation-backend" in content
    assert "--sam2-model-size" in content
    assert "--strict-segmentation" in content


def test_portal_html_resets_to_static_operator_shell_without_background_video() -> None:
    html_content = _portal_html_content()
    css_content = _portal_css_content()

    assert "portal-video-backdrop" not in html_content
    assert 'class="portal-video-media"' not in html_content
    assert "/portal/video/dna-portal-video-2.mp4" not in html_content
    assert "shell-noise" in html_content
    assert ".shell-noise" in css_content
    assert ".shell-bg" in css_content


def test_portal_phase1_accessibility_tokens_align_focus_and_target_size() -> None:
    css_content = _portal_css_content()
    shared_tokens = orchestrator_app.PORTAL_ASSET_PATHS["shared-ui-tokens.css"].read_text(encoding="utf-8")

    assert "--ux-focus-ring:" in shared_tokens
    assert "--ux-focus-shadow:" in shared_tokens
    assert re.search(r"--ux-target-min-size:\s*44px;", shared_tokens)
    assert re.search(r"font-size:\s*var\(--ux-body-size\)", css_content)
    assert len(re.findall(r"--shell-border:\s*var\(--ux-panel-border\)", css_content)) >= 2
    assert "--shell-border: rgba(148, 163, 184, 0.22);" not in css_content
    assert "summary:focus-visible" in css_content
    assert "#build-shell label:not(.sr-only)" in css_content


def test_portal_routed_shell_hidden_rules_preserve_responsive_display_utilities() -> None:
    html_content = _portal_html_content()
    css_content = _portal_css_content()

    hidden_rule = re.search(r"\.hidden\s*\{(?P<body>[^}]*)}", css_content, re.S)
    assert hidden_rule is not None
    hidden_body = hidden_rule.group("body")
    assert re.search(r"display:\s*none;?", hidden_body)
    assert "!important" not in hidden_body
    assert 'class="topbar-status hidden lg:flex"' in html_content

    route_shell_rule = re.search(
        (
            r"#overview-shell\.hidden,\s*"
            r"#console-grid\.hidden,\s*"
            r"#build-shell\.hidden,\s*"
            r"#jobs-shell\.hidden\s*\{(?P<body>[^}]*)}"
        ),
        css_content,
        re.S,
    )
    assert route_shell_rule is not None
    assert re.search(r"display:\s*none;?", route_shell_rule.group("body"))
    assert "!important" not in route_shell_rule.group("body")


def test_portal_shell_veil_tokens_use_shell_namespace_and_ordered_opacity() -> None:
    css_content = _portal_css_content()

    assert "--s-veil" not in css_content
    assert "--s-tint-faint" not in css_content
    assert "--shell-tint-faint:" in css_content

    def token_alpha(selector: str, token: str) -> float:
        block_body = None
        for block_match in re.finditer(rf"{re.escape(selector)}\s*\{{(?P<body>[^}}]*)\}}", css_content):
            if token in block_match.group("body"):
                block_body = block_match.group("body")
                break
        assert block_body is not None, f"{selector} token block missing"
        value_match = re.search(
            rf"{re.escape(token)}:\s*rgba\(\s*\d+,\s*\d+,\s*\d+,\s*(?P<alpha>(?:0?\.)?\d+)\s*\)",
            block_body,
        )
        assert value_match is not None, f"{token} missing from {selector}"
        return float(value_match.group("alpha"))

    for selector in (":root", ".dark:root"):
        assert token_alpha(selector, "--shell-veil-soft") < token_alpha(
            selector,
            "--shell-veil",
        )
        assert token_alpha(selector, "--shell-veil") < token_alpha(
            selector,
            "--shell-veil-strong",
        )


def test_portal_html_externalizes_direct_debug_assets_without_third_party_hosts() -> None:
    html_content = _portal_html_content()
    css_content = _portal_css_content()
    bundle = orchestrator_app._get_portal_asset_bundle()

    assert (
        f'<link rel="preload" as="font" type="font/woff2" crossorigin href="{bundle.urls["fonts/portal-sans.woff2"]}" />'
        in html_content
    )
    assert f'href="{bundle.urls["portal.css"]}"' in html_content
    assert f'src="{bundle.urls["portal.js"]}"' in html_content
    assert f'data-review-surface-js-url="{bundle.urls["portal-review.js"]}"' in html_content
    assert f'data-review-surface-css-url="{bundle.urls["portal-review.css"]}"' in html_content
    assert "@import" not in css_content
    assert "--ux-target-min-size:" in css_content
    assert '<meta name="theme-color" content="#F4F7FB" media="(prefers-color-scheme: light)" />' in html_content
    assert '<meta name="theme-color" content="#020617" media="(prefers-color-scheme: dark)" />' in html_content
    assert "<style>" not in html_content
    assert "<script>" not in html_content
    assert "https://cdn.tailwindcss.com" not in html_content
    assert "https://fonts.googleapis.com" not in html_content
    assert "https://fonts.gstatic.com" not in html_content
    assert "tailwind.config" not in css_content
    assert ".text-\\[10px\\]" in css_content


def test_portal_html_signature_tracks_preloaded_font_fingerprint() -> None:
    signature = orchestrator_app._portal_html_signature()
    font_fingerprint = orchestrator_app._get_portal_direct_asset_fingerprint("fonts/portal-sans.woff2")

    assert ("fonts/portal-sans.woff2", font_fingerprint) in signature


def test_portal_css_signature_tracks_active_template_token_assets_only() -> None:
    signature = orchestrator_app._portal_css_signature()
    sans_fingerprint = orchestrator_app._get_portal_direct_asset_fingerprint("fonts/portal-sans.woff2")
    mono_fingerprint = orchestrator_app._get_portal_direct_asset_fingerprint("fonts/portal-mono.woff2")
    shared_token_fingerprint = orchestrator_app._get_portal_direct_asset_fingerprint("shared-ui-tokens.css")

    assert orchestrator_app._portal_css_dependency_asset_names() == (
        "fonts/portal-sans.woff2",
        "fonts/portal-mono.woff2",
    )
    assert ("fonts/portal-sans.woff2", sans_fingerprint) in signature
    assert ("fonts/portal-mono.woff2", mono_fingerprint) in signature
    assert ("shared-ui-tokens.css", shared_token_fingerprint) not in signature


def test_portal_runtime_helpers_read_served_js_assets_from_rendered_html() -> None:
    bundle = orchestrator_app._get_portal_asset_bundle()
    served_portal_js = _portal_js_content()
    served_review_js = _portal_review_bundle_content()
    runtime_content = _portal_runtime_bundle_content()

    assert served_portal_js == _portal_asset_path(bundle.urls["portal.js"]).read_text(encoding="utf-8")
    assert served_review_js == _portal_asset_path(bundle.urls["portal-review.js"]).read_text(encoding="utf-8")
    assert served_portal_js != _portal_js_source_content()
    assert "overviewStatsSkeletonState" in served_portal_js
    assert "overviewCapabilitySkeletonState" in served_portal_js
    assert (
        "_toggleSurfaceSkeleton(els.overviewStatsRow,els.overviewStatsRow,els.overviewStatsSkeletonState,bootstrapLoading)"
        in served_portal_js
    )
    assert (
        "_toggleSurfaceSkeleton(els.overviewCapabilityRow,els.overviewCapabilityRow,els.overviewCapabilitySkeletonState,bootstrapLoading)"
        in served_portal_js
    )
    assert "Review surfaces failed to load. Reload the portal and retry the review action." in runtime_content
    assert "deferredReviewSurfaceLoadFailedAt" in runtime_content
    assert "DEFERRED_REVIEW_SURFACE_RETRY_WINDOW_MS" in runtime_content
    assert "createDeferredReviewSurfaceApi" in served_review_js
    assert "Inline preview unavailable" in served_review_js


def test_advisory_caption_panel_uses_bounded_payload_cache() -> None:
    content = _portal_review_source_content()
    served_review_js = _portal_review_bundle_content()
    loader_body = _extract_js_function_body(content, "_loadAdvisoryCaptionPayload")
    panel_body = _extract_js_function_body(content, "_renderAdvisoryCaptionPanel")

    assert "const advisoryCaptionPayloadCache = new Map();" in content
    assert 'let advisoryCaptionCacheScope = "";' in content
    assert "advisoryCaptionPayloadCache" in served_review_js
    assert "const ADVISORY_CAPTION_CACHE_MAX_ENTRIES = 24;" in content
    assert "function _resetAdvisoryCaptionCacheForAuth(" in content
    assert "_advisoryCaptionCredentialSignature" in content
    assert "const requestHeaders = _buildAuthHeaders();" in loader_body
    assert "_resetAdvisoryCaptionCacheForAuth(requestHeaders);" in loader_body
    assert "const cacheKey = fetchUrl;" in loader_body
    assert "advisoryCaptionPayloadCache.delete(cacheKey);" in loader_body
    assert '_rememberAdvisoryCaptionCacheEntry(cacheKey, { status: "pending", promise });' in loader_body
    assert '_rememberAdvisoryCaptionCacheEntry(cacheKey, { status: "fulfilled", payload });' in loader_body
    assert "fetch(fetchUrl, {" in loader_body
    assert "headers: requestHeaders" in loader_body
    assert "fetch(" not in panel_body
    assert "_loadAdvisoryCaptionPayload(url)" in panel_body
    assert "requestId !== advisoryCaptionRenderRequestId" in panel_body


def test_portal_rum_rollout_reuses_shared_rollout_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = {"username": "portal-admin"}
    captured: list[tuple[str, dict[str, str]]] = []

    monkeypatch.setattr(orchestrator_app, "_env_bool", lambda name, default=False: name == "TP_PORTAL_RUM_ENABLED")
    monkeypatch.setattr(
        orchestrator_app,
        "_portal_rollout_enabled",
        lambda env_name, actor=None: captured.append((env_name, actor)) or True,
    )

    assert orchestrator_app._portal_rum_enabled(actor) is True
    assert captured == [("TP_PORTAL_RUM_ROLLOUT_PERCENT", actor)]


def test_portal_asset_manifest_is_explicit_and_repo_local() -> None:
    assert orchestrator_app.PORTAL_ASSET_MANIFEST_PATH.is_file()
    assert orchestrator_app.PORTAL_ASSET_PATHS == {
        "portal.css": orchestrator_app.PORTAL_ASSETS_DIR / "portal.css",
        "shared-ui-tokens.css": orchestrator_app.PORTAL_ASSETS_DIR / "shared-ui-tokens.css",
        "portal.js": orchestrator_app.PORTAL_ASSETS_DIR / "portal.js",
        "portal-review.js": orchestrator_app.PORTAL_ASSETS_DIR / "portal-review.js",
        "portal-review.css": orchestrator_app.PORTAL_ASSETS_DIR / "portal-review.css",
        "fonts/portal-sans.woff2": orchestrator_app.PORTAL_ASSETS_DIR / "fonts" / "portal-sans.woff2",
        "fonts/portal-mono.woff2": orchestrator_app.PORTAL_ASSETS_DIR / "fonts" / "portal-mono.woff2",
        "brand/dna-symbol-dark.svg": orchestrator_app.PORTAL_ASSETS_DIR / "brand" / "dna-symbol-dark.svg",
        "brand/dna-symbol-light.svg": orchestrator_app.PORTAL_ASSETS_DIR / "brand" / "dna-symbol-light.svg",
    }
    assert orchestrator_app.PORTAL_ASSET_MEDIA_TYPES == {
        "portal.css": "text/css; charset=utf-8",
        "shared-ui-tokens.css": "text/css; charset=utf-8",
        "portal.js": "text/javascript; charset=utf-8",
        "portal-review.js": "text/javascript; charset=utf-8",
        "portal-review.css": "text/css; charset=utf-8",
        "fonts/portal-sans.woff2": "font/woff2",
        "fonts/portal-mono.woff2": "font/woff2",
        "brand/dna-symbol-dark.svg": "image/svg+xml",
        "brand/dna-symbol-light.svg": "image/svg+xml",
    }
    for asset_path in orchestrator_app.PORTAL_ASSET_PATHS.values():
        assert asset_path.is_file()
        assert asset_path.is_relative_to(orchestrator_app.PORTAL_ASSETS_DIR)


def test_portal_asset_manifest_rejects_paths_outside_portal_assets_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(orchestrator_app, "PORTAL_ASSET_MANIFEST_PATH", manifest_path)

    with pytest.raises(RuntimeError, match="points outside"):
        orchestrator_app._load_portal_asset_manifest()

    assert portal_asset_bundle.PORTAL_ASSET_MANIFEST_PATH != manifest_path


def test_portal_html_asset_references_are_covered_by_manifest() -> None:
    bundle = orchestrator_app._get_portal_asset_bundle()
    html_asset_urls = _portal_asset_urls_from_html()
    bundled_asset_urls = html_asset_urls | _portal_asset_urls_from_css()
    manifest_asset_urls = {f"/portal/assets/{asset_name}" for asset_name in orchestrator_app.PORTAL_ASSET_MANIFEST.keys()}
    normalized_asset_urls = {urlparse(asset_url).path for asset_url in bundled_asset_urls}

    assert html_asset_urls
    assert {urlparse(asset_url).path for asset_url in html_asset_urls} <= manifest_asset_urls
    assert "/portal/assets/shared-ui-tokens.css" in manifest_asset_urls
    assert normalized_asset_urls == manifest_asset_urls - {"/portal/assets/shared-ui-tokens.css"}
    for asset_url in bundled_asset_urls:
        asset_name = _portal_asset_name(asset_url)
        assert _portal_asset_version(asset_url) == bundle.fingerprints[asset_name]
        _portal_asset_path(asset_url)


def test_portal_brand_asset_references_are_manifest_backed_and_repo_local() -> None:
    bundle = orchestrator_app._get_portal_asset_bundle()
    brand_asset_urls = {url for url in _portal_asset_urls_from_html() if url.startswith("/portal/assets/brand/")}
    manifest_asset_urls = {f"/portal/assets/{asset_name}" for asset_name in orchestrator_app.PORTAL_ASSET_MANIFEST.keys()}

    assert brand_asset_urls == {
        bundle.urls["brand/dna-symbol-dark.svg"],
        bundle.urls["brand/dna-symbol-light.svg"],
    }
    assert {urlparse(asset_url).path for asset_url in brand_asset_urls} <= manifest_asset_urls
    for asset_url in brand_asset_urls:
        assert _portal_asset_version(asset_url) == bundle.fingerprints[_portal_asset_name(asset_url)]
        assert _portal_asset_path(asset_url).is_relative_to(orchestrator_app.PORTAL_ASSETS_DIR)


@pytest.mark.parametrize("asset_name", ["dna-symbol-dark.svg", "dna-symbol-light.svg"])
def test_portal_brand_assets_match_frontdoor_sources(asset_name: str) -> None:
    frontdoor_asset = FRONTDOOR_BRAND_ROOT / asset_name
    portal_asset = PORTAL_ASSET_ROOT / "brand" / asset_name

    assert frontdoor_asset.is_file()
    assert portal_asset.is_file()

    frontdoor_bytes = frontdoor_asset.read_bytes()
    portal_bytes = portal_asset.read_bytes()
    frontdoor_sha = hashlib.sha256(frontdoor_bytes).hexdigest()
    portal_sha = hashlib.sha256(portal_bytes).hexdigest()

    assert (
        portal_bytes == frontdoor_bytes
    ), f"brand asset drift for {asset_name}: frontdoor={frontdoor_sha} portal={portal_sha}"


def test_portal_fetch_sse_reconnect_scheduler_has_terminal_guard_and_backoff() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "scheduleSseReconnect")

    assert "_isJobStreamRecoverable(job)" in body
    assert "if (job.reconnectBlocked) return;" in body
    assert "if (job.sseRetry.timer || _jobHasActiveStream(job)) return;" in body
    assert "SSE_RECONNECT_BASE_DELAY_MS" in body
    assert "setTimeout" in body
    assert "startJobEventStream(job, job.eventStreamUrl);" in body


def test_portal_bundle_embeds_internal_modules_without_changing_public_contracts() -> None:
    content = _portal_bundle_content()

    assert "const portalInternals = __PortalInternal;" in content
    assert "const portalRoute = portalInternals.createPortalRouteHelpers(window);" in content
    assert "const portalDom = portalInternals.createDomContract(document, {" in content
    assert "const _domId = (id, required = false) => portalDom.id(id, { required });" in content
    assert "config: portalInternals.createPortalConfigState()," in content
    assert "auth: portalInternals.createPortalAuthState()," in content
    assert "bootstrap: portalInternals.createPortalBootstrapState(Date.now())," in content
    assert "rum: portalInternals.createPortalRumState()" in content
    assert "portalDom.assertPresent(els, [" in content
    assert "portalRenderSurfaces.register('jobQueue'" in content


def test_portal_rum_wires_native_observers_and_keepalive_posts() -> None:
    content = _portal_bundle_content()
    observer_body = _extract_js_function_body(content, "_startPortalRumObservers")
    flush_body = _extract_js_function_block(content, "_flushQueuedPortalRumSamples")
    finalize_body = _extract_js_function_body(content, "_finalizePortalRumVitals")

    assert "if (state.rum.observersStarted || typeof window.PerformanceObserver !== 'function') return;" in observer_body
    assert "new PerformanceObserver" in observer_body
    assert "type: 'largest-contentful-paint'" in observer_body
    assert "type: 'layout-shift'" in observer_body
    assert "type: 'event'" in observer_body
    assert "durationThreshold: 16" in observer_body
    assert "fetch(`${API_BASE}/v1/portal/rum`" in flush_body
    assert "keepalive: keepalive || sample.keepalive" in flush_body
    assert "traceparent: sample.traceparent" in flush_body
    assert "eventType: 'core_web_vital'" in finalize_body
    assert "metric: 'lcp'" in finalize_body
    assert "metric: 'inp'" in finalize_body
    assert "metric: 'cls'" in finalize_body


def test_portal_rum_emits_once_per_page_load_and_keys_samples_by_route_view() -> None:
    content = _portal_bundle_content()
    base_payload_body = _extract_js_function_block(content, "_portalRumBasePayload")
    record_body = _extract_js_function_block(content, "_recordPortalRumMilestone")
    first_view_body = _extract_js_function_body(content, "_scheduleFirstViewInteractiveRum")

    assert "route: '/portal'" in base_payload_body
    assert "view: portalInternals.normalizePortalRumView(sampleOptions.view || state.currentView)" in base_payload_body
    assert "if (state.rum.emittedMilestones[eventType]) return;" in record_body
    assert "state.rum.emittedMilestones[eventType] = true;" in record_body
    assert "_queuePortalRumSample({" in record_body
    assert "metric: 'duration'" in record_body
    assert "window.requestAnimationFrame(emit);" in first_view_body
    assert "window.setTimeout(emit, 0);" in first_view_body
    assert "_recordPortalRumMilestone('first_view_interactive', _portalRumNow()," in first_view_body
    assert "_recordPortalRumMilestone('portal_shell_rendered', _portalRumNow()," in content
    assert "_recordPortalRumMilestone('bootstrap_ready', _portalRumNow()," in content
    assert "_recordPortalRumMilestone('first_view_interactive', _portalRumNow()," in content


def test_portal_rum_tracks_queue_actions_and_sse_reconnects() -> None:
    content = _portal_bundle_content()
    submit_body = _extract_js_function_body(content, "submitJob")
    cancel_body = _extract_js_function_body(content, "cancelJob")
    fetch_sse_body = _extract_js_function_body(content, "_startAuthorizedFetchSse")
    native_sse_body = _extract_js_function_body(content, "startJobEventStream")

    assert "eventType: 'queue_request'" in submit_body
    assert "metric: 'submit'" in submit_body
    assert "eventType: 'queue_request'" in cancel_body
    assert "metric: 'cancel'" in cancel_body
    assert "eventType: 'sse_reconnect'" in fetch_sse_body
    assert "eventType: 'sse_reconnect'" in native_sse_body


def test_portal_fetch_sse_reconnect_schedules_on_unexpected_disconnect_only() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "_startAuthorizedFetchSse")

    assert "let sawDoneEvent = false;" in body
    assert "let shouldReconnect = true;" in body
    assert "const suppressed = await _maybeSuppressOnProtectedResponse('jobs_events', res);" in body
    assert "const isAuthError = status === 401 || status === 403;" in body
    assert "const isRetryableStatus = status === 429 || status >= 500;" in body
    assert "shouldReconnect = isRetryableStatus && !suppressed;" in body
    assert "if (suppressed || !isRetryableStatus) {" in body
    assert "job.reconnectBlocked = true;" in body
    assert "if (shouldReconnect && !sawDoneEvent && !controller.signal.aborted && _isJobStreamRecoverable(job)) {" in body
    assert "scheduleSseReconnect(job);" in body


def test_portal_sse_watchdog_reconnects_stalled_streams_for_fetch_and_native_transports() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "startSseWatchdog")

    assert "if (job.reconnectBlocked) return;" in body
    assert "SSE_STALL_THRESHOLD_MS" in body
    assert "Fetch event stream is not active. Reconnecting to restore live telemetry." in body
    assert "Native SSE stream is not active. Reconnecting to restore live telemetry." in body
    assert "Fetch event stream stalled. Reconnecting to restore live telemetry." in body
    assert "Native SSE stream stalled. Reconnecting to restore live telemetry." in body
    assert "_teardownJobEventStream(job);" in body
    assert "scheduleSseReconnect(job);" in body


def test_portal_start_job_stream_avoids_duplicate_readers() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "startJobEventStream")

    assert "if (_isProtectedFamilySuppressed('jobs_events')) {" in body
    assert "_clearSseRetry(job, false);" in body
    assert "if (_jobHasActiveStream(job)) return;" in body


def test_portal_eventsource_wraps_json_parse_in_try_catch() -> None:
    """Verify EventSource handlers guard against malformed JSON data."""
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "startJobEventStream")

    # The EventSource handlers must use a safe parsing wrapper
    assert "safeParseSseEvent" in body
    assert "try {" in body
    assert "JSON.parse(e.data)" in body
    assert "catch {" in body
    # Verify all event types use the safe wrapper
    assert "es.addEventListener('log', (e) => safeParseSseEvent('log', e));" in body
    assert "es.addEventListener('progress', (e) => safeParseSseEvent('progress', e));" in body
    assert "es.addEventListener('state', (e) => safeParseSseEvent('state', e));" in body
    assert "es.addEventListener('artifact', (e) => safeParseSseEvent('artifact', e));" in body
    assert "es.addEventListener('done', (e) => safeParseSseEvent('done', e));" in body


def test_portal_native_eventsource_surfaces_state_and_terminal_errors() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "startJobEventStream")
    active_body = _extract_js_function_body(content, "_jobHasActiveStream")
    transport_body = _extract_js_function_body(content, "formatTransportLabel")

    assert "const EVENT_SOURCE_READY_STATE_CONNECTING = 0;" in content
    assert "const EVENT_SOURCE_READY_STATE_OPEN = 1;" in content
    assert "const EVENT_SOURCE_READY_STATE_CLOSED = 2;" in content
    assert "function _isNativeEventSourceHandle(handle) {" in content
    assert "function _nativeEventSourceReadyState(handle) {" in content
    assert "const nativeReadyState = _nativeEventSourceReadyState(job.eventSource);" in active_body
    assert "if (nativeReadyState === EVENT_SOURCE_READY_STATE_CLOSED) return false;" in active_body
    assert "if (nativeReadyState === EVENT_SOURCE_READY_STATE_CONNECTING) return 'event reconnecting';" in transport_body
    assert "if (nativeReadyState === EVENT_SOURCE_READY_STATE_OPEN) return 'event stream';" in transport_body
    assert "if (nativeReadyState === EVENT_SOURCE_READY_STATE_CLOSED) return 'event closed';" in transport_body
    assert "es.onopen = () => {" in body
    assert "const readyState = _nativeEventSourceReadyState(es);" in body
    assert "if (readyState === EVENT_SOURCE_READY_STATE_CONNECTING) {" in body
    assert (
        "_noteTransportWarning(job, 'eventsource_reconnecting', 'Native SSE connection dropped. Browser is retrying in the background.', 'warn');"
        in body
    )
    assert (
        "const warningCode = readyState === EVENT_SOURCE_READY_STATE_CLOSED ? 'eventsource_closed' : 'eventsource_error';"
        in body
    )
    assert "appendJobLog(job, logLine);" in body
    assert "_teardownJobEventStream(job);" in body
    assert "scheduleSseReconnect(job);" in body


def test_portal_resumes_blocked_streams_after_api_key_update() -> None:
    content = _portal_bundle_content()
    helper_body = _extract_js_function_body(content, "resumeBlockedJobStreamsAfterAuthUpdate")
    api_key_update_body = _extract_js_function_body(content, "_handleDirectDebugApiKeyUpdate")
    bind_body = _extract_js_function_body(content, "bindInputs")

    assert "if (!job.reconnectBlocked) return;" in helper_body
    assert "startJobEventStream(job, job.eventStreamUrl);" in helper_body
    assert "resumeBlockedJobStreamsAfterAuthUpdate();" in api_key_update_body
    assert "_resetProtectedFamilySuppression();" in api_key_update_body
    assert "_handleDirectDebugApiKeyUpdate({ resumeStreams: true });" in bind_body


def test_portal_bootstrap_loader_uses_abortable_timeout_and_state_contract() -> None:
    content = _portal_bundle_content()
    default_body = _extract_js_function_body(content, "_defaultPortalBootstrap")
    login_url_body = _extract_js_function_body(content, "_managedLoginUrlForCurrentRoute")
    body = _extract_js_function_body(content, "loadPortalBootstrap")
    failure_details_body = _extract_js_function_body(content, "_bootstrapFailureDetails")
    followup_body = _extract_js_function_body(content, "_flushBootstrapOnlineFollowup")
    normalize_body = _extract_js_function_body(content, "_normalizeFetchFailureReason")
    retryable_body = _extract_js_function_body(content, "_isBootstrapRetryableFailure")
    retry_body = _extract_js_function_body(content, "_scheduleBootstrapRetry")
    delay_body = _extract_js_function_body(content, "_nextBootstrapRetryDelayMs")

    assert "authMode: 'managed_unavailable'" in default_body
    assert "apiKeyInput: false" in default_body
    assert "directDebug: false" in default_body
    assert "artifactViewerModal: false" in default_body
    assert "reviewSurfaceDeferred: false" in default_body
    assert "stagedUploads: false" in default_body
    assert "rumTelemetry: false" in default_body
    assert "fastVlmCaptioning: false" in default_body
    assert "const BOOTSTRAP_TIMEOUT_MS = 3500;" in content
    assert "const BOOTSTRAP_RETRY_BASE_DELAY_MS = 1000;" in content
    assert "const BOOTSTRAP_RETRY_MAX_DELAY_MS = 12000;" in content
    assert "const BOOTSTRAP_RETRY_MAX_ATTEMPTS = 4;" in content
    assert "const BOOTSTRAP_RETRY_MAX_WINDOW_MS = 60000;" in content
    assert "const TRANSIENT_DRAFT_STORAGE_KEY = 'tp_portal_transient_draft';" in content
    assert "const TRANSIENT_DRAFT_SCHEMA = 'tp.portal.transient_draft.v1';" in content
    assert "const BOOTSTRAP_RETRIABLE_HTTP_STATUSES = new Set([500, 502, 503, 504]);" in content
    assert "fetchWithTimeout(" in body
    assert "return `/login?returnTo=${encodeURIComponent(_managedReturnToPath())}`;" in login_url_body
    assert "`${API_BASE}/portal/bootstrap`" in body
    assert "BOOTSTRAP_TIMEOUT_MS" in body
    assert "const bootstrapOptions = options && typeof options === 'object' ? options : null;" in body
    assert "const retryAttempt = Number.isInteger(bootstrapOptions && bootstrapOptions.attempt)" in body
    assert "const retryReason = String(bootstrapOptions && bootstrapOptions.retryReason" in body
    assert "const isRetryAttempt = Boolean((bootstrapOptions && bootstrapOptions.isRetryAttempt) || retryAttempt > 0);" in body
    assert "const bootstrapCancelReason = isRetryAttempt" in body
    assert "_cancelPendingBootstrapRequest(bootstrapCancelReason);" in body
    assert "_applyPortalBootstrap(fallback, { status: 'pending' });" in body
    assert "onStart: _trackBootstrapRequest" in body
    assert "onFinally: _clearTrackedBootstrapRequest" in body
    assert "if (res.status === 401 || res.status === 403)" in body
    assert "let payload = null;" in body
    assert "let payloadParsed = false;" in body
    assert "const failure = _bootstrapFailureDetails(" in body
    assert "_finalizeBootstrapRetry('terminal_auth_redirect', { reason: failure.reason, httpStatus: res.status });" in body
    assert "window.location.assign(_managedLoginUrlForCurrentRoute());" in body
    assert "const status = failure.retryable ? 'degraded' : 'unavailable';" in body
    assert "const retryScheduled = failure.retryable && _scheduleBootstrapRetry(failure.reason, res.status);" in body
    assert "const retryScheduled = failure.retryable && _scheduleBootstrapRetry(failure.reason, 0);" in body
    assert "_finalizeBootstrapRetry('terminal_invalid_json', { reason: 'invalid_json' });" in body
    assert "_finalizeBootstrapRetry('succeeded', {" in body
    assert "_applyPortalBootstrap(payload, { status: 'ready', traceparent: bootstrapTraceparent });" in body
    assert "artifactViewerModal: Boolean(bootstrap.features?.artifactViewerModal)" in content
    assert "stagedUploads: Boolean(bootstrap.features?.stagedUploads)" in content
    assert "rumTelemetry: Boolean(bootstrap.features?.rumTelemetry)" in content
    assert "fastVlmCaptioning: Boolean(bootstrap.features?.fastVlmCaptioning)" in content
    assert "res.headers.get('traceparent')" in body
    assert "previousHealthEndpointPath !== nextHealthEndpointPath" in body
    assert "_queueBootstrapOnlineFollowup();" in body
    assert "_flushBootstrapOnlineFollowup();" in body
    assert "_normalizeFetchFailureReason(error, 'bootstrap_timeout')" in body
    assert (
        "async function fetchWithTimeout(url, options = {}, timeoutMs = HEALTH_CHECK_TIMEOUT_MS, timeoutReason = 'request_timeout', lifecycle = null)"
        in content
    )
    assert "timeoutError.name = 'AppTimeoutError';" in content
    assert "timeoutError.reason = timeoutReason;" in content
    assert "if (!_isBootstrapReady()) {" in followup_body
    assert "_queueBootstrapOnlineFollowup();" in followup_body
    assert "void fetchPresetsForPipeline(state.pipeline, true);" in followup_body
    assert "void recoverJobs();" in followup_body
    assert "normalizedReason === 'auth_failure' || normalizedReason === 'auth'" in failure_details_body
    assert "reason: 'access_outage'" in failure_details_body
    assert "reason: 'config_failure'" in failure_details_body
    assert "reason: 'invalid_json'" in failure_details_body
    assert (
        "reason: normalizedReason === 'timeout' || normalizedReason === 'network' ? normalizedReason : 'upstream_unavailable'"
        in failure_details_body
    )
    assert "reason === String(timeoutReason || '').trim().toLowerCase()" in normalize_body
    assert "name === 'timeouterror'" in normalize_body
    assert "name === 'aborterror'" in normalize_body
    assert "return _bootstrapFailureDetails(reason, httpStatus).retryable;" in retryable_body
    assert "Math.random()" in delay_body
    assert "BOOTSTRAP_RETRY_MAX_EXPONENT" in delay_body
    assert "BOOTSTRAP_RETRY_MAX_DELAY_MS" in delay_body
    assert "if (state.bootstrap.retry.timer !== null) {" in retry_body
    assert "skipped_already_scheduled" in retry_body
    assert "attempt > BOOTSTRAP_RETRY_MAX_ATTEMPTS" in retry_body
    assert "(now + delayMs) > state.bootstrap.retry.deadlineAt" in retry_body
    assert "window.setTimeout(() => {" in retry_body
    assert "void loadPortalBootstrap({ isRetryAttempt: true, attempt, retryReason: reason });" in retry_body


def test_portal_protected_suppression_reads_managed_error_envelope_details() -> None:
    content = _portal_bundle_content()
    details_body = _extract_js_function_body(content, "_protectedErrorDetailsFromPayload")
    suppress_body = _extract_js_function_body(content, "_maybeSuppressOnProtectedResponse")
    api_key_update_body = _extract_js_function_body(content, "_handleDirectDebugApiKeyUpdate")
    recover_body = _extract_js_function_body(content, "recoverJobs")
    upload_body = _extract_js_function_body(content, "_submitStagedUploadSelection")

    assert "payload.error.details" in details_body
    assert "const details = _nonRetryableProtectedDetails(body);" in suppress_body
    assert "_resetProtectedFamilySuppression();" in api_key_update_body
    assert "void fetchConfigMetadata(state.pipeline, true);" in api_key_update_body
    assert "void fetchConfigPreview(generatePayload());" in api_key_update_body
    assert "if (_isProtectedFamilySuppressed('jobs_list')) return;" in recover_body
    assert "await _maybeSuppressOnProtectedResponse('jobs_list', res);" in recover_body
    assert "if (_isProtectedFamilySuppressed('uploads_staging')) {" in upload_body
    assert "_recordProtectedFamilySuppression('uploads_staging', nonRetryableDetails);" in upload_body


def test_portal_staged_upload_ui_contract_is_present_in_markup_and_source() -> None:
    html = _portal_html_content()
    content = _portal_bundle_content()

    assert 'id="stagedUploadShell"' in html
    assert 'data-ui="staged-upload-shell"' in html
    assert 'id="stagedUploadDropzone"' in html
    assert 'data-ui="staged-upload-dropzone"' in html
    assert 'role="button"' in html
    assert 'aria-label="Choose files or drop files for staged upload"' in html
    assert 'aria-describedby="stagedUploadStatus"' in html
    assert 'id="stagedUploadFilesInput"' in html
    assert 'id="stagedUploadFolderInput"' in html
    assert "const STAGED_UPLOAD_SUPPORTED_PIPELINES = new Set(['lux-depth-v3', 'archive-gate-a']);" in content
    assert "function _stagedUploadsVisibleForState() {" in content
    assert "_buildAuthHeaders({}, 'POST', { traceparent: requestTraceparent });" in content
    assert "xhr.open('POST', `${API_BASE}/v1/uploads/staging`);" in content
    assert "formData.append('files', file, relativePath);" in content
    assert "els.inputDir.dispatchEvent(new Event('input', { bubbles: true }));" in content
    assert "els.inputDir.dispatchEvent(new Event('change', { bubbles: true }));" in content
    assert "function _renderStagedUploadSummary(container, uploadState) {" in content
    assert "Upload receipt" in content
    assert "Inline failures" in content


def test_portal_managed_mode_clears_api_keys_and_hides_secret_ui() -> None:
    content = _portal_bundle_content()
    clear_body = _extract_js_function_body(content, "_clearStoredApiKeyState")
    summary_body = _extract_js_function_body(content, "_bootstrapSurfaceSummary")
    sync_body = _extract_js_function_body(content, "_syncBootstrapUi")

    assert "localStorage.removeItem(API_KEY_STORAGE_KEY);" in clear_body
    assert "sessionStorage.removeItem(API_KEY_STORAGE_KEY);" in clear_body
    assert "_clearStoredApiKeyState(true);" in content
    assert "_loadApiKeyIntoInputs();" in content
    assert 'id="connectionDetails"' in content
    assert 'data-ui="connection-details"' in content
    assert 'id="portalAccessState"' in content
    assert 'id="bootstrapStatusBadge"' in content
    assert 'id="bootstrapRecoveryHint"' in content
    assert "badge: 'Managed access'" in summary_body
    assert "badge: 'Direct debug'" in summary_body
    assert "badge: failure.retryable ? 'Recovery pending' : 'Recovery required'" in summary_body
    assert "badge: 'Confirming access'" in summary_body
    assert "const showApiKeyInput = bootstrapReady && state.auth.features.apiKeyInput;" in sync_body
    assert "els.apiKeySection.classList.toggle('hidden', !showApiKeyInput);" in sync_body
    assert "document.body.dataset.bootstrapReason = String(state.bootstrap.lastErrorReason || '');" in sync_body
    assert "document.body.dataset.authMode = String(state.auth.mode || 'managed_unavailable');" in sync_body
    assert "els.portalAccessState.dataset.bootstrapStatus = String(state.bootstrap.status || 'pending');" in sync_body
    assert "els.bootstrapStatusBadge.textContent = summary.badge;" in sync_body
    assert "els.bootstrapRecoveryHint.textContent = summary.detail;" in sync_body
    assert "els.apiKeyInput.disabled = !showApiKeyInput;" in sync_body
    assert "rememberApiKey" not in content


def test_portal_fastvlm_captioning_controls_are_feature_gated_and_advisory_only() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()
    canonical_body = _extract_js_function_body(content, "buildCanonicalLuxDepthArgs")
    applicability_body = _extract_js_function_body(content, "syncBuildSurfaceApplicability")
    cli_body = _extract_js_function_body(content, "renderCLI")
    diagnostics_body = _extract_js_function_body(content, "renderPreRunDiagnostics")
    effective_drawer_body = _extract_js_function_body(content, "renderEffectiveConfigDrawer")
    captioning_status_body = _extract_js_function_body(content, "normalizeCaptioningRunStatus")
    run_summary_body = _extract_js_function_body(content, "normalizeRunSummary")
    queue_body = _extract_js_function_body(content, "renderJobQueue")
    bind_body = _extract_js_function_body(content, "bindInputs")

    assert 'id="captioningDetails"' in content
    assert 'data-ui="captioning-controls"' in content
    assert 'id="enableFastVlmCaptioning"' in content
    assert 'id="fastVlmCaptioningModel"' in content
    assert 'id="fastVlmProxyFormat"' in content
    assert 'id="fastVlmMaxSidePx"' in content
    assert 'id="fastVlmTimeoutSeconds"' in content
    assert 'id="fastVlmPythonExecutable"' in content
    assert 'id="fastVlmMlxVlmDir"' in content
    assert 'id="captioningReadinessScope"' in content
    assert 'id="captioningReadinessList"' in content
    assert 'id="reviewProvenanceCaptioning"' in content
    assert 'data-ui="captioning-readiness"' in content
    assert 'data-ui="captioning-run-status"' in content
    assert '"captioning-evidence-strip"' in review_content
    assert '"captioning-evidence-link"' in review_content
    assert '"captioning-sidecar-link"' in review_content
    assert '"captioning-raw-link"' in review_content
    assert '"captioning-proxy-link"' in review_content
    assert "FastVLM captions are advisory and never satisfy quality gates." in content
    assert "FastVLM readiness: Off" in content
    assert "function _fastVlmCaptioningFeatureEnabled()" in content
    assert "const CAPTIONING_RUN_STATUS_VALUES = new Set([" in content
    assert "function toNonNegativeCaptioningRunStatusInt(value)" in content
    assert "function normalizeCaptioningRunStatus(rawStatus)" in content
    assert "function captioningRunStatusSummary(status)" in content
    assert "function captioningRunStatusDetail(status)" in content
    assert "function createCaptioningRunStatusChip(status)" in content
    assert "function _captioningRuntimeReadiness(summary = {})" in content
    assert "function _renderCaptioningReadiness(summary = {}, options = {})" in content
    assert "function _captioningEvidenceArtifacts(job, artifact)" in review_content
    assert "function _captioningStatusFromEvidence(status, artifacts)" in review_content
    assert "function _renderCaptioningEvidenceStrip(job, artifact)" in review_content
    assert "function _renderAdvisoryCaptionUnavailable(panel)" in review_content
    assert (
        'return type === "vlm_caption_proxy" || /(^|\\/)captioning\\/.*_proxy\\.(png|jpe?g)$/i.test(relPath);'
        in review_content
    )
    assert "const stemMatchedArtifacts = selectedStem" in review_content
    assert (
        "const candidates = selectedStem && stemMatchedArtifacts.length === 0 ? captionArtifacts : stemMatchedArtifacts;"
        in review_content
    )
    assert 'status: "succeeded",' in review_content
    assert "state.auth?.features?.fastVlmCaptioning" in content
    assert "const captioningFeatureVisible = isLuxPipeline && _fastVlmCaptioningFeatureEnabled();" in applicability_body
    assert "_setContextVisibility(els.captioningDetails, captioningFeatureVisible);" in applicability_body
    assert "state.config.captioning.enableFastVlm = false;" in applicability_body
    assert "runtimeStatus === 'missing_runtime'" in applicability_body
    assert "runtimeStatus === 'invalid_config'" in applicability_body
    assert "_renderCaptioningReadiness(summary, {" in applicability_body
    assert "if (captioningFeatureEnabled) {" in canonical_body
    assert "args.vlm_captioning_enabled = enableFastVlmCaptioning;" in canonical_body
    assert "args.vlm_captioning_backend = 'fastvlm';" in canonical_body
    assert "parseBoolLike(payload.args.vlm_captioning_enabled, false)" in cli_body
    assert "--vlm-captioning" in cli_body
    assert "--fastvlm-python" in cli_body
    assert "--fastvlm-mlx-vlm-dir" in cli_body
    assert "Advisory FastVLM caption sidecars" in diagnostics_body
    assert "new Set([" not in captioning_status_body
    assert "FastVLM captions are advisory sidecar metadata and do not satisfy quality gates." in diagnostics_body
    assert "FastVLM advisory captioning is enabled and remains outside quality gates." in effective_drawer_body
    assert "FastVLM readiness is" in effective_drawer_body
    assert "captioningReadiness.verification_scope" in effective_drawer_body
    assert "captioning_status: normalizeCaptioningRunStatus(rawSummary.captioning_status)" in run_summary_body
    assert "createCaptioningRunStatusChip(captioningRunStatus)" in queue_body
    assert "chip.dataset.sidecarCount" in content
    assert "chip.dataset.rawCount" in content
    assert "chip.dataset.proxyCount" in content
    assert "chip.setAttribute('aria-label', captioningRunStatusDetail(status));" in content
    assert "'captioning:enableFastVlm': 'vlm_captioning_enabled'" in bind_body
    assert "safeBindCheck(els.captioning.enableFastVlm, 'captioning', 'enableFastVlm');" in bind_body


def test_portal_direct_debug_api_key_storage_is_session_only() -> None:
    content = _portal_bundle_content()
    persist_body = _extract_js_function_body(content, "_persistApiKeyFromInputs")
    load_body = _extract_js_function_body(content, "_loadApiKeyIntoInputs")
    current_token_body = _extract_js_function_body(content, "_currentApiToken")

    assert "localStorage.setItem(API_KEY_STORAGE_KEY, token);" not in persist_body
    assert "sessionStorage.setItem(API_KEY_STORAGE_KEY, token);" in persist_body
    assert "localStorage.removeItem(API_KEY_STORAGE_KEY);" in persist_body
    assert "const localValue = localStorage.getItem(API_KEY_STORAGE_KEY) || '';" in load_body
    assert "const sessionValue = sessionStorage.getItem(API_KEY_STORAGE_KEY) || '';" in load_body
    assert "const stored = sessionValue || localValue;" in load_body
    assert "if (localValue && !sessionValue) {" in load_body
    assert "sessionStorage.setItem(API_KEY_STORAGE_KEY, localValue);" in load_body
    assert "localStorage.removeItem(API_KEY_STORAGE_KEY);" in load_body
    assert "localStorage.getItem(API_KEY_STORAGE_KEY)" not in current_token_body
    assert "sessionStorage.getItem(API_KEY_STORAGE_KEY)" in current_token_body


def test_portal_managed_mode_uses_csrf_instead_of_browser_backend_secrets() -> None:
    content = _portal_bundle_content()
    current_token_body = _extract_js_function_body(content, "_currentApiToken")
    auth_headers_block = _extract_js_function_block(content, "_buildAuthHeaders")

    assert "if (_isManagedAuthMode()) return '';" in current_token_body
    assert "function _buildAuthHeaders(base = {}, method = 'GET', options = null) {" in auth_headers_block
    assert "headers.traceparent = traceparent;" in auth_headers_block
    assert "if (_isManagedAuthMode()) {" in auth_headers_block
    assert "headers['X-CSRF-Token'] = state.auth.csrfToken;" in auth_headers_block
    assert "headers['Authorization'] = `Bearer ${token}`;" in auth_headers_block
    assert "headers['x-api-key'] = token;" in auth_headers_block

    managed_guard_index = auth_headers_block.index("if (_isManagedAuthMode()) {")
    csrf_header_index = auth_headers_block.index("headers['X-CSRF-Token'] = state.auth.csrfToken;")
    managed_return_index = auth_headers_block.index("return headers;", managed_guard_index)
    authorization_header_index = auth_headers_block.index("headers['Authorization'] = `Bearer ${token}`;")
    api_key_header_index = auth_headers_block.index("headers['x-api-key'] = token;")

    assert managed_guard_index < csrf_header_index < managed_return_index
    assert managed_return_index < authorization_header_index < api_key_header_index


def test_portal_auth_helpers_fail_closed_until_bootstrap_ready() -> None:
    content = _portal_bundle_content()
    persist_body = _extract_js_function_body(content, "_persistApiKeyFromInputs")
    load_body = _extract_js_function_body(content, "_loadApiKeyIntoInputs")
    current_token_body = _extract_js_function_body(content, "_currentApiToken")

    assert 'data-bootstrap-status="pending"' in content
    assert "if (!_isBootstrapReady()) {" in persist_body
    assert "_clearStoredApiKeyState(false);" in persist_body
    assert "if (!_isBootstrapReady()) {" in load_body
    assert "_clearStoredApiKeyState(false);" in load_body
    assert "if (!_isBootstrapReady()) return '';" in current_token_body
    assert "function _buildAuthHeaders(base = {}, method = 'GET', options = null) {" in content
    assert "if (!_isBootstrapReady()) {" in content
    assert "return headers;" in content


def test_portal_health_checks_route_to_front_door_in_managed_mode() -> None:
    content = _portal_bundle_content()
    helper_body = _extract_js_function_body(content, "_healthEndpointPath")
    check_body = _extract_js_function_body(content, "checkBackend")

    assert "return _isManagedAuthMode() ? '/healthz' : '/ready';" in helper_body
    assert "const healthEndpointPath = _healthEndpointPath();" in check_body
    assert "state.bootstrap.lastHealthEndpointPath = healthEndpointPath;" in check_body
    assert "fetchWithTimeout(`${API_BASE}${healthEndpointPath}`" in check_body
    assert "_queueBootstrapOnlineFollowup();" in check_body
    assert "_flushBootstrapOnlineFollowup(force);" in check_body


def test_portal_transient_draft_restore_is_scoped_to_the_current_owner_and_excludes_auth_state() -> None:
    content = _portal_bundle_content()
    route_body = _extract_js_function_body(content, "_managedReturnToPath")
    owner_body = _extract_js_function_body(content, "_transientDraftOwnerKey")
    managed_owner_body = _extract_js_function_body(content, "_managedDraftOwnerKey")
    clear_body = _extract_js_function_body(content, "_clearTransientPortalDraft")
    read_body = _extract_js_function_body(content, "_readTransientPortalDraft")
    persist_body = _extract_js_function_body(content, "_persistTransientPortalDraft")
    schedule_body = _extract_js_function_body(content, "_scheduleTransientPortalDraftPersist")
    flush_body = _extract_js_function_body(content, "_flushPendingTransientPortalDraftPersist")
    restore_body = _extract_js_function_body(content, "_restoreTransientPortalDraft")

    assert "const url = new URL(window.location.href);" in route_body
    assert "return `${url.pathname}${url.search}`;" in route_body
    assert "if (_isManagedAuthMode()) return _managedDraftOwnerKey();" in owner_body
    assert "return 'direct_debug';" in owner_body
    assert "managed:" in managed_owner_body
    assert "sessionStorage.removeItem(TRANSIENT_DRAFT_STORAGE_KEY);" in clear_body
    assert "sessionStorage.getItem(TRANSIENT_DRAFT_STORAGE_KEY)" in read_body
    assert "schema !== TRANSIENT_DRAFT_SCHEMA" in read_body
    assert "config: _copyTransientDraftConfig(config)" in read_body
    assert "sessionStorage.setItem(TRANSIENT_DRAFT_STORAGE_KEY, JSON.stringify(snapshot));" in persist_body
    assert "schema: TRANSIENT_DRAFT_SCHEMA" in persist_body
    assert "ownerKey" in persist_body
    assert "savedAt: Date.now()" in persist_body
    assert "buildStep: resolveBuildStep(state.portalUi.buildStep)" in persist_body
    assert "window.setTimeout(scheduleCommit, TRANSIENT_DRAFT_PERSIST_DEBOUNCE_MS);" in schedule_body
    assert "window.requestIdleCallback" in schedule_body
    assert "_persistTransientPortalDraft();" in schedule_body
    assert "_cancelScheduledTransientPortalDraftPersist();" in flush_body
    assert "return _persistTransientPortalDraft();" in flush_body
    assert "API_KEY_STORAGE_KEY" not in persist_body
    assert "csrfToken" not in persist_body
    assert "state.jobs" not in persist_body
    assert "snapshot.ownerKey !== ownerKey" in restore_body
    assert "_clearTransientPortalDraft();" in restore_body
    assert "state.pipeline = snapshot.pipeline;" in restore_body
    assert "state.config = _copyTransientDraftConfig(snapshot.config);" in restore_body
    assert "state.portalUi.buildStep = resolveBuildStep(snapshot.buildStep);" in restore_body


def test_portal_transient_draft_restores_before_preview_and_readiness_hydration() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "init")
    bind_body = _extract_js_function_body(content, "bindInputs")
    set_build_step_body = _extract_js_function_body(content, "setBuildStep")

    assert "_restoreTransientPortalDraft();" in body
    assert "_persistTransientPortalDraft();" in body
    assert "_scheduleTransientPortalDraftPersist();" in bind_body
    assert "_scheduleTransientPortalDraftPersist({ immediate: true });" in bind_body
    assert "_persistTransientPortalDraft();" in set_build_step_body
    assert "window.addEventListener('pagehide', _flushPendingTransientPortalDraftPersist);" in content
    assert "window.addEventListener('beforeunload', _flushPendingTransientPortalDraftPersist);" in content

    restore_index = body.index("_restoreTransientPortalDraft();")
    update_index = body.index("updateUIFromState();", restore_index)
    check_index = body.index("void checkBackend(true);")
    metadata_index = body.index("void fetchConfigMetadata(state.pipeline, true);")

    assert restore_index < update_index < check_index < metadata_index


def test_portal_managed_unavailable_mode_blocks_dispatch_and_api_key_recovery_prompts() -> None:
    content = _portal_bundle_content()
    block_body = _extract_js_function_body(content, "_blockManagedUnavailableAction")
    cancel_body = _extract_js_function_body(content, "cancelJob")
    submit_body = _extract_js_function_body(content, "submitJob")
    sse_body = _extract_js_function_body(content, "_startAuthorizedFetchSse")

    assert "_isManagedUnavailableMode()" in block_body
    assert "_bootstrapFailureDetails(state.bootstrap.lastErrorReason, state.bootstrap.lastHttpStatus)" in block_body
    assert "Unable to ${actionLabel} until recovery completes." in block_body
    assert "_blockManagedUnavailableAction('change job state')" in cancel_body
    assert "_blockManagedUnavailableAction('dispatch jobs')" in submit_body
    assert "Restore the managed session to resume live job events." in sse_body
    assert "Restore the managed session to resume live logs." in sse_body


def test_portal_artifact_gallery_renders_visual_review_controls() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()
    review_body = _extract_js_function_body(review_content, "renderArtifactPanel")
    reset_body = _extract_js_function_body(content, "_resetArtifactActionButtons")
    sanitize_body = _extract_js_function_body(content, "sanitizeManagedAssetUrl")
    open_artifact_body = _extract_js_function_body(content, "_openArtifactForSelection")
    rank_body = _extract_js_function_body(content, "rankArtifactsForDisplay")
    compare_body = _extract_js_function_body(content, "findCompareArtifact")
    normalize_body = _extract_js_function_body(content, "normalizeArtifactItems")
    route_key_body = _extract_js_function_body(content, "_artifactRouteKey")

    assert 'id="artifactPreviewStage"' in content
    assert 'id="artifactThumbnailRail"' in content
    assert 'id="artifactSelectionMeta"' in content
    assert 'id="openArtifactBtn"' in content
    assert 'id="downloadArtifactBtn"' in content
    assert 'id="copyArtifactPathBtn"' in content
    assert 'id="copyArtifactFingerprintBtn"' in content
    assert "artifactHeroScore" in content
    assert "findCompareArtifact" in content
    assert "artifactDisplayHint" in content
    assert "artifactDisplayPriority" in content
    assert "artifactDisplayLabel" in content
    assert "function _artifactRouteKey(artifact) {" in content
    assert "function _openManagedArtifactWindow(job, artifact, surface = 'artifact_review') {" in content
    assert "buildArtifactUrl(selected, selectedArtifact)" in review_body
    # Browser-previewable check (item 8 of executive diagnosis): TIFF/EXR are
    # not browser-previewable, so the inline <img> path is gated on the
    # narrower flag rather than the legacy `previewable` field.
    assert "artifactIsBrowserPreviewable(selectedArtifact)" in review_body
    assert "artifactDisplayLabel(selectedArtifact)" in review_body
    assert "artifactDisplayLabel(artifact)" in review_body
    assert "button.dataset.artifactPath = _artifactRouteKey(artifact);" in review_body
    assert "artifactDisplayPriority(right)" in rank_body
    assert "artifactIsBrowserPreviewable(primaryArtifact)" in compare_body
    assert "artifactIsBrowserPreviewable(candidate)" in compare_body
    assert "artifactCompareGroup(candidate) === primaryGroup" in compare_body
    assert "display_hint: _normalizeArtifactDisplayHint(item.display_hint)" in normalize_body
    assert "browser_previewable: Boolean(item.browser_previewable)" in normalize_body
    assert "preview_url: typeof item.preview_url === 'string' ? item.preview_url : ''" in normalize_body
    assert "preview_mime_type: typeof item.preview_mime_type === 'string' ? item.preview_mime_type : ''" in normalize_body
    assert "if (!artifact || typeof artifact !== 'object') return '';" in route_key_body
    assert "_resetArtifactActionButtons();" in review_body
    assert "renderConsoleContextRibbon();" in review_body
    assert "_syncConsoleRoute(true);" in review_body
    assert "delete els.openArtifactBtn.dataset.url;" in reset_body
    assert "delete els.downloadArtifactBtn.dataset.filename;" in reset_body
    assert "delete els.copyArtifactPathBtn.dataset.path;" in reset_body
    assert "delete els.copyArtifactFingerprintBtn.dataset.fingerprint;" in reset_body
    assert "parsed.origin !== window.location.origin" in sanitize_body
    assert "parsed.pathname.startsWith('/v1/jobs/')" in sanitize_body
    assert "_artifactViewerEnabled()" in open_artifact_body
    assert "const openedInViewer = _openArtifactViewer(job, artifact, document.activeElement, surface);" in open_artifact_body
    assert "if (openedInViewer) {" in open_artifact_body
    assert "return true;" in open_artifact_body
    assert "return _openManagedArtifactWindow(job, artifact, surface);" in open_artifact_body
    assert "sanitizeManagedAssetUrl(els.downloadArtifactBtn.dataset.url)" in content


def test_portal_artifact_missing_preview_cache_clears_when_artifacts_update() -> None:
    content = _portal_bundle_content()
    upsert_body = _extract_js_function_body(content, "upsertArtifact")
    stream_body = _extract_js_function_body(content, "_applyJobStreamEvent")

    assert "function _clearArtifactUrlNotFoundCache() {" in content
    assert "_artifactNotFoundUrls.clear();" in content
    assert "_clearArtifactUrlNotFoundCache();" in upsert_body
    assert "job.artifacts = normalizeArtifactItems(parsed.artifacts);" in stream_body
    assert "_clearArtifactUrlNotFoundCache();" in stream_body


def test_portal_deferred_review_surface_failures_back_off_until_retry_window_expires() -> None:
    content = _portal_bundle_content()
    load_body = _extract_js_function_body(content, "_loadDeferredReviewSurface")
    note_body = _extract_js_function_body(content, "_noteDeferredReviewSurfaceLoadFailure")
    clear_body = _extract_js_function_body(content, "_clearDeferredReviewSurfaceLoadFailure")
    prime_body = _extract_js_function_body(content, "_primeDeferredReviewSurface")
    fallback_body = _extract_js_function_body(content, "_renderDeferredReviewSurfaceFallback")

    assert "const DEFERRED_REVIEW_SURFACE_RETRY_WINDOW_MS = 30000;" in content
    assert "let deferredReviewSurfaceLoadFailedAt = 0;" in content
    assert "let deferredReviewSurfaceLoadLastToastAt = 0;" in content
    assert "if (_deferredReviewSurfaceLoadRetryBlocked()) return null;" in load_body
    assert "_clearDeferredReviewSurfaceLoadFailure();" in load_body
    assert "_noteDeferredReviewSurfaceLoadFailure();" in load_body
    assert "deferredReviewSurfaceLoadFailedAt = now;" in note_body
    assert "deferredReviewSurfaceLoadLastToastAt = now;" in note_body
    assert (
        "createToast('Review surfaces failed to load. Reload the portal and retry the review action.', 'error');" in note_body
    )
    assert "deferredReviewSurfaceLoadFailedAt = 0;" in clear_body
    assert "deferredReviewSurfaceLoadLastToastAt = 0;" in clear_body
    assert "if (!_shouldLoadDeferredReviewSurface(reason) || _deferredReviewSurfaceLoadRetryBlocked()) return;" in prime_body
    assert "els.artifactMeta.textContent = 'Review surface unavailable';" in fallback_body
    assert "Reload the portal to retry loading the review surface assets for this artifact context." in fallback_body


def test_portal_review_surface_exposes_warning_banner_and_provenance_contract() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()
    render_body = _extract_js_function_body(review_content, "renderArtifactPanel")
    state_body = _extract_js_function_body(review_content, "_reviewStatusState")
    status_body = _extract_js_function_body(review_content, "_reviewStatusSnapshot")
    banner_body = _extract_js_function_body(review_content, "_renderReviewStatusBanner")
    provenance_body = _extract_js_function_body(review_content, "_renderArtifactProvenance")

    assert 'id="reviewStatusBanner"' in content
    assert 'id="reviewStatusTitle"' in content
    assert 'id="reviewStatusDetail"' in content
    assert 'id="reviewProvenanceGrid"' in content
    assert 'id="reviewProvenanceArtifactRole"' in content
    assert 'id="reviewProvenanceRunState"' in content
    assert 'id="reviewProvenancePath"' in content
    assert 'id="reviewProvenanceFingerprint"' in content
    assert 'id="reviewProvenanceFreshness"' in content
    assert 'id="reviewProvenanceSource"' in content
    assert 'id="reviewProvenanceBatch"' in content
    assert 'id="reviewProvenanceCaptioning"' in content
    assert 'data-ui="review-status-banner"' in content
    assert 'data-ui="review-provenance-grid"' in content
    assert 'data-ui="captioning-run-status"' in content
    assert '"captioning-evidence-strip"' in review_content
    assert '"captioning-evidence-link"' in review_content
    assert '"captioning-sidecar-link"' in review_content
    assert '"captioning-raw-link"' in review_content
    assert '"captioning-proxy-link"' in review_content
    assert "function _reviewStatusState(job, reviewableOutputs, visibleWarning) {" in review_content
    assert "function _captioningEvidenceArtifacts(job, artifact)" in review_content
    assert "function _captioningStatusFromEvidence(status, artifacts)" in review_content
    assert "function _renderCaptioningEvidenceStrip(job, artifact)" in review_content
    assert "function _renderAdvisoryCaptionUnavailable(panel)" in review_content
    assert (
        "const candidates = selectedStem && stemMatchedArtifacts.length === 0 ? captionArtifacts : stemMatchedArtifacts;"
        in review_content
    )
    assert "const status = _captioningStatusFromEvidence(summary?.captioning_status || null, artifacts);" in review_content
    assert "_renderReviewStatusBanner(selected, selectedArtifact);" in render_body
    assert "_renderArtifactProvenance(selected, selectedArtifact);" in render_body
    assert "_renderReviewStatusBanner(selected, null);" in render_body
    assert "_renderArtifactProvenance(selected, null);" in render_body
    assert "_renderReviewStatusBanner(null, null);" in render_body
    assert "_renderArtifactProvenance(null, null);" in render_body
    for state_token in (
        "awaiting_job",
        "partial_reviewable",
        "failed_reviewable",
        "failed_unreviewable",
        "canceled_reviewable",
        "canceled_unreviewable",
        "offline_reviewable",
        "offline_unreviewable",
        "transport_blocked",
        "transport_warning",
        "in_progress",
        "ready",
    ):
        assert f'"{state_token}"' in state_body
    for job_state in ("partial", "failed", "canceled", "offline"):
        assert re.search(rf"job\.state === [\"']{job_state}[\"']", state_body)
    assert "job.reconnectBlocked" in state_body
    assert "Outputs ready for review" in review_content
    assert "Run canceled after partial output capture" in review_content
    assert "Run is offline with reviewable outputs" in review_content
    assert "_jobFreshnessLabel(job)" in status_body
    assert "els.reviewStatusBanner.dataset.tone = snapshot.tone;" in banner_body
    assert "els.reviewStatusBanner.dataset.reviewState = snapshot.state;" in banner_body
    assert "artifactDisplayLabel(artifact)" in provenance_body
    assert "artifactLabel(artifact)" in provenance_body
    assert "_artifactFingerprintLabel(artifact)" in provenance_body
    assert 'titleCaseToken(job.state, "Unknown")' in provenance_body
    assert "captioningRunStatusSummary(captioningStatus)" in provenance_body
    assert "els.reviewProvenanceCaptioning.dataset.status" in provenance_body
    assert "summary?.batch_id" in provenance_body
    assert "summary?.source" in provenance_body
    assert '_appendCaptionRow(fields, "Issues", caption.issues);' in review_content
    assert '_appendCaptionRow(fields, "Uncertain", caption.uncertain);' in review_content
    assert '_appendCaptionRow(fields, "Validated", _captionBooleanLabel(root.validated));' in review_content
    assert '_appendCaptionRow(fields, "Model role", root.model_role);' in review_content
    assert '_appendCaptionRow(fields, "Runtime status", root.runtime_diagnostics?.status);' in review_content


def test_portal_artifact_viewer_modal_is_feature_flagged_and_keyboard_complete() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()
    active_overlay_body = _extract_js_function_body(content, "_activeOverlayPanel")
    fallback_event_body = _extract_js_function_body(review_content, "_emitArtifactViewerFallback")
    show_fallback_body = _extract_js_function_body(review_content, "_showArtifactViewerFallback")
    render_body = _extract_js_function_body(review_content, "renderArtifactViewer")
    preview_load_body = _extract_js_function_body(review_content, "_loadArtifactViewerInlinePreview")
    open_body = _extract_js_function_body(review_content, "_openArtifactViewer")
    navigate_body = _extract_js_function_body(review_content, "_navigateArtifactViewerSelection")
    close_body = _extract_js_function_body(review_content, "_closeArtifactViewer")
    viewer_metadata_body = _extract_js_function_body(review_content, "_artifactViewerEventMetadata")

    assert 'id="artifactViewerModal"' in content
    assert 'id="artifactViewerPanel"' in content
    assert 'id="artifactViewerTitle"' in content
    assert 'id="artifactViewerImage"' in content
    assert 'id="artifactViewerFallback"' in content
    assert 'id="artifactViewerPrevBtn"' in content
    assert 'id="artifactViewerNextBtn"' in content
    assert 'id="artifactViewerZoomOutBtn"' in content
    assert 'id="artifactViewerZoomInBtn"' in content
    assert 'id="artifactViewerResetZoomBtn"' in content
    assert 'id="artifactViewerOpenRawBtn"' in content
    assert 'id="artifactViewerCopyPathBtn"' in content
    assert 'id="artifactViewerCopyFingerprintBtn"' in content
    assert 'id="artifactViewerStatus"' in content
    assert 'aria-describedby="artifactViewerMeta artifactViewerStatus"' in content
    assert "return Boolean(state.auth?.features?.artifactViewerModal);" in content
    assert "if (els.artifactViewerModal && !els.artifactViewerModal.classList.contains('hidden'))" in active_overlay_body
    assert "void _loadDeferredReviewSurface().then((api) => {" in _extract_js_function_body(content, "_openArtifactViewer")
    assert "_openManagedArtifactWindow(job, artifact, surface);" in _extract_js_function_body(content, "_openArtifactViewer")
    assert "state.portalUi.artifactViewer.open = true;" in open_body
    assert "_rememberOverlayTrigger(trigger);" in open_body
    assert "els.closeArtifactViewerBtn.focus();" in open_body
    assert "artifact_viewer_opened" in open_body
    assert 'surface: "artifact_review"' in open_body
    assert "_artifactViewerEventMetadata(job, artifact)" in open_body
    assert "state.portalUi.artifactViewer.open = false;" in close_body
    assert "_restoreOverlayFocus();" in close_body
    assert 'viewer_mode: "modal"' in review_content
    assert "artifact_fingerprint" in review_content
    assert 'pipeline: String(job?.pipeline || "")' not in viewer_metadata_body
    assert "artifact_viewer_fallback" in fallback_event_body
    assert 'surface: "artifact_review"' in fallback_event_body
    assert "fallback_reason: fallbackReason" in fallback_event_body
    assert "const hasRawAsset = Boolean(downloadUrl);" in show_fallback_body
    assert 'hasRawAsset ? "inline_preview_unavailable" : "asset_url_unavailable";' in show_fallback_body
    assert "const isRetryable = Boolean(fallbackOptions?.retryable && url);" in show_fallback_body
    assert "_renderArtifactViewerRetry(isRetryable ? { context, artifactName } : null);" in show_fallback_body
    assert "_emitArtifactViewerFallback(context, fallbackReason);" in show_fallback_body
    assert 'els.artifactViewerModal.classList.toggle("hidden", !shouldShow);' in render_body
    assert "_closeArtifactViewer(false);" in render_body
    assert "els.artifactViewerImage.style.transform = `scale(${zoomPercent / 100})`;" in render_body
    assert "els.artifactViewerImage.onerror = null;" in render_body
    assert "_showArtifactViewerFallback(activeContext, artifactName, { retryable: true });" in render_body
    assert "_showArtifactViewerFallback(context, artifactName);" in render_body
    assert "const ARTIFACT_VIEWER_FETCH_TIMEOUT_MS = 15000;" in review_content
    assert (
        'headers: _buildAuthHeaders({ Accept: artifactContentType(context.artifact) || "*/*" }, "GET"),' in preview_load_body
    )
    assert '_abortArtifactViewerPreview("superseded");' in preview_load_body
    assert '_abortArtifactViewerPreview("timeout", controller);' in preview_load_body
    assert 'const abortReason = err?.name === "AbortError" ? _artifactViewerAbortReason(controller) : "";' in preview_load_body
    assert 'const retryable = err?.name !== "AbortError" || abortReason === "timeout";' in preview_load_body
    assert "URL.createObjectURL(await response.blob())" in preview_load_body
    assert "_showArtifactViewerFallback(context, artifactName, { retryable });" in preview_load_body
    assert "els.artifactViewerCopyFingerprintBtn.dataset.fingerprint = fingerprint;" in render_body
    assert "const { artifact, index, artifacts, inlinePreview, job, url, downloadUrl, zoomPercent } = context;" in render_body
    assert "els.artifactViewerOpenRawBtn.disabled = !downloadUrl;" in render_body
    assert "els.artifactViewerOpenRawBtn.dataset.url = downloadUrl;" in render_body
    assert "_setArtifactViewerStatus(" in render_body
    assert "_rememberArtifactSelection(context.job.id, nextPath);" in navigate_body
    assert "renderReviewSurfaces();" in navigate_body
    assert '_abortArtifactViewerPreview("close");' in close_body
    assert "_setArtifactViewerBackgroundInert(false);" in close_body
    assert "_renderArtifactViewerRetry(null);" in close_body
    assert "_setArtifactViewerBackgroundInert(true);" in open_body
    assert 'document.addEventListener("keydown", artifactViewerKeydownHandler, true);' in open_body
    assert (
        'if (e.key === "Escape" && els.artifactViewerModal && !els.artifactViewerModal.classList.contains("hidden"))'
        in content
    )
    assert "if (key === 'ArrowLeft')" in content
    assert "if (key === 'ArrowRight')" in content
    assert "if (key === '+' || key === '=')" in content
    assert "if (key === '-')" in content
    assert "if (key === '0')" in content


def test_portal_review_surface_supports_compare_summary_and_keyboard_selection() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()
    render_body = _extract_js_function_body(review_content, "renderArtifactPanel")
    compare_summary_body = _extract_js_function_body(review_content, "_renderReviewCompareSummary")
    compare_copy_body = _extract_js_function_body(content, "_compareSurfaceCopy")
    focus_body = _extract_js_function_body(content, "_focusArtifactRailButton")
    keydown_body = _extract_js_function_body(content, "handleArtifactRailKeydown")

    assert 'id="reviewCompareSummary"' in content
    assert 'id="reviewCompareTitle"' in content
    assert 'id="reviewCompareDetail"' in content
    assert 'data-ui="review-compare-summary"' in content
    assert re.search(
        r"els\.artifactThumbnailRail\.setAttribute\([\"']role[\"'],\s*[\"']listbox[\"']\);",
        render_body,
    )
    assert re.search(
        r"els\.artifactThumbnailRail\.setAttribute\([\"']aria-label[\"'],\s*[\"']Artifact thumbnails[\"']\);",
        render_body,
    )
    assert re.search(r"button\.setAttribute\([\"']role[\"'],\s*[\"']option[\"']\);", render_body)
    assert re.search(
        r"button\.setAttribute\([\"']aria-selected[\"'],\s*active \? [\"']true[\"'] : [\"']false[\"']\);",
        render_body,
    )
    assert "button.tabIndex = active ? 0 : -1;" in render_body
    assert (
        "_renderReviewCompareSummary(selectedArtifact, compareAvailable ? compareCandidate : null, compareEnabled);"
        in render_body
    )
    assert "const compareAvailable = Boolean(" in render_body
    assert "selectedPreviewAvailable &&" in render_body
    assert "comparePreviewAvailable" in render_body
    assert re.search(
        r"if \(compareEnabled && selectedArtifact && compareCandidate\) \{[\s\S]*?"
        r"if \(captioningEvidenceVisible\) _renderArtifactMetadataCard\(selected, selectedArtifact\);[\s\S]*?"
        r"\} else if \(selectedPreviewAvailable\)",
        render_body,
    )
    assert re.search(
        r"els\.artifactCompareBtn\.setAttribute\([\"']aria-pressed[\"'],\s*compareEnabled \? [\"']true[\"'] : [\"']false[\"']\);",
        render_body,
    )
    assert re.search(r"els\.artifactCompareBtn\.removeAttribute\([\"']aria-controls[\"']\);", render_body)
    assert re.search(
        r"els\.artifactCompareStage\.setAttribute\([\"']aria-hidden[\"'],\s*compareEnabled \? [\"']false[\"'] : [\"']true[\"']\);",
        render_body,
    )
    assert "const compareCopy = _compareSurfaceCopy(primaryArtifact, compareArtifact, compareEnabled);" in compare_summary_body
    assert "No compare pair" in compare_copy_body
    assert "No paired comparison is available for the current artifact." in compare_copy_body
    assert "Comparing paired outputs" in compare_copy_body
    assert "Paired comparison available" in compare_copy_body
    assert "button[data-artifact-path]" in focus_body
    assert "_focusArtifactRailButton(path);" in content
    assert "const shouldRestoreFocus = event.detail === 0;" in content
    assert "requestAnimationFrame(() => {" in content
    assert "if (event.key === 'Enter' || event.key === ' ') {" in keydown_body
    assert (
        "if (!['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;" in keydown_body
    )
    assert "if (els.artifactThumbnailRail) {" in content
    assert "els.artifactThumbnailRail.addEventListener('keydown', handleArtifactRailKeydown);" in content


def test_portal_selection_review_surfaces_have_single_render_owner() -> None:
    content = _portal_bundle_content()
    review_body = _extract_js_function_body(content, "renderReviewSurfaces")
    artifact_body = _extract_js_function_body(content, "renderArtifactPanel")
    queue_body = _extract_js_function_body(content, "renderJobQueue")
    select_body = _extract_js_function_body(content, "selectJob")
    schedule_body = _extract_js_function_body(content, "scheduleRenderJobQueue")
    presets_body = _extract_js_function_body(content, "fetchPresetsForPipeline")
    diagnostics_body = _extract_js_function_body(content, "renderPreRunDiagnostics")
    backend_body = _extract_js_function_body(content, "checkBackend")
    init_body = _extract_js_function_body(content, "init")

    assert "let queuedReviewSurfaceRefresh = false;" in content
    assert "renderArtifactPanel();" in review_body
    assert "renderSelectedJobInspector();" in review_body
    assert "renderMissionControl(payload);" in review_body
    assert "renderSelectedJobInspector();" not in artifact_body
    assert "queuedReviewSurfaceRefresh = queuedReviewSurfaceRefresh || includeReviewSurfaces;" in schedule_body
    assert "const shouldRenderReviewSurfaces = queuedReviewSurfaceRefresh;" in schedule_body
    assert "queuedReviewSurfaceRefresh = false;" in schedule_body
    assert "renderJobQueue(shouldRenderReviewSurfaces);" in schedule_body
    assert "if (includeReviewSurfaces) renderReviewSurfaces();" in queue_body
    assert "renderReviewSurfaces();" in select_body
    assert "scheduleRenderJobQueue(false);" in select_body
    assert "renderReviewSurfaces();" in presets_body
    assert "renderMissionControl();" not in presets_body
    assert "renderReviewSurfaces(payload);" in diagnostics_body
    assert "renderMissionControl(payload);" not in diagnostics_body
    assert "renderReviewSurfaces();" in backend_body
    assert "renderMissionControl();" not in backend_body
    assert "renderJobQueue();" in init_body
    assert "renderArtifactPanel();" not in init_body
    assert "renderSelectedJobInspector();" not in init_body
    assert "renderMissionControl();" not in init_body


def test_portal_selected_job_inspector_uses_timeline_tabs_and_log_secondary_view() -> None:
    content = _portal_bundle_content()
    inspector_body = _extract_js_function_body(content, "renderSelectedJobInspector")
    tab_body = _extract_js_function_body(content, "setInspectorTab")

    assert 'id="inspectorOverviewTab"' in content
    assert 'id="inspectorTimelineTab"' in content
    assert 'id="inspectorLogsTab"' in content
    assert 'id="selectedJobTimelineList"' in content
    assert 'id="selectedJobLogPreview"' in content
    assert "Primary run context stays pinned above while you inspect the timeline and log stream." in content
    assert "_reconcileJobTimeline(selected);" in inspector_body
    assert "formatDuration" in inspector_body
    assert "_noteTransportWarning" in content
    assert "const showLogsShell = nextTab === 'logs' || state.currentView === 'review';" in tab_body
    assert "button.setAttribute('aria-selected', active ? 'true' : 'false');" in tab_body
    assert "els.logsShell.classList.toggle('hidden', !showLogsShell);" in tab_body


def test_portal_operate_surfaces_use_jobs_hydration_skeletons_before_empty_state() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()
    helper_body = _extract_js_function_body(content, "_isJobsHydrationPending")
    queue_empty_body = _extract_js_function_body(content, "_queueEmptyStateCopy")
    artifact_empty_body = _extract_js_function_body(review_content, "_artifactEmptyStateCopy")
    toggle_body = _extract_js_function_body(content, "_toggleSurfaceSkeleton")
    queue_body = _extract_js_function_body(content, "renderJobQueue")
    inspector_body = _extract_js_function_body(content, "renderSelectedJobInspector")
    artifact_body = _extract_js_function_body(content, "renderArtifactPanel")
    review_body = _extract_js_function_body(review_content, "renderArtifactPanel")
    recover_body = _extract_js_function_body(content, "recoverJobs")
    flush_body = _extract_js_function_body(content, "_flushBootstrapOnlineFollowup")
    backend_body = _extract_js_function_body(content, "checkBackend")

    assert "jobsLoadStatus: 'pending'," in content
    assert 'id="queueSkeletonState"' in content
    assert 'id="selectedJobSkeletonState"' in content
    assert 'id="artifactSkeletonState"' in content
    assert 'id="emptyQueueState"' in content
    assert 'id="emptyQueueTitle"' in content
    assert 'id="emptyQueueDetail"' in content
    assert 'id="emptyArtifactState"' in content
    assert 'id="emptyArtifactTitle"' in content
    assert 'id="emptyArtifactDetail"' in content
    assert 'data-ui="queue-empty-state"' in content
    assert 'data-ui="artifact-empty-state"' in content
    assert "state.jobsLoadStatus === 'loading'" in helper_body
    assert "state.bootstrap.status === 'pending' || state.bootstrap.status === 'degraded'" in helper_body
    assert "Queue unavailable" in queue_empty_body
    assert "Queue recovery needs attention" in queue_empty_body
    assert "Select a completed run" in artifact_empty_body
    assert "Outputs are still arriving" in artifact_empty_body
    assert "skeleton.setAttribute('aria-hidden', 'true');" in toggle_body
    assert "const queueLoading = _isJobsHydrationPending();" in queue_body
    assert "els.queueShell.setAttribute('aria-busy', queueLoading ? 'true' : 'false');" in queue_body
    assert "els.queueSkeletonState.setAttribute('aria-hidden', 'true');" in queue_body
    assert "_setSurfaceEmptyState(els.emptyQueueState, els.emptyQueueTitle, els.emptyQueueDetail, emptyCopy);" in queue_body
    assert (
        "_toggleSurfaceSkeleton(els.selectedJobShell, els.selectedJobShellContent, els.selectedJobSkeletonState, jobsLoading);"
        in inspector_body
    )
    assert (
        "_toggleSurfaceSkeleton(els.artifactsShell, els.artifactShellContent, els.artifactSkeletonState, jobsLoading);"
        in review_body
    )
    assert "_renderDeferredReviewSurfaceFallback(jobsLoading);" in artifact_body
    assert "_setSurfaceEmptyState(" in review_body
    assert "state.jobsLoadStatus = 'loading';" in recover_body
    assert "state.jobsLoadStatus = 'ready';" in recover_body
    assert "state.jobsLoadStatus = 'loading';" in flush_body
    assert "(state.jobsLoadStatus === 'pending' || state.jobsLoadStatus === 'loading')" in backend_body
    assert "state.jobsLoadStatus = 'offline';" in backend_body


def test_portal_theme_preference_defaults_to_system_without_persisting_boot_value() -> None:
    content = _portal_bundle_content()
    apply_body = _extract_js_function_body(content, "applyThemePreference")
    migrate_body = _extract_js_function_body(content, "_migrateThemePreferenceStorage")
    init_body = _extract_js_function_body(content, "init")

    assert "const THEME_STORAGE_KEY = 'tp_theme';" in content
    assert "const THEME_STORAGE_VERSION_KEY = 'tp_theme_version';" in content
    assert "const THEME_STORAGE_VERSION = '2';" in content
    assert "const THEME_PREFERENCES = Object.freeze(['system', 'dark', 'light']);" in content
    assert "themePreference: 'system'," in content
    assert "localStorage.setItem('tp_theme', mode);" not in content
    assert "const storageVersion = localStorage.getItem(THEME_STORAGE_VERSION_KEY);" in migrate_body
    assert "if (storageVersion === THEME_STORAGE_VERSION) return;" in migrate_body
    assert "if (localStorage.getItem(THEME_STORAGE_KEY) !== null) {" in migrate_body
    assert "localStorage.removeItem(THEME_STORAGE_KEY);" in migrate_body
    assert "localStorage.setItem(THEME_STORAGE_VERSION_KEY, THEME_STORAGE_VERSION);" in migrate_body
    assert "localStorage.setItem(THEME_STORAGE_VERSION_KEY, THEME_STORAGE_VERSION);" in apply_body
    assert "if (normalizedPreference === 'system') localStorage.removeItem(THEME_STORAGE_KEY);" in apply_body
    assert "else localStorage.setItem(THEME_STORAGE_KEY, normalizedPreference);" in apply_body
    assert "_migrateThemePreferenceStorage();" in init_body
    assert (
        "const savedThemePreference = _normalizeThemePreference(localStorage.getItem(THEME_STORAGE_KEY)) || 'system';"
        in init_body
    )
    assert "applyThemePreference(savedThemePreference, { persist: false, themeQuery });" in init_body


def test_portal_theme_control_cycles_system_dark_light_preferences() -> None:
    content = _portal_bundle_content()
    next_body = _extract_js_function_body(content, "_nextThemePreference")
    sync_body = _extract_js_function_body(content, "_syncThemeButton")
    apply_body = _extract_js_function_body(content, "applyThemePreference")

    assert "Theme: System" in content
    assert "Theme preference: system. Click to switch to dark." in content
    assert "Cycle Theme" in content
    assert "const nextIndex = (currentIndex + 1) % THEME_PREFERENCES.length;" in next_body
    assert "return THEME_PREFERENCES[nextIndex];" in next_body
    assert "applyThemePreference(_nextThemePreference(state.themePreference));" in content
    assert "Theme: System (${effectiveLabel})" in sync_body
    assert "Theme preference: ${preference}. Click to switch to ${nextPreference}." in sync_body
    assert "document.documentElement.classList.toggle('dark', mode === 'dark');" in apply_body
    assert "document.documentElement.classList.toggle('light', mode === 'light');" in apply_body


def test_portal_theme_system_listener_only_reacts_while_following_system() -> None:
    content = _portal_bundle_content()
    init_body = _extract_js_function_body(content, "init")

    assert "const themeQuery = window.matchMedia('(prefers-color-scheme: dark)');" in init_body
    assert "themeQuery.addEventListener('change', () => {" in init_body
    assert "if (state.themePreference === 'system') {" in init_body
    assert "applyThemePreference('system', { persist: false, themeQuery });" in init_body


def test_portal_console_views_use_query_param_navigation_without_backend_route_changes() -> None:
    content = _portal_bundle_content()
    rail_body = _extract_js_function_body(content, "setupSectionRail")
    route_body = _extract_js_function_body(content, "_routeUrlForView")
    apply_view_body = _extract_js_function_body(content, "applyConsoleViewLayout")

    assert 'data-view-link="overview"' in content
    assert 'data-view-link="build"' in content
    assert 'data-view-link="operate"' in content
    assert 'data-view-link="review"' in content
    assert 'data-ui="workspace-shortcut-hint"' in content
    assert 'aria-keyshortcuts="1"' in content
    assert 'aria-keyshortcuts="2"' in content
    assert 'aria-keyshortcuts="3"' in content
    assert 'aria-keyshortcuts="4"' in content
    assert "return portalRoute.build({" in route_body
    assert "resolveView: resolveConsoleView," in route_body
    assert "normalizeSelectedJobId: _normalizeSelectedJobId," in route_body
    assert "normalizeArtifactRoutePath: _normalizeArtifactRoutePath," in route_body
    assert "activeContext: _activeRouteContext" in route_body
    assert "const routeState = portalRoute.read({" in content
    assert "resolveView: resolveConsoleView," in content
    assert "candidate === 'run'" not in content
    assert "document.body.dataset.consoleView = state.currentView;" in apply_view_body
    assert "els.queueShell.classList.toggle('hidden', state.currentView === 'review');" in apply_view_body
    assert "const isPlainPrimaryClick = event.button === 0" in rail_body
    assert "if (event.defaultPrevented || !isPlainPrimaryClick)" in rail_body
    assert "navigateConsoleView(nextView);" in rail_body
    assert "viewName," in route_body
    assert "compareEnabled," in route_body


def test_portal_console_routes_reuse_last_selected_job_across_operate_and_review() -> None:
    content = _portal_bundle_content()
    navigate_block = _extract_js_function_block(content, "navigateConsoleView")
    apply_route_body = _extract_js_function_body(content, "applyConsoleRouteFromLocation")
    select_body = _extract_js_function_body(content, "selectJob")
    selected_artifact_body = _extract_js_function_body(content, "_selectedArtifactForJob")
    recover_body = _extract_js_function_body(content, "recoverJobs")

    assert "function _rememberSelectedJob(jobId) {" in content
    assert "function _preferredSelectedJobId() {" in content
    assert "function _rememberArtifactSelection(jobId, artifactPath) {" in content
    assert "function _rememberComparePreference(jobId, enabled) {" in content
    assert "const explicitJobId = _normalizeSelectedJobId(options.jobId);" in navigate_block
    assert "const hasArtifactOption = Object.prototype.hasOwnProperty.call(options, 'artifactPath');" in navigate_block
    assert "const hasCompareOption = Object.prototype.hasOwnProperty.call(options, 'compareEnabled');" in navigate_block
    assert "const preferredJobId = _preferredSelectedJobId();" in navigate_block
    assert "_rememberSelectedJob(explicitJobId);" in navigate_block
    assert "_rememberSelectedJob(routeJobId);" in apply_route_body
    assert "const routeState = portalRoute.read({" in apply_route_body
    assert "const routeArtifactPath = routeState.artifactPath;" in apply_route_body
    assert "const routeCompareEnabled = routeState.compareEnabled;" in apply_route_body
    assert "_rememberArtifactSelection(routeJobId, routeArtifactPath);" in apply_route_body
    assert "_rememberComparePreference(routeJobId, routeCompareEnabled);" in apply_route_body
    assert "_rememberSelectedJob(jobId);" in select_body
    assert "delete state.artifactUi.compareByJob[String(jobId || '')];" not in select_body
    assert "if (selectedPath && !selected) {" in selected_artifact_body
    assert "delete state.artifactUi.compareByJob[normalizedJobId];" in selected_artifact_body
    assert "state.currentView === 'operate' || state.currentView === 'review'" in select_body
    assert "const retained = state.jobs.find((job) => job.id === state.portalUi.lastSelectedJobId) || null;" in recover_body


def test_portal_console_context_ribbon_tracks_selected_job_and_review_state() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()
    ribbon_body = _extract_js_function_body(content, "renderConsoleContextRibbon")
    apply_view_body = _extract_js_function_body(content, "applyConsoleViewLayout")
    inspector_body = _extract_js_function_body(content, "renderSelectedJobInspector")
    artifact_body = _extract_js_function_body(review_content, "renderArtifactPanel")
    operate_branch_idx = ribbon_body.index("if (state.currentView === 'operate' || state.currentView === 'review') {")
    operate_return_idx = ribbon_body.index("        return;", operate_branch_idx)
    selected_idx = ribbon_body.index("const selected = state.jobs.find((job) => job.id === state.selectedJobId) || null;")
    current_payload_idx = ribbon_body.index("const currentPayload = generatePayload();")

    assert 'id="consoleContextRibbon"' in content
    assert 'id="contextRibbonJob"' in content
    assert 'id="contextRibbonState"' in content
    assert 'id="contextRibbonFreshness"' in content
    assert 'id="contextRibbonArtifact"' in content
    assert 'id="contextRibbonCompare"' in content
    assert "const ribbonVisible = ['overview', 'build', 'operate', 'review'].includes(state.currentView);" in ribbon_body
    assert "els.consoleContextRibbon.classList.toggle('hidden', !ribbonVisible);" in ribbon_body
    assert (
        "_setSummaryCard(els.contextRibbonCard1, els.contextRibbonCard1Label, els.contextRibbonJob, els.contextRibbonJobMeta, {"
        in ribbon_body
    )
    assert "const compareCopy = _compareSurfaceCopy(selectedArtifact, compareCandidate, compareEnabled);" in ribbon_body
    assert "label: 'Artifact'," in ribbon_body
    assert "value: selected ? compareCopy.ribbonValue : 'No compare pair'" in ribbon_body
    assert selected_idx > operate_branch_idx
    assert current_payload_idx > operate_return_idx
    assert "renderConsoleContextRibbon();" in apply_view_body
    assert "renderConsoleContextRibbon();" in inspector_body
    assert "renderConsoleContextRibbon();" in artifact_body


def test_portal_build_stepper_and_quick_actions_drive_task_first_navigation() -> None:
    content = _portal_bundle_content()
    stepper_body = _extract_js_function_body(content, "syncBuildStepUi")
    update_body = _extract_js_function_body(content, "updateUIFromState")
    init_body = _extract_js_function_body(content, "init")

    assert 'id="buildStepTabs"' in content
    assert 'id="buildStepTab1"' in content
    assert 'id="buildStepTab4"' in content
    assert 'id="resumeDraftBtn"' in content
    assert "const BUILD_STEP_CONTENT = Object.freeze({" in content
    assert "button.setAttribute('aria-selected', active ? 'true' : 'false');" in stepper_body
    assert "panel.hidden = !active;" in stepper_body
    assert "panel.setAttribute('data-step-active', active ? 'true' : 'false');" in stepper_body
    assert "panel.setAttribute('data-step-hidden', active ? 'false' : 'true');" in stepper_body
    assert "panel.classList.toggle('hidden', !active);" not in stepper_body
    assert "function setBuildStep(nextStep, options) {" in content
    assert "const settings = options && typeof options === 'object' ? options : {};" in content
    assert "emitPortalEvent('step_completed'" in content
    assert "state.pipeline !== 'lux-depth-v3' && state.portalUi.buildStep < 2" in update_body
    assert "setupBuildStepper();" in init_body
    assert "if (els.heroRunBtn) {" in content
    assert "navigateConsoleView('build');" in content
    assert "navigateConsoleView('operate', { jobId });" in content
    assert "navigateConsoleView('review', { jobId });" in content
    assert "els.heroRunBtn.addEventListener('click', submitJob);" not in content


def test_portal_dispatch_review_keeps_cli_parity_in_secondary_disclosure() -> None:
    content = _portal_bundle_content()

    assert 'data-ui="dispatch-primary-lane"' in content
    assert 'data-ui="dispatch-launch"' in content
    assert 'data-ui="dispatch-shortcut-hint"' in content
    assert 'data-ui="governance-posture-hint"' in content
    assert 'id="dispatchReadinessReason"' in content
    assert 'aria-live="polite"' in content
    assert 'aria-atomic="true"' in content
    assert 'aria-describedby="dispatchReadinessReason"' in content
    assert 'id="dispatchToolsDetails"' in content
    assert 'data-ui="dispatch-tools"' in content
    assert "Review dispatch posture" in content
    assert "JSON / CLI / Effective Config" in content
    assert "CLI Parity & Config Tools" not in content
    assert 'id="effectiveConfigBtn"' in content
    assert 'id="importBtn"' in content
    assert 'id="exportBtn"' in content
    assert 'id="copyCliBtn"' in content
    assert 'id="cliPreview"' in content


def test_portal_keyboard_shortcuts_cover_view_navigation_and_help_without_text_fragility() -> None:
    content = _portal_bundle_content()
    typing_body = _extract_js_function_body(content, "_isTypingTarget")

    assert "const WORKSPACE_VIEW_SHORTCUTS = Object.freeze({" in content
    assert "'1': 'overview'" in content
    assert "'4': 'review'" in content
    assert "function _isTypingTarget(target) {" in content
    assert "target.isContentEditable || target.closest('[contenteditable=\"true\"]')" in typing_body
    assert "tagName === 'textarea' || tagName === 'select'" in typing_body
    assert "if (isPlainShortcut && (key === '?' || (key === '/' && e.shiftKey)))" in content
    assert "toggleModal(true);" in content
    assert "if (isPlainShortcut && Object.prototype.hasOwnProperty.call(WORKSPACE_VIEW_SHORTCUTS, key)) {" in content
    assert "const nextView = WORKSPACE_VIEW_SHORTCUTS[key];" in content
    assert "navigateConsoleView(nextView);" in content


def test_portal_build_surface_keeps_primary_posture_band_outside_contextual_disclosures() -> None:
    content = _portal_bundle_content()
    summary_body = _extract_js_function_body(content, "renderReconstructionRuntimeSummary")

    assert 'data-ui="build-posture-band"' in content
    assert 'data-ui="reconstruction-runtime-summary"' in content
    assert "Current Run Posture" in content
    assert content.index('id="reconstructionRuntimeSummary"') < content.index('id="reconstructionDetails"')
    assert "Preview-backed validation, normalization, and runtime estimates reflect the next dispatch." in summary_body
    assert "Primary run posture updates here before you open contextual runtime or research controls." in summary_body


def test_portal_dispatch_lane_surfaces_live_readiness_reason() -> None:
    content = _portal_bundle_content()
    guard_body = _extract_js_function_body(content, "_syncBootstrapGuardedControls")
    snapshot_body = _extract_js_function_body(content, "_dispatchReadinessSnapshot")

    assert "function _dispatchReadinessSnapshot(payload = null) {" in content
    assert (
        "const DISPATCH_BACKEND_OFFLINE_MESSAGE = 'Backend is offline. Dispatch is disabled until connectivity is restored.';"
        in content
    )
    assert "Preview-backed validation is refreshing. Dispatch unlocks when the current draft settles." in snapshot_body
    assert "Debug bundle acknowledgement is required before dispatch." in snapshot_body
    assert "detail: DISPATCH_BACKEND_OFFLINE_MESSAGE" in snapshot_body
    assert "els.dispatchReadinessReason.textContent = readiness.detail;" in guard_body
    assert "els.dispatchReadinessReason.dataset.tone = readiness.tone;" in guard_body


def test_portal_display_job_state_keeps_terminal_retained_outputs_reviewable_without_dead_indexing_branch() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "_displayJobState")

    assert "const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;" in body
    assert "const reviewableOutputs = _jobHasReviewableOutputs(job);" in body
    assert "Terminal runs stay reviewable once outputs are retained" in body
    assert "if ((rawState === 'succeeded' || rawState === 'ready') && reviewableOutputs) return 'reviewable';" in body
    assert "if ((rawState === 'succeeded' || rawState === 'ready') && artifactCount > 0) return 'indexing';" not in body


def test_portal_switch_state_badges_share_the_toggle_control_wrapper() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "_syncSwitchStateLabels")

    assert "let controlsWrap = label.querySelector('[data-switch-controls-wrap=\"true\"]');" in body
    assert "controlsWrap.dataset.switchControlsWrap = 'true';" in body
    assert "controlsWrap.className = 'ml-3 inline-flex items-center gap-3';" in body
    assert "controlsWrap.appendChild(toggleWrap);" in body
    assert "controlsWrap.insertBefore(stateLabel, controlsWrap.firstChild);" in body
    assert "label.insertBefore(stateLabel, toggleWrap);" not in body
    assert "'mr-3 inline-flex min-w-[3rem]" not in body


def test_portal_disclosure_defaults_are_state_driven_instead_of_static() -> None:
    content = _portal_bundle_content()
    sync_body = _extract_js_function_body(content, "syncDisclosurePanels")
    init_body = _extract_js_function_body(content, "init")

    assert 'id="advancedFlagsDetails"' in content
    assert 'id="governanceDetails"' in content
    assert 'id="reconstructionDetails"' in content
    assert 'id="dispatchToolsDetails"' in content
    assert 'id="advancedFlagsSummary"' in content
    assert 'id="governanceDetailsSummary"' in content
    assert 'id="reconstructionDetailsSummary"' in content
    assert 'id="dispatchToolsSummary"' in content
    assert 'id="advancedFlagsDetails" class="disclosure-panel disclosure-panel-secondary mt-6">' in content
    assert 'id="governanceDetails" class="hidden disclosure-panel disclosure-panel-secondary mt-6">' in content
    assert 'id="reconstructionDetails" class="hidden disclosure-panel disclosure-panel-secondary mt-6">' in content
    assert "function _setDisclosureSummaryBadge(element, text, tone = 'info') {" in content
    assert "element.dataset.tone = String(tone || 'info').trim().toLowerCase() || 'info';" in content
    assert "const previewFieldGroups = {" in sync_body
    assert "const researchPreset = _presetRequiresResearchAcknowledgments(preset, args);" in sync_body
    assert "element.dataset.autoOpen = autoOpenState[name] ? 'true' : 'false';" in sync_body
    assert "disclosurePrefs.dispatchTools === true" in sync_body
    assert "els.advancedFlagsSummary" in sync_body
    assert "els.governanceDetailsSummary" in sync_body
    assert "els.reconstructionDetailsSummary" in sync_body
    assert "els.dispatchToolsSummary" in sync_body
    assert "advancedNeedsAttention ? 'Needs attention' : advancedActive ? 'Contextual' : 'Secondary'" in sync_body
    assert "governanceNeedsAttention ? 'Needs attention' : governanceActive ? 'Contextual' : 'Contextual'" in sync_body
    assert "reconstructionNeedsAttention ? 'Needs attention' : reconstructionActive ? 'Contextual' : 'Contextual'" in sync_body
    assert "String(args.preset || '').toLowerCase().includes('v3.1')" not in sync_body
    assert "setupDisclosurePanels();" in init_body


def test_portal_html_first_paint_lux_defaults_match_premium_posture() -> None:
    html = _portal_html_content()

    assert re.search(r'id="enableSegmentation"[^>]*checked', html)
    assert '<option value="efficientsam" selected>efficientsam</option>' in html
    assert '<option value="stub">stub</option>' in html
    assert '<option value="custom">custom (Manual)</option>' in html
    assert re.search(r'id="strictSegmentation"[^>]*checked', html)
    assert re.search(r'id="sam2ModelSizeField" class="[^"]*\bhidden\b', html)
    assert re.search(r'id="sam2CheckpointField" class="[^"]*\bhidden\b', html)
    assert re.search(r'id="sam2TuningPanel" class="[^"]*\bhidden\b', html)
    assert re.search(r'id="sam2TilingConfigFields" class="[^"]*\bhidden\b', html)
    assert re.search(r'id="v2PresetField" class="[^"]*\bhidden\b', html)
    assert re.search(r'id="governanceDetails" class="[^"]*\bhidden\b', html)
    assert re.search(r'id="reconstructionDetails" class="[^"]*\bhidden\b', html)
    assert "Segmentation is active via efficientsam." in html
    assert "Turn segmentation on to choose a backend and strictness policy." not in html
    assert re.search(r'id="runCardVersion"[^>]*>[\s\S]*?<option value="v1" selected>v1</option>', html)


def test_portal_overview_and_build_surfaces_sync_bootstrap_skeletons_and_preview_loading() -> None:
    content = _portal_bundle_content()
    helper_body = _extract_js_function_body(content, "_syncOverviewBuildLoadingState")
    preview_body = _extract_js_function_body(content, "_isBuildPreviewRefreshing")
    mission_body = _extract_js_function_body(content, "renderMissionControl")
    bootstrap_body = _extract_js_function_body(content, "_syncBootstrapUi")
    cli_body = _extract_js_function_body(content, "renderCLI")

    assert 'id="missionShellSkeletonState"' in content
    assert 'id="intelligenceShellSkeletonState"' in content
    assert 'id="overviewStatsSkeletonState"' in content
    assert 'id="overviewCapabilitySkeletonState"' in content
    assert 'id="profileShellSkeletonState"' in content
    assert 'id="buildStepperSkeletonState"' in content
    assert 'id="parametersShellSkeletonState"' in content
    assert "function _isBootstrapSurfaceLoading()" in content
    assert "function _isBuildPreviewRefreshing(payload = null) {" in content
    assert "return Boolean(preview && preview.status === 'loading');" in preview_body
    assert (
        "_toggleSurfaceSkeleton(els.missionShell, els.missionShellContent, els.missionShellSkeletonState, bootstrapLoading);"
        in helper_body
    )
    assert (
        "_toggleSurfaceSkeleton(els.intelligenceShell, els.intelligenceShellContent, els.intelligenceShellSkeletonState, bootstrapLoading);"
        in helper_body
    )
    assert (
        "_toggleSurfaceSkeleton(els.overviewStatsRow, els.overviewStatsRow, els.overviewStatsSkeletonState, bootstrapLoading);"
        in helper_body
    )
    assert (
        "_toggleSurfaceSkeleton(els.overviewCapabilityRow, els.overviewCapabilityRow, els.overviewCapabilitySkeletonState, bootstrapLoading);"
        in helper_body
    )
    assert (
        "_toggleSurfaceSkeleton(els.profileShell, els.profileShellContent, els.profileShellSkeletonState, bootstrapLoading);"
        in helper_body
    )
    assert (
        "_toggleSurfaceSkeleton(els.buildStepperShell, els.buildStepperShellContent, els.buildStepperSkeletonState, bootstrapLoading);"
        in helper_body
    )
    assert (
        "_toggleSurfaceSkeleton(els.parametersShell, els.parametersShellContent, els.parametersShellSkeletonState, bootstrapLoading);"
        in helper_body
    )
    assert "document.body.dataset.bootstrapLoading = bootstrapLoading ? 'true' : 'false';" in helper_body
    assert "document.body.dataset.buildPreviewLoading = previewRefreshing ? 'true' : 'false';" in helper_body
    assert "_syncOverviewBuildLoadingState(currentPayload);" in mission_body
    assert "_syncOverviewBuildLoadingState();" in bootstrap_body
    assert "_syncOverviewBuildLoadingState(payload);" in cli_body


def test_portal_loading_tokens_and_reduced_motion_cover_overview_and_build_surfaces() -> None:
    css = _portal_css_content()

    assert ".skeleton-pill" in css
    assert ".surface-loading" in css
    assert ".surface-loading:after" in css
    assert re.search(r"\.status-dot\.running[^{}]*,\s*\.status-dot\.partial", css)
    assert re.search(r"\.skeleton-pill[^{}]*,", css)
    assert re.search(r"\.toast-enter[^{}]*,", css)
    assert re.search(r"\.surface-loading:after\s*\{", css)
    assert re.search(r"transition:\s*none!important", css)


def test_portal_runtime_css_ships_short_viewport_modal_and_phone_stepper_rules() -> None:
    css = _portal_css_content()
    base_stepper_rule = re.search(
        r"\.build-step-tabs\s*\{[^}]*display:\s*grid[^}]*grid-template-columns:\s*repeat\(4,minmax\(0,1fr\)\)[^}]*gap:\s*\.75rem",
        css,
    )
    phone_stepper_rule = re.search(
        r"@media\(max-width:639px\)\{\.build-step-tabs\s*\{[^}]*grid-template-columns:\s*repeat\(2,minmax\(0,1fr\)\)[^}]*gap:\s*\.5rem[^}]*}\.build-step-tab\s*\{[^}]*min-height:\s*0",
        css,
    )

    assert ".max-h-\\[92vh\\]" in css
    assert "max-height:92vh" in css
    assert base_stepper_rule is not None
    assert phone_stepper_rule is not None
    assert base_stepper_rule.start() < phone_stepper_rule.start()


def test_portal_preview_statuses_render_inline_for_build_fields() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "renderFieldPreviewStatuses")

    assert 'id="inputDirStatus"' in content
    assert 'id="outputDirStatus"' in content
    assert 'id="archiveIndexStatus"' in content
    assert 'id="rightsManifestStatus"' in content
    assert "_previewIssueForField('input_dir', currentPayload)" in body
    assert "_previewIssueForField('output_dir', currentPayload)" in body
    assert "_previewIssueForField('archive_index', currentPayload)" in body
    assert "_previewIssueForField('manifest_jsonl', currentPayload)" in body
    assert "renderFieldPreviewStatuses(payload);" in _extract_js_function_body(content, "renderCLI")


def test_portal_overlay_focus_management_traps_and_restores_focus() -> None:
    content = _portal_bundle_content()
    trap_body = _extract_js_function_body(content, "_trapOverlayFocus")

    assert "function _rememberOverlayTrigger" in content
    assert "function _restoreOverlayFocus" in content
    assert "function _overlayFocusableElements" in content
    assert "const toggleModal = (show, trigger = document.activeElement) => {" in content
    assert "const toggleEffectiveConfigDrawer = (show, trigger = document.activeElement) => {" in content
    assert "const panel = _activeOverlayPanel();" in trap_body
    assert "if (event.key !== 'Tab') return false;" in trap_body
    assert "_rememberOverlayTrigger(trigger);" in content
    assert "_restoreOverlayFocus();" in content


def test_portal_queue_rows_support_keyboard_selection_navigation() -> None:
    content = _portal_bundle_content()
    keydown_body = _extract_js_function_body(content, "handleJobListKeydown")
    queue_body = _extract_js_function_body(content, "renderJobQueue")

    assert "li.setAttribute('role', 'option');" in queue_body
    assert "li.setAttribute('aria-selected', isSelected ? 'true' : 'false');" in queue_body
    assert "if (event.key === 'Enter' || event.key === ' ') {" in keydown_body
    assert "if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;" in keydown_body
    assert "if (els.jobList) els.jobList.addEventListener('keydown', handleJobListKeydown);" in content


def test_portal_timestamp_parsing_normalizes_second_precision_epochs() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "parseTimestamp")

    assert "value > 0 && value < 1e12 ? value * 1000 : value" in body
    assert "const numeric = Number(value);" in body


def test_portal_preset_selection_applies_recommended_defaults_without_changing_contract_shape() -> None:
    content = _portal_bundle_content()
    bind_body = _extract_js_function_body(content, "bindInputs")
    preset_body = _extract_js_function_body(content, "applyPresetRecommendedArgs")
    option_body = _extract_js_function_body(content, "applyPipelinePresetOptions")
    fetch_body = _extract_js_function_body(content, "fetchPresetsForPipeline")

    assert "applyPresetRecommendedArgs(nextPreset);" in bind_body
    assert "if (String(presetName || '').trim() === 'custom') {" in preset_body
    assert "state.config.preset = 'custom';" in preset_body
    assert "quality_tier" in preset_body
    assert "depth_backend" in preset_body
    assert "segmentation_backend" in preset_body
    assert "sam2_tiling_enabled" in preset_body
    assert "sam2_tile_size_px" in preset_body
    assert "sam2_points_per_side" in preset_body
    assert "emit_run_card" in preset_body
    assert "run_card_version" in preset_body
    assert "run_card_include_proofs" in preset_body
    assert "customOption.value = 'custom';" in option_body
    assert "customOption.textContent = 'custom (Manual)';" in option_body
    assert "const names = [...els.presetSelect.options].map((option) => String(option.value || ''));" in option_body
    assert "advanced_sections" in fetch_body
    assert "recommended_args" in fetch_body
    assert "applyPresetRecommendedArgs(" not in fetch_body
    assert "state.config =" not in fetch_body
    assert "state.config.preset =" not in fetch_body


def test_portal_init_establishes_interactive_shell_before_bootstrap_settles() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "init")

    assert "const bootstrapPromise = loadPortalBootstrap();" in body
    assert "_syncBootstrapUi();" in body
    assert "renderJobQueue();" in body
    assert "startHealthPolling();" in body
    assert "await bootstrapPromise;" in body
    assert "await loadPortalBootstrap();" not in body


def test_portal_bootstrap_online_followup_state_is_tracked_until_auth_ready() -> None:
    content = _portal_bundle_content()
    state_source = _portal_internal_state_source_content()

    assert "bootstrap: portalInternals.createPortalBootstrapState(Date.now())," in content
    assert 'lastHealthEndpointPath: ""' in state_source
    assert "pendingOnlineFollowup: false" in state_source
    assert "onlineFollowupComplete: false" in state_source
    assert "deadlineAt: 0" in state_source
    assert 'lastOutcome: ""' in state_source


def test_portal_bootstrap_retry_lifecycle_tracks_active_state_and_teardown() -> None:
    content = _portal_bundle_content()
    active_body = _extract_js_function_body(content, "_hasActiveBootstrapRetryState")
    history_body = _extract_js_function_body(content, "_hasBootstrapRetryHistory")
    record_body = _extract_js_function_body(content, "_recordBootstrapRetryEvent")
    cleanup_body = _extract_js_function_body(content, "cleanupActiveJobHandles")

    assert "state.bootstrap.retry.timer !== null" in active_body
    assert "state.bootstrap.retry.deadlineAt > 0" in active_body
    assert "state.bootstrap.retry.attempt > 0" in history_body
    assert "state.bootstrap.retry.lastOutcome" in history_body
    assert "console.warn('[portal bootstrap retry]', payload);" in record_body
    assert "console.info('[portal bootstrap retry]', payload);" in record_body
    assert "_finalizeBootstrapRetry('terminal_navigation_abort', { reason: 'navigation_abort' });" in cleanup_body
    assert "_cancelPendingBootstrapRequest('navigation_abort');" in cleanup_body


def test_portal_bootstrap_retry_scheduler_rejects_tight_loops_under_persistent_failure() -> None:
    content = _portal_bundle_content()
    retry_body = _extract_js_function_body(content, "_scheduleBootstrapRetry")

    assert "if (state.bootstrap.retry.timer !== null) {" in retry_body
    assert "skipped_already_scheduled" in retry_body
    assert "state.bootstrap.retry.deadlineAt = now + BOOTSTRAP_RETRY_MAX_WINDOW_MS;" in retry_body
    assert "attempt > BOOTSTRAP_RETRY_MAX_ATTEMPTS" in retry_body
    assert "(now + delayMs) > state.bootstrap.retry.deadlineAt" in retry_body
    assert "window.setTimeout(() => {" in retry_body


def test_portal_verbose_quiet_conflict_is_notified_and_blocked() -> None:
    content = _portal_bundle_content()
    bind_body = _extract_js_function_body(content, "bindInputs")
    submit_body = _extract_js_function_body(content, "submitJob")

    assert "verbose and quiet are mutually exclusive; disabled" in bind_body
    assert "verbose and quiet are mutually exclusive; disable one flag." in submit_body


def test_portal_reconstruction_runtime_summary_and_effective_config_surfaces_are_present() -> None:
    content = _portal_bundle_content()
    summary_body = _extract_js_function_body(content, "renderReconstructionRuntimeSummary")
    drawer_body = _extract_js_function_body(content, "renderEffectiveConfigDrawer")

    assert 'id="summaryReconstructionState"' in content
    assert 'id="summaryRuntimeWorkers"' in content
    assert 'id="summaryRawIngest"' in content
    assert 'id="summaryDebugBundle"' in content
    assert 'id="summaryPreviewState"' in content
    assert 'id="estimateRuntimeBand"' in content
    assert 'id="debugBundleGuardrail"' in content
    assert 'id="effectiveConfigDrawer"' in content
    assert 'id="requestedConfigJson"' in content
    assert 'id="effectiveConfigJson"' in content
    assert 'id="inactiveConfigJson"' in content
    assert "renderDebugBundleGuardrail(currentPayload" in summary_body
    assert "renderEffectiveConfigDrawer(currentPayload" in summary_body
    assert "effectivePreview.normalized_args" in drawer_body
    assert "effectivePreview.inactive_fields" in drawer_body


def test_portal_preview_metadata_worker_modes_and_export_contract_are_wired() -> None:
    content = _portal_bundle_content()
    state_source = _portal_internal_state_source_content()
    update_body = _extract_js_function_body(content, "updateUIFromState")
    bind_body = _extract_js_function_body(content, "bindInputs")
    metadata_body = _extract_js_function_body(content, "fetchConfigMetadata")
    preview_body = _extract_js_function_body(content, "fetchConfigPreview")
    reconcile_body = _extract_js_function_body(content, "_reconcilePreviewRepairedPaths")
    setter_body = _extract_js_function_body(content, "_setBuildSurfacePathFieldValue")

    assert 'modelKey: "da3-metric",' in state_source
    assert 'id="modelKey"' in content
    assert "c.modelKey = _resolveDa3ModelKey(c.modelKey);" in update_body
    assert "safeBindText(els.modelKey, null, 'modelKey');" in bind_body
    assert (
        "model_catalog: data.model_catalog && typeof data.model_catalog === 'object' ? data.model_catalog : {},"
        in metadata_body
    )
    assert 'maxWorkersMode: "auto",' in state_source
    assert 'maxGpuWorkersMode: "auto",' in state_source
    assert 'id="maxWorkersMode"' in content
    assert 'id="maxGpuWorkersMode"' in content
    assert "function fetchConfigMetadata" in content
    assert "function fetchConfigPreview" in content
    assert "function scheduleConfigPreview" in content
    assert "syncRuntimeWorkerModeControls();" in update_body
    assert "state.config.runtime.maxWorkersMode = _normalizeWorkerMode(e.target.value);" in bind_body
    assert "state.config.runtime.maxGpuWorkersMode = _normalizeWorkerMode(e.target.value);" in bind_body
    assert "schema: 'tp.portal.export.v1'" in content
    assert "effective_args:" in content
    assert "execution_args:" in content
    assert "inactive_fields:" in content
    assert "estimate_summary:" in content
    assert "argv_preview:" in content
    assert "submitted_args:" in preview_body
    assert "execution_args:" in preview_body
    assert "const refreshPreviewDrivenSurfaces = (nextPayload = currentPayload) => {" in preview_body
    assert "renderPreRunDiagnostics(nextPayload);" in preview_body
    assert "_syncBootstrapGuardedControls();" in preview_body
    assert "refreshPreviewDrivenSurfaces(generatePayload());" in preview_body
    assert "backend_catalog: {}," in content
    assert (
        "backend_catalog: data.backend_catalog && typeof data.backend_catalog === 'object' ? data.backend_catalog : {},"
        in metadata_body
    )
    assert "function renderRuntimeBriefing(payload = null) {" in content
    assert "scheduleConfigPreview(true);" in metadata_body
    assert "repo_local_path_repaired" in reconcile_body
    assert "_setBuildSurfacePathFieldValue(fieldName, normalizedValue)" in reconcile_body
    assert "state.config = state.config || {};" in setter_body
    assert "state.config.gate = state.config.gate || {};" in setter_body
    assert "state.config.segmentation = state.config.segmentation || {};" in setter_body
    assert "state.config.reconstruction = state.config.reconstruction || {};" in setter_body


def test_portal_runtime_briefing_and_recovery_surfaces_stay_additive_and_selector_stable() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()
    mission_body = _extract_js_function_body(content, "renderMissionControl")
    review_status_body = _extract_js_function_body(review_content, "_reviewStatusSnapshot")
    queue_empty_body = _extract_js_function_body(content, "_queueEmptyStateCopy")
    artifact_empty_body = _extract_js_function_body(review_content, "_artifactEmptyStateCopy")
    inspector_body = _extract_js_function_body(content, "renderSelectedJobInspector")

    assert 'id="overviewRuntimeBriefing"' in content
    assert 'data-ui="runtime-clarity-grid"' in content
    assert 'id="buildRuntimeBriefing"' in content
    assert 'data-ui="build-runtime-clarity"' in content
    assert 'id="reviewStatusAction"' in content
    assert 'id="emptyQueueAction"' in content
    assert 'id="emptyArtifactAction"' in content
    assert 'id="selectedJobRecoveryTitle"' in content
    assert 'id="selectedJobRecoveryDetail"' in content
    assert "renderRuntimeBriefing(currentPayload);" in mission_body
    assert (
        "action: 'Next action: open Build to prepare the next run or restore backend connectivity to recover recent history.'"
        in queue_empty_body
    )
    assert re.search(
        r"action:\s*[\"\']Next action: inspect the selected run in Operate or wait for indexed outputs before reopening review\.[\"\']",
        artifact_empty_body,
    )
    assert "const builder = REVIEW_STATUS_BUILDERS[stateToken] || REVIEW_STATUS_BUILDERS.ready;" in review_status_body
    assert (
        "Next action: use the selected run state, warning context, and freshness above to decide whether to recover or open review."
        in review_content
    )
    assert "els.reviewStatusAction.textContent = snapshot.action;" in review_content
    assert "els.selectedJobRecoveryTitle.textContent = recovery.title;" in inspector_body
    assert "els.selectedJobRecoveryDetail.textContent = recovery.detail;" in inspector_body


def test_portal_contextual_action_rail_reuses_existing_route_and_recovery_contracts() -> None:
    content = _portal_bundle_content()
    rail_snapshot_body = _extract_js_function_body(content, "_operatorActionRailSnapshot")
    recovery_snapshot_body = _extract_js_function_body(content, "_operatorRecoveryActionSnapshot")
    rail_render_body = _extract_js_function_body(content, "renderOperatorActionRail")
    inspector_actions_body = _extract_js_function_body(content, "renderSelectedJobRecoveryActions")
    review_actions_body = _extract_js_function_body(content, "renderReviewStatusActions")
    handler_body = _extract_js_function_body(content, "handleOperatorActionClick")

    assert 'data-ui="console-action-rail"' in content
    assert 'data-ui="console-action-shortcuts"' in content
    assert 'data-ui="console-action-primary"' in content
    assert 'data-ui="console-action-secondary-1"' in content
    assert 'data-ui="console-action-secondary-2"' in content
    assert 'data-ui="selected-job-recovery-actions"' in content
    assert 'data-ui="selected-job-recovery-primary"' in content
    assert 'data-ui="selected-job-recovery-secondary"' in content
    assert 'data-ui="review-status-actions"' in content
    assert 'data-ui="review-status-primary"' in content
    assert 'data-ui="review-status-secondary"' in content
    assert "Open Review" in rail_snapshot_body
    assert "Open Latest Artifact" in rail_snapshot_body
    assert "Toggle Compare" in rail_snapshot_body
    assert "Stay in Operate" in rail_snapshot_body
    assert "Open Early Artifacts" in rail_snapshot_body
    assert "Review Retained Outputs" in rail_snapshot_body
    assert "Return to Build" in rail_snapshot_body
    assert "Restore Access" in recovery_snapshot_body
    assert "Retry Status Check" in recovery_snapshot_body
    assert "els.consoleActionRailHint.innerHTML = _operatorActionHintHtml();" in rail_render_body
    assert "const snapshot = _operatorActionRailSnapshot(job);" in inspector_actions_body
    assert "const snapshot = _operatorActionRailSnapshot(job);" in review_actions_body
    assert "_openReviewSurfaceForJob(job, 'action_rail');" in handler_body
    assert "_openArtifactForSelection(job, context.heroArtifact || context.selectedArtifact, 'action_rail');" in handler_body
    assert "_toggleCompareSurface(job, 'action_rail');" in handler_body
    assert "_retryPortalStatus(job);" in handler_body
    assert "window.location.assign(_managedLoginUrlForCurrentRoute());" in handler_body
    assert "_rememberArtifactSelection(explicitJobId, '');" in content
    assert "_rememberArtifactSelection(preferredJobId, '');" in content
    assert "url.searchParams.set('action'" not in content
    assert "url.searchParams.set('mode'" not in content
    assert "url.searchParams.set('shortcut'" not in content


def test_portal_preview_recovers_from_stale_and_transient_service_failures() -> None:
    content = _portal_bundle_content()
    fetch_body = _extract_js_function_body(content, "fetchConfigPreview")
    schedule_body = _extract_js_function_body(content, "scheduleConfigPreview")
    clear_body = _extract_js_function_body(content, "_clearConfigPreviewServiceRetry")
    retry_body = _extract_js_function_body(content, "_scheduleConfigPreviewServiceRetry")

    assert "const CONFIG_PREVIEW_SERVICE_RETRY_BASE_MS = 2500;" in content
    assert "const CONFIG_PREVIEW_SERVICE_RETRY_MAX_ATTEMPTS = 3;" in content
    assert "let configPreviewServiceRetryTimerId = null;" in content
    assert "let configPreviewServiceRetryAttempts = 0;" in content

    catch_marker = "} catch {"
    catch_index = fetch_body.rfind(catch_marker)
    assert catch_index >= 0, "fetchConfigPreview catch block not found"
    catch_block = fetch_body[catch_index:]
    assert "if (_configPreviewRequestKey(generatePayload()) !== requestKey) {" in catch_block
    assert "_scheduleConfigPreviewServiceRetry();" in catch_block

    assert "_scheduleConfigPreviewServiceRetry();" in fetch_body
    assert "_clearConfigPreviewServiceRetry();" in fetch_body
    assert "_clearConfigPreviewServiceRetry();" in schedule_body
    assert "configPreviewServiceRetryAttempts >= CONFIG_PREVIEW_SERVICE_RETRY_MAX_ATTEMPTS" in retry_body
    assert "CONFIG_PREVIEW_SERVICE_RETRY_BASE_MS * configPreviewServiceRetryAttempts" in retry_body
    assert "configPreviewServiceRetryAttempts = 0;" in clear_body


def test_portal_submit_blocks_preview_unavailable_and_debug_bundle_without_acknowledgement() -> None:
    content = _portal_bundle_content()
    guard_body = _extract_js_function_body(content, "_syncBootstrapGuardedControls")
    readiness_body = _extract_js_function_body(content, "_dispatchReadinessSnapshot")
    submit_body = _extract_js_function_body(content, "submitJob")
    preview_failure_body = _extract_js_function_body(content, "_previewFailureDetails")
    summary_body = _extract_js_function_body(content, "renderReconstructionRuntimeSummary")
    guardrail_body = _extract_js_function_body(content, "renderDebugBundleGuardrail")

    assert "Configuration preview is still refreshing." in submit_body
    assert "Preview-backed validation could not authenticate." in preview_failure_body
    assert "Preview-backed validation rejected the current configuration." in preview_failure_body
    assert "Preview-backed validation is unavailable." in preview_failure_body
    assert "preview_auth_failed" in preview_failure_body
    assert "preview_validation_error" in preview_failure_body
    assert "preview_service_unavailable" in preview_failure_body
    assert "Acknowledge the reconstruction debug-bundle guardrail before dispatch." in submit_body
    assert "debug_bundle_acknowledgement_required" in submit_body
    assert "const readiness = _dispatchReadinessSnapshot();" in guard_body
    assert "_effectiveDebugBundleEnabled(preview, currentPayload)" in readiness_body
    assert "_effectiveDebugBundleEnabled(preview, payload)" in submit_body
    assert "_effectiveDebugBundleEnabled(matchedPreview, currentPayload)" in summary_body
    assert "_effectiveDebugBundleEnabled(currentPreview, currentPayload)" in guardrail_body


def test_lux_cli_parity_links_portal_canonical_args_and_backend_argv() -> None:
    content = _portal_bundle_content()
    canonical_keys = _extract_portal_canonical_lux_arg_keys(content)
    portal_cli_flags = _extract_portal_lux_cli_flags(content)

    arg_to_flag = {
        "preset": "--preset",
        "quality_tier": "--quality-tier",
        "depth_backend": "--depth-backend",
        "model_key": "--model-key",
        "depth_device": "--depth-device",
        "enable_segmentation": "--enable-segmentation",
        "segmentation_backend": "--segmentation-backend",
        "sam2_model_size": "--sam2-model-size",
        "sam2_tiling_enabled": "--sam2-tiling-enabled",
        "sam2_tile_size_px": "--sam2-tile-size-px",
        "sam2_overlap_px": "--sam2-overlap-px",
        "sam2_global_pass_longest_side": "--sam2-global-pass-longest-side",
        "sam2_max_concurrency": "--sam2-max-concurrency",
        "sam2_points_per_side": "--sam2-points-per-side",
        "sam2_points_per_batch": "--sam2-points-per-batch",
        "sam2_pred_iou_thresh": "--sam2-pred-iou-thresh",
        "sam2_stability_score_thresh": "--sam2-stability-score-thresh",
        "sam2_crop_n_layers": "--sam2-crop-n-layers",
        "sam2_checkpoint_path": "--sam2-checkpoint-path",
        "strict_segmentation": "--strict-segmentation",
        "materials_v3": "--materials-v3",
        "pbr": "--pbr",
        "save_float_depth": "--save-float-depth",
        "cache_depth": "--cache-depth",
        "enable_v2": "--enable-v2",
        "v2_preset": "--v2-preset",
        "emit_master16": "--emit-master16",
        "emit_upscaled16": "--emit-upscaled16",
        "emit_marketing": "--emit-marketing",
        "emit_report": "--emit-report",
        "emit_run_card": "--emit-run-card",
        "run_card_version": "--run-card-version",
        "run_card_include_proofs": "--run-card-include-proofs",
        "non_commercial_ok": "--non-commercial-ok",
        "accept_apple_depth_pro_research_license": "--accept-apple-depth-pro-research-license",
        "accept_research_tools_license": "--accept-research-tools-license",
        "enable_reconstruction": "--enable-reconstruction",
        "grouping_mode": "--grouping-mode",
        "cameras_sidecar_path": "--cameras-sidecar-path",
        "reconstruction_iterations": "--reconstruction-iterations",
        "reconstruction_tier": "--reconstruction-tier",
        "emit_scene_debug_bundle": "--emit-scene-debug-bundle",
        "force_depth": "--force-depth",
        "strict_inputs": "--strict-inputs",
        "raw_ingest_mode": "--raw-ingest-mode",
        "raw_wb_mode": "--raw-wb-mode",
        "raw_demosaic": "--raw-demosaic",
        "max_workers": "--max-workers",
        "max_gpu_workers": "--max-gpu-workers",
        "verify_images": "--verify-images",
        "allow_semantic_fallback": "--allow-semantic-fallback",
        "verbose": "--verbose",
        "quiet": "--quiet",
        "log_level": "--log-level",
        "vlm_captioning_enabled": "--vlm-captioning",
        "vlm_captioning_backend": "--vlm-captioning-backend",
        "vlm_captioning_model": "--vlm-captioning-model",
        "vlm_captioning_proxy_format": "--vlm-captioning-proxy-format",
        "vlm_captioning_max_side_px": "--vlm-captioning-max-side-px",
        "fastvlm_python_executable": "--fastvlm-python",
        "fastvlm_mlx_vlm_dir": "--fastvlm-mlx-vlm-dir",
        "fastvlm_timeout_seconds": "--fastvlm-timeout-seconds",
    }

    for key, flag in arg_to_flag.items():
        assert key in canonical_keys, f"portal canonical args missing key '{key}'"
        assert flag in portal_cli_flags, f"portal CLI preview missing flag '{flag}'"

    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output/lux_depth_v3_apex_verify",
            "preset": "depth-pro-research-m4",
            "quality_tier": "apex",
            "depth_backend": "depth_pro",
            "depth_device": "cpu",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_model_size": "large",
            "sam2_tiling_enabled": True,
            "sam2_tile_size_px": 1536,
            "sam2_overlap_px": 256,
            "sam2_global_pass_longest_side": 1280,
            "sam2_max_concurrency": 1,
            "sam2_points_per_side": 32,
            "sam2_points_per_batch": 64,
            "sam2_pred_iou_thresh": 0.88,
            "sam2_stability_score_thresh": 0.85,
            "sam2_crop_n_layers": 1,
            "sam2_checkpoint_path": "./models/sam2/sam2.1_hiera_large.pt",
            "strict_segmentation": True,
            "materials_v3": True,
            "pbr": True,
            "save_float_depth": True,
            "cache_depth": False,
            "emit_master16": True,
            "emit_upscaled16": False,
            "emit_marketing": False,
            "emit_report": True,
            "emit_run_card": True,
            "run_card_version": "v2",
            "run_card_include_proofs": True,
            "enable_v2": True,
            "v2_preset": "default",
            "non_commercial_ok": True,
            "accept_apple_depth_pro_research_license": True,
            "accept_research_tools_license": True,
            "enable_reconstruction": True,
            "grouping_mode": "parent_dir",
            "cameras_sidecar_path": "./manifests/scene_cameras.json",
            "reconstruction_iterations": 1500,
            "reconstruction_tier": "apex_research",
            "emit_scene_debug_bundle": True,
            "force_depth": True,
            "strict_inputs": True,
            "raw_ingest_mode": "force_rawpy",
            "raw_wb_mode": "camera",
            "raw_demosaic": "AHD",
            "max_workers": 6,
            "max_gpu_workers": 2,
            "verify_images": True,
            "allow_semantic_fallback": True,
            "verbose": True,
            "quiet": False,
            "log_level": "DEBUG",
            "vlm_captioning_enabled": True,
            "vlm_captioning_backend": "fastvlm",
            "vlm_captioning_model": "review",
            "vlm_captioning_proxy_format": "jpeg",
            "vlm_captioning_max_side_px": 1200,
            "fastvlm_python_executable": "/tmp/fastvlm-python",
            "fastvlm_mlx_vlm_dir": "/tmp/mlx-vlm",
            "fastvlm_timeout_seconds": 60,
        },
    }
    argv = orchestrator_app._argv_from_request(payload)

    expected_present_flags = {flag for key, flag in arg_to_flag.items() if key not in {"model_key", "quiet"}}
    for flag in expected_present_flags:
        assert flag in argv, f"backend argv missing flag '{flag}'"

    assert _flag_value(argv, "--preset") == "depth-pro-research-m4"
    assert _flag_value(argv, "--quality-tier") == "apex"
    assert _flag_value(argv, "--depth-backend") == "depth_pro"
    assert _flag_value(argv, "--depth-device") == "cpu"
    assert _flag_value(argv, "--enable-segmentation") == "on"
    assert _flag_value(argv, "--segmentation-backend") == "sam2"
    assert _flag_value(argv, "--sam2-model-size") == "large"
    assert "--sam2-tiling-enabled" in argv
    assert _flag_value(argv, "--sam2-tile-size-px") == "1536"
    assert _flag_value(argv, "--sam2-overlap-px") == "256"
    assert _flag_value(argv, "--sam2-global-pass-longest-side") == "1280"
    assert _flag_value(argv, "--sam2-max-concurrency") == "1"
    assert _flag_value(argv, "--sam2-points-per-side") == "32"
    assert _flag_value(argv, "--sam2-points-per-batch") == "64"
    assert _flag_value(argv, "--sam2-pred-iou-thresh") == "0.88"
    assert _flag_value(argv, "--sam2-stability-score-thresh") == "0.85"
    assert _flag_value(argv, "--sam2-crop-n-layers") == "1"
    assert _flag_value(argv, "--sam2-checkpoint-path").endswith("models/sam2/sam2.1_hiera_large.pt")
    assert "--strict-segmentation" in argv
    assert _flag_value(argv, "--materials-v3") == "on"
    assert _flag_value(argv, "--pbr") == "on"
    assert _flag_value(argv, "--save-float-depth") == "on"
    assert _flag_value(argv, "--cache-depth") == "off"
    assert _flag_value(argv, "--emit-master16") == "on"
    assert _flag_value(argv, "--emit-upscaled16") == "off"
    assert _flag_value(argv, "--emit-marketing") == "off"
    assert _flag_value(argv, "--emit-report") == "on"
    assert _flag_value(argv, "--emit-run-card") == "on"
    assert _flag_value(argv, "--run-card-version") == "v2"
    assert _flag_value(argv, "--enable-v2") == "on"
    assert _flag_value(argv, "--v2-preset") == "default"
    assert _flag_value(argv, "--non-commercial-ok") == "true"
    assert _flag_value(argv, "--accept-apple-depth-pro-research-license") == "true"
    assert _flag_value(argv, "--accept-research-tools-license") == "true"
    assert _flag_value(argv, "--enable-reconstruction") == "on"
    assert _flag_value(argv, "--grouping-mode") == "parent_dir"
    assert _flag_value(argv, "--cameras-sidecar-path").endswith("manifests/scene_cameras.json")
    assert _flag_value(argv, "--reconstruction-iterations") == "1500"
    assert _flag_value(argv, "--reconstruction-tier") == "apex_research"
    assert _flag_value(argv, "--emit-scene-debug-bundle") == "on"
    assert "--force-depth" in argv
    assert "--strict-inputs" in argv
    assert _flag_value(argv, "--raw-ingest-mode") == "force_rawpy"
    assert _flag_value(argv, "--raw-wb-mode") == "camera"
    assert _flag_value(argv, "--raw-demosaic") == "AHD"
    assert _flag_value(argv, "--max-workers") == "6"
    assert _flag_value(argv, "--max-gpu-workers") == "2"
    assert "--verify-images" in argv
    assert "--allow-semantic-fallback" in argv
    assert "--verbose" in argv
    assert "--quiet" not in argv
    assert _flag_value(argv, "--log-level") == "DEBUG"
    assert _flag_value(argv, "--vlm-captioning") == "on"
    assert _flag_value(argv, "--vlm-captioning-backend") == "fastvlm"
    assert _flag_value(argv, "--vlm-captioning-model") == "review"
    assert _flag_value(argv, "--vlm-captioning-proxy-format") == "jpeg"
    assert _flag_value(argv, "--vlm-captioning-max-side-px") == "1200"
    assert _flag_value(argv, "--fastvlm-python") == "/tmp/fastvlm-python"
    assert _flag_value(argv, "--fastvlm-mlx-vlm-dir") == "/tmp/mlx-vlm"
    assert _flag_value(argv, "--fastvlm-timeout-seconds") == "60"


def test_argv_rejects_verbose_and_quiet_combination() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output/lux_depth_v3_apex_verify",
            "verbose": True,
            "quiet": True,
        },
    }

    with pytest.raises(ValueError, match="verbose and quiet are mutually exclusive"):
        orchestrator_app._argv_from_request(payload)


def test_lux_ui_backend_and_direct_cli_paths_share_config_fingerprint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "frame.jpg").touch()

    portal_payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(tmp_path / "output_ui"),
            "preset": "depth-pro-research-m4",
            "quality_tier": "apex",
            "depth_backend": "depth_pro",
            "depth_device": "cpu",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_model_size": "large",
            "strict_segmentation": True,
            "materials_v3": True,
            "pbr": True,
            "cache_depth": False,
            "emit_master16": True,
            "emit_upscaled16": False,
            "emit_marketing": False,
            "emit_report": True,
            "emit_run_card": True,
            "run_card_version": "v2",
            "run_card_include_proofs": False,
            "enable_v2": True,
            "v2_preset": "default",
            "non_commercial_ok": True,
            "accept_apple_depth_pro_research_license": True,
        },
    }
    portal_argv = orchestrator_app._argv_from_request(portal_payload)
    assert portal_argv[:3] == [sys.executable, "-m", orchestrator_app.LUX_DEPTH_MODULE]

    portal_path_config = _capture_lux_cli_config_from_args(portal_argv[3:])
    portal_path_fingerprint = _run_card_fingerprint_from_config(portal_path_config)

    direct_cli_args = [
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(tmp_path / "output_cli"),
        "--preset",
        "depth-pro-research-m4",
        "--quality-tier",
        "apex",
        "--depth-backend",
        "depth_pro",
        "--depth-device",
        "cpu",
        "--materials-v3",
        "yes",
        "--enable-segmentation",
        "on",
        "--segmentation-backend",
        "sam2",
        "--sam2-model-size",
        "large",
        "--strict-segmentation",
        "--pbr",
        "true",
        "--cache-depth",
        "0",
        "--emit-master16",
        "1",
        "--emit-upscaled16",
        "off",
        "--emit-marketing",
        "false",
        "--emit-report",
        "on",
        "--emit-run-card",
        "on",
        "--run-card-version",
        "v2",
        "--enable-v2",
        "on",
        "--v2-preset",
        "default",
        "--non-commercial-ok",
        "true",
        "--accept-apple-depth-pro-research-license",
        "true",
    ]
    direct_cli_config = _capture_lux_cli_config_from_args(direct_cli_args)
    direct_cli_fingerprint = _run_card_fingerprint_from_config(direct_cli_config)

    assert portal_path_fingerprint["canonical_json"] == direct_cli_fingerprint["canonical_json"]
    assert portal_path_fingerprint["sha256"] == direct_cli_fingerprint["sha256"]


def test_argv_normalization_includes_segmentation_controls() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_model_size": "large",
            "strict_segmentation": True,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)
    assert _flag_value(argv, "--enable-segmentation") == "on"
    assert _flag_value(argv, "--segmentation-backend") == "sam2"
    assert _flag_value(argv, "--sam2-model-size") == "large"
    assert "--strict-segmentation" in argv


def test_portal_default_da3_dispatch_uses_metric_model_without_research_ack() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "depth_backend": "da3",
        },
    }

    preview = orchestrator_app._build_lux_config_preview(payload["args"])  # type: ignore[arg-type]
    errors = {item["code"] for item in preview["field_errors"]}
    argv = orchestrator_app._argv_from_request(payload)

    assert "da3_model_non_commercial_required" not in errors
    assert preview["execution_args"]["model_key"] == "da3-metric"
    assert _flag_value(argv, "--depth-backend") == "da3"
    assert _flag_value(argv, "--model-key") == "da3-metric"
    assert "--non-commercial-ok" not in argv


def test_portal_da3_research_model_requires_noncommercial_ack() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "depth_backend": "da3",
            "model_key": "da3-research",
        },
    }

    preview = orchestrator_app._build_lux_config_preview(payload["args"])  # type: ignore[arg-type]
    errors = {item["code"] for item in preview["field_errors"]}

    assert "da3_model_non_commercial_required" in errors

    payload["args"]["non_commercial_ok"] = True  # type: ignore[index]
    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--model-key") == "da3-research"
    assert _flag_value(argv, "--non-commercial-ok") == "true"


def test_argv_normalization_includes_sam2_tiling_and_generator_controls() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_model_size": "large",
            "sam2_tiling_enabled": True,
            "sam2_tile_size_px": 1024,
            "sam2_overlap_px": 128,
            "sam2_global_pass_longest_side": 900,
            "sam2_max_concurrency": 1,
            "sam2_points_per_side": 16,
            "sam2_points_per_batch": 32,
            "sam2_pred_iou_thresh": 0.77,
            "sam2_stability_score_thresh": 0.66,
            "sam2_crop_n_layers": 2,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)
    assert "--sam2-tiling-enabled" in argv
    assert _flag_value(argv, "--sam2-tile-size-px") == "1024"
    assert _flag_value(argv, "--sam2-overlap-px") == "128"
    assert _flag_value(argv, "--sam2-global-pass-longest-side") == "900"
    assert _flag_value(argv, "--sam2-max-concurrency") == "1"
    assert _flag_value(argv, "--sam2-points-per-side") == "16"
    assert _flag_value(argv, "--sam2-points-per-batch") == "32"
    assert _flag_value(argv, "--sam2-pred-iou-thresh") == "0.77"
    assert _flag_value(argv, "--sam2-stability-score-thresh") == "0.66"
    assert _flag_value(argv, "--sam2-crop-n-layers") == "2"


def test_argv_rejects_non_finite_sam2_probability_controls() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_pred_iou_thresh": float("nan"),
        },
    }

    with pytest.raises(ValueError, match="Invalid sam2_pred_iou_thresh"):
        orchestrator_app._argv_from_request(payload)


def test_lux_config_preview_rejects_non_finite_sam2_probability_controls(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    preview = orchestrator_app._build_lux_config_preview(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_stability_score_thresh": float("inf"),
        }
    )

    assert any(
        error["field"] == "sam2_stability_score_thresh" and error["code"] == "invalid_sam2_stability_score_thresh"
        for error in preview["field_errors"]
    )


def test_argv_normalization_ignores_sam2_model_size_when_backend_is_not_sam2() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "segmentation_backend": "stub",
            "sam2_model_size": "tiny",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)
    assert _flag_value(argv, "--segmentation-backend") == "stub"
    assert "--sam2-model-size" not in argv


def test_argv_normalization_ignores_sam2_checkpoint_path_when_backend_is_not_sam2() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "segmentation_backend": "stub",
            "sam2_checkpoint_path": "./models/sam2/sam2.1_hiera_large.pt",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)
    assert _flag_value(argv, "--segmentation-backend") == "stub"
    assert "--sam2-checkpoint-path" not in argv


def test_validate_managed_sam2_checkpoint_path_allows_repo_controlled_missing_path() -> None:
    normalized = orchestrator_app._validate_managed_sam2_checkpoint_path("./models/sam2/sam2.1_hiera_large.pt")

    assert normalized.endswith("models/sam2/sam2.1_hiera_large.pt")


def test_resolve_managed_sam2_checkpoint_validation_preserves_untrusted_reason(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "sam2-untrusted.pt"
    checkpoint_path.write_bytes(b"untrusted checkpoint bytes")
    orchestrator_app._clear_managed_sam2_checksum_cache()

    validation = orchestrator_app._resolve_managed_sam2_checkpoint_validation(str(checkpoint_path))

    assert validation.normalized_path is None
    assert validation.reason == "untrusted_checkpoint_path"


def test_argv_rejects_untrusted_sam2_checkpoint_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    checkpoint_path = tmp_path / "sam2-untrusted.pt"
    input_dir.mkdir()
    output_dir.mkdir()
    checkpoint_path.write_bytes(b"untrusted checkpoint bytes")

    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_checkpoint_path": str(checkpoint_path),
        },
    }

    with pytest.raises(orchestrator_app._PortalValidationReasonError, match="SAM2 checkpoint path is not trusted") as exc_info:
        orchestrator_app._argv_from_request(payload)

    assert exc_info.value.reason == "untrusted_checkpoint_path"


def test_ensure_safe_regular_file_path_preserves_outside_root_reason(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    checkpoint_path = outside_root / "sam2-outside.pt"
    allowed_root.mkdir()
    outside_root.mkdir()
    checkpoint_path.write_bytes(b"outside")

    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._ensure_safe_regular_file_path(checkpoint_path, [allowed_root])

    assert exc_info.value.reason == "path_outside_allowed_roots"


def test_argv_rejects_non_file_sam2_checkpoint_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "sam2-checkpoint-dir"
    input_dir.mkdir()
    output_dir.mkdir()
    checkpoint_dir.mkdir()

    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_checkpoint_path": str(checkpoint_dir),
        },
    }

    with pytest.raises(orchestrator_app._PortalValidationReasonError, match="Invalid path value") as exc_info:
        orchestrator_app._argv_from_request(payload)

    assert exc_info.value.reason == "invalid_path_value"


def test_argv_rejects_oversized_sam2_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    checkpoint_path = tmp_path / "sam2-oversized.pt"
    input_dir.mkdir()
    output_dir.mkdir()
    checkpoint_path.write_bytes(b"oversized")
    monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_CHECKSUM_MAX_BYTES", 1)

    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_checkpoint_path": str(checkpoint_path),
        },
    }

    with pytest.raises(
        orchestrator_app._PortalValidationReasonError,
        match="checksum verification size limit",
    ) as exc_info:
        orchestrator_app._argv_from_request(payload)

    assert exc_info.value.reason == "checkpoint_file_too_large"


def test_managed_sam2_checkpoint_validation_reuses_cached_hash_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = (tmp_path / "sam2-governed.pt").resolve()
    checkpoint_path.write_bytes(b"trusted checkpoint bytes")
    digest = hashlib.sha256(b"trusted checkpoint bytes").hexdigest()
    orchestrator_app._clear_managed_sam2_checksum_cache()
    monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_TRUSTED_SHA256", {digest})

    hash_calls: list[Path] = []
    original_hash = orchestrator_app._hash_file_sha256

    def _counting_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
        hash_calls.append(path)
        return original_hash(path, chunk_size)

    monkeypatch.setattr(orchestrator_app, "_hash_file_sha256", _counting_hash)

    first = orchestrator_app._validate_managed_sam2_checkpoint_path(str(checkpoint_path))
    second = orchestrator_app._validate_managed_sam2_checkpoint_path(str(checkpoint_path))

    assert first == str(checkpoint_path)
    assert second == str(checkpoint_path)
    assert hash_calls == [checkpoint_path]


def test_managed_sam2_bounded_checksum_cache_eviction() -> None:
    """Verify bounded cache evicts oldest entries when capacity is exceeded (FIFO)."""
    cache = orchestrator_app._ManagedSam2BoundedChecksumCache(max_entries=3)

    # Insert 3 entries at capacity
    key1 = ("/path/a.pt", 100, 1000, 1, 1001, 2000)
    key2 = ("/path/b.pt", 200, 1000, 1, 1002, 2000)
    key3 = ("/path/c.pt", 300, 1000, 1, 1003, 2000)
    entry = orchestrator_app._ManagedSam2ChecksumCacheEntry(digest="abc", reason=None)

    cache[key1] = entry
    cache[key2] = entry
    cache[key3] = entry

    assert len(cache) == 3
    assert key1 in cache
    assert key2 in cache
    assert key3 in cache

    # Insert a 4th entry, should evict the oldest (key1)
    key4 = ("/path/d.pt", 400, 1000, 1, 1004, 2000)
    cache[key4] = entry

    assert len(cache) == 3
    assert key1 not in cache  # evicted (oldest)
    assert key2 in cache
    assert key3 in cache
    assert key4 in cache

    # Insert a 5th entry, should evict key2
    key5 = ("/path/e.pt", 500, 1000, 1, 1005, 2000)
    cache[key5] = entry

    assert len(cache) == 3
    assert key2 not in cache  # evicted
    assert key3 in cache
    assert key4 in cache
    assert key5 in cache

    # Updating an existing key should not evict
    cache[key3] = orchestrator_app._ManagedSam2ChecksumCacheEntry(digest="updated", reason=None)
    assert len(cache) == 3
    assert cache[key3].digest == "updated"

    # Clear should empty both dict and deque
    cache.clear()
    assert len(cache) == 0
    assert len(cache._insertion_order) == 0


@pytest.mark.parametrize(
    ("artifact_name", "artifact_bytes", "size_limit", "expected_reason"),
    [
        ("sam2-untrusted.pt", b"untrusted checkpoint bytes", None, "untrusted_checkpoint_path"),
        ("sam2-checkpoint-dir", None, None, "invalid_path_value"),
        ("sam2-oversized.pt", b"oversized", 1, "checkpoint_file_too_large"),
    ],
)
def test_sam2_checkpoint_preview_and_dispatch_share_reason_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    artifact_bytes: bytes | None,
    size_limit: int | None,
    expected_reason: str,
) -> None:
    input_dir = (tmp_path / "input").resolve()
    output_dir = (tmp_path / "output").resolve()
    input_dir.mkdir()
    output_dir.mkdir()
    checkpoint_path = (tmp_path / artifact_name).resolve()
    if artifact_bytes is None:
        checkpoint_path.mkdir()
    else:
        checkpoint_path.write_bytes(artifact_bytes)
    if size_limit is not None:
        monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_CHECKSUM_MAX_BYTES", size_limit)
    orchestrator_app._clear_managed_sam2_checksum_cache()

    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_checkpoint_path": str(checkpoint_path),
        },
    }

    preview = orchestrator_app._build_config_preview(payload)
    preview_error = next(item for item in preview["field_errors"] if item["field"] == "sam2_checkpoint_path")

    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._argv_from_request(payload)

    assert preview_error["code"] == expected_reason
    assert exc_info.value.reason == expected_reason


def test_argv_rejects_invalid_raw_ingest_mode() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "raw_ingest_mode": "bad_mode",
        },
    }

    with pytest.raises(ValueError, match="Invalid raw_ingest_mode"):
        orchestrator_app._argv_from_request(payload)


def test_argv_rejects_invalid_raw_wb_mode() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "raw_wb_mode": "daylight",
        },
    }

    with pytest.raises(ValueError, match="Invalid raw_wb_mode"):
        orchestrator_app._argv_from_request(payload)


def test_argv_rejects_invalid_raw_demosaic() -> None:
    """Orchestrator rejects values that are syntactically invalid as a
    rawpy.DemosaicAlgorithm member (special chars, embedded shell metas)."""
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "raw_demosaic": "amaze; rm -rf /",
        },
    }

    with pytest.raises(ValueError, match="Invalid raw_demosaic"):
        orchestrator_app._argv_from_request(payload)


def test_argv_accepts_any_syntactic_raw_demosaic_name() -> None:
    """Orchestrator accepts any rawpy.DemosaicAlgorithm member name, including
    build-specific ones (AFD, VCD, VCD_MODIFIED_AHD) that were previously
    rejected by the curated allowlist. The decode subprocess is the
    authoritative gate that fails closed for names this LibRaw build does
    not expose."""
    for demosaic in (
        "AMAZE",
        "DCB",
        "LMMSE",
        "VNG",
        "PPG",
        "AFD",
        "VCD",
        "VCD_MODIFIED_AHD",
    ):
        payload: Dict[str, object] = {
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./input_images",
                "output_dir": "./output",
                "raw_demosaic": demosaic,
            },
        }
        argv = orchestrator_app._argv_from_request(payload)
        assert "--raw-demosaic" in argv
        idx = argv.index("--raw-demosaic")
        assert argv[idx + 1] == demosaic, f"expected {demosaic} preserved in argv, got {argv[idx + 1]}"


def test_argv_rejects_invalid_reconstruction_tier() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "reconstruction_tier": "invalid_tier",
        },
    }

    with pytest.raises(ValueError, match="Invalid reconstruction_tier"):
        orchestrator_app._argv_from_request(payload)


def test_argv_rejects_invalid_log_level() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "log_level": "TRACE",
        },
    }

    with pytest.raises(ValueError, match="Invalid log_level"):
        orchestrator_app._argv_from_request(payload)


def test_portal_segmentation_defaults_align_with_cli_defaults() -> None:
    state_source = _portal_internal_state_source_content()
    assert "enable: true," in state_source
    assert 'backend: "efficientsam",' in state_source
    assert 'sam2ModelSize: "base",' in state_source
    assert "sam2TilingEnabled: false," in state_source
    assert "sam2TileSizePx: 1536," in state_source
    assert "sam2OverlapPx: 256," in state_source
    assert "sam2GlobalPassLongestSide: 1280," in state_source
    assert "sam2MaxConcurrency: 1," in state_source
    assert "sam2PointsPerSide: 32," in state_source
    assert "sam2PointsPerBatch: 64," in state_source
    assert "sam2PredIouThresh: 0.88," in state_source
    assert "sam2StabilityScoreThresh: 0.85," in state_source
    assert "sam2CropNLayers: 1," in state_source
    assert "strict: true" in state_source


def test_portal_run_card_version_control_is_explicit_in_state_and_bundle() -> None:
    state_source = _portal_internal_state_source_content()
    content = _portal_bundle_content()
    update_body = _extract_js_function_body(content, "updateUIFromState")
    bind_text_body = _extract_js_function_body(content, "bindInputs")

    assert 'runCardVersion: "v1"' in state_source
    assert "runCardVersion: _domId('runCardVersion')" in content
    assert "runCardVersionField: _domId('runCardVersionField')" in content
    assert "syncRunCardControlState(c);" in update_body
    assert "const syncDependentControlState = (category, key) => {" in bind_text_body
    assert "if (category === 'emits' && key === 'runCard') {" in bind_text_body
    assert "syncRunCardControlState(state.config);" in bind_text_body
    assert "els.emits.runCard.addEventListener('change'" not in bind_text_body


def test_portal_canonical_lux_args_gate_sam2_fields_to_active_backend_and_tiling() -> None:
    content = _portal_bundle_content()
    build_body = _extract_js_function_body(content, "buildCanonicalLuxDepthArgs")

    assert "const sam2Active = segmentationEnable && segmentationBackend === 'sam2';" in build_body
    assert "if (sam2Active) {" in build_body
    assert "args.sam2_model_size = sam2ModelSize;" in build_body
    assert "if (sam2CheckpointPath) args.sam2_checkpoint_path = sam2CheckpointPath;" in build_body
    assert "if (sam2PointsPerSide !== null) args.sam2_points_per_side = sam2PointsPerSide;" in build_body
    assert "if (sam2TilingEnabled) {" in build_body
    assert "args.sam2_tiling_enabled = true;" in build_body
    assert "if (sam2TileSizePx !== null) args.sam2_tile_size_px = sam2TileSizePx;" in build_body
    assert "if (sam2MaxConcurrency !== null) args.sam2_max_concurrency = sam2MaxConcurrency;" in build_body


def test_portal_surfaces_pre_run_diagnostics_and_expected_outputs() -> None:
    content = _portal_bundle_content()
    render_cli_body = _extract_js_function_body(content, "renderCLI")
    diagnostics_body = _extract_js_function_body(content, "renderPreRunDiagnostics")
    next_action_body = _extract_js_function_body(content, "renderNextBestAction")
    local_next_action_body = _extract_js_function_body(content, "_buildLocalNextBestAction")

    assert 'id="preRunWarnings"' in content
    assert 'id="expectedOutputsList"' in content
    assert 'id="datasetHealthText"' in content
    assert 'id="nextBestActionLabel"' in content
    assert 'id="nextBestActionDetail"' in content
    assert 'id="nextBestActionTone"' in content
    assert "next_best_action: null" in content
    assert "_normalizeNextBestAction(data.next_best_action)" in content
    assert "renderPreRunDiagnostics(payload);" in render_cli_body
    assert "renderNextBestAction(payload" in diagnostics_body
    assert "const action = _effectiveNextBestAction(payload, preview);" in next_action_body
    assert "_normalizeNextBestAction(currentPreview?.next_best_action)" in content
    assert "Wait for preview to refresh" in local_next_action_body
    assert "Restore backend connection" in local_next_action_body


def test_portal_operator_briefing_and_build_pulse_surfaces_are_present() -> None:
    content = _portal_bundle_content()
    ribbon_body = _extract_js_function_body(content, "renderConsoleContextRibbon")
    pulse_body = _extract_js_function_body(content, "renderBuildStepPulse")
    mission_body = _extract_js_function_body(content, "renderMissionControl")
    stepper_body = _extract_js_function_body(content, "syncBuildStepUi")

    assert 'id="contextRibbonCard1Label"' in content
    assert 'id="contextRibbonCard4Label"' in content
    assert 'id="buildPulseDraft"' in content
    assert 'id="buildPulsePreview"' in content
    assert 'id="buildPulseDispatch"' in content
    assert 'data-ui="build-step-pulse"' in content
    assert "label: 'Live lane'" in ribbon_body
    assert "label: 'Review lane'" in ribbon_body
    assert "label: 'Dispatch lane'" in ribbon_body
    assert "state.currentView === 'build' ? 'Current focus' : 'Draft'" in ribbon_body
    assert "_previewSurfaceSummary(currentPayload)" in pulse_body
    assert "_effectiveNextBestAction(currentPayload)" in pulse_body
    assert "renderBuildStepPulse(currentPayload);" in mission_body
    assert "renderConsoleContextRibbon();" in mission_body
    assert "renderBuildStepPulse(generatePayload());" in stepper_body


def test_portal_exposes_run_card_quick_actions() -> None:
    content = _portal_bundle_content()
    review_content = _portal_review_source_content()

    assert 'id="runCardActions"' in content
    assert 'id="viewRunCardBtn"' in content
    assert 'id="copyRunCardPathBtn"' in content
    assert 'id="copyRunCardFingerprintBtn"' in content
    assert "els.viewRunCardBtn.dataset.url = runCardUrl;" in review_content
    assert "sanitizeManagedAssetUrl(els.viewRunCardBtn.dataset.url)" in content
    assert "window.open(runCardUrl, '_blank', 'noopener,noreferrer');" in content


def test_portal_supports_partial_review_state() -> None:
    content = _portal_bundle_content()

    assert "'partial'" in content
    assert (
        "SAFE_JOB_STATES = new Set(['queued', 'running', 'succeeded', 'partial', 'failed', 'canceled', 'ready', 'offline']);"
        in content
    )
    assert "Run partially completed" in content
    assert "outputs remain reviewable" in content


def test_portal_selected_job_progress_bar_has_accessible_label() -> None:
    content = _portal_bundle_content()

    assert 'id="selectedJobProgressLabel"' in content
    assert 'aria-labelledby="selectedJobProgressLabel selectedJobProgressText"' in content


def test_portal_archive_controls_expose_canonical_stage_labels() -> None:
    content = _portal_bundle_content()

    assert "archive-gate-a (Fixity / Manifest Prep)" in content
    assert "archive-gate-b (BagIt Build)" in content
    assert "archive-gate-c (METS Export)" in content
    assert "Rights Manifest JSONL" in content
    assert "Canonical Command" in content


def test_portal_archive_build_surface_uses_manifest_input_and_never_derives_archive_index() -> None:
    content = _portal_bundle_content()
    bind_body = _extract_js_function_body(content, "bindInputs")

    assert "safeBindInput(els.rightsManifestPath, 'gate', 'manifestJsonl');" in bind_body
    assert "deriveArchiveIndexPath" not in bind_body
    assert "gateDedup" not in bind_body
    assert "gateSign" not in bind_body


def test_portal_archive_payload_and_cli_preview_use_canonical_archive_contract() -> None:
    content = _portal_bundle_content()
    payload_body = _extract_js_function_body(content, "generatePayload")
    cli_body = _extract_js_function_body(content, "renderCLI")
    payload_init_segment = payload_body.split("if (p === 'lux-depth-v3') {", maxsplit=1)[0]
    cli_archive_segment = cli_body.split("} else {", maxsplit=1)[1].split("const cli =", maxsplit=1)[0]

    assert "archive_command: archiveCommand" in payload_body
    assert "args.manifest_jsonl" in payload_body
    assert "args.archive_index" in payload_body
    assert "overwrite:" not in payload_init_segment
    assert payload_body.count("overwrite:") == 1
    assert "dedup" not in payload_body
    assert "sign" not in payload_body
    assert "--archive-command" in cli_body
    assert "--manifest-jsonl" in cli_body
    assert "--overwrite" in _extract_portal_lux_cli_flags(content)
    assert "--overwrite" not in cli_archive_segment
    assert "--dedup" not in cli_body
    assert "--sign" not in cli_body


def test_portal_archive_preview_and_readiness_flow_share_preview_state() -> None:
    content = _portal_bundle_content()
    readiness_body = _extract_js_function_body(content, "currentPipelineReadiness")
    issues_body = _extract_js_function_body(content, "currentPipelineReadinessIssues")
    cli_body = _extract_js_function_body(content, "renderCLI")
    preview_enabled_body = _extract_js_function_body(content, "_configPreviewEnabledForPipeline")
    preview_schedule_body = _extract_js_function_body(content, "scheduleConfigPreview")
    diagnostics_body = _extract_js_function_body(content, "renderPreRunDiagnostics")

    assert "_currentPreviewReadiness(payload)" in readiness_body
    assert "if (previewReadiness) return previewReadiness;" in readiness_body
    assert "if (previewReadiness) return issues;" in issues_body
    assert "CONFIG_PREVIEW_SUPPORTED_PIPELINES" in content
    assert "CONFIG_PREVIEW_SUPPORTED_PIPELINES.has" in preview_enabled_body
    assert "payload.pipeline !== 'lux-depth-v3'" not in preview_schedule_body
    assert "preview && preview.status === 'ready'" in cli_body
    assert "payload.pipeline === 'lux-depth-v3' && preview" not in cli_body
    assert "const readinessIssues = currentPipelineReadinessIssues(payload);" in diagnostics_body
    assert "Archive index path is missing;" not in diagnostics_body
    assert "Rights Manifest JSONL is missing;" not in diagnostics_body


def test_portal_archive_pipelines_hide_lux_flag_shell() -> None:
    content = _portal_bundle_content()
    update_body = _extract_js_function_body(content, "updateUIFromState")

    assert "flagsShell: _domId('flags-shell')" in content
    assert "if (els.flagsShell) els.flagsShell.classList.remove('hidden');" in update_body
    assert "if (els.flagsShell) els.flagsShell.classList.add('hidden');" in update_body


def test_portal_lux_build_surface_hides_inapplicable_optional_controls_until_needed() -> None:
    content = _portal_bundle_content()
    applicability_body = _extract_js_function_body(content, "syncBuildSurfaceApplicability")
    mission_control_body = _extract_js_function_body(content, "renderMissionControl")
    preset_body = _extract_js_function_body(content, "currentPresetDescriptor")
    preset_research_body = _extract_js_function_body(content, "_derivePresetResearchFlag")
    fallback_body = _extract_js_function_body(content, "seedPresetFallbacks")
    fetch_body = _extract_js_function_body(content, "fetchPresetsForPipeline")
    bind_inputs_body = _extract_js_function_body(content, "bindInputs")

    assert "segmentationBackendField: _domId('segmentationBackendField')" in content
    assert "sam2ModelSizeField: _domId('sam2ModelSizeField')" in content
    assert "strictSegmentationField: _domId('strictSegmentationField')" in content
    assert "sam2CheckpointField: _domId('sam2CheckpointField')" in content
    assert "sam2TuningPanel: _domId('sam2TuningPanel')" in content
    assert "sam2TilingConfigFields: _domId('sam2TilingConfigFields')" in content
    assert "sam2GeneratorConfigFields: _domId('sam2GeneratorConfigFields')" in content
    assert "runCardVersionField: _domId('runCardVersionField')" in content
    assert "v2PresetField: _domId('v2PresetField')" in content
    assert "governanceDetailsHint: _domId('governanceDetailsHint')" in content
    assert "licenseAppleField: _domId('licenseAppleField')" in content
    assert "reconstructionConfigFields: _domId('reconstructionConfigFields')" in content
    assert "function _derivePresetResearchFlag" in content
    assert ".includes('research')" in preset_research_body
    assert "if (String(state.config.preset || '').trim() === 'custom') {" in preset_body
    assert "Manual configuration mode." in preset_body
    assert "is_research: _derivePresetResearchFlag({" in preset_body
    assert "is_research: _derivePresetResearchFlag({" in fallback_body
    assert "is_research: _derivePresetResearchFlag(preset)" in fetch_body
    assert (
        "if (category === 'segmentation' && (key === 'enable' || key === 'backend' || key === 'sam2TilingEnabled')) {"
        in bind_inputs_body
    )
    assert "syncSegmentationControlState(state.config);" in bind_inputs_body
    assert "[\n        els.segmentation.enable," not in bind_inputs_body
    assert "_setContextVisibility(els.segmentationBackendField, isLuxPipeline && segmentationEnabled);" in applicability_body
    assert "_setContextVisibility(els.sam2ModelSizeField, isLuxPipeline && showSam2Controls);" in applicability_body
    assert "_setContextVisibility(els.sam2TuningPanel, isLuxPipeline && showSam2Controls);" in applicability_body
    assert (
        "_setContextVisibility(els.sam2TilingConfigFields, isLuxPipeline && showSam2Controls && sam2TilingEnabled);"
        in applicability_body
    )
    assert "_setContextVisibility(els.sam2GeneratorConfigFields, isLuxPipeline && showSam2Controls);" in applicability_body
    assert (
        "SAM2 is active, so generator controls are live now. Tiling values matter only when tiling is enabled."
        in applicability_body
    )
    assert "_setContextVisibility(els.v2PresetField, isLuxPipeline && enableV2);" in applicability_body
    assert "_setContextVisibility(els.governanceDetails, governanceVisible);" in applicability_body
    assert "els.licenseAppleField" in applicability_body
    assert "_setContextVisibility(els.reconstructionConfigFields, reconstructionEnabled);" in applicability_body
    assert "syncBuildSurfaceApplicability(currentPayload);" in mission_control_body


def test_portal_dispatch_controls_require_backend_readiness_and_live_backend() -> None:
    content = _portal_bundle_content()
    guard_body = _extract_js_function_body(content, "_syncBootstrapGuardedControls")
    readiness_body = _extract_js_function_body(content, "_dispatchReadinessSnapshot")
    submit_body = _extract_js_function_body(content, "submitJob")

    assert "const readiness = _dispatchReadinessSnapshot();" in guard_body
    assert "state.backendOk" in readiness_body
    assert "currentPipelineDispatchStatus(currentPayload)" in readiness_body
    assert "canRun: true" in readiness_body
    assert "Execution readiness is still loading." in submit_body
    assert "createToast(DISPATCH_BACKEND_OFFLINE_MESSAGE, 'error');" in submit_body
    assert "Pipeline is blocked by missing prerequisites." in submit_body
    assert "mock simulation" not in submit_body


def test_portal_reconciles_restored_build_surface_values_without_field_events() -> None:
    content = _portal_bundle_content()
    reconcile_body = _extract_js_function_body(content, "reconcileBuildSurfaceFromDom")
    init_body = _extract_js_function_body(content, "init")

    assert "els.archiveIndexPath ? els.archiveIndexPath.value" in reconcile_body
    assert "els.rightsManifestPath ? els.rightsManifestPath.value" in reconcile_body
    assert "renderCLI();" in reconcile_body
    assert "scheduleConfigPreview(true);" in reconcile_body
    assert "window.addEventListener('pageshow'" in content
    assert "window.addEventListener('focus'" in content
    assert "reconcileBuildSurfaceFromDom();" in init_body


def test_argv_archive_gate_a_defaults_to_fixity_scan_runner() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[0] == sys.executable
    assert argv[1].endswith("tools/archive_governance.py")
    assert argv[2] == "--json"
    assert argv[3] == "fixity-scan"
    expected_archive_index = str(
        (orchestrator_app.REPO_ROOT / "archive_reports" / "archive_index_normalized.csv.gz").resolve()
    )
    expected_archive_root = str((orchestrator_app.REPO_ROOT / "archive_root").resolve())
    expected_out_dir = str((orchestrator_app.REPO_ROOT / "archive_reports").resolve())
    assert _flag_value(argv, "--archive-index") == expected_archive_index
    assert _flag_value(argv, "--archive-root") == expected_archive_root
    assert _flag_value(argv, "--out-dir") == expected_out_dir
    assert "--strict" in argv
    assert "--strict-identity" in argv
    assert "--no-strict" not in argv
    assert "--no-strict-identity" not in argv


def test_argv_archive_gate_b_allows_command_override_and_sign_maps_to_bagit_validation() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-b",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "bag-validate",
            "sign": True,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[3] == "bag-validate"
    expected_bag_dir = str((orchestrator_app.REPO_ROOT / "archive_reports" / "bag").resolve())
    assert _flag_value(argv, "--bag-dir") == expected_bag_dir
    assert "--validate-with-bagit-python" in argv


def test_argv_archive_gate_fixity_verify_uses_canonical_default_report_filename() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "fixity-verify",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[3] == "fixity-verify"
    expected_report_path = str((orchestrator_app.REPO_ROOT / "archive_reports" / "verification_report.json").resolve())
    assert _flag_value(argv, "--report-path") == expected_report_path


def test_argv_archive_gate_manifest_build_defaults_archive_index_under_output_dir() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "manifest-build",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[3] == "manifest-build"
    expected_archive_index = str(
        (orchestrator_app.REPO_ROOT / "archive_reports" / "archive_index_normalized.csv.gz").resolve()
    )
    assert _flag_value(argv, "--archive-index") == expected_archive_index


def test_argv_archive_gate_preserves_explicit_archive_index_override() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "fixity-scan",
            "archive_index": "./archive_root/custom_archive_index.csv.gz",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    expected_archive_index = str((orchestrator_app.REPO_ROOT / "archive_root" / "custom_archive_index.csv.gz").resolve())
    assert _flag_value(argv, "--archive-index") == expected_archive_index


def test_argv_repairs_repo_local_leading_slash_paths() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "/tests/fixtures/archive_small/archive_root",
            "output_dir": "/tests/fixtures/portal_runtime_output/lux_depth_repo_local_repair",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--input-dir") == str(
        (orchestrator_app.REPO_ROOT / "tests" / "fixtures" / "archive_small" / "archive_root").resolve()
    )
    assert _flag_value(argv, "--output-dir") == str(
        (orchestrator_app.REPO_ROOT / "tests" / "fixtures" / "portal_runtime_output" / "lux_depth_repo_local_repair").resolve()
    )


def test_argv_preserves_valid_absolute_allow_root_paths(tmp_path: Path) -> None:
    input_dir = (tmp_path / "input_abs").resolve()
    output_dir = (tmp_path / "output_abs").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--input-dir") == str(input_dir)
    assert _flag_value(argv, "--output-dir") == str(output_dir)


def test_argv_rejects_repo_local_shorthand_with_traversal_segments() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "/tests/../output",
            "output_dir": "./output",
        },
    }

    with pytest.raises(ValueError, match="Path shorthand traversal disallowed"):
        orchestrator_app._argv_from_request(payload)


def test_argv_archive_gate_rejects_workers_below_minimum() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "fixity-scan",
            "workers": 0,
        },
    }

    with pytest.raises(ValueError, match="Invalid archive integer option"):
        orchestrator_app._argv_from_request(payload)


def test_argv_archive_gate_rejects_negative_verify_sample() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "fixity-verify",
            "verify_sample": -1,
        },
    }

    with pytest.raises(ValueError, match="Invalid archive integer option"):
        orchestrator_app._argv_from_request(payload)


def test_argv_archive_gate_invalid_command_is_rejected() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-c",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "bag-build",
        },
    }

    with pytest.raises(ValueError, match="Invalid archive_command"):
        orchestrator_app._argv_from_request(payload)


def test_archive_gate_a_readiness_is_degraded_until_archive_index_is_supplied() -> None:
    readiness = orchestrator_app._archive_gate_readiness("archive-gate-a", require_dispatch_inputs=False)

    assert readiness["status"] == "degraded"
    assert readiness["canonical_command"] == "fixity-scan"
    assert readiness["missing_prerequisites"][0]["reason"] == "archive_index_required"


def test_archive_gate_a_readiness_blocks_fixture_index_against_nonmatching_root(tmp_path: Path) -> None:
    raw_root = tmp_path / "Raw_16-bit_Source"
    raw_root.mkdir()
    (raw_root / "DJI_0018.DNG").write_bytes(b"raw")
    fixture_index = orchestrator_app.REPO_ROOT / "tests" / "fixtures" / "archive_small" / "archive_index_normalized.csv.gz"

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-a",
        args={
            "input_dir": str(raw_root),
            "output_dir": str(tmp_path / "out"),
            "archive_command": "fixity-scan",
            "archive_root": str(raw_root),
            "archive_index": str(fixture_index),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    issues = {item["reason"]: item for item in readiness["missing_prerequisites"]}
    assert issues["archive_index_root_mismatch"]["field"] == "archive_index"
    assert "3/3 rows blocked" in issues["archive_index_root_mismatch"]["message"]


def test_archive_gate_a_readiness_is_ready_when_index_matches_root(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    (archive_root / "asset-001.dng").write_bytes(b"raw")
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-a",
        args={
            "input_dir": str(archive_root),
            "output_dir": str(tmp_path / "out"),
            "archive_command": "fixity-scan",
            "archive_root": str(archive_root),
            "archive_index": str(archive_index),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["missing_prerequisites"] == []


def test_archive_gate_a_readiness_reports_missing_archive_root_on_root_field(tmp_path: Path) -> None:
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])
    missing_root = tmp_path / "missing-root"

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-a",
        args={
            "input_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
            "archive_command": "fixity-scan",
            "archive_root": str(missing_root),
            "archive_index": str(archive_index),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    issues = {item["field"]: item for item in readiness["missing_prerequisites"] if "field" in item}
    assert issues["archive_root"]["reason"] == "input_dir_required"
    assert "archive_index_root_mismatch" not in {item["reason"] for item in readiness["missing_prerequisites"]}


def test_archive_gate_a_readiness_rejects_symlink_archive_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    (real_root / "asset-001.dng").write_bytes(b"raw")
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])
    link_root = tmp_path / "link-root"
    try:
        link_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-a",
        args={
            "input_dir": str(real_root),
            "output_dir": str(tmp_path / "out"),
            "archive_command": "fixity-scan",
            "archive_root": str(link_root),
            "archive_index": str(archive_index),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    issues = {item["field"]: item for item in readiness["missing_prerequisites"] if "field" in item}
    assert issues["archive_root"]["reason"] == "unsafe_path"
    assert "archive_index_root_mismatch" not in {item["reason"] for item in readiness["missing_prerequisites"]}


@pytest.mark.parametrize(
    ("relpath", "reason"),
    [
        ("../escape.dng", "parent_traversal"),
        ("/absolute.dng", "absolute_relpath"),
        ("C:/absolute.dng", "drive_prefixed_relpath"),
        ("", "empty_relpath"),
        ("missing.dng", "missing"),
        ("folder", "directory"),
    ],
)
def test_archive_index_preflight_rejects_invalid_relpaths(tmp_path: Path, relpath: str, reason: str) -> None:
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    (archive_root / "folder").mkdir()
    archive_index = tmp_path / f"archive_index_{reason}.csv.gz"
    _write_archive_index(archive_index, [relpath])

    result = orchestrator_app._validate_archive_index_against_root(
        archive_index,
        archive_root,
        scan_mode="full",
    )

    assert result["ok"] is False
    assert result["blocked_rows"] == 1
    assert result["examples"][0]["reason"] == reason


def test_archive_index_preflight_rejects_symlink_relpath(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    outside = tmp_path / "outside.dng"
    outside.write_bytes(b"outside")
    link_path = archive_root / "asset-link.dng"
    try:
        link_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    archive_index = tmp_path / "archive_index_symlink.csv.gz"
    _write_archive_index(archive_index, ["asset-link.dng"])

    result = orchestrator_app._validate_archive_index_against_root(
        archive_index,
        archive_root,
        scan_mode="full",
    )

    assert result["ok"] is False
    assert result["examples"][0]["reason"] == "symlink_traversal"


def test_archive_index_preflight_rejects_missing_columns_and_bad_gzip(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    missing_columns = tmp_path / "archive_index_missing_columns.csv"
    missing_columns.write_text("relpath\nasset-001.dng\n", encoding="utf-8")
    bad_gzip = tmp_path / "archive_index_bad.csv.gz"
    bad_gzip.write_bytes(b"not a gzip stream")

    missing_columns_result = orchestrator_app._validate_archive_index_against_root(
        missing_columns,
        archive_root,
        scan_mode="full",
    )
    bad_gzip_result = orchestrator_app._validate_archive_index_against_root(
        bad_gzip,
        archive_root,
        scan_mode="full",
    )

    assert missing_columns_result["ok"] is False
    assert missing_columns_result["examples"][0]["reason"].startswith("missing_columns:")
    assert bad_gzip_result["ok"] is False
    assert bad_gzip_result["examples"][0]["reason"] == "archive_index_unreadable:BadGzipFile"


def test_archive_index_preflight_preview_is_bounded_and_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    existing_relpaths = [f"asset-{idx:03d}.dng" for idx in range(orchestrator_app.ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT)]
    for relpath in existing_relpaths:
        (archive_root / relpath).write_bytes(b"raw")
    relpaths = [*existing_relpaths, "late-missing.dng"]
    archive_index = tmp_path / "archive_index_bounded.csv.gz"
    _write_archive_index(archive_index, relpaths)

    with orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK:
        orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE.clear()

    preview_result = orchestrator_app._validate_archive_index_against_root(
        archive_index,
        archive_root,
        scan_mode="preview",
    )
    assert preview_result["ok"] is True
    assert preview_result["truncated"] is True
    assert preview_result["rows_total"] == orchestrator_app.ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT

    def fail_if_rescanned(*_args, **_kwargs):
        raise AssertionError("cached preview result should avoid rescanning rows")

    monkeypatch.setattr(orchestrator_app, "_validate_archive_index_relpath", fail_if_rescanned)
    cached_preview = orchestrator_app._validate_archive_index_against_root(
        archive_index,
        archive_root,
        scan_mode="preview",
    )
    assert cached_preview == preview_result

    monkeypatch.undo()
    full_result = orchestrator_app._validate_archive_index_against_root(
        archive_index,
        archive_root,
        scan_mode="full",
    )
    assert full_result["ok"] is False
    assert full_result["examples"][0]["reason"] == "missing"


def test_archive_gate_b_readiness_fails_closed_without_manifest_jsonl() -> None:
    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-b",
        args={"input_dir": "./archive_root", "output_dir": "./archive_reports"},
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    assert readiness["canonical_command"] == "bag-build"
    assert readiness["missing_prerequisites"][0]["reason"] == "rights_manifest_required"
    assert readiness["missing_prerequisites"][0]["field"] == "manifest_jsonl"


def test_archive_gate_b_readiness_is_ready_when_manifest_jsonl_exists(tmp_path: Path) -> None:
    manifest_jsonl = tmp_path / "archive_manifest_v2.rights.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-b",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "bag-build",
            "manifest_jsonl": str(manifest_jsonl),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "bag-build"
    assert readiness["missing_prerequisites"] == []


def test_archive_gate_readiness_blocks_unsafe_input_dir() -> None:
    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-b",
        args={
            "input_dir": "~/.ssh",
            "output_dir": "./archive_reports",
            "archive_command": "bag-build",
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    assert readiness["missing_prerequisites"][0]["reason"] == "unsafe_path"
    assert readiness["missing_prerequisites"][0]["field"] == "input_dir"


def test_validate_existing_path_accepts_trusted_regular_file(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    checkpoint_path = allowed_root / "sam2.pt"
    allowed_root.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")

    resolved, issue = orchestrator_app._validate_existing_path(
        str(checkpoint_path),
        field="sam2_checkpoint_path",
        allowed_roots=[allowed_root],
        missing_reason="missing_checkpoint",
        missing_message="Checkpoint required.",
        expected_type="file",
        required=True,
    )

    assert resolved == str(checkpoint_path.resolve())
    assert issue is None


def test_validate_existing_path_accepts_trusted_directory(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    input_dir = allowed_root / "inputs"
    allowed_root.mkdir()
    input_dir.mkdir()

    resolved, issue = orchestrator_app._validate_existing_path(
        str(input_dir),
        field="input_dir",
        allowed_roots=[allowed_root],
        missing_reason="input_dir_required",
        missing_message="Input directory required.",
        expected_type="dir",
        required=True,
    )

    assert resolved == str(input_dir.resolve())
    assert issue is None


def test_validate_existing_path_preserves_missing_reason_for_path_inside_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    missing_file = allowed_root / "missing.pt"
    allowed_root.mkdir()

    resolved, issue = orchestrator_app._validate_existing_path(
        str(missing_file),
        field="sam2_checkpoint_path",
        allowed_roots=[allowed_root],
        missing_reason="missing_checkpoint",
        missing_message="Checkpoint required.",
        expected_type="file",
        required=True,
    )

    assert resolved == str(missing_file.resolve())
    assert issue is not None
    assert issue["reason"] == "missing_checkpoint"
    assert issue["path"] == str(missing_file.resolve())


def test_validate_existing_path_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    outside_file = outside_root / "secret.pt"
    symlink_path = allowed_root / "escape.pt"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_file.write_bytes(b"secret")
    try:
        symlink_path.symlink_to(outside_file)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")

    resolved, issue = orchestrator_app._validate_existing_path(
        str(symlink_path),
        field="sam2_checkpoint_path",
        allowed_roots=[allowed_root],
        missing_reason="missing_checkpoint",
        missing_message="Checkpoint required.",
        expected_type="file",
        required=True,
    )

    assert resolved is None
    assert issue is not None
    assert issue["reason"] == "unsafe_path"
    assert issue["field"] == "sam2_checkpoint_path"
    assert issue["path"] == str(symlink_path)


# --- Archive Gate E2E Test Extensions (Phase 2) ---


def test_archive_gate_a_fixity_verify_requires_hash_manifest(tmp_path: Path) -> None:
    """Gate A fixity-verify command requires an existing hash_manifest file."""
    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-a",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "fixity-verify",
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    # canonical_command is always the default for the pipeline
    assert readiness["canonical_command"] == "fixity-scan"
    # The actual requested command is in runner_details
    assert "fixity-verify" in readiness["runner_details"]["command"]
    assert any(p["reason"] == "hash_manifest_required" for p in readiness["missing_prerequisites"])


def test_archive_gate_a_fixity_verify_ready_when_hash_manifest_exists(tmp_path: Path) -> None:
    """Gate A fixity-verify becomes ready once hash_manifest is provided."""
    hash_manifest = tmp_path / "hash_manifest.csv.gz"
    hash_manifest.write_bytes(b"fixture-manifest")

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-a",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "fixity-verify",
            "hash_manifest": str(hash_manifest),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "fixity-scan"  # canonical is always the default
    assert "fixity-verify" in readiness["runner_details"]["command"]
    assert readiness["missing_prerequisites"] == []


def test_archive_gate_a_rights_apply_requires_manifest_and_policy(tmp_path: Path) -> None:
    """Gate A rights-apply command requires manifest_jsonl and policy_yaml files."""
    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-a",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "rights-apply",
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    assert readiness["canonical_command"] == "fixity-scan"  # canonical is always the default
    assert "rights-apply" in readiness["runner_details"]["command"]
    prerequisites = {p["field"]: p["reason"] for p in readiness["missing_prerequisites"]}
    assert "manifest_jsonl" in prerequisites
    assert "policy_yaml" in prerequisites


def test_archive_gate_a_rights_apply_ready_when_inputs_exist(tmp_path: Path) -> None:
    """Gate A rights-apply becomes ready once all required inputs are provided."""
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")
    policy_yaml = tmp_path / "rights_flags.yml"
    policy_yaml.write_text(
        "version: 1\n" "default_owner: archive-ops\n" "default_flags:\n" "  - review_required\n",
        encoding="utf-8",
    )

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-a",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "rights-apply",
            "manifest_jsonl": str(manifest_jsonl),
            "policy_yaml": str(policy_yaml),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "fixity-scan"  # canonical is always the default
    assert "rights-apply" in readiness["runner_details"]["command"]
    assert readiness["missing_prerequisites"] == []


def test_archive_gate_b_dedup_plan_requires_manifest_jsonl(tmp_path: Path) -> None:
    """Gate B dedup-plan command requires manifest_jsonl file."""
    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-b",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "dedup-plan",
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    assert readiness["canonical_command"] == "bag-build"  # canonical is always the default
    assert "dedup-plan" in readiness["runner_details"]["command"]
    assert any(p["field"] == "manifest_jsonl" for p in readiness["missing_prerequisites"])


def test_archive_gate_b_dedup_plan_ready_when_manifest_exists(tmp_path: Path) -> None:
    """Gate B dedup-plan becomes ready once manifest_jsonl is provided."""
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-b",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "dedup-plan",
            "manifest_jsonl": str(manifest_jsonl),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "bag-build"  # canonical is always the default
    assert "dedup-plan" in readiness["runner_details"]["command"]
    assert readiness["missing_prerequisites"] == []


def test_archive_gate_b_bag_validate_requires_bag_dir(tmp_path: Path) -> None:
    """Gate B bag-validate command requires an existing bag_dir."""
    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-b",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "bag-validate",
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    assert readiness["canonical_command"] == "bag-build"  # canonical is always the default
    assert "bag-validate" in readiness["runner_details"]["command"]
    assert any(p["field"] == "bag_dir" for p in readiness["missing_prerequisites"])


def test_archive_gate_b_bag_validate_ready_when_bag_dir_exists(tmp_path: Path) -> None:
    """Gate B bag-validate becomes ready once bag_dir is provided."""
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-b",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "bag-validate",
            "bag_dir": str(bag_dir),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "bag-build"  # canonical is always the default
    assert "bag-validate" in readiness["runner_details"]["command"]
    assert readiness["missing_prerequisites"] == []


def test_archive_gate_c_prov_export_requires_manifest_jsonl(tmp_path: Path) -> None:
    """Gate C prov-export command requires manifest_jsonl file."""
    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-c",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "prov-export",
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    assert readiness["canonical_command"] == "mets-export"  # canonical is always the default
    assert "prov-export" in readiness["runner_details"]["command"]
    assert any(p["field"] == "manifest_jsonl" for p in readiness["missing_prerequisites"])


def test_archive_gate_c_prov_export_ready_when_manifest_exists(tmp_path: Path) -> None:
    """Gate C prov-export becomes ready once manifest_jsonl is provided."""
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-c",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "prov-export",
            "manifest_jsonl": str(manifest_jsonl),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "mets-export"  # canonical is always the default
    assert "prov-export" in readiness["runner_details"]["command"]
    assert readiness["missing_prerequisites"] == []


def test_archive_gate_c_stac_export_requires_manifest_jsonl(tmp_path: Path) -> None:
    """Gate C stac-export command requires manifest_jsonl file."""
    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-c",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "stac-export",
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "blocked"
    assert readiness["canonical_command"] == "mets-export"  # canonical is always the default
    assert "stac-export" in readiness["runner_details"]["command"]
    assert any(p["field"] == "manifest_jsonl" for p in readiness["missing_prerequisites"])


def test_archive_gate_c_stac_export_ready_when_manifest_exists(tmp_path: Path) -> None:
    """Gate C stac-export becomes ready once manifest_jsonl is provided."""
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-c",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "stac-export",
            "manifest_jsonl": str(manifest_jsonl),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "mets-export"  # canonical is always the default
    assert "stac-export" in readiness["runner_details"]["command"]
    assert readiness["missing_prerequisites"] == []


def test_archive_gate_c_mets_export_ready_when_manifest_exists(tmp_path: Path) -> None:
    """Gate C mets-export becomes ready once manifest_jsonl is provided."""
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")

    readiness = orchestrator_app._archive_gate_readiness(
        "archive-gate-c",
        args={
            "input_dir": str(tmp_path / "archive_root"),
            "output_dir": str(tmp_path / "archive_reports"),
            "archive_command": "mets-export",
            "manifest_jsonl": str(manifest_jsonl),
        },
        require_dispatch_inputs=True,
    )

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "mets-export"
    assert readiness["missing_prerequisites"] == []


def test_lux_depth_readiness_separates_base_ready_from_canary_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator_app,
        "_module_available",
        lambda module_name: module_name == orchestrator_app.LUX_DEPTH_MODULE,
    )
    monkeypatch.setattr(orchestrator_app, "_resolve_lux_depth_canary_runtime", lambda: None)

    readiness = orchestrator_app._lux_depth_readiness()

    assert readiness["status"] == "ready"
    assert readiness["canonical_command"] == "lux-depth-v3"
    assert readiness["canary_status"] == "unavailable"
    assert readiness["missing_prerequisites"] == []


def test_lux_depth_readiness_blocks_selected_da3_when_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator_app,
        "_module_available",
        lambda module_name: module_name == orchestrator_app.LUX_DEPTH_MODULE,
    )
    monkeypatch.setattr(orchestrator_app, "_resolve_lux_depth_canary_runtime", lambda: None)

    readiness = orchestrator_app._lux_depth_readiness(
        {
            "depth_backend": "da3",
            "model_key": "da3-metric",
            "input_dir": "./input_images",
            "output_dir": "./output",
        }
    )

    assert readiness["status"] == "blocked"
    assert readiness["selected_model"]["model_key"] == "da3-metric"
    assert readiness["selected_model"]["canonical_model_key"] == "da3_metric"
    assert readiness["selected_model"]["runtime_available"] is False
    assert any(item["reason"] == "da3_runtime_unavailable" for item in readiness["missing_prerequisites"])


def test_run_job_is_async_and_does_not_block_event_loop() -> None:
    async def scenario() -> None:
        job = orchestrator_app.Job(id="job_async", created_at=orchestrator_app._now())
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        ticks = 0
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        runner = asyncio.create_task(
            orchestrator_app._run_job(
                job,
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys,time;print('progress=10%');sys.stdout.flush();time.sleep(0.15);"
                    "print('progress=100%');sys.stdout.flush()",
                ],
            )
        )
        ticker_task = asyncio.create_task(ticker())

        await asyncio.wait_for(runner, timeout=5)
        stop.set()
        await asyncio.wait_for(ticker_task, timeout=1)

        assert ticks > 5
        assert job.state == "succeeded"
        assert job.exit_code == 0
        assert job.progress == 100

    asyncio.run(scenario())


def test_cancel_request_terminates_running_job() -> None:
    async def scenario() -> None:
        job = orchestrator_app.Job(id="job_cancel", created_at=orchestrator_app._now())
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        runner = asyncio.create_task(
            orchestrator_app._run_job(
                job,
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys,time;print('progress=1%');sys.stdout.flush();time.sleep(30)",
                ],
            )
        )

        for _ in range(200):
            if job.proc is not None:
                break
            await asyncio.sleep(0.01)

        assert job.proc is not None
        await orchestrator_app._request_cancel(job)
        await asyncio.wait_for(runner, timeout=5)

        assert job.cancel_requested is True
        assert job.state == "canceled"

    asyncio.run(scenario())


def test_sse_broadcast_delivers_events_to_multiple_subscribers() -> None:
    async def scenario() -> None:
        job = orchestrator_app.Job(id="job_sse", created_at=orchestrator_app._now(), state="running")
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        req_one = _FakeRequest()
        req_two = _FakeRequest()
        response_one = await orchestrator_app.job_events(req_one, job.id)
        response_two = await orchestrator_app.job_events(req_two, job.id)

        async def collect(response) -> str:
            chunks = []
            saw_done = False
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                if "event: done" in chunk:
                    saw_done = True
            assert saw_done is True
            return "".join(chunks)

        task_one = asyncio.create_task(collect(response_one))
        task_two = asyncio.create_task(collect(response_two))

        await asyncio.sleep(0)
        await orchestrator_app._publish_event(job.id, "progress", {"id": job.id, "progress": 42})
        await orchestrator_app._publish_event(job.id, "done", {"id": job.id, "state": "succeeded", "exit_code": 0})

        out_one, out_two = await asyncio.wait_for(asyncio.gather(task_one, task_two), timeout=3)
        assert "event: progress" in out_one
        assert "event: progress" in out_two
        assert "event: done" in out_one
        assert "event: done" in out_two
        assert orchestrator_app.EVENT_SUBSCRIBERS[job.id] == {}

    asyncio.run(scenario())


def test_sse_disconnect_cleans_up_subscriber_queue() -> None:
    async def scenario() -> None:
        job = orchestrator_app.Job(id="job_disconnect", created_at=orchestrator_app._now(), state="running")
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        request = _FakeRequest()
        response = await orchestrator_app.job_events(request, job.id)

        async def read_until_disconnect() -> None:
            async for _chunk in response.body_iterator:
                request.disconnected = True

        await asyncio.wait_for(read_until_disconnect(), timeout=3)
        assert orchestrator_app.EVENT_SUBSCRIBERS[job.id] == {}

    asyncio.run(scenario())


def test_cleanup_expired_jobs_prunes_old_finished_entries() -> None:
    now = orchestrator_app._now()
    old_job = orchestrator_app.Job(
        id="job_old",
        created_at=now - 5000,
        finished_at=now - orchestrator_app.JOB_RETENTION_SECONDS - 10,
        state="succeeded",
    )
    fresh_job = orchestrator_app.Job(
        id="job_fresh",
        created_at=now - 120,
        finished_at=now - 30,
        state="succeeded",
    )
    running_job = orchestrator_app.Job(id="job_running", created_at=now, state="running")

    orchestrator_app.JOBS[old_job.id] = old_job
    orchestrator_app.JOBS[fresh_job.id] = fresh_job
    orchestrator_app.JOBS[running_job.id] = running_job

    orchestrator_app.EVENT_SUBSCRIBERS[old_job.id] = {"s1": asyncio.Queue()}
    orchestrator_app.EVENT_SUBSCRIBERS[fresh_job.id] = {"s2": asyncio.Queue()}

    orchestrator_app._cleanup_expired_jobs(now)

    assert old_job.id not in orchestrator_app.JOBS
    assert old_job.id not in orchestrator_app.EVENT_SUBSCRIBERS
    assert fresh_job.id in orchestrator_app.JOBS
    assert running_job.id in orchestrator_app.JOBS


def test_mutating_job_route_detection() -> None:
    assert orchestrator_app._is_mutating_job_endpoint("POST", "/v1/jobs") is True
    assert orchestrator_app._is_mutating_job_endpoint("POST", "/v1/jobs/job_123/cancel") is True
    assert orchestrator_app._is_mutating_job_endpoint("POST", "/v2/jobs") is True
    assert orchestrator_app._is_mutating_job_endpoint("POST", "/v2/jobs/job_123/cancel") is True
    assert orchestrator_app._is_mutating_job_endpoint("GET", "/v1/jobs/job_123") is False
    assert orchestrator_app._is_mutating_job_endpoint("GET", "/v1/jobs/job_123/events") is False
    assert orchestrator_app._is_mutating_job_endpoint("GET", "/v2/jobs/job_123") is False
    assert orchestrator_app._is_mutating_job_endpoint("GET", "/v2/jobs/job_123/events") is False


def test_extract_client_ip_does_not_trust_forwarded_header_by_default() -> None:
    previous_trust = orchestrator_app.TRUST_X_FORWARDED_FOR
    previous_proxies = orchestrator_app.TRUSTED_PROXY_IPS
    try:
        orchestrator_app.TRUST_X_FORWARDED_FOR = False
        orchestrator_app.TRUSTED_PROXY_IPS = set()
        request = _build_request(
            "POST",
            "/v1/jobs",
            headers={"x-forwarded-for": "198.51.100.8, 203.0.113.3"},
            client_host="10.0.0.9",
        )
        assert orchestrator_app._extract_client_ip(request) == "10.0.0.9"
    finally:
        orchestrator_app.TRUST_X_FORWARDED_FOR = previous_trust
        orchestrator_app.TRUSTED_PROXY_IPS = previous_proxies


def test_extract_client_ip_trusts_forwarded_header_for_trusted_proxy() -> None:
    previous_trust = orchestrator_app.TRUST_X_FORWARDED_FOR
    previous_proxies = orchestrator_app.TRUSTED_PROXY_IPS
    try:
        orchestrator_app.TRUST_X_FORWARDED_FOR = False
        orchestrator_app.TRUSTED_PROXY_IPS = {"10.0.0.9"}
        request = _build_request(
            "POST",
            "/v1/jobs",
            headers={"x-forwarded-for": "198.51.100.8, 203.0.113.3"},
            client_host="10.0.0.9",
        )
        assert orchestrator_app._extract_client_ip(request) == "198.51.100.8"
    finally:
        orchestrator_app.TRUST_X_FORWARDED_FOR = previous_trust
        orchestrator_app.TRUSTED_PROXY_IPS = previous_proxies


def test_api_key_validation_accepts_header_and_bearer() -> None:
    previous_key = orchestrator_app.API_KEY_SECRET
    previous_header = orchestrator_app.API_KEY_HEADER
    try:
        orchestrator_app.API_KEY_SECRET = "test-api-key"
        orchestrator_app.API_KEY_HEADER = "x-api-key"
        missing = _build_request("POST", "/v1/jobs")
        header_ok = _build_request("POST", "/v1/jobs", headers={"x-api-key": "test-api-key"})
        bearer_ok = _build_request("POST", "/v1/jobs", headers={"authorization": "Bearer test-api-key"})
        wrong = _build_request("POST", "/v1/jobs", headers={"x-api-key": "wrong"})

        assert orchestrator_app._has_valid_api_key(missing) is False
        assert orchestrator_app._has_valid_api_key(header_ok) is True
        assert orchestrator_app._has_valid_api_key(bearer_ok) is True
        assert orchestrator_app._has_valid_api_key(wrong) is False
    finally:
        orchestrator_app.API_KEY_SECRET = previous_key
        orchestrator_app.API_KEY_HEADER = previous_header


def test_content_length_limit_blocks_oversized_payloads() -> None:
    previous_limit = orchestrator_app.MAX_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 64
        request = _build_request(
            "POST",
            "/v1/jobs",
            headers={"content-type": "application/json", "content-length": "256"},
        )
        response = orchestrator_app._enforce_content_length_limit(request)
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_limit

    assert response is not None
    assert response.status_code == 413


def test_request_body_limit_uses_upload_override_for_staging_path() -> None:
    previous_max_request_bytes = orchestrator_app.MAX_REQUEST_BYTES
    previous_max_upload_request_bytes = orchestrator_app.MAX_UPLOAD_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 256
        orchestrator_app.MAX_UPLOAD_REQUEST_BYTES = 64
        assert orchestrator_app._request_body_limit_bytes("/v1/jobs") == 256
        assert orchestrator_app._request_body_limit_bytes("/v2/jobs") == 256
        assert orchestrator_app._request_body_limit_bytes("/v1/uploads/staging") == 64
        assert orchestrator_app._public_http_error_message(413, "/v1/uploads/staging") == (
            "request body too large (max 64 bytes)"
        )
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_max_request_bytes
        orchestrator_app.MAX_UPLOAD_REQUEST_BYTES = previous_max_upload_request_bytes


def test_stream_body_limit_blocks_oversized_chunked_payloads() -> None:
    previous_limit = orchestrator_app.MAX_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 8
        request = _build_request("POST", "/v1/jobs", headers={"content-type": "application/json"})
        chunks = [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": False},
        ]

        async def receive():
            if chunks:
                return chunks.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        setattr(request, "_receive", receive)
        orchestrator_app._install_stream_body_limit(request)

        first = asyncio.run(request._receive())  # type: ignore[attr-defined]
        assert first["body"] == b"12345"

        with pytest.raises(orchestrator_app.HTTPException) as exc:
            asyncio.run(request._receive())  # type: ignore[attr-defined]
        assert exc.value.status_code == 413
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_limit


def test_stream_body_limit_uses_upload_override_for_chunked_uploads() -> None:
    previous_max_request_bytes = orchestrator_app.MAX_REQUEST_BYTES
    previous_max_upload_request_bytes = orchestrator_app.MAX_UPLOAD_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 64
        orchestrator_app.MAX_UPLOAD_REQUEST_BYTES = 8
        request = _build_request("POST", "/v1/uploads/staging", headers={"content-type": "application/octet-stream"})
        chunks = [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": False},
        ]

        async def receive():
            if chunks:
                return chunks.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        setattr(request, "_receive", receive)
        orchestrator_app._install_stream_body_limit(request)

        first = asyncio.run(request._receive())  # type: ignore[attr-defined]
        assert first["body"] == b"12345"

        with pytest.raises(orchestrator_app.HTTPException) as exc:
            asyncio.run(request._receive())  # type: ignore[attr-defined]
        assert exc.value.status_code == 413
        assert exc.value.detail == "request body too large (max 8 bytes)"
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_max_request_bytes
        orchestrator_app.MAX_UPLOAD_REQUEST_BYTES = previous_max_upload_request_bytes


def test_parse_portal_upload_multipart_streams_chunked_payloads() -> None:
    boundary = "tp-boundary"
    request = _build_request(
        "POST",
        "/v1/uploads/staging",
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    body = _build_multipart_form_body(
        boundary,
        fields=[
            (
                "client_manifest",
                json.dumps(
                    {
                        "schema": "tp.portal.upload_manifest.v1",
                        "files": [{"relative_path": "nested/sample.txt", "size_bytes": 11}],
                    }
                ),
            )
        ],
        files=[("files", "nested/sample.txt", b"hello world", "text/plain")],
    )
    chunks = [
        body[:17],
        body[17:53],
        body[53:111],
        body[111:167],
        body[167:],
    ]

    async def receive():
        if chunks:
            chunk = chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(chunks)}
        return {"type": "http.request", "body": b"", "more_body": False}

    setattr(request, "_receive", receive)

    payload = asyncio.run(orchestrator_app._parse_portal_upload_multipart(request))
    try:
        assert payload.client_manifest_raw is not None
        assert len(payload.uploads) == 1
        assert payload.uploads[0].filename == "nested/sample.txt"
        assert payload.uploads[0].stream.read() == b"hello world"
    finally:
        payload.close()


def test_stage_upload_batch_normalizes_paths_and_writes_deterministic_manifest(tmp_path: Path) -> None:
    result = upload_staging.stage_upload_batch(
        upload_root=tmp_path / "uploads",
        uploads=[
            upload_staging.IncomingUpload(filename="nested/sample.txt", stream=io.BytesIO(b"hello world")),
            upload_staging.IncomingUpload(filename="nested/child/readme.md", stream=io.BytesIO(b"# hi\n")),
        ],
        client_manifest_paths=["nested/sample.txt", "nested/child/readme.md"],
        capture_metadata_enabled=False,
        now=1234.0,
    )

    assert result.file_count == 2
    assert result.total_bytes == 16
    assert (result.input_dir / "nested" / "sample.txt").read_text(encoding="utf-8") == "hello world"
    baseline_manifest_payload = json.loads(result.baseline_manifest_path.read_text(encoding="utf-8"))
    assert baseline_manifest_payload["schema"] == "tp.meta.baseline_manifest.v1"
    assert baseline_manifest_payload["record_count"] == 2
    assert [record["relative_path"] for record in baseline_manifest_payload["records"]] == [
        "nested/child/readme.md",
        "nested/sample.txt",
    ]
    assert baseline_manifest_payload["records"][1]["sha256"] == hashlib.sha256(b"hello world").hexdigest()
    assert baseline_manifest_payload["records"][1]["mime_type"] == "text/plain"
    assert baseline_manifest_payload["records"][1]["media_kind"] == "text"
    assert "image" not in baseline_manifest_payload["records"][1]
    assert "pdf" not in baseline_manifest_payload["records"][1]
    assert json.loads(result.capture_metadata_path.read_text(encoding="utf-8")) == []


def test_stage_upload_batch_manifest_is_dependency_independent_for_known_extensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = upload_staging.stage_upload_batch(
        upload_root=tmp_path / "uploads-a",
        uploads=[upload_staging.IncomingUpload(filename="nested/sample.png", stream=io.BytesIO(b"png-bytes"))],
        client_manifest_paths=["nested/sample.png"],
        capture_metadata_enabled=False,
        now=111.0,
    )

    monkeypatch.setitem(upload_staging.__dict__, "Image", object())
    monkeypatch.setitem(upload_staging.__dict__, "PdfReader", object())
    second = upload_staging.stage_upload_batch(
        upload_root=tmp_path / "uploads-b",
        uploads=[upload_staging.IncomingUpload(filename="nested/sample.png", stream=io.BytesIO(b"png-bytes"))],
        client_manifest_paths=["nested/sample.png"],
        capture_metadata_enabled=False,
        now=222.0,
    )

    assert first.baseline_manifest_path.read_bytes() == second.baseline_manifest_path.read_bytes()


def test_cleanup_expired_batches_skips_non_managed_directories(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    expired_at = 100.0

    unrelated_dir = upload_root / "manual_batch"
    unrelated_dir.mkdir()
    (unrelated_dir / "notes.txt").write_text("keep", encoding="utf-8")
    os.utime(unrelated_dir, (expired_at, expired_at))

    managed_dir = upload_root / "upload_123"
    (managed_dir / "input").mkdir(parents=True)
    portal_dir = managed_dir / "_portal"
    portal_dir.mkdir()
    (portal_dir / upload_staging.UPLOAD_RECEIPT_FILENAME).write_text("{}", encoding="utf-8")
    os.utime(managed_dir, (expired_at, expired_at))

    removed = upload_staging.cleanup_expired_batches(
        upload_root,
        now=10_000.0,
        ttl_seconds=1.0,
        retained_input_dirs=[],
    )

    assert removed == ["upload_123"]
    assert unrelated_dir.exists()
    assert not managed_dir.exists()


def test_stage_upload_batch_rejects_duplicate_relative_paths_and_cleans_batch(tmp_path: Path) -> None:
    with pytest.raises(upload_staging.UploadStagingError) as exc:
        upload_staging.stage_upload_batch(
            upload_root=tmp_path / "uploads",
            uploads=[
                upload_staging.IncomingUpload(filename="nested/sample.txt", stream=io.BytesIO(b"one")),
                upload_staging.IncomingUpload(filename="nested/sample.txt", stream=io.BytesIO(b"two")),
            ],
            client_manifest_paths=["nested/sample.txt", "nested/sample.txt"],
            capture_metadata_enabled=False,
            now=4321.0,
        )

    assert exc.value.reason == "duplicate_relative_path"
    upload_root = tmp_path / "uploads"
    if upload_root.exists():
        assert list(upload_root.iterdir()) == []


def test_stage_upload_batch_capture_metadata_failure_falls_back_to_empty_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _explode(*_args, **_kwargs):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(upload_staging, "extract_capture_metadata_records", _explode)
    result = upload_staging.stage_upload_batch(
        upload_root=tmp_path / "uploads",
        uploads=[upload_staging.IncomingUpload(filename="sample.txt", stream=io.BytesIO(b"hello"))],
        client_manifest_paths=["sample.txt"],
        capture_metadata_enabled=True,
        now=9876.0,
    )

    receipt_payload = json.loads(result.upload_receipt_path.read_text(encoding="utf-8"))
    assert receipt_payload["summary"]["warnings"] == ["capture_metadata_extraction_failed"]
    assert receipt_payload["summary"]["top_level_roots"] == ["sample.txt"]
    assert result.received_at_epoch_seconds == 9876.0
    assert result.top_level_roots == ("sample.txt",)
    assert json.loads(result.capture_metadata_path.read_text(encoding="utf-8")) == []


def test_sanitized_child_env_redacts_secret_like_values() -> None:
    old_values = {
        "TP_API_KEY": os.environ.get("TP_API_KEY"),
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "PATH": os.environ.get("PATH"),
    }
    try:
        os.environ["TP_API_KEY"] = "tp-secret"
        os.environ["HF_TOKEN"] = "hf-secret"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "aws-secret"
        os.environ["PATH"] = "/usr/bin"
        child_env = orchestrator_app._sanitized_child_env()
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert "TP_API_KEY" not in child_env
    assert "HF_TOKEN" not in child_env
    assert "AWS_SECRET_ACCESS_KEY" not in child_env
    assert child_env["PATH"] == "/usr/bin"


def test_rate_limiting_returns_true_after_threshold() -> None:
    previous_limit = orchestrator_app.RATE_LIMIT_PER_MINUTE
    try:
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 1
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()
        now = orchestrator_app._now()
        first = orchestrator_app._is_rate_limited("127.0.0.1", now)
        second = orchestrator_app._is_rate_limited("127.0.0.1", now + 0.1)
    finally:
        orchestrator_app.RATE_LIMIT_PER_MINUTE = previous_limit
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()

    assert first is False
    assert second is True


def test_client_ip_prefers_peer_by_default() -> None:
    request = _build_request("GET", "/ready", headers={"x-forwarded-for": "203.0.113.9, 127.0.0.1"}, client_host="10.0.0.1")
    assert orchestrator_app._extract_client_ip(request) == "10.0.0.1"


def test_ready_verbose_reports_lux_runner_from_module_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    previous_verbose = orchestrator_app.READY_VERBOSE
    try:
        orchestrator_app.READY_VERBOSE = True
        monkeypatch.setattr(orchestrator_app, "_lux_depth_runner_available", lambda: True)
        payload = asyncio.run(orchestrator_app.ready())
    finally:
        orchestrator_app.READY_VERBOSE = previous_verbose

    assert payload["ok"] is True
    assert payload["cli"]["lux-depth-v3"] is True


def test_api_key_validation_rejects_query_param_by_default() -> None:
    previous_key = orchestrator_app.API_KEY_SECRET
    previous_flag = orchestrator_app.ALLOW_SSE_QUERY_API_KEY
    try:
        orchestrator_app.API_KEY_SECRET = "query-secret"
        orchestrator_app.ALLOW_SSE_QUERY_API_KEY = False
        request = _build_request("GET", "/v1/jobs/job_1/events", query_string="api_key=query-secret")
        assert orchestrator_app._has_valid_api_key(request) is False
    finally:
        orchestrator_app.API_KEY_SECRET = previous_key
        orchestrator_app.ALLOW_SSE_QUERY_API_KEY = previous_flag


def test_api_key_validation_accepts_query_param_when_explicitly_enabled() -> None:
    previous_key = orchestrator_app.API_KEY_SECRET
    previous_flag = orchestrator_app.ALLOW_SSE_QUERY_API_KEY
    try:
        orchestrator_app.API_KEY_SECRET = "query-secret"
        orchestrator_app.ALLOW_SSE_QUERY_API_KEY = True
        request = _build_request("GET", "/v1/jobs/job_1/events", query_string="api_key=query-secret")
        assert orchestrator_app._has_valid_api_key(request) is True
        v2_request = _build_request("GET", "/v2/jobs/job_1/events", query_string="api_key=query-secret")
        assert orchestrator_app._has_valid_api_key(v2_request) is True
    finally:
        orchestrator_app.API_KEY_SECRET = previous_key
        orchestrator_app.ALLOW_SSE_QUERY_API_KEY = previous_flag


def test_api_key_query_param_is_rejected_for_non_event_endpoints() -> None:
    previous_key = orchestrator_app.API_KEY_SECRET
    try:
        orchestrator_app.API_KEY_SECRET = "query-secret"
        request = _build_request("GET", "/v1/jobs", query_string="api_key=query-secret")
        assert orchestrator_app._has_valid_api_key(request) is False
    finally:
        orchestrator_app.API_KEY_SECRET = previous_key


def test_protected_job_route_detection() -> None:
    assert orchestrator_app._is_protected_job_endpoint("/v1/jobs") is True
    assert orchestrator_app._is_protected_job_endpoint("/v1/jobs/job_123") is True
    assert orchestrator_app._is_protected_job_endpoint("/v1/jobs/job_123/events") is True
    assert orchestrator_app._is_protected_job_endpoint("/v2/jobs") is True
    assert orchestrator_app._is_protected_job_endpoint("/v2/jobs/job_123") is True
    assert orchestrator_app._is_protected_job_endpoint("/v2/jobs/job_123/events") is True
    assert orchestrator_app._is_protected_job_endpoint("/ready") is False


def test_protected_api_key_route_detection() -> None:
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/jobs") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/v2/jobs") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/config-metadata") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/config-preview") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/portal/events") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/portal/rum") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/ready") is False


def test_portal_event_log_persistence_keeps_jsonl_parseable_under_concurrent_appends(tmp_path: Path) -> None:
    log_path = tmp_path / "portal-rum.jsonl"
    records = [
        {
            "schema": "tp.orchestrator.portal_rum.v1",
            "sequence": index,
            "event_type": "queue_request",
        }
        for index in range(32)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(orchestrator_app._persist_portal_event_record, record, log_path) for record in records]
        for future in futures:
            future.result()

    persisted_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(persisted_lines) == len(records)
    parsed_records = [json.loads(line) for line in persisted_lines]
    assert {item["sequence"] for item in parsed_records} == {record["sequence"] for record in records}


def test_index_job_artifacts_populates_job_payload(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "render.png").write_bytes(b"png")

    job = orchestrator_app.Job(
        id="job_artifacts",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    indexed = orchestrator_app._index_job_artifacts(job)

    assert len(indexed) == 2
    assert job.artifacts["output_dir"] == str(output_dir)
    assert {item["artifact_type"] for item in indexed} == {"metadata", "image"}
    assert {item["path"] for item in indexed} == {"manifest.json", "render.png"}
    render_item = next(item for item in indexed if item["path"] == "render.png")
    assert render_item["media_kind"] == "image"
    assert render_item["previewable"] is True
    assert render_item["browser_previewable"] is True
    assert render_item["content_type"] == "image/png"
    assert render_item["mime_type"] == "image/png"
    assert render_item["display_hint"]["role"] == "primary_preview"
    assert render_item["display_hint"]["priority"] == 1000
    assert render_item["display_hint"]["label"] == "Primary Preview"
    assert render_item["display_hint"]["compare_group"]
    assert render_item["url"] == f"/v1/jobs/{job.id}/artifacts/render.png"
    assert render_item["download_url"] == render_item["url"]
    assert "preview_url" not in render_item  # browser handles PNG natively
    manifest_item = next(item for item in indexed if item["path"] == "manifest.json")
    assert manifest_item["media_kind"] == "metadata"
    assert manifest_item["previewable"] is False
    assert manifest_item["browser_previewable"] is False
    assert manifest_item["display_hint"]["role"] == "manifest"
    assert manifest_item["display_hint"]["priority"] == 240
    assert manifest_item["display_hint"]["label"] == "Manifest"


def test_index_job_artifacts_marks_tiff_without_proxy_as_not_browser_previewable(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "render.tif").write_bytes(b"II*\x00")

    job = orchestrator_app.Job(
        id="job_tiff",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    indexed = orchestrator_app._index_job_artifacts(job)
    tiff_item = next(item for item in indexed if item["path"] == "render.tif")

    assert tiff_item["previewable"] is True  # legacy field stays as-is
    assert tiff_item["browser_previewable"] is False
    assert "preview_url" not in tiff_item


def test_index_job_artifacts_surfaces_tiff_preview_proxy_when_present(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "render.tif").write_bytes(b"II*\x00")
    (output_dir / "render.tif.preview.png").write_bytes(b"\x89PNG")

    job = orchestrator_app.Job(
        id="job_tiff_proxy",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    indexed = orchestrator_app._index_job_artifacts(job)
    tiff_item = next(item for item in indexed if item["path"] == "render.tif")

    assert tiff_item["browser_previewable"] is True
    assert tiff_item["preview_url"] == f"/v1/jobs/{job.id}/artifacts/render.tif.preview.png"
    assert tiff_item["preview_mime_type"] == "image/png"
    assert tiff_item["download_url"] == f"/v1/jobs/{job.id}/artifacts/render.tif"
    assert job.artifact_lookup["render.tif.preview.png"] == (output_dir / "render.tif.preview.png").resolve()

    job.artifact_lookup = {}
    hydrated = orchestrator_app._hydrate_artifact_lookup_from_items(job)
    assert hydrated["render.tif.preview.png"] == (output_dir / "render.tif.preview.png").resolve()


def test_index_job_artifacts_indexes_scoped_tiff_preview_proxy_for_download(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "render.tif").write_bytes(b"II*\x00")
    proxy_path = output_dir / "render.tif.preview.png"
    proxy_path.write_bytes(b"\x89PNG")
    run_card = output_dir / "run_card_2026-05-05_120000.json"
    run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-05-05_120000",
                "artifact_index": [
                    {"relative_path": "render.tif"},
                ],
            }
        ),
        encoding="utf-8",
    )

    job = orchestrator_app.Job(
        id="job_scoped_tiff_proxy",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    indexed = orchestrator_app._index_job_artifacts(job)
    tiff_item = next(item for item in indexed if item["path"] == "render.tif")

    assert tiff_item["browser_previewable"] is True
    assert tiff_item["preview_url"] == f"/v1/jobs/{job.id}/artifacts/render.tif.preview.png"
    assert job.artifact_lookup["render.tif.preview.png"] == proxy_path.resolve()


def test_index_job_artifacts_does_not_misclassify_catalog_metadata_as_log(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "catalog.json").write_text("{}", encoding="utf-8")
    (output_dir / "job.log").write_text("ok", encoding="utf-8")

    job = orchestrator_app.Job(
        id="job_artifacts_catalog",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )

    indexed = orchestrator_app._index_job_artifacts(job)

    catalog_item = next(item for item in indexed if item["path"] == "catalog.json")
    log_item = next(item for item in indexed if item["path"] == "job.log")

    assert catalog_item["media_kind"] == "metadata"
    assert catalog_item["display_hint"]["role"] == "metadata"
    assert catalog_item["display_hint"]["label"] == "Metadata"
    assert log_item["display_hint"]["role"] == "log"
    assert log_item["display_hint"]["label"] == "Log"


def test_index_job_artifacts_classifies_captioning_sidecar_only_as_advisory_caption(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    caption_dir = output_dir / "captioning"
    caption_dir.mkdir(parents=True, exist_ok=True)
    (caption_dir / "image.vlm_captioning.sidecar.json").write_text("{}", encoding="utf-8")
    (caption_dir / "image.vlm_captioning.raw.txt").write_text("SCENE=Pool", encoding="utf-8")
    (caption_dir / "image_proxy.png").write_bytes(b"png")

    job = orchestrator_app.Job(
        id="job_artifacts_captioning",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )

    indexed = orchestrator_app._index_job_artifacts(job)

    sidecar_item = next(item for item in indexed if item["path"] == "captioning/image.vlm_captioning.sidecar.json")
    raw_item = next(item for item in indexed if item["path"] == "captioning/image.vlm_captioning.raw.txt")
    proxy_item = next(item for item in indexed if item["path"] == "captioning/image_proxy.png")
    assert sidecar_item["display_hint"]["role"] == "vlm_caption"
    assert sidecar_item["display_hint"]["label"] == "Advisory Caption"
    assert raw_item["display_hint"]["role"] == "log"
    assert proxy_item["display_hint"]["role"] == "review_preview"
    for item in (sidecar_item, raw_item, proxy_item):
        assert not Path(item["path"]).is_absolute()
        assert str(output_dir) not in item["path"]
        assert item["relative_path"] == item["path"]


def test_index_job_artifacts_skips_entries_resolving_outside_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    escaped_target = tmp_path / "secret.png"
    escaped_target.write_bytes(b"secret")
    symlink_path = output_dir / "escape.png"
    symlink_path.symlink_to(escaped_target)

    job = orchestrator_app.Job(
        id="job_artifacts_symlink_skip",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )

    indexed = orchestrator_app._index_job_artifacts(job)

    assert indexed == []
    assert job.artifacts["items"] == []
    assert job.artifact_lookup == {}


def test_hydrate_artifact_lookup_from_items_reuses_existing_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "renders" / "hero.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"png")

    job = orchestrator_app.Job(
        id="job_artifacts_lookup",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        artifacts={
            "output_dir": str(output_dir),
            "items": [
                {
                    "path": "renders/hero.png",
                    "relative_path": "renders/hero.png",
                }
            ],
        },
    )

    lookup = orchestrator_app._hydrate_artifact_lookup_from_items(job)

    assert lookup["renders/hero.png"] == artifact_path.resolve()
    assert job.artifact_lookup["renders/hero.png"] == artifact_path.resolve()


def test_hydrate_artifact_lookup_from_items_registers_preview_proxy(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "renders" / "hero.tif"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"II*\x00")
    proxy_path = output_dir / "renders" / "hero.tif.preview.png"
    proxy_path.write_bytes(b"\x89PNG")

    job = orchestrator_app.Job(
        id="job_artifacts_lookup_proxy",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        artifacts={
            "output_dir": str(output_dir),
            "items": [
                {
                    "path": "renders/hero.tif",
                    "relative_path": "renders/hero.tif",
                    "preview_url": "/v1/jobs/job_artifacts_lookup_proxy/artifacts/renders/hero.tif.preview.png",
                }
            ],
        },
    )

    lookup = orchestrator_app._hydrate_artifact_lookup_from_items(job)

    assert lookup["renders/hero.tif"] == artifact_path.resolve()
    assert lookup["renders/hero.tif.preview.png"] == proxy_path.resolve()


def test_index_job_artifacts_truncation_is_sorted_and_stable(tmp_path: Path) -> None:
    previous_limit = orchestrator_app.MAX_INDEXED_ARTIFACTS
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "zeta.txt").write_text("z", encoding="utf-8")
    (output_dir / "alpha.txt").write_text("a", encoding="utf-8")
    (output_dir / "mid.txt").write_text("m", encoding="utf-8")

    try:
        orchestrator_app.MAX_INDEXED_ARTIFACTS = 2
        job = orchestrator_app.Job(
            id="job_artifacts_sorted",
            created_at=orchestrator_app._now(),
            request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        )
        indexed = orchestrator_app._index_job_artifacts(job)
    finally:
        orchestrator_app.MAX_INDEXED_ARTIFACTS = previous_limit

    assert [item["path"] for item in indexed] == ["alpha.txt", "mid.txt"]
    assert job.artifacts["truncated"] is True
    assert job.artifacts["indexed_count"] == 2


def test_index_job_artifacts_prefers_current_run_card_artifact_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    depth_dir = output_dir / "depth"
    manifests_dir = output_dir / "manifests"
    depth_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    current_depth = depth_dir / "current_depth.png"
    current_depth.write_bytes(b"current-depth")
    (depth_dir / "stale_depth.png").write_bytes(b"stale-depth")
    batch_manifest = manifests_dir / "batch_2026-04-09_132300.json"
    batch_manifest.write_text(json.dumps({"batch_id": "2026-04-09_132300", "results": []}), encoding="utf-8")
    run_card = output_dir / "run_card_2026-04-09_132300.json"
    run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-04-09_132300",
                "success_count": 1,
                "error_count": 0,
                "artifact_index": [
                    {"relative_path": "depth/current_depth.png"},
                    {"relative_path": "manifests/batch_2026-04-09_132300.json"},
                ],
            }
        ),
        encoding="utf-8",
    )

    job = orchestrator_app.Job(
        id="job_artifacts_scoped_run_card",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )

    indexed = orchestrator_app._index_job_artifacts(job)

    assert {item["path"] for item in indexed} == {
        "depth/current_depth.png",
        "manifests/batch_2026-04-09_132300.json",
        "run_card_2026-04-09_132300.json",
    }
    assert "depth/stale_depth.png" not in job.artifact_lookup


def test_index_job_artifacts_uses_current_batch_manifest_when_run_card_lacks_artifact_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    depth_dir = output_dir / "depth"
    manifests_dir = output_dir / "manifests"
    depth_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    (depth_dir / "stale_depth.png").write_bytes(b"stale-depth")
    batch_manifest = manifests_dir / "batch_2026-04-09_132300.json"
    batch_manifest.write_text(
        json.dumps(
            {
                "batch_id": "2026-04-09_132300",
                "results": [{"status": "error"}],
                "stats": {"total_images": 1},
            }
        ),
        encoding="utf-8",
    )
    run_card = output_dir / "run_card_2026-04-09_132300.json"
    run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-04-09_132300",
                "success_count": 0,
                "error_count": 1,
            }
        ),
        encoding="utf-8",
    )

    job = orchestrator_app.Job(
        id="job_artifacts_scoped_manifest",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )

    indexed = orchestrator_app._index_job_artifacts(job)

    assert {item["path"] for item in indexed} == {
        "manifests/batch_2026-04-09_132300.json",
    }
    assert "depth/stale_depth.png" not in job.artifact_lookup


def test_index_job_artifacts_prefers_batch_matched_run_card_over_newer_unmatched_run_card(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    depth_dir = output_dir / "depth"
    manifests_dir = output_dir / "manifests"
    depth_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    matched_depth = depth_dir / "matched_depth.png"
    unmatched_depth = depth_dir / "unmatched_depth.png"
    matched_depth.write_bytes(b"matched")
    unmatched_depth.write_bytes(b"unmatched")

    batch_manifest = manifests_dir / "batch_2026-04-09_132300.json"
    batch_manifest.write_text(json.dumps({"batch_id": "2026-04-09_132300", "results": []}), encoding="utf-8")

    matched_run_card = output_dir / "run_card_2026-04-09_132300.json"
    matched_run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-04-09_132300",
                "artifact_index": [{"relative_path": "depth/matched_depth.png"}],
            }
        ),
        encoding="utf-8",
    )
    unmatched_run_card = output_dir / "run_card_2026-04-10_120000.json"
    unmatched_run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-04-10_120000",
                "artifact_index": [{"relative_path": "depth/unmatched_depth.png"}],
            }
        ),
        encoding="utf-8",
    )
    now = orchestrator_app._now()
    os.utime(matched_run_card, (now - 5, now - 5))
    os.utime(unmatched_run_card, (now, now))

    job = orchestrator_app.Job(
        id="job_artifacts_prefers_matched_run_card",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )

    indexed = orchestrator_app._index_job_artifacts(job)

    assert {item["path"] for item in indexed} == {
        "depth/matched_depth.png",
        "run_card_2026-04-09_132300.json",
    }
    assert "depth/unmatched_depth.png" not in job.artifact_lookup


def test_refresh_job_run_summary_uses_current_output_metadata_not_scoped_items(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    old_run_card = output_dir / "run_card_2026-04-06_232022.json"
    old_run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-04-06_232022",
                "total_images": 5,
                "success_count": 4,
                "error_count": 1,
            }
        ),
        encoding="utf-8",
    )
    current_batch_manifest = manifests_dir / "batch_2026-04-09_132300.json"
    current_batch_manifest.write_text(
        json.dumps(
            {
                "batch_id": "2026-04-09_132300",
                "results": [{"status": "error"}] * 6,
                "stats": {"total_images": 6},
            }
        ),
        encoding="utf-8",
    )
    current_run_card = output_dir / "run_card_2026-04-09_132300.json"
    current_run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-04-09_132300",
                "total_images": 6,
                "success_count": 0,
                "error_count": 6,
                "artifact_index": [
                    {"relative_path": "manifests/batch_2026-04-09_132300.json"},
                ],
            }
        ),
        encoding="utf-8",
    )

    job = orchestrator_app.Job(
        id="job_run_summary_current_metadata",
        created_at=orchestrator_app._now(),
        state="failed",
        exit_code=1,
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        artifacts={
            "output_dir": str(output_dir),
            "items": [{"path": "run_card_2026-04-06_232022.json", "relative_path": "run_card_2026-04-06_232022.json"}],
            "indexed_count": 1,
            "truncated": False,
        },
        error={
            "code": "RUNNER_EXIT_NONZERO",
            "message": "runner exited with code 1",
            "details": {"exit_code": 1},
        },
    )

    summary = orchestrator_app._refresh_job_run_summary(job)

    assert summary["batch_id"] == "2026-04-09_132300"
    assert summary["success_count"] == 0
    assert summary["error_count"] == 6
    assert summary["partial"] is False
    assert summary["reviewable_outputs"] is False
    assert job.state == "failed"
    assert job.error["code"] == "RUNNER_EXIT_NONZERO"


def test_create_job_validation_uses_typed_error_envelope() -> None:
    response = asyncio.run(orchestrator_app.create_job({"pipeline": "unsupported", "args": {}}))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"


def test_create_job_archive_gate_invalid_command_uses_typed_error_envelope() -> None:
    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "archive-gate-a",
                "args": {"input_dir": "./input_images", "output_dir": "./output", "archive_command": "stac-export"},
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_archive_command"
    assert orchestrator_app.JOBS == {}


def test_create_job_archive_gate_invalid_integer_option_uses_typed_error_envelope(tmp_path: Path) -> None:
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    archive_index.write_bytes(b"fixture-index")
    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "archive-gate-a",
                "args": {
                    "input_dir": str(tmp_path / "archive_root"),
                    "output_dir": str(tmp_path / "output"),
                    "archive_command": "fixity-scan",
                    "archive_index": str(archive_index),
                    "workers": 0,
                },
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_archive_integer_option"
    assert orchestrator_app.JOBS == {}


def test_create_job_invalid_log_level_uses_typed_error_envelope() -> None:
    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "lux-depth-v3",
                "args": {
                    "input_dir": "./input_images",
                    "output_dir": "./output",
                    "log_level": "TRACE",
                },
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_log_level"
    assert "Traceback" not in response.body.decode("utf-8")
    assert orchestrator_app.JOBS == {}


def test_create_job_archive_gate_b_requires_manifest_jsonl_with_typed_error() -> None:
    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "archive-gate-b",
                "args": {
                    "input_dir": "./input_images",
                    "output_dir": "./output",
                    "archive_command": "bag-build",
                },
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["field"] == "manifest_jsonl"
    assert body["error"]["details"]["reason"] == "required"
    assert orchestrator_app.JOBS == {}


def test_argv_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    previous_path_roots = orchestrator_app.ALLOWED_PATH_ROOTS
    try:
        allowed_root = (tmp_path / "allowed").resolve()
        allowed_root.mkdir(parents=True, exist_ok=True)
        orchestrator_app.ALLOWED_INPUT_ROOTS = [allowed_root]
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = [allowed_root]
        orchestrator_app.ALLOWED_PATH_ROOTS = [allowed_root]

        payload: Dict[str, object] = {
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./input_images",
                "output_dir": "./output",
            },
        }

        with pytest.raises(ValueError, match="Path outside allowed roots"):
            orchestrator_app._argv_from_request(payload)
    finally:
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
        orchestrator_app.ALLOWED_PATH_ROOTS = previous_path_roots


def test_argv_archive_gate_rejects_output_path_under_input_root(tmp_path: Path) -> None:
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    previous_path_roots = orchestrator_app.ALLOWED_PATH_ROOTS
    try:
        input_root = (tmp_path / "input_root").resolve()
        output_root = (tmp_path / "output_root").resolve()
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        orchestrator_app.ALLOWED_INPUT_ROOTS = [input_root]
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = [output_root]
        orchestrator_app.ALLOWED_PATH_ROOTS = [input_root, output_root]

        payload: Dict[str, object] = {
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(input_root),
                "output_dir": str(output_root),
                "archive_command": "fixity-scan",
                "out_dir": str(input_root / "reports"),
            },
        }
        with pytest.raises(ValueError, match="Path outside allowed roots"):
            orchestrator_app._argv_from_request(payload)
    finally:
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
        orchestrator_app.ALLOWED_PATH_ROOTS = previous_path_roots


def test_argv_archive_gate_archive_index_default_accepts_output_root_via_allowed_path_roots(tmp_path: Path) -> None:
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    previous_path_roots = orchestrator_app.ALLOWED_PATH_ROOTS
    try:
        input_root = (tmp_path / "input_root").resolve()
        output_root = (tmp_path / "output_root").resolve()
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        orchestrator_app.ALLOWED_INPUT_ROOTS = [input_root]
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = [output_root]
        orchestrator_app.ALLOWED_PATH_ROOTS = [input_root, output_root]

        payload: Dict[str, object] = {
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(input_root),
                "output_dir": str(output_root),
                "archive_command": "fixity-scan",
            },
        }
        argv = orchestrator_app._argv_from_request(payload)
    finally:
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
        orchestrator_app.ALLOWED_PATH_ROOTS = previous_path_roots

    expected_archive_index = str((output_root / "archive_index_normalized.csv.gz").resolve())
    assert _flag_value(argv, "--archive-index") == expected_archive_index


def test_argv_archive_gate_manifest_build_keeps_hash_manifest_and_rights_scopes_narrow(tmp_path: Path) -> None:
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    previous_path_roots = orchestrator_app.ALLOWED_PATH_ROOTS
    try:
        input_root = (tmp_path / "input_root").resolve()
        output_root = (tmp_path / "output_root").resolve()
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        orchestrator_app.ALLOWED_INPUT_ROOTS = [input_root]
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = [output_root]
        orchestrator_app.ALLOWED_PATH_ROOTS = [input_root, output_root]

        payload_bad_hash: Dict[str, object] = {
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(input_root),
                "output_dir": str(output_root),
                "archive_command": "manifest-build",
                "hash_manifest": str(input_root / "hash_manifest.csv.gz"),
            },
        }
        payload_bad_rights: Dict[str, object] = {
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(input_root),
                "output_dir": str(output_root),
                "archive_command": "manifest-build",
                "rights_jsonl": str(output_root / "archive_manifest_v2.rights.jsonl"),
            },
        }

        with pytest.raises(ValueError, match="Path outside allowed roots"):
            orchestrator_app._argv_from_request(payload_bad_hash)
        with pytest.raises(ValueError, match="Path outside allowed roots"):
            orchestrator_app._argv_from_request(payload_bad_rights)
    finally:
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
        orchestrator_app.ALLOWED_PATH_ROOTS = previous_path_roots


def test_env_path_roots_rejects_invalid_configured_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TP_ALLOWED_INPUT_ROOTS", "~")
    with pytest.raises(RuntimeError, match="contains no valid roots"):
        orchestrator_app._env_path_roots("TP_ALLOWED_INPUT_ROOTS", [orchestrator_app.REPO_ROOT])


def test_default_allowed_path_roots_accept_tmp_alias_on_posix() -> None:
    if os.name == "nt":
        pytest.skip("POSIX-only tmp alias behavior")

    candidate = "/tmp/tp-default-allowlist-smoke"
    roots = orchestrator_app._default_allowed_path_roots()

    assert orchestrator_app._validate_path_against_roots(candidate, roots) == os.path.realpath(candidate)


def test_argv_rejects_tilde_prefixed_paths() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "~/.ssh",
            "output_dir": "./output",
        },
    }
    with pytest.raises(ValueError, match="Invalid path value"):
        orchestrator_app._argv_from_request(payload)


def test_create_job_rejects_paths_outside_allowed_roots_with_typed_error(tmp_path: Path) -> None:
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    previous_path_roots = orchestrator_app.ALLOWED_PATH_ROOTS
    try:
        allowed_root = (tmp_path / "allowed").resolve()
        allowed_root.mkdir(parents=True, exist_ok=True)
        orchestrator_app.ALLOWED_INPUT_ROOTS = [allowed_root]
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = [allowed_root]
        orchestrator_app.ALLOWED_PATH_ROOTS = [allowed_root]

        response = asyncio.run(
            orchestrator_app.create_job(
                {
                    "pipeline": "lux-depth-v3",
                    "args": {"input_dir": "./input_images", "output_dir": "./output"},
                }
            )
        )
        body = json.loads(response.body.decode("utf-8"))
    finally:
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
        orchestrator_app.ALLOWED_PATH_ROOTS = previous_path_roots

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "path_outside_allowed_roots"
    assert orchestrator_app.JOBS == {}


def test_create_job_rejects_tilde_prefixed_paths_with_typed_error() -> None:
    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "lux-depth-v3",
                "args": {"input_dir": "~/.ssh", "output_dir": "./output"},
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_path_value"
    assert orchestrator_app.JOBS == {}


def test_create_job_preflight_sanitizes_exception_derived_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_readiness(_pipeline, _args, require_dispatch_inputs=False):  # noqa: ANN001
        assert require_dispatch_inputs is True
        return {
            "status": "blocked",
            "canonical_command": "lux-depth-v3",
            "missing_prerequisites": [
                {
                    "reason": "unsafe_path",
                    "severity": "blocked",
                    "message": "Traceback: leaked preflight internals",
                    "field": "input_dir",
                }
            ],
            "runner_details": {},
            "notes": [],
        }

    monkeypatch.setattr(orchestrator_app, "_evaluate_pipeline_readiness", fake_readiness)

    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "lux-depth-v3",
                "args": {"input_dir": "./input_images", "output_dir": "./output"},
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["message"] == "Configured paths must stay within the allowed workspace roots."
    assert body["error"]["details"] == {"field": "input_dir", "reason": "unsafe_path"}
    assert "Traceback" not in response.body.decode("utf-8")
    assert orchestrator_app.JOBS == {}


def test_create_job_uses_preview_errors_before_readiness_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_preview(  # noqa: ANN001
        _payload,
        *,
        readiness_snapshot=None,
        archive_index_scan_mode="preview",
        portal_actor=None,
    ):
        del readiness_snapshot
        del archive_index_scan_mode
        del portal_actor
        return {
            "pipeline": "lux-depth-v3",
            "execution_args": {
                "input_dir": "./input_images",
                "output_dir": "./output",
            },
            "readiness": {
                "status": "ready",
                "canonical_command": "lux-depth-v3",
                "missing_prerequisites": [],
                "runner_details": {},
                "notes": [],
            },
            "field_errors": [
                {
                    "field": "accept_research_tools_license",
                    "code": "reconstruction_license_required",
                    "message": "Scene reconstruction requires the research-tools license acknowledgment.",
                }
            ],
        }

    monkeypatch.setattr(orchestrator_app, "_build_config_preview", fake_preview)

    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "lux-depth-v3",
                "args": {
                    "input_dir": "./input_images",
                    "output_dir": "./output",
                    "enable_reconstruction": True,
                },
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["error"]["message"] == "Scene reconstruction requires the research-tools license acknowledgment."
    assert body["error"]["details"] == {
        "field": "accept_research_tools_license",
        "reason": "reconstruction_license_required",
    }
    assert orchestrator_app.JOBS == {}


def test_create_job_config_preview_is_threaded_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_preview(_payload: Dict[str, Any], **_kwargs: Any) -> Dict[str, Any]:
        return {
            "pipeline": "lux-depth-v3",
            "execution_args": {
                "input_dir": "./input_images",
                "output_dir": "./output",
            },
            "readiness": {
                "status": "ready",
                "canonical_command": "lux-depth-v3",
                "missing_prerequisites": [],
                "runner_details": {},
                "notes": [],
            },
            "field_errors": [
                {
                    "field": "accept_research_tools_license",
                    "code": "reconstruction_license_required",
                    "message": "Scene reconstruction requires the research-tools license acknowledgment.",
                }
            ],
        }

    async def fake_to_thread(func: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(orchestrator_app, "_build_config_preview", fake_preview)
    monkeypatch.setattr(orchestrator_app.asyncio, "to_thread", fake_to_thread)

    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "lux-depth-v3",
                "args": {
                    "input_dir": "./input_images",
                    "output_dir": "./output",
                    "enable_reconstruction": True,
                },
            }
        )
    )

    assert response.status_code == 400
    assert len(calls) == 1
    func, args, kwargs = calls[0]
    assert func is fake_preview
    assert args == (
        {
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./input_images",
                "output_dir": "./output",
                "enable_reconstruction": True,
            },
        },
    )
    assert kwargs["archive_index_scan_mode"] == "full"
    assert kwargs["portal_actor"] is None
    assert orchestrator_app.JOBS == {}


def test_create_job_preserves_raw_request_and_stores_effective_request(
    monkeypatch: pytest.MonkeyPatch,
    mark_da3_runtime_available: None,
) -> None:
    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    try:
        response = asyncio.run(
            orchestrator_app.create_job(
                {
                    "pipeline": "lux-depth-v3",
                    "args": {
                        "input_dir": "/tests/fixtures/archive_small/archive_root",
                        "output_dir": "/tests/fixtures/portal_runtime_output/lux_depth_effective_request",
                    },
                }
            )
        )
        body = json.loads(response.body.decode("utf-8"))

        assert response.status_code == 200
        assert body["success"] is True
        job_id = body["data"]["id"]
        job = orchestrator_app.JOBS[job_id]
        assert job.request["args"]["input_dir"] == "/tests/fixtures/archive_small/archive_root"
        assert job.request["args"]["output_dir"] == "/tests/fixtures/portal_runtime_output/lux_depth_effective_request"
        assert job.effective_request["args"]["input_dir"] == "./tests/fixtures/archive_small/archive_root"
        assert (
            job.effective_request["args"]["output_dir"] == "./tests/fixtures/portal_runtime_output/lux_depth_effective_request"
        )
    finally:
        orchestrator_app.JOBS.clear()
        orchestrator_app.EVENT_SUBSCRIBERS.clear()


def test_job_output_dir_prefers_effective_request_repo_relative_path() -> None:
    job = orchestrator_app.Job(
        id="job_effective_output_dir",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": "/tmp/ignored"}},
        effective_request={"pipeline": "lux-depth-v3", "args": {"output_dir": "./output"}},
    )

    assert orchestrator_app._job_output_dir(job) == (orchestrator_app.REPO_ROOT / "output").resolve()


def test_create_job_archive_gate_rejects_unsafe_input_dir_before_argv() -> None:
    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "archive-gate-b",
                "args": {
                    "input_dir": "~/.ssh",
                    "output_dir": "./output",
                    "archive_command": "bag-build",
                },
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_path_value"
    assert body["error"]["details"]["field"] == "input_dir"
    assert orchestrator_app.JOBS == {}


def test_create_job_rejects_when_concurrency_limit_is_reached(
    monkeypatch: pytest.MonkeyPatch,
    mark_da3_runtime_available: None,
) -> None:
    previous_limit = orchestrator_app.MAX_CONCURRENT_JOBS
    try:
        orchestrator_app.MAX_CONCURRENT_JOBS = 1
        orchestrator_app.JOBS["job_running"] = orchestrator_app.Job(
            id="job_running",
            created_at=orchestrator_app._now(),
            state="running",
            request={"pipeline": "lux-depth-v3", "args": {"input_dir": "./input_images", "output_dir": "./output"}},
        )

        response = asyncio.run(
            orchestrator_app.create_job(
                {
                    "pipeline": "lux-depth-v3",
                    "args": {"input_dir": "./input_images", "output_dir": "./output"},
                }
            )
        )
        body = json.loads(response.body.decode("utf-8"))
    finally:
        orchestrator_app.MAX_CONCURRENT_JOBS = previous_limit
        orchestrator_app.JOBS.clear()

    assert response.status_code == 429
    assert body["error"]["code"] == "RATE_LIMITED"
    assert body["error"]["details"]["active_jobs"] == 1
    assert body["error"]["details"]["max_concurrent_jobs"] == 1


def test_importing_lux_depth_main_does_not_eagerly_import_depth_models() -> None:
    probe = (
        "import sys\n"
        "import transformation_portal.lux_depth_v3.__main__\n"
        "mods = sorted(m for m in sys.modules if m.startswith('transformation_portal.depth.models'))\n"
        "print('\\n'.join(mods))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True)

    assert result.stdout.strip() == ""


def test_list_presets_filters_pipeline() -> None:
    response = asyncio.run(orchestrator_app.list_presets("lux-depth-v3"))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert body["success"] is True
    assert body["data"]["pipeline"] == "lux-depth-v3"
    premium = next(item for item in body["data"]["presets"] if item["name"] == "premium")
    assert premium["recommended_args"]["quality_tier"] == "premium"
    assert premium["advanced_sections"] == []


def test_list_jobs_includes_error_and_artifacts() -> None:
    job = orchestrator_app.Job(
        id="job_summary",
        created_at=orchestrator_app._now(),
        last_event_at=123.0,
        state="failed",
        progress=72,
        request={"pipeline": "lux-depth-v3"},
        logs_tail=["line-1", "line-2"],
        artifacts={
            "output_dir": "/tmp/out",
            "items": [
                {
                    "artifact_type": "metadata",
                    "path": "/tmp/out/manifest.json",
                    "display_hint": {"role": "manifest", "priority": 240, "label": "Manifest"},
                }
            ],
        },
        error={"code": "RUNNER_ERROR", "message": "boom", "details": {}},
    )
    orchestrator_app.JOBS[job.id] = job

    response = asyncio.run(orchestrator_app.list_jobs())
    body = json.loads(response.body.decode("utf-8"))
    first = body["data"]["jobs"][0]

    assert response.status_code == 200
    assert first["id"] == "job_summary"
    assert first["error"]["code"] == "RUNNER_ERROR"
    assert first["last_event_at"] == 123.0
    assert first["artifacts"]["items"][0]["path"].endswith("manifest.json")
    assert first["artifacts"]["items"][0]["display_hint"]["role"] == "manifest"


def test_get_job_includes_artifacts_and_error() -> None:
    job = orchestrator_app.Job(
        id="job_details",
        created_at=orchestrator_app._now(),
        last_event_at=456.0,
        state="failed",
        request={"pipeline": "lux-depth-v3"},
        artifacts={
            "output_dir": "/tmp/out",
            "items": [
                {
                    "artifact_type": "metadata",
                    "path": "manifest.json",
                    "relative_path": "manifest.json",
                    "display_hint": {"role": "manifest", "priority": 240, "label": "Manifest"},
                }
            ],
        },
        error={"code": "RUNNER_ERROR", "message": "boom", "details": {}},
    )
    orchestrator_app.JOBS[job.id] = job

    response = asyncio.run(orchestrator_app.get_job(job.id))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert body["data"]["artifacts"]["output_dir"] == "/tmp/out"
    assert body["data"]["error"]["code"] == "RUNNER_ERROR"
    assert body["data"]["last_event_at"] == 456.0
    assert body["data"]["artifacts"]["items"][0]["display_hint"]["role"] == "manifest"


def test_late_connecting_client_receives_real_events_during_artifact_indexing() -> None:
    """Regression test: client connecting during artifact indexing gets real events.

    This tests the race condition where:
    1. Job state becomes terminal (succeeded/failed)
    2. Artifact indexing starts (can be slow on large directories)
    3. Client connects during this window (after terminal state, before done_published_at)
    4. Artifact and done events are published

    The client should receive the real artifact/done events, NOT a synthetic done.
    The deterministic fix uses done_published_at to distinguish between:
    - job.finished_at is set but events not published yet -> wait for real events
    - job.done_published_at is set -> safe to synthesize from job state
    """

    async def scenario() -> None:
        # Setup: create job in terminal state but WITHOUT done_published_at set
        # This simulates the window during artifact indexing
        job = orchestrator_app.Job(
            id="job_late_connect_test",
            created_at=orchestrator_app._now(),
            state="succeeded",
            exit_code=0,
        )
        # Do NOT set done_published_at yet - simulates artifact indexing in progress.
        # Note: Neither finished_at nor done_published_at is set, simulating the
        # window after terminal state is reached but before events are published.
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        request = _FakeRequest()
        response = await orchestrator_app.job_events(request, job.id)

        collected_events = []

        async def collect_events() -> None:
            async for chunk in response.body_iterator:
                collected_events.append(chunk)
                if "event: done" in chunk:
                    break

        # Start collecting events in background
        collect_task = asyncio.create_task(collect_events())
        await asyncio.sleep(0)  # Let collector start

        # Simulate the real artifact and done events being published
        # (as would happen after artifact indexing completes)
        await orchestrator_app._publish_event(
            job.id,
            "artifact",
            {"id": job.id, "path": "output.jpg", "size": 1234, "mime_type": "image/jpeg"},
        )
        await orchestrator_app._publish_event(
            job.id,
            "done",
            {"id": job.id, "state": "succeeded", "exit_code": 0, "error": None, "artifacts": {}},
        )

        # Wait for collector to complete
        await asyncio.wait_for(collect_task, timeout=3)

        # Join all chunks
        full_output = "".join(collected_events)

        # Verify we received the REAL artifact event (not just a synthetic done)
        assert "event: artifact" in full_output, "Client should receive real artifact event"
        assert '"path": "output.jpg"' in full_output or "output.jpg" in full_output
        assert "event: done" in full_output

        # Cleanup
        orchestrator_app.JOBS.pop(job.id, None)
        orchestrator_app.EVENT_SUBSCRIBERS.pop(job.id, None)

    asyncio.run(scenario())


def test_late_connecting_client_synthesizes_done_after_done_published_at() -> None:
    """Test that late clients get synthetic done when done_published_at is set.

    When done_published_at is set, the real events have already been published,
    so the endpoint can safely synthesize a done event from job state.
    """

    async def scenario() -> None:
        now = orchestrator_app._now()
        job = orchestrator_app.Job(
            id="job_fully_finished",
            created_at=now - 10,
            state="succeeded",
            exit_code=0,
            artifacts={"output_dir": "/tmp/out", "items": [{"path": "result.png"}]},
        )
        # Set both timestamps - this means events were already published
        job.finished_at = now
        job.done_published_at = now
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        request = _FakeRequest()
        response = await orchestrator_app.job_events(request, job.id)

        collected_events = []
        async for chunk in response.body_iterator:
            collected_events.append(chunk)
            if "event: done" in chunk:
                break

        full_output = "".join(collected_events)

        # Should get state and done events
        assert "event: state" in full_output
        assert "event: done" in full_output
        assert '"state":"succeeded"' in full_output

        # Cleanup
        orchestrator_app.JOBS.pop(job.id, None)
        orchestrator_app.EVENT_SUBSCRIBERS.pop(job.id, None)

    asyncio.run(scenario())
