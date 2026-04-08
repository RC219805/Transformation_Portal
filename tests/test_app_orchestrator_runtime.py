#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for root FastAPI orchestrator app runtime behavior."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest
from starlette.requests import Request as StarletteRequest

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")
PORTAL_HTML_PATH = Path(__file__).resolve().parents[1] / "portal.html"
PORTAL_ASSET_ROOT = PORTAL_HTML_PATH.parent / "public" / "portal-assets"


class _FakeRequest:
    """Lightweight request stub for SSE generator tests."""

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _flag_value(argv: list[str], flag: str) -> str:
    idx = argv.index(flag)
    return argv[idx + 1]


@lru_cache(maxsize=1)
def _portal_html_content() -> str:
    return PORTAL_HTML_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _portal_asset_urls_from_html() -> set[str]:
    return set(re.findall(r'["\'](/portal/assets/[^"\']+)["\']', _portal_html_content()))


@lru_cache(maxsize=1)
def _portal_asset_urls_from_css() -> set[str]:
    return set(re.findall(r'url\(["\']?(/portal/assets/[^)"\']+)["\']?\)', _portal_css_content()))


def _portal_asset_path(asset_url: str) -> Path:
    if not asset_url.startswith("/portal/assets/"):
        raise AssertionError(f"unexpected portal asset url: {asset_url}")
    candidate = PORTAL_ASSET_ROOT / asset_url.removeprefix("/portal/assets/")
    if not candidate.is_file():
        raise AssertionError(f"portal asset missing: {candidate}")
    return candidate


@lru_cache(maxsize=1)
def _portal_css_content() -> str:
    html = _portal_html_content()
    match = re.search(r'<link rel="stylesheet" href="(/portal/assets/[^"]+)"\s*/?>', html)
    if match is None:
        raise AssertionError("portal stylesheet link not found")
    return _portal_asset_path(match.group(1)).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _portal_js_content() -> str:
    html = _portal_html_content()
    match = re.search(r'<script src="(/portal/assets/[^"]+)"[^>]*></script>', html)
    if match is None:
        raise AssertionError("portal script asset not found")
    return _portal_asset_path(match.group(1)).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _portal_bundle_content() -> str:
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


def test_portal_html_externalizes_direct_debug_assets_without_third_party_hosts() -> None:
    html_content = _portal_html_content()
    css_content = _portal_css_content()

    assert 'href="/portal/assets/portal.css"' in html_content
    assert 'src="/portal/assets/portal.js"' in html_content
    assert "<style>" not in html_content
    assert "<script>" not in html_content
    assert "https://cdn.tailwindcss.com" not in html_content
    assert "https://fonts.googleapis.com" not in html_content
    assert "https://fonts.gstatic.com" not in html_content
    assert "tailwind.config" not in css_content
    assert "Phase 1 local utility snapshot replacing Tailwind CDN for portal.html" in css_content


def test_portal_asset_manifest_is_explicit_and_repo_local() -> None:
    assert orchestrator_app.PORTAL_ASSET_MANIFEST_PATH.is_file()
    assert orchestrator_app.PORTAL_ASSET_PATHS == {
        "portal.css": orchestrator_app.PORTAL_ASSETS_DIR / "portal.css",
        "portal.js": orchestrator_app.PORTAL_ASSETS_DIR / "portal.js",
        "fonts/portal-sans.woff2": orchestrator_app.PORTAL_ASSETS_DIR / "fonts" / "portal-sans.woff2",
        "fonts/portal-mono.woff2": orchestrator_app.PORTAL_ASSETS_DIR / "fonts" / "portal-mono.woff2",
    }
    assert orchestrator_app.PORTAL_ASSET_MEDIA_TYPES == {
        "portal.css": "text/css; charset=utf-8",
        "portal.js": "text/javascript; charset=utf-8",
        "fonts/portal-sans.woff2": "font/woff2",
        "fonts/portal-mono.woff2": "font/woff2",
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


def test_portal_html_asset_references_are_covered_by_manifest() -> None:
    html_asset_urls = _portal_asset_urls_from_html()
    bundled_asset_urls = html_asset_urls | _portal_asset_urls_from_css()
    manifest_asset_urls = {f"/portal/assets/{asset_name}" for asset_name in orchestrator_app.PORTAL_ASSET_MANIFEST.keys()}

    assert html_asset_urls
    assert html_asset_urls <= manifest_asset_urls
    assert bundled_asset_urls == manifest_asset_urls
    for asset_url in bundled_asset_urls:
        _portal_asset_path(asset_url)


def test_portal_fetch_sse_reconnect_scheduler_has_terminal_guard_and_backoff() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "scheduleSseReconnect")

    assert "_isJobStreamRecoverable(job)" in body
    assert "if (job.reconnectBlocked) return;" in body
    assert "if (job.sseRetry.timer || _jobHasActiveStream(job)) return;" in body
    assert "SSE_RECONNECT_BASE_DELAY_MS" in body
    assert "setTimeout" in body
    assert "startJobEventStream(job, job.eventStreamUrl);" in body


def test_portal_fetch_sse_reconnect_schedules_on_unexpected_disconnect_only() -> None:
    content = _portal_bundle_content()
    body = _extract_js_function_body(content, "_startAuthorizedFetchSse")

    assert "let sawDoneEvent = false;" in body
    assert "let shouldReconnect = true;" in body
    assert "const isAuthError = status === 401 || status === 403;" in body
    assert "const isRetryableStatus = status === 429 || status >= 500;" in body
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
    bind_body = _extract_js_function_body(content, "bindInputs")

    assert "if (!job.reconnectBlocked) return;" in helper_body
    assert "startJobEventStream(job, job.eventStreamUrl);" in helper_body
    assert "resumeBlockedJobStreamsAfterAuthUpdate();" in bind_body


def test_portal_bootstrap_loader_uses_abortable_timeout_and_state_contract() -> None:
    content = _portal_bundle_content()
    default_body = _extract_js_function_body(content, "_defaultPortalBootstrap")
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
    assert "const BOOTSTRAP_TIMEOUT_MS = 3500;" in content
    assert "const BOOTSTRAP_RETRY_BASE_DELAY_MS = 1000;" in content
    assert "const BOOTSTRAP_RETRY_MAX_DELAY_MS = 12000;" in content
    assert "const BOOTSTRAP_RETRY_MAX_ATTEMPTS = 4;" in content
    assert "const BOOTSTRAP_RETRY_MAX_WINDOW_MS = 60000;" in content
    assert "const BOOTSTRAP_RETRIABLE_HTTP_STATUSES = new Set([500, 502, 503, 504]);" in content
    assert "fetchWithTimeout(" in body
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
    assert "window.location.assign('/login');" in body
    assert "const status = failure.retryable ? 'degraded' : 'unavailable';" in body
    assert "const retryScheduled = failure.retryable && _scheduleBootstrapRetry(failure.reason, res.status);" in body
    assert "const retryScheduled = failure.retryable && _scheduleBootstrapRetry(failure.reason, 0);" in body
    assert "_finalizeBootstrapRetry('terminal_invalid_json', { reason: 'invalid_json' });" in body
    assert "_finalizeBootstrapRetry('succeeded', {" in body
    assert "_applyPortalBootstrap(payload, { status: 'ready' });" in body
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


def test_portal_managed_mode_clears_api_keys_and_hides_secret_ui() -> None:
    content = _portal_bundle_content()
    clear_body = _extract_js_function_body(content, "_clearStoredApiKeyState")
    sync_body = _extract_js_function_body(content, "_syncBootstrapUi")

    assert "localStorage.removeItem(API_KEY_STORAGE_KEY);" in clear_body
    assert "sessionStorage.removeItem(API_KEY_STORAGE_KEY);" in clear_body
    assert "_clearStoredApiKeyState(true);" in content
    assert "_loadApiKeyIntoInputs();" in content
    assert "const showApiKeyInput = bootstrapReady && state.auth.features.apiKeyInput;" in sync_body
    assert "els.apiKeySection.classList.toggle('hidden', !showApiKeyInput);" in sync_body
    assert "els.apiKeyInput.disabled = !showApiKeyInput;" in sync_body
    assert "rememberApiKey" not in content


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
    assert "function _buildAuthHeaders(base = {}, method = 'GET') {" in auth_headers_block
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
    assert "function _buildAuthHeaders(base = {}, method = 'GET') {" in content
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
    body = _extract_js_function_body(content, "renderArtifactPanel")
    reset_body = _extract_js_function_body(content, "_resetArtifactActionButtons")
    sanitize_body = _extract_js_function_body(content, "sanitizeManagedAssetUrl")
    rank_body = _extract_js_function_body(content, "rankArtifactsForDisplay")
    compare_body = _extract_js_function_body(content, "findCompareArtifact")
    normalize_body = _extract_js_function_body(content, "normalizeArtifactItems")

    assert 'id="artifactPreviewStage"' in content
    assert 'id="artifactThumbnailRail"' in content
    assert 'id="artifactSelectionMeta"' in content
    assert 'id="openArtifactBtn"' in content
    assert 'id="downloadArtifactBtn"' in content
    assert 'id="copyArtifactPathBtn"' in content
    assert "artifactHeroScore" in content
    assert "findCompareArtifact" in content
    assert "artifactDisplayHint" in content
    assert "artifactDisplayPriority" in content
    assert "artifactDisplayLabel" in content
    assert "buildArtifactUrl(selected, selectedArtifact)" in body
    assert "artifactIsPreviewable(selectedArtifact)" in body
    assert "artifactDisplayLabel(selectedArtifact)" in body
    assert "artifactDisplayLabel(artifact)" in body
    assert "artifactDisplayPriority(right)" in rank_body
    assert "artifactCompareGroup(candidate) === primaryGroup" in compare_body
    assert "display_hint: _normalizeArtifactDisplayHint(item.display_hint)" in normalize_body
    assert "_resetArtifactActionButtons();" in body
    assert "delete els.openArtifactBtn.dataset.url;" in reset_body
    assert "delete els.downloadArtifactBtn.dataset.filename;" in reset_body
    assert "delete els.copyArtifactPathBtn.dataset.path;" in reset_body
    assert "parsed.origin !== window.location.origin" in sanitize_body
    assert "parsed.pathname.startsWith('/v1/jobs/')" in sanitize_body
    assert "sanitizeManagedAssetUrl(els.openArtifactBtn.dataset.url)" in content
    assert "sanitizeManagedAssetUrl(els.downloadArtifactBtn.dataset.url)" in content


def test_portal_review_surface_exposes_warning_banner_and_provenance_contract() -> None:
    content = _portal_bundle_content()
    render_body = _extract_js_function_body(content, "renderArtifactPanel")
    status_body = _extract_js_function_body(content, "_reviewStatusSnapshot")
    banner_body = _extract_js_function_body(content, "_renderReviewStatusBanner")
    provenance_body = _extract_js_function_body(content, "_renderArtifactProvenance")

    assert 'id="reviewStatusBanner"' in content
    assert 'id="reviewStatusTitle"' in content
    assert 'id="reviewStatusDetail"' in content
    assert 'id="reviewProvenanceGrid"' in content
    assert 'id="reviewProvenanceArtifactRole"' in content
    assert 'id="reviewProvenanceRunState"' in content
    assert 'id="reviewProvenancePath"' in content
    assert 'id="reviewProvenanceFreshness"' in content
    assert 'id="reviewProvenanceSource"' in content
    assert 'id="reviewProvenanceBatch"' in content
    assert 'data-ui="review-status-banner"' in content
    assert 'data-ui="review-provenance-grid"' in content
    assert "_renderReviewStatusBanner(selected, selectedArtifact);" in render_body
    assert "_renderArtifactProvenance(selected, selectedArtifact);" in render_body
    assert "_renderReviewStatusBanner(selected, null);" in render_body
    assert "_renderArtifactProvenance(selected, null);" in render_body
    assert "_renderReviewStatusBanner(null, null);" in render_body
    assert "_renderArtifactProvenance(null, null);" in render_body
    assert "job.state === 'partial'" in status_body
    assert "job.state === 'failed'" in status_body
    assert "job.state === 'canceled'" in status_body
    assert "job.state === 'offline'" in status_body
    assert "job.reconnectBlocked" in status_body
    assert "Outputs ready for review" in status_body
    assert "Run canceled after partial output capture" in status_body
    assert "Run is offline with reviewable outputs" in status_body
    assert "formatRelativeTime" in status_body
    assert "els.reviewStatusBanner.dataset.tone = snapshot.tone;" in banner_body
    assert "artifactDisplayLabel(artifact)" in provenance_body
    assert "artifactLabel(artifact)" in provenance_body
    assert "titleCaseToken(job.state, 'Unknown')" in provenance_body
    assert "summary?.batch_id" in provenance_body
    assert "summary?.source" in provenance_body


def test_portal_review_surface_supports_compare_summary_and_keyboard_selection() -> None:
    content = _portal_bundle_content()
    render_body = _extract_js_function_body(content, "renderArtifactPanel")
    compare_summary_body = _extract_js_function_body(content, "_renderReviewCompareSummary")
    focus_body = _extract_js_function_body(content, "_focusArtifactRailButton")
    keydown_body = _extract_js_function_body(content, "handleArtifactRailKeydown")

    assert 'id="reviewCompareSummary"' in content
    assert 'id="reviewCompareTitle"' in content
    assert 'id="reviewCompareDetail"' in content
    assert 'data-ui="review-compare-summary"' in content
    assert "els.artifactThumbnailRail.setAttribute('role', 'listbox');" in render_body
    assert "button.setAttribute('role', 'option');" in render_body
    assert "button.setAttribute('aria-selected', active ? 'true' : 'false');" in render_body
    assert "button.tabIndex = active ? 0 : -1;" in render_body
    assert "_renderReviewCompareSummary(selectedArtifact, compareCandidate, compareEnabled);" in render_body
    assert "els.artifactCompareBtn.setAttribute('aria-pressed', compareEnabled ? 'true' : 'false');" in render_body
    assert "els.artifactCompareStage.setAttribute('aria-hidden', compareEnabled ? 'false' : 'true');" in render_body
    assert "Comparing paired outputs" in compare_summary_body
    assert "Compare pair available" in compare_summary_body
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
    assert "_reconcileJobTimeline(selected);" in inspector_body
    assert "formatDuration" in inspector_body
    assert "_noteTransportWarning" in content
    assert "const showLogsShell = nextTab === 'logs' || state.currentView === 'review';" in tab_body
    assert "button.setAttribute('aria-selected', active ? 'true' : 'false');" in tab_body
    assert "els.logsShell.classList.toggle('hidden', !showLogsShell);" in tab_body


def test_portal_operate_surfaces_use_jobs_hydration_skeletons_before_empty_state() -> None:
    content = _portal_bundle_content()
    helper_body = _extract_js_function_body(content, "_isJobsHydrationPending")
    toggle_body = _extract_js_function_body(content, "_toggleSurfaceSkeleton")
    queue_body = _extract_js_function_body(content, "renderJobQueue")
    inspector_body = _extract_js_function_body(content, "renderSelectedJobInspector")
    artifact_body = _extract_js_function_body(content, "renderArtifactPanel")
    recover_body = _extract_js_function_body(content, "recoverJobs")
    flush_body = _extract_js_function_body(content, "_flushBootstrapOnlineFollowup")
    backend_body = _extract_js_function_body(content, "checkBackend")

    assert "jobsLoadStatus: 'pending'," in content
    assert 'id="queueSkeletonState"' in content
    assert 'id="selectedJobSkeletonState"' in content
    assert 'id="artifactSkeletonState"' in content
    assert "state.jobsLoadStatus === 'loading'" in helper_body
    assert "state.bootstrap.status === 'pending' || state.bootstrap.status === 'degraded'" in helper_body
    assert "skeleton.setAttribute('aria-hidden', 'true');" in toggle_body
    assert "const queueLoading = _isJobsHydrationPending();" in queue_body
    assert "els.queueShell.setAttribute('aria-busy', queueLoading ? 'true' : 'false');" in queue_body
    assert "els.queueSkeletonState.setAttribute('aria-hidden', 'true');" in queue_body
    assert (
        "_toggleSurfaceSkeleton(els.selectedJobShell, els.selectedJobShellContent, els.selectedJobSkeletonState, jobsLoading);"
        in inspector_body
    )
    assert (
        "_toggleSurfaceSkeleton(els.artifactsShell, els.artifactShellContent, els.artifactSkeletonState, jobsLoading);"
        in artifact_body
    )
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

    assert "Theme: System" in content
    assert "Theme preference: system. Click to switch to dark." in content
    assert "Cycle Theme" in content
    assert "const nextIndex = (currentIndex + 1) % THEME_PREFERENCES.length;" in next_body
    assert "return THEME_PREFERENCES[nextIndex];" in next_body
    assert "applyThemePreference(_nextThemePreference(state.themePreference));" in content
    assert "Theme: System (${effectiveLabel})" in sync_body
    assert "Theme preference: ${preference}. Click to switch to ${nextPreference}." in sync_body


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
    apply_view_body = _extract_js_function_body(content, "applyConsoleViewLayout")

    assert 'data-view-link="overview"' in content
    assert 'data-view-link="build"' in content
    assert 'data-view-link="operate"' in content
    assert 'data-view-link="review"' in content
    assert "url.searchParams.set('view', resolveConsoleView(viewName));" in content
    assert "state.currentView = resolveConsoleView(url.searchParams.get('view'));" in content
    assert "candidate === 'run'" not in content
    assert "document.body.dataset.consoleView = state.currentView;" in apply_view_body
    assert "els.queueShell.classList.toggle('hidden', state.currentView === 'review');" in apply_view_body
    assert "const isPlainPrimaryClick = event.button === 0" in rail_body
    assert "if (event.defaultPrevented || !isPlainPrimaryClick)" in rail_body
    assert "navigateConsoleView(nextView);" in rail_body


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
    assert "function setBuildStep(nextStep, options = {}) {" in content
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

    assert 'id="dispatchToolsDetails"' in content
    assert 'data-ui="dispatch-tools"' in content
    assert "Review dispatch posture" in content
    assert "CLI Parity & Config Tools" in content
    assert 'id="effectiveConfigBtn"' in content
    assert 'id="importBtn"' in content
    assert 'id="exportBtn"' in content
    assert 'id="copyCliBtn"' in content
    assert 'id="cliPreview"' in content


def test_portal_overview_and_build_surfaces_sync_bootstrap_skeletons_and_preview_loading() -> None:
    content = _portal_bundle_content()
    helper_body = _extract_js_function_body(content, "_syncOverviewBuildLoadingState")
    preview_body = _extract_js_function_body(content, "_isBuildPreviewRefreshing")
    mission_body = _extract_js_function_body(content, "renderMissionControl")
    bootstrap_body = _extract_js_function_body(content, "_syncBootstrapUi")
    cli_body = _extract_js_function_body(content, "renderCLI")

    assert 'id="missionShellSkeletonState"' in content
    assert 'id="intelligenceShellSkeletonState"' in content
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
    assert ".surface-loading::after" in css
    assert ".status-dot.running," in css
    assert ".status-dot.partial," in css
    assert ".skeleton-pill," in css
    assert ".toast-enter," in css
    assert ".surface-loading::after {" in css
    assert "transition: none !important;" in css


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
    fetch_body = _extract_js_function_body(content, "fetchPresetsForPipeline")

    assert "applyPresetRecommendedArgs(nextPreset);" in bind_body
    assert "quality_tier" in preset_body
    assert "depth_backend" in preset_body
    assert "segmentation_backend" in preset_body
    assert "emit_run_card" in preset_body
    assert "advanced_sections" in fetch_body
    assert "recommended_args" in fetch_body


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

    assert "lastHealthEndpointPath: ''" in content
    assert "pendingOnlineFollowup: false" in content
    assert "onlineFollowupComplete: false" in content
    assert "deadlineAt: 0" in content
    assert "lastOutcome: ''" in content


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
    update_body = _extract_js_function_body(content, "updateUIFromState")
    bind_body = _extract_js_function_body(content, "bindInputs")
    preview_body = _extract_js_function_body(content, "fetchConfigPreview")
    reconcile_body = _extract_js_function_body(content, "_reconcilePreviewRepairedPaths")
    setter_body = _extract_js_function_body(content, "_setBuildSurfacePathFieldValue")

    assert "maxWorkersMode: 'auto'," in content
    assert "maxGpuWorkersMode: 'auto'," in content
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
    assert "repo_local_path_repaired" in reconcile_body
    assert "_setBuildSurfacePathFieldValue(fieldName, normalizedValue)" in reconcile_body
    assert "state.config = state.config || {};" in setter_body
    assert "state.config.gate = state.config.gate || {};" in setter_body
    assert "state.config.segmentation = state.config.segmentation || {};" in setter_body
    assert "state.config.reconstruction = state.config.reconstruction || {};" in setter_body


def test_portal_submit_blocks_preview_unavailable_and_debug_bundle_without_acknowledgement() -> None:
    content = _portal_bundle_content()
    guard_body = _extract_js_function_body(content, "_syncBootstrapGuardedControls")
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
    assert "_effectiveDebugBundleEnabled(preview)" in guard_body
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
        "depth_device": "--depth-device",
        "enable_segmentation": "--enable-segmentation",
        "segmentation_backend": "--segmentation-backend",
        "sam2_model_size": "--sam2-model-size",
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
    }

    for key, flag in arg_to_flag.items():
        assert key in canonical_keys, f"portal canonical args missing key '{key}'"
        assert flag in portal_cli_flags, f"portal CLI preview missing flag '{flag}'"

    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output/lux_depth_v3_apex_verify",
            "preset": "depth-anything-v3.1-research-m4",
            "quality_tier": "apex",
            "depth_backend": "depth_pro",
            "depth_device": "cpu",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_model_size": "large",
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
        },
    }
    argv = orchestrator_app._argv_from_request(payload)

    expected_present_flags = {flag for key, flag in arg_to_flag.items() if key != "quiet"}
    for flag in expected_present_flags:
        assert flag in argv, f"backend argv missing flag '{flag}'"

    assert _flag_value(argv, "--preset") == "depth-anything-v3.1-research-m4"
    assert _flag_value(argv, "--quality-tier") == "apex"
    assert _flag_value(argv, "--depth-backend") == "depth_pro"
    assert _flag_value(argv, "--depth-device") == "cpu"
    assert _flag_value(argv, "--enable-segmentation") == "on"
    assert _flag_value(argv, "--segmentation-backend") == "sam2"
    assert _flag_value(argv, "--sam2-model-size") == "large"
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
            "preset": "depth-anything-v3.1-research-m4",
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
        "depth-anything-v3.1-research-m4",
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
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "raw_demosaic": "VNG",
        },
    }

    with pytest.raises(ValueError, match="Invalid raw_demosaic"):
        orchestrator_app._argv_from_request(payload)


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
    content = _portal_bundle_content()
    assert "enable: false," in content
    assert "backend: 'stub'," in content
    assert "sam2ModelSize: 'base'," in content
    assert "strict: false" in content


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


def test_portal_exposes_run_card_quick_actions() -> None:
    content = _portal_bundle_content()

    assert 'id="runCardActions"' in content
    assert 'id="viewRunCardBtn"' in content
    assert 'id="copyRunCardPathBtn"' in content
    assert 'id="copyRunCardFingerprintBtn"' in content
    assert "els.viewRunCardBtn.dataset.url = runCardUrl;" in content
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

    assert "flagsShell: document.getElementById('flags-shell')" in content
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

    assert "segmentationBackendField: document.getElementById('segmentationBackendField')" in content
    assert "sam2ModelSizeField: document.getElementById('sam2ModelSizeField')" in content
    assert "strictSegmentationField: document.getElementById('strictSegmentationField')" in content
    assert "sam2CheckpointField: document.getElementById('sam2CheckpointField')" in content
    assert "v2PresetField: document.getElementById('v2PresetField')" in content
    assert "governanceDetailsHint: document.getElementById('governanceDetailsHint')" in content
    assert "licenseAppleField: document.getElementById('licenseAppleField')" in content
    assert "reconstructionConfigFields: document.getElementById('reconstructionConfigFields')" in content
    assert "function _derivePresetResearchFlag" in content
    assert ".includes('research')" in preset_research_body
    assert "is_research: _derivePresetResearchFlag({" in preset_body
    assert "is_research: _derivePresetResearchFlag({" in fallback_body
    assert "is_research: _derivePresetResearchFlag(preset)" in fetch_body
    assert "_setContextVisibility(els.segmentationBackendField, isLuxPipeline && segmentationEnabled);" in applicability_body
    assert "_setContextVisibility(els.sam2ModelSizeField, isLuxPipeline && showSam2Controls);" in applicability_body
    assert "_setContextVisibility(els.v2PresetField, isLuxPipeline && enableV2);" in applicability_body
    assert "_setContextVisibility(els.governanceDetails, governanceVisible);" in applicability_body
    assert "els.licenseAppleField" in applicability_body
    assert "_setContextVisibility(els.reconstructionConfigFields, reconstructionEnabled);" in applicability_body
    assert "syncBuildSurfaceApplicability(currentPayload);" in mission_control_body


def test_portal_dispatch_controls_require_backend_readiness_and_live_backend() -> None:
    content = _portal_bundle_content()
    guard_body = _extract_js_function_body(content, "_syncBootstrapGuardedControls")
    submit_body = _extract_js_function_body(content, "submitJob")

    assert "state.backendOk" in guard_body
    assert "currentPipelineDispatchStatus()" in guard_body
    assert "readinessStatus === 'ready'" in guard_body
    assert "Execution readiness is still loading." in submit_body
    assert "Backend is offline. Dispatch is disabled until connectivity is restored." in submit_body
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
    assert orchestrator_app._is_mutating_job_endpoint("GET", "/v1/jobs/job_123") is False
    assert orchestrator_app._is_mutating_job_endpoint("GET", "/v1/jobs/job_123/events") is False


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
    assert orchestrator_app._is_protected_job_endpoint("/ready") is False


def test_protected_api_key_route_detection() -> None:
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/jobs") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/config-metadata") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/config-preview") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/v1/portal/events") is True
    assert orchestrator_app._is_protected_api_key_endpoint("/ready") is False


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
    assert render_item["content_type"] == "image/png"
    assert render_item["display_hint"]["role"] == "primary_preview"
    assert render_item["display_hint"]["priority"] == 1000
    assert render_item["display_hint"]["label"] == "Primary Preview"
    assert render_item["display_hint"]["compare_group"]
    assert render_item["url"] == f"/v1/jobs/{job.id}/artifacts/render.png"
    manifest_item = next(item for item in indexed if item["path"] == "manifest.json")
    assert manifest_item["media_kind"] == "metadata"
    assert manifest_item["previewable"] is False
    assert manifest_item["display_hint"]["role"] == "manifest"
    assert manifest_item["display_hint"]["priority"] == 240
    assert manifest_item["display_hint"]["label"] == "Manifest"


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
    def fake_preview(_payload, *, readiness_snapshot=None):  # noqa: ANN001
        del readiness_snapshot
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


def test_create_job_preserves_raw_request_and_stores_effective_request(
    monkeypatch: pytest.MonkeyPatch,
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


def test_create_job_rejects_when_concurrency_limit_is_reached() -> None:
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
