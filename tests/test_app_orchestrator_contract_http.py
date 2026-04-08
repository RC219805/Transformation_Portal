#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP contract tests for the root FastAPI orchestrator app."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request as StarletteRequest

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")


def _collect_sse_events(response) -> List[Tuple[str, Dict[str, Any]]]:
    events: List[Tuple[str, Dict[str, Any]]] = []
    current_event = ""
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line.split("event: ", 1)[1].strip()
            continue
        if line.startswith("data: "):
            payload = json.loads(line.split("data: ", 1)[1])
            events.append((current_event, payload))
            if current_event == "done":
                break
    return events


@pytest.fixture(autouse=True)
def _reset_orchestrator_globals() -> None:
    previous_api_key = orchestrator_app.API_KEY_SECRET
    previous_enforce_job_api_key = orchestrator_app.ENFORCE_JOB_API_KEY
    previous_allow_sse_query_api_key = orchestrator_app.ALLOW_SSE_QUERY_API_KEY
    previous_max_request_bytes = orchestrator_app.MAX_REQUEST_BYTES
    previous_max_indexed_artifacts = orchestrator_app.MAX_INDEXED_ARTIFACTS
    previous_rate_limit_per_minute = orchestrator_app.RATE_LIMIT_PER_MINUTE
    previous_max_concurrent_jobs = orchestrator_app.MAX_CONCURRENT_JOBS
    orchestrator_app.API_KEY_SECRET = "contract-secret"
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    orchestrator_app.ALLOW_SSE_QUERY_API_KEY = False
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()
    try:
        yield
    finally:
        orchestrator_app.API_KEY_SECRET = previous_api_key
        orchestrator_app.ENFORCE_JOB_API_KEY = previous_enforce_job_api_key
        orchestrator_app.ALLOW_SSE_QUERY_API_KEY = previous_allow_sse_query_api_key
        orchestrator_app.MAX_REQUEST_BYTES = previous_max_request_bytes
        orchestrator_app.MAX_INDEXED_ARTIFACTS = previous_max_indexed_artifacts
        orchestrator_app.RATE_LIMIT_PER_MINUTE = previous_rate_limit_per_minute
        orchestrator_app.MAX_CONCURRENT_JOBS = previous_max_concurrent_jobs
        orchestrator_app.JOBS.clear()
        orchestrator_app.EVENT_SUBSCRIBERS.clear()
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()


@pytest.fixture(name="client")
def _client_fixture() -> TestClient:
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as test_client:
        yield test_client


def test_ready_keeps_non_enveloped_shape(client: TestClient) -> None:
    response = client.get("/ready")
    body = response.json()
    assert response.status_code == 200
    assert body["ok"] is True
    assert "success" not in body
    assert "schema" not in body


def test_healthz_returns_minimal_health_response(client: TestClient) -> None:
    """Validate /healthz endpoint matches portal.html expectations for managed auth mode."""
    response = client.get("/healthz")
    body = response.json()
    assert response.status_code == 200
    assert body["ok"] is True
    assert "time" in body
    # The /healthz endpoint must be minimal - no verbose cli/jobs/security fields
    assert "cli" not in body
    assert "jobs" not in body
    assert "security" not in body
    assert "version" not in body
    # Health checks must not be cached to ensure outages are detected immediately
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Pragma"] == "no-cache"


def test_portal_bootstrap_reports_direct_debug_mode(client: TestClient) -> None:
    response = client.get("/portal/bootstrap")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    body = response.json()
    assert body["authMode"] == "direct_debug"
    assert body["csrfToken"] is None
    assert body["actor"] is None
    assert body["features"]["apiKeyInput"] is True
    assert body["features"]["directDebug"] is True


def test_root_ui_response_is_not_cached(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    csp = response.headers.get("Content-Security-Policy")

    assert csp is not None
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Pragma"] == "no-cache"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"
    assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert response.headers["X-Permitted-Cross-Domain-Policies"] == "none"
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp
    assert "style-src 'self'" in csp
    assert "style-src 'self' 'unsafe-inline'" not in csp
    assert "font-src 'self'" in csp
    assert "img-src 'self' data: blob:" in csp
    assert "media-src 'self'" in csp
    assert "connect-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "form-action 'self'" in csp
    assert "https://cdn.tailwindcss.com" not in csp
    assert "https://fonts.googleapis.com" not in csp
    assert "https://fonts.gstatic.com" not in csp
    assert "https://cdn.tailwindcss.com" not in response.text
    assert "https://fonts.googleapis.com" not in response.text
    assert "https://fonts.gstatic.com" not in response.text
    assert '<link rel="stylesheet" href="/portal/assets/portal.css"' in response.text
    assert '<script src="/portal/assets/portal.js" defer></script>' in response.text
    assert "<style>" not in response.text
    assert "<script>" not in response.text
    assert "Content-Security-Policy" not in response.text
    assert "Remember in local storage" not in response.text
    assert "Transformation Portal" in response.text


def test_portal_asset_endpoint_serves_css_and_js(client: TestClient) -> None:
    css_response = client.get("/portal/assets/portal.css")
    js_response = client.get("/portal/assets/portal.js")

    assert css_response.status_code == 200
    assert css_response.headers["Cache-Control"] == orchestrator_app.PORTAL_ASSET_CACHE_CONTROL
    assert css_response.headers["content-type"] == orchestrator_app.PORTAL_ASSET_MEDIA_TYPES["portal.css"]
    assert "@font-face" in css_response.text
    assert "Portal Sans" in css_response.text
    assert "https://fonts.googleapis.com" not in css_response.text

    assert js_response.status_code == 200
    assert js_response.headers["Cache-Control"] == orchestrator_app.PORTAL_ASSET_CACHE_CONTROL
    assert js_response.headers["content-type"] == orchestrator_app.PORTAL_ASSET_MEDIA_TYPES["portal.js"]
    assert "const BOOTSTRAP_TIMEOUT_MS = 3500;" in js_response.text


def test_portal_asset_endpoint_serves_repo_local_fonts(client: TestClient) -> None:
    response = client.get("/portal/assets/fonts/portal-sans.woff2")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == orchestrator_app.PORTAL_ASSET_CACHE_CONTROL
    assert response.headers["content-type"] == orchestrator_app.PORTAL_ASSET_MEDIA_TYPES["fonts/portal-sans.woff2"]
    assert response.content


def test_portal_asset_endpoint_rejects_path_traversal(client: TestClient) -> None:
    response = client.get("/portal/assets/../portal.html")

    assert response.status_code == 404


def test_portal_asset_endpoint_rejects_unknown_assets(client: TestClient) -> None:
    response = client.get("/portal/assets/fonts/not-real.woff2")

    assert response.status_code == 404


def test_portal_video_endpoint_serves_background_asset(
    client: TestClient,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_path = tmp_path / "dna-portal-video-2.mp4"
    asset_bytes = b"\x00\x00\x00\x20ftypisomportal-video"
    asset_path.write_bytes(asset_bytes)
    monkeypatch.setattr(orchestrator_app, "PORTAL_VIDEO_PATH", asset_path)

    response = client.get("/portal/video/dna-portal-video-2.mp4")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == orchestrator_app.PORTAL_VIDEO_CACHE_CONTROL
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content == asset_bytes


def test_legacy_portal_video_endpoint_redirects_to_cacheable_asset_route(client: TestClient) -> None:
    response = client.get("/v1/portal/video/dna-portal-video-2.mp4", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["location"] == "/portal/video/dna-portal-video-2.mp4"


def test_portal_video_endpoint_returns_not_found_for_unknown_assets(client: TestClient) -> None:
    response = client.get("/portal/video/not-allowed.mp4")

    assert response.status_code == 404


def test_presets_contract_for_lux_depth_pipeline(client: TestClient) -> None:
    response = client.get("/v1/presets", params={"pipeline": "lux-depth-v3"})
    body = response.json()
    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.presets.v1"
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["pipeline"] == "lux-depth-v3"
    premium = next(item for item in body["data"]["presets"] if item["name"] == "premium")
    assert premium["recommended_args"]["quality_tier"] == "premium"
    assert premium["advanced_sections"] == []


def test_config_metadata_contract_for_lux_depth_pipeline(client: TestClient) -> None:
    response = client.get("/v1/config-metadata", params={"pipeline": "lux-depth-v3"})
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_metadata.v1"
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["pipeline"] == "lux-depth-v3"
    assert body["data"]["fields"]["reconstruction_tier"]["default"] == "apex_research"
    assert body["data"]["fields"]["reconstruction_iterations"]["recommended"]["balanced"] == 1000
    assert body["data"]["fields"]["raw_wb_mode"]["kind"] == "locked"
    assert body["data"]["debug_bundle_policy"]["acknowledgement_required"] is True


def test_config_metadata_request_validation_errors_are_sanitized(client: TestClient) -> None:
    response = client.get("/v1/config-metadata")
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["message"] == "request validation failed"
    assert body["error"]["details"] == {
        "path": "/v1/config-metadata",
        "reason": "request_validation_failed",
    }


def test_config_preview_rejects_unsupported_pipeline_with_sanitized_reason(client: TestClient) -> None:
    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "not-a-real-pipeline",
            "args": {},
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["message"] == "invalid config preview request"
    assert body["error"]["details"] == {"field": "payload", "reason": "unsupported_pipeline"}


def test_lux_config_preview_returns_execution_args_and_repair_warning_for_repo_local_shorthand(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "/tests/fixtures/archive_small/archive_root",
                "output_dir": "/tests/fixtures/portal_contract_output/lux_depth_preview_contract",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    preview = body["data"]
    warning_codes = {item["code"] for item in preview["field_warnings"]}
    assert "repo_local_path_repaired" in warning_codes
    assert preview["normalized_args"]["input_dir"] == "./tests/fixtures/archive_small/archive_root"
    assert preview["normalized_args"]["output_dir"] == "./tests/fixtures/portal_contract_output/lux_depth_preview_contract"
    assert preview["execution_args"]["input_dir"] == "./tests/fixtures/archive_small/archive_root"
    assert preview["execution_args"]["output_dir"] == "./tests/fixtures/portal_contract_output/lux_depth_preview_contract"


def test_config_preview_contract_rejects_repo_local_shorthand_traversal(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "/tests/../output",
                "output_dir": "./output",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert errors["input_dir"]["code"] == "path_shorthand_traversal_disallowed"


def test_archive_config_preview_returns_field_specific_path_errors(client: TestClient) -> None:
    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./output/archive_contract",
                "archive_command": "fixity-scan",
                "archive_index": "/tests/../fixtures/archive_small/archive_index_normalized.csv.gz",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    preview = body["data"]
    errors = {item["field"]: item for item in preview["field_errors"]}
    assert errors["archive_index"]["code"] == "path_shorthand_traversal_disallowed"
    assert all(item["code"] != "invalid_request" for item in preview["field_errors"])


def test_archive_config_preview_rejects_non_directory_bag_dir(client: TestClient, tmp_path: Path) -> None:
    bag_file = tmp_path / "not_a_bag.txt"
    bag_file.write_text("not a directory", encoding="utf-8")

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "archive-gate-b",
            "args": {
                "archive_command": "bag-validate",
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": str(tmp_path),
                "bag_dir": str(bag_file),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert errors["bag_dir"]["code"] == "not_a_directory"


def test_archive_gate_a_config_preview_returns_authoritative_readiness_and_argv(
    client: TestClient,
    tmp_path: Path,
) -> None:
    archive_root = (tmp_path / "archive_root").resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_index = (tmp_path / "archive_index_normalized.csv.gz").resolve()
    archive_index.write_bytes(b"fixture-index")
    output_dir = (tmp_path / "preview-output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(archive_root),
                "output_dir": str(output_dir),
                "archive_command": "fixity-scan",
                "archive_root": str(archive_root),
                "archive_index": str(archive_index),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    preview = body["data"]
    assert preview["pipeline"] == "archive-gate-a"
    assert preview["field_errors"] == []
    assert preview["field_warnings"] == []
    assert preview["readiness"]["status"] == "ready"
    assert preview["readiness"]["missing_prerequisites"] == []
    assert "fixity-scan" in preview["argv_preview"]
    assert "--archive-index" in preview["argv_preview"]
    assert preview["execution_args"] == preview["normalized_args"]


def test_config_preview_contract_normalizes_inactive_reconstruction_fields(
    client: TestClient,
    tmp_path: Path,
) -> None:
    fixture_input_dir = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_root"
    ).resolve()
    output_dir = (tmp_path / "preview-output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(fixture_input_dir),
                "output_dir": str(output_dir),
                "enable_reconstruction": False,
                "grouping_mode": "parent_dir",
                "reconstruction_iterations": 2000,
                "reconstruction_tier": "apex_research_ultra",
                "emit_scene_debug_bundle": True,
                "max_workers": 2,
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    assert body["error"] is None
    preview = body["data"]
    assert preview["pipeline"] == "lux-depth-v3"
    assert "grouping_mode" not in preview["normalized_args"]
    assert "reconstruction_tier" not in preview["normalized_args"]
    assert "emit_scene_debug_bundle" not in preview["normalized_args"]
    assert preview["execution_args"]["grouping_mode"] == "parent_dir"
    assert preview["execution_args"]["reconstruction_tier"] == "apex_research_ultra"
    assert preview["execution_args"]["emit_scene_debug_bundle"] is True
    inactive_fields = {item["field"]: item for item in preview["inactive_fields"]}
    assert inactive_fields["grouping_mode"]["reason"] == "enable_reconstruction_disabled"
    assert inactive_fields["reconstruction_tier"]["value"] == "apex_research_ultra"
    assert inactive_fields["emit_scene_debug_bundle"]["value"] is True
    assert preview["estimate_summary"]["summary_label"]
    assert preview["debug_bundle_summary"]["enabled"] is True
    assert preview["debug_bundle_summary"]["output_root"] == str(output_dir)
    assert preview["debug_bundle_summary"]["destination"] == "reconstruction/<scene-fingerprint>/debug"
    warning_reasons = {item["code"] for item in preview["field_warnings"]}
    assert "debug_bundle_sensitive_output" in warning_reasons


def test_config_preview_contract_omits_default_reconstruction_inactive_fields_when_toggle_is_off(
    client: TestClient,
    tmp_path: Path,
) -> None:
    fixture_input_dir = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_root"
    ).resolve()
    output_dir = (tmp_path / "preview-output-defaults").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(fixture_input_dir),
                "output_dir": str(output_dir),
                "enable_reconstruction": False,
                "grouping_mode": "single",
                "reconstruction_iterations": 1000,
                "reconstruction_tier": "apex_research",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    assert body["data"]["inactive_fields"] == []


def test_config_preview_contract_rejects_missing_cameras_sidecar_when_reconstruction_enabled(
    client: TestClient,
    tmp_path: Path,
) -> None:
    fixture_input_dir = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_root"
    ).resolve()
    output_dir = (tmp_path / "preview-output-sidecar").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_sidecar = fixture_input_dir / "missing_scene_cameras.json"

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(fixture_input_dir),
                "output_dir": str(output_dir),
                "enable_reconstruction": True,
                "non_commercial_ok": True,
                "accept_research_tools_license": True,
                "cameras_sidecar_path": str(missing_sidecar),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert errors["cameras_sidecar_path"]["code"] == "missing"
    assert errors["cameras_sidecar_path"]["message"] == "cameras_sidecar_path does not exist."
    warning_codes = {item["code"] for item in body["data"]["field_warnings"]}
    assert "camera_sidecar_missing" not in warning_codes
    assert "cameras_sidecar_path" not in body["data"]["normalized_args"]


def test_config_preview_contract_rejects_directory_cameras_sidecar_when_reconstruction_enabled(
    client: TestClient,
    tmp_path: Path,
) -> None:
    fixture_input_dir = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_root"
    ).resolve()
    output_dir = (tmp_path / "preview-output-sidecar-dir").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(fixture_input_dir),
                "output_dir": str(output_dir),
                "enable_reconstruction": True,
                "non_commercial_ok": True,
                "accept_research_tools_license": True,
                "cameras_sidecar_path": str(fixture_input_dir),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert errors["cameras_sidecar_path"]["code"] == "not_a_file"
    assert errors["cameras_sidecar_path"]["message"] == "cameras_sidecar_path must be a file."
    warning_codes = {item["code"] for item in body["data"]["field_warnings"]}
    assert "camera_sidecar_missing" not in warning_codes
    assert "cameras_sidecar_path" not in body["data"]["normalized_args"]


def test_config_preview_contract_rejects_invalid_cameras_sidecar_path_values(
    client: TestClient,
    tmp_path: Path,
) -> None:
    fixture_input_dir = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_root"
    ).resolve()
    output_dir = (tmp_path / "preview-output-sidecar-invalid").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(fixture_input_dir),
                "output_dir": str(output_dir),
                "enable_reconstruction": True,
                "non_commercial_ok": True,
                "accept_research_tools_license": True,
                "cameras_sidecar_path": "~/scene_cameras.json",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert errors["cameras_sidecar_path"]["code"] == "invalid_path_value"
    assert errors["cameras_sidecar_path"]["message"] == "cameras_sidecar_path contains an invalid path value."
    assert "cameras_sidecar_path" not in body["data"]["normalized_args"]


def test_config_preview_contract_sanitizes_archive_validation_errors(
    client: TestClient,
    tmp_path: Path,
) -> None:
    input_dir = (tmp_path / "archive-input").resolve()
    output_dir = (tmp_path / "archive-output").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "archive_command": "not-a-real-command",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    preview = body["data"]
    assert preview["field_errors"] == [
        {
            "field": "payload",
            "code": "invalid_archive_command",
            "message": "The selected archive command is not supported.",
        }
    ]


def test_portal_events_contract_sanitizes_metadata_and_writes_optional_log(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_log_path = tmp_path / "portal-events.jsonl"
    monkeypatch.setattr(orchestrator_app, "PORTAL_EVENT_LOG_PATH", event_log_path)

    response = client.post(
        "/v1/portal/events",
        json={
            "event_type": "config_exported",
            "pipeline": "lux-depth-v3",
            "surface": "effective_config",
            "field": "reconstruction_tier",
            "metadata": {
                "mode": "auto",
                "count": 2,
                "raw_path": "/private/tmp/should-not-pass",
            },
            "reasons": ["preview_ready", "Not A Token", "EXPORT"],
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.portal_event.v1"
    assert body["success"] is True
    assert body["error"] is None
    event = body["data"]["event"]
    assert event["event_type"] == "config_exported"
    assert event["surface"] == "effective_config"
    assert event["field"] == "reconstruction_tier"
    assert event["metadata"] == {"mode": "auto", "count": 2}
    assert event["reasons"] == ["preview_ready", "export"]
    lines = event_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_type"] == "config_exported"


def test_portal_events_invalid_payload_returns_sanitized_reason(client: TestClient) -> None:
    response = client.post(
        "/v1/portal/events",
        json={
            "event_type": "not-a-real-event",
            "pipeline": "lux-depth-v3",
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["message"] == "invalid portal telemetry payload"
    assert body["error"]["details"] == {"field": "payload", "reason": "invalid_event_type"}


def test_portal_events_allow_operator_console_review_and_stream_events(client: TestClient) -> None:
    response = client.post(
        "/v1/portal/events",
        json={
            "event_type": "stream_reconnected",
            "pipeline": "lux-depth-v3",
            "surface": "stream_transport",
            "metadata": {
                "attempt": 2,
                "job_id": "job_1234abcd",
                "transport": "fetch",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "stream_reconnected"
    assert event["surface"] == "stream_transport"
    assert event["metadata"] == {"attempt": 2, "job_id": "job_1234abcd", "transport": "fetch"}


def test_portal_events_contract_ignores_log_sink_write_failures(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    blocked_parent = tmp_path / "portal-events-blocked"
    blocked_parent.write_text("not-a-directory", encoding="utf-8")
    monkeypatch.setattr(orchestrator_app, "PORTAL_EVENT_LOG_PATH", blocked_parent / "events.jsonl")

    with caplog.at_level("WARNING"):
        response = client.post(
            "/v1/portal/events",
            json={
                "event_type": "config_exported",
                "pipeline": "lux-depth-v3",
                "surface": "effective_config",
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert "failed to persist portal event telemetry" in caplog.text


def test_portal_events_contract_offloads_log_persistence_to_thread(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    event_log_path = tmp_path / "portal-events-threaded.jsonl"

    async def fake_to_thread(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(orchestrator_app, "PORTAL_EVENT_LOG_PATH", event_log_path)
    monkeypatch.setattr(orchestrator_app.asyncio, "to_thread", fake_to_thread)

    response = client.post(
        "/v1/portal/events",
        json={
            "event_type": "config_exported",
            "pipeline": "lux-depth-v3",
            "surface": "effective_config",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    assert len(calls) == 1
    func, args, kwargs = calls[0]
    assert func is orchestrator_app._persist_portal_event_record
    assert args[1] == event_log_path
    assert kwargs == {}
    assert event_log_path.read_text(encoding="utf-8").strip()


def test_portal_events_contract_skips_thread_offload_when_log_sink_is_unset(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_to_thread(*_args, **_kwargs):
        raise AssertionError("portal telemetry should not offload when no log sink is configured")

    monkeypatch.setattr(orchestrator_app, "PORTAL_EVENT_LOG_PATH", None)
    monkeypatch.setattr(orchestrator_app.asyncio, "to_thread", fail_to_thread)

    response = client.post(
        "/v1/portal/events",
        json={
            "event_type": "config_exported",
            "pipeline": "lux-depth-v3",
            "surface": "effective_config",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True


def test_readiness_contract_reports_pipeline_status_matrix(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_map = {
        "lux-depth-v3": {
            "status": "ready",
            "canonical_command": "lux-depth-v3",
            "missing_prerequisites": [],
            "runner_details": {"type": "python_module", "available": True},
            "notes": ["safe lane ready"],
            "canary_status": "degraded",
        },
        "archive-gate-a": {
            "status": "degraded",
            "canonical_command": "fixity-scan",
            "missing_prerequisites": [
                {
                    "reason": "archive_index_required",
                    "severity": "degraded",
                    "message": "archive_index is required for archive-gate-a dispatch.",
                    "field": "archive_index",
                }
            ],
            "runner_details": {"type": "python_script", "available": True},
            "notes": ["existing archive index required"],
        },
        "archive-gate-b": {
            "status": "blocked",
            "canonical_command": "bag-build",
            "missing_prerequisites": [
                {
                    "reason": "rights_manifest_required",
                    "severity": "blocked",
                    "message": "manifest_jsonl is required for archive-gate-b dispatch.",
                    "field": "manifest_jsonl",
                }
            ],
            "runner_details": {"type": "python_script", "available": True},
            "notes": ["rights manifest missing"],
        },
        "archive-gate-c": {
            "status": "ready",
            "canonical_command": "mets-export",
            "missing_prerequisites": [],
            "runner_details": {"type": "python_script", "available": True},
            "notes": ["fixture-backed prereqs satisfied"],
        },
    }

    def _fake_evaluate(pipeline: str, args=None, require_dispatch_inputs: bool = False):  # noqa: ANN001
        del args, require_dispatch_inputs
        return readiness_map[pipeline]

    monkeypatch.setattr(orchestrator_app, "_evaluate_pipeline_readiness", _fake_evaluate)

    response = client.get("/v1/readiness")
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.readiness.v1"
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["server"]["backend_live"] is True
    assert body["data"]["server"]["version"] == orchestrator_app.APP_VERSION
    assert body["data"]["server"]["auth_mode"] == "direct_debug"
    assert body["data"]["pipelines"]["lux-depth-v3"]["canary_status"] == "degraded"
    assert body["data"]["pipelines"]["archive-gate-a"]["status"] == "degraded"
    assert body["data"]["pipelines"]["archive-gate-b"]["status"] == "blocked"
    assert body["data"]["pipelines"]["archive-gate-c"]["canonical_command"] == "mets-export"


def test_jobs_list_and_detail_include_recovery_fields(client: TestClient) -> None:
    job = orchestrator_app.Job(
        id="job_contract_recovery",
        created_at=orchestrator_app._now(),
        state="failed",
        progress=55,
        request={"pipeline": "lux-depth-v3"},
        logs_tail=["line-a", "line-b"],
        artifacts={
            "output_dir": "/tmp/out",
            "items": [{"artifact_type": "metadata", "path": "manifest.json", "relative_path": "manifest.json"}],
            "indexed_count": 1,
            "truncated": False,
        },
        error={"code": "RUNNER_ERROR", "message": "boom", "details": {}},
    )
    orchestrator_app.JOBS[job.id] = job

    list_response = client.get("/v1/jobs")
    list_body = list_response.json()
    assert list_response.status_code == 200
    assert list_body["schema"] == "tp.orchestrator.jobs.v1"
    first = list_body["data"]["jobs"][0]
    assert first["id"] == job.id
    assert first["events_url"] == f"/v1/jobs/{job.id}/events"
    assert first["error"]["code"] == "RUNNER_ERROR"
    assert first["artifacts"]["items"][0]["relative_path"] == "manifest.json"

    detail_response = client.get(f"/v1/jobs/{job.id}")
    detail_body = detail_response.json()
    assert detail_response.status_code == 200
    assert detail_body["schema"] == "tp.orchestrator.job_status.v1"
    assert detail_body["data"]["events_url"] == f"/v1/jobs/{job.id}/events"
    assert detail_body["data"]["artifacts"]["indexed_count"] == 1
    assert detail_body["data"]["error"]["code"] == "RUNNER_ERROR"


def test_partial_run_card_promotes_reviewable_failed_job_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_card_2026-04-06_232022.json").write_text(
        json.dumps(
            {
                "batch_id": "2026-04-06_232022",
                "total_images": 5,
                "success_count": 4,
                "error_count": 1,
                "artifact_index": [
                    {
                        "artifact_type": "run_card",
                        "path": "run_card_2026-04-06_232022.json",
                        "relative_path": "run_card_2026-04-06_232022.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    job = orchestrator_app.Job(
        id="job_partial_review",
        created_at=orchestrator_app._now(),
        state="failed",
        exit_code=1,
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        error={
            "code": "RUNNER_EXIT_NONZERO",
            "message": "runner exited with code 1",
            "details": {"exit_code": 1},
        },
    )

    orchestrator_app._index_job_artifacts(job)
    summary = orchestrator_app._refresh_job_run_summary(job)

    assert summary["partial"] is True
    assert summary["success_count"] == 4
    assert summary["error_count"] == 1
    assert job.state == "partial"
    assert job.run_summary["batch_id"] == "2026-04-06_232022"
    assert job.error["code"] == "RUNNER_PARTIAL_FAILURE"


def test_partial_run_summary_prefers_newest_run_card_when_output_dir_reused(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_card_2026-04-06_232022.json").write_text(
        json.dumps(
            {
                "batch_id": "2026-04-06_232022",
                "total_images": 2,
                "success_count": 1,
                "error_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "run_card_2026-04-07_001500.json").write_text(
        json.dumps(
            {
                "batch_id": "2026-04-07_001500",
                "total_images": 5,
                "success_count": 4,
                "error_count": 1,
            }
        ),
        encoding="utf-8",
    )

    job = orchestrator_app.Job(
        id="job_partial_review_reused_output_dir",
        created_at=orchestrator_app._now(),
        state="failed",
        exit_code=1,
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        error={
            "code": "RUNNER_EXIT_NONZERO",
            "message": "runner exited with code 1",
            "details": {"exit_code": 1},
        },
    )

    orchestrator_app._index_job_artifacts(job)
    summary = orchestrator_app._refresh_job_run_summary(job)

    assert summary["batch_id"] == "2026-04-07_001500"
    assert summary["success_count"] == 4
    assert summary["error_count"] == 1
    assert job.state == "partial"


def test_partial_run_summary_prefers_newest_batch_manifest_when_run_card_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "batch_2026-04-06_232022.json").write_text(
        json.dumps(
            {
                "batch_id": "2026-04-06_232022",
                "results": [{"status": "ok"}, {"status": "error"}],
                "stats": {"total_images": 2},
            }
        ),
        encoding="utf-8",
    )
    (manifests_dir / "batch_2026-04-07_001500.json").write_text(
        json.dumps(
            {
                "batch_id": "2026-04-07_001500",
                "results": [{"status": "ok"}] * 4 + [{"status": "error"}],
                "stats": {"total_images": 5},
            }
        ),
        encoding="utf-8",
    )

    job = orchestrator_app.Job(
        id="job_partial_review_manifest_fallback",
        created_at=orchestrator_app._now(),
        state="failed",
        exit_code=1,
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        error={
            "code": "RUNNER_EXIT_NONZERO",
            "message": "runner exited with code 1",
            "details": {"exit_code": 1},
        },
    )

    orchestrator_app._index_job_artifacts(job)
    summary = orchestrator_app._refresh_job_run_summary(job)

    assert summary["source"] == "batch_manifest"
    assert summary["batch_id"] == "2026-04-07_001500"
    assert summary["success_count"] == 4
    assert summary["error_count"] == 1
    assert job.state == "partial"


def test_jobs_list_and_detail_include_partial_run_summary(client: TestClient) -> None:
    job = orchestrator_app.Job(
        id="job_contract_partial",
        created_at=orchestrator_app._now(),
        state="partial",
        progress=100,
        request={"pipeline": "lux-depth-v3"},
        artifacts={
            "output_dir": "/tmp/out",
            "items": [{"artifact_type": "metadata", "path": "run_card.json", "relative_path": "run_card.json"}],
            "indexed_count": 1,
            "truncated": False,
        },
        run_summary={
            "source": "run_card",
            "batch_id": "2026-04-06_232022",
            "total_images": 5,
            "success_count": 4,
            "error_count": 1,
            "partial": True,
            "reviewable_outputs": True,
        },
        error={
            "code": "RUNNER_PARTIAL_FAILURE",
            "message": "1/5 images failed; 4 outputs remain reviewable",
            "details": {"exit_code": 1},
        },
    )
    orchestrator_app.JOBS[job.id] = job

    list_response = client.get("/v1/jobs")
    list_body = list_response.json()
    assert list_response.status_code == 200
    first = list_body["data"]["jobs"][0]
    assert first["state"] == "partial"
    assert first["run_summary"]["partial"] is True
    assert first["run_summary"]["success_count"] == 4

    detail_response = client.get(f"/v1/jobs/{job.id}")
    detail_body = detail_response.json()
    assert detail_response.status_code == 200
    assert detail_body["data"]["state"] == "partial"
    assert detail_body["data"]["run_summary"]["error_count"] == 1


def test_job_artifact_endpoint_serves_indexed_binary_without_exposing_absolute_path(
    client: TestClient,
    tmp_path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "renders" / "hero.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"\x89PNG\r\n\x1a\npreview")

    job = orchestrator_app.Job(
        id="job_artifact_read",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app.JOBS[job.id] = job
    orchestrator_app._index_job_artifacts(job)

    response = client.get(f"/v1/jobs/{job.id}/artifacts/renders/hero.png")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["content-type"].startswith("image/png")
    assert "attachment" not in response.headers.get("content-disposition", "").lower()
    assert response.content == b"\x89PNG\r\n\x1a\npreview"
    assert str(output_dir) not in response.text


def test_job_artifact_endpoint_uses_existing_index_without_full_rescan(
    client: TestClient,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "renders" / "hero.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"\x89PNG\r\n\x1a\npreview")

    job = orchestrator_app.Job(
        id="job_artifact_cached",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        artifacts={
            "output_dir": str(output_dir),
            "items": [
                orchestrator_app._serialize_indexed_artifact(
                    job_id="job_artifact_cached",
                    relative_path="renders/hero.png",
                    path=artifact_path,
                )
            ],
            "indexed_count": 1,
            "truncated": False,
        },
    )
    orchestrator_app.JOBS[job.id] = job

    def _fail_reindex(_job) -> None:
        raise AssertionError("artifact fetch should not rebuild the full artifact index")

    monkeypatch.setattr(orchestrator_app, "_index_job_artifacts", _fail_reindex)

    response = client.get(f"/v1/jobs/{job.id}/artifacts/renders/hero.png")

    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n\x1a\npreview"


def test_job_artifact_endpoint_rejects_traversal_outside_job_output_dir(
    client: TestClient,
    tmp_path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    job = orchestrator_app.Job(
        id="job_artifact_traversal",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app.JOBS[job.id] = job

    response = client.get(f"/v1/jobs/{job.id}/artifacts/%2E%2E/secret.txt")
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "artifact_path_outside_job_output_dir"


def test_job_artifact_endpoint_uses_bounded_reason_for_absolute_path(
    client: TestClient,
    tmp_path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    job = orchestrator_app.Job(
        id="job_artifact_absolute",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app.JOBS[job.id] = job

    response = client.get(f"/v1/jobs/{job.id}/artifacts//tmp/secret.txt")
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "absolute_artifact_path"
    assert "/tmp/secret.txt" not in response.text


def test_job_artifact_endpoint_returns_typed_not_found_for_missing_file(
    client: TestClient,
    tmp_path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    job = orchestrator_app.Job(
        id="job_artifact_missing",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app.JOBS[job.id] = job

    response = client.get(f"/v1/jobs/{job.id}/artifacts/missing.png")
    body = response.json()

    assert response.status_code == 404
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["details"]["path"] == "missing.png"


def test_v1_routes_enforce_api_key_for_reads_and_events(client: TestClient) -> None:
    orchestrator_app.API_KEY_SECRET = "contract-secret"
    now = orchestrator_app._now()
    finished_job = orchestrator_app.Job(
        id="job_auth",
        created_at=now,
        finished_at=now,
        done_published_at=now,  # Required for SSE endpoint to synthesize done event
        state="succeeded",
        exit_code=0,
        request={"pipeline": "lux-depth-v3"},
    )
    orchestrator_app.JOBS[finished_job.id] = finished_job
    orchestrator_app.EVENT_SUBSCRIBERS[finished_job.id] = {}

    list_unauthorized = client.get("/v1/jobs", headers={"x-api-key": "wrong"})
    assert list_unauthorized.status_code == 401
    assert list_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    list_authorized = client.get("/v1/jobs", headers={"x-api-key": "contract-secret"})
    assert list_authorized.status_code == 200
    assert list_authorized.json()["success"] is True

    events_unauthorized = client.get(f"/v1/jobs/{finished_job.id}/events", headers={"x-api-key": "wrong"})
    assert events_unauthorized.status_code == 401
    assert events_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    events_authorized = client.get(f"/v1/jobs/{finished_job.id}/events", headers={"x-api-key": "contract-secret"})
    assert events_authorized.status_code == 200
    assert "event: state" in events_authorized.text
    assert "event: done" in events_authorized.text


def test_config_preview_and_portal_event_routes_enforce_api_key(client: TestClient, tmp_path: Path) -> None:
    fixture_input_dir = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_root"
    ).resolve()
    output_dir = (tmp_path / "protected-preview-out").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_unauthorized = client.get(
        "/v1/config-metadata",
        params={"pipeline": "lux-depth-v3"},
        headers={"x-api-key": "wrong"},
    )
    assert metadata_unauthorized.status_code == 401
    assert metadata_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    preview_unauthorized = client.post(
        "/v1/config-preview",
        headers={"x-api-key": "wrong"},
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(fixture_input_dir),
                "output_dir": str(output_dir),
            },
        },
    )
    assert preview_unauthorized.status_code == 401
    assert preview_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    telemetry_unauthorized = client.post(
        "/v1/portal/events",
        headers={"x-api-key": "wrong"},
        json={
            "event_type": "config_exported",
            "pipeline": "lux-depth-v3",
            "surface": "effective_config",
        },
    )
    assert telemetry_unauthorized.status_code == 401
    assert telemetry_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"


def test_slash_redirect_variants_of_protected_preview_routes_enforce_api_key_before_redirect(
    client: TestClient,
    tmp_path: Path,
) -> None:
    fixture_input_dir = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_root"
    ).resolve()
    output_dir = (tmp_path / "protected-preview-out-slash").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_unauthorized = client.get(
        "/v1/config-metadata/",
        params={"pipeline": "lux-depth-v3"},
        headers={"x-api-key": "wrong"},
        follow_redirects=False,
    )
    assert metadata_unauthorized.status_code == 401
    assert metadata_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    preview_unauthorized = client.post(
        "/v1/config-preview/",
        headers={"x-api-key": "wrong"},
        follow_redirects=False,
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(fixture_input_dir),
                "output_dir": str(output_dir),
            },
        },
    )
    assert preview_unauthorized.status_code == 401
    assert preview_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    telemetry_unauthorized = client.post(
        "/v1/portal/events/",
        headers={"x-api-key": "wrong"},
        follow_redirects=False,
        json={
            "event_type": "config_exported",
            "pipeline": "lux-depth-v3",
            "surface": "effective_config",
        },
    )
    assert telemetry_unauthorized.status_code == 401
    assert telemetry_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"


def test_v1_routes_fail_closed_when_auth_enforced_without_secret(client: TestClient) -> None:
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    orchestrator_app.API_KEY_SECRET = ""

    response = client.get("/v1/jobs", headers={"x-api-key": "irrelevant"})
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "AUTH_CONFIGURATION_ERROR"
    assert body["error"]["details"]["env"] == "TP_API_KEY"

    config_response = client.get("/v1/config-metadata", params={"pipeline": "lux-depth-v3"})
    assert config_response.status_code == 503
    config_body = config_response.json()
    assert config_body["error"]["code"] == "AUTH_CONFIGURATION_ERROR"
    assert config_body["error"]["details"]["env"] == "TP_API_KEY"


def test_invalid_job_payload_returns_typed_invalid_argument(client: TestClient) -> None:
    response = client.post("/v1/jobs", json={"pipeline": "not-allowed", "args": {}})
    body = response.json()
    assert response.status_code == 400
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"


def test_archive_gate_pipeline_submission_returns_job_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    archive_index.write_bytes(b"fixture-index")

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(archive_root),
                "output_dir": str(tmp_path / "out"),
                "archive_command": "fixity-scan",
                "archive_index": str(archive_index),
                "archive_root": str(archive_root),
            },
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["id"].startswith("job_")


def test_create_job_preserves_raw_request_and_internal_execution_args(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "/tests/fixtures/archive_small/archive_root",
                "output_dir": "/tests/fixtures/portal_contract_output/http_effective_request_contract",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    job_id = body["data"]["id"]
    job = orchestrator_app.JOBS[job_id]
    assert job.request["args"]["input_dir"] == "/tests/fixtures/archive_small/archive_root"
    assert job.request["args"]["output_dir"] == "/tests/fixtures/portal_contract_output/http_effective_request_contract"
    assert job.effective_request["args"]["input_dir"] == "./tests/fixtures/archive_small/archive_root"
    assert (
        job.effective_request["args"]["output_dir"]
        == "./tests/fixtures/portal_contract_output/http_effective_request_contract"
    )


def test_archive_gate_b_submission_fails_closed_without_rights_manifest(client: TestClient) -> None:
    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-b",
            "args": {
                "input_dir": "./in",
                "output_dir": "./out",
                "archive_command": "bag-build",
            },
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["field"] == "manifest_jsonl"
    assert body["error"]["details"]["reason"] == "required"


def test_oversized_v1_request_returns_typed_413_envelope(client: TestClient) -> None:
    orchestrator_app.MAX_REQUEST_BYTES = 32
    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "lux-depth-v3",
            "args": {"input_dir": "a" * 64, "output_dir": "b" * 64},
        },
    )
    body = response.json()
    assert response.status_code == 413
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["success"] is False
    assert body["error"]["code"] == "REQUEST_TOO_LARGE"


def test_v1_jobs_rejects_requests_outside_allowed_roots(client: TestClient, tmp_path) -> None:
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    previous_path_roots = orchestrator_app.ALLOWED_PATH_ROOTS
    try:
        allowed_root = (tmp_path / "allowed").resolve()
        allowed_root.mkdir(parents=True, exist_ok=True)
        orchestrator_app.ALLOWED_INPUT_ROOTS = [allowed_root]
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = [allowed_root]
        orchestrator_app.ALLOWED_PATH_ROOTS = [allowed_root]

        response = client.post(
            "/v1/jobs",
            json={
                "pipeline": "lux-depth-v3",
                "args": {"input_dir": "./input", "output_dir": "./output"},
            },
        )
        body = response.json()
    finally:
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
        orchestrator_app.ALLOWED_PATH_ROOTS = previous_path_roots

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "path_outside_allowed_roots"


def test_v1_jobs_archive_gate_rejects_unsafe_input_dir_with_typed_error(client: TestClient) -> None:
    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-b",
            "args": {
                "input_dir": "~/.ssh",
                "output_dir": "./output",
                "archive_command": "bag-build",
            },
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_path_value"
    assert body["error"]["details"]["field"] == "input_dir"


def test_v1_jobs_rejects_when_max_concurrent_jobs_reached(client: TestClient) -> None:
    previous_limit = orchestrator_app.MAX_CONCURRENT_JOBS
    try:
        orchestrator_app.MAX_CONCURRENT_JOBS = 1
        orchestrator_app.JOBS["job_busy"] = orchestrator_app.Job(
            id="job_busy",
            created_at=orchestrator_app._now(),
            state="running",
            request={"pipeline": "lux-depth-v3", "args": {"input_dir": "./input", "output_dir": "./output"}},
        )
        response = client.post(
            "/v1/jobs",
            json={
                "pipeline": "lux-depth-v3",
                "args": {"input_dir": "./input", "output_dir": "./output"},
            },
        )
        body = response.json()
    finally:
        orchestrator_app.MAX_CONCURRENT_JOBS = previous_limit
        orchestrator_app.JOBS.clear()

    assert response.status_code == 429
    assert body["error"]["code"] == "RATE_LIMITED"
    assert body["error"]["details"]["active_jobs"] == 1
    assert body["error"]["details"]["max_concurrent_jobs"] == 1


def test_unknown_v1_route_returns_typed_not_found_envelope(client: TestClient) -> None:
    v1_missing = client.get("/v1/not-a-route")
    assert v1_missing.status_code == 404
    assert v1_missing.json()["error"]["code"] == "NOT_FOUND"

    non_v1_missing = client.get("/not-a-route")
    assert non_v1_missing.status_code == 404
    assert non_v1_missing.json() == {"detail": "Not Found"}


def test_http_exception_handler_preserves_headers_for_v1_and_non_v1(client: TestClient) -> None:
    v1_method_not_allowed = client.get("/v1/jobs/job_method/cancel")
    assert v1_method_not_allowed.status_code == 405
    assert v1_method_not_allowed.json()["error"]["code"] == "METHOD_NOT_ALLOWED"
    assert v1_method_not_allowed.json()["error"]["message"] == "method not allowed"
    assert v1_method_not_allowed.headers.get("allow") == "POST"

    non_v1_method_not_allowed = client.post("/ready")
    assert non_v1_method_not_allowed.status_code == 405
    assert non_v1_method_not_allowed.json() == {"detail": "Method Not Allowed"}
    assert non_v1_method_not_allowed.headers.get("allow") == "GET"


def test_request_validation_errors_return_typed_envelope_for_v1(client: TestClient) -> None:
    response = client.get("/v1/jobs", params={"limit": "not-an-int"})
    body = response.json()
    assert response.status_code == 400
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"] == {
        "path": "/v1/jobs",
        "reason": "request_validation_failed",
    }


def test_http_exception_handler_sanitizes_v1_exception_detail_and_logs_it(caplog: pytest.LogCaptureFixture) -> None:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/test",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    request = StarletteRequest(scope)

    with caplog.at_level(logging.WARNING):
        response = asyncio.run(
            orchestrator_app.http_exception_handler(
                request,
                StarletteHTTPException(status_code=500, detail="Traceback: /srv/app.py secret boom"),
            )
        )

    body = json.loads(response.body.decode("utf-8"))
    assert response.status_code == 500
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "internal server error"
    assert "Traceback" not in response.body.decode("utf-8")
    assert any("Sanitized HTTPException detail" in record.message for record in caplog.records)


def test_public_http_error_message_preserves_safe_request_size_detail() -> None:
    previous_max_request_bytes = orchestrator_app.MAX_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 123
        message = orchestrator_app._public_http_error_message(413)
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_max_request_bytes

    assert message == "request body too large (max 123 bytes)"


def test_http_exception_handler_preserves_safe_413_detail_for_v1_requests() -> None:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/jobs",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    request = StarletteRequest(scope)

    previous_max_request_bytes = orchestrator_app.MAX_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 123
        response = asyncio.run(
            orchestrator_app.http_exception_handler(
                request,
                StarletteHTTPException(status_code=413, detail="internal body parsing detail"),
            )
        )
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_max_request_bytes

    body = json.loads(response.body.decode("utf-8"))
    assert response.status_code == 413
    assert body["error"]["code"] == "REQUEST_TOO_LARGE"
    assert body["error"]["message"] == "request body too large (max 123 bytes)"
    assert body["error"]["details"] == {"path": "/v1/jobs"}


def test_job_events_stream_emits_state_log_progress_artifact_done(client: TestClient, monkeypatch) -> None:
    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "running"
        job.started_at = orchestrator_app._now()
        await orchestrator_app._publish_event(job.id, "state", {"id": job.id, "state": "running", "progress": 0})

        # Wait for stream subscribers so events are deterministic in test collection.
        for _ in range(200):
            if orchestrator_app.EVENT_SUBSCRIBERS.get(job.id):
                break
            await asyncio.sleep(0.005)
        if not orchestrator_app.EVENT_SUBSCRIBERS.get(job.id):
            raise AssertionError(f"subscriber registration timed out for job {job.id}")

        log_line = "progress=33%"
        job.add_log(log_line)
        await orchestrator_app._publish_event(job.id, "log", {"id": job.id, "line": log_line})
        job.progress = 33
        await orchestrator_app._publish_event(job.id, "progress", {"id": job.id, "progress": job.progress})
        artifact = {
            "artifact_type": "metadata",
            "path": "report.json",
            "relative_path": "report.json",
            "size_bytes": 12,
        }
        job.artifacts = {"output_dir": "./output", "items": [artifact], "indexed_count": 1, "truncated": False}
        await orchestrator_app._publish_event(job.id, "artifact", {"id": job.id, **artifact})

        job.state = "succeeded"
        job.exit_code = 0
        await orchestrator_app._publish_event(
            job.id,
            "done",
            {
                "id": job.id,
                "state": job.state,
                "exit_code": job.exit_code,
                "error": job.error,
                "artifacts": job.artifacts,
            },
        )
        # Set timestamps AFTER publishing done event (matches real _run_job behavior)
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    create = client.post(
        "/v1/jobs",
        json={
            "pipeline": "lux-depth-v3",
            "args": {"input_dir": "./input", "output_dir": "./output"},
        },
    )
    assert create.status_code == 200
    job_id = create.json()["data"]["id"]

    with client.stream("GET", f"/v1/jobs/{job_id}/events") as stream_response:
        assert stream_response.status_code == 200
        events = _collect_sse_events(stream_response)

    event_names = [name for name, _payload in events]
    assert "state" in event_names
    assert "log" in event_names
    assert "progress" in event_names
    assert "artifact" in event_names
    assert event_names[-1] == "done"

    artifact_payload = next(payload for name, payload in events if name == "artifact")
    assert artifact_payload["artifact_type"] == "metadata"
    assert artifact_payload["relative_path"] == "report.json"


def test_artifact_indexing_truncation_visible_via_job_status(client: TestClient, tmp_path) -> None:
    orchestrator_app.MAX_INDEXED_ARTIFACTS = 2
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "render.png").write_bytes(b"png")
    (output_dir / "run.log").write_text("ok", encoding="utf-8")

    job = orchestrator_app.Job(
        id="job_artifact_truncation",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app._index_job_artifacts(job)
    orchestrator_app.JOBS[job.id] = job

    response = client.get(f"/v1/jobs/{job.id}")
    body = response.json()
    artifacts = body["data"]["artifacts"]
    assert response.status_code == 200
    assert artifacts["indexed_count"] == 2
    assert artifacts["truncated"] is True
    assert len(artifacts["items"]) == 2
    for item in artifacts["items"]:
        assert "artifact_type" in item
        assert "path" in item
        assert "relative_path" in item
        assert "size_bytes" in item
