#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP contract tests for the root FastAPI orchestrator app."""

from __future__ import annotations

import asyncio
import csv
import gzip
import importlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request as StarletteRequest

from transformation_portal.orchestrator import reset_singletons
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")


def _reset_job_repository() -> None:
    reset_singletons()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False


def _seed_job(job: Any) -> Any:
    repo = orchestrator_app._job_repository()
    asyncio.run(repo.create(orchestrator_app._record_from_job(job)))
    if job.artifacts or job.artifact_lookup:
        asyncio.run(repo.set_artifacts(job.id, job.artifacts, job.artifact_lookup))
    orchestrator_app.JOBS[job.id] = job
    orchestrator_app.EVENT_SUBSCRIBERS.setdefault(job.id, {})
    return job


def _sync_seeded_job(job: Any) -> Any:
    repo = orchestrator_app._job_repository()
    asyncio.run(orchestrator_app._persist_job_state(job))
    asyncio.run(repo.set_artifacts(job.id, job.artifacts, job.artifact_lookup))
    return job


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


def _flag_value(argv: list[str], flag: str) -> str:
    try:
        index = argv.index(flag)
    except ValueError as exc:
        raise AssertionError(f"missing flag {flag}") from exc
    if index + 1 >= len(argv):
        raise AssertionError(f"flag {flag} missing value")
    return argv[index + 1]


@pytest.fixture(autouse=True)
def _reset_orchestrator_globals() -> None:
    previous_api_key = orchestrator_app.API_KEY_SECRET
    previous_enforce_job_api_key = orchestrator_app.ENFORCE_JOB_API_KEY
    previous_allow_sse_query_api_key = orchestrator_app.ALLOW_SSE_QUERY_API_KEY
    previous_max_request_bytes = orchestrator_app.MAX_REQUEST_BYTES
    previous_max_upload_request_bytes = orchestrator_app.MAX_UPLOAD_REQUEST_BYTES
    previous_max_indexed_artifacts = orchestrator_app.MAX_INDEXED_ARTIFACTS
    previous_rate_limit_per_minute = orchestrator_app.RATE_LIMIT_PER_MINUTE
    previous_max_concurrent_jobs = orchestrator_app.MAX_CONCURRENT_JOBS
    previous_portal_upload_root = orchestrator_app.PORTAL_UPLOAD_ROOT
    previous_portal_upload_max_files = orchestrator_app.PORTAL_UPLOAD_MAX_FILES
    previous_portal_upload_max_fields = orchestrator_app.PORTAL_UPLOAD_MAX_FIELDS
    previous_portal_upload_max_part_bytes = orchestrator_app.PORTAL_UPLOAD_MAX_PART_BYTES
    previous_portal_upload_ttl_seconds = orchestrator_app.PORTAL_UPLOAD_TTL_SECONDS
    orchestrator_app.API_KEY_SECRET = "contract-secret"
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    orchestrator_app.ALLOW_SSE_QUERY_API_KEY = False
    _reset_job_repository()
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
        orchestrator_app.MAX_UPLOAD_REQUEST_BYTES = previous_max_upload_request_bytes
        orchestrator_app.MAX_INDEXED_ARTIFACTS = previous_max_indexed_artifacts
        orchestrator_app.RATE_LIMIT_PER_MINUTE = previous_rate_limit_per_minute
        orchestrator_app.MAX_CONCURRENT_JOBS = previous_max_concurrent_jobs
        orchestrator_app.PORTAL_UPLOAD_ROOT = previous_portal_upload_root
        orchestrator_app.PORTAL_UPLOAD_MAX_FILES = previous_portal_upload_max_files
        orchestrator_app.PORTAL_UPLOAD_MAX_FIELDS = previous_portal_upload_max_fields
        orchestrator_app.PORTAL_UPLOAD_MAX_PART_BYTES = previous_portal_upload_max_part_bytes
        orchestrator_app.PORTAL_UPLOAD_TTL_SECONDS = previous_portal_upload_ttl_seconds
        orchestrator_app.JOBS.clear()
        orchestrator_app.EVENT_SUBSCRIBERS.clear()
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()
        _reset_job_repository()


@pytest.fixture(name="client")
def _client_fixture() -> TestClient:
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as test_client:
        yield test_client


# Healthcheck contract tests for /healthz and /ready were extracted to
# tests/test_app_healthcheck_contract.py — see that file's module
# docstring for the rationale (first family-scoped slice of this
# historically-monolithic file).


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
    assert body["features"]["artifactViewerModal"] is False
    assert body["features"]["reviewSurfaceDeferred"] is False
    assert body["features"]["stagedUploads"] is False
    assert body["features"]["rumTelemetry"] is False
    assert body["features"]["fastVlmCaptioning"] is False


def test_portal_bootstrap_exposes_artifact_viewer_rollout_flag_when_enabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "contract-smoke")

    response = client.get("/portal/bootstrap")

    assert response.status_code == 200
    assert response.json()["features"]["artifactViewerModal"] is True


def test_portal_bootstrap_exposes_review_surface_defer_rollout_flag_when_enabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "contract-smoke")

    response = client.get("/portal/bootstrap")

    assert response.status_code == 200
    assert response.json()["features"]["reviewSurfaceDeferred"] is True


def test_portal_bootstrap_exposes_staged_uploads_rollout_flag_when_enabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "contract-smoke")

    response = client.get("/portal/bootstrap")

    assert response.status_code == 200
    assert response.json()["features"]["stagedUploads"] is True


def test_portal_bootstrap_exposes_rum_rollout_flag_when_enabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "contract-smoke")

    response = client.get("/portal/bootstrap")

    assert response.status_code == 200
    assert response.json()["features"]["rumTelemetry"] is True


def test_portal_bootstrap_exposes_fastvlm_captioning_rollout_flag_when_enabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "contract-smoke")

    response = client.get("/portal/bootstrap")

    assert response.status_code == 200
    assert response.json()["features"]["fastVlmCaptioning"] is True


def test_portal_bootstrap_and_v1_echo_traceparent_header(client: TestClient) -> None:
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

    bootstrap_response = client.get("/portal/bootstrap", headers={"traceparent": traceparent})
    metadata_response = client.get(
        "/v1/config-metadata",
        headers={"traceparent": traceparent},
        params={"pipeline": "lux-depth-v3"},
    )

    assert bootstrap_response.status_code == 200
    assert bootstrap_response.headers["traceparent"] == traceparent
    assert metadata_response.status_code == 200
    assert metadata_response.headers["traceparent"] == traceparent


def test_staged_upload_route_requires_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "1")
    orchestrator_app.PORTAL_UPLOAD_ROOT = tmp_path / "uploads"

    with TestClient(orchestrator_app.app) as unauthenticated_client:
        response = unauthenticated_client.post(
            "/v1/uploads/staging",
            files=[("files", ("sample.txt", b"hello", "text/plain"))],
        )

    body = response.json()
    assert response.status_code == 401
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert body["error"]["details"]["path"] == "/v1/uploads/staging"


def test_staged_upload_route_returns_not_found_when_disabled(client: TestClient) -> None:
    response = client.post(
        "/v1/uploads/staging",
        files=[("files", ("sample.txt", b"hello", "text/plain"))],
    )

    body = response.json()
    assert response.status_code == 404
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["details"]["path"] == "/v1/uploads/staging"


def test_staged_upload_route_stages_files_and_writes_artifacts(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "100")
    orchestrator_app.PORTAL_UPLOAD_ROOT = tmp_path / "uploads"

    response = client.post(
        "/v1/uploads/staging",
        data={
            "client_manifest": json.dumps(
                {
                    "schema": "tp.portal.upload_manifest.v1",
                    "files": [
                        {"relative_path": "nested/sample.txt", "size_bytes": 11},
                        {"relative_path": "nested/child/readme.md", "size_bytes": 5},
                    ],
                }
            )
        },
        files=[
            ("files", ("nested/sample.txt", b"hello world", "text/plain")),
            ("files", ("nested/child/readme.md", b"# hi\n", "text/markdown")),
        ],
    )

    body = response.json()
    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.upload_staging.v1"
    assert body["success"] is True

    data = body["data"]
    input_dir = Path(data["input_dir"])
    metadata_dir = Path(data["metadata_dir"])
    assert input_dir.parent.parent == orchestrator_app.PORTAL_UPLOAD_ROOT
    assert (input_dir / "nested" / "sample.txt").read_text(encoding="utf-8") == "hello world"
    assert (input_dir / "nested" / "child" / "readme.md").read_text(encoding="utf-8") == "# hi\n"

    baseline_manifest_path = Path(data["artifacts"]["baseline_manifest_path"])
    capture_metadata_path = Path(data["artifacts"]["capture_metadata_path"])
    upload_receipt_path = Path(data["artifacts"]["upload_receipt_path"])
    assert baseline_manifest_path.parent == metadata_dir
    assert capture_metadata_path.parent == metadata_dir
    assert upload_receipt_path.parent == metadata_dir

    baseline_payload = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    assert baseline_payload["schema"] == "tp.meta.baseline_manifest.v1"
    assert baseline_payload["record_count"] == 2
    assert [record["relative_path"] for record in baseline_payload["records"]] == [
        "nested/child/readme.md",
        "nested/sample.txt",
    ]
    assert json.loads(capture_metadata_path.read_text(encoding="utf-8")) == []

    receipt_payload = json.loads(upload_receipt_path.read_text(encoding="utf-8"))
    assert receipt_payload["schema"] == "tp.orchestrator.upload_staging.v1"
    assert data["received_at_epoch_seconds"] > 0
    assert data["summary"]["top_level_roots"] == ["nested"]
    assert receipt_payload["summary"]["file_count"] == 2
    assert receipt_payload["summary"]["total_bytes"] == 16
    assert receipt_payload["summary"]["top_level_roots"] == ["nested"]


def test_staged_upload_route_rejects_invalid_relative_paths(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "100")
    orchestrator_app.PORTAL_UPLOAD_ROOT = tmp_path / "uploads"

    response = client.post(
        "/v1/uploads/staging",
        files=[("files", ("../escape.txt", b"bad", "text/plain"))],
    )

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_relative_path"


def test_staged_upload_route_rejects_invalid_client_manifest(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "100")
    orchestrator_app.PORTAL_UPLOAD_ROOT = tmp_path / "uploads"

    response = client.post(
        "/v1/uploads/staging",
        data={"client_manifest": json.dumps({"files": []})},
        files=[("files", ("sample.txt", b"hello", "text/plain"))],
    )

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["field"] == "client_manifest"


def test_staged_upload_route_returns_typed_413_for_oversized_parts(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "100")
    orchestrator_app.PORTAL_UPLOAD_ROOT = tmp_path / "uploads"
    previous_part_limit = orchestrator_app.PORTAL_UPLOAD_MAX_PART_BYTES
    try:
        orchestrator_app.PORTAL_UPLOAD_MAX_PART_BYTES = 4
        response = client.post(
            "/v1/uploads/staging",
            files=[("files", ("sample.txt", b"hello", "text/plain"))],
        )
    finally:
        orchestrator_app.PORTAL_UPLOAD_MAX_PART_BYTES = previous_part_limit

    body = response.json()
    assert response.status_code == 413
    assert body["error"]["code"] == "REQUEST_TOO_LARGE"
    assert body["error"]["details"] == {
        "field": "files",
        "reason": "multipart_part_too_large",
    }


def test_staged_upload_route_uses_upload_specific_request_size_limit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "1")
    previous_max_request_bytes = orchestrator_app.MAX_REQUEST_BYTES
    previous_max_upload_request_bytes = orchestrator_app.MAX_UPLOAD_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 1024
        orchestrator_app.MAX_UPLOAD_REQUEST_BYTES = 64
        response = client.post(
            "/v1/uploads/staging",
            content=b"x" * 128,
            headers={"content-type": "application/octet-stream"},
        )
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_max_request_bytes
        orchestrator_app.MAX_UPLOAD_REQUEST_BYTES = previous_max_upload_request_bytes

    body = response.json()
    assert response.status_code == 413
    assert body["error"]["code"] == "REQUEST_TOO_LARGE"
    assert body["error"]["message"] == "request body too large (max 64 bytes)"
    assert body["error"]["details"] == {
        "path": "/v1/uploads/staging",
        "max_request_bytes": 64,
    }


def test_staged_upload_route_returns_not_found_when_rollout_disabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "0")

    response = client.post(
        "/v1/uploads/staging",
        files=[("files", ("sample.txt", b"hello", "text/plain"))],
    )

    body = response.json()
    assert response.status_code == 404
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["details"]["path"] == "/v1/uploads/staging"


def test_root_ui_response_is_not_cached(client: TestClient) -> None:
    bundle = orchestrator_app._get_portal_asset_bundle()
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
    assert f'<link rel="stylesheet" href="{bundle.urls["portal.css"]}"' in response.text
    assert f'<script src="{bundle.urls["portal.js"]}" defer></script>' in response.text
    assert "<style>" not in response.text
    assert "<script>" not in response.text
    assert "Content-Security-Policy" not in response.text
    assert "Remember in local storage" not in response.text
    assert "Transformation Portal" in response.text


def test_portal_asset_endpoint_serves_css_and_js(client: TestClient) -> None:
    bundle = orchestrator_app._get_portal_asset_bundle()
    css_response = client.get(bundle.urls["portal.css"])
    shared_tokens_response = client.get(bundle.urls["shared-ui-tokens.css"])
    js_response = client.get(bundle.urls["portal.js"])

    assert css_response.status_code == 200
    assert css_response.headers["Cache-Control"] == orchestrator_app.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert css_response.headers["content-type"] == orchestrator_app.PORTAL_ASSET_MEDIA_TYPES["portal.css"]
    assert "@font-face" in css_response.text
    assert "Portal Sans" in css_response.text
    assert "@import" not in css_response.text
    assert "--ux-target-min-size:" in css_response.text
    assert "__PORTAL_" not in css_response.text
    assert bundle.urls["fonts/portal-sans.woff2"] in css_response.text
    assert bundle.urls["fonts/portal-mono.woff2"] in css_response.text
    assert "https://fonts.googleapis.com" not in css_response.text

    assert shared_tokens_response.status_code == 200
    assert shared_tokens_response.headers["Cache-Control"] == orchestrator_app.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert shared_tokens_response.headers["content-type"] == orchestrator_app.PORTAL_ASSET_MEDIA_TYPES["shared-ui-tokens.css"]
    assert re.search(r"--ux-target-min-size:\s*44px;", shared_tokens_response.text)

    assert js_response.status_code == 200
    assert js_response.headers["Cache-Control"] == orchestrator_app.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert js_response.headers["content-type"] == orchestrator_app.PORTAL_ASSET_MEDIA_TYPES["portal.js"]
    assert "BOOTSTRAP_TIMEOUT_MS=3500" in js_response.text


def test_portal_asset_endpoint_serves_repo_local_fonts(client: TestClient) -> None:
    bundle = orchestrator_app._get_portal_asset_bundle()
    response = client.get(bundle.urls["fonts/portal-sans.woff2"])

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == orchestrator_app.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert response.headers["content-type"] == orchestrator_app.PORTAL_ASSET_MEDIA_TYPES["fonts/portal-sans.woff2"]
    assert response.content


def test_portal_asset_endpoint_keeps_unversioned_and_stale_requests_backward_compatible(client: TestClient) -> None:
    response = client.get("/portal/assets/portal.css")
    stale_response = client.get("/portal/assets/portal.css", params={"v": "stale-version"})

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == orchestrator_app.PORTAL_ASSET_CACHE_CONTROL
    assert stale_response.status_code == 200
    assert stale_response.headers["Cache-Control"] == orchestrator_app.PORTAL_ASSET_CACHE_CONTROL
    assert response.text == stale_response.text


def test_portal_css_asset_endpoint_returns_304_for_matching_etag(client: TestClient) -> None:
    css_asset = orchestrator_app._get_portal_css_asset()
    response = client.get(
        "/portal/assets/portal.css",
        params={"v": css_asset.fingerprint},
        headers={"If-None-Match": f'"{css_asset.fingerprint}"'},
    )

    assert response.status_code == 304
    assert response.headers["Cache-Control"] == orchestrator_app.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert response.headers["ETag"] == f'"{css_asset.fingerprint}"'
    assert response.content == b""


def test_portal_direct_asset_endpoint_returns_304_for_matching_etag(client: TestClient) -> None:
    current_fingerprint = orchestrator_app._get_portal_direct_asset_fingerprint("portal.js")
    response = client.get(
        "/portal/assets/portal.js",
        params={"v": current_fingerprint},
        headers={"If-None-Match": f'"{current_fingerprint}"'},
    )

    assert response.status_code == 304
    assert response.headers["Cache-Control"] == orchestrator_app.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert response.headers["ETag"] == f'"{current_fingerprint}"'
    assert response.content == b""


def test_portal_css_endpoint_does_not_depend_on_html_bundle_inputs(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    css_asset = orchestrator_app._get_portal_css_asset()
    monkeypatch.setattr(orchestrator_app, "PORTAL_HTML", tmp_path / "missing-portal.html")
    orchestrator_app._build_portal_asset_bundle.cache_clear()

    response = client.get("/portal/assets/portal.css", params={"v": css_asset.fingerprint})

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == orchestrator_app.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert response.headers["ETag"] == f'"{css_asset.fingerprint}"'
    assert response.text == css_asset.text


def test_portal_direct_asset_endpoint_does_not_depend_on_html_bundle_inputs(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_fingerprint = orchestrator_app._get_portal_direct_asset_fingerprint("fonts/portal-sans.woff2")
    monkeypatch.setattr(orchestrator_app, "PORTAL_HTML", tmp_path / "missing-portal.html")
    orchestrator_app._build_portal_asset_bundle.cache_clear()

    response = client.get(
        "/portal/assets/fonts/portal-sans.woff2",
        params={"v": current_fingerprint},
    )

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == orchestrator_app.PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL
    assert response.headers["ETag"] == f'"{current_fingerprint}"'
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
    assert premium["recommended_args"]["model_key"] == "da3-metric"
    assert premium["advanced_sections"] == []
    da3_research = next(item for item in body["data"]["presets"] if item["name"] == "depth-anything-v3.1-research-m4")
    assert da3_research["recommended_args"]["depth_backend"] == "da3"
    assert da3_research["recommended_args"]["model_key"] == "da3-research"
    depth_pro = next(item for item in body["data"]["presets"] if item["name"] == "depth-pro-research-m4")
    assert depth_pro["recommended_args"]["depth_backend"] == "depth_pro"


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
    assert body["data"]["backend_catalog"]["da3"]["policy_posture"]["code"] == "governed_default"
    assert body["data"]["backend_catalog"]["da3"]["default_model_key"] == "da3-metric"
    da3_model_options = {item["value"]: item for item in body["data"]["fields"]["model_key"]["options"]}
    assert da3_model_options["da3-metric"]["requires_non_commercial_ok"] is False
    assert da3_model_options["da3-research"]["requires_non_commercial_ok"] is True
    assert body["data"]["backend_catalog"]["depth_pro"]["required_acknowledgments"][0]["field"] == "non_commercial_ok"
    assert body["data"]["backend_catalog"]["sam2"]["checkpoint_expectation"]["field"] == "sam2_checkpoint_path"
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
    assert preview["captioning_summary"]["feature_enabled"] is False
    assert preview["captioning_summary"]["enabled"] is False
    assert preview["captioning_summary"]["used_for_quality_gate"] is False
    assert preview["captioning_summary"]["runtime_readiness"]["status"] == "off"
    assert preview["captioning_summary"]["runtime_readiness"]["verification_scope"] == "path-existence"


def test_lux_config_preview_rejects_fastvlm_captioning_when_feature_disabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", raising=False)
    monkeypatch.delenv("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", raising=False)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/lux_depth_captioning_disabled",
                "vlm_captioning_enabled": True,
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    preview = body["data"]
    errors = {item["field"]: item for item in preview["field_errors"]}
    assert errors["vlm_captioning_enabled"]["code"] == "captioning_feature_disabled"
    assert preview["normalized_args"]["vlm_captioning_enabled"] is False
    assert preview["captioning_summary"]["feature_enabled"] is False
    assert preview["captioning_summary"]["enabled"] is False
    assert preview["captioning_summary"]["runtime_readiness"]["status"] == "off"
    assert preview["argv_preview"] == ""


def test_lux_config_preview_accepts_fastvlm_captioning_aliases_when_enabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "captioning-contract")

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/lux_depth_captioning_enabled",
                "vlmCaptioningEnabled": True,
                "vlmCaptioningBackend": "fastvlm",
                "vlmCaptioningModel": "review",
                "vlmCaptioningProxyFormat": "jpeg",
                "vlmCaptioningMaxSidePx": 1200,
                "fastvlmPythonExecutable": "./.runtime/fastvlm/pytest-python-shim",
                "fastvlmMlxVlmDir": "./.runtime/fastvlm/pytest-mlx-vlm",
                "fastvlmTimeoutSeconds": 60,
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    preview = body["data"]
    warning_codes = {item["code"] for item in preview["field_warnings"]}
    normalized = preview["normalized_args"]
    assert preview["field_errors"] == []
    assert "vlm_captioning_advisory_only" in warning_codes
    assert "fastvlm_runtime_missing" in warning_codes
    assert "fastvlm_runtime_python_executable_missing" in warning_codes
    assert "fastvlm_runtime_mlx_vlm_dir_missing" in warning_codes
    assert normalized["vlm_captioning_enabled"] is True
    assert normalized["vlm_captioning_backend"] == "fastvlm"
    assert normalized["vlm_captioning_model"] == "review"
    assert normalized["vlm_captioning_proxy_format"] == "jpeg"
    assert normalized["vlm_captioning_max_side_px"] == 1200
    assert normalized["fastvlm_timeout_seconds"] == 60
    assert preview["execution_args"]["vlm_captioning_enabled"] is True
    assert preview["captioning_summary"]["feature_enabled"] is True
    assert preview["captioning_summary"]["enabled"] is True
    assert preview["captioning_summary"]["role"] == "advisory"
    assert preview["captioning_summary"]["used_for_quality_gate"] is False
    assert preview["captioning_summary"]["runtime_readiness"]["status"] == "missing_runtime"
    assert "--vlm-captioning on" in preview["argv_preview"]
    assert "--vlm-captioning-model review" in preview["argv_preview"]
    assert "--vlm-captioning-proxy-format jpeg" in preview["argv_preview"]
    assert "--fastvlm-timeout-seconds 60" in preview["argv_preview"]


def test_lux_config_preview_reports_fastvlm_runtime_ready_for_manifest_backed_paths(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "fastvlm"
    python_path = runtime_root / ".venv-fastvlm" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    python_path.chmod(0o755)
    mlx_dir = runtime_root / "mlx-vlm"
    model_dir = runtime_root / "checkpoints" / "FastVLM-0.5B-fp16"
    mlx_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(
        orchestrator_app,
        "FASTVLM_RUNTIME_ALLOWED_ROOTS",
        [*orchestrator_app.FASTVLM_RUNTIME_ALLOWED_ROOTS, Path(os.path.realpath(tmp_path))],
    )
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "captioning-contract")

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/lux_depth_captioning_ready",
                "vlm_captioning_enabled": True,
                "vlm_captioning_model": str(model_dir),
                "fastvlm_python_executable": str(python_path),
                "fastvlm_mlx_vlm_dir": str(mlx_dir),
            },
        },
    )

    assert response.status_code == 200
    preview = response.json()["data"]
    assert preview["field_errors"] == []
    assert preview["captioning_summary"]["runtime_status"] == "ready"
    runtime_status = preview["captioning_summary"]["runtime_path_status"]
    readiness = preview["captioning_summary"]["runtime_readiness"]
    assert readiness["status"] == "ready"
    assert readiness["verification_scope"] == "path-existence"
    assert runtime_status["python_executable"]["status"] == "ready"
    assert runtime_status["mlx_vlm_dir"]["status"] == "ready"
    assert runtime_status["model_path"]["status"] == "ready"
    assert readiness["checks"]["python_executable"]["status"] == "ready"
    assert readiness["checks"]["mlx_vlm_dir"]["status"] == "ready"
    assert readiness["checks"]["model_path"]["status"] == "ready"
    assert readiness["checks"]["python_executable"]["required"] is True
    assert readiness["checks"]["mlx_vlm_dir"]["required"] is True
    assert readiness["checks"]["model_path"]["required"] is True


def test_lux_config_preview_treats_default_fastvlm_venv_python_symlink_as_ready(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "fastvlm"
    python_path = runtime_root / ".venv-fastvlm" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(Path(sys.executable))
    mlx_dir = runtime_root / "mlx-vlm"
    model_dir = runtime_root / "checkpoints" / "FastVLM-0.5B-fp16"
    mlx_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(orchestrator_app, "default_fastvlm_runtime_root", lambda: runtime_root)
    monkeypatch.setattr(
        orchestrator_app,
        "FASTVLM_RUNTIME_ALLOWED_ROOTS",
        [*orchestrator_app.FASTVLM_RUNTIME_ALLOWED_ROOTS, Path(os.path.realpath(runtime_root))],
    )
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "captioning-contract")

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/lux_depth_captioning_default_python",
                "vlm_captioning_enabled": True,
                "vlm_captioning_model": str(model_dir),
            },
        },
    )

    assert response.status_code == 200
    summary = response.json()["data"]["captioning_summary"]
    assert summary["runtime_status"] == "ready"
    assert summary["runtime_readiness"]["status"] == "ready"
    assert summary["runtime_path_status"]["python_executable"]["status"] == "ready"


def test_lux_config_preview_validates_fastvlm_captioning_fields(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "captioning-contract")

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/lux_depth_captioning_invalid",
                "vlm_captioning_enabled": True,
                "vlm_captioning_backend": "other",
                "vlm_captioning_model": "not-a-role",
                "vlm_captioning_proxy_format": "gif",
                "vlm_captioning_max_side_px": 0,
                "fastvlm_python_executable": "/etc/passwd",
                "fastvlm_timeout_seconds": 0,
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    errors = {(item["field"], item["code"]) for item in body["data"]["field_errors"]}
    assert ("vlm_captioning_backend", "invalid_vlm_captioning_backend") in errors
    assert ("vlm_captioning_model", "invalid_vlm_captioning_model") in errors
    assert ("vlm_captioning_proxy_format", "invalid_vlm_captioning_proxy_format") in errors
    assert ("vlm_captioning_max_side_px", "invalid_vlm_captioning_max_side_px") in errors
    assert ("fastvlm_timeout_seconds", "invalid_fastvlm_timeout_seconds") in errors
    assert any(field == "fastvlm_python_executable" for field, _code in errors)
    summary = body["data"]["captioning_summary"]
    assert summary["runtime_readiness"]["status"] == "invalid_config"
    assert summary["runtime_readiness"]["checks"]["python_executable"]["status"] == "invalid_path"
    assert summary["runtime_readiness"]["checks"]["model_path"]["status"] == "invalid_path"
    assert summary["used_for_quality_gate"] is False
    assert body["data"]["argv_preview"] == ""


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
    (archive_root / "asset-001.dng").write_bytes(b"raw")
    archive_index = (tmp_path / "archive_index_normalized.csv.gz").resolve()
    _write_archive_index(archive_index, ["asset-001.dng"])
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
    assert preview["next_best_action"]["action"] == "dispatch_ready"
    assert preview["next_best_action"]["tone"] == "ready"
    assert "fixity-scan" in preview["argv_preview"]
    assert "--archive-index" in preview["argv_preview"]
    assert preview["execution_args"] == preview["normalized_args"]


def test_archive_gate_a_config_preview_blocks_archive_index_root_mismatch(
    client: TestClient,
    tmp_path: Path,
) -> None:
    archive_root = (tmp_path / "raw_root").resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "DJI_0018.DNG").write_bytes(b"raw")
    archive_index = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_index_normalized.csv.gz"
    )
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
    errors = {item["field"]: item for item in preview["field_errors"]}
    assert errors["archive_index"]["code"] == "archive_index_root_mismatch"
    assert "3/3 rows blocked" in errors["archive_index"]["message"]
    assert preview["readiness"]["status"] == "blocked"
    assert preview["argv_preview"] == ""


def test_archive_gate_a_config_preview_reports_missing_archive_root_on_root_field(
    client: TestClient,
    tmp_path: Path,
) -> None:
    archive_root = (tmp_path / "missing-root").resolve()
    archive_index = (tmp_path / "archive_index_normalized.csv.gz").resolve()
    _write_archive_index(archive_index, ["asset-001.dng"])
    output_dir = (tmp_path / "preview-output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(tmp_path),
                "output_dir": str(output_dir),
                "archive_command": "fixity-scan",
                "archive_root": str(archive_root),
                "archive_index": str(archive_index),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    preview = body["data"]
    errors = {item["field"]: item for item in preview["field_errors"]}
    assert errors["archive_root"]["code"] in {"missing", "not_a_directory"}
    assert "archive_index" not in errors
    assert preview["argv_preview"] == ""


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


def test_config_preview_rejects_untrusted_sam2_checkpoint_path(client: TestClient, tmp_path: Path) -> None:
    input_dir = (tmp_path / "preview-input").resolve()
    output_dir = (tmp_path / "preview-output").resolve()
    checkpoint_path = (tmp_path / "sam2-untrusted.pt").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"untrusted checkpoint bytes")

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "enable_segmentation": True,
                "segmentation_backend": "sam2",
                "sam2_checkpoint_path": str(checkpoint_path),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert errors["sam2_checkpoint_path"]["code"] == "untrusted_checkpoint_path"
    assert "sam2_checkpoint_path" not in body["data"]["normalized_args"]


def test_config_preview_contract_offloads_preview_build_to_thread(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_build_config_preview(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {
            "pipeline": payload["pipeline"],
            "normalized_args": {},
            "execution_args": {},
            "argv_preview": "",
            "field_errors": [],
            "field_warnings": [],
            "inactive_fields": [],
            "readiness": {"status": "ready"},
            "estimate_summary": {},
            "debug_bundle_summary": {},
            "next_best_action": {},
            "kwargs": kwargs,
        }

    async def fake_to_thread(func: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(orchestrator_app, "_build_config_preview", fake_build_config_preview)
    monkeypatch.setattr(orchestrator_app.asyncio, "to_thread", fake_to_thread)

    response = client.post(
        "/v1/config-preview",
        json={"pipeline": "lux-depth-v3", "args": {"input_dir": "./input_images", "output_dir": "./output"}},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    assert len(calls) == 1
    func, args, kwargs = calls[0]
    assert func is fake_build_config_preview
    assert args == ({"pipeline": "lux-depth-v3", "args": {"input_dir": "./input_images", "output_dir": "./output"}},)
    assert kwargs["archive_index_scan_mode"] == "preview"
    assert kwargs["portal_actor"] == {}


def test_config_preview_accepts_repo_controlled_missing_sam2_checkpoint_path(
    client: TestClient,
    tmp_path: Path,
) -> None:
    input_dir = (tmp_path / "preview-input-managed").resolve()
    output_dir = (tmp_path / "preview-output-managed").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "enable_segmentation": True,
                "segmentation_backend": "sam2",
                "sam2_checkpoint_path": "./models/sam2/sam2.1_hiera_large.pt",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert "sam2_checkpoint_path" not in errors
    assert body["data"]["normalized_args"]["sam2_checkpoint_path"].endswith("models/sam2/sam2.1_hiera_large.pt")


def test_config_preview_rejects_non_file_sam2_checkpoint_path(client: TestClient, tmp_path: Path) -> None:
    input_dir = (tmp_path / "preview-input-dir").resolve()
    output_dir = (tmp_path / "preview-output-dir").resolve()
    checkpoint_dir = (tmp_path / "sam2-checkpoint-dir").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "enable_segmentation": True,
                "segmentation_backend": "sam2",
                "sam2_checkpoint_path": str(checkpoint_dir),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert errors["sam2_checkpoint_path"]["code"] == "invalid_path_value"
    assert "sam2_checkpoint_path" not in body["data"]["normalized_args"]


def test_config_preview_rejects_oversized_sam2_checkpoint_path(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = (tmp_path / "preview-input-oversized").resolve()
    output_dir = (tmp_path / "preview-output-oversized").resolve()
    checkpoint_path = (tmp_path / "sam2-oversized.pt").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"oversized")
    monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_CHECKSUM_MAX_BYTES", 1)

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "enable_segmentation": True,
                "segmentation_backend": "sam2",
                "sam2_checkpoint_path": str(checkpoint_path),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    errors = {item["field"]: item for item in body["data"]["field_errors"]}
    assert errors["sam2_checkpoint_path"]["code"] == "checkpoint_file_too_large"
    assert (
        errors["sam2_checkpoint_path"]["message"]
        == "Managed SAM2 checkpoint overrides exceed the checksum verification size limit."
    )
    assert "sam2_checkpoint_path" not in body["data"]["normalized_args"]


def test_lux_config_preview_preserves_custom_preset_and_advanced_sam2_controls(
    client: TestClient,
    tmp_path: Path,
) -> None:
    input_dir = (tmp_path / "preview-input-custom-sam2").resolve()
    output_dir = (tmp_path / "preview-output-custom-sam2").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "frame.jpg").write_bytes(b"fixture-image")

    response = client.post(
        "/v1/config-preview",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "preset": "custom",
                "enable_segmentation": True,
                "segmentation_backend": "sam2",
                "strict_segmentation": True,
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
                "emit_run_card": True,
                "run_card_version": "v2",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    preview = body["data"]
    normalized = preview["normalized_args"]

    assert preview["field_errors"] == []
    assert normalized["preset"] == "custom"
    assert normalized["segmentation_backend"] == "sam2"
    assert normalized["sam2_model_size"] == "large"
    assert normalized["sam2_tiling_enabled"] is True
    assert normalized["sam2_tile_size_px"] == 1536
    assert normalized["sam2_overlap_px"] == 256
    assert normalized["sam2_global_pass_longest_side"] == 1280
    assert normalized["sam2_max_concurrency"] == 1
    assert normalized["sam2_points_per_side"] == 32
    assert normalized["sam2_points_per_batch"] == 64
    assert normalized["sam2_pred_iou_thresh"] == pytest.approx(0.88)
    assert normalized["sam2_stability_score_thresh"] == pytest.approx(0.85)
    assert normalized["sam2_crop_n_layers"] == 1
    assert normalized["run_card_version"] == "v2"
    assert preview["execution_args"]["preset"] == "custom"
    assert preview["execution_args"]["sam2_tiling_enabled"] is True
    assert preview["execution_args"]["run_card_version"] == "v2"


@pytest.mark.parametrize(
    ("jobs_path", "expected_events_prefix"),
    [
        ("/v1/jobs", "/v1/jobs"),
        ("/v2/jobs", "/v2/jobs"),
    ],
)
def test_lux_jobs_dispatch_accepts_custom_manual_preset(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mark_da3_runtime_available: None,
    jobs_path: str,
    expected_events_prefix: str,
) -> None:
    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    input_dir = (tmp_path / "manual-input").resolve()
    output_dir = (tmp_path / "manual-output").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "frame.jpg").write_bytes(b"fixture-image")

    response = client.post(
        jobs_path,
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "preset": "custom",
                "quality_tier": "apex",
                "depth_backend": "da3",
                "enable_segmentation": True,
                "segmentation_backend": "sam2",
                "strict_segmentation": True,
                "materials_v3": True,
                "pbr": True,
                "emit_run_card": True,
                "run_card_version": "v2",
                "non_commercial_ok": True,
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["data"]["events_url"].startswith(f"{expected_events_prefix}/job_")
    job = orchestrator_app.JOBS[body["data"]["id"]]
    assert job.request["args"]["preset"] == "custom"
    assert job.effective_request["args"]["preset"] == "custom"


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


@pytest.mark.parametrize("event_type", ["artifact_viewer_opened", "artifact_viewer_fallback"])
def test_portal_events_accept_artifact_viewer_review_events(
    client: TestClient,
    event_type: str,
) -> None:
    response = client.post(
        "/v1/portal/events",
        json={
            "event_type": event_type,
            "pipeline": "lux-depth-v3",
            "surface": "artifact_review",
            "metadata": {
                "job_id": "job_1234abcd",
                "pipeline": "lux-depth-v3",
                "media_kind": "image",
                "artifact_fingerprint": "abcdef1234",
                "viewer_mode": "modal",
                "fallback_reason": "inline_preview_unavailable",
                "email": "admin@example.com",
                "raw_path": "/private/tmp/should-not-pass",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    event = body["data"]["event"]
    assert event["event_type"] == event_type
    assert event["surface"] == "artifact_review"
    assert event["metadata"] == {
        "artifact_fingerprint": "abcdef1234",
        "fallback_reason": "inline_preview_unavailable",
        "job_id": "job_1234abcd",
        "media_kind": "image",
        "viewer_mode": "modal",
    }


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


def test_portal_rum_contract_noops_cleanly_when_disabled(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rum_log_path = tmp_path / "portal-rum-disabled.jsonl"
    monkeypatch.setattr(orchestrator_app, "PORTAL_RUM_LOG_PATH", rum_log_path)
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

    response = client.post(
        "/v1/portal/rum",
        headers={"traceparent": traceparent},
        json={
            "event_type": "not-a-real-event",
            "route": "/not-portal",
            "view": "invalid",
            "value": "boom",
            "unit": "bad",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert response.headers["traceparent"] == traceparent
    assert body["schema"] == "tp.orchestrator.portal_rum_ingest.v1"
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"] == {"accepted": False, "disabled": True}
    assert rum_log_path.exists() is False


def test_portal_rum_contract_sanitizes_metadata_and_writes_optional_log(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rum_log_path = tmp_path / "portal-rum.jsonl"
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")
    monkeypatch.setattr(orchestrator_app, "PORTAL_RUM_LOG_PATH", rum_log_path)
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

    response = client.post(
        "/v1/portal/rum",
        headers={
            "traceparent": traceparent,
            "x-tp-actor": "admin",
            "x-tp-actor-email": "admin@example.com",
            "x-tp-actor-role": "admin",
        },
        json={
            "event_type": "queue_request",
            "route": "/portal",
            "view": "build",
            "metric": "submit",
            "value": 183.42,
            "unit": "ms",
            "metadata": {
                "transport": "fetch",
                "job_id": "job_1234abcd",
                "attempt": 2,
                "email": "admin@example.com",
                "path": "/private/tmp/should-not-pass",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert response.headers["traceparent"] == traceparent
    assert body["schema"] == "tp.orchestrator.portal_rum_ingest.v1"
    assert body["success"] is True
    assert body["error"] is None
    event = body["data"]["event"]
    assert body["data"]["accepted"] is True
    assert event["event_type"] == "queue_request"
    assert event["route"] == "/portal"
    assert event["view"] == "build"
    assert event["metric"] == "submit"
    assert event["value"] == 183.42
    assert event["unit"] == "ms"
    assert event["metadata"] == {"attempt": 2, "job_id": "job_1234abcd", "transport": "fetch"}
    assert event["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert event["cohort_bucket"] == orchestrator_app._stable_rollout_bucket("admin")
    assert event["auth_mode"] == "managed"
    assert "username" not in event
    assert "accessEmail" not in event
    persisted = json.loads(rum_log_path.read_text(encoding="utf-8").strip())
    assert persisted == event
    assert "admin@example.com" not in rum_log_path.read_text(encoding="utf-8")


def test_portal_rum_contract_avoids_high_volume_info_logs(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    with caplog.at_level(logging.INFO):
        response = client.post(
            "/v1/portal/rum",
            headers={"x-tp-actor": "admin"},
            json={
                "event_type": "queue_request",
                "route": "/portal",
                "view": "build",
                "metric": "submit",
                "value": 183.42,
                "unit": "ms",
            },
        )

    assert response.status_code == 200
    assert "portal_rum" not in caplog.text


@pytest.mark.parametrize(
    ("payload_overrides", "reason"),
    [
        ({"event_type": "not-a-real-event"}, "invalid_event_type"),
        ({"route": "/ready"}, "invalid_route"),
        ({"view": "invalid"}, "invalid_view"),
        ({"unit": "seconds"}, "invalid_unit"),
        ({"value": -1}, "invalid_value"),
        ({"metric": "restart"}, "invalid_metric"),
    ],
)
def test_portal_rum_invalid_payload_returns_sanitized_reason(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    payload_overrides: Dict[str, Any],
    reason: str,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")
    payload = {
        "event_type": "queue_request",
        "route": "/portal",
        "view": "build",
        "metric": "submit",
        "value": 183.42,
        "unit": "ms",
    }
    payload.update(payload_overrides)

    response = client.post("/v1/portal/rum", json=payload)
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["message"] == "invalid portal rum payload"
    assert body["error"]["details"] == {"field": "payload", "reason": reason}


def test_portal_rum_contract_accepts_landing_rendered_event(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "landing_rendered",
            "route": "/",
            "view": "landing",
            "metric": "duration",
            "value": 142.5,
            "unit": "ms",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "landing_rendered"
    assert event["route"] == "/"
    assert event["view"] == "landing"
    assert event["metric"] == "duration"
    assert event["value"] == 142.5
    assert event["unit"] == "ms"


def test_portal_rum_contract_accepts_login_rendered_event(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_rendered",
            "route": "/login",
            "view": "login",
            "metric": "duration",
            "value": 87.25,
            "unit": "ms",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "login_rendered"
    assert event["route"] == "/login"
    assert event["view"] == "login"


def test_portal_rum_contract_accepts_core_web_vital_from_root_route(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "core_web_vital",
            "route": "/",
            "view": "landing",
            "metric": "lcp",
            "value": 1820.5,
            "unit": "ms",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    assert body["data"]["event"]["route"] == "/"
    assert body["data"]["event"]["view"] == "landing"
    assert body["data"]["event"]["metric"] == "lcp"


def test_portal_rum_contract_accepts_core_web_vital_from_login_route(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "core_web_vital",
            "route": "/login",
            "view": "login",
            "metric": "cls",
            "value": 0.0125,
            "unit": "score",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    assert body["data"]["event"]["route"] == "/login"
    assert body["data"]["event"]["view"] == "login"
    assert body["data"]["event"]["metric"] == "cls"


def test_portal_rum_contract_accepts_login_submit_attempt(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_attempt",
            "route": "/login",
            "view": "login",
            "metric": "count",
            "value": 1,
            "unit": "count",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "login_submit_attempt"
    assert event["route"] == "/login"
    assert event["view"] == "login"
    assert event["metric"] == "count"
    assert event["unit"] == "count"
    assert event["metadata"] == {}


def test_portal_rum_contract_accepts_client_login_submit_attempt_metadata(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Survival contract for the client-side login_submit_attempt emission.

    The browser-side counterpart to the server-side login_submit_attempt
    (#1684) re-uses the existing event_type/metric/unit allowlist and
    carries the form-render-to-submit duration as metadata. This test
    pins that the existing _portal_sanitize_metadata accepts the
    {source: "client", duration_ms: <int>} shape unchanged so we never
    have to widen the backend allowlist.
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_attempt",
            "route": "/login",
            "view": "login",
            "metric": "count",
            "value": 1,
            "unit": "count",
            "metadata": {"source": "client", "duration_ms": 18420},
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "login_submit_attempt"
    assert event["metric"] == "count"
    # Both metadata keys survive the sanitizer round-trip unchanged.
    assert event["metadata"] == {"source": "client", "duration_ms": 18420}


def test_portal_rum_contract_drops_non_token_metadata_for_client_login_submit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the actual contract of ``_portal_sanitize_metadata`` for the
    client-side login_submit_attempt path: values that fail the
    ``_portal_is_token`` regex (emails with '@', free-text containing
    spaces, raw JWTs, etc.) are dropped entirely.

    Token-shaped metadata keys/values (``username``, ``csrf_token``,
    ``role``, etc.) DO survive the sanitizer if a caller stuffs them in.
    The client emitter never sets those keys — the contract is enforced
    at the source (``rum-client.js`` only ever emits
    ``{source, duration_ms}``). A sanitizer-level denylist is a
    separate, broader change deferred from this PR; aggregators that
    consume the persisted JSONL should treat any unexpected metadata
    key as suspect.
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_attempt",
            "route": "/login",
            "view": "login",
            "metric": "count",
            "value": 1,
            "unit": "count",
            "metadata": {
                "source": "client",
                "duration_ms": 12345,
                # Non-token VALUES (contain '@' or whitespace) are dropped
                # wholesale by the sanitizer's string branch.
                "email": "admin@example.com",
                "password": "correct horse battery staple",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    metadata = body["data"]["event"]["metadata"]

    # Allowed metadata: the two keys the client emitter actually sets.
    assert metadata.get("source") == "client"
    assert metadata.get("duration_ms") == 12345

    # Non-token VALUES are dropped: the sanitizer's string branch only
    # admits values passing _portal_is_token, so emails (contain '@')
    # and free-text passwords (contain spaces) are filtered out cleanly.
    assert "email" not in metadata
    assert "password" not in metadata


def test_portal_rum_contract_accepts_login_submit_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_success",
            "route": "/login",
            "view": "login",
            "metric": "duration",
            "value": 142.5,
            "unit": "ms",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "login_submit_success"
    assert event["metric"] == "duration"
    assert event["unit"] == "ms"
    assert event["value"] == 142.5
    assert event["metadata"] == {}


def test_portal_rum_contract_accepts_client_login_submit_success_metadata(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Survival contract for the client-side login_submit_success emission.

    The portal-bundle counterpart in ``portal.template.js`` posts
    ``{source: "client"}`` metadata to differentiate browser-side
    samples from the server-side emission. This test pins that the
    existing ``_portal_sanitize_metadata`` accepts the key without any
    allowlist widening, mirroring the equivalent attempt/failure
    survival contracts established in #1689 and #1694.
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_success",
            "route": "/portal",
            # The portal bundle normalizes state.currentView before
            # emitting; "overview" is the default first view after
            # bootstrap_ready, so it's the realistic value to assert.
            "view": "overview",
            "metric": "duration",
            "value": 312,
            "unit": "ms",
            "metadata": {"source": "client"},
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "login_submit_success"
    assert event["metric"] == "duration"
    assert event["unit"] == "ms"
    assert event["value"] == 312
    assert event["metadata"] == {"source": "client"}


def test_portal_rum_contract_drops_non_token_metadata_for_client_login_submit_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth for the client-side success path: even if a
    caller stuffs PII into login_submit_success metadata, the existing
    sanitizer drops values that fail ``_portal_is_token`` (emails with
    '@', free-text passwords with whitespace) before persistence. The
    portal-bundle emitter only ever sets ``{source: "client"}``; this
    test pins the round-trip discipline at the boundary.
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_success",
            "route": "/portal",
            "view": "overview",
            "metric": "duration",
            "value": 220,
            "unit": "ms",
            "metadata": {
                "source": "client",
                # Non-token VALUES (contain '@' or whitespace) are dropped
                # wholesale by the sanitizer's string branch.
                "email": "victim@example.com",
                "password": "correct horse battery staple",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    metadata = body["data"]["event"]["metadata"]

    # Allowed metadata: the single key the portal emitter actually sets.
    assert metadata.get("source") == "client"

    # Non-token VALUES are dropped wholesale.
    assert "email" not in metadata
    assert "password" not in metadata


def test_portal_rum_contract_accepts_login_submit_failure_with_failure_code(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_failure",
            "route": "/login",
            "view": "login",
            "metric": "duration",
            "value": 87.0,
            "unit": "ms",
            "metadata": {"failure_code": "invalid"},
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "login_submit_failure"
    assert event["metric"] == "duration"
    assert event["metadata"] == {"failure_code": "invalid"}
    # Round-trip discipline: PII must never appear in the persisted event,
    # even if a caller stuffs it into metadata.
    assert "username" not in event["metadata"]
    assert "accessEmail" not in event["metadata"]


@pytest.mark.parametrize(
    "failure_code",
    ["csrf", "configuration", "access", "throttled", "invalid"],
)
def test_portal_rum_contract_accepts_client_login_submit_failure_metadata(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
) -> None:
    """Survival contract for the client-side login_submit_failure emission.

    The browser-side counterpart in ``rum-client.js`` posts
    ``{source: "client", failure_code: <code>}`` for each of the five
    server-side LOGIN_RUM_FAILURE_CODES. This test pins that the existing
    ``_portal_sanitize_metadata`` accepts both keys together for every
    code, so we never have to widen ``PORTAL_ALLOWED_RUM_METRICS`` or the
    sanitizer to ship the failure mirror.
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_failure",
            "route": "/login",
            "view": "login",
            "metric": "duration",
            "value": 142,
            "unit": "ms",
            "metadata": {"source": "client", "failure_code": failure_code},
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "login_submit_failure"
    assert event["metric"] == "duration"
    assert event["unit"] == "ms"
    assert event["value"] == 142
    # Both keys survive the sanitizer with values intact and lowercased.
    assert event["metadata"] == {"source": "client", "failure_code": failure_code}


def test_portal_rum_contract_drops_non_token_metadata_for_client_login_submit_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: even if a caller stuffs PII into the
    login_submit_failure metadata, ``_portal_sanitize_metadata`` drops
    values that fail ``_portal_is_token`` (emails with '@', free-text
    passwords with whitespace) before persistence. The client emitter
    never sets those keys — the contract is enforced at the source
    (``rum-client.js`` only ever emits ``{source, failure_code}``).
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_failure",
            "route": "/login",
            "view": "login",
            "metric": "duration",
            "value": 95,
            "unit": "ms",
            "metadata": {
                "source": "client",
                "failure_code": "invalid",
                # Non-token VALUES (contain '@' or whitespace) are dropped
                # wholesale by the sanitizer's string branch.
                "email": "victim@example.com",
                "password": "correct horse battery staple",
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    metadata = body["data"]["event"]["metadata"]

    # Allowed metadata: the two keys the client emitter actually sets.
    assert metadata.get("source") == "client"
    assert metadata.get("failure_code") == "invalid"

    # Non-token VALUES are dropped wholesale by the sanitizer's string branch.
    assert "email" not in metadata
    assert "password" not in metadata


def test_portal_rum_contract_accepts_logout_submit_attempt(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-side logout_submit_attempt round-trip.

    Mirrors the login_submit_attempt contract: the front-door's POST
    /logout handler emits this event at handler entry to anchor the
    paired attempt/terminal latency calculation. ``route="/logout"``
    requires the new entry in PORTAL_ALLOWED_RUM_ROUTES; ``view="login"``
    matches the redirect destination, paralleling how the login event
    types report the user-facing shell.
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "logout_submit_attempt",
            "route": "/logout",
            "view": "login",
            "metric": "count",
            "value": 1,
            "unit": "count",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "logout_submit_attempt"
    assert event["route"] == "/logout"
    assert event["view"] == "login"
    assert event["metric"] == "count"
    assert event["unit"] == "count"
    assert event["metadata"] == {}


def test_portal_rum_contract_accepts_logout_submit_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "logout_submit_success",
            "route": "/logout",
            "view": "login",
            "metric": "duration",
            "value": 73.4,
            "unit": "ms",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "logout_submit_success"
    assert event["route"] == "/logout"
    assert event["view"] == "login"
    assert event["metric"] == "duration"
    assert event["unit"] == "ms"
    assert event["value"] == 73.4
    assert event["metadata"] == {}


def test_portal_rum_contract_accepts_logout_submit_failure_with_csrf_code(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The logout route currently has only one failure surface (CSRF).

    Both the Origin/Referrer mismatch path and the x-csrf-token mismatch
    path fold into ``failure_code: "csrf"``. If a future change widens
    the failure surface, both this test and ``ALLOWED_LOGOUT_FAILURE_CODES``
    in ``lib/rum-emitter.js`` must be extended together.
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "logout_submit_failure",
            "route": "/logout",
            "view": "login",
            "metric": "duration",
            "value": 41.0,
            "unit": "ms",
            "metadata": {"failure_code": "csrf"},
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["accepted"] is True
    event = body["data"]["event"]
    assert event["event_type"] == "logout_submit_failure"
    assert event["metric"] == "duration"
    assert event["metadata"] == {"failure_code": "csrf"}


def test_portal_rum_contract_rejects_logout_submit_attempt_with_unknown_route(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin exact-match routing on the new ``/logout`` allowlist entry.

    Paired with ``test_portal_rum_contract_accepts_logout_submit_attempt``
    above (which pins exactly ``/logout`` returns 200), this test pins
    that a sibling sub-path under ``/logout/...`` is rejected. Together
    the pair encodes the invariant: ``/logout`` is allowed iff the
    request route equals ``/logout`` exactly. If a refactor ever
    weakens the route check from a set-membership test (``route in
    PORTAL_ALLOWED_RUM_ROUTES``) to a prefix/glob match (e.g.
    ``any(route.startswith(p) for p in ...)``), ``/logout/anything-else``
    would suddenly start being accepted and this test would fail.
    """
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "logout_submit_attempt",
            "route": "/logout/anything-else",
            "view": "login",
            "metric": "count",
            "value": 1,
            "unit": "count",
        },
    )

    assert response.status_code == 400


def test_portal_rum_contract_rejects_login_submit_attempt_with_unknown_metric(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")

    response = client.post(
        "/v1/portal/rum",
        json={
            "event_type": "login_submit_attempt",
            "route": "/login",
            "view": "login",
            "metric": "duration",
            "value": 10,
            "unit": "ms",
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["details"] == {"field": "payload", "reason": "invalid_metric"}


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
        last_event_at=987.0,
        state="failed",
        progress=55,
        request={"pipeline": "lux-depth-v3"},
        logs_tail=["line-a", "line-b"],
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
            "indexed_count": 1,
            "truncated": False,
        },
        error={"code": "RUNNER_ERROR", "message": "boom", "details": {}},
    )
    _seed_job(job)

    list_response = client.get("/v1/jobs")
    list_body = list_response.json()
    assert list_response.status_code == 200
    assert list_body["schema"] == "tp.orchestrator.jobs.v1"
    first = list_body["data"]["jobs"][0]
    assert first["id"] == job.id
    assert first["events_url"] == f"/v1/jobs/{job.id}/events"
    assert first["error"]["code"] == "RUNNER_ERROR"
    assert first["last_event_at"] == 987.0
    assert first["artifacts"]["items"][0]["relative_path"] == "manifest.json"
    assert first["artifacts"]["items"][0]["display_hint"]["role"] == "manifest"

    detail_response = client.get(f"/v1/jobs/{job.id}")
    detail_body = detail_response.json()
    assert detail_response.status_code == 200
    assert detail_body["schema"] == "tp.orchestrator.job_status.v1"
    assert detail_body["data"]["events_url"] == f"/v1/jobs/{job.id}/events"
    assert detail_body["data"]["last_event_at"] == 987.0
    assert detail_body["data"]["artifacts"]["indexed_count"] == 1
    assert detail_body["data"]["artifacts"]["items"][0]["display_hint"]["label"] == "Manifest"
    assert detail_body["data"]["error"]["code"] == "RUNNER_ERROR"

    v2_list_response = client.get("/v2/jobs")
    v2_list_body = v2_list_response.json()
    assert v2_list_response.status_code == 200
    assert v2_list_body["schema"] == "tp.orchestrator.jobs.v1"
    assert v2_list_body["data"]["jobs"][0]["events_url"] == f"/v2/jobs/{job.id}/events"

    v2_detail_response = client.get(f"/v2/jobs/{job.id}")
    v2_detail_body = v2_detail_response.json()
    assert v2_detail_response.status_code == 200
    assert v2_detail_body["schema"] == "tp.orchestrator.job_status.v1"
    assert v2_detail_body["data"]["events_url"] == f"/v2/jobs/{job.id}/events"


def test_jobs_list_and_detail_are_repository_authoritative_after_runtime_cache_clear(client: TestClient) -> None:
    job = orchestrator_app.Job(
        id="job_repo_authority",
        created_at=orchestrator_app._now(),
        state="failed",
        progress=33,
        request={"pipeline": "lux-depth-v3"},
        error={"code": "RUNNER_ERROR", "message": "boom", "details": {}},
    )
    _seed_job(job)
    orchestrator_app.JOBS.clear()

    list_response = client.get("/v1/jobs")
    detail_response = client.get(f"/v1/jobs/{job.id}")

    assert list_response.status_code == 200
    assert list_response.json()["data"]["jobs"][0]["id"] == job.id
    assert detail_response.status_code == 200
    detail_body = detail_response.json()
    assert detail_body["data"]["id"] == job.id
    assert detail_body["data"]["error"]["code"] == "RUNNER_ERROR"


def test_cancel_reads_repository_state_after_runtime_cache_clear(client: TestClient) -> None:
    job = orchestrator_app.Job(
        id="job_repo_cancel",
        created_at=orchestrator_app._now(),
        state="queued",
        request={"pipeline": "lux-depth-v3"},
    )
    _seed_job(job)
    orchestrator_app.JOBS.clear()

    response = client.post(f"/v1/jobs/{job.id}/cancel")

    assert response.status_code == 200
    assert response.json()["data"] == {"id": job.id, "state": "queued"}
    record = asyncio.run(orchestrator_app._job_repository().get(job.id))
    assert record is not None
    assert record.cancel_requested is True


def test_detail_and_list_preserve_terminal_runtime_overlay_when_repository_lags(client: TestClient) -> None:
    job = orchestrator_app.Job(
        id="job_terminal_overlay_http",
        created_at=orchestrator_app._now(),
        started_at=orchestrator_app._now(),
        state="running",
        progress=70,
        request={"pipeline": "lux-depth-v3"},
    )
    _seed_job(job)
    cached_job = orchestrator_app.Job(
        id=job.id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=orchestrator_app._now(),
        done_published_at=orchestrator_app._now(),
        last_event_at=orchestrator_app._now(),
        state="succeeded",
        progress=100,
        exit_code=0,
        request=job.request,
        logs_tail=["finished locally"],
    )
    cached_job.proc = SimpleNamespace(returncode=0)
    orchestrator_app.JOBS[job.id] = cached_job

    detail_response = client.get(f"/v1/jobs/{job.id}")
    list_response = client.get("/v1/jobs")

    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["state"] == "succeeded"
    assert detail_response.json()["data"]["progress"] == 100
    assert list_response.status_code == 200
    listed = list_response.json()["data"]["jobs"][0]
    assert listed["id"] == job.id
    assert listed["state"] == "succeeded"
    assert listed["progress"] == 100


def test_job_routes_fail_closed_when_repository_is_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mark_da3_runtime_available: None,
) -> None:
    allowed_root = tmp_path.resolve()
    input_dir = allowed_root / "input"
    output_dir = allowed_root / "output"
    input_dir.mkdir(parents=True)
    (input_dir / "frame.jpg").write_bytes(b"fixture")
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [allowed_root])
    monkeypatch.setattr(orchestrator_app, "ALLOWED_OUTPUT_ROOTS", [allowed_root])
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = True

    checks = [
        client.get("/v1/jobs"),
        client.get("/v1/jobs/missing"),
        client.post("/v1/jobs/missing/cancel"),
        client.get("/v1/jobs/missing/artifacts/output.txt"),
        client.get("/v1/jobs/missing/events"),
        client.post(
            "/v1/jobs",
            json={
                "pipeline": "lux-depth-v3",
                "args": {
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "non_commercial_ok": True,
                },
            },
        ),
    ]

    for response in checks:
        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "JOB_REPOSITORY_UNAVAILABLE"
        assert body["error"]["message"] == "job repository unavailable"
        assert "cached repository construction failure" not in response.text


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


def _write_phase22_run_card(
    output_dir: Path,
    *,
    captioning_status: dict[str, Any] | None,
    artifact_index: list[dict[str, Any]] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_card_version": "v1",
        "batch_id": "2026-05-04_120000",
        "total_images": 1,
        "success_count": 1,
        "error_count": 0,
    }
    if artifact_index is not None:
        payload["artifact_index"] = artifact_index
    if captioning_status is not None:
        payload["captioning_status"] = captioning_status
    (output_dir / "run_card_2026-05-04_120000.json").write_text(json.dumps(payload), encoding="utf-8")


def _phase22_job(output_dir: Path, *, state: str = "succeeded") -> Any:
    return orchestrator_app.Job(
        id=f"job_phase22_{state}",
        created_at=orchestrator_app._now(),
        state=state,
        exit_code=0 if state == "succeeded" else None,
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )


def test_run_summary_maps_fastvlm_sidecar_evidence_to_succeeded(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_phase22_run_card(
        output_dir,
        captioning_status={
            "enabled": True,
            "backend": "fastvlm",
            "status": "ok",
            "model_role": "smoke",
            "model_id": "apple/FastVLM-0.5B-fp16",
            "role": "advisory",
            "sidecar_count": 0,
            "failed_count": 0,
            "used_for_quality_gate": False,
        },
        artifact_index=[
            {"relative_path": "captioning/image.vlm_captioning.sidecar.json"},
            {"relative_path": "captioning/image.vlm_captioning.raw.txt"},
        ],
    )

    job = _phase22_job(output_dir)
    job.artifacts = {
        "items": [
            {"relative_path": r"captioning\image_proxy.png"},
        ],
    }
    summary = orchestrator_app._refresh_job_run_summary(job)

    status = summary["captioning_status"]
    assert status["status"] == "succeeded"
    assert status["role"] == "advisory"
    assert status["sidecar_count"] == 1
    assert status["raw_count"] == 1
    assert status["proxy_count"] == 1
    assert status["used_for_quality_gate"] is False


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        (
            {"enabled": True, "backend": "fastvlm", "status": "missing_runtime", "used_for_quality_gate": False},
            "missing_runtime",
        ),
        (
            {"enabled": True, "backend": "fastvlm", "status": "invalid_config", "used_for_quality_gate": False},
            "invalid_config",
        ),
        ({"enabled": True, "backend": "other", "status": "ok", "used_for_quality_gate": False}, "unsupported_backend"),
        (
            {"enabled": True, "backend": "fastvlm", "status": "skipped", "sidecar_count": 0, "used_for_quality_gate": False},
            "skipped",
        ),
        (
            {"enabled": True, "backend": "fastvlm", "status": "error", "failed_count": 1, "used_for_quality_gate": False},
            "failed",
        ),
    ],
)
def test_run_summary_normalizes_fastvlm_terminal_statuses(
    tmp_path: Path,
    raw_status: dict[str, Any],
    expected: str,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_phase22_run_card(output_dir, captioning_status=raw_status)

    summary = orchestrator_app._refresh_job_run_summary(_phase22_job(output_dir))

    assert summary["captioning_status"]["status"] == expected
    assert summary["captioning_status"]["used_for_quality_gate"] is False


def test_run_summary_reports_fastvlm_quality_gate_policy_violation(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_phase22_run_card(
        output_dir,
        captioning_status={
            "enabled": True,
            "backend": "other",
            "status": "ok",
            "role": "advisory",
            "sidecar_count": 1,
            "failed_count": 0,
            "used_for_quality_gate": True,
        },
    )

    summary = orchestrator_app._refresh_job_run_summary(_phase22_job(output_dir))

    status = summary["captioning_status"]
    assert status["status"] == "failed"
    assert status["policy_violation"] is True
    assert status["quality_gate_claimed"] is True
    assert status["failed_count"] == 1
    assert status["used_for_quality_gate"] is False
    assert "used_for_quality_gate" in status["error"]


def test_fastvlm_model_role_normalization_uses_allowed_roles() -> None:
    for role in orchestrator_app.ALLOWED_VLM_CAPTIONING_MODEL_ROLES:
        assert orchestrator_app._fastvlm_model_role_from_value(role) == role
        assert orchestrator_app._fastvlm_model_role_from_value(role.upper()) == role

    assert orchestrator_app._fastvlm_model_role_from_value("unexpected") == "custom"


def test_active_fastvlm_job_serializes_requested_captioning_status(client: TestClient) -> None:
    job = orchestrator_app.Job(
        id="job_phase22_requested",
        created_at=orchestrator_app._now(),
        state="queued",
        request={"pipeline": "lux-depth-v3", "args": {}},
        effective_request={
            "pipeline": "lux-depth-v3",
            "args": {
                "vlm_captioning_enabled": True,
                "vlm_captioning_backend": "fastvlm",
                "vlm_captioning_model": "smoke",
            },
        },
    )
    _seed_job(job)

    list_body = client.get("/v1/jobs").json()
    detail_body = client.get(f"/v1/jobs/{job.id}").json()
    v2_list_body = client.get("/v2/jobs").json()
    v2_detail_body = client.get(f"/v2/jobs/{job.id}").json()

    for serialized in (
        list_body["data"]["jobs"][0],
        detail_body["data"],
        v2_list_body["data"]["jobs"][0],
        v2_detail_body["data"],
    ):
        status = serialized["run_summary"]["captioning_status"]
        assert status["status"] == "requested"
        assert status["enabled"] is True
        assert status["model_role"] == "smoke"
        assert status["used_for_quality_gate"] is False


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


def test_partial_run_summary_ignores_summaryless_run_card_payload(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_card_2026-04-07_001500.json").write_text(
        json.dumps(
            {
                "batch_id": "2026-04-07_001500",
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
        id="job_partial_review_summaryless_run_card",
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


def test_failed_run_summary_prefers_newest_all_error_run_card_when_output_dir_reused(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_card_2026-04-06_232022.json").write_text(
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
    (manifests_dir / "batch_2026-04-09_132300.json").write_text(
        json.dumps(
            {
                "batch_id": "2026-04-09_132300",
                "results": [{"status": "error"}] * 6,
                "stats": {"total_images": 6},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "run_card_2026-04-09_132300.json").write_text(
        json.dumps(
            {
                "batch_id": "2026-04-09_132300",
                "total_images": 6,
                "success_count": 0,
                "error_count": 6,
                "artifact_index": [
                    {
                        "artifact_type": "batch_manifest",
                        "path": "manifests/batch_2026-04-09_132300.json",
                        "relative_path": "manifests/batch_2026-04-09_132300.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "stale_preview.png").write_bytes(b"stale-preview")

    job = orchestrator_app.Job(
        id="job_failed_reused_output_dir",
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

    indexed = orchestrator_app._index_job_artifacts(job)
    summary = orchestrator_app._refresh_job_run_summary(job)

    assert {item["path"] for item in indexed} == {
        "manifests/batch_2026-04-09_132300.json",
        "run_card_2026-04-09_132300.json",
    }
    assert summary["batch_id"] == "2026-04-09_132300"
    assert summary["success_count"] == 0
    assert summary["error_count"] == 6
    assert summary["reviewable_outputs"] is False
    assert summary["partial"] is False
    assert job.state == "failed"
    assert job.error["code"] == "RUNNER_EXIT_NONZERO"


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
    _seed_job(job)

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


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
def test_job_artifact_endpoint_serves_indexed_binary_without_exposing_absolute_path(
    client: TestClient,
    tmp_path,
    jobs_base: str,
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
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)

    response = client.get(f"{jobs_base}/{job.id}/artifacts/renders/hero.png")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["content-type"].startswith("image/png")
    assert "attachment" not in response.headers.get("content-disposition", "").lower()
    assert response.content == b"\x89PNG\r\n\x1a\npreview"
    assert str(output_dir) not in response.text


def test_artifact_fetch_reads_repository_metadata_after_runtime_cache_clear(
    client: TestClient,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "repo-artifact-fetch"
    artifact_path = output_dir / "renders" / "hero.png"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"\x89PNG\r\n\x1a\nrepo")
    job = orchestrator_app.Job(
        id="job_repo_artifact_fetch",
        created_at=orchestrator_app._now(),
        state="succeeded",
        finished_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)
    orchestrator_app.JOBS.clear()

    response = client.get(f"/v1/jobs/{job.id}/artifacts/renders/hero.png")

    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n\x1a\nrepo"


def test_artifact_fetch_hydration_metadata_persist_is_best_effort(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "repo-artifact-hydrate"
    artifact_path = output_dir / "renders" / "hero.png"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"\x89PNG\r\n\x1a\nhydrate")
    job = orchestrator_app.Job(
        id="job_repo_artifact_hydrate",
        created_at=orchestrator_app._now(),
        state="succeeded",
        finished_at=orchestrator_app._now(),
        artifacts={"lifecycle": {"mirror_status": "mirrored", "artifact_store_backend": "local"}},
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    repo = orchestrator_app._job_repository()
    asyncio.run(repo.create(orchestrator_app._record_from_job(job)))

    async def set_artifacts(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("transient repository artifact write failure")

    monkeypatch.setattr(repo, "set_artifacts", set_artifacts)
    orchestrator_app.JOBS.clear()

    response = client.get(f"/v1/jobs/{job.id}/artifacts/renders/hero.png")

    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n\x1a\nhydrate"


def test_artifact_delete_persists_lifecycle_after_runtime_cache_clear(
    client: TestClient,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "repo-artifact-delete"
    artifact_path = output_dir / "result.txt"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("delete me", encoding="utf-8")
    job = orchestrator_app.Job(
        id="job_repo_artifact_delete",
        created_at=orchestrator_app._now(),
        state="succeeded",
        finished_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)
    orchestrator_app.JOBS.clear()

    delete_response = client.delete(f"/v1/jobs/{job.id}/artifacts")

    assert delete_response.status_code == 200
    record = asyncio.run(orchestrator_app._job_repository().get(job.id))
    assert record is not None
    assert record.artifacts["lifecycle"]["deletion_status"] == "deleted"
    orchestrator_app.JOBS.clear()
    fetch_response = client.get(f"/v1/jobs/{job.id}/artifacts/result.txt")
    assert fetch_response.status_code == 410
    assert fetch_response.json()["error"]["code"] == "ARTIFACT_DELETED"


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
    _seed_job(job)

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
    _seed_job(job)

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
    _seed_job(job)

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
    _seed_job(job)

    response = client.get(f"/v1/jobs/{job.id}/artifacts/missing.png")
    body = response.json()

    assert response.status_code == 404
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["details"]["path"] == "missing.png"


def test_job_artifact_endpoint_serves_local_store_after_source_file_is_gone(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "renders" / "result.bin"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"store-backed-bytes")

    store = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)

    job = orchestrator_app.Job(
        id="job_artifact_store_local",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        state="succeeded",
        finished_at=orchestrator_app._now(),
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)
    asyncio.run(orchestrator_app._mirror_job_artifacts_to_store(job))
    artifact_path.unlink()
    job.artifact_lookup = {}

    response = client.get(f"/v1/jobs/{job.id}/artifacts/renders/result.bin")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "attachment" in response.headers.get("content-disposition", "").lower()
    assert response.content == b"store-backed-bytes"


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
@pytest.mark.parametrize(
    ("artifact_relative_path", "body", "content_type", "expect_disposition"),
    [
        ("outputs/payload.txt", b"s3-bytes", "text/plain", True),
        ("outputs/preview.png", b"\x89PNG\r\n\x1a\npreview", "image/png", False),
    ],
)
def test_job_artifact_endpoint_redirects_to_s3_presigned_url_after_auth(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs_base: str,
    artifact_relative_path: str,
    body: bytes,
    content_type: str,
    expect_disposition: bool,
) -> None:
    try:
        import boto3
        from moto import mock_aws
    except ImportError:
        pytest.skip("moto not available; install CI/dev requirements to exercise the S3 app route")

    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

    previous_env = {key: os.environ.get(key) for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")}
    os.environ.update(
        AWS_ACCESS_KEY_ID="test",
        AWS_SECRET_ACCESS_KEY="test",
        AWS_DEFAULT_REGION="us-east-1",
    )
    try:
        with mock_aws():
            bucket = f"tp-route-{orchestrator_app.uuid.uuid4().hex[:8]}"
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
            store = S3ArtifactStore(bucket=bucket, prefix="phase4b", region_name="us-east-1")
            monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)

            job = orchestrator_app.Job(
                id="job_artifact_store_s3",
                created_at=orchestrator_app._now(),
                request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(tmp_path / "unused")}},
                state="succeeded",
                finished_at=orchestrator_app._now(),
                artifacts={
                    "items": [{"path": artifact_relative_path, "relative_path": artifact_relative_path}],
                    "indexed_count": 1,
                    "truncated": False,
                },
            )
            _seed_job(job)
            asyncio.run(store.write_bytes(job.id, artifact_relative_path, body, content_type=content_type))

            response = client.get(
                f"{jobs_base}/{job.id}/artifacts/{artifact_relative_path}",
                follow_redirects=False,
            )

            assert response.status_code == 307
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["location"].startswith("https://")
            assert response.content == b""
            presign_query = parse_qs(urlparse(response.headers["location"]).query)
            assert presign_query["response-content-type"] == [content_type]
            assert presign_query["response-cache-control"] == ["no-store"]
            if expect_disposition:
                assert presign_query["response-content-disposition"][0].startswith("attachment; filename=")
            else:
                assert "response-content-disposition" not in presign_query

            status_response = client.get(f"{jobs_base}/{job.id}")
            assert response.headers["location"] not in status_response.text
            assert response.headers["location"] not in json.dumps(job.artifacts)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
def test_job_artifact_endpoint_mirrors_legacy_file_to_s3_on_first_access(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs_base: str,
) -> None:
    try:
        import boto3
        from moto import mock_aws
    except ImportError:
        pytest.skip("moto not available; install CI/dev requirements to exercise the S3 app route")

    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

    previous_env = {key: os.environ.get(key) for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")}
    os.environ.update(
        AWS_ACCESS_KEY_ID="test",
        AWS_SECRET_ACCESS_KEY="test",
        AWS_DEFAULT_REGION="us-east-1",
    )
    try:
        with mock_aws():
            bucket = f"tp-mirror-{orchestrator_app.uuid.uuid4().hex[:8]}"
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
            store = S3ArtifactStore(bucket=bucket, prefix="phase4b", region_name="us-east-1")
            monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)

            output_dir = tmp_path / "legacy-output"
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = output_dir / "outputs" / "payload.txt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"legacy-to-s3")
            job = orchestrator_app.Job(
                id="job_artifact_legacy_s3_mirror",
                created_at=orchestrator_app._now(),
                request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
                state="succeeded",
                finished_at=orchestrator_app._now(),
            )
            orchestrator_app._index_job_artifacts(job)
            _seed_job(job)

            response = client.get(
                f"{jobs_base}/{job.id}/artifacts/outputs/payload.txt",
                follow_redirects=False,
            )

            assert response.status_code == 307
            assert job.artifact_store_mirrored is True
            stream = asyncio.run(store.open_bytes(job.id, "outputs/payload.txt"))
            chunks: list[bytes] = []

            async def _collect() -> None:
                async for chunk in stream:
                    chunks.append(chunk)

            asyncio.run(_collect())
            assert b"".join(chunks) == b"legacy-to-s3"
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_job_artifact_endpoint_returns_503_when_s3_first_access_mirror_fails(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingMirrorS3Store:
        backend = "s3"

        async def write_file(
            self, job_id: str, relative_path: str, source_path: Path, *, content_type: str | None = None
        ):  # noqa: ARG002
            raise orchestrator_app.ArtifactStoreError("mirror write failed with private path")

        async def head(self, job_id: str, relative_path: str):  # noqa: ARG002
            raise orchestrator_app.StoreArtifactNotFoundError("missing after failed mirror")

        async def presign_get(self, job_id: str, relative_path: str, *, expires_seconds: int, **_kwargs):  # noqa: ARG002
            raise AssertionError("presign must not run when requested artifact failed to mirror")

    output_dir = tmp_path / "legacy-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "outputs" / "payload.txt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("legacy source", encoding="utf-8")
    job = orchestrator_app.Job(
        id="job_artifact_s3_mirror_failed",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        state="succeeded",
        finished_at=orchestrator_app._now(),
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)
    monkeypatch.setattr(orchestrator_app, "_artifact_store", FailingMirrorS3Store)

    response = client.get(f"/v1/jobs/{job.id}/artifacts/outputs/payload.txt")

    assert response.status_code == 503
    assert response.json()["error"]["details"] == {
        "job_id": job.id,
        "reason": "artifact_store_mirror_failed",
    }


def test_job_artifact_endpoint_returns_503_when_s3_failed_path_sample_is_truncated(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingMirrorS3Store:
        backend = "s3"

        async def write_file(
            self, job_id: str, relative_path: str, source_path: Path, *, content_type: str | None = None
        ):  # noqa: ARG002
            raise orchestrator_app.ArtifactStoreError("mirror write failed")

        async def head(self, job_id: str, relative_path: str):  # noqa: ARG002
            raise orchestrator_app.StoreArtifactNotFoundError("missing after failed mirror")

        async def presign_get(self, job_id: str, relative_path: str, *, expires_seconds: int, **_kwargs):  # noqa: ARG002
            raise AssertionError("presign must not run when mirror failures are unsampled")

    output_dir = tmp_path / "legacy-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_lookup: dict[str, Path] = {}
    items: list[dict[str, str]] = []
    for index in range(11):
        relative_path = f"outputs/payload-{index:02d}.txt"
        artifact_path = output_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(f"legacy source {index}", encoding="utf-8")
        artifact_lookup[relative_path] = artifact_path
        items.append({"path": relative_path, "relative_path": relative_path})

    job = orchestrator_app.Job(
        id="job_artifact_s3_mirror_truncated_failures",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        state="succeeded",
        finished_at=orchestrator_app._now(),
        artifacts={"items": items, "indexed_count": len(items), "truncated": False},
        artifact_lookup=artifact_lookup,
    )
    _seed_job(job)
    monkeypatch.setattr(orchestrator_app, "_artifact_store", FailingMirrorS3Store)

    unsampled_path = "outputs/payload-10.txt"
    response = client.get(f"/v1/jobs/{job.id}/artifacts/{unsampled_path}")

    lifecycle = job.artifacts[orchestrator_app._ARTIFACT_LIFECYCLE_KEY]
    assert response.status_code == 503
    assert response.json()["error"]["details"] == {
        "job_id": job.id,
        "reason": "artifact_store_mirror_failed",
    }
    assert lifecycle["mirror_failed_paths_complete"] is False
    assert unsampled_path not in lifecycle["mirror_failed_paths"]


def test_job_artifact_endpoint_auth_failure_does_not_presign(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = orchestrator_app.Job(
        id="job_artifact_no_presign",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(tmp_path / "out")}},
        state="succeeded",
        artifacts={
            "items": [{"path": "outputs/payload.txt", "relative_path": "outputs/payload.txt"}],
            "indexed_count": 1,
            "truncated": False,
        },
    )
    _seed_job(job)

    def _fail_store_access():
        raise AssertionError("unauthorized artifact route must not touch artifact store")

    monkeypatch.setattr(orchestrator_app, "_artifact_store", _fail_store_access)

    response = client.get(
        f"/v1/jobs/{job.id}/artifacts/outputs/payload.txt",
        headers={"x-api-key": "wrong"},
        follow_redirects=False,
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_job_artifact_endpoint_redacts_store_setup_exception(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_message = "Traceback leaked /tmp/internal-store-secret"
    job = orchestrator_app.Job(
        id="job_artifact_store_unavailable",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(tmp_path / "out")}},
        state="succeeded",
        artifacts={
            "items": [{"path": "outputs/payload.txt", "relative_path": "outputs/payload.txt"}],
            "indexed_count": 1,
            "truncated": False,
        },
    )
    _seed_job(job)

    def _fail_store_access():
        raise RuntimeError(secret_message)

    monkeypatch.setattr(orchestrator_app, "_artifact_store", _fail_store_access)

    response = client.get(f"/v1/jobs/{job.id}/artifacts/outputs/payload.txt")

    assert response.status_code == 503
    assert secret_message not in response.text
    assert response.json()["error"]["details"] == {
        "job_id": job.id,
        "reason": "artifact_store_unavailable",
    }
    assert job.artifacts["lifecycle"]["mirror_error"] == "artifact_store_unavailable"


def test_job_artifact_endpoint_redacts_s3_operation_exception(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_message = "Traceback leaked from boto credentials"

    class FailingS3Store:
        backend = "s3"

        async def head(self, job_id: str, relative_path: str):  # noqa: ARG002
            raise orchestrator_app.ArtifactStoreError(secret_message)

        async def presign_get(self, job_id: str, relative_path: str, *, expires_seconds: int, **_kwargs):  # noqa: ARG002
            raise AssertionError("presign must not run after failed head")

    job = orchestrator_app.Job(
        id="job_artifact_s3_unavailable",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(tmp_path / "out")}},
        state="succeeded",
        artifact_store_mirrored=True,
        artifact_store_backend="s3",
        artifacts={
            "items": [{"path": "outputs/payload.txt", "relative_path": "outputs/payload.txt"}],
            "indexed_count": 1,
            "truncated": False,
        },
    )
    _seed_job(job)
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: FailingS3Store())

    response = client.get(f"/v1/jobs/{job.id}/artifacts/outputs/payload.txt")

    assert response.status_code == 503
    assert secret_message not in response.text
    assert response.json()["error"]["details"] == {
        "job_id": job.id,
        "reason": "artifact_store_operation_failed",
    }


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
def test_delete_job_artifacts_marks_lifecycle_and_returns_gone_on_fetch(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs_base: str,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "result.txt"
    artifact_path.write_text("delete me", encoding="utf-8")
    store = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)

    job = orchestrator_app.Job(
        id="job_artifact_delete",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        state="succeeded",
        finished_at=orchestrator_app._now(),
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)
    asyncio.run(orchestrator_app._mirror_job_artifacts_to_store(job))

    delete_response = client.delete(f"{jobs_base}/{job.id}/artifacts")

    assert delete_response.status_code == 200
    lifecycle = delete_response.json()["data"]["artifacts"]["lifecycle"]
    assert lifecycle["deletion_status"] == "deleted"
    assert lifecycle["deleted_count"] == 1

    fetch_response = client.get(f"{jobs_base}/{job.id}/artifacts/result.txt")
    assert fetch_response.status_code == 410
    assert fetch_response.json()["error"]["code"] == "ARTIFACT_DELETED"


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
def test_delete_job_artifacts_removes_legacy_files_when_store_is_empty(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs_base: str,
) -> None:
    output_dir = tmp_path / "legacy-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "legacy.txt"
    artifact_path.write_text("legacy artifact", encoding="utf-8")
    store = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)

    job = orchestrator_app.Job(
        id="job_artifact_delete_legacy",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        state="succeeded",
        finished_at=orchestrator_app._now(),
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)

    delete_response = client.delete(f"{jobs_base}/{job.id}/artifacts")

    assert delete_response.status_code == 200
    assert not artifact_path.exists()
    lifecycle = delete_response.json()["data"]["artifacts"]["lifecycle"]
    assert lifecycle["deletion_status"] == "deleted"
    assert lifecycle["deleted_count"] == 1
    assert lifecycle["store_deleted_count"] == 0
    assert lifecycle["legacy_deleted_count"] == 1

    fetch_response = client.get(f"{jobs_base}/{job.id}/artifacts/legacy.txt")
    assert fetch_response.status_code == 410
    assert fetch_response.json()["error"]["code"] == "ARTIFACT_DELETED"


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
def test_delete_job_artifacts_reports_partial_mirror_unique_count(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs_base: str,
) -> None:
    output_dir = tmp_path / "partial-mirror-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = output_dir / "mirrored.txt"
    legacy_only_path = output_dir / "legacy-only.txt"
    mirrored_path.write_text("mirrored artifact", encoding="utf-8")
    legacy_only_path.write_text("legacy-only artifact", encoding="utf-8")
    store = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)

    job = orchestrator_app.Job(
        id="job_artifact_delete_partial_mirror",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        state="succeeded",
        finished_at=orchestrator_app._now(),
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)
    asyncio.run(store.write_file(job.id, "mirrored.txt", mirrored_path))

    delete_response = client.delete(f"{jobs_base}/{job.id}/artifacts")

    assert delete_response.status_code == 200
    assert not mirrored_path.exists()
    assert not legacy_only_path.exists()
    lifecycle = delete_response.json()["data"]["artifacts"]["lifecycle"]
    assert lifecycle["deletion_status"] == "deleted"
    assert lifecycle["deleted_count"] == 2
    assert lifecycle["store_deleted_count"] == 1
    assert lifecycle["legacy_deleted_count"] == 2


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
def test_delete_job_artifacts_rejects_active_jobs(client: TestClient, tmp_path: Path, jobs_base: str) -> None:
    job = orchestrator_app.Job(
        id="job_artifact_delete_active",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(tmp_path / "out")}},
        state="running",
    )
    _seed_job(job)

    response = client.delete(f"{jobs_base}/{job.id}/artifacts")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONFLICT"


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
def test_delete_job_artifacts_requires_api_key_before_store_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs_base: str,
) -> None:
    job = orchestrator_app.Job(
        id="job_artifact_delete_auth",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(tmp_path / "out")}},
        state="succeeded",
        finished_at=orchestrator_app._now(),
        artifacts={
            "items": [{"path": "outputs/payload.txt", "relative_path": "outputs/payload.txt"}],
            "indexed_count": 1,
            "truncated": False,
        },
    )
    _seed_job(job)

    def _fail_store_access():
        raise AssertionError("unauthorized artifact delete must not touch artifact store")

    monkeypatch.setattr(orchestrator_app, "_artifact_store", _fail_store_access)

    with TestClient(orchestrator_app.app) as anonymous_client:
        missing_key = anonymous_client.delete(f"{jobs_base}/{job.id}/artifacts")
    assert missing_key.status_code == 401
    assert missing_key.json()["error"]["code"] == "UNAUTHORIZED"

    with TestClient(orchestrator_app.app) as wrong_key_client:
        wrong_key = wrong_key_client.delete(f"{jobs_base}/{job.id}/artifacts", headers={"x-api-key": "wrong"})
    assert wrong_key.status_code == 401
    assert wrong_key.json()["error"]["code"] == "UNAUTHORIZED"


def test_delete_job_artifacts_routes_use_job_status_response_model() -> None:
    route_models = {}
    for route in orchestrator_app.app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set()) or set()
        if path in {"/v1/jobs/{job_id}/artifacts", "/v2/jobs/{job_id}/artifacts"} and "DELETE" in methods:
            route_models[path] = getattr(route, "response_model", None)

    assert route_models == {
        "/v1/jobs/{job_id}/artifacts": orchestrator_app.JobStatusEnvelope,
        "/v2/jobs/{job_id}/artifacts": orchestrator_app.JobStatusEnvelope,
    }


def test_artifact_retention_cleanup_skips_active_jobs_and_deletes_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)
    now = orchestrator_app._now()

    terminal_output = tmp_path / "terminal"
    terminal_output.mkdir()
    (terminal_output / "result.txt").write_text("terminal", encoding="utf-8")
    terminal_job = orchestrator_app.Job(
        id="job_artifact_retention_terminal",
        created_at=now,
        finished_at=now,
        state="succeeded",
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(terminal_output)}},
    )
    orchestrator_app._index_job_artifacts(terminal_job)
    _seed_job(terminal_job)
    asyncio.run(orchestrator_app._mirror_job_artifacts_to_store(terminal_job))
    terminal_job.artifacts["lifecycle"]["expires_at"] = now - 1
    _sync_seeded_job(terminal_job)

    active_output = tmp_path / "active"
    active_output.mkdir()
    (active_output / "result.txt").write_text("active", encoding="utf-8")
    active_job = orchestrator_app.Job(
        id="job_artifact_retention_active",
        created_at=now,
        state="running",
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(active_output)}},
    )
    orchestrator_app._index_job_artifacts(active_job)
    _seed_job(active_job)
    asyncio.run(orchestrator_app._mirror_job_artifacts_to_store(active_job))
    active_job.artifacts["lifecycle"]["expires_at"] = now - 1
    _sync_seeded_job(active_job)

    asyncio.run(orchestrator_app._cleanup_expired_job_artifacts(now))

    assert terminal_job.artifacts["lifecycle"]["deletion_status"] == "deleted"
    assert "deleted_at" in terminal_job.artifacts["lifecycle"]
    assert active_job.artifacts["lifecycle"]["deletion_status"] == "available"
    assert asyncio.run(store.delete(active_job.id)) == 1


def test_artifact_retention_cleanup_keeps_job_when_delete_fails_then_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingDeleteStore:
        backend = "local"

        async def delete(self, job_id: str, relative_path: str | None = None):  # noqa: ARG002
            raise orchestrator_app.ArtifactStoreError("delete failed")

    now = orchestrator_app._now()
    output_dir = tmp_path / "retry-delete"
    output_dir.mkdir()
    artifact_path = output_dir / "result.txt"
    artifact_path.write_text("retry me", encoding="utf-8")
    job = orchestrator_app.Job(
        id="job_artifact_retention_retry",
        created_at=now - orchestrator_app.JOB_RETENTION_SECONDS - 10,
        finished_at=now - orchestrator_app.JOB_RETENTION_SECONDS - 10,
        state="succeeded",
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)
    orchestrator_app._ensure_artifact_lifecycle(job, now=now - orchestrator_app.JOB_RETENTION_SECONDS - 10)
    job.artifacts["lifecycle"]["expires_at"] = now - 1
    _sync_seeded_job(job)
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: FailingDeleteStore())

    asyncio.run(orchestrator_app._cleanup_expired_job_artifacts(now))
    asyncio.run(orchestrator_app._cleanup_expired_jobs(now))

    assert job.id in orchestrator_app.JOBS
    assert job.artifacts["lifecycle"]["deletion_status"] == "failed"
    assert job.artifacts["lifecycle"]["deletion_error"] == "artifact_deletion_failed"
    assert artifact_path.exists()
    assert asyncio.run(orchestrator_app._job_repository().get(job.id)) is not None

    store = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)
    asyncio.run(orchestrator_app._cleanup_expired_job_artifacts(now))
    asyncio.run(orchestrator_app._cleanup_expired_jobs(now))

    assert job.artifacts["lifecycle"]["deletion_status"] == "deleted"
    assert not artifact_path.exists()
    assert job.id not in orchestrator_app.JOBS
    assert asyncio.run(orchestrator_app._job_repository().get(job.id)) is None


def test_ready_reports_artifact_store_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = LocalArtifactStore(root_dir=tmp_path / "artifact-store")
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)

    response = asyncio.run(orchestrator_app.ready())

    assert response["ok"] is True
    assert response["artifact_store"] == {
        "backend": "local",
        "configured": True,
        "prefix": "",
        "signed_urls": False,
    }


def test_ready_redacts_artifact_store_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_message = "Traceback leaked /tmp/internal-bucket-secret"

    def _fail_store_access():
        raise RuntimeError(secret_message)

    monkeypatch.setattr(orchestrator_app, "_artifact_store", _fail_store_access)

    response = asyncio.run(orchestrator_app.ready())

    assert response["ok"] is False
    assert response["artifact_store"]["configured"] is False
    assert response["artifact_store"]["error"] == "artifact_store_unavailable"
    assert secret_message not in json.dumps(response)


def test_ready_fails_closed_when_s3_head_bucket_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

    secret_message = "Traceback leaked bucket-name-and-credentials"

    class FailingHeadBucketClient:
        def head_bucket(self, **_kwargs):
            raise RuntimeError(secret_message)

    store = S3ArtifactStore(bucket="private-bucket", prefix="phase4b", client=FailingHeadBucketClient())
    monkeypatch.setenv("TP_ARTIFACT_STORE", "s3")
    monkeypatch.setattr(orchestrator_app, "_artifact_store", lambda: store)

    response = asyncio.run(orchestrator_app.ready())

    assert response["ok"] is False
    assert response["artifact_store"]["backend"] == "s3"
    assert response["artifact_store"]["configured"] is False
    assert response["artifact_store"]["error"] == "artifact_store_unavailable"
    assert secret_message not in json.dumps(response)


@pytest.mark.parametrize("jobs_base", ["/v1/jobs", "/v2/jobs"])
def test_versioned_job_routes_enforce_api_key_for_reads_and_events(client: TestClient, jobs_base: str) -> None:
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
    _seed_job(finished_job)
    orchestrator_app.EVENT_SUBSCRIBERS[finished_job.id] = {}

    list_unauthorized = client.get(jobs_base, headers={"x-api-key": "wrong"})
    assert list_unauthorized.status_code == 401
    assert list_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    list_authorized = client.get(jobs_base, headers={"x-api-key": "contract-secret"})
    assert list_authorized.status_code == 200
    assert list_authorized.json()["success"] is True

    events_unauthorized = client.get(f"{jobs_base}/{finished_job.id}/events", headers={"x-api-key": "wrong"})
    assert events_unauthorized.status_code == 401
    assert events_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

    events_authorized = client.get(f"{jobs_base}/{finished_job.id}/events", headers={"x-api-key": "contract-secret"})
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

    rum_unauthorized = client.post(
        "/v1/portal/rum",
        headers={"x-api-key": "wrong"},
        json={
            "event_type": "queue_request",
            "route": "/portal",
            "view": "build",
            "metric": "submit",
            "value": 183.42,
            "unit": "ms",
        },
    )
    assert rum_unauthorized.status_code == 401
    assert rum_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"


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

    rum_unauthorized = client.post(
        "/v1/portal/rum/",
        headers={"x-api-key": "wrong"},
        follow_redirects=False,
        json={
            "event_type": "queue_request",
            "route": "/portal",
            "view": "build",
            "metric": "submit",
            "value": 183.42,
            "unit": "ms",
        },
    )
    assert rum_unauthorized.status_code == 401
    assert rum_unauthorized.json()["error"]["code"] == "UNAUTHORIZED"


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


def test_malformed_job_args_return_typed_invalid_argument(client: TestClient) -> None:
    response = client.post("/v1/jobs", json={"pipeline": "lux-depth-v3", "args": "not-a-dict"})
    body = response.json()
    assert response.status_code == 400
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"] == {"field": "input_dir", "reason": "required"}


def test_job_create_request_adapter_rejects_non_dict_args() -> None:
    response = orchestrator_app._validated_job_create_request({"pipeline": "lux-depth-v3", "args": "not-a-dict"})
    assert isinstance(response, orchestrator_app.JSONResponse)
    body = json.loads(response.body.decode("utf-8"))
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"] == {"field": "args", "reason": "invalid_request"}


def test_job_create_request_adapter_keeps_required_reason_vocabulary() -> None:
    response = orchestrator_app._validated_job_create_request({"args": {}})
    assert isinstance(response, orchestrator_app.JSONResponse)
    body = json.loads(response.body.decode("utf-8"))
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"] == {"field": "pipeline", "reason": "required"}


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
    (archive_root / "asset-001.dng").write_bytes(b"raw")
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])

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


def test_archive_gate_pipeline_submission_rejects_archive_index_root_mismatch(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_run_job(job, _argv):  # noqa: ANN001
        raise AssertionError("mismatched archive index must be rejected before dispatch")

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    archive_root = tmp_path / "raw_root"
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "DJI_0018.DNG").write_bytes(b"raw")
    archive_index = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive_small" / "archive_index_normalized.csv.gz"
    )

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

    assert response.status_code == 400
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"] == {
        "field": "archive_index",
        "reason": "archive_index_root_mismatch",
    }


def test_archive_gate_pipeline_submission_reports_missing_archive_root_on_root_field(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_run_job(job, _argv):  # noqa: ANN001
        raise AssertionError("missing archive root must be rejected before dispatch")

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])
    archive_root = tmp_path / "missing-root"

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(tmp_path),
                "output_dir": str(tmp_path / "out"),
                "archive_command": "fixity-scan",
                "archive_index": str(archive_index),
                "archive_root": str(archive_root),
            },
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["field"] == "archive_root"
    assert body["error"]["details"]["reason"] in {"missing", "not_a_directory"}


def test_create_job_preserves_raw_request_and_internal_execution_args(
    client: TestClient,
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


def test_create_job_rejects_fastvlm_captioning_when_feature_disabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    mark_da3_runtime_available: None,
) -> None:
    monkeypatch.delenv("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", raising=False)
    monkeypatch.delenv("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", raising=False)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/http_captioning_disabled",
                "vlm_captioning_enabled": True,
            },
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["error"]["details"] == {
        "field": "vlm_captioning_enabled",
        "reason": "captioning_feature_disabled",
    }


def test_create_job_normalizes_fastvlm_captioning_and_emits_backend_argv(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    mark_da3_runtime_available: None,
) -> None:
    captured_argv: list[str] = []
    original_argv_from_request = orchestrator_app._argv_from_request

    async def fake_run_job(job, argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    def capture_argv(payload, *, execution_args=None):  # noqa: ANN001
        argv = original_argv_from_request(payload, execution_args=execution_args)
        captured_argv[:] = list(argv)
        return argv

    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", "1")
    monkeypatch.setenv("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "captioning-contract")
    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    monkeypatch.setattr(orchestrator_app, "_argv_from_request", capture_argv)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "/tests/fixtures/archive_small/archive_root",
                "output_dir": "/tests/fixtures/portal_contract_output/http_captioning_enabled",
                "vlmCaptioningEnabled": True,
                "vlmCaptioningModel": "smoke",
                "vlmCaptioningProxyFormat": "png",
                "vlmCaptioningMaxSidePx": 960,
                "fastvlmTimeoutSeconds": 45,
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    job = orchestrator_app.JOBS[body["data"]["id"]]
    assert job.request["args"]["vlmCaptioningEnabled"] is True
    assert "vlmCaptioningEnabled" not in job.effective_request["args"]
    assert job.effective_request["args"]["vlm_captioning_enabled"] is True
    assert job.effective_request["args"]["vlm_captioning_model"] == "smoke"
    assert job.effective_request["args"]["vlm_captioning_proxy_format"] == "png"
    assert job.effective_request["args"]["vlm_captioning_max_side_px"] == 960
    assert job.effective_request["args"]["fastvlm_timeout_seconds"] == 45
    assert "--vlm-captioning" in captured_argv
    assert _flag_value(captured_argv, "--vlm-captioning") == "on"
    assert _flag_value(captured_argv, "--vlm-captioning-backend") == "fastvlm"
    assert _flag_value(captured_argv, "--vlm-captioning-model") == "smoke"
    assert _flag_value(captured_argv, "--vlm-captioning-proxy-format") == "png"
    assert _flag_value(captured_argv, "--vlm-captioning-max-side-px") == "960"
    assert _flag_value(captured_argv, "--fastvlm-timeout-seconds") == "45"


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


# --- Archive Gate E2E HTTP Contract Test Extensions (Phase 3) ---


def test_archive_gate_a_fixity_verify_submission_returns_job_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate A fixity-verify command submits successfully with required hash_manifest."""

    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    hash_manifest = tmp_path / "hash_manifest.csv.gz"
    hash_manifest.write_bytes(b"fixture-manifest")
    (tmp_path / "archive_root").mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(tmp_path / "archive_root"),
                "output_dir": str(tmp_path / "out"),
                "archive_command": "fixity-verify",
                "hash_manifest": str(hash_manifest),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["data"]["id"].startswith("job_")


def test_archive_gate_b_bag_validate_submission_returns_job_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate B bag-validate command submits successfully with required bag_dir."""

    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "archive_root").mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-b",
            "args": {
                "input_dir": str(tmp_path / "archive_root"),
                "output_dir": str(tmp_path / "out"),
                "archive_command": "bag-validate",
                "bag_dir": str(bag_dir),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["data"]["id"].startswith("job_")


def test_archive_gate_b_dedup_plan_submission_returns_job_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate B dedup-plan command submits successfully with required manifest_jsonl."""

    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")
    (tmp_path / "archive_root").mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-b",
            "args": {
                "input_dir": str(tmp_path / "archive_root"),
                "output_dir": str(tmp_path / "out"),
                "archive_command": "dedup-plan",
                "manifest_jsonl": str(manifest_jsonl),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["data"]["id"].startswith("job_")


def test_archive_gate_c_prov_export_submission_returns_job_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate C prov-export command submits successfully with required manifest_jsonl."""

    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")
    (tmp_path / "archive_root").mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-c",
            "args": {
                "input_dir": str(tmp_path / "archive_root"),
                "output_dir": str(tmp_path / "out"),
                "archive_command": "prov-export",
                "manifest_jsonl": str(manifest_jsonl),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["data"]["id"].startswith("job_")


def test_archive_gate_c_stac_export_submission_returns_job_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate C stac-export command submits successfully with required manifest_jsonl."""

    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")
    (tmp_path / "archive_root").mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-c",
            "args": {
                "input_dir": str(tmp_path / "archive_root"),
                "output_dir": str(tmp_path / "out"),
                "archive_command": "stac-export",
                "manifest_jsonl": str(manifest_jsonl),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["data"]["id"].startswith("job_")


def test_archive_gate_c_mets_export_submission_returns_job_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate C mets-export command submits successfully with required manifest_jsonl."""

    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text('{"id":"asset-1"}\n', encoding="utf-8")
    (tmp_path / "archive_root").mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-c",
            "args": {
                "input_dir": str(tmp_path / "archive_root"),
                "output_dir": str(tmp_path / "out"),
                "archive_command": "mets-export",
                "manifest_jsonl": str(manifest_jsonl),
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["data"]["id"].startswith("job_")


def test_archive_gate_b_bag_validate_rejects_missing_bag_dir(client: TestClient) -> None:
    """Gate B bag-validate fails with typed error when bag_dir is missing."""
    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-b",
            "args": {
                "input_dir": "./archive_root",
                "output_dir": "./out",
                "archive_command": "bag-validate",
            },
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["field"] == "bag_dir"


def test_archive_gate_c_prov_export_rejects_missing_manifest(client: TestClient) -> None:
    """Gate C prov-export fails with typed error when manifest_jsonl is missing."""
    response = client.post(
        "/v1/jobs",
        json={
            "pipeline": "archive-gate-c",
            "args": {
                "input_dir": "./archive_root",
                "output_dir": "./out",
                "archive_command": "prov-export",
            },
        },
    )
    body = response.json()

    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["field"] == "manifest_jsonl"


def test_v1_jobs_rejects_when_max_concurrent_jobs_reached(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    mark_da3_runtime_available: None,
) -> None:
    # Use per-test isolated directories under tmp_path so the test stays
    # parallel-safe under pytest-xdist (no shared repo-root paths).
    previous_limit = orchestrator_app.MAX_CONCURRENT_JOBS
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    allowed_root = (tmp_path / "rate-limit-isolated").resolve()
    allowed_root.mkdir(parents=True, exist_ok=True)
    input_dir = allowed_root / "input"
    output_dir = allowed_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        orchestrator_app.ALLOWED_INPUT_ROOTS = [allowed_root]
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = [allowed_root]
        orchestrator_app.MAX_CONCURRENT_JOBS = 1
        _seed_job(
            orchestrator_app.Job(
                id="job_busy",
                created_at=orchestrator_app._now(),
                state="running",
                request={
                    "pipeline": "lux-depth-v3",
                    "args": {"input_dir": str(input_dir), "output_dir": str(output_dir)},
                },
            )
        )
        response = client.post(
            "/v1/jobs",
            json={
                "pipeline": "lux-depth-v3",
                "args": {"input_dir": str(input_dir), "output_dir": str(output_dir)},
            },
        )
        body = response.json()
    finally:
        orchestrator_app.MAX_CONCURRENT_JOBS = previous_limit
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
        orchestrator_app.JOBS.clear()

    assert response.status_code == 429
    assert body["error"]["code"] == "RATE_LIMITED"
    assert body["error"]["details"]["active_jobs"] == 1
    assert body["error"]["details"]["max_concurrent_jobs"] == 1

    # The job-admission 429 path emits the same rate-limit header contract as
    # the per-IP gate, with values describing the concurrency cap rather than
    # the per-minute window. We assert parse + range, not exact values.
    import time as _time

    retry_after_raw = response.headers.get("Retry-After")
    assert retry_after_raw is not None, "Retry-After header missing on job-admission 429"
    assert int(retry_after_raw) >= 1

    limit_raw = response.headers.get("X-RateLimit-Limit")
    assert limit_raw is not None, "X-RateLimit-Limit header missing on job-admission 429"
    assert int(limit_raw) == 1  # MAX_CONCURRENT_JOBS for this test

    remaining_raw = response.headers.get("X-RateLimit-Remaining")
    assert remaining_raw is not None, "X-RateLimit-Remaining header missing on job-admission 429"
    assert int(remaining_raw) == 0

    reset_raw = response.headers.get("X-RateLimit-Reset")
    assert reset_raw is not None, "X-RateLimit-Reset header missing on job-admission 429"
    assert int(reset_raw) >= int(_time.time()) - 1  # tolerate <1s clock skew


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


def test_job_events_stream_emits_state_log_progress_artifact_done(
    client: TestClient,
    monkeypatch,
    tmp_path,
    mark_da3_runtime_available: None,
) -> None:
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

    # Use per-test isolated paths so the SSE stream test stays parallel-safe
    # under pytest-xdist (no shared repo-root directories).
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    allowed_root = (tmp_path / "sse-stream-isolated").resolve()
    allowed_root.mkdir(parents=True, exist_ok=True)
    input_dir = allowed_root / "input"
    output_dir = allowed_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        orchestrator_app.ALLOWED_INPUT_ROOTS = [allowed_root]
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = [allowed_root]
        create = client.post(
            "/v1/jobs",
            json={
                "pipeline": "lux-depth-v3",
                "args": {"input_dir": str(input_dir), "output_dir": str(output_dir)},
            },
        )
    finally:
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
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


def test_job_events_replays_persisted_events_from_last_event_id(client: TestClient) -> None:
    now = orchestrator_app._now()
    job = orchestrator_app.Job(
        id="job_sse_replay",
        created_at=now,
        state="running",
        progress=0,
        request={"pipeline": "lux-depth-v3"},
    )
    _seed_job(job)

    asyncio.run(orchestrator_app._publish_event(job.id, "state", {"id": job.id, "state": "running", "progress": 0}))
    job.progress = 45
    asyncio.run(orchestrator_app._publish_event(job.id, "progress", {"id": job.id, "progress": 45}))
    job.state = "succeeded"
    job.exit_code = 0
    asyncio.run(
        orchestrator_app._publish_event(
            job.id,
            "done",
            {
                "id": job.id,
                "state": "succeeded",
                "exit_code": 0,
                "error": None,
                "artifacts": {},
            },
        )
    )
    job.done_published_at = orchestrator_app._now()
    job.finished_at = job.done_published_at

    with client.stream("GET", f"/v1/jobs/{job.id}/events", headers={"Last-Event-ID": "1"}) as stream_response:
        assert stream_response.status_code == 200
        events = _collect_sse_events(stream_response)

    assert [name for name, _payload in events] == ["progress", "done"]
    assert events[0][1]["progress"] == 45


def test_job_events_replay_beyond_latest_seq_falls_back_to_state_first(client: TestClient) -> None:
    now = orchestrator_app._now()
    job = orchestrator_app.Job(
        id="job_sse_replay_beyond_latest",
        created_at=now,
        state="running",
        progress=12,
        request={"pipeline": "lux-depth-v3"},
    )
    _seed_job(job)
    asyncio.run(orchestrator_app._publish_event(job.id, "state", {"id": job.id, "state": "running", "progress": 12}))
    job.state = "succeeded"
    job.exit_code = 0
    job.finished_at = orchestrator_app._now()
    job.done_published_at = job.finished_at
    _sync_seeded_job(job)

    with client.stream("GET", f"/v1/jobs/{job.id}/events", headers={"Last-Event-ID": "999"}) as stream_response:
        assert stream_response.status_code == 200
        events = _collect_sse_events(stream_response)

    assert [name for name, _payload in events] == ["state", "done"]
    assert events[0][1]["state"] == "succeeded"
    assert events[1][1]["state"] == "succeeded"


def test_job_events_replay_failure_falls_back_to_state_first(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAsyncIterator:
        def __aiter__(self):  # noqa: ANN204
            return self

        async def __anext__(self):  # noqa: ANN204
            raise RuntimeError("event store unavailable")

    class FailingEventStore:
        def events_since(self, _job_id: str, *, after_seq: int) -> FailingAsyncIterator:
            return FailingAsyncIterator()

    def failing_event_store() -> FailingEventStore:
        return FailingEventStore()

    now = orchestrator_app._now()
    job = orchestrator_app.Job(
        id="job_sse_replay_failure",
        created_at=now,
        state="succeeded",
        progress=100,
        finished_at=now,
        done_published_at=now,
        exit_code=0,
        request={"pipeline": "lux-depth-v3"},
    )
    _seed_job(job)
    monkeypatch.setattr(orchestrator_app, "_job_event_store", failing_event_store)

    with client.stream("GET", f"/v1/jobs/{job.id}/events", headers={"Last-Event-ID": "1"}) as stream_response:
        assert stream_response.status_code == 200
        events = _collect_sse_events(stream_response)

    assert [name for name, _payload in events] == ["state", "done"]
    assert events[0][1]["progress"] == 100


def test_job_events_rejects_invalid_last_event_id(client: TestClient) -> None:
    now = orchestrator_app._now()
    job = orchestrator_app.Job(
        id="job_sse_bad_last_event_id",
        created_at=now,
        state="running",
        progress=0,
        request={"pipeline": "lux-depth-v3"},
    )
    _seed_job(job)

    response = client.get(f"/v1/jobs/{job.id}/events", headers={"Last-Event-ID": "not-a-seq"})
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"] == {"field": "Last-Event-ID", "reason": "invalid_request"}


def test_late_job_events_done_payload_includes_fastvlm_captioning_status(client: TestClient) -> None:
    now = orchestrator_app._now()
    job = orchestrator_app.Job(
        id="job_phase22_late_sse",
        created_at=now,
        state="succeeded",
        progress=100,
        exit_code=0,
        done_published_at=now,
        finished_at=now,
        request={"pipeline": "lux-depth-v3"},
        run_summary={
            "source": "run_card",
            "batch_id": "2026-05-04_120000",
            "success_count": 1,
            "error_count": 0,
            "captioning_status": {
                "enabled": True,
                "backend": "fastvlm",
                "status": "succeeded",
                "role": "advisory",
                "sidecar_count": 1,
                "failed_count": 0,
                "used_for_quality_gate": False,
            },
        },
    )
    _seed_job(job)

    with client.stream("GET", f"/v1/jobs/{job.id}/events") as stream_response:
        assert stream_response.status_code == 200
        events = _collect_sse_events(stream_response)

    done_payload = next(payload for name, payload in events if name == "done")
    status = done_payload["run_summary"]["captioning_status"]
    assert status["status"] == "succeeded"
    assert status["sidecar_count"] == 1
    assert status["used_for_quality_gate"] is False


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
    _seed_job(job)

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
