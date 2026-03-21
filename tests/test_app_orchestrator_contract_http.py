#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP contract tests for the root FastAPI orchestrator app."""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

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


def test_presets_contract_for_lux_depth_pipeline(client: TestClient) -> None:
    response = client.get("/v1/presets", params={"pipeline": "lux-depth-v3"})
    body = response.json()
    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.presets.v1"
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["pipeline"] == "lux-depth-v3"
    assert any(item["name"] == "premium" for item in body["data"]["presets"])


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


def test_v1_routes_enforce_api_key_for_reads_and_events(client: TestClient) -> None:
    orchestrator_app.API_KEY_SECRET = "contract-secret"
    finished_job = orchestrator_app.Job(
        id="job_auth",
        created_at=orchestrator_app._now(),
        finished_at=orchestrator_app._now(),
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


def test_v1_routes_fail_closed_when_auth_enforced_without_secret(client: TestClient) -> None:
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    orchestrator_app.API_KEY_SECRET = ""

    response = client.get("/v1/jobs", headers={"x-api-key": "irrelevant"})
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "AUTH_CONFIGURATION_ERROR"
    assert body["error"]["details"]["env"] == "TP_API_KEY"


def test_invalid_job_payload_returns_typed_invalid_argument(client: TestClient) -> None:
    response = client.post("/v1/jobs", json={"pipeline": "not-allowed", "args": {}})
    body = response.json()
    assert response.status_code == 400
    assert body["schema"] == "tp.orchestrator.error.v1"
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"


def test_archive_gate_pipeline_submission_returns_job_envelope(client: TestClient, monkeypatch) -> None:
    async def fake_run_job(job, _argv):  # noqa: ANN001
        job.state = "succeeded"
        job.exit_code = 0
        job.finished_at = orchestrator_app._now()

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    response = client.post(
        "/v1/jobs",
        json={"pipeline": "archive-gate-a", "args": {"input_dir": "./in", "output_dir": "./out"}},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["schema"] == "tp.orchestrator.job.v1"
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["id"].startswith("job_")


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
    assert v1_method_not_allowed.json()["error"]["code"] == "HTTP_ERROR"
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
    assert body["error"]["details"]["path"] == "/v1/jobs"
    assert body["error"]["details"]["errors"]


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
        job.finished_at = orchestrator_app._now()
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
