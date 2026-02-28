#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for root FastAPI orchestrator app runtime behavior."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from typing import Dict

import pytest
from starlette.requests import Request as StarletteRequest

orchestrator_app = importlib.import_module("app")


class _FakeRequest:
    """Lightweight request stub for SSE generator tests."""

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _flag_value(argv: list[str], flag: str) -> str:
    idx = argv.index(flag)
    return argv[idx + 1]


def _build_request(
    method: str, path: str, headers: Dict[str, str] | None = None, client_host: str = "127.0.0.1"
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
        "query_string": b"",
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
