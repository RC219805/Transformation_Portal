#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coverage for the backend + portal hardening pass.

Focused tests for the P0/P1 changes introduced in
``claude/harden-backend-portal-soFcZ``:

* ``/portal`` is the canonical UI route; ``/`` 307s to it.
* ``/v1/jobs`` performs a filesystem preflight on ``input_dir`` / ``output_dir``.
* Value preflight rejects unknown preset / depth_device / run_card_version.
* Artifact lookup never exposes files that were walked but not indexed.
* Artifact serving sends ``Content-Disposition: attachment`` for non-previewable
  content.
* Runner logs are redacted before both persistence and SSE publication.
* ``/v1/jobs`` list responses do not include ``logs_tail``; ``/v1/jobs/{id}``
  still does.
* Telemetry metadata drops non-finite floats; RUM metric allowlist rejects
  unknown metrics on event types that accept none.
* Artifact fingerprinting emits sha256 for bounded files and reports
  ``skipped_size`` above the cap.

Lifespan and portal.html static contracts are pinned by
``tests/test_lifespan_cleanup.py`` and ``tests/test_portal_html_static.py``
respectively.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import signal
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from transformation_portal.orchestrator import reset_singletons

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")


@pytest.fixture(autouse=True)
def _reset_orchestrator_state(tmp_path: Path, mark_da3_runtime_available: None) -> Iterator[None]:
    previous_api_key = orchestrator_app.API_KEY_SECRET
    previous_enforce = orchestrator_app.ENFORCE_JOB_API_KEY
    previous_input_roots = orchestrator_app.ALLOWED_INPUT_ROOTS
    previous_output_roots = orchestrator_app.ALLOWED_OUTPUT_ROOTS
    previous_path_roots = orchestrator_app.ALLOWED_PATH_ROOTS
    previous_max_indexed = orchestrator_app.MAX_INDEXED_ARTIFACTS
    previous_fingerprint_cap = orchestrator_app.ARTIFACT_FINGERPRINT_MAX_BYTES
    orchestrator_app.API_KEY_SECRET = "hardening-secret"
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()
    reset_singletons()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False
    allowed_root = (tmp_path / "allowed").resolve()
    allowed_root.mkdir(parents=True, exist_ok=True)
    orchestrator_app.ALLOWED_INPUT_ROOTS = [allowed_root]
    orchestrator_app.ALLOWED_OUTPUT_ROOTS = [allowed_root]
    orchestrator_app.ALLOWED_PATH_ROOTS = [allowed_root]
    try:
        yield
    finally:
        orchestrator_app.API_KEY_SECRET = previous_api_key
        orchestrator_app.ENFORCE_JOB_API_KEY = previous_enforce
        orchestrator_app.ALLOWED_INPUT_ROOTS = previous_input_roots
        orchestrator_app.ALLOWED_OUTPUT_ROOTS = previous_output_roots
        orchestrator_app.ALLOWED_PATH_ROOTS = previous_path_roots
        orchestrator_app.MAX_INDEXED_ARTIFACTS = previous_max_indexed
        orchestrator_app.ARTIFACT_FINGERPRINT_MAX_BYTES = previous_fingerprint_cap
        orchestrator_app.JOBS.clear()
        orchestrator_app.EVENT_SUBSCRIBERS.clear()
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()
        reset_singletons()
        orchestrator_app.app.state.job_repository = None
        orchestrator_app.app.state.job_repository_unavailable = False


@pytest.fixture(name="client")
def _client_fixture() -> Iterator[TestClient]:
    with TestClient(
        orchestrator_app.app,
        headers={"x-api-key": "hardening-secret"},
    ) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_root_redirects_to_portal(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/portal"


def test_root_redirect_preserves_query_string(client: TestClient) -> None:
    # Legacy/deep links such as `/?view=review` must continue to land on
    # the right workspace tab after the redirect; query string carries the
    # routing context we don't want the canonicalisation to drop.
    response = client.get("/?view=review&job=job_xyz", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/portal?view=review&job=job_xyz"


def test_portal_route_serves_html(client: TestClient) -> None:
    response = client.get("/portal")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert b"<html" in response.content.lower() or b"<!doctype" in response.content.lower()


def test_portal_route_registered_on_app() -> None:
    paths = {getattr(r, "path", "") for r in orchestrator_app.app.routes}
    assert "/portal" in paths
    assert "/" in paths


# ---------------------------------------------------------------------------
# Dispatch preflight: filesystem + values
# ---------------------------------------------------------------------------


def _lux_payload(input_dir: Path, output_dir: Path, **extra: object) -> dict:
    args = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }
    args.update(extra)
    return {"pipeline": "lux-depth-v3", "args": args}


def _seed_job(job: Any) -> Any:
    async def _seed() -> None:
        repo = orchestrator_app._job_repository()
        await repo.create(orchestrator_app._record_from_job(job))
        if job.artifacts or job.artifact_lookup:
            await repo.set_artifacts(job.id, job.artifacts, job.artifact_lookup)

    asyncio.run(_seed())
    orchestrator_app.JOBS[job.id] = job
    orchestrator_app.EVENT_SUBSCRIBERS.setdefault(job.id, {})
    return job


def test_dispatch_rejects_missing_input_dir(client: TestClient, tmp_path: Path) -> None:
    allowed = orchestrator_app.ALLOWED_INPUT_ROOTS[0]
    missing_input = allowed / "does-not-exist"
    output_dir = allowed / "out"
    response = client.post("/v1/jobs", json=_lux_payload(missing_input, output_dir))
    body = response.json()
    assert response.status_code == 400
    assert body["error"]["details"]["reason"] == "input_dir_required"
    assert body["error"]["details"]["field"] == "input_dir"


def test_dispatch_output_dir_gets_created_when_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_job(job, _argv):
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.finished_at = now
        job.done_published_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)
    allowed = orchestrator_app.ALLOWED_INPUT_ROOTS[0]
    input_dir = allowed / "inp"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir = allowed / "will-be-created"
    assert not output_dir.exists()

    response = client.post("/v1/jobs", json=_lux_payload(input_dir, output_dir))
    assert response.status_code == 200, response.text
    assert output_dir.is_dir()


def test_dispatch_does_not_mkdir_when_admission_rejects(client: TestClient) -> None:
    # Filesystem preflight is read-only; the mkdir must only run after the
    # admission gate succeeds, so a 429 response leaves no directories on
    # disk for the requested output path.
    allowed = orchestrator_app.ALLOWED_INPUT_ROOTS[0]
    input_dir = allowed / "inp"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir = allowed / "should-not-exist-after-429"
    assert not output_dir.exists()

    previous_limit = orchestrator_app.MAX_CONCURRENT_JOBS
    try:
        orchestrator_app.MAX_CONCURRENT_JOBS = 1
        _seed_job(
            orchestrator_app.Job(
                id="job_busy",
                created_at=orchestrator_app._now(),
                state="running",
                request={"pipeline": "lux-depth-v3", "args": {}},
            )
        )
        response = client.post("/v1/jobs", json=_lux_payload(input_dir, output_dir))
    finally:
        orchestrator_app.MAX_CONCURRENT_JOBS = previous_limit

    assert response.status_code == 429
    assert not output_dir.exists(), "output_dir was created despite admission rejection"


def test_artifact_fingerprint_default_cap_is_inexpensive() -> None:
    # The default cap must stay small enough that hashing up to
    # MAX_INDEXED_ARTIFACTS (200) artifacts per job remains an inexpensive
    # background-thread workload rather than blocking the event loop.
    assert orchestrator_app.ARTIFACT_FINGERPRINT_MAX_BYTES <= 16 * 1024 * 1024


def test_dispatch_rejects_unknown_preset(client: TestClient) -> None:
    allowed = orchestrator_app.ALLOWED_INPUT_ROOTS[0]
    input_dir = allowed / "inp"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir = allowed / "out"
    payload = _lux_payload(input_dir, output_dir, preset="no-such-preset")
    response = client.post("/v1/jobs", json=payload)
    body = response.json()
    assert response.status_code == 400
    assert body["error"]["details"]["reason"] == "invalid_preset"


def test_dispatch_rejects_unknown_depth_device(client: TestClient) -> None:
    allowed = orchestrator_app.ALLOWED_INPUT_ROOTS[0]
    input_dir = allowed / "inp"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir = allowed / "out"
    payload = _lux_payload(input_dir, output_dir, depth_device="rocm")
    response = client.post("/v1/jobs", json=payload)
    body = response.json()
    assert response.status_code == 400
    # Preview validation will normally convert this to invalid_depth_device;
    # the dispatch-side preflight is the last line of defence and should agree.
    assert body["error"]["details"]["reason"] == "invalid_depth_device"


# ---------------------------------------------------------------------------
# Artifact lookup boundary + attachment headers
# ---------------------------------------------------------------------------


def test_artifact_lookup_excludes_non_indexed_files(tmp_path: Path) -> None:
    orchestrator_app.MAX_INDEXED_ARTIFACTS = 2
    allowed = orchestrator_app.ALLOWED_OUTPUT_ROOTS[0]
    output_dir = allowed / "job_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "a.json").write_text("{}", encoding="utf-8")
    (output_dir / "b.json").write_text("{}", encoding="utf-8")
    (output_dir / "secret.txt").write_text("sensitive", encoding="utf-8")

    job = orchestrator_app.Job(
        id="job_lookup",
        created_at=orchestrator_app._now(),
        state="succeeded",
        request={
            "pipeline": "lux-depth-v3",
            "args": {"input_dir": str(output_dir), "output_dir": str(output_dir)},
        },
        effective_request={
            "pipeline": "lux-depth-v3",
            "args": {"input_dir": str(output_dir), "output_dir": str(output_dir)},
        },
    )
    orchestrator_app._index_job_artifacts(job)

    # Only the two indexed artifacts are reachable through the lookup; the
    # third file on disk must be excluded to preserve least privilege.
    assert set(job.artifact_lookup.keys()) == {"a.json", "b.json"}


def test_artifact_endpoint_attaches_non_previewable(client: TestClient, tmp_path: Path) -> None:
    allowed = orchestrator_app.ALLOWED_OUTPUT_ROOTS[0]
    output_dir = allowed / "job_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.html").write_text("<html></html>", encoding="utf-8")

    job = orchestrator_app.Job(
        id="job_attach",
        created_at=orchestrator_app._now(),
        state="succeeded",
        request={
            "pipeline": "lux-depth-v3",
            "args": {"input_dir": str(output_dir), "output_dir": str(output_dir)},
        },
        effective_request={
            "pipeline": "lux-depth-v3",
            "args": {"input_dir": str(output_dir), "output_dir": str(output_dir)},
        },
    )
    orchestrator_app._index_job_artifacts(job)
    _seed_job(job)

    response = client.get(f"/v1/jobs/{job.id}/artifacts/report.html")
    assert response.status_code == 200
    disposition = response.headers.get("content-disposition", "")
    assert "attachment" in disposition
    assert 'filename="report.html"' in disposition
    assert response.headers["X-Content-Type-Options"] == "nosniff"


# ---------------------------------------------------------------------------
# Log redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        # Keep these fixtures non-secret-shaped: CI runs gitleaks on pushed
        # diffs and on clean checked-out source trees.
        ("api_key=query-secret", "api_key=<redacted>"),
        ("OPENAI_API_KEY=query-secret", "OPENAI_API_KEY=<redacted>"),
        ("password: hunter2", "password: <redacted>"),
        ("token = abcdef", "token = <redacted>"),
        ("Authorization: Bearer abcdef.token", "Authorization: <redacted>"),
        ("plain runner progress=50%", "plain runner progress=50%"),
    ],
)
def test_log_redaction_rewrites_secret_shapes(line: str, expected: str) -> None:
    assert orchestrator_app._redact_log_line(line) == expected


# ---------------------------------------------------------------------------
# Job list payload
# ---------------------------------------------------------------------------


def test_list_jobs_omits_log_tail(client: TestClient) -> None:
    job = orchestrator_app.Job(
        id="job_list",
        created_at=orchestrator_app._now(),
        state="succeeded",
        request={"pipeline": "lux-depth-v3", "args": {}},
    )
    job.add_log("example log line")
    _seed_job(job)

    list_response = client.get("/v1/jobs")
    assert list_response.status_code == 200
    job_payloads = list_response.json()["data"]["jobs"]
    assert len(job_payloads) == 1
    assert "logs_tail" not in job_payloads[0]


def test_get_job_includes_log_tail_by_default(client: TestClient) -> None:
    job = orchestrator_app.Job(
        id="job_detail",
        created_at=orchestrator_app._now(),
        state="succeeded",
        request={"pipeline": "lux-depth-v3", "args": {}},
    )
    job.add_log("detail log line")
    _seed_job(job)

    detail_response = client.get(f"/v1/jobs/{job.id}")
    assert detail_response.status_code == 200
    assert "logs_tail" in detail_response.json()["data"]

    trimmed = client.get(f"/v1/jobs/{job.id}?include_logs=false")
    assert trimmed.status_code == 200
    assert "logs_tail" not in trimmed.json()["data"]


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_sanitize_metadata_drops_non_finite_floats() -> None:
    sanitized = orchestrator_app._portal_sanitize_metadata(
        {
            "good": 1.25,
            "nanfield": float("nan"),
            "inf": float("inf"),
            "negative_inf": float("-inf"),
        }
    )
    assert sanitized == {"good": 1.25}


def test_rum_metric_allowlist_rejects_unknown_metric_for_empty_allowlist() -> None:
    # `sse_reconnect` intentionally carries no metric token; any caller-supplied
    # token must be rejected rather than silently accepted.
    record, reason = orchestrator_app._record_portal_rum(
        {
            "event_type": "sse_reconnect",
            "route": "/portal",
            "view": "overview",
            "unit": "count",
            "value": 1,
            "metric": "unexpected-metric",
        },
        _fake_request(),
    )
    assert record is None
    assert reason == "invalid_metric"


def _fake_request():
    """Minimal stand-in for Starlette ``Request`` to drive _record_portal_rum."""

    class _DummyState:
        pass

    class _DummyRequest:
        def __init__(self) -> None:
            self.headers: dict = {}
            self.state = _DummyState()

    return _DummyRequest()


# ---------------------------------------------------------------------------
# Artifact fingerprints
# ---------------------------------------------------------------------------


def test_artifact_fingerprint_emits_sha256_for_bounded_files(tmp_path: Path) -> None:
    path = tmp_path / "small.bin"
    path.write_bytes(b"hello world")
    sha, status = orchestrator_app._artifact_fingerprint(path, path.stat().st_size)
    assert status == "ok"
    assert sha is not None and len(sha) == 64


def test_artifact_fingerprint_reports_skipped_size_above_cap(tmp_path: Path) -> None:
    orchestrator_app.ARTIFACT_FINGERPRINT_MAX_BYTES = 4
    path = tmp_path / "too-big.bin"
    path.write_bytes(b"0123456789")
    sha, status = orchestrator_app._artifact_fingerprint(path, path.stat().st_size)
    assert sha is None
    assert status == "skipped_size"


# ---------------------------------------------------------------------------
# Subprocess cancellation helpers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="process-group cancel is POSIX-only")
def test_signal_process_tree_uses_killpg_when_session_matches() -> None:
    recorded: dict[str, object] = {}

    class _FakeProc:
        pid = 12345

        @property
        def returncode(self):
            return None

    def fake_getpgid(pid: int) -> int:
        assert pid == 12345
        return 12345  # same as pid => the child is its own session leader

    def fake_killpg(pgid: int, sig: int) -> None:
        recorded["pgid"] = pgid
        recorded["sig"] = sig

    original_getpgid = os.getpgid
    original_killpg = os.killpg
    os.getpgid = fake_getpgid  # type: ignore[assignment]
    os.killpg = fake_killpg  # type: ignore[assignment]
    try:
        delivered = orchestrator_app._signal_process_tree(_FakeProc(), signal.SIGTERM)
    finally:
        os.getpgid = original_getpgid  # type: ignore[assignment]
        os.killpg = original_killpg  # type: ignore[assignment]

    assert delivered is True
    assert recorded == {"pgid": 12345, "sig": signal.SIGTERM}


@pytest.mark.skipif(os.name == "nt", reason="process-group cancel is POSIX-only")
def test_signal_process_tree_returns_false_when_not_session_leader() -> None:
    class _FakeProc:
        pid = 4242

        @property
        def returncode(self):
            return None

    original_getpgid = os.getpgid
    os.getpgid = lambda _pid: 9999  # type: ignore[assignment]
    try:
        delivered = orchestrator_app._signal_process_tree(_FakeProc(), signal.SIGTERM)
    finally:
        os.getpgid = original_getpgid  # type: ignore[assignment]

    assert delivered is False
