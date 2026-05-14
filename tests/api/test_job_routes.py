"""Unit-level route contract tests for the job-lifecycle HTTP routes.

Tests the FastAPI app's route handlers end-to-end via TestClient, covering:
  - GET /healthz
  - GET /v1/readiness
  - GET /v1/jobs (list, empty)
  - GET /v1/jobs/{id} (found and 404)
  - POST /v1/jobs (invalid pipeline → 400)
  - POST /v1/jobs (path validation error → 400)
  - POST /v1/jobs/{id}/cancel (found and 404)

Jobs are injected directly into the in-process JOBS dict to test retrieval and
runtime overlay behavior without executing actual subprocesses; durable route
state is seeded through the JobRepository.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import pytest

from transformation_portal.orchestrator import reset_singletons

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------


_TEST_API_KEY = "test-route-key"


@pytest.fixture(name="client")
def _client_fixture(monkeypatch) -> Iterator[Any]:
    """Yield a TestClient with a known API key patched into the live app module."""
    from fastapi.testclient import TestClient

    import app as orchestrator_app

    # Patch the module-level constant so the auth middleware accepts our key.
    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", _TEST_API_KEY)

    with TestClient(
        orchestrator_app.app,
        headers={"x-api-key": _TEST_API_KEY},
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Ensure the global JOBS dict and rate-limit buckets are clean before and after each test."""
    import app as orchestrator_app

    reset_singletons()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()
    yield
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()
    reset_singletons()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False


def _inject_job(job_id: str = "test-job-001", state: str = "queued", *, cache_runtime: bool = True) -> Any:
    """Insert a minimal Job through the durable repository, with optional runtime cache overlay."""
    import app as orchestrator_app

    job = orchestrator_app.Job(
        id=job_id,
        created_at=time.time(),
        state=state,
        request={"pipeline": "lux-depth-v3"},
    )

    async def _seed() -> None:
        await orchestrator_app._job_repository().create(orchestrator_app._record_from_job(job))

    asyncio.run(_seed())
    if cache_runtime:
        orchestrator_app.JOBS[job_id] = job
        orchestrator_app.EVENT_SUBSCRIBERS.setdefault(job_id, {})
    return job


# ---------------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------------


class TestHealthz:
    def test_returns_200(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_body_ok_true(self, client):
        body = client.get("/healthz").json()
        assert body.get("ok") is True

    def test_no_cache_header(self, client):
        response = client.get("/healthz")
        assert "no-store" in response.headers.get("cache-control", "")


# ---------------------------------------------------------------------------
# GET /v1/readiness
# ---------------------------------------------------------------------------


class TestReadiness:
    def test_returns_200(self, client):
        response = client.get("/v1/readiness")
        assert response.status_code == 200

    def test_envelope_success_true(self, client):
        body = client.get("/v1/readiness").json()
        assert body.get("success") is True

    def test_envelope_has_data(self, client):
        body = client.get("/v1/readiness").json()
        assert "data" in body


# ---------------------------------------------------------------------------
# GET /v1/jobs — listing
# ---------------------------------------------------------------------------


class TestListJobs:
    def test_empty_list_returns_200(self, client):
        response = client.get("/v1/jobs")
        assert response.status_code == 200

    def test_empty_list_envelope_success(self, client):
        body = client.get("/v1/jobs").json()
        assert body.get("success") is True

    def test_empty_list_data_has_jobs_key(self, client):
        body = client.get("/v1/jobs").json()
        assert "jobs" in body.get("data", {})

    def test_injected_job_appears_in_list(self, client):
        _inject_job("listed-job")
        body = client.get("/v1/jobs").json()
        job_ids = [j.get("id") for j in body["data"]["jobs"]]
        assert "listed-job" in job_ids

    def test_repository_job_appears_in_list_after_runtime_cache_clear(self, client):
        import app as orchestrator_app

        _inject_job("repo-listed-job")
        orchestrator_app.JOBS.clear()
        body = client.get("/v1/jobs").json()
        job_ids = [j.get("id") for j in body["data"]["jobs"]]
        assert "repo-listed-job" in job_ids


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id} — single job
# ---------------------------------------------------------------------------


class TestGetJob:
    def test_returns_404_for_unknown_job(self, client):
        response = client.get("/v1/jobs/nonexistent-id-xyz")
        assert response.status_code == 404

    def test_404_envelope_success_false(self, client):
        body = client.get("/v1/jobs/ghost").json()
        assert body.get("success") is False

    def test_returns_200_for_known_job(self, client):
        _inject_job("known-job")
        response = client.get("/v1/jobs/known-job")
        assert response.status_code == 200

    def test_returned_job_has_correct_id(self, client):
        _inject_job("find-me")
        body = client.get("/v1/jobs/find-me").json()
        assert body["data"]["id"] == "find-me"

    def test_returns_repository_job_after_runtime_cache_clear(self, client):
        import app as orchestrator_app

        _inject_job("repo-find-me")
        orchestrator_app.JOBS.clear()
        body = client.get("/v1/jobs/repo-find-me").json()
        assert body["data"]["id"] == "repo-find-me"

    def test_returned_job_has_state(self, client):
        _inject_job("state-job", state="queued")
        body = client.get("/v1/jobs/state-job").json()
        assert "state" in body["data"]


# ---------------------------------------------------------------------------
# POST /v1/jobs — invalid pipeline → 400
# ---------------------------------------------------------------------------


class TestCreateJobInvalidPipeline:
    def test_returns_400_for_unknown_pipeline(self, client):
        response = client.post("/v1/jobs", json={"pipeline": "no-such-pipeline", "args": {}})
        assert response.status_code == 400

    def test_400_envelope_success_false(self, client):
        body = client.post("/v1/jobs", json={"pipeline": "bad-pipe", "args": {}}).json()
        assert body.get("success") is False

    def test_400_code_is_invalid_argument(self, client):
        body = client.post("/v1/jobs", json={"pipeline": "bad-pipe", "args": {}}).json()
        error = body.get("error") or {}
        assert error.get("code") == "INVALID_ARGUMENT"

    def test_returns_400_for_missing_pipeline(self, client):
        """Empty pipeline field is also unsupported."""
        response = client.post("/v1/jobs", json={"args": {}})
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /v1/jobs — missing input path → 400
# ---------------------------------------------------------------------------


class TestCreateJobMissingPath:
    def test_returns_400_for_lux_without_input_dir(self, client):
        """lux-depth-v3 requires input_dir; omitting it should produce a 400."""
        response = client.post("/v1/jobs", json={"pipeline": "lux-depth-v3", "args": {}})
        assert response.status_code == 400

    def test_400_for_nonexistent_input_dir(self, client, tmp_path):
        missing = str(tmp_path / "does_not_exist")
        response = client.post(
            "/v1/jobs",
            json={"pipeline": "lux-depth-v3", "args": {"input_dir": missing, "output_dir": missing}},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /v1/jobs/{id}/cancel
# ---------------------------------------------------------------------------


class TestCancelJob:
    def test_returns_404_for_nonexistent_job(self, client):
        response = client.post("/v1/jobs/ghost-cancel/cancel")
        assert response.status_code == 404

    def test_404_envelope_success_false(self, client):
        body = client.post("/v1/jobs/ghost/cancel").json()
        assert body.get("success") is False

    def test_cancel_existing_queued_job_returns_200(self, client):
        _inject_job("to-cancel", state="queued")
        response = client.post("/v1/jobs/to-cancel/cancel")
        assert response.status_code == 200

    def test_cancel_response_envelope_success_true(self, client):
        _inject_job("cancel-ok", state="queued")
        body = client.post("/v1/jobs/cancel-ok/cancel").json()
        assert body.get("success") is True
