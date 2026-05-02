"""End-to-end rejection-path tests for the FastAPI orchestrator.

The lower-level helpers (``_has_valid_api_key``, ``_is_rate_limited``,
``_is_protected_api_key_endpoint``) are unit-tested in
``test_app_orchestrator_runtime.py``. These tests exercise the
*middleware-integrated* rejection paths through ``TestClient`` so the
contract surface — status codes, error envelope shape, header echo —
is pinned down for:

* API-key missing or wrong on a protected endpoint (401 UNAUTHORIZED)
* API-key authentication misconfigured (503 AUTH_CONFIGURATION_ERROR)
* TrustedHost middleware rejecting an off-allowlist Host header (400)
* Rate-limit middleware returning 429 RATE_LIMITED beyond the bucket
* Public liveness routes (``/healthz``, ``/ready``) bypassing auth

Tests follow the fixture conventions of
``test_app_orchestrator_contract_http.py``: an autouse fixture
snapshots and restores the configuration knobs that the rejection
logic reads.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")


@pytest.fixture(autouse=True)
def _reset_orchestrator_globals():
    previous_api_key = orchestrator_app.API_KEY_SECRET
    previous_enforce = orchestrator_app.ENFORCE_JOB_API_KEY
    previous_rate_limit = orchestrator_app.RATE_LIMIT_PER_MINUTE
    orchestrator_app.API_KEY_SECRET = "rejection-secret"
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()
    orchestrator_app.JOBS.clear()
    try:
        yield
    finally:
        orchestrator_app.API_KEY_SECRET = previous_api_key
        orchestrator_app.ENFORCE_JOB_API_KEY = previous_enforce
        orchestrator_app.RATE_LIMIT_PER_MINUTE = previous_rate_limit
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()
        orchestrator_app.JOBS.clear()


@pytest.fixture(name="anon_client")
def _anon_client_fixture():
    with TestClient(orchestrator_app.app) as client:
        yield client


@pytest.fixture(name="auth_client")
def _auth_client_fixture():
    with TestClient(
        orchestrator_app.app,
        headers={"x-api-key": "rejection-secret"},
    ) as client:
        yield client


class TestApiKeyRejection:
    def test_missing_key_on_protected_endpoint_returns_401(self, anon_client):
        response = anon_client.get("/v1/jobs")
        assert response.status_code == 401
        body = response.json()
        # Error envelope contract: code + message + details
        error = body.get("error") or body
        assert error.get("code") == "UNAUTHORIZED"
        assert "invalid or missing api key" in error.get("message", "").lower()
        assert error.get("details", {}).get("path") == "/v1/jobs"

    def test_wrong_key_on_protected_endpoint_returns_401(self, anon_client):
        response = anon_client.get(
            "/v1/jobs",
            headers={"x-api-key": "this-is-not-the-key"},
        )
        assert response.status_code == 401
        error = response.json().get("error") or response.json()
        assert error.get("code") == "UNAUTHORIZED"

    def test_valid_bearer_token_passes_auth_layer(self, anon_client):
        # /v1/jobs requires auth; with a valid bearer it must reach the route
        # handler and return the documented envelope. A weaker `!= 401`
        # assertion would silently accept a 5xx regression in routing.
        response = anon_client.get(
            "/v1/jobs",
            headers={"authorization": "Bearer rejection-secret"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body.get("schema") == "tp.orchestrator.jobs.v1"
        assert body.get("success") is True
        assert body.get("error") is None
        data = body.get("data", {})
        assert "jobs" in data
        assert "total" in data
        assert "returned" in data

    def test_protected_endpoint_with_no_secret_configured_returns_503(self, anon_client):
        orchestrator_app.API_KEY_SECRET = ""
        # Enforcement still on, but no secret to compare against.
        response = anon_client.get("/v1/jobs")
        assert response.status_code == 503
        error = response.json().get("error") or response.json()
        assert error.get("code") == "AUTH_CONFIGURATION_ERROR"
        assert error.get("details", {}).get("env") == "TP_API_KEY"

    @pytest.mark.parametrize("path", ["/healthz", "/ready"])
    def test_public_liveness_bypasses_auth(self, anon_client, path):
        response = anon_client.get(path)
        assert response.status_code == 200


class TestTrustedHostRejection:
    def test_off_allowlist_host_returns_400(self, auth_client):
        response = auth_client.get("/healthz", headers={"host": "evil.example.com"})
        assert response.status_code == 400
        # Starlette's TrustedHostMiddleware returns plain text "Invalid host header".
        assert "invalid host" in response.text.lower()

    def test_localhost_is_allowed(self, auth_client):
        response = auth_client.get("/healthz", headers={"host": "localhost"})
        assert response.status_code == 200

    def test_loopback_ipv4_is_allowed(self, auth_client):
        response = auth_client.get("/healthz", headers={"host": "127.0.0.1"})
        assert response.status_code == 200


class TestRateLimitRejection:
    def test_burst_above_threshold_returns_429(self, auth_client):
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 3
        # First 3 requests succeed; the 4th from the same client should 429.
        for _ in range(3):
            assert auth_client.get("/healthz").status_code == 200
        response = auth_client.get("/healthz")
        assert response.status_code == 429
        error = response.json().get("error") or response.json()
        assert error.get("code") == "RATE_LIMITED"
        assert "rate limit exceeded" in error.get("message", "").lower()
        assert "client_ip" in error.get("details", {})

    def test_disabled_rate_limit_never_rejects(self, auth_client):
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 0
        for _ in range(20):
            assert auth_client.get("/healthz").status_code == 200
