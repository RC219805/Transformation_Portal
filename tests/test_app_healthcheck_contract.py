#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP contract tests for the FastAPI orchestrator's healthcheck routes.

This file is the first family-scoped slice of the historically-monolithic
``tests/test_app_orchestrator_contract_http.py`` (~3.6k LOC). Healthcheck
routes (``/healthz``, ``/ready``) are a clean starting point because:

* They are pure read-only routes with no job-state dependencies.
* They have only three contract behaviours (response shape, no-cache
  headers, trusted-host enforcement) that are unlikely to grow.
* They sit at the fail-loud boundary external probes call, so it's
  worth surfacing them in their own file rather than burying them in
  a 4k-line bag of mixed contracts.

The fixtures below are deliberately a minimal duplicate of the ones in
``tests/test_app_orchestrator_contract_http.py``; future family extracts
may want to consolidate via ``tests/conftest.py``, but until at least
two or three families are split a shared conftest would change the
fixture scope of dozens of in-place tests and is not worth the risk.

Family registration: ``healthcheck`` in ``tests/test_app_route_inventory.py``.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")


@pytest.fixture(autouse=True)
def _reset_orchestrator_globals():
    # Healthcheck routes don't read JOBS/EVENT_SUBSCRIBERS/RATE_LIMIT_BUCKETS,
    # but we still snapshot+restore the trust-boundary globals so a
    # previously-mutated state from another test in the same session can't
    # leak in. Mirror the relevant subset of the original fixture in
    # tests/test_app_orchestrator_contract_http.py.
    previous_api_key = orchestrator_app.API_KEY_SECRET
    previous_enforce_job_api_key = orchestrator_app.ENFORCE_JOB_API_KEY
    orchestrator_app.API_KEY_SECRET = "contract-secret"
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    try:
        yield
    finally:
        orchestrator_app.API_KEY_SECRET = previous_api_key
        orchestrator_app.ENFORCE_JOB_API_KEY = previous_enforce_job_api_key


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


def test_healthz_rejects_untrusted_host_header(client: TestClient) -> None:
    response = client.get("/healthz", headers={"host": "evil.example.com"})

    assert response.status_code == 400
    assert "Invalid host header" in response.text
