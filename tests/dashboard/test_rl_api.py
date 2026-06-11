"""Behavioral coverage for ``dashboard.rl_api``.

Covers the torch-free surface of the RL optimization router: the
action-listing and policy-config endpoints (backed by the pure
``transformation_portal.rl.action_space`` / ``policy_guard`` modules), the
job-start handshake, the unknown-job status path, and the FastAPI guard.
The actual RL training that the start endpoint schedules requires torch and
is intentionally not asserted here.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import rl_api


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(rl_api.create_rl_api_router())
    return TestClient(app)


def test_list_actions(client: TestClient) -> None:
    body = client.get("/rl/actions").json()
    assert body["count"] > 0
    first = body["actions"][0]
    assert {"index", "node", "action_type", "params"} <= set(first)


def test_get_policy_config(client: TestClient) -> None:
    body = client.get("/rl/policy").json()
    assert {"safe_actions", "risky_actions", "blocked_actions"} <= set(body)
    assert isinstance(body["safe_actions"], list)


def test_status_unknown_job(client: TestClient) -> None:
    assert client.get("/rl/status/ghost").json() == {"error": "Job not found"}


def test_optimize_returns_job_handshake(client: TestClient) -> None:
    # The scheduled background task needs torch (absent in the core lane) and
    # will fail closed; we only assert the synchronous handshake here.
    resp = client.post("/rl/optimize", json={"max_iterations": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert isinstance(body["job_id"], str) and body["job_id"]


def test_create_router_returns_none_without_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl_api, "FASTAPI_AVAILABLE", False)
    assert rl_api.create_rl_api_router() is None
