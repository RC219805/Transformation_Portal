"""Unit tests for the dashboard RL optimization API.

The deterministic, dependency-light endpoints (/actions, /policy, /status)
are covered directly. The /optimize endpoint is covered for its synchronous
contract only; its background training task degrades gracefully when ML
runtimes are absent (core CI lane) and is not asserted here.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard.rl_api import create_rl_api_router

pytestmark = pytest.mark.unit


@pytest.fixture
def client() -> TestClient:
    """A test client backed by a fresh RL router (fresh job state per test)."""
    router = create_rl_api_router()
    assert router is not None, "FastAPI must be available in the test environment"
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestRouterFactory:
    """Tests for create_rl_api_router."""

    def test_router_has_rl_prefix(self) -> None:
        router = create_rl_api_router()

        assert router is not None
        assert router.prefix == "/rl"


class TestActionsEndpoint:
    """Tests for GET /rl/actions."""

    def test_lists_actions_with_count(self, client: TestClient) -> None:
        body = client.get("/rl/actions").json()

        assert body["count"] > 0
        assert len(body["actions"]) <= 50
        assert len(body["actions"]) == min(body["count"], 50)

    def test_action_entries_have_expected_shape(self, client: TestClient) -> None:
        body = client.get("/rl/actions").json()

        entry = body["actions"][0]
        assert set(entry) == {"index", "node", "action_type", "params"}
        assert isinstance(entry["index"], int)
        assert isinstance(entry["node"], str)


class TestPolicyEndpoint:
    """Tests for GET /rl/policy."""

    def test_returns_action_classification_lists(self, client: TestClient) -> None:
        body = client.get("/rl/policy").json()

        assert set(body) == {"safe_actions", "risky_actions", "blocked_actions"}
        assert isinstance(body["safe_actions"], list)
        assert isinstance(body["risky_actions"], list)
        assert isinstance(body["blocked_actions"], list)


class TestStatusEndpoint:
    """Tests for GET /rl/status/{job_id}."""

    def test_unknown_job_returns_error(self, client: TestClient) -> None:
        body = client.get("/rl/status/nonexistent").json()

        assert body == {"error": "Job not found"}


class TestOptimizeEndpoint:
    """Tests for POST /rl/optimize (synchronous contract only)."""

    def test_optimize_returns_job_id_and_started_status(self, client: TestClient) -> None:
        response = client.post("/rl/optimize", json={"pipeline": {}, "max_iterations": 1})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "started"
        assert isinstance(body["job_id"], str)
        assert len(body["job_id"]) > 0

    def test_started_job_is_queryable(self, client: TestClient) -> None:
        job_id = client.post("/rl/optimize", json={"pipeline": {}}).json()["job_id"]

        body = client.get(f"/rl/status/{job_id}").json()

        # The background training task may finish or fail depending on the
        # runtime, but the job record must exist and carry a status.
        assert "error" not in body or body.get("status") is not None
        assert "status" in body
