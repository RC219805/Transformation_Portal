"""API contract tests for execution endpoints.

Tests cover:
1. POST /api/exec/run returns 202 with run_id
2. Returned run_id is immediately queryable via GET /runs/{run_id}
3. POST /runs/{run_id}/cancel returns appropriate status codes
4. Cancel endpoint is idempotent
5. Missing run returns 404
6. Response payload structure validation
"""

from __future__ import annotations

from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard.execution_api import (
    create_execution_router,
    set_manager,
)
from transformation_portal.dashboard.execution_manager import (
    ExecutionManager,
    RunStatus,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def test_app() -> FastAPI:
    """Create a FastAPI app with execution router for testing."""
    app = FastAPI()
    router = create_execution_router()
    app.include_router(router)
    return app


@pytest.fixture
def manager() -> ExecutionManager:
    """Create a fresh ExecutionManager for each test."""
    mgr = ExecutionManager()
    set_manager(mgr)
    return mgr


@pytest.fixture
def client(test_app: FastAPI, manager: ExecutionManager) -> Generator[TestClient, None, None]:
    """Create a test client for testing."""
    with TestClient(test_app) as tc:
        yield tc


class TestPostRun:
    """Tests for POST /api/exec/run endpoint."""

    def test_returns_202_accepted(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that POST /run returns 202 Accepted."""
        pipeline = {
            "nodes": [{"id": "n1", "type": "passthrough"}],
            "edges": [],
        }

        response = client.post("/api/exec/run", json=pipeline)

        assert response.status_code == 202

    def test_returns_run_id_in_response(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that response includes run_id."""
        pipeline = {"nodes": [], "edges": []}

        response = client.post("/api/exec/run", json=pipeline)
        data = response.json()

        assert "run_id" in data
        assert isinstance(data["run_id"], str)
        assert len(data["run_id"]) > 0

    def test_response_has_expected_fields(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that response has status, run_id, and message fields."""
        pipeline = {"nodes": [], "edges": []}

        response = client.post("/api/exec/run", json=pipeline)
        data = response.json()

        assert data["status"] == "accepted"
        assert "run_id" in data
        assert "message" in data

    def test_run_id_immediately_queryable(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that returned run_id is immediately visible via GET."""
        pipeline = {
            "nodes": [{"id": "n1", "type": "passthrough"}],
            "edges": [],
        }

        # Start the run
        post_response = client.post("/api/exec/run", json=pipeline)
        run_id = post_response.json()["run_id"]

        # Query immediately (no delay)
        get_response = client.get(f"/api/exec/runs/{run_id}")

        assert get_response.status_code == 200
        data = get_response.json()
        assert data["run_id"] == run_id


class TestGetRun:
    """Tests for GET /api/exec/runs/{run_id} endpoint."""

    def test_returns_run_details(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that GET returns full run details."""
        # Create a run manually
        run_id = manager.allocate_run_id()
        manager.prepare_run(
            run_id,
            {
                "nodes": [{"id": "node-a", "type": "test"}],
                "edges": [],
            },
        )

        response = client.get(f"/api/exec/runs/{run_id}")
        data = response.json()

        assert data["run_id"] == run_id
        assert "status" in data
        assert "nodes" in data
        assert "node-a" in data["nodes"]

    def test_includes_cancel_requested_field(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that response includes cancel_requested field."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [], "edges": []})

        response = client.get(f"/api/exec/runs/{run_id}")
        data = response.json()

        assert "cancel_requested" in data
        assert data["cancel_requested"] is False

    def test_includes_current_node_id_field(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that response includes current_node_id field."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [], "edges": []})

        response = client.get(f"/api/exec/runs/{run_id}")
        data = response.json()

        assert "current_node_id" in data
        assert data["current_node_id"] is None

    def test_returns_404_for_missing_run(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that GET returns 404 for non-existent run."""
        response = client.get("/api/exec/runs/nonexistent")

        assert response.status_code == 404


class TestCancelRun:
    """Tests for POST /api/exec/runs/{run_id}/cancel endpoint."""

    def test_returns_202_for_active_run(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that cancel returns 202 for an active run."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.RUNNING

        response = client.post(f"/api/exec/runs/{run_id}/cancel")

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "cancelling"
        assert data["run_id"] == run_id

    def test_returns_200_for_already_cancelled(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that cancel returns 200 for already cancelled run."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.CANCELLED

        response = client.post(f"/api/exec/runs/{run_id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_returns_200_for_complete_run(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that cancel returns 200 for complete run."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.COMPLETE

        response = client.post(f"/api/exec/runs/{run_id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"

    def test_returns_404_for_missing_run(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that cancel returns 404 for non-existent run."""
        response = client.post("/api/exec/runs/nonexistent/cancel")

        assert response.status_code == 404

    def test_cancel_is_idempotent(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that multiple cancel calls are idempotent."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.RUNNING

        # First cancel
        response1 = client.post(f"/api/exec/runs/{run_id}/cancel")
        assert response1.status_code == 202

        # Second cancel (now in CANCELLING state)
        response2 = client.post(f"/api/exec/runs/{run_id}/cancel")
        assert response2.status_code == 202  # Still in cancelling state
        assert response2.json()["status"] == "cancelling"


class TestListRuns:
    """Tests for GET /api/exec/runs endpoint."""

    def test_returns_empty_list_initially(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that runs list is empty initially."""
        response = client.get("/api/exec/runs")

        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert data["runs"] == []

    def test_includes_registered_runs(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that registered runs appear in list."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [{"id": "n1", "type": "test"}], "edges": []})

        response = client.get("/api/exec/runs")
        data = response.json()

        assert len(data["runs"]) == 1
        assert data["runs"][0]["run_id"] == run_id

    def test_run_summary_has_expected_fields(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that run summary has expected fields."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [{"id": "n1", "type": "test"}], "edges": []})

        response = client.get("/api/exec/runs")
        data = response.json()

        run_summary = data["runs"][0]
        assert "run_id" in run_summary
        assert "status" in run_summary
        assert "node_count" in run_summary
        assert run_summary["node_count"] == 1


class TestRaceConditionPrevention:
    """Tests specifically for race condition prevention."""

    def test_run_visible_before_execution_starts(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that run is visible even if execution hasn't started."""
        # Manually pre-register without starting execution
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [{"id": "n1", "type": "test"}], "edges": []})

        # Query immediately
        response = client.get(f"/api/exec/runs/{run_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"  # Pre-registered but not started

    def test_nodes_visible_before_execution_starts(self, client: TestClient, manager: ExecutionManager) -> None:
        """Test that nodes are visible even before execution starts."""
        run_id = manager.allocate_run_id()
        manager.prepare_run(
            run_id,
            {
                "nodes": [
                    {"id": "input", "type": "source"},
                    {"id": "output", "type": "sink"},
                ],
                "edges": [],
            },
        )

        response = client.get(f"/api/exec/runs/{run_id}")
        data = response.json()

        assert "input" in data["nodes"]
        assert "output" in data["nodes"]
        # All nodes should be PENDING before execution
        assert data["nodes"]["input"]["status"] == "pending"
        assert data["nodes"]["output"]["status"] == "pending"


class TestManagerAccessors:
    """Tests for get_manager / set_manager."""

    def test_get_manager_creates_singleton_when_unset(self) -> None:
        from transformation_portal.dashboard import execution_api

        execution_api._manager = None
        first = execution_api.get_manager()
        second = execution_api.get_manager()

        assert isinstance(first, ExecutionManager)
        assert first is second

    def test_set_manager_replaces_global(self) -> None:
        from transformation_portal.dashboard import execution_api

        replacement = ExecutionManager()
        set_manager(replacement)

        assert execution_api.get_manager() is replacement


class TestBroadcast:
    """Tests for the WebSocket broadcast helper."""

    @pytest.fixture(autouse=True)
    def _reset_clients(self) -> Generator[None, None, None]:
        from transformation_portal.dashboard import execution_api

        execution_api._websocket_clients.clear()
        yield
        execution_api._websocket_clients.clear()

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_clients(self) -> None:
        from transformation_portal.dashboard import execution_api

        class _Good:
            def __init__(self) -> None:
                self.sent: list[dict] = []

            async def send_json(self, msg: dict) -> None:
                self.sent.append(msg)

        good = _Good()
        execution_api._websocket_clients.append(good)

        await execution_api.broadcast({"type": "ping"})

        assert good.sent == [{"type": "ping"}]

    @pytest.mark.asyncio
    async def test_broadcast_drops_disconnected_clients(self) -> None:
        from transformation_portal.dashboard import execution_api

        class _Broken:
            async def send_json(self, msg: dict) -> None:
                raise RuntimeError("gone")

        broken = _Broken()
        execution_api._websocket_clients.append(broken)

        await execution_api.broadcast({"type": "ping"})

        assert broken not in execution_api._websocket_clients


class TestExecutionWebSocket:
    """Tests for the /api/exec/ws WebSocket endpoint."""

    @pytest.fixture(autouse=True)
    def _reset_clients(self) -> Generator[None, None, None]:
        from transformation_portal.dashboard import execution_api

        execution_api._websocket_clients.clear()
        yield
        execution_api._websocket_clients.clear()

    def test_ping_pong_and_client_tracking(self, client: TestClient) -> None:
        from transformation_portal.dashboard import execution_api

        with client.websocket_connect("/api/exec/ws") as ws:
            assert len(execution_api._websocket_clients) == 1
            ws.send_text("ping")
            assert ws.receive_text() == "pong"

        assert len(execution_api._websocket_clients) == 0


class TestRunFailurePath:
    """Tests for the POST /run error handling."""

    def test_returns_500_when_background_start_fails(self, client: TestClient, manager: ExecutionManager) -> None:
        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("scheduler unavailable")

        manager.start_pipeline_background = _boom  # type: ignore[method-assign]

        response = client.post("/api/exec/run", json={"nodes": [], "edges": []})

        assert response.status_code == 500


class TestExecutionUi:
    """Tests for the served HTML route and helper."""

    def test_ui_route_serves_html(self, client: TestClient) -> None:
        response = client.get("/api/exec/")

        assert response.status_code == 200
        assert "Pipeline Execution Monitor" in response.text

    def test_html_helper_is_self_contained(self) -> None:
        from transformation_portal.dashboard.execution_api import get_execution_ui_html

        html = get_execution_ui_html()
        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")
