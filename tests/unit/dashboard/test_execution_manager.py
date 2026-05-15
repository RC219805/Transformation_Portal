"""Unit tests for ExecutionManager lifecycle and cancellation semantics.

Tests cover:
1. Pre-registration of run state via prepare_run()
2. Task registry tracking
3. Cooperative cancellation with cancel_requested flag
4. Proper node skipping when cancelled
5. State transitions (PENDING → RUNNING → CANCELLING → CANCELLED)
6. Terminal state invariants
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from transformation_portal.dashboard.execution_manager import (
    ExecutionManager,
    NodeState,
    NodeStatus,
    RunState,
    RunStatus,
)

pytestmark = pytest.mark.unit


class TestPrepareRun:
    """Tests for prepare_run() pre-registration semantics."""

    def test_prepare_run_creates_run_state(self) -> None:
        """Test that prepare_run creates a RunState in active_runs."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()

        pipeline = {
            "nodes": [
                {"id": "node-1", "type": "passthrough"},
                {"id": "node-2", "type": "passthrough"},
            ],
            "edges": [],
        }

        run_state = manager.prepare_run(run_id, pipeline)

        assert run_id in manager.active_runs
        assert run_state.run_id == run_id
        assert run_state.status == RunStatus.PENDING
        assert "node-1" in run_state.nodes
        assert "node-2" in run_state.nodes

    def test_prepare_run_initializes_node_states_as_pending(self) -> None:
        """Test that all nodes are initialized with PENDING status."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()

        pipeline = {
            "nodes": [
                {"id": "a", "type": "test"},
                {"id": "b", "type": "test"},
            ],
            "edges": [],
        }

        run_state = manager.prepare_run(run_id, pipeline)

        assert run_state.nodes["a"].status == NodeStatus.PENDING
        assert run_state.nodes["b"].status == NodeStatus.PENDING

    def test_prepare_run_adds_to_history(self) -> None:
        """Test that prepare_run adds run_id to run_history."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()

        manager.prepare_run(run_id, {"nodes": [], "edges": []})

        assert run_id in manager.run_history

    def test_prepare_run_rejects_duplicate_run_id(self) -> None:
        """Test that prepare_run raises error for duplicate run_id."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [], "edges": []})

        with pytest.raises(ValueError, match="already registered"):
            manager.prepare_run(run_id, {"nodes": [], "edges": []})

    def test_prepare_run_trims_history(self) -> None:
        """Test that prepare_run trims history when exceeding max."""
        manager = ExecutionManager()
        manager._max_history = 5

        # Create runs and mark some as complete (terminal state)
        created_ids = []
        for i in range(7):
            run_id = manager.allocate_run_id()
            manager.prepare_run(run_id, {"nodes": [], "edges": []})
            # Mark first few as complete so they can be trimmed
            if i < 4:
                manager.active_runs[run_id].status = RunStatus.COMPLETE
            created_ids.append(run_id)

        # First two completed runs should be trimmed
        assert created_ids[0] not in manager.active_runs
        assert created_ids[1] not in manager.active_runs
        # Last five (mix of complete and pending) should remain
        assert created_ids[2] in manager.active_runs
        assert created_ids[6] in manager.active_runs
        assert len(manager.run_history) == 5


class TestStartPipelineBackground:
    """Tests for start_pipeline_background() task management."""

    @pytest.mark.asyncio
    async def test_start_pipeline_background_pre_registers_run(self) -> None:
        """Test that run is queryable immediately after start_pipeline_background."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        pipeline = {
            "nodes": [{"id": "n1", "type": "passthrough"}],
            "edges": [],
        }

        task = manager.start_pipeline_background(run_id, pipeline, broadcast)

        # Run should be immediately visible
        assert run_id in manager.active_runs
        run_state = manager.get_run_state(run_id)
        assert run_state is not None
        assert run_state.status in (RunStatus.PENDING, RunStatus.RUNNING)

        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_task_registered_in_tasks_by_run_id(self) -> None:
        """Test that background task is tracked in _tasks_by_run_id."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        pipeline = {"nodes": [], "edges": []}

        task = manager.start_pipeline_background(run_id, pipeline, broadcast)

        assert run_id in manager._tasks_by_run_id
        assert manager._tasks_by_run_id[run_id] is task

        # Wait for task to complete
        await task

        # Task should be cleaned up from registry
        assert run_id not in manager._tasks_by_run_id

    @pytest.mark.asyncio
    async def test_task_cleanup_on_completion(self) -> None:
        """Test that task is removed from registry after completion."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        # Simple pipeline that completes immediately
        pipeline = {"nodes": [], "edges": []}

        task = manager.start_pipeline_background(run_id, pipeline, broadcast)
        await task

        # Task should be cleaned up
        assert run_id not in manager._tasks_by_run_id


class TestCancelRequestSemantics:
    """Tests for cancel_requested flag and CANCELLING status."""

    @pytest.mark.asyncio
    async def test_cancel_sets_cancel_requested(self) -> None:
        """Test that cancel_run sets cancel_requested flag."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        # Pre-register with RUNNING status to simulate active run
        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.RUNNING

        result = await manager.cancel_run(run_id, broadcast)

        assert result == "cancelling"
        assert manager.active_runs[run_id].cancel_requested is True

    @pytest.mark.asyncio
    async def test_cancel_transitions_to_cancelling(self) -> None:
        """Test that cancel_run transitions status to CANCELLING."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.RUNNING

        await manager.cancel_run(run_id, broadcast)

        assert manager.active_runs[run_id].status == RunStatus.CANCELLING

    @pytest.mark.asyncio
    async def test_cancel_broadcasts_run_cancelling_event(self) -> None:
        """Test that cancel_run broadcasts run_cancelling event."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.RUNNING

        await manager.cancel_run(run_id, broadcast)

        broadcast.assert_called()
        call_args = broadcast.call_args_list[-1][0][0]
        assert call_args["type"] == "run_cancelling"
        assert call_args["run_id"] == run_id

    @pytest.mark.asyncio
    async def test_cancel_idempotent_when_already_cancelling(self) -> None:
        """Test that cancel_run is idempotent for CANCELLING status."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.CANCELLING
        manager.active_runs[run_id].cancel_requested = True

        result = await manager.cancel_run(run_id, broadcast)

        assert result == "cancelling"
        # Should not broadcast again
        broadcast.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_returns_complete_status_for_complete_run(self) -> None:
        """Test that cancel_run returns 'complete' for already complete runs."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.COMPLETE

        result = await manager.cancel_run(run_id, broadcast)

        assert result == "complete"
        broadcast.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_returns_none_for_missing_run(self) -> None:
        """Test that cancel_run returns None for non-existent run."""
        manager = ExecutionManager()
        broadcast = AsyncMock()

        result = await manager.cancel_run("nonexistent", broadcast)

        assert result is None


class TestRunStateFields:
    """Tests for new RunState fields (cancel_requested, current_node_id)."""

    def test_run_state_has_cancel_requested_field(self) -> None:
        """Test that RunState has cancel_requested field defaulting to False."""
        state = RunState(run_id="test")
        assert state.cancel_requested is False

    def test_run_state_has_current_node_id_field(self) -> None:
        """Test that RunState has current_node_id field defaulting to None."""
        state = RunState(run_id="test")
        assert state.current_node_id is None


class TestRunStatusEnum:
    """Tests for RunStatus enum values."""

    def test_cancelling_status_exists(self) -> None:
        """Test that CANCELLING status is defined."""
        assert RunStatus.CANCELLING.value == "cancelling"

    def test_all_expected_statuses_exist(self) -> None:
        """Test that all expected statuses are defined."""
        expected = {"pending", "running", "complete", "error", "cancelled", "cancelling"}
        actual = {s.value for s in RunStatus}
        assert expected == actual


class TestCooperativeCancellation:
    """Tests for cooperative cancellation during execution."""

    @pytest.mark.asyncio
    async def test_cancelled_run_does_not_emit_run_complete(self) -> None:
        """Test that a cancelled run emits run_cancelled, not run_complete."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        # Empty pipeline that will complete quickly
        pipeline = {"nodes": [], "edges": []}

        # Pre-register with cancellation already requested
        manager.prepare_run(run_id, pipeline)
        manager.active_runs[run_id].cancel_requested = True

        task = manager.start_pipeline_background(run_id, pipeline, broadcast)
        await task

        # Check that run_cancelled was emitted
        event_types = [call[0][0]["type"] for call in broadcast.call_args_list]
        assert "run_cancelled" in event_types
        # Verify run_complete is NOT emitted when cancel_requested is set
        assert "run_complete" not in event_types

    @pytest.mark.asyncio
    async def test_cancellation_marks_remaining_nodes_skipped(self) -> None:
        """Test that nodes after cancellation point are marked SKIPPED."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        executed = []

        class ImmediateNode:
            def __init__(self, node_id: str) -> None:
                self.node_id = node_id

            def run(self, **inputs: Any) -> Dict[str, Any]:
                executed.append(self.node_id)
                return {"done": True}

        # Register node implementations
        manager.node_registry = {
            "immediate": lambda **config: ImmediateNode(config.get("id", "unknown")),
        }

        pipeline = {
            "nodes": [
                {"id": "n1", "type": "passthrough"},
                {"id": "n2", "type": "passthrough"},
                {"id": "n3", "type": "passthrough"},
            ],
            "edges": [],
        }

        # Pre-register with cancellation already requested
        manager.prepare_run(run_id, pipeline)
        manager.active_runs[run_id].cancel_requested = True

        task = manager.start_pipeline_background(run_id, pipeline, broadcast)
        await task

        # All nodes should be skipped since cancel was requested before start
        run_state = manager.get_run_state(run_id)
        assert run_state is not None
        for node_id in ["n1", "n2", "n3"]:
            assert run_state.nodes[node_id].status == NodeStatus.SKIPPED


class TestCurrentNodeIdTracking:
    """Tests for current_node_id tracking during execution."""

    @pytest.mark.asyncio
    async def test_current_node_id_cleared_after_completion(self) -> None:
        """Test that current_node_id is None after run completes."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        pipeline = {
            "nodes": [{"id": "n1", "type": "passthrough"}],
            "edges": [],
        }

        task = manager.start_pipeline_background(run_id, pipeline, broadcast)
        await task

        run_state = manager.get_run_state(run_id)
        assert run_state is not None
        assert run_state.current_node_id is None


class TestTerminalStateInvariants:
    """Tests for terminal state invariants."""

    @pytest.mark.asyncio
    async def test_complete_run_cannot_become_cancelled(self) -> None:
        """Test that a COMPLETE run cannot transition to CANCELLED."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.COMPLETE

        result = await manager.cancel_run(run_id, broadcast)

        # Should return current status, not change it
        assert result == "complete"
        assert manager.active_runs[run_id].status == RunStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_error_run_cannot_become_cancelled(self) -> None:
        """Test that an ERROR run cannot transition to CANCELLED."""
        manager = ExecutionManager()
        run_id = manager.allocate_run_id()
        broadcast = AsyncMock()

        manager.prepare_run(run_id, {"nodes": [], "edges": []})
        manager.active_runs[run_id].status = RunStatus.ERROR

        result = await manager.cancel_run(run_id, broadcast)

        assert result == "error"
        assert manager.active_runs[run_id].status == RunStatus.ERROR


class _ConfigNode:
    """A registry node that records the config it was constructed with."""

    def __init__(self, **config: Any) -> None:
        self.config = config

    def run(self, **inputs: Any) -> Dict[str, Any]:
        return {"echoed": inputs, "config": self.config}


class _FailingNode:
    """A registry node whose run() always raises."""

    def __init__(self, **config: Any) -> None:
        pass

    def run(self, **inputs: Any) -> Dict[str, Any]:
        raise RuntimeError("node boom")


class TestDataclasses:
    """Tests for the NodeState / RunState dataclasses."""

    def test_node_state_defaults(self) -> None:
        node = NodeState(node_id="n1")
        assert node.status == NodeStatus.PENDING
        assert node.outputs == {}
        assert node.logs == []
        assert node.merkle_hash is None

    def test_run_state_defaults(self) -> None:
        run = RunState(run_id="r1")
        assert run.status == RunStatus.PENDING
        assert run.nodes == {}
        assert run.cancel_requested is False
        assert run.current_node_id is None


class TestResolveNodeImpl:
    """Tests for _resolve_node_impl."""

    def test_resolves_from_registry_with_config(self) -> None:
        manager = ExecutionManager(node_registry={"config_node": _ConfigNode})

        impl = manager._resolve_node_impl({"type": "config_node", "config": {"k": "v"}})

        assert isinstance(impl, _ConfigNode)
        assert impl.config == {"k": "v"}

    def test_falls_back_to_passthrough_for_unknown_type(self) -> None:
        from transformation_portal.execution_graph.nodes.base import PassthroughNode

        manager = ExecutionManager()

        impl = manager._resolve_node_impl({"type": "unknown"})

        assert isinstance(impl, PassthroughNode)


class TestGetNodeDeps:
    """Tests for _get_node_deps."""

    def test_extracts_dependencies_from_edges(self) -> None:
        manager = ExecutionManager()
        edges = [
            {"source": "a", "target": "c"},
            {"source": "b", "target": "c"},
            {"source": "a", "target": "b"},
        ]

        assert manager._get_node_deps("c", edges) == ["a", "b"]

    def test_returns_empty_for_root_node(self) -> None:
        manager = ExecutionManager()

        assert manager._get_node_deps("a", [{"source": "a", "target": "b"}]) == []


class TestAllocateRunId:
    """Tests for allocate_run_id collision avoidance."""

    def test_skips_ids_already_in_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from transformation_portal.dashboard import execution_manager

        ids = iter(["AAAAAAAA-collision", "BBBBBBBB-fresh"])

        def _fake_uuid4() -> str:
            return next(ids)

        monkeypatch.setattr(execution_manager.uuid, "uuid4", _fake_uuid4)
        manager = ExecutionManager()
        manager.run_history.append("AAAAAAAA")

        assert manager.allocate_run_id() == "BBBBBBBB"


class TestPrepareRunHistoryTrimming:
    """Tests for the history-trimming branches in prepare_run."""

    def test_keeps_non_terminal_run_during_trim(self) -> None:
        manager = ExecutionManager()
        manager._max_history = 1

        first = manager.allocate_run_id()
        manager.prepare_run(first, {"nodes": [], "edges": []})  # stays PENDING
        second = manager.allocate_run_id()
        manager.prepare_run(second, {"nodes": [], "edges": []})

        # The PENDING (non-terminal) run is not evicted.
        assert first in manager.active_runs
        assert second in manager.active_runs

    def test_keeps_run_with_live_task_during_trim(self) -> None:
        manager = ExecutionManager()
        manager._max_history = 1

        first = manager.allocate_run_id()
        manager.prepare_run(first, {"nodes": [], "edges": []})
        manager.active_runs[first].status = RunStatus.COMPLETE
        manager._tasks_by_run_id[first] = object()  # type: ignore[assignment]

        second = manager.allocate_run_id()
        manager.prepare_run(second, {"nodes": [], "edges": []})

        # A run with a live task is not evicted even though it is terminal.
        assert first in manager.active_runs

    def test_drops_history_id_missing_from_active_runs(self) -> None:
        manager = ExecutionManager()
        manager._max_history = 1
        manager.run_history.append("ghost")  # in history, never in active_runs

        run_id = manager.allocate_run_id()
        manager.prepare_run(run_id, {"nodes": [], "edges": []})

        assert "ghost" not in manager.run_history
        assert run_id in manager.active_runs


class TestStartPipelineBackgroundGuards:
    """Tests for start_pipeline_background duplicate protection and callbacks."""

    @pytest.mark.asyncio
    async def test_rejects_duplicate_run_id(self) -> None:
        manager = ExecutionManager()
        broadcast = AsyncMock()
        run_id = manager.allocate_run_id()
        pipeline = {"nodes": [{"id": "a", "type": "passthrough"}], "edges": []}

        task = manager.start_pipeline_background(run_id, pipeline, broadcast)
        try:
            with pytest.raises(ValueError, match="already registered"):
                manager.start_pipeline_background(run_id, pipeline, broadcast)
        finally:
            await task

    @pytest.mark.asyncio
    async def test_done_callback_clears_task_registry(self) -> None:
        manager = ExecutionManager()
        broadcast = AsyncMock()
        run_id = manager.allocate_run_id()

        task = manager.start_pipeline_background(
            run_id, {"nodes": [{"id": "a", "type": "passthrough"}], "edges": []}, broadcast
        )
        await task

        assert run_id not in manager._tasks_by_run_id

    @pytest.mark.asyncio
    async def test_done_callback_handles_cancellation(self) -> None:
        manager = ExecutionManager()
        broadcast = AsyncMock()
        run_id = manager.allocate_run_id()

        task = manager.start_pipeline_background(
            run_id, {"nodes": [{"id": "a", "type": "passthrough"}], "edges": []}, broadcast
        )
        await asyncio.sleep(0)  # let the task start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert run_id not in manager._tasks_by_run_id


class TestRunPipelineExecution:
    """Tests for the end-to-end run_pipeline / _execute_pipeline path."""

    @pytest.mark.asyncio
    async def test_run_pipeline_completes_successfully(self) -> None:
        manager = ExecutionManager()
        events: list[str] = []

        async def broadcast(msg: Dict[str, Any]) -> None:
            events.append(msg["type"])

        run_id = await manager.run_pipeline(
            {"nodes": [{"id": "a", "type": "passthrough"}], "edges": []},
            broadcast,
        )

        run = manager.get_run_state(run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETE
        assert run.nodes["a"].status == NodeStatus.COMPLETE
        assert "run_started" in events
        assert "run_complete" in events

    @pytest.mark.asyncio
    async def test_node_error_is_isolated_but_run_completes(self) -> None:
        manager = ExecutionManager(node_registry={"failing": _FailingNode})
        events: list[str] = []

        async def broadcast(msg: Dict[str, Any]) -> None:
            events.append(msg["type"])

        run_id = await manager.run_pipeline(
            {"nodes": [{"id": "a", "type": "failing"}], "edges": []},
            broadcast,
        )

        run = manager.get_run_state(run_id)
        assert run.nodes["a"].status == NodeStatus.ERROR
        assert "node boom" in run.nodes["a"].error
        assert "node_error" in events
        assert run.status == RunStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_run_level_error_marks_run_errored(self) -> None:
        manager = ExecutionManager()
        events: list[str] = []

        async def broadcast(msg: Dict[str, Any]) -> None:
            events.append(msg["type"])

        # A node definition missing "id" raises inside _execute_pipeline's
        # scheduler-build loop, exercising the run-level error handler.
        run_id = await manager.run_pipeline(
            {"nodes": [{"type": "passthrough"}], "edges": []},
            broadcast,
        )

        run = manager.get_run_state(run_id)
        assert run.status == RunStatus.ERROR
        assert run.error is not None
        assert "run_error" in events

    @pytest.mark.asyncio
    async def test_merkle_hash_recorded_when_dag_provided(self) -> None:
        from transformation_portal.storage.merkle_dag import MerkleDAG

        dag = MerkleDAG()
        manager = ExecutionManager(merkle_dag=dag)
        broadcast = AsyncMock()

        run_id = await manager.run_pipeline(
            {"nodes": [{"id": "a", "type": "passthrough"}], "edges": []},
            broadcast,
        )

        run = manager.get_run_state(run_id)
        assert run.nodes["a"].merkle_hash is not None
        assert dag.get_node(run.nodes["a"].merkle_hash) is not None
