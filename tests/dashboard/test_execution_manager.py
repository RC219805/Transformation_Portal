"""Behavioral coverage for ``dashboard.execution_manager``.

``ExecutionManager`` drives async pipeline execution with event streaming.
These tests isolate it against a fake broadcast sink and tiny passthrough
pipelines so the deterministic seams are exercised without a FastAPI app,
WebSocket, or ML backend:

- run-id allocation + collision avoidance
- ``prepare_run`` registration and the duplicate-registration guard
- history trimming (evict terminal runs, retain active / live-task runs)
- dependency extraction + node-impl resolution (registry hit vs passthrough)
- a full happy-path execution (run/node lifecycle events, Merkle wiring)
- per-node error isolation (one node fails, the run still completes)
- cooperative cancellation before any node starts
- the ``cancel_run`` status state machine (idempotent + terminal cases)

``asyncio_mode = auto`` (pyproject) lets the ``async def`` tests run directly.
The only real waits are ``ExecutionManager``'s built-in 0.1s per-node UI delay,
so pipelines are kept to one or two nodes to stay sub-second.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.dashboard.execution_manager import (
    ExecutionManager,
    NodeStatus,
    RunStatus,
)
from transformation_portal.execution_graph.nodes.base import DAGNode, NodeResult


class _Recorder:
    """Async broadcast sink that records every emitted event."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def __call__(self, message: Dict[str, Any]) -> None:
        self.events.append(message)

    def types(self) -> List[str]:
        return [e["type"] for e in self.events]


class _DictNode(DAGNode):
    """Registry node returning a bare dict (exercises the dict branch)."""

    def __init__(self, value: int = 1) -> None:
        self.value = value

    def run(self, **inputs: Any) -> Dict[str, Any]:  # type: ignore[override]
        return {"value": self.value, "seen": sorted(inputs)}


class _BoomNode(DAGNode):
    """Registry node that always raises (exercises the node_error branch)."""

    def run(self, **inputs: Any) -> NodeResult:
        raise RuntimeError("kaboom")


class _ScalarNode(DAGNode):
    """Registry node returning a bare scalar (exercises the wrap branch)."""

    def run(self, **inputs: Any) -> int:  # type: ignore[override]
        return 42


def _linear_pipeline() -> Dict[str, Any]:
    return {
        "nodes": [
            {"id": "a", "type": "passthrough"},
            {"id": "b", "type": "passthrough"},
        ],
        "edges": [{"source": "a", "target": "b"}],
    }


@pytest.fixture
def manager() -> ExecutionManager:
    return ExecutionManager()


# --------------------------------------------------------------------------- #
# run-id allocation + prepare_run
# --------------------------------------------------------------------------- #


def test_allocate_run_id_is_unique_and_avoids_collisions(manager: ExecutionManager) -> None:
    a = manager.allocate_run_id()
    # Pre-seed both the active set and history to force the collision loop.
    manager.active_runs[a] = manager.prepare_run(a, {"nodes": []})
    b = manager.allocate_run_id()
    assert b != a
    assert b not in manager.active_runs
    assert b not in manager.run_history


def test_prepare_run_registers_pending_nodes(manager: ExecutionManager) -> None:
    run_id = manager.allocate_run_id()
    state = manager.prepare_run(run_id, _linear_pipeline())

    assert state.status == RunStatus.PENDING
    assert set(state.nodes) == {"a", "b"}
    assert all(n.status == NodeStatus.PENDING for n in state.nodes.values())
    assert manager.get_run_state(run_id) is state
    assert run_id in manager.run_history


def test_prepare_run_skips_nodes_without_id(manager: ExecutionManager) -> None:
    state = manager.prepare_run("r", {"nodes": [{"type": "passthrough"}, {"id": "ok"}]})
    assert set(state.nodes) == {"ok"}


def test_prepare_run_duplicate_registration_raises(manager: ExecutionManager) -> None:
    manager.prepare_run("dup", {"nodes": []})
    with pytest.raises(ValueError, match="already registered"):
        manager.prepare_run("dup", {"nodes": []})


# --------------------------------------------------------------------------- #
# history trimming branches
# --------------------------------------------------------------------------- #


def test_history_trim_evicts_terminal_runs(manager: ExecutionManager) -> None:
    manager._max_history = 1
    manager.prepare_run("old", {"nodes": []})
    manager.active_runs["old"].status = RunStatus.COMPLETE  # terminal -> evictable

    manager.prepare_run("new", {"nodes": []})

    assert "old" not in manager.active_runs
    assert manager.run_history == ["new"]


def test_history_trim_retains_active_runs(manager: ExecutionManager) -> None:
    manager._max_history = 1
    manager.prepare_run("active", {"nodes": []})  # left PENDING (non-terminal)

    manager.prepare_run("new", {"nodes": []})

    # Non-terminal run is re-queued, not evicted; trimming stops.
    assert "active" in manager.active_runs
    assert "active" in manager.run_history


def test_history_trim_retains_runs_with_live_task(manager: ExecutionManager) -> None:
    manager._max_history = 1
    manager.prepare_run("busy", {"nodes": []})
    manager.active_runs["busy"].status = RunStatus.COMPLETE
    manager._tasks_by_run_id["busy"] = object()  # type: ignore[assignment]

    manager.prepare_run("new", {"nodes": []})

    # A live task pins the run in active_runs even though it is terminal.
    assert "busy" in manager.active_runs


# --------------------------------------------------------------------------- #
# dependency + node-impl resolution
# --------------------------------------------------------------------------- #


def test_get_node_deps_filters_by_target(manager: ExecutionManager) -> None:
    edges = [
        {"source": "a", "target": "c"},
        {"source": "b", "target": "c"},
        {"source": "a", "target": "d"},
        {"target": "c"},  # missing source is ignored
    ]
    assert sorted(manager._get_node_deps("c", edges)) == ["a", "b"]
    assert manager._get_node_deps("z", edges) == []


def test_resolve_node_impl_uses_registry_with_config() -> None:
    manager = ExecutionManager(node_registry={"dict": _DictNode})
    impl = manager._resolve_node_impl({"type": "dict", "config": {"value": 9}})
    assert isinstance(impl, _DictNode)
    assert impl.value == 9


def test_resolve_node_impl_falls_back_to_passthrough(manager: ExecutionManager) -> None:
    from transformation_portal.execution_graph.nodes.base import PassthroughNode

    impl = manager._resolve_node_impl({"id": "x"})  # no type, not in registry
    assert isinstance(impl, PassthroughNode)


# --------------------------------------------------------------------------- #
# full execution lifecycle
# --------------------------------------------------------------------------- #


async def test_run_pipeline_happy_path_completes_all_nodes(manager: ExecutionManager) -> None:
    recorder = _Recorder()
    run_id = await manager.run_pipeline(_linear_pipeline(), recorder)

    state = manager.get_run_state(run_id)
    assert state.status == RunStatus.COMPLETE
    assert state.end_time is not None
    assert state.current_node_id is None
    assert all(n.status == NodeStatus.COMPLETE for n in state.nodes.values())
    assert all(n.progress == 100 for n in state.nodes.values())

    types = recorder.types()
    assert types[0] == "run_started"
    assert "execution_plan" in types
    assert types.count("node_start") == 2
    assert types.count("node_complete") == 2
    assert types[-1] == "run_complete"


async def test_run_pipeline_records_merkle_hashes_when_dag_present() -> None:
    class _FakeMerkle:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def add_computation(self, *, node_id: str, inputs: Any, outputs: Any, metadata: Any) -> str:
            self.calls.append(node_id)
            return f"merkle-{node_id}"

    merkle = _FakeMerkle()
    manager = ExecutionManager(merkle_dag=merkle)
    run_id = await manager.run_pipeline(_linear_pipeline(), _Recorder())

    state = manager.get_run_state(run_id)
    assert merkle.calls == ["a", "b"]
    assert state.nodes["a"].merkle_hash == "merkle-a"
    assert state.nodes["b"].merkle_hash == "merkle-b"


async def test_node_error_is_isolated_and_run_still_completes() -> None:
    manager = ExecutionManager(node_registry={"boom": _BoomNode})
    pipeline = {
        "nodes": [
            {"id": "ok", "type": "passthrough"},
            {"id": "bad", "type": "boom"},
        ],
        "edges": [{"source": "ok", "target": "bad"}],
    }
    recorder = _Recorder()
    run_id = await manager.run_pipeline(pipeline, recorder)

    state = manager.get_run_state(run_id)
    assert state.status == RunStatus.COMPLETE  # errors do not abort the run
    assert state.nodes["ok"].status == NodeStatus.COMPLETE
    assert state.nodes["bad"].status == NodeStatus.ERROR
    assert "RuntimeError: kaboom" in state.nodes["bad"].error
    assert "node_error" in recorder.types()
    # Failed node is excluded from the broadcast results payload.
    complete_event = next(e for e in recorder.events if e["type"] == "run_complete")
    assert "bad" not in complete_event["results"]


async def test_scalar_node_output_is_wrapped(manager: ExecutionManager) -> None:
    manager = ExecutionManager(node_registry={"scalar": _ScalarNode})
    pipeline = {"nodes": [{"id": "s", "type": "scalar"}], "edges": []}
    run_id = await manager.run_pipeline(pipeline, _Recorder())

    state = manager.get_run_state(run_id)
    assert state.status == RunStatus.COMPLETE
    assert state.nodes["s"].outputs == {"result": 42}


async def test_run_error_is_broadcast_when_scheduling_raises(manager: ExecutionManager) -> None:
    # A node dict missing the "id" key survives run-state init (which skips it)
    # but trips the scheduler-build loop, exercising the outer error handler.
    pipeline = {"nodes": [{"type": "passthrough"}], "edges": []}
    run_id = manager.allocate_run_id()
    manager.active_runs[run_id] = manager.prepare_run(run_id, {"nodes": []})

    recorder = _Recorder()
    await manager._execute_pipeline(run_id, pipeline, recorder)

    state = manager.get_run_state(run_id)
    assert state.status == RunStatus.ERROR
    assert state.error is not None
    error_event = next(e for e in recorder.events if e["type"] == "run_error")
    assert "traceback" in error_event


async def test_start_pipeline_background_returns_queryable_task(manager: ExecutionManager) -> None:
    run_id = manager.allocate_run_id()
    recorder = _Recorder()
    task = manager.start_pipeline_background(run_id, _linear_pipeline(), recorder)

    # Run is immediately queryable before the task finishes.
    assert manager.get_run_state(run_id) is not None
    await task

    assert manager.get_run_state(run_id).status == RunStatus.COMPLETE
    # Done-callback clears the task registry.
    assert run_id not in manager._tasks_by_run_id


async def test_start_pipeline_background_rejects_duplicate_task(manager: ExecutionManager) -> None:
    run_id = manager.allocate_run_id()
    task = manager.start_pipeline_background(run_id, _linear_pipeline(), _Recorder())
    try:
        with pytest.raises(ValueError, match="already registered"):
            manager.start_pipeline_background(run_id, _linear_pipeline(), _Recorder())
    finally:
        await task


async def test_cancellation_before_first_node_skips_all(manager: ExecutionManager) -> None:
    run_id = manager.allocate_run_id()
    manager.prepare_run(run_id, _linear_pipeline())
    manager.active_runs[run_id].cancel_requested = True

    recorder = _Recorder()
    await manager._execute_pipeline(run_id, _linear_pipeline(), recorder)

    state = manager.get_run_state(run_id)
    assert state.status == RunStatus.CANCELLED
    assert all(n.status == NodeStatus.SKIPPED for n in state.nodes.values())
    assert "run_cancelled" in recorder.types()
    assert recorder.types().count("node_skipped") == 2
    assert "run_complete" not in recorder.types()


# --------------------------------------------------------------------------- #
# cancel_run state machine + summaries
# --------------------------------------------------------------------------- #


async def test_cancel_run_unknown_returns_none(manager: ExecutionManager) -> None:
    assert await manager.cancel_run("ghost", _Recorder()) is None


async def test_cancel_run_requests_cancellation_then_is_idempotent(manager: ExecutionManager) -> None:
    manager.prepare_run("r", _linear_pipeline())  # PENDING
    recorder = _Recorder()

    first = await manager.cancel_run("r", recorder)
    assert first == RunStatus.CANCELLING.value
    assert manager.get_run_state("r").cancel_requested is True
    assert "run_cancelling" in recorder.types()

    # Second call on an already-cancelling run is idempotent (no new request).
    second = await manager.cancel_run("r", recorder)
    assert second == RunStatus.CANCELLING.value


@pytest.mark.parametrize(
    "terminal",
    [RunStatus.COMPLETE, RunStatus.ERROR, RunStatus.CANCELLED],
)
async def test_cancel_run_on_terminal_returns_current_status(manager: ExecutionManager, terminal: RunStatus) -> None:
    manager.prepare_run("r", {"nodes": []})
    manager.active_runs["r"].status = terminal
    assert await manager.cancel_run("r", _Recorder()) == terminal.value


def test_get_active_runs_summary_shape(manager: ExecutionManager) -> None:
    manager.prepare_run("r", _linear_pipeline())
    summaries = manager.get_active_runs()
    assert summaries == [
        {
            "run_id": "r",
            "status": RunStatus.PENDING.value,
            "start_time": None,
            "node_count": 2,
        }
    ]
