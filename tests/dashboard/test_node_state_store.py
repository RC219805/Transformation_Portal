"""Behavioral coverage for ``dashboard.node_state_store``.

``NodeStateStore`` is a pure, deterministic per-node execution state tracker
with no external dependencies, which makes it an ideal first cold-zone target
for the dashboard package. These tests exercise the full lifecycle (init ->
status transitions -> inputs/outputs/artifacts/logs/metrics -> error/complete),
the history-trim eviction branch, the missing-run / missing-node no-op
branches, and the global-store singleton accessor. Every test is ``unit``,
offline, and sub-second.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.dashboard.node_state_store import (
    NodeExecutionState,
    NodeStateStore,
    RunExecutionState,
    get_store,
    set_store,
)


@pytest.fixture
def store() -> NodeStateStore:
    s = NodeStateStore()
    s.init_run("run_1", ["ingest", "segment", "export"], config={"tier": "apex"})
    return s


# --------------------------------------------------------------------------- #
# init / structure
# --------------------------------------------------------------------------- #


def test_init_run_creates_running_state_with_all_nodes(store: NodeStateStore) -> None:
    run = store.get_run("run_1")
    assert isinstance(run, RunExecutionState)
    assert run.status == "running"
    assert run.start_time is not None
    assert run.config == {"tier": "apex"}
    assert set(run.nodes) == {"ingest", "segment", "export"}
    assert all(isinstance(n, NodeExecutionState) for n in run.nodes.values())
    assert all(n.status == "idle" for n in run.nodes.values())


def test_init_run_without_config_defaults_to_empty_dict() -> None:
    store = NodeStateStore()
    store.init_run("r", ["a"])
    assert store.get_run("r").config == {}


def test_get_run_and_get_node_miss_returns_none(store: NodeStateStore) -> None:
    assert store.get_run("nope") is None
    assert store.get_node("nope", "ingest") is None
    assert store.get_node("run_1", "nope") is None


# --------------------------------------------------------------------------- #
# status transitions (timestamp side effects)
# --------------------------------------------------------------------------- #


def test_set_status_running_sets_start_time_once(store: NodeStateStore) -> None:
    store.set_status("run_1", "ingest", "running")
    first_start = store.get_node("run_1", "ingest").start_time
    assert first_start is not None

    # Re-entering running must not overwrite the original start_time.
    store.set_status("run_1", "ingest", "running")
    assert store.get_node("run_1", "ingest").start_time == first_start


@pytest.mark.parametrize("terminal", ["complete", "error"])
def test_set_status_terminal_sets_end_time_once(store: NodeStateStore, terminal: str) -> None:
    store.set_status("run_1", "ingest", terminal)
    node = store.get_node("run_1", "ingest")
    assert node.status == terminal
    end = node.end_time
    assert end is not None

    store.set_status("run_1", "ingest", terminal)
    assert store.get_node("run_1", "ingest").end_time == end


def test_set_status_on_missing_node_is_noop(store: NodeStateStore) -> None:
    # Must not raise; simply ignores the unknown target.
    store.set_status("run_1", "ghost", "running")
    store.set_status("ghost_run", "ingest", "running")


# --------------------------------------------------------------------------- #
# data mutations
# --------------------------------------------------------------------------- #


def test_update_inputs_outputs_metrics(store: NodeStateStore) -> None:
    store.update_inputs("run_1", "ingest", {"image": "a.png"})
    store.update_outputs("run_1", "ingest", {"rgb": [1, 2, 3]})
    store.update_metrics("run_1", "ingest", {"ms": 12})
    store.update_metrics("run_1", "ingest", {"mem": 4})  # merges, not replaces

    node = store.get_node("run_1", "ingest")
    assert node.inputs == {"image": "a.png"}
    assert node.outputs == {"rgb": [1, 2, 3]}
    assert node.metrics == {"ms": 12, "mem": 4}


def test_artifacts_bulk_update_and_single_add(store: NodeStateStore) -> None:
    store.update_artifacts("run_1", "ingest", {"depth": "sha-d"})
    store.add_artifact("run_1", "ingest", "mask", "sha-m")
    assert store.get_node("run_1", "ingest").artifacts == {"depth": "sha-d", "mask": "sha-m"}


def test_add_log_prefixes_timestamp(store: NodeStateStore) -> None:
    store.add_log("run_1", "ingest", "loaded")
    logs = store.get_node("run_1", "ingest").logs
    assert len(logs) == 1
    assert logs[0].startswith("[") and logs[0].endswith("] loaded")


def test_set_merkle_hash(store: NodeStateStore) -> None:
    store.set_merkle_hash("run_1", "ingest", "merkle-abc")
    assert store.get_node("run_1", "ingest").merkle_hash == "merkle-abc"


def test_data_mutations_on_missing_node_are_noops(store: NodeStateStore) -> None:
    # None of these should raise when the node does not exist.
    store.update_inputs("run_1", "ghost", {"x": 1})
    store.update_outputs("run_1", "ghost", {"x": 1})
    store.update_artifacts("run_1", "ghost", {"x": "y"})
    store.add_artifact("run_1", "ghost", "x", "y")
    store.add_log("run_1", "ghost", "msg")
    store.update_metrics("run_1", "ghost", {"x": 1})
    store.set_merkle_hash("run_1", "ghost", "h")
    store.set_error("run_1", "ghost", "boom")


# --------------------------------------------------------------------------- #
# error / completion
# --------------------------------------------------------------------------- #


def test_set_error_marks_error_status_and_end_time(store: NodeStateStore) -> None:
    store.set_error("run_1", "segment", "OOM")
    node = store.get_node("run_1", "segment")
    assert node.error == "OOM"
    assert node.status == "error"
    assert node.end_time is not None


def test_set_error_preserves_existing_end_time(store: NodeStateStore) -> None:
    store.set_status("run_1", "segment", "complete")
    original_end = store.get_node("run_1", "segment").end_time
    store.set_error("run_1", "segment", "late failure")
    assert store.get_node("run_1", "segment").end_time == original_end


def test_complete_run_sets_status_and_end_time(store: NodeStateStore) -> None:
    store.complete_run("run_1")
    run = store.get_run("run_1")
    assert run.status == "complete"
    assert run.end_time is not None


def test_complete_run_custom_status(store: NodeStateStore) -> None:
    store.complete_run("run_1", status="cancelled")
    assert store.get_run("run_1").status == "cancelled"


def test_complete_run_missing_run_is_noop(store: NodeStateStore) -> None:
    store.complete_run("ghost")  # must not raise


# --------------------------------------------------------------------------- #
# summaries + history trimming
# --------------------------------------------------------------------------- #


def test_get_all_runs_summary_shape(store: NodeStateStore) -> None:
    summaries = store.get_all_runs()
    assert summaries == [
        {
            "run_id": "run_1",
            "status": "running",
            "start_time": store.get_run("run_1").start_time,
            "end_time": None,
            "node_count": 3,
        }
    ]


def test_history_trim_evicts_oldest_runs() -> None:
    store = NodeStateStore(max_runs=2)
    store.init_run("a", ["n"])
    store.init_run("b", ["n"])
    store.init_run("c", ["n"])  # exceeds max_runs -> "a" evicted

    assert store.get_run("a") is None
    assert store.get_run("b") is not None
    assert store.get_run("c") is not None
    assert store.run_history == ["b", "c"]


# --------------------------------------------------------------------------- #
# dataclass serialization + global singleton
# --------------------------------------------------------------------------- #


def test_node_execution_state_to_dict_roundtrip() -> None:
    node = NodeExecutionState(node_id="x", status="running")
    node.inputs = {"a": 1}
    node.artifacts = {"out": "sha"}
    node.logs = ["hi"]
    d = node.to_dict()
    assert d["node_id"] == "x"
    assert d["status"] == "running"
    assert d["inputs"] == {"a": 1}
    assert d["artifacts"] == {"out": "sha"}
    assert d["logs"] == ["hi"]
    # All documented keys are present.
    assert set(d) == {
        "node_id",
        "status",
        "inputs",
        "outputs",
        "artifacts",
        "logs",
        "metrics",
        "error",
        "start_time",
        "end_time",
        "merkle_hash",
    }


def test_global_store_is_lazy_singleton_and_settable() -> None:
    original = get_store()
    try:
        assert get_store() is original  # stable across calls

        replacement = NodeStateStore()
        set_store(replacement)
        assert get_store() is replacement
    finally:
        set_store(original)
