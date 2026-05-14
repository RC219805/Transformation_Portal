"""Unit tests for the dashboard node state store.

Covers NodeExecutionState / RunExecutionState dataclasses, the NodeStateStore
lifecycle (init/get/mutate/complete), history trimming, and the global
store accessors.
"""

from __future__ import annotations

import pytest

from transformation_portal.dashboard.node_state_store import (
    NodeExecutionState,
    NodeStateStore,
    RunExecutionState,
    get_store,
    set_store,
)

pytestmark = pytest.mark.unit


class TestNodeExecutionState:
    """Tests for the NodeExecutionState dataclass."""

    def test_defaults(self) -> None:
        node = NodeExecutionState(node_id="ingest")

        assert node.node_id == "ingest"
        assert node.status == "idle"
        assert node.inputs == {}
        assert node.outputs == {}
        assert node.artifacts == {}
        assert node.logs == []
        assert node.metrics == {}
        assert node.error is None
        assert node.start_time is None
        assert node.end_time is None
        assert node.merkle_hash is None

    def test_to_dict_roundtrips_all_fields(self) -> None:
        node = NodeExecutionState(
            node_id="segment",
            status="complete",
            inputs={"image": "path"},
            outputs={"mask": "data"},
            artifacts={"mask.png": "abc123"},
            logs=["[ts] done"],
            metrics={"ms": 12},
            error=None,
            start_time="t0",
            end_time="t1",
            merkle_hash="deadbeef",
        )

        payload = node.to_dict()

        assert payload == {
            "node_id": "segment",
            "status": "complete",
            "inputs": {"image": "path"},
            "outputs": {"mask": "data"},
            "artifacts": {"mask.png": "abc123"},
            "logs": ["[ts] done"],
            "metrics": {"ms": 12},
            "error": None,
            "start_time": "t0",
            "end_time": "t1",
            "merkle_hash": "deadbeef",
        }

    def test_default_collections_are_not_shared(self) -> None:
        first = NodeExecutionState(node_id="a")
        second = NodeExecutionState(node_id="b")

        first.logs.append("only-a")

        assert second.logs == []


class TestRunExecutionState:
    """Tests for the RunExecutionState dataclass."""

    def test_defaults(self) -> None:
        run = RunExecutionState(run_id="run_1")

        assert run.run_id == "run_1"
        assert run.status == "pending"
        assert run.nodes == {}
        assert run.start_time is None
        assert run.end_time is None
        assert run.config == {}


class TestInitAndGet:
    """Tests for run initialization and lookup."""

    def test_init_run_creates_running_state_with_nodes(self) -> None:
        store = NodeStateStore()

        store.init_run("run_1", ["ingest", "segment"], config={"preset": "apex"})

        run = store.get_run("run_1")
        assert run is not None
        assert run.status == "running"
        assert run.start_time is not None
        assert run.config == {"preset": "apex"}
        assert set(run.nodes) == {"ingest", "segment"}
        assert run.nodes["ingest"].status == "idle"

    def test_init_run_without_config_uses_empty_dict(self) -> None:
        store = NodeStateStore()

        store.init_run("run_1", ["only"])

        run = store.get_run("run_1")
        assert run is not None
        assert run.config == {}

    def test_get_run_returns_none_for_unknown(self) -> None:
        store = NodeStateStore()

        assert store.get_run("missing") is None

    def test_get_node_returns_state(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        node = store.get_node("run_1", "ingest")

        assert node is not None
        assert node.node_id == "ingest"

    def test_get_node_returns_none_for_unknown_run(self) -> None:
        store = NodeStateStore()

        assert store.get_node("missing", "ingest") is None

    def test_get_node_returns_none_for_unknown_node(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        assert store.get_node("run_1", "missing") is None


class TestHistoryTrimming:
    """Tests for the bounded run history."""

    def test_trims_oldest_runs_beyond_max(self) -> None:
        store = NodeStateStore(max_runs=2)

        store.init_run("run_1", ["n"])
        store.init_run("run_2", ["n"])
        store.init_run("run_3", ["n"])

        assert store.get_run("run_1") is None
        assert store.get_run("run_2") is not None
        assert store.get_run("run_3") is not None
        assert store.run_history == ["run_2", "run_3"]


class TestStatusTransitions:
    """Tests for set_status timing semantics."""

    def test_set_status_running_sets_start_time_once(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.set_status("run_1", "ingest", "running")
        first_start = store.get_node("run_1", "ingest").start_time
        assert first_start is not None

        store.set_status("run_1", "ingest", "running")
        assert store.get_node("run_1", "ingest").start_time == first_start

    def test_set_status_complete_sets_end_time(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.set_status("run_1", "ingest", "complete")

        node = store.get_node("run_1", "ingest")
        assert node.status == "complete"
        assert node.end_time is not None

    def test_set_status_error_sets_end_time(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.set_status("run_1", "ingest", "error")

        assert store.get_node("run_1", "ingest").end_time is not None

    def test_set_status_on_missing_node_is_noop(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.set_status("run_1", "missing", "running")  # must not raise


class TestNodeMutations:
    """Tests for input/output/artifact/log/metric mutators."""

    def test_update_inputs_and_outputs(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.update_inputs("run_1", "ingest", {"image": "img.png"})
        store.update_outputs("run_1", "ingest", {"rgb": "array"})

        node = store.get_node("run_1", "ingest")
        assert node.inputs == {"image": "img.png"}
        assert node.outputs == {"rgb": "array"}

    def test_update_artifacts_merges(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.update_artifacts("run_1", "ingest", {"a.png": "h1"})
        store.update_artifacts("run_1", "ingest", {"b.png": "h2"})

        assert store.get_node("run_1", "ingest").artifacts == {"a.png": "h1", "b.png": "h2"}

    def test_add_artifact_sets_single_entry(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.add_artifact("run_1", "ingest", "mask.png", "hashval")

        assert store.get_node("run_1", "ingest").artifacts["mask.png"] == "hashval"

    def test_add_log_appends_timestamped_message(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.add_log("run_1", "ingest", "loaded image")

        logs = store.get_node("run_1", "ingest").logs
        assert len(logs) == 1
        assert logs[0].endswith("loaded image")
        assert logs[0].startswith("[")

    def test_update_metrics_merges(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.update_metrics("run_1", "ingest", {"ms": 10})
        store.update_metrics("run_1", "ingest", {"mb": 64})

        assert store.get_node("run_1", "ingest").metrics == {"ms": 10, "mb": 64}

    def test_set_error_marks_node_failed(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.set_error("run_1", "ingest", "boom")

        node = store.get_node("run_1", "ingest")
        assert node.error == "boom"
        assert node.status == "error"
        assert node.end_time is not None

    def test_set_error_preserves_existing_end_time(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])
        store.set_status("run_1", "ingest", "complete")
        existing_end = store.get_node("run_1", "ingest").end_time

        store.set_error("run_1", "ingest", "late failure")

        assert store.get_node("run_1", "ingest").end_time == existing_end

    def test_set_merkle_hash(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.set_merkle_hash("run_1", "ingest", "merkle123")

        assert store.get_node("run_1", "ingest").merkle_hash == "merkle123"

    def test_mutators_on_missing_node_are_noops(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        # None of these should raise for an unknown node.
        store.update_inputs("run_1", "missing", {"x": 1})
        store.update_outputs("run_1", "missing", {"x": 1})
        store.update_artifacts("run_1", "missing", {"x": "h"})
        store.add_artifact("run_1", "missing", "x", "h")
        store.add_log("run_1", "missing", "msg")
        store.update_metrics("run_1", "missing", {"x": 1})
        store.set_error("run_1", "missing", "err")
        store.set_merkle_hash("run_1", "missing", "h")


class TestRunCompletionAndSummary:
    """Tests for complete_run and get_all_runs."""

    def test_complete_run_sets_status_and_end_time(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.complete_run("run_1")

        run = store.get_run("run_1")
        assert run.status == "complete"
        assert run.end_time is not None

    def test_complete_run_accepts_custom_status(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["ingest"])

        store.complete_run("run_1", status="failed")

        assert store.get_run("run_1").status == "failed"

    def test_complete_run_on_missing_run_is_noop(self) -> None:
        store = NodeStateStore()

        store.complete_run("missing")  # must not raise

    def test_get_all_runs_returns_summaries(self) -> None:
        store = NodeStateStore()
        store.init_run("run_1", ["a", "b"])
        store.init_run("run_2", ["c"])

        summaries = store.get_all_runs()

        by_id = {s["run_id"]: s for s in summaries}
        assert by_id["run_1"]["node_count"] == 2
        assert by_id["run_2"]["node_count"] == 1
        assert by_id["run_1"]["status"] == "running"
        assert "start_time" in by_id["run_1"]
        assert "end_time" in by_id["run_1"]


class TestGlobalStore:
    """Tests for the module-level global store accessors."""

    def test_get_store_returns_singleton(self) -> None:
        set_store(NodeStateStore())  # reset to a known state

        first = get_store()
        second = get_store()

        assert first is second

    def test_set_store_replaces_global(self) -> None:
        replacement = NodeStateStore(max_runs=7)
        set_store(replacement)

        assert get_store() is replacement
        assert get_store().max_runs == 7
