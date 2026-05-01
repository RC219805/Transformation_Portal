"""Tests for execution_graph.scheduler — PriorityDAGScheduler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from transformation_portal.execution_graph.scheduler import (
    PriorityDAGScheduler,
    ResourceRequirements,
    SchedulerError,
)

pytestmark = pytest.mark.unit


def _make_node(return_value=None):
    node = MagicMock()
    node.run.return_value = return_value
    return node


class TestSchedulerAddNode:
    def test_add_node_succeeds(self):
        """add_node() does not raise for a new node."""
        s = PriorityDAGScheduler()
        s.add_node("a", _make_node())

    def test_add_duplicate_node_raises(self):
        """Adding the same node_id twice raises SchedulerError."""
        s = PriorityDAGScheduler()
        s.add_node("a", _make_node())
        with pytest.raises(SchedulerError, match="already exists"):
            s.add_node("a", _make_node())

    def test_node_stored_in_nodes_dict(self):
        """After add_node(), node_id appears in scheduler.nodes."""
        s = PriorityDAGScheduler()
        s.add_node("mynode", _make_node())
        assert "mynode" in s.nodes

    def test_add_node_with_deps_stores_deps(self):
        """Dependencies are stored on the ScheduledNode."""
        s = PriorityDAGScheduler()
        s.add_node("a", _make_node())
        s.add_node("b", _make_node(), deps=["a"])
        assert "a" in s.nodes["b"].deps


class TestSchedulerValidate:
    def test_empty_graph_valid(self):
        """validate() returns empty list for an empty graph."""
        s = PriorityDAGScheduler()
        assert not s.validate()

    def test_single_node_valid(self):
        """A single node with no deps is valid."""
        s = PriorityDAGScheduler()
        s.add_node("a", _make_node())
        assert not s.validate()

    def test_missing_dep_reported(self):
        """A dependency on an unknown node generates a validation error."""
        s = PriorityDAGScheduler()
        s.add_node("b", _make_node(), deps=["missing"])
        errors = s.validate()
        assert len(errors) > 0
        assert any("missing" in e for e in errors)

    def test_cycle_detected(self):
        """A→B→A cycle produces a validation error."""
        s = PriorityDAGScheduler()
        s.add_node("a", _make_node(), deps=["b"])
        s.add_node("b", _make_node(), deps=["a"])
        errors = s.validate()
        assert len(errors) > 0

    def test_valid_linear_chain_ok(self):
        """A→B→C is a valid DAG (no errors)."""
        s = PriorityDAGScheduler()
        s.add_node("a", _make_node())
        s.add_node("b", _make_node(), deps=["a"])
        s.add_node("c", _make_node(), deps=["b"])
        assert not s.validate()


class TestSchedulerRun:
    def test_single_node_runs(self):
        """A single node is executed and its result returned."""
        s = PriorityDAGScheduler()
        node = _make_node(return_value="output_a")
        s.add_node("a", node)
        results = s.run()
        assert results["a"] == "output_a"
        node.run.assert_called_once()

    def test_linear_chain_executes_in_order(self):
        """A→B→C: node A runs before B before C."""
        call_order = []
        s = PriorityDAGScheduler()

        def make_ordered_node(name, rv=None):
            n = MagicMock()
            n.run.side_effect = lambda **kw: call_order.append(name) or rv
            return n

        s.add_node("a", make_ordered_node("a", "ra"))
        s.add_node("b", make_ordered_node("b", "rb"), deps=["a"])
        s.add_node("c", make_ordered_node("c", "rc"), deps=["b"])
        s.run()
        assert call_order == ["a", "b", "c"]

    def test_priority_respected_for_independent_nodes(self):
        """Higher-priority root node is executed first."""
        call_order = []
        s = PriorityDAGScheduler()

        def ordered(name):
            n = MagicMock()
            n.run.side_effect = lambda **kw: call_order.append(name)
            return n

        s.add_node("low", ordered("low"), priority=1)
        s.add_node("high", ordered("high"), priority=10)
        s.run()
        assert call_order.index("high") < call_order.index("low")

    def test_dep_output_available_to_dependent(self):
        """Node B receives node A's result via keyword argument."""
        received = {}
        s = PriorityDAGScheduler()

        node_a = _make_node(return_value="value_from_a")
        node_b = MagicMock()

        def b_run(**kwargs):
            received.update(kwargs)
            return "b_done"

        node_b.run.side_effect = b_run
        s.add_node("a", node_a)
        s.add_node("b", node_b, deps=["a"])
        s.run()
        assert received.get("a") == "value_from_a"

    def test_failed_node_raises_scheduler_error(self):
        """If a node's run() raises, SchedulerError is re-raised."""
        s = PriorityDAGScheduler()
        node = MagicMock()
        node.run.side_effect = RuntimeError("boom")
        s.add_node("a", node)
        with pytest.raises(SchedulerError, match="boom"):
            s.run()

    def test_invalid_dag_raises_before_run(self):
        """Validation errors cause SchedulerError before any node runs."""
        s = PriorityDAGScheduler()
        s.add_node("b", _make_node(), deps=["missing"])
        with pytest.raises(SchedulerError, match="validation failed"):
            s.run()

    def test_all_nodes_produce_results(self):
        """All node IDs appear in the returned results dict."""
        s = PriorityDAGScheduler()
        s.add_node("x", _make_node(return_value=1))
        s.add_node("y", _make_node(return_value=2))
        results = s.run()
        assert "x" in results and "y" in results


class TestGetExecutionOrder:
    def test_order_respects_deps(self):
        """get_execution_order returns dep before dependent."""
        s = PriorityDAGScheduler()
        s.add_node("a", _make_node())
        s.add_node("b", _make_node(), deps=["a"])
        order = s.get_execution_order()
        assert order.index("a") < order.index("b")

    def test_all_nodes_included(self):
        """Every node_id appears in the execution order."""
        s = PriorityDAGScheduler()
        for name in ["x", "y", "z"]:
            s.add_node(name, _make_node())
        order = s.get_execution_order()
        assert set(order) == {"x", "y", "z"}

    def test_empty_scheduler_returns_empty_list(self):
        """No nodes → empty order."""
        assert not PriorityDAGScheduler().get_execution_order()


class TestGetResourceSummary:
    def test_counts_gpu_nodes(self):
        """GPU node count is correct."""
        s = PriorityDAGScheduler()
        s.add_node("cpu", _make_node())
        s.add_node("gpu", _make_node(), resources=ResourceRequirements(gpu=True, gpu_memory_mb=4000))
        summary = s.get_resource_summary()
        assert summary["gpu_nodes"] == 1
        assert summary["cpu_only_nodes"] == 1

    def test_sums_gpu_memory(self):
        """total_gpu_memory_mb sums all GPU nodes."""
        s = PriorityDAGScheduler()
        s.add_node("g1", _make_node(), resources=ResourceRequirements(gpu=True, gpu_memory_mb=2000))
        s.add_node("g2", _make_node(), resources=ResourceRequirements(gpu=True, gpu_memory_mb=3000))
        summary = s.get_resource_summary()
        assert summary["total_gpu_memory_mb"] == 5000

    def test_total_nodes_correct(self):
        """total_nodes equals the number of added nodes."""
        s = PriorityDAGScheduler()
        for i in range(4):
            s.add_node(str(i), _make_node())
        assert s.get_resource_summary()["total_nodes"] == 4

    def test_empty_scheduler_summary_zeros(self):
        """Empty scheduler: all counts are zero."""
        summary = PriorityDAGScheduler().get_resource_summary()
        assert summary["total_nodes"] == 0
        assert summary["gpu_nodes"] == 0


class TestGPUPoolIntegration:
    def test_gpu_node_routes_to_pool(self):
        """A GPU-required node is routed through the gpu_pool."""
        gpu_pool = MagicMock()
        gpu_pool.get.return_value = "gpu_result"

        s = PriorityDAGScheduler(gpu_pool=gpu_pool)
        s.add_node(
            "gpu_node",
            _make_node(),
            resources=ResourceRequirements(gpu=True),
        )
        results = s.run()
        gpu_pool.submit.assert_called_once()
        assert results["gpu_node"] == "gpu_result"

    def test_cpu_node_does_not_use_pool(self):
        """A CPU-only node does not call gpu_pool.submit."""
        gpu_pool = MagicMock()
        s = PriorityDAGScheduler(gpu_pool=gpu_pool)
        s.add_node("cpu", _make_node(return_value="ok"))
        s.run()
        gpu_pool.submit.assert_not_called()
