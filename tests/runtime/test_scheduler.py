"""Tests for ClusterScheduler priority ordering and retry accounting."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from transformation_portal.runtime.scheduler import ClusterScheduler, TaskStatus

pytestmark = [pytest.mark.unit, pytest.mark.regression]


class _Node:
    """Stand-in DAG node class (never instantiated by the fake engine)."""


class _RecordingEngine:
    """Engine double that records the execution order of node ids."""

    def __init__(self, *, fail: bool = False) -> None:
        self.order: List[str] = []
        self.fail = fail

    def run_node(self, node_cls: type, *, inputs: Dict[str, Any], node_id: str, **_: Any) -> Tuple[str, Dict[str, Any]]:
        self.order.append(node_id)
        if self.fail:
            raise RuntimeError("node failed")
        return "h" * 64, {}


def test_priority_zero_is_not_replaced_by_default() -> None:
    engine = _RecordingEngine()
    scheduler = ClusterScheduler(engine, default_priority=10)  # type: ignore[arg-type]
    scheduler.submit(_Node, inputs={}, node_id="normal", priority=5)
    scheduler.submit(_Node, inputs={}, node_id="urgent", priority=0)

    scheduler.run_all()

    assert engine.order == ["urgent", "normal"]


def test_max_retries_zero_runs_exactly_once() -> None:
    engine = _RecordingEngine(fail=True)
    scheduler = ClusterScheduler(engine, default_max_retries=2)  # type: ignore[arg-type]
    scheduler.submit(_Node, inputs={}, node_id="once", max_retries=0)

    results = scheduler.run_all()

    assert engine.order == ["once"]
    assert [r.status for r in results] == [TaskStatus.FAILED]
    assert results[0].retries_used == 0


def test_omitted_options_still_use_defaults() -> None:
    engine = _RecordingEngine(fail=True)
    scheduler = ClusterScheduler(engine, default_max_retries=2)  # type: ignore[arg-type]
    scheduler.submit(_Node, inputs={}, node_id="flaky")

    scheduler.run_all()

    assert engine.order == ["flaky", "flaky", "flaky"]


def test_equal_priority_tasks_run_in_submission_order() -> None:
    engine = _RecordingEngine()
    scheduler = ClusterScheduler(engine)  # type: ignore[arg-type]
    names = list("ABCDEFGHIJ")
    for name in names:
        scheduler.submit(_Node, inputs={}, node_id=name)

    scheduler.run_all()

    assert engine.order == names
