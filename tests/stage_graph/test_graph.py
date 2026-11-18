"""Tests for StageGraph construction and execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from transformation_portal.stage_graph.graph import StageGraph
from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus

pytestmark = pytest.mark.unit


class _ScriptedStage(Stage):
    """Stage with a fixed outcome that records the artifacts it was handed."""

    def __init__(
        self,
        name: str,
        *,
        deps: Optional[List[str]] = None,
        status: StageStatus = StageStatus.COMPLETED,
        artifacts: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        super().__init__(name)
        self._deps = list(deps or [])
        self._status = status
        self._artifacts = dict(artifacts or {})
        self._error = error
        self.calls = 0
        self.seen: Optional[Dict[str, Any]] = None

    def get_dependencies(self) -> List[str]:
        return list(self._deps)

    def get_cache_key(self, context: StageContext) -> str:
        return f"{self.name}-key"

    def compute(self, context: StageContext) -> StageResult:
        self.calls += 1
        self.seen = dict(context.artifacts)
        return StageResult(self.name, self.version, self._status, artifacts=dict(self._artifacts), error=self._error)


def _pipeline(middle: _ScriptedStage) -> tuple[StageGraph, _ScriptedStage]:
    graph = StageGraph("scripted")
    graph.add_stage(_ScriptedStage("source", artifacts={"image": "pixels"}))
    graph.add_stage(middle)
    sink = _ScriptedStage("sink", deps=[middle.name], artifacts={"report": "done"})
    graph.add_stage(sink)
    return graph, sink


def test_stage_rejected_for_a_missing_dependency_is_not_left_in_the_graph():
    graph = StageGraph("partial")
    source = _ScriptedStage("source", artifacts={"image": "pixels"})
    graph.add_stage(source)

    with pytest.raises(ValueError, match="depends on 'not_installed', but 'not_installed' is not in graph"):
        graph.add_stage(_ScriptedStage("optional_extra", deps=["not_installed"]))

    assert list(graph.stages) == ["source"]
    assert graph.get_execution_order() == ["source"]
    execution = graph.execute(StageContext(cache_enabled=False), parallel=False)
    assert execution.success is True and execution.error is None
    assert source.calls == 1


def test_rejected_stage_can_be_added_once_its_dependency_exists():
    graph = StageGraph("retry")
    extra = _ScriptedStage("optional_extra", deps=["backend"])
    with pytest.raises(ValueError, match="is not in graph"):
        graph.add_stage(extra)

    graph.add_stage(_ScriptedStage("backend"))
    graph.add_stage(extra)

    assert graph.get_execution_order() == ["backend", "optional_extra"]


def test_stage_that_depends_on_itself_is_rejected_when_added():
    graph = StageGraph("self-cycle")

    with pytest.raises(ValueError, match="depends on 'loop'"):
        graph.add_stage(_ScriptedStage("loop", deps=["loop"]))

    assert graph.stages == {}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (StageStatus.COMPLETED, True),
        (StageStatus.CACHED, True),
        (StageStatus.SKIPPED, True),
        (StageStatus.FAILED, False),
        (StageStatus.UNAVAILABLE, False),
        (StageStatus.PENDING, False),
        (StageStatus.RUNNING, False),
    ],
)
def test_stage_result_success_by_status(status: StageStatus, expected: bool):
    assert StageResult("stage", "1.0.0", status).is_success() is expected


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
def test_skipped_stage_passes_its_artifacts_on_instead_of_failing_the_pipeline(parallel: bool):
    skipped = _ScriptedStage("optional", deps=["source"], status=StageStatus.SKIPPED, artifacts={"upscaled_image": "pixels"})
    graph, sink = _pipeline(skipped)

    execution = graph.execute(StageContext(cache_enabled=False), parallel=parallel)

    assert execution.success is True and execution.error is None
    assert execution.execution_order == ["source", "optional", "sink"]
    assert execution.stage_results["optional"].status == StageStatus.SKIPPED
    assert sink.calls == 1
    assert sink.seen is not None and sink.seen["upscaled_image"] == "pixels"


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
@pytest.mark.parametrize("status", [StageStatus.FAILED, StageStatus.UNAVAILABLE])
def test_failed_or_unavailable_stage_still_stops_the_pipeline(status: StageStatus, parallel: bool):
    broken = _ScriptedStage("optional", deps=["source"], status=status, error="backend missing")
    graph, sink = _pipeline(broken)

    execution = graph.execute(StageContext(cache_enabled=False), parallel=parallel)

    assert execution.success is False
    assert execution.error == "Stage optional failed: backend missing"
    assert sink.calls == 0


def test_skipped_result_is_not_written_to_the_stage_cache(tmp_path):
    skipped = _ScriptedStage("optional", deps=["source"], status=StageStatus.SKIPPED, artifacts={"upscaled_image": "pixels"})
    graph, sink = _pipeline(skipped)
    cache_dir = tmp_path / "cache"

    for _ in range(2):
        execution = graph.execute(StageContext(cache_enabled=True, cache_dir=cache_dir), parallel=False)
        assert execution.success is True
        assert execution.stage_results["optional"].status == StageStatus.SKIPPED

    assert skipped.calls == 2
    assert sink.calls == 1  # the stage that did real work is served from its cache
    assert sorted(path.name for path in cache_dir.iterdir()) == ["sink-key.json", "source-key.json"]
