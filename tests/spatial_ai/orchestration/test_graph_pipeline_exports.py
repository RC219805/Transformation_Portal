from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.orchestration.error_handler import ErrorRecoveryStrategy, PipelineError
from transformation_portal.spatial_ai.orchestration.pipeline import PipelineConfig, PipelineResult, SpatialAIPipeline

pytestmark = pytest.mark.unit


def _import_graph_pipeline() -> ModuleType:
    return importlib.import_module("transformation_portal.spatial_ai.orchestration.graph_pipeline")


def _make_pipeline(*, error_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY) -> SpatialAIPipeline:
    config = PipelineConfig(
        tier="standard",
        stages=["ingest", "segment"],
        use_execution_graph=True,
        error_strategy=error_strategy,
    )
    pipeline = SpatialAIPipeline(config)
    pipeline.resource_manager.select_device = lambda: "cpu"
    pipeline.resource_manager.get_peak_memory_mb = lambda: 42.0
    return pipeline


def _make_exec_result() -> SimpleNamespace:
    linear_rgb = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    masks = np.zeros((2, 2, 2), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, :, 1] = True
    return SimpleNamespace(
        stages_executed=1,
        stages_cached=1,
        total_time_ms=250.0,
        stage_results=[
            SimpleNamespace(stage_id="ingest", cache_hit=False, execution_time_ms=100.0, cache_key="ingest-cache"),
            SimpleNamespace(stage_id="segment", cache_hit=True, execution_time_ms=0.0, cache_key="segment-cache"),
        ],
        outputs={
            "ingest.linear_rgb": linear_rgb,
            "ingest.input_size": (2, 2),
            "ingest.input_format": "TIFF",
            "segment.masks": masks,
            "segment.scores": np.array([0.95, 0.55], dtype=np.float32),
            "segment.metadata.area": np.array([1, 2], dtype=np.int64),
            "segment.metadata.bbox": np.array([[0, 0, 1, 1], [1, 0, 1, 2]], dtype=np.int64),
            "segment.metadata.stability_score": np.array([0.91, 0.73], dtype=np.float32),
        },
    )


def test_pipeline_graph_wrapper_delegates_to_extracted_bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    graph_pipeline = _import_graph_pipeline()
    pipeline = _make_pipeline()
    expected = PipelineResult(input_path=tmp_path / "input.tiff", output_dir=tmp_path / "output", stages_completed=[])
    calls: dict[str, Any] = {}

    def fake_process_with_graph(
        pipeline_arg: SpatialAIPipeline,
        input_path: Path,
        output_dir: Path,
        save_intermediates: bool,
    ) -> PipelineResult:
        calls.update(
            {
                "pipeline": pipeline_arg,
                "input_path": input_path,
                "output_dir": output_dir,
                "save_intermediates": save_intermediates,
            }
        )
        return expected

    monkeypatch.setattr(graph_pipeline, "process_with_graph", fake_process_with_graph)

    result = pipeline._process_with_graph(tmp_path / "input.tiff", tmp_path / "output", False)

    assert result is expected
    assert calls == {
        "pipeline": pipeline,
        "input_path": tmp_path / "input.tiff",
        "output_dir": tmp_path / "output",
        "save_intermediates": False,
    }


def test_direct_graph_pipeline_preserves_result_and_metadata_shape(tmp_path: Path) -> None:
    graph_pipeline = _import_graph_pipeline()
    pipeline = _make_pipeline()
    exec_result = _make_exec_result()
    input_path = tmp_path / "input.tiff"
    input_path.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("transformation_portal.spatial_ai.orchestration.graph_pipeline.Executor.execute") as mock_execute:
        mock_execute.return_value = exec_result
        result = graph_pipeline.process_with_graph(pipeline, input_path, output_dir, True)

    assert isinstance(result, PipelineResult)
    assert result.input_path == input_path
    assert result.output_dir == output_dir
    assert result.stages_completed == ["ingest", "segment"]
    assert result.execution_time == 0.25
    assert result.peak_memory_mb == 42.0
    assert [report["stage"] for report in result.stage_reports] == ["ingest", "segment"]
    assert [report["status"] for report in result.stage_reports] == ["completed", "cached"]
    assert result.linear_image is not None
    assert result.linear_image.input_size == (2, 2)
    assert result.segmentation is not None
    assert np.array_equal(result.segmentation.masks, exec_result.outputs["segment.masks"])
    assert np.allclose(result.segmentation.scores, exec_result.outputs["segment.scores"])
    assert [metadata.area for metadata in result.segmentation.metadata] == [1, 2]
    assert [metadata.bbox for metadata in result.segmentation.metadata] == [(0, 0, 1, 1), (1, 0, 1, 2)]

    graph_metadata = json.loads((output_dir / "graph_execution.json").read_text(encoding="utf-8"))
    assert graph_metadata == {
        "stages_executed": 1,
        "stages_cached": 1,
        "total_time_ms": 250.0,
        "stage_results": [
            {
                "stage_id": "ingest",
                "cache_hit": False,
                "execution_time_ms": 100.0,
                "cache_key": "ingest-cache",
            },
            {
                "stage_id": "segment",
                "cache_hit": True,
                "execution_time_ms": 0.0,
                "cache_key": "segment-cache",
            },
        ],
    }
    assert (output_dir / "pipeline_summary.json").is_file()


def test_direct_graph_pipeline_wraps_executor_errors(tmp_path: Path) -> None:
    graph_pipeline = _import_graph_pipeline()
    pipeline = _make_pipeline(error_strategy=ErrorRecoveryStrategy.FAIL_FAST)
    input_path = tmp_path / "input.tiff"
    input_path.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("transformation_portal.spatial_ai.orchestration.graph_pipeline.Executor.execute") as mock_execute:
        mock_execute.side_effect = RuntimeError("executor failed")
        with pytest.raises(PipelineError) as exc_info:
            graph_pipeline.process_with_graph(pipeline, input_path, output_dir, False)

    assert exc_info.value.stage == "graph"
    assert "executor failed" in str(exc_info.value)


def test_direct_graph_pipeline_returns_partial_for_return_partial_strategy(tmp_path: Path) -> None:
    graph_pipeline = _import_graph_pipeline()
    pipeline = _make_pipeline(error_strategy=ErrorRecoveryStrategy.RETURN_PARTIAL)
    input_path = tmp_path / "input.tiff"
    input_path.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("transformation_portal.spatial_ai.orchestration.graph_pipeline.Executor.execute") as mock_execute:
        mock_execute.side_effect = RuntimeError("executor failed")
        result = graph_pipeline.process_with_graph(pipeline, input_path, output_dir, False)

    assert isinstance(result, PipelineResult)
    assert result.stages_completed == []
    assert result.errors == ["executor failed"]


def test_process_rejects_reconstruction_before_graph_bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    graph_pipeline = _import_graph_pipeline()
    config = PipelineConfig(
        tier="apex_research",
        stages=["ingest", "segment", "reconstruction"],
        use_execution_graph=True,
    )
    pipeline = SpatialAIPipeline(config)
    input_path = tmp_path / "input.tiff"
    input_path.touch()

    def fail_if_called(*args: Any, **kwargs: Any) -> PipelineResult:
        raise AssertionError("graph bridge should not be called")

    monkeypatch.setattr(graph_pipeline, "process_with_graph", fail_if_called)

    with pytest.raises(PipelineError) as exc_info:
        pipeline.process(input_path=input_path, output_dir=tmp_path / "output")

    assert exc_info.value.stage == "reconstruction"
    assert "process_multiview" in str(exc_info.value)
