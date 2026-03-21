"""Smoke coverage for the unified pipeline orchestrator."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from PIL import Image
import pytest


pytestmark = pytest.mark.unit

PIPELINE_MODULE = "transformation_portal.pipeline_unified"
QUALITY_BRIDGE_MODULE = "transformation_portal.pipelines.quality_feedback_bridge"
RENDERING_4K_MODULE = "transformation_portal.pipelines.rendering_4k_pipeline"


def _load_pipeline_unified_with_stubbed_optionals(monkeypatch):
    quality_module = types.ModuleType(QUALITY_BRIDGE_MODULE)

    class QualityTargets:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class QualityFeedbackBridge:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    quality_module.QualityFeedbackBridge = QualityFeedbackBridge
    quality_module.QualityTargets = QualityTargets

    rendering_module = types.ModuleType(RENDERING_4K_MODULE)

    class Rendering4KPipeline:
        pass

    rendering_module.Rendering4KPipeline = Rendering4KPipeline

    monkeypatch.setitem(sys.modules, QUALITY_BRIDGE_MODULE, quality_module)
    monkeypatch.setitem(sys.modules, RENDERING_4K_MODULE, rendering_module)
    monkeypatch.delitem(sys.modules, PIPELINE_MODULE, raising=False)

    return importlib.import_module(PIPELINE_MODULE)


def test_pipeline_unified_smoke_with_stubbed_dependencies(tmp_path: Path, monkeypatch) -> None:
    pipeline_module = _load_pipeline_unified_with_stubbed_optionals(monkeypatch)

    try:
        input_path = tmp_path / "input.png"
        Image.new("RGB", (8, 6), color=(64, 96, 128)).save(input_path)

        recipe = {
            "name": "Smoke Pipeline",
            "description": "Tiny happy-path smoke test",
            "stages": ["color_grading"],
            "color_grading": {
                "enabled": True,
                "contrast": 1.05,
                "saturation": 1.1,
            },
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(input_path)

        assert result.success
        assert result.error_message is None
        assert result.stages_executed == ["color_grading"]
        assert "color_grading" in result.stage_times
        assert result.output_path is not None
        assert result.output_path == tmp_path / "processed" / "input_smoke_pipeline.png"
        assert result.output_path.exists()

        with Image.open(result.output_path) as output_image:
            assert output_image.mode == "RGB"
            assert output_image.size == (8, 6)
    finally:
        sys.modules.pop(PIPELINE_MODULE, None)
