"""Smoke coverage for the 4K rendering pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from PIL import Image

PIPELINE_MODULE = "transformation_portal.pipelines.rendering_4k_pipeline"
CONTROLNET_AUX_MODULE = "controlnet_aux"


def _load_rendering_4k_pipeline_with_stubbed_optionals(monkeypatch):
    # Force optional ControlNet support unavailable so the core smoke test
    # does not drift into heavier enhancement paths if defaults change later.
    monkeypatch.setitem(sys.modules, CONTROLNET_AUX_MODULE, None)
    monkeypatch.delitem(sys.modules, PIPELINE_MODULE, raising=False)

    return importlib.import_module(PIPELINE_MODULE)


def test_rendering_4k_pipeline_smoke_with_stubbed_external_calls(tmp_path: Path, monkeypatch) -> None:
    pipeline_module = _load_rendering_4k_pipeline_with_stubbed_optionals(monkeypatch)

    try:
        input_path = tmp_path / "input.png"
        output_dir = tmp_path / "output"
        Image.new("RGB", (8, 6), color=(64, 96, 128)).save(input_path)

        pipeline = pipeline_module.Rendering4KPipeline.from_preset("preview")
        result = pipeline.process(input_path, output_dir)

        assert result.config_used is not None
        assert result.config_used.name == "preview"
        assert result.depth_map is None
        assert result.iterations == 1
        assert result.image.mode == "RGB"
        assert result.image.size == (8, 6)
        assert result.output_paths.keys() == {"delivery_jpeg"}

        output_path = result.output_paths["delivery_jpeg"]
        assert output_path == output_dir / "input_DELIVERY.jpg"
        assert output_path.exists()

        with Image.open(output_path) as output_image:
            assert output_image.mode == "RGB"
            assert output_image.size == (8, 6)
    finally:
        sys.modules.pop(PIPELINE_MODULE, None)
