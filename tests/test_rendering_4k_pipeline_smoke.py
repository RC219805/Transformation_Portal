"""Smoke coverage for the 4K rendering pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.unit

PIPELINE_MODULE = "transformation_portal.pipelines.rendering_4k_pipeline"
CONTROLNET_AUX_MODULE = "controlnet_aux"


def _load_rendering_4k_pipeline_for_preview_smoke(monkeypatch):
    # The preview preset is expected to keep AI enhancement disabled.
    # Force optional ControlNet support unavailable as well so this smoke test
    # stays deterministic even when extra local dependencies are installed.
    monkeypatch.setitem(sys.modules, CONTROLNET_AUX_MODULE, None)
    monkeypatch.delitem(sys.modules, PIPELINE_MODULE, raising=False)

    return importlib.import_module(PIPELINE_MODULE)


def test_rendering_4k_pipeline_preview_smoke_writes_delivery_artifact(tmp_path: Path, monkeypatch) -> None:
    pipeline_module = _load_rendering_4k_pipeline_for_preview_smoke(monkeypatch)

    try:
        input_path = tmp_path / "input.png"
        output_dir = tmp_path / "output"
        Image.new("RGB", (8, 6), color=(64, 96, 128)).save(input_path)

        pipeline = pipeline_module.Rendering4KPipeline.from_preset("preview")
        assert pipeline.config.name == "preview"
        assert not pipeline.config.ai_enhancement.enabled

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


def test_rendering_4k_depth_cache_does_not_alias_images_sharing_a_prefix() -> None:
    """M-01: two frames that differ only after the first rows must not share a depth cache entry."""
    import numpy as np

    from transformation_portal.pipelines.rendering_4k import stages
    from transformation_portal.pipelines.rendering_4k.pipeline import Rendering4KPipeline

    pipeline = Rendering4KPipeline.from_preset("default")
    assert pipeline.config.depth.cache_enabled
    # Force the deterministic fallback estimator; no model is loaded.
    pipeline._depth_model_initialized = True
    pipeline._depth_model = None

    image_a = np.zeros((128, 512, 3), np.float32)
    image_a[:1] = 1.0
    image_a[1:, :256] = 0.9
    image_b = np.zeros((128, 512, 3), np.float32)
    image_b[:1] = 1.0
    image_b[1:, 256:] = 0.9

    assert pipeline._compute_cache_key(image_a) != pipeline._compute_cache_key(image_b)

    depth_a = pipeline._estimate_depth(image_a, Path("a.png")).copy()
    depth_b = pipeline._estimate_depth(image_b, Path("b.png"))

    np.testing.assert_array_equal(depth_b, stages.estimate_depth_simple(image_b))
    assert not np.array_equal(depth_a, depth_b)
    # Identical content still hits the cache.
    assert (
        pipeline._estimate_depth(image_a.copy(), Path("a2.png")) is pipeline._depth_cache[pipeline._compute_cache_key(image_a)]
    )
