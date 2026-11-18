"""Regression coverage for the DepthProcessor adapter and the depth model config mapping (D-20)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from transformation_portal.depth import depth_processor as depth_processor_mod
from transformation_portal.depth import pipeline as pipeline_mod
from transformation_portal.depth.models import ModelBackend, ModelVariant

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.fixture
def model_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace the depth model with a stub that records its constructor arguments."""
    calls: list = []

    class _StubDepthModel:
        variant = SimpleNamespace(name="STUB")

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def estimate_depth(self, image):
            h, w = image.shape[:2]
            return {"depth": np.zeros((h, w), np.float32), "metadata": {"inference_time_ms": 0.0}}

    monkeypatch.setattr(pipeline_mod, "DepthAnythingV2Model", _StubDepthModel)
    return calls


@pytest.mark.parametrize(
    ("device", "backend", "model_device"),
    [
        ("cpu", ModelBackend.PYTORCH_CPU, "cpu"),
        ("cuda", ModelBackend.PYTORCH_CPU, "cuda"),
        ("mps", ModelBackend.COREML, None),
    ],
)
def test_default_config_honours_the_requested_device(model_calls, device, backend, model_device) -> None:
    depth_processor_mod.DepthProcessor(device=device)

    assert len(model_calls) == 1
    kwargs = model_calls[0]
    assert kwargs["variant"] is ModelVariant.SMALL
    assert kwargs["backend"] is backend
    assert kwargs["device"] == model_device


def test_default_config_rejects_an_unknown_device(model_calls) -> None:
    with pytest.raises(ValueError, match="Unsupported DepthProcessor device"):
        depth_processor_mod.DepthProcessor(device="tpu")
    assert model_calls == []


def test_config_path_builds_the_depth_model_once(model_calls, tmp_path) -> None:
    config = tmp_path / "depth.yaml"
    config.write_text("depth_model:\n  variant: base\n  backend: pytorch_cpu\n  device: cpu\nprocessing: {}\n")

    processor = depth_processor_mod.DepthProcessor(config_path=config)

    assert len(model_calls) == 1
    assert model_calls[0]["variant"] is ModelVariant.BASE
    assert model_calls[0]["backend"] is ModelBackend.PYTORCH_CPU
    assert model_calls[0]["device"] == "cpu"
    assert processor._pipeline.config["depth_model"]["variant"] == "base"


def test_cache_stats_report_the_depth_cache_counters(model_calls) -> None:
    processor = depth_processor_mod.DepthProcessor(device="cpu")
    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)

    for _ in range(3):
        processor.estimate_depth(image)

    assert processor.get_cache_stats() == {"hits": 2, "misses": 1}


@pytest.mark.parametrize(
    ("depth_model", "message"),
    [
        ({"variant": "vitl"}, "Unknown depth_model.variant"),
        ({"variant": "small", "backend": "pytorch"}, "Unknown depth_model.backend"),
    ],
)
def test_pipeline_rejects_unknown_variant_and_backend(model_calls, depth_model, message) -> None:
    with pytest.raises(ValueError, match=message):
        pipeline_mod.ArchitecturalDepthPipeline({"depth_model": depth_model, "processing": {}})
    assert model_calls == []


def test_pipeline_without_backend_or_device_keeps_auto_detection(model_calls) -> None:
    pipeline_mod.ArchitecturalDepthPipeline({"depth_model": {"variant": "large"}, "processing": {}})

    assert model_calls[0]["variant"] is ModelVariant.LARGE
    assert model_calls[0]["backend"] is None
    assert model_calls[0]["device"] is None
