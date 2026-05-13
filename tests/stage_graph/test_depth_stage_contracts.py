"""CPU/core contracts for the stage-graph depth estimation stage."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from transformation_portal.stage_graph.stage import StageContext, StageStatus
from transformation_portal.stage_graph.stages.depth import DepthEstimationStage

pytestmark = pytest.mark.unit


def _image_uint8() -> np.ndarray:
    return np.array(
        [
            [[0, 32, 64], [96, 128, 160], [192, 224, 255]],
            [[255, 224, 192], [160, 128, 96], [64, 32, 0]],
        ],
        dtype=np.uint8,
    )


def _image_float() -> np.ndarray:
    return _image_uint8().astype(np.float32) / 255.0


def test_compute_fails_with_explicit_missing_image_error() -> None:
    stage = DepthEstimationStage()

    result = stage.compute(StageContext(artifacts={}))

    assert result.status is StageStatus.FAILED
    assert result.error == "Missing 'image' artifact in context"


def test_cache_key_is_deterministic_and_model_configuration_sensitive() -> None:
    image = _image_uint8()
    context = StageContext(artifacts={"image": image})

    first = DepthEstimationStage(model_size="small", version="1.0.0").get_cache_key(context)
    second = DepthEstimationStage(model_size="small", version="1.0.0").get_cache_key(context)
    different_size = DepthEstimationStage(model_size="large", version="1.0.0").get_cache_key(context)
    different_version = DepthEstimationStage(model_size="small", version="2.0.0").get_cache_key(context)

    assert first == second
    assert first.startswith("depth_small_1.0.0_")
    assert different_size != first
    assert different_version != first
    assert DepthEstimationStage().get_cache_key(StageContext(artifacts={})) == "no_image"


def test_model_load_import_failure_uses_placeholder_without_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    image = _image_uint8()
    stage = DepthEstimationStage()

    result = stage.compute(StageContext(artifacts={"image": image}, device="cpu"))

    assert result.status is StageStatus.COMPLETED
    assert stage._model == "placeholder"
    depth = result.artifacts["depth_map"]
    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.float32
    assert depth.min() == pytest.approx(0.0)
    assert depth.max() == pytest.approx(1.0)
    assert result.artifacts["depth_metadata"] == {
        "model_size": "small",
        "device": "cpu",
        "shape": image.shape[:2],
    }


def test_placeholder_output_is_float32_normalized_to_image_shape() -> None:
    image = _image_uint8()
    stage = DepthEstimationStage()
    stage._model = "placeholder"

    depth = stage._estimate_depth(image, device="cpu")

    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.float32
    assert np.all(depth >= 0.0)
    assert np.all(depth <= 1.0)
    assert np.allclose(depth[0], np.linspace(0.0, 1.0, image.shape[1], dtype=np.float32))


def test_fake_transformers_pipeline_returns_normalized_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_calls: list[dict[str, object]] = []

    class FakePipeline:
        def __call__(self, image):
            pipeline_calls.append({"image_size": image.size, "mode": image.mode})
            return {
                "depth": np.array(
                    [
                        [[10.0], [20.0], [30.0]],
                        [[40.0], [50.0], [60.0]],
                    ],
                    dtype=np.float32,
                )
            }

    def fake_pipeline(task: str, *, model: str, device: int) -> FakePipeline:
        pipeline_calls.append({"task": task, "model": model, "device": device})
        return FakePipeline()

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    stage = DepthEstimationStage()
    result = stage.compute(StageContext(artifacts={"image": _image_uint8()}, device="cuda"))

    assert result.status is StageStatus.COMPLETED
    assert pipeline_calls[0] == {
        "task": "depth-estimation",
        "model": "depth-anything/Depth-Anything-V2-Small-hf",
        "device": 0,
    }
    assert pipeline_calls[1] == {"image_size": (3, 2), "mode": "RGB"}
    depth = result.artifacts["depth_map"]
    assert depth.shape == (2, 3)
    assert depth.dtype == np.float32
    assert depth.min() == pytest.approx(0.0)
    assert depth.max() == pytest.approx(1.0)
    assert result.metadata["model_size"] == "small"


def test_inference_exception_returns_constant_fallback_depth() -> None:
    class FailingPipeline:
        def __call__(self, image):
            raise RuntimeError("backend unavailable")

    image = _image_uint8()
    stage = DepthEstimationStage()
    stage._model = FailingPipeline()

    depth = stage._estimate_depth(image, device="cpu")

    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.float32
    assert np.allclose(depth, 0.5)


@pytest.mark.parametrize("image", [_image_float(), _image_uint8()])
def test_input_scaling_accepts_float_unit_range_and_uint8(image: np.ndarray) -> None:
    captured_inputs: list[np.ndarray] = []

    class RecordingPipeline:
        def __call__(self, image):
            captured_inputs.append(np.array(image))
            return {"depth": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)}

    stage = DepthEstimationStage()
    stage._model = RecordingPipeline()

    depth = stage._estimate_depth(image, device="cpu")

    assert captured_inputs[0].dtype == np.uint8
    assert captured_inputs[0].shape == image.shape
    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.float32
    assert depth.min() == pytest.approx(0.0)
    assert depth.max() == pytest.approx(1.0)
