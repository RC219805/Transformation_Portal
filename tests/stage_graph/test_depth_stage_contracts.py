"""CPU/core contracts for the stage-graph depth estimation stage."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from transformation_portal.stage_graph.stage import StageContext, StageStatus
from transformation_portal.stage_graph.stages import depth_pro as depth_pro_module
from transformation_portal.stage_graph.stages.depth import DepthEstimationStage
from transformation_portal.stage_graph.stages.depth_pro import DepthProStage

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

    assert result.status == StageStatus.FAILED
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


def test_model_load_import_failure_fails_closed_without_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    stage = DepthEstimationStage()
    result = stage.compute(StageContext(artifacts={"image": _image_uint8()}, device="cpu"))
    assert result.status == StageStatus.FAILED
    assert "unavailable" in result.error
    assert result.artifacts == {}
    assert stage._model is None


def test_legacy_placeholder_cannot_produce_successful_depth() -> None:
    stage = DepthEstimationStage()
    stage._model = "placeholder"
    result = stage.compute(StageContext(artifacts={"image": _image_uint8()}, device="cpu"))
    assert result.status == StageStatus.FAILED
    assert result.artifacts == {}


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

    assert result.status == StageStatus.COMPLETED
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


def test_inference_exception_fails_without_synthetic_artifacts() -> None:
    class FailingPipeline:
        def __call__(self, image):
            raise RuntimeError("backend unavailable")

    stage = DepthEstimationStage()
    stage._model = FailingPipeline()
    result = stage.compute(StageContext(artifacts={"image": _image_uint8()}, device="cpu"))
    assert result.status == StageStatus.FAILED
    assert "backend unavailable" in result.error
    assert result.artifacts == {}
    with pytest.raises(RuntimeError, match="Depth inference failed"):
        stage._estimate_depth(_image_uint8(), device="cpu")


class _StubDepthPro(DepthProStage):
    """Depth Pro stage with the model replaced; cache and output handling are real."""

    def __init__(self, checkpoint_path):
        super().__init__(checkpoint_path=checkpoint_path, device="cpu")
        self.inference_runs = 0

    def _load_model(self) -> None:
        self._transform = "Compose(ToTensor(), Normalize())"
        self._model_loaded = True

    def _run_inference(self, image, *, focal_length_px=None):
        self.inference_runs += 1
        camera = {
            "focal_length_px": None,
            "focal_length_source": "unavailable",
            "fov_deg": None,
            "coordinate_space": "input_image",
        }
        return np.ones(image.shape[:2], dtype=np.float32), 0.01, camera


@pytest.fixture
def depth_pro_checkpoint(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(depth_pro_module, "DEPTH_PRO_AVAILABLE", True)
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    return checkpoint


def test_depth_pro_run_that_asks_for_files_is_not_answered_from_the_cache(tmp_path, depth_pro_checkpoint) -> None:
    cache_dir = tmp_path / "cache"
    image = _image_uint8()
    warm = _StubDepthPro(depth_pro_checkpoint)
    assert warm.execute(StageContext(artifacts={"image": image}, cache_enabled=True, cache_dir=cache_dir)).is_success()

    stage = _StubDepthPro(depth_pro_checkpoint)
    output_dir = tmp_path / "delivery"
    result = stage.execute(
        StageContext(artifacts={"image": image, "output_dir": output_dir}, cache_enabled=True, cache_dir=cache_dir)
    )

    assert result.status == StageStatus.COMPLETED, result.error
    assert result.cache_hit is False
    assert stage.inference_runs == 1
    assert result.artifacts["depth_float_path"] == output_dir / "depth_depthpro.npy"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "depth_depthpro.npy",
        "depth_depthpro_preview.png",
        "depth_depthpro_provenance.json",
    ]
    assert np.array_equal(np.load(output_dir / "depth_depthpro.npy"), result.artifacts["depth_map"])


def test_depth_pro_file_outputs_are_not_written_into_the_stage_cache(tmp_path, depth_pro_checkpoint) -> None:
    cache_dir = tmp_path / "cache"
    image = _image_uint8()

    with_files = _StubDepthPro(depth_pro_checkpoint)
    for _ in range(2):
        context = StageContext(
            artifacts={"image": image, "output_dir": tmp_path / "delivery"}, cache_enabled=True, cache_dir=cache_dir
        )
        assert with_files.execute(context).status == StageStatus.COMPLETED
    assert with_files.inference_runs == 2
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []

    # In-memory runs still use the cache. Each run gets a stage that has not
    # loaded its model, as a new process would: the key of a loaded stage
    # carries the transform hash (ADR-018) and differs from a fresh one.
    in_memory = [_StubDepthPro(depth_pro_checkpoint) for _ in range(2)]
    results = [
        stage.execute(StageContext(artifacts={"image": image}, cache_enabled=True, cache_dir=cache_dir)) for stage in in_memory
    ]
    assert [result.cache_hit for result in results] == [False, True]
    assert [stage.inference_runs for stage in in_memory] == [1, 0]
    assert set(results[1].artifacts) == {"depth_map", "depth_provenance"}


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
