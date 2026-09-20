"""Regression tests for unquantized DA2 predictions and manual model dispatch."""

import hashlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.models import depth_anything_v2 as da2_module
from transformation_portal.depth.pipeline import ArchitecturalDepthPipeline
from transformation_portal.depth.utils.cache import DepthCache
from transformation_portal.lux_depth_v3 import inference as inference_module

pytestmark = pytest.mark.unit


class _Tensor:
    """Small tensor boundary double, without importing the optional ML runtime."""

    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def to(self, device):
        return self


@pytest.fixture(params=["model_wrapper", "v3_adapter"])
def adapter(request, monkeypatch):
    """Exercise both Transformers consumers without downloading model weights."""
    module = da2_module if request.param == "model_wrapper" else inference_module
    monkeypatch.setattr(module, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(module, "torch", SimpleNamespace(Tensor=_Tensor, no_grad=nullcontext))
    monkeypatch.setattr(module, "_ensure_optional_runtime_imports", lambda: None)
    if request.param == "model_wrapper":
        result = da2_module.DepthAnythingV2Model.__new__(da2_module.DepthAnythingV2Model)
        result.processor = None
        result.backend = da2_module.ModelBackend.PYTORCH_CPU
        result.variant = da2_module.ModelVariant.SMALL
        result.device = "cpu"
    else:
        result = inference_module.DA3InferenceEngine.__new__(inference_module.DA3InferenceEngine)
        monkeypatch.setitem(
            sys.modules,
            "depth_anything_3.api",
            SimpleNamespace(DepthAnything3=type("DA3", (), {})),
        )
    return result


@pytest.mark.parametrize("shape", [(32, 32), (1, 32, 32), (1, 1, 1024), (1, 1024, 1)])
@pytest.mark.parametrize("as_tensor", [False, True])
def test_pipeline_uses_float_prediction_without_display_quantization(adapter, shape, as_tensor):
    raw = np.linspace(0.125, 2.25, 1024, dtype=np.float32).reshape(shape)
    prediction = _Tensor(raw) if as_tensor else raw
    adapter.model = lambda image: {
        "predicted_depth": prediction,
        "depth": Image.new("L", image.size, color=0),
    }

    result = adapter._estimate_depth_pytorch(Image.new("RGB", (32, 32)))

    expected = raw[0] if raw.ndim == 3 else raw
    np.testing.assert_array_equal(result["depth_raw"], expected)
    assert result["depth_raw"].dtype == np.float32
    assert result["depth"].shape == expected.shape
    assert result["metadata"]["shape"] == expected.shape
    assert np.unique(result["depth"]).size == 1024
    np.testing.assert_allclose(result["depth"], (expected - expected.min()) / np.ptp(expected))


def test_pipeline_rejects_missing_float_prediction(adapter):
    adapter.model = lambda image: {"depth": Image.new("L", image.size)}
    with pytest.raises(KeyError, match="predicted_depth"):
        adapter._estimate_depth_pytorch(Image.new("RGB", (2, 2)))


def test_pipeline_rejects_multiple_depth_maps(adapter):
    adapter.model = lambda image: {"predicted_depth": np.ones((2, 2, 2), dtype=np.float32)}
    with pytest.raises(ValueError, match="one HxW"):
        adapter._estimate_depth_pytorch(Image.new("RGB", (2, 2)))


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("image_shape", [(40, 60), (1, 1024), (1024, 1), (1, 1)])
def test_callable_manual_fallback_receives_processed_tensors(monkeypatch, device, image_shape):
    """Manual inference must restore float predictions to the source image grid."""
    raw = np.linspace(0.1, 2.0, 512, dtype=np.float32).reshape(1, 16, 32)
    height, width = image_shape
    image = Image.new("RGB", (width, height))
    pixel_values = _Tensor(np.zeros((1, 3, 16, 32), dtype=np.float32))
    expected = np.asarray(Image.fromarray(raw[0]).resize(image.size, Image.Resampling.BICUBIC))
    calls = []

    class ManualModel:
        def to(self, target_device):
            calls.append(("device", target_device))
            return self

        def __call__(self, *, pixel_values):
            calls.append(("model", pixel_values))
            return SimpleNamespace(predicted_depth=_Tensor(raw))

    class Processor:
        def __call__(self, *, images, return_tensors):
            assert images is image
            assert return_tensors == "pt"
            return {"pixel_values": pixel_values}

        def post_process_depth_estimation(self, outputs, *, target_sizes):
            assert target_sizes == [image_shape]
            np.testing.assert_array_equal(outputs.predicted_depth.values, raw)
            calls.append(("postprocess", image_shape))
            # Match upstream float interpolation and its squeeze of all singleton axes.
            resized = np.asarray(
                Image.fromarray(outputs.predicted_depth.values[0]).resize(
                    (target_sizes[0][1], target_sizes[0][0]), Image.Resampling.BICUBIC
                )
            )
            return [{"predicted_depth": _Tensor(resized.squeeze())}]

    def unavailable_pipeline(**kwargs):
        raise RuntimeError("pipeline unavailable")

    monkeypatch.setattr(da2_module, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(da2_module, "TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(da2_module, "TRANSFORMERS_TORCH_BACKEND_ISSUE", None)
    monkeypatch.setattr(da2_module, "torch", SimpleNamespace(Tensor=_Tensor, no_grad=nullcontext))
    monkeypatch.setattr(da2_module, "pipeline", unavailable_pipeline)
    monkeypatch.setattr(da2_module, "AutoImageProcessor", SimpleNamespace(from_pretrained=lambda *a, **k: Processor()))
    monkeypatch.setattr(
        da2_module, "AutoModelForDepthEstimation", SimpleNamespace(from_pretrained=lambda *a, **k: ManualModel())
    )
    model = da2_module.DepthAnythingV2Model.__new__(da2_module.DepthAnythingV2Model)
    model.variant = da2_module.ModelVariant.SMALL
    model.backend = da2_module.ModelBackend.PYTORCH_CPU
    model.model_revision = None
    model.device = device
    model.processor = None
    model._load_pytorch_model()

    result = model.estimate_depth(image)

    assert ("model", pixel_values) in calls
    assert calls.count(("postprocess", image_shape)) == 1
    if device == "mps":
        assert ("device", "mps") in calls
    np.testing.assert_array_equal(result["depth_raw"], expected)
    assert result["depth_raw"].dtype == np.float32
    assert result["depth"].shape == image_shape
    assert result["metadata"]["shape"] == image_shape
    np.testing.assert_allclose(result["depth"], (expected - expected.min()) / (np.ptp(expected) + 1e-8))
    if image_shape != (1, 1):
        assert np.unique(result["depth"]).size > 256


def test_legacy_pipeline_cache_excludes_old_quantized_entries(tmp_path):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    legacy = DepthCache(cache_dir=tmp_path, enable_disk_cache=True)
    quantized = {"depth": np.zeros((2, 2), dtype=np.float32)}
    legacy.put(image, quantized)
    legacy_key = legacy._generate_key(image)
    assert legacy_key == hashlib.md5(image.tobytes(), usedforsecurity=False).hexdigest()
    assert (tmp_path / f"{legacy_key}.pkl").is_file()

    pipeline = ArchitecturalDepthPipeline.__new__(ArchitecturalDepthPipeline)
    pipeline.config = {"depth_model": {"enable_disk_cache": False}}
    cache = pipeline._init_cache()
    cache.cache_dir = tmp_path
    cache.enable_disk_cache = True
    unquantized = {"depth": np.full((2, 2), 0.125, dtype=np.float32)}
    calls = []

    def compute():
        calls.append(True)
        return unquantized

    result = cache.get_or_compute(image, compute)

    assert calls == [True]
    assert result is unquantized
    assert cache._generate_key(image) != legacy_key
    assert (tmp_path / f"{legacy_key}.pkl").is_file()
    assert (tmp_path / f"{cache._generate_key(image)}.pkl").is_file()
    reopened = DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=cache.namespace)
    np.testing.assert_array_equal(reopened.get_or_compute(image, compute)["depth"], unquantized["depth"])
    assert calls == [True]


@pytest.mark.parametrize("namespace", ["", " ", 42])
def test_depth_cache_rejects_empty_or_nontext_recipe(namespace):
    with pytest.raises(ValueError, match="namespace"):
        DepthCache(namespace=namespace)
