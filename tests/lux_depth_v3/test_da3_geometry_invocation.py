"""Bounded corrected DA3 model-grid contracts without optional inference.

The API spy exercises the real engine boundary while no models, tensors, or
external runtime are loaded. Geometry restoration remains native and explicit.
"""

from __future__ import annotations

import io
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.config import DA3Config, EnhanceConfig
from transformation_portal.lux_depth_v3.geometry import (
    CORRECTED_MODEL_MAX_LONG_EDGE,
    LEGACY_GEOMETRY_RECIPE,
    NORMALIZED_GEOMETRY_RECIPE,
)
from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot, preprocess_image

pytestmark = pytest.mark.unit


def _spy_engine(monkeypatch, recipe, *, prediction_shape=None):
    import transformation_portal.lux_depth_v3.inference as inference

    calls = []

    class SpyDA3:
        def inference(self, images, **kwargs):
            calls.append((images, kwargs))
            height, width = prediction_shape or (images[0].height, images[0].width)
            depth = np.arange(height * width, dtype=np.float32).reshape(1, height, width)
            return SimpleNamespace(depth=depth)

    package = ModuleType("depth_anything_3")
    api = ModuleType("depth_anything_3.api")
    api.DepthAnything3 = SpyDA3
    monkeypatch.setitem(sys.modules, "depth_anything_3", package)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api)
    monkeypatch.setattr(inference, "_ensure_optional_runtime_imports", lambda: None)
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(inference, "torch", None)
    engine = inference.DA3InferenceEngine.__new__(inference.DA3InferenceEngine)
    engine.config = DA3Config(geometry_recipe=recipe)
    engine.model = SpyDA3()
    return engine, calls


@pytest.mark.parametrize("orientation", [1, 6])
def test_corrected_large_source_is_bounded_before_exact_da3_invocation(tmp_path, monkeypatch, orientation):
    path = tmp_path / "large.png"
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (1300, 900), (71, 113, 157)).save(path, exif=exif)
    geometry = preprocess_geometry_snapshot(path)
    expected_shape = (336, 504) if orientation == 1 else (504, 336)

    assert geometry.transform.target_size == CORRECTED_MODEL_MAX_LONG_EDGE == 504
    assert geometry.transform.resized_shape == ((348, 504) if orientation == 1 else (504, 348))
    assert geometry.transform.model_shape == expected_shape
    assert geometry.canonical_rgb.shape[:2] == ((900, 1300) if orientation == 1 else (1300, 900))
    image = Image.fromarray((geometry.pixels * 255).astype(np.uint8))
    engine, calls = _spy_engine(monkeypatch, NORMALIZED_GEOMETRY_RECIPE)
    result = engine._estimate_depth_pytorch(image)

    assert calls == [([image], {"process_res": 504, "process_res_method": "upper_bound_resize"})]
    assert result["depth_raw"].shape == geometry.transform.model_shape
    restored, support = geometry.transform.restore_numeric(result["depth_raw"])
    assert restored.shape == geometry.transform.normalized_shape
    assert np.count_nonzero(~support) > 0
    assert not np.any(restored[~support])


@pytest.mark.parametrize("shape", [(31, 45), (5, 9), (504, 504)])
def test_corrected_small_source_never_upscales_before_crop_or_required_padding(tmp_path, monkeypatch, shape):
    path = tmp_path / "small.png"
    Image.new("RGB", shape[::-1], (31, 79, 173)).save(path)
    geometry = preprocess_geometry_snapshot(path)
    assert geometry.transform.target_size == max(shape)
    assert geometry.transform.resized_shape == shape
    image = Image.fromarray((geometry.pixels * 255).astype(np.uint8))
    engine, calls = _spy_engine(monkeypatch, NORMALIZED_GEOMETRY_RECIPE)
    result = engine._estimate_depth_pytorch(image)
    assert calls[0][1] == {"process_res": max(image.size), "process_res_method": "upper_bound_resize"}
    assert result["depth_raw"].shape == geometry.transform.model_shape


def test_legacy_preprocessing_and_model_kwargs_remain_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "legacy.png"
    Image.new("RGB", (1300, 900), (71, 113, 157)).save(path)
    pixels, original_shape = preprocess_image(path)
    assert original_shape == (900, 1300)
    assert pixels.shape == (896, 1288, 3)
    image = Image.fromarray((pixels * 255).astype(np.uint8))
    engine, calls = _spy_engine(monkeypatch, LEGACY_GEOMETRY_RECIPE, prediction_shape=(350, 504))
    result = engine._estimate_depth_pytorch(image)
    assert calls == [([image], {})]
    assert result["depth_raw"].shape == (350, 504)


@pytest.mark.parametrize("size", [(518, 14), (28, 27), (9, 14)])
def test_corrected_engine_rejects_unbounded_or_unrecorded_grid_before_model_compute(monkeypatch, size):
    engine, calls = _spy_engine(monkeypatch, NORMALIZED_GEOMETRY_RECIPE)
    with pytest.raises(ValueError, match="bounded multiple-of-14"):
        engine._estimate_depth_pytorch(Image.new("RGB", size))
    assert not calls


def test_corrected_engine_rejects_internally_resized_prediction(monkeypatch):
    engine, calls = _spy_engine(monkeypatch, NORMALIZED_GEOMETRY_RECIPE, prediction_shape=(336, 504))
    with pytest.raises(ValueError, match="exact finite recorded model grid"):
        engine._estimate_depth_pytorch(Image.new("RGB", (42, 28)))
    assert len(calls) == 1


def test_corrected_engine_rejects_nonfinite_prediction(monkeypatch):
    engine, _ = _spy_engine(monkeypatch, NORMALIZED_GEOMETRY_RECIPE)
    engine.model.inference = lambda *_args, **_kwargs: SimpleNamespace(depth=np.full((1, 28, 42), np.nan))
    with pytest.raises(ValueError, match="exact finite recorded model grid"):
        engine._estimate_depth_pytorch(Image.new("RGB", (42, 28)))


@pytest.mark.parametrize("recipe", [LEGACY_GEOMETRY_RECIPE, NORMALIZED_GEOMETRY_RECIPE])
def test_backend_local_engine_receives_exact_recipe(monkeypatch, recipe):
    import transformation_portal.lux_depth_v3.inference as inference
    from transformation_portal.depth.backends.da3 import DA3Backend

    captured = []
    monkeypatch.setattr(inference, "DA3InferenceEngine", lambda **kwargs: captured.append(kwargs))
    backend = DA3Backend(EnhanceConfig(depth_backend="da3", depth_device="cpu", geometry_recipe=recipe))
    backend._load_engine("cpu")
    assert captured[0]["config"].geometry_recipe == recipe


@pytest.mark.parametrize("recipe", [LEGACY_GEOMETRY_RECIPE, NORMALIZED_GEOMETRY_RECIPE])
def test_noncanonical_backend_carries_only_opted_in_geometry_to_worker(monkeypatch, recipe):
    from transformation_portal.depth.backends import da3

    backend = da3.DA3Backend(EnhanceConfig(depth_backend="da3", depth_device="cpu", geometry_recipe=recipe))
    backend._python_executable = "/test-only/python"
    monkeypatch.setattr(backend, "_worker_cwd", lambda: None)
    monkeypatch.setattr(backend, "_build_worker_env", lambda: {})
    commands = []

    def record_launch(command, **_kwargs):
        commands.append(command)
        raise FileNotFoundError("test stops before process launch")

    monkeypatch.setattr(da3.subprocess, "run", record_launch)
    with pytest.raises(RuntimeError, match="executable_not_found"):
        backend._compute_subprocess(Image.new("RGB", (42, 28)))
    if recipe == LEGACY_GEOMETRY_RECIPE:
        assert "--geometry-recipe" not in commands[0]
    else:
        assert commands[0][commands[0].index("--geometry-recipe") + 1] == recipe


@pytest.mark.parametrize("recipe", [LEGACY_GEOMETRY_RECIPE, NORMALIZED_GEOMETRY_RECIPE])
def test_canonical_worker_derives_recipe_from_validated_plan(monkeypatch, tmp_path, recipe):
    import transformation_portal.lux_depth_v3.inference as inference
    from transformation_portal.depth.backends import da3_worker
    from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution

    source = tmp_path / "source.png"
    Image.new("RGB", (45, 31), (31, 79, 173)).save(source)
    prepared = prepare_lux_execution(
        EnhanceConfig(depth_backend="da3", depth_device="cpu", geometry_recipe=recipe), tmp_path, [source]
    )
    monkeypatch.setattr(da3_worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(prepared.canonical_plan_bytes)))
    authority = da3_worker._consume_canonical_worker_authority(candidate_id="da3", model_backend_id=None)
    captured = []
    monkeypatch.setattr(inference, "DA3InferenceEngine", lambda **kwargs: captured.append(kwargs))
    kwargs = dict(
        model_variant_name="METRIC_LARGE",
        model_key=authority.resolved_model_contract.canonical_key,
        model_revision=authority.resolved_model_contract.revision,
        device="cpu",
        use_coreml=False,
        non_commercial_ok=False,
        canonical_authority=authority,
    )
    da3_worker._build_inference_engine(**kwargs)
    assert captured[0]["config"].geometry_recipe == recipe
    with pytest.raises(ValueError, match="only from its execution plan"):
        da3_worker._build_inference_engine(**kwargs, geometry_recipe=recipe)


@pytest.mark.parametrize("recipe", [None, NORMALIZED_GEOMETRY_RECIPE])
def test_noncanonical_worker_preserves_legacy_default_and_explicit_corrected_recipe(monkeypatch, recipe):
    import transformation_portal.lux_depth_v3.inference as inference
    from transformation_portal.depth.backends import da3_worker

    captured = []
    monkeypatch.setattr(inference, "DA3InferenceEngine", lambda **kwargs: captured.append(kwargs))
    da3_worker._build_inference_engine(
        model_variant_name="METRIC_LARGE",
        model_key="da3_metric",
        model_revision=None,
        device="cpu",
        use_coreml=False,
        non_commercial_ok=False,
        canonical_authority=None,
        geometry_recipe=recipe,
    )
    assert captured[0]["config"].geometry_recipe == (recipe or LEGACY_GEOMETRY_RECIPE)


def test_canonical_worker_rejects_geometry_flag_override():
    from transformation_portal.depth.backends import da3_worker

    parser = da3_worker._build_parser()
    args = parser.parse_args(
        ["--execution-plan-stdin", "--candidate-id", "da3", "--geometry-recipe", NORMALIZED_GEOMETRY_RECIPE, "--check"]
    )
    with pytest.raises(SystemExit):
        da3_worker._validate_execution_mode(parser, args)


def test_unknown_geometry_recipe_fails_closed():
    with pytest.raises(ValueError, match="Unsupported Lux geometry recipe"):
        DA3Config(geometry_recipe="unbounded")


def test_mutated_engine_recipe_fails_before_model_compute(monkeypatch):
    engine, calls = _spy_engine(monkeypatch, LEGACY_GEOMETRY_RECIPE)
    engine.config.geometry_recipe = ""
    with pytest.raises(ValueError, match="Unsupported Lux geometry recipe"):
        engine._estimate_depth_pytorch(Image.new("RGB", (42, 28)))
    assert not calls


def test_explicit_empty_worker_recipe_cannot_select_legacy():
    from transformation_portal.depth.backends import da3_worker

    with pytest.raises(ValueError, match="Unsupported Lux geometry recipe"):
        da3_worker._build_inference_engine(
            model_variant_name="METRIC_LARGE",
            model_key="da3_metric",
            model_revision=None,
            device="cpu",
            use_coreml=False,
            non_commercial_ok=False,
            canonical_authority=None,
            geometry_recipe="",
        )


def _forbid_model_load(monkeypatch, engine):
    loads = []
    engine._model_loaded = False

    def load():
        loads.append(True)
        raise AssertionError("MODEL_LOAD_REACHED")

    monkeypatch.setattr(engine, "_load_model", load)
    return loads


@pytest.mark.parametrize("input_kind", ["pil", "numpy"])
@pytest.mark.parametrize("size", [(518, 14), (28, 27), (9, 14)])
def test_public_corrected_predict_rejects_invalid_grid_before_lazy_model_load(monkeypatch, input_kind, size):
    engine, calls = _spy_engine(monkeypatch, NORMALIZED_GEOMETRY_RECIPE)
    loads = _forbid_model_load(monkeypatch, engine)
    image = Image.new("RGB", size)
    value = image if input_kind == "pil" else np.asarray(image)
    with pytest.raises(ValueError, match="bounded multiple-of-14"):
        engine.predict(value)
    assert loads == calls == []


@pytest.mark.parametrize("input_kind", ["pil", "numpy"])
def test_public_predict_rejects_mutated_recipe_before_lazy_model_load(monkeypatch, input_kind):
    engine, calls = _spy_engine(monkeypatch, NORMALIZED_GEOMETRY_RECIPE)
    engine.config.geometry_recipe = ""
    loads = _forbid_model_load(monkeypatch, engine)
    image = Image.new("RGB", (42, 28))
    value = image if input_kind == "pil" else np.asarray(image)
    with pytest.raises(ValueError, match="Unsupported Lux geometry recipe"):
        engine.predict(value)
    assert loads == calls == []


@pytest.mark.parametrize("input_kind", ["pil", "numpy"])
def test_public_legacy_predict_preserves_loader_before_input_validation(monkeypatch, input_kind):
    engine, calls = _spy_engine(monkeypatch, LEGACY_GEOMETRY_RECIPE)
    loads = _forbid_model_load(monkeypatch, engine)
    image = Image.new("RGB", (518, 14))
    value = image if input_kind == "pil" else np.asarray(image)
    with pytest.raises(AssertionError, match="MODEL_LOAD_REACHED"):
        engine.predict(value)
    assert loads == [True]
    assert calls == []


@pytest.mark.parametrize("failure", ["grid", "recipe"])
def test_corrected_worker_public_predict_rejects_before_model_load_or_output(tmp_path, monkeypatch, failure):
    from transformation_portal.depth.backends import da3_worker

    engine, calls = _spy_engine(monkeypatch, NORMALIZED_GEOMETRY_RECIPE)
    loads = _forbid_model_load(monkeypatch, engine)
    if failure == "recipe":
        engine.config.geometry_recipe = ""
    source = tmp_path / "input.png"
    Image.new("RGB", (518, 14) if failure == "grid" else (42, 28)).save(source)
    depth = tmp_path / "result.npy"
    report = tmp_path / "result.json"
    monkeypatch.setattr(da3_worker, "_build_inference_engine", lambda **_kwargs: engine)
    monkeypatch.setattr(da3_worker, "_seed_isolated_inference", lambda _image: 0)
    with pytest.raises(ValueError, match="bounded multiple-of-14|Unsupported Lux geometry recipe"):
        da3_worker._run_inference(
            input_image=source,
            output_depth=depth,
            output_json=report,
            model_variant_name="METRIC_LARGE",
            model_key="da3_metric",
            device="cpu",
            use_coreml=False,
            non_commercial_ok=False,
            geometry_recipe=NORMALIZED_GEOMETRY_RECIPE,
        )
    assert loads == calls == []
    assert not depth.exists()
    assert not report.exists()
