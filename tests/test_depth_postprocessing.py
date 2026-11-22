import sys
from unittest.mock import Mock

import numpy as np
import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

# Save original modules before mocking
_ORIGINAL_TORCH = sys.modules.get("torch")
_ORIGINAL_TRANSFORMERS = sys.modules.get("transformers")
_ORIGINAL_COREMLTOOLS = sys.modules.get("coremltools")

# Mock torch and other heavy dependencies before importing pipeline
sys.modules["torch"] = Mock()
sys.modules["transformers"] = Mock()
sys.modules["coremltools"] = Mock()

import transformation_portal.depth.pipeline as pipeline_mod
import transformation_portal.depth.utils.depth_utils as depth_utils_mod
from transformation_portal.depth.pipeline import ArchitecturalDepthPipeline

# Immediately restore sys.modules to prevent pollution
# The pipeline module is already imported and cached, so this is safe
if _ORIGINAL_TORCH is not None:
    sys.modules["torch"] = _ORIGINAL_TORCH
else:
    sys.modules.pop("torch", None)

if _ORIGINAL_TRANSFORMERS is not None:
    sys.modules["transformers"] = _ORIGINAL_TRANSFORMERS
else:
    sys.modules.pop("transformers", None)

if _ORIGINAL_COREMLTOOLS is not None:
    sys.modules["coremltools"] = _ORIGINAL_COREMLTOOLS
else:
    sys.modules.pop("coremltools", None)


class _DummyVariant:
    name = "DUMMY"


class _DummyDepthModel:
    variant = _DummyVariant()

    def __init__(self, depth: np.ndarray):
        self._depth = depth

    def estimate_depth(self, image):
        return {
            "depth": self._depth,
            "metadata": {"inference_time_ms": 0.0},
        }


class _DummyCache:
    def get_or_compute(self, image, compute):
        return compute()

    def get_stats(self):
        return {"hit_rate": 0.0, "size": 0, "max_size": 0}

    def clear(self, clear_disk: bool = False):
        return None


def _build_pipeline(monkeypatch, config, depth):
    monkeypatch.setattr(
        pipeline_mod.ArchitecturalDepthPipeline,
        "_init_depth_model",
        lambda self: _DummyDepthModel(depth),
    )
    monkeypatch.setattr(
        pipeline_mod.ArchitecturalDepthPipeline,
        "_init_cache",
        lambda self: _DummyCache(),
    )
    monkeypatch.setattr(
        pipeline_mod.ArchitecturalDepthPipeline,
        "_init_processors",
        lambda self: {},
    )
    monkeypatch.setattr(
        pipeline_mod,
        "load_image",
        lambda path, normalize=True: np.zeros((4, 4, 3), dtype=np.float32),
    )
    return ArchitecturalDepthPipeline(config)


def test_depth_postprocessing_disabled_skips_smoothing(monkeypatch):
    depth = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    config = {
        "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
        "processing": {},
    }

    calls = {"count": 0}

    def fake_smooth_depth(*args, **kwargs):
        calls["count"] += 1
        return np.full_like(depth, 0.5, dtype=np.float32)

    monkeypatch.setattr(pipeline_mod, "smooth_depth", fake_smooth_depth)

    pipeline = _build_pipeline(monkeypatch, config, depth)
    out = pipeline.process_render("dummy.png")["depth"]

    assert calls["count"] == 0
    assert np.array_equal(out, depth)


def test_depth_postprocessing_enabled_applies_smoothing_with_scale_preserved(monkeypatch):
    depth = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    config = {
        "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
        "processing": {
            "depth_postprocessing": {
                "enabled": True,
                "method": "bilateral",
                "sigma": 5.0,
                "edge_preserve": 0.1,
                "preserve_scale": True,
            }
        },
    }

    def fake_smooth_depth(depth, method="bilateral", sigma=5.0, edge_preserve=0.1):
        # Return a normalized [0,1] "smoothed" map to exercise preserve_scale logic.
        return np.full_like(depth, 0.5, dtype=np.float32)

    monkeypatch.setattr(pipeline_mod, "smooth_depth", fake_smooth_depth)

    pipeline = _build_pipeline(monkeypatch, config, depth)
    out = pipeline.process_render("dummy.png")["depth"]

    # min=10, max=40 => midpoint = 25
    assert np.allclose(out, 25.0)


def test_depth_postprocessing_unknown_method_is_a_noop(monkeypatch):
    depth = np.array([[1.0, 2.0]], dtype=np.float32)
    config = {
        "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
        "processing": {
            "depth_postprocessing": {
                "enabled": True,
                "method": "not-a-method",
                "sigma": 5.0,
                "edge_preserve": 0.1,
            }
        },
    }

    def fake_smooth_depth(*args, **kwargs):
        raise AssertionError("smooth_depth should not be called for unknown method")

    monkeypatch.setattr(pipeline_mod, "smooth_depth", fake_smooth_depth)

    pipeline = _build_pipeline(monkeypatch, config, depth)
    out = pipeline.process_render("dummy.png")["depth"]

    assert np.array_equal(out, depth)


def test_depth_postprocessing_median_default_sigma_smooths_float_depth(monkeypatch):
    """Regression (D-03): median at the default sigma must not be a silent no-op for float32 depth."""
    rng = np.random.default_rng(0)
    ramp = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))
    depth = (ramp + rng.normal(0, 0.05, (32, 32))).astype(np.float32)
    config = {
        "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
        "processing": {"depth_postprocessing": {"enabled": True, "method": "median", "sigma": 5.0}},
    }

    pipeline = _build_pipeline(monkeypatch, config, depth)
    out = pipeline.process_render("dummy.png")["depth"]

    assert out.shape == depth.shape
    assert not np.array_equal(out, depth)
    # A median filter reduces the injected per-pixel noise.
    assert float(np.abs(np.diff(out, axis=0)).mean()) < float(np.abs(np.diff(depth, axis=0)).mean())


@pytest.mark.skipif(not depth_utils_mod.CV2_AVAILABLE, reason="opencv-python required")
@pytest.mark.parametrize("preserve_scale", [True, False])
def test_depth_postprocessing_bilateral_keeps_depth_range(monkeypatch, preserve_scale):
    """Regression (D-06): bilateral smoothing returned depth renormalised to [0, 1]."""
    rng = np.random.default_rng(1)
    ramp = np.tile(np.linspace(0.2, 0.9, 32, dtype=np.float32), (32, 1))
    depth = (ramp + rng.normal(0, 0.01, (32, 32))).astype(np.float32)
    config = {
        "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
        "processing": {
            "depth_postprocessing": {
                "enabled": True,
                "method": "bilateral",
                "sigma": 2.0,
                "edge_preserve": 0.1,
                "preserve_scale": preserve_scale,
            }
        },
    }

    pipeline = _build_pipeline(monkeypatch, config, depth)
    out = pipeline.process_render("dummy.png")["depth"]

    assert out.shape == depth.shape
    assert float(out.min()) >= float(depth.min()) - 1e-6
    assert float(out.max()) <= float(depth.max()) + 1e-6
    # Within the 8-bit quantisation step of the input span of the original values.
    np.testing.assert_allclose(out, depth, atol=0.05)


def _marking_postprocess_config():
    return {
        "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
        "processing": {"depth_postprocessing": {"enabled": True, "method": "gaussian", "sigma": 1.0}},
        "output": {},
    }


def _marking_smooth_depth(depth, method="bilateral", sigma=5.0, edge_preserve=0.1):
    # A recognisable "smoothed" map: the raw depth shifted by +100.
    return np.asarray(depth, dtype=np.float32) + 100.0


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(
            lambda p, paths, out: p.batch_process(paths, out, False, False, parallel=True, max_workers=2),
            id="parallel",
        ),
        pytest.param(
            lambda p, paths, out: p.batch_process(paths, out, False, False, parallel=False, preload_images=True),
            id="sequential",
        ),
        pytest.param(lambda p, paths, out: list(p.batch_process_pipelined(paths, out, False, False)), id="pipelined"),
        pytest.param(lambda p, paths, out: list(p.batch_process_streaming(paths, out, False, False)), id="streaming"),
        pytest.param(
            lambda p, paths, out: p.process_render_progressive(paths[0], [1.0], return_all_levels=True),
            id="progressive",
        ),
    ],
)
def test_every_processing_path_applies_depth_postprocessing(monkeypatch, tmp_path, run):
    """Regression (D-02): batch/pipelined/progressive paths used raw depth while process_render smoothed it."""
    depth = np.tile(np.linspace(0.0, 1.0, 4, dtype=np.float32), (4, 1))
    monkeypatch.setattr(pipeline_mod, "smooth_depth", _marking_smooth_depth)

    pipeline = _build_pipeline(monkeypatch, _marking_postprocess_config(), depth)
    expected = pipeline.process_render("a.png")["depth"]
    np.testing.assert_array_equal(expected, depth + 100.0)

    results = run(pipeline, [str(tmp_path / "a.png"), str(tmp_path / "b.png")], tmp_path / "out")

    assert results
    for result in results:
        np.testing.assert_array_equal(result["depth"], expected)
        assert result["metadata"]["depth_stats"] == pipeline_mod.depth_statistics(expected)


def test_depth_postprocessing_leaves_the_cached_depth_result_unchanged(monkeypatch):
    """The postprocessed depth must not be written back into the model/cache result."""
    depth = np.tile(np.linspace(0.0, 1.0, 4, dtype=np.float32), (4, 1))
    raw = depth.copy()
    monkeypatch.setattr(pipeline_mod, "smooth_depth", _marking_smooth_depth)

    pipeline = _build_pipeline(monkeypatch, _marking_postprocess_config(), depth)
    depth_result, smoothed = pipeline._estimate_depth(np.zeros((4, 4, 3), dtype=np.float32))

    np.testing.assert_array_equal(depth_result["depth"], raw)
    np.testing.assert_array_equal(smoothed, raw + 100.0)
