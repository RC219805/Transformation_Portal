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
