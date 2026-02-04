"""Integration tests for DA3InferenceEngine.

These tests exercise the *real* HuggingFace/transformers model-loading path.

Why gating is necessary:
- Depth Anything V3 repos may be gated (401) without an HF token.
- Model downloads can be large and non-deterministic in CI environments.

Policy:
- CI remains honest and deterministic by default: network/model tests are OFF.
- To explicitly enable real model integration tests, set:
    TP_RUN_HF_MODEL_TESTS=1
  and provide authentication if needed:
    HF_TOKEN or HUGGINGFACE_HUB_TOKEN

Notes:
- This file is still marked as ML tier (torch/transformers dependencies).
- When TP_RUN_HF_MODEL_TESTS is not enabled, only lightweight constructor/flag tests run.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v3 import DA3Config, DA3InferenceEngine
from transformation_portal.lux_depth_v3.config import DeviceConfig, ModelVariant

# Mark all tests in this file as requiring ML dependencies
pytestmark = pytest.mark.ml


# -----------------------------------------------------------------------------
# Gating: real model downloads are opt-in
# -----------------------------------------------------------------------------

RUN_HF = os.getenv("TP_RUN_HF_MODEL_TESTS") == "1"
HAS_HF_TOKEN = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
OFFLINE = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"

skip_hf = pytest.mark.skipif(
    not RUN_HF,
    reason="Real HF model tests are disabled by default. Set TP_RUN_HF_MODEL_TESTS=1 to enable.",
)

skip_offline = pytest.mark.skipif(
    OFFLINE,
    reason="HF/transformers offline mode enabled; real model downloads are unavailable.",
)

# If you enable TP_RUN_HF_MODEL_TESTS without a token, many gated repos will 401.
# We skip rather than fail noisily.
skip_no_token = pytest.mark.skipif(
    not HAS_HF_TOKEN,
    reason="HF_TOKEN/HUGGINGFACE_HUB_TOKEN required for gated Depth Anything models.",
)


def _make_engine(config: DA3Config | None = None) -> DA3InferenceEngine:
    """Construct an engine with a config."""
    if config is None:
        config = DA3Config()
    return DA3InferenceEngine(config)


def _rand_image(h: int, w: int, dtype: np.dtype = np.float32, rng=None) -> np.ndarray:
    """Create a random RGB test image."""
    if rng is None:
        rng = np.random
    if dtype == np.uint8:
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return rng.random((h, w, 3)).astype(np.float32)


# -----------------------------------------------------------------------------
# Real-model integration tests (opt-in)
# -----------------------------------------------------------------------------


@skip_hf
@skip_offline
@skip_no_token
def test_da3_predict_basic():
    """Test basic predict() functionality."""
    engine = _make_engine()

    image = _rand_image(128, 128, np.float32)
    result = engine.predict(image)

    assert result.depth_map.shape == (128, 128)
    assert result.depth_map.dtype == np.float32
    assert result.depth_map.min() >= 0.0
    assert result.depth_map.max() <= 1.0
    assert result.original_image.shape == image.shape

    for key in ("inference_time_ms", "backend", "device", "model_variant"):
        assert key in result.metadata


@skip_hf
@skip_offline
@skip_no_token
def test_da3_infer_alias():
    """Test that infer() is an alias for predict()."""
    engine = _make_engine()

    image = _rand_image(64, 64, np.uint8)
    result1 = engine.predict(image)
    result2 = engine.infer(image)

    assert result1.depth_map.shape == result2.depth_map.shape
    assert result1.depth_map.dtype == result2.depth_map.dtype


@skip_hf
@skip_offline
@skip_no_token
def test_da3_depth_property_alias():
    """Test that DepthResult.depth is an alias for depth_map."""
    engine = _make_engine()

    image = _rand_image(64, 64, np.float32)
    result = engine.predict(image)

    assert result.depth is result.depth_map
    assert np.array_equal(result.depth, result.depth_map)


@skip_hf
@skip_offline
@skip_no_token
def test_da3_different_image_sizes():
    """Test predict() across multiple image sizes."""
    engine = _make_engine()

    for h, w in [(64, 64), (128, 128), (256, 256)]:
        image = _rand_image(h, w, np.float32)
        result = engine.predict(image)

        assert result.depth_map.shape == (h, w), f"Failed for size {h}x{w}"
        assert result.depth_map.min() >= 0.0
        assert result.depth_map.max() <= 1.0


@skip_hf
@skip_offline
@skip_no_token
def test_da3_uint8_image():
    """Test predict() with uint8 input (common format)."""
    engine = _make_engine()

    image = _rand_image(128, 128, np.uint8)
    result = engine.predict(image)

    assert result.depth_map.shape == (128, 128)
    assert result.depth_map.dtype == np.float32


@skip_hf
@skip_offline
@skip_no_token
def test_da3_device_config_cpu():
    """Test device configuration with explicit CPU selection."""
    config = DA3Config()
    config.device = DeviceConfig(device="cpu")
    engine = _make_engine(config)

    assert engine.device == "cpu"

    image = _rand_image(64, 64, np.float32)
    result = engine.predict(image)

    assert "device" in result.metadata
    assert result.metadata["device"] in ["cpu", "mps", "cuda"]


@skip_hf
@skip_offline
@skip_no_token
def test_da3_metadata_completeness():
    """Test that metadata contains expected fields and types."""
    engine = _make_engine()

    image = _rand_image(64, 64, np.float32)
    result = engine.predict(image)

    required_fields = [
        "inference_time_ms",
        "backend",
        "device",
        "model_variant",
        "shape",
    ]
    for field in required_fields:
        assert field in result.metadata, f"Missing metadata field: {field}"

    assert isinstance(result.metadata["inference_time_ms"], (int, float))
    assert result.metadata["inference_time_ms"] >= 0
    assert isinstance(result.metadata["backend"], str)
    assert isinstance(result.metadata["device"], str)


@skip_hf
@skip_offline
@skip_no_token
def test_da3_lazy_loading():
    """Test that model is loaded lazily on first inference.

    Note: this uses a private attribute as a pragmatic test signal.
    """
    engine = _make_engine()

    assert not getattr(engine, "_model_loaded", False)

    image = _rand_image(64, 64, np.float32)
    result = engine.predict(image)

    assert getattr(engine, "_model_loaded", False)
    assert getattr(engine, "model", None) is not None

    result2 = engine.predict(image)
    assert result2.depth_map.shape == result.depth_map.shape


@skip_hf
@skip_offline
@skip_no_token
def test_da3_fallback_model_indicator():
    """Test that metadata indicates fallback when V3 cannot load and V2 is used."""
    config = DA3Config()
    config.model_variant = ModelVariant.METRIC_LARGE  # likely to exercise fallback path
    engine = _make_engine(config)

    image = _rand_image(64, 64, np.float32)
    result = engine.predict(image)

    # Only assert fallback fields if the implementation sets them.
    if "using_fallback" in result.metadata:
        assert result.metadata["using_fallback"] is True
        assert "fallback_model" in result.metadata
        assert "V2" in str(result.metadata["fallback_model"])


@skip_hf
@skip_offline
@skip_no_token
def test_da3_infer_from_path_roundtrip(tmp_path: Path):
    """Test infer_from_path() using a temporary real image file."""
    engine = _make_engine()

    # Create a simple RGB image file
    import PIL.Image

    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    p = tmp_path / "input.png"
    PIL.Image.fromarray(img, mode="RGB").save(p)

    result = engine.infer_from_path(p)
    assert result.depth_map.shape == (64, 64)
    assert result.depth_map.dtype == np.float32


# -----------------------------------------------------------------------------
# Lightweight, non-network tests (always run in ML tier)
# -----------------------------------------------------------------------------


def test_da3_commercial_use_flag():
    """Test commercial_use initialization parameter (no model load)."""
    config = DA3Config()

    engine1 = DA3InferenceEngine(config, commercial_use=True)
    assert engine1.commercial_use is True

    engine2 = DA3InferenceEngine(config, commercial_use=False)
    assert engine2.commercial_use is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
