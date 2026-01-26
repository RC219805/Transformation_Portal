from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lux_depth_v3.config import DA3Config
from lux_depth_v3.enhance.preprocessing import normalize_exif_orientation
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.input_manager import ImageInput


@pytest.fixture
def tiny_rgb_png(tmp_path: Path) -> Path:
    # 32x32 RGB image
    import PIL.Image

    p = tmp_path / "tiny.png"
    img = PIL.Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8), mode="RGB")
    img.save(p)
    return p


def _patch_model_backend(monkeypatch, available: bool) -> None:
    # Patch DA3ModelBackend so it never downloads HF assets.
    import lux_depth_v3.da3_model_backend as mb

    monkeypatch.setattr(mb.DA3ModelBackend, "is_available", lambda self: available)

    def fake_predict(self, rgb01):
        # Return a stable synthetic depth map.
        h, w = rgb01.shape[:2]
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        return np.repeat(y, w, axis=1)

    monkeypatch.setattr(mb.DA3ModelBackend, "predict_depth01_from_rgb01", fake_predict)


def test_model_backend_selected_when_available(monkeypatch, tiny_rgb_png: Path) -> None:
    _patch_model_backend(monkeypatch, available=True)

    cfg = DA3Config()
    engine = DA3InferenceEngine(config=cfg, commercial_use=False, validate_license_strict=False)

    # Prepare input (match orchestrator behavior: normalize EXIF)
    tmp_norm = tiny_rgb_png.parent / "norm.png"
    normalize_exif_orientation(tiny_rgb_png, tmp_norm)
    out = engine.predict(ImageInput(path=tmp_norm))

    assert out.depth_map.shape == (32, 32)
    # The engine should report model backend in metadata (added in your PR)
    assert out.metadata.get("inference_mode") in ("model_backend", "model-backend", "model_backend_hf")


def test_kill_switch_disables_model_backend(monkeypatch, tiny_rgb_png: Path) -> None:
    _patch_model_backend(monkeypatch, available=True)
    monkeypatch.setenv("LUX_DA3_DISABLE_MODEL_BACKEND", "1")

    cfg = DA3Config()
    engine = DA3InferenceEngine(config=cfg, commercial_use=False, validate_license_strict=False)

    tmp_norm = tiny_rgb_png.parent / "norm.png"
    normalize_exif_orientation(tiny_rgb_png, tmp_norm)

    out = engine.predict(ImageInput(path=tmp_norm))

    # Must NOT claim model backend if kill-switch set.
    assert out.metadata.get("inference_mode") not in ("model_backend", "model-backend", "model_backend_hf")
