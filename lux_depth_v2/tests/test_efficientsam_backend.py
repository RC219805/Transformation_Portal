# lux_depth_v2/tests/test_efficientsam_backend.py

import numpy as np
import pytest

from lux_depth_v2.backends.efficientsam_backend import (
    EfficientSAMBackend,
    EfficientSAMNotAvailable,
    PointPrompt,
    BoxPrompt,
)


def test_backend_available_flag_without_onnx(monkeypatch):
    # Simulate missing onnxruntime
    import lux_depth_v2.backends.efficientsam_backend as backend_mod

    monkeypatch.setattr(backend_mod, "ort", None, raising=False)

    backend = EfficientSAMBackend(lazy_load=True)
    assert backend.available is False

    with pytest.raises(EfficientSAMNotAvailable):
        backend.segment(np.zeros((32, 32, 3), dtype=np.uint8), [PointPrompt(0.5, 0.5)])


def test_preprocess_builds_prompt_tensors():
    backend = EfficientSAMBackend(lazy_load=True)

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    prompts = [
        PointPrompt(0.25, 0.25, 1),
        PointPrompt(0.75, 0.75, 0),
        BoxPrompt(0.1, 0.1, 0.9, 0.9),
    ]

    img_out, tensors = backend._preprocess(img, prompts)

    assert img_out.shape == (16, 16, 3)
    assert img_out.dtype == np.float32
    assert "points" in tensors and "boxes" in tensors
    assert tensors["points"].shape == (2, 3)
    assert tensors["boxes"].shape == (1, 4)


@pytest.mark.skip(reason="Stage 2: requires real EfficientSAM ONNX model and I/O wiring")
def test_segment_runs_with_real_model():
    backend = EfficientSAMBackend(
        model_path="weights/efficientsam/efficientsam_ti_vit_s.onnx",
        lazy_load=False,
    )
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = backend.segment(img, [PointPrompt(0.5, 0.5)])
    assert mask.shape == (32, 32)
    assert mask.dtype == np.float32
