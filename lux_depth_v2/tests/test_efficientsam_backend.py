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


def test_backend_available_with_model_missing(monkeypatch, tmp_path):
    """Stage 5B: available is False when model doesn't exist (stricter semantics)."""
    import lux_depth_v2.backends.efficientsam_backend as backend_mod
    
    # Mock onnxruntime as available
    monkeypatch.setattr(backend_mod, "ort", type("ort", (), {"InferenceSession": type}))
    
    # Model doesn't exist, auto_download=False
    backend = EfficientSAMBackend(
        model_name="missing_model",
        cache_dir=tmp_path,
        auto_download=False,
        lazy_load=True,
    )
    
    # Stage 5B: available should be False (model missing)
    assert backend.available is False


def test_backend_available_with_model_present(monkeypatch, tmp_path):
    """Stage 5B: available is True when model exists."""
    import lux_depth_v2.backends.efficientsam_backend as backend_mod
    
    # Mock onnxruntime
    monkeypatch.setattr(backend_mod, "ort", type("ort", (), {"InferenceSession": type}))
    
    # Create model file
    model_name = "test_model"
    model_file = tmp_path / f"{model_name}.onnx"
    model_file.touch()
    
    backend = EfficientSAMBackend(
        model_path=model_file,
        lazy_load=True,
    )
    
    # Stage 5B: available should be True
    assert backend.available is True


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


def _model_exists() -> bool:
    """Check if efficientsam_s.onnx model exists locally."""
    from pathlib import Path
    return (Path("weights") / "efficientsam" / "efficientsam_s.onnx").exists()


@pytest.mark.skipif(not _model_exists(), reason="efficientsam_s.onnx model not available")
def test_segment_runs_with_real_model_efficientsam_s():
    """
    Stage 5A: Real ONNX inference test with efficientsam_s.onnx.
    
    Only runs when model is present (e.g., after manual download via CLI).
    CI skips this test by default (offline, no model).
    """
    backend = EfficientSAMBackend(
        model_name="efficientsam_s",
        lazy_load=False,
    )
    
    # Create simple test image: 64x64 with a white square in center
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:44, 20:44] = 255  # white square
    
    # Test with box prompt around the square
    mask = backend.segment(img, [BoxPrompt(0.25, 0.25, 0.75, 0.75)])
    
    # Validate output
    assert mask.shape == (64, 64), f"Expected (64,64), got {mask.shape}"
    assert mask.dtype == np.float32
    assert np.all(np.isfinite(mask)), "Mask contains NaN or Inf"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, f"Mask values outside [0,1]: [{mask.min()}, {mask.max()}]"
    
    # Mask should not be constant (variance > epsilon)
    assert mask.std() > 0.01, "Mask is constant (no segmentation detected)"
    
    # Center region should have higher confidence than edges (basic sanity)
    center_val = mask[32, 32]
    edge_val = mask[4, 4]
    assert center_val > edge_val, f"Center ({center_val}) should be > edge ({edge_val})"


@pytest.mark.skipif(not _model_exists(), reason="efficientsam_s.onnx model not available")
def test_segment_with_point_prompts_real_model():
    """Stage 5A: Test with point prompts on real model."""
    backend = EfficientSAMBackend(model_name="efficientsam_s", lazy_load=False)
    
    img = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    
    # Single foreground point at center
    mask = backend.segment(img, [PointPrompt(0.5, 0.5, label=1)])
    
    assert mask.shape == (48, 48)
    assert mask.dtype == np.float32
    assert np.all(np.isfinite(mask))
    assert mask.std() > 0.01


@pytest.mark.skip(reason="Requires real EfficientSAM ONNX model; use mocked tests for CI")
def test_segment_runs_with_real_model():
    backend = EfficientSAMBackend(
        model_path="weights/efficientsam/efficientsam_ti_vit_s.onnx",
        lazy_load=False,
    )
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = backend.segment(img, [PointPrompt(0.5, 0.5)])
    assert mask.shape == (32, 32)
    assert mask.dtype == np.float32


# ============================================================================
# Stage 4: Mocked ONNX Runtime Tests (CI-safe, no model required)
# ============================================================================


class MockONNXInput:
    def __init__(self, name):
        self.name = name


class MockONNXOutput:
    def __init__(self, name):
        self.name = name


class MockONNXSession:
    """Mock ONNX InferenceSession for testing without real model."""

    def __init__(self, model_path, providers=None):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self._inputs = [MockONNXInput("image"), MockONNXInput("boxes")]
        self._outputs = [MockONNXOutput("mask")]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, input_feed):
        """Return a synthetic mask based on input shape."""
        # Extract image shape from feed dict
        img_key = "image" if "image" in input_feed else list(input_feed.keys())[0]
        img_tensor = input_feed[img_key]  # (1, 3, H, W)

        _, _, h, w = img_tensor.shape

        # Create a simple synthetic mask (center circle-like pattern)
        mask = np.zeros((1, 1, h, w), dtype=np.float32)
        cy, cx = h // 2, w // 2
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                if dist < min(h, w) / 4:
                    mask[0, 0, y, x] = 0.8

        return [mask]


def test_prepare_onnx_inputs_box_prompts(monkeypatch, tmp_path):
    """Test ONNX input preparation with box prompts."""
    import lux_depth_v2.backends.efficientsam_backend as backend_mod

    # Mock onnxruntime
    monkeypatch.setattr(backend_mod, "ort", type("ort", (), {"InferenceSession": MockONNXSession}))

    # Create fake model file
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    backend = EfficientSAMBackend(model_path=model_path, lazy_load=False)

    # Prepare inputs
    img = np.random.rand(32, 32, 3).astype(np.float32)
    prompts = [BoxPrompt(0.1, 0.1, 0.9, 0.9)]
    img_preprocessed, prompt_tensors = backend._preprocess(img, prompts)

    feed = backend._prepare_onnx_inputs(img_preprocessed, prompt_tensors, 32, 32)

    # Stage 5A: Check actual efficientsam_s tensor names
    assert "batched_images" in feed
    assert feed["batched_images"].shape == (1, 3, 32, 32)

    # Box converted to point prompts (center only in Stage 5A)
    assert "batched_point_coords" in feed
    assert "batched_point_labels" in feed
    coords = feed["batched_point_coords"]
    labels = feed["batched_point_labels"]
    assert coords.shape == (1, 1, 1, 2)  # 1 box → 1 center point
    assert labels.shape == (1, 1, 1)
    # Center of box (0.5, 0.5) in pixel coords = (16, 16)
    np.testing.assert_allclose(coords[0, 0, 0], [16.0, 16.0], rtol=1e-5)
    assert labels[0, 0, 0] == 1.0  # foreground


def test_postprocess_outputs_handles_4d_tensor():
    """Test postprocessing of typical (1, 1, H, W) output."""
    backend = EfficientSAMBackend(lazy_load=True)

    # Simulate ONNX output
    raw_output = np.random.rand(1, 1, 64, 64).astype(np.float32)
    outputs = [raw_output]

    mask = backend._postprocess_outputs(outputs, h_orig=64, w_orig=64)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert 0.0 <= mask.min() <= mask.max() <= 1.0


def test_postprocess_outputs_applies_sigmoid_to_logits():
    """Test sigmoid application when output contains logits."""
    backend = EfficientSAMBackend(lazy_load=True)

    # Simulate logit output (values outside [0,1])
    logits = np.array([[[-5.0, 0.0, 5.0]]]).astype(np.float32)  # (1, 1, 3)
    outputs = [logits]

    mask = backend._postprocess_outputs(outputs, h_orig=1, w_orig=3)

    # Check sigmoid applied
    assert mask.shape == (1, 3)
    assert mask[0, 0] < 0.1  # sigmoid(-5) ≈ 0.007
    assert 0.4 < mask[0, 1] < 0.6  # sigmoid(0) = 0.5
    assert mask[0, 2] > 0.9  # sigmoid(5) ≈ 0.993


def test_postprocess_outputs_resizes_when_needed():
    """Test output resizing to match original image dimensions."""
    backend = EfficientSAMBackend(lazy_load=True)

    # Output is smaller than original
    raw_output = np.ones((1, 1, 16, 16), dtype=np.float32) * 0.7
    outputs = [raw_output]

    # Request resize to 32x32
    mask = backend._postprocess_outputs(outputs, h_orig=32, w_orig=32)

    assert mask.shape == (32, 32)
    # Values should be close to 0.7 (interpolated)
    assert 0.65 < mask.mean() < 0.75


def test_segment_end_to_end_with_mocked_onnx(monkeypatch, tmp_path):
    """Test full segment() path with mocked ONNX runtime (Stage 4 complete)."""
    import lux_depth_v2.backends.efficientsam_backend as backend_mod

    # Mock onnxruntime
    monkeypatch.setattr(backend_mod, "ort", type("ort", (), {"InferenceSession": MockONNXSession}))

    # Create fake model file
    model_path = tmp_path / "efficientsam_ti_vit_s.onnx"
    model_path.touch()

    backend = EfficientSAMBackend(model_path=model_path, lazy_load=False)

    # Run segment
    img = np.random.rand(64, 64, 3).astype(np.uint8)
    prompts = [BoxPrompt(0.2, 0.2, 0.8, 0.8)]

    mask = backend.segment(img, prompts)

    # Verify output
    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert 0.0 <= mask.min() <= mask.max() <= 1.0

    # Check mask has some structure (center should have higher values)
    center_val = mask[32, 32]
    corner_val = mask[0, 0]
    assert center_val > corner_val  # Mocked session creates center blob


def test_segment_raises_on_missing_model(monkeypatch, tmp_path):
    """Test that segment() raises clear error when model file doesn't exist."""
    import lux_depth_v2.backends.efficientsam_backend as backend_mod

    # Mock onnxruntime
    monkeypatch.setattr(backend_mod, "ort", type("ort", (), {"InferenceSession": MockONNXSession}))

    # Point to non-existent model
    model_path = tmp_path / "nonexistent.onnx"

    backend = EfficientSAMBackend(model_path=model_path, lazy_load=True)

    img = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(EfficientSAMNotAvailable, match="not available"):
        backend.segment(img, [PointPrompt(0.5, 0.5)])
