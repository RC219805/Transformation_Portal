# lux_depth_v2/tests/test_stage4_end_to_end.py
"""
Stage 4 end-to-end integration tests.

Tests the complete EfficientSAM V3 pipeline with mocked ONNX runtime:
- EfficientSAMBackend (mocked ONNX)
- EfficientSAMRefinementProvider
- FusedMaterialSegmenter
- Fusion stats and fallback behavior
"""

import numpy as np
import pytest


# Mock ONNX runtime components (same as in test_efficientsam_backend.py)
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

        # Create refined mask: slightly expand base region from box prompts
        mask = np.zeros((1, 1, h, w), dtype=np.float32)

        if "boxes" in input_feed:
            boxes_px = input_feed["boxes"][0]  # (N, 4)
            for box in boxes_px:
                x0, y0, x1, y1 = box.astype(int)
                # Expand box slightly (refinement simulation)
                x0 = max(0, x0 - 2)
                y0 = max(0, y0 - 2)
                x1 = min(w, x1 + 2)
                y1 = min(h, y1 + 2)
                mask[0, 0, y0:y1, x0:x1] = 0.85

        return [mask]


@pytest.fixture
def mock_onnx_env(monkeypatch):
    """Set up mocked ONNX environment."""
    import lux_depth_v2.backends.efficientsam_backend as backend_mod

    monkeypatch.setattr(
        backend_mod, "ort", type("ort", (), {"InferenceSession": MockONNXSession})
    )


def test_stage4_complete_pipeline_with_real_backend(mock_onnx_env, tmp_path):
    """
    Test complete Stage 4 pipeline:
    EfficientSAMBackend (mocked ONNX) -> EfficientSAMRefinementProvider -> FusedMaterialSegmenter
    """
    from lux_depth_v2.backends.efficientsam_backend import EfficientSAMBackend
    from lux_depth_v2.backends.refinement_provider import EfficientSAMRefinementProvider
    from lux_depth_v2.material_segmentation import FusedMaterialSegmenter
    from lux_depth_v2.config import SegmentationConfig, SegmentationBackend, FusionMode
    from lux_depth_v2 import torch_ops

    torch_ops.require_torch()
    torch = torch_ops.torch

    # Create mocked EfficientSAM backend
    model_path = tmp_path / "efficientsam.onnx"
    model_path.touch()

    backend = EfficientSAMBackend(model_path=model_path, lazy_load=False)
    assert backend.available

    device = torch.device("cpu")
    provider = EfficientSAMRefinementProvider(backend, device)

    # Create config with fusion enabled
    cfg = SegmentationConfig(
        backend_v3=SegmentationBackend.FUSED,
        fusion_mode=FusionMode.CONFIDENCE_WEIGHTED,
        fusion_min_iou=0.2,
    )

    # Create dummy base segmenter (returns simple masks)
    class DummyBaseSegmenter:
        def __init__(self, device):
            self.device = device

        def segment(self, rgb):
            """Return a simple centered square mask."""
            _, _, h, w = rgb.shape
            mask = torch.zeros((1, 1, h, w), device=self.device, dtype=torch.float32)
            # Simple square in center
            h0, h1 = h // 4, 3 * h // 4
            w0, w1 = w // 4, 3 * w // 4
            mask[0, 0, h0:h1, w0:w1] = 0.8
            return mask

    base_segmenter = DummyBaseSegmenter(device)

    # Create fused segmenter
    fused = FusedMaterialSegmenter(base_segmenter, cfg, provider)

    # Run segmentation
    rgb = torch.rand(1, 3, 64, 64, device=device, dtype=torch.float32)
    result = fused.predict(rgb)

    # Verify output - FusedMaterialSegmenter.predict() returns dict of masks
    assert isinstance(result, dict)
    # Should have at least water (our default edge refinement class)
    assert len(result) > 0
    
    # Check one of the masks
    sample_mask = next(iter(result.values()))
    assert sample_mask.shape == (1, 1, 64, 64)
    assert sample_mask.dtype == torch.float32
    assert 0.0 <= sample_mask.min() <= sample_mask.max() <= 1.0

    # Check fusion stats
    stats = fused.get_fusion_stats()
    assert "water" in stats  # Default edge class
    assert "iou_base_vs_refined" in stats["water"]
    assert "fusion_applied" in stats["water"]


def test_stage4_real_backend_generates_different_mask(mock_onnx_env, tmp_path):
    """
    Test that EfficientSAM refinement actually produces a different mask than base.
    """
    from lux_depth_v2.backends.efficientsam_backend import EfficientSAMBackend, BoxPrompt
    from lux_depth_v2 import torch_ops

    torch_ops.require_torch()
    torch = torch_ops.torch

    # Create backend
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    backend = EfficientSAMBackend(model_path=model_path, lazy_load=False)

    # Create base mask
    base_np = np.zeros((64, 64), dtype=np.float32)
    base_np[20:40, 20:40] = 0.8  # Simple square

    # Create RGB image
    rgb_np = np.random.rand(64, 64, 3).astype(np.float32)

    # Generate refined mask via EfficientSAM
    box = BoxPrompt(20.0 / 64, 20.0 / 64, 40.0 / 64, 40.0 / 64)
    refined_np = backend.segment(rgb_np, [box])

    # Verify refined mask is different (mock expands by 2 pixels)
    assert refined_np.shape == (64, 64)
    assert not np.allclose(refined_np, base_np)

    # Check expansion happened (refined should have more non-zero pixels)
    assert refined_np.sum() > base_np.sum()


def test_stage4_fallback_on_empty_base_mask(mock_onnx_env, tmp_path):
    """
    Test that refinement provider handles empty base masks gracefully.
    """
    from lux_depth_v2.backends.efficientsam_backend import EfficientSAMBackend
    from lux_depth_v2.backends.refinement_provider import EfficientSAMRefinementProvider
    from lux_depth_v2 import torch_ops

    torch_ops.require_torch()
    torch = torch_ops.torch

    model_path = tmp_path / "model.onnx"
    model_path.touch()
    backend = EfficientSAMBackend(model_path=model_path, lazy_load=False)

    device = torch.device("cpu")
    provider = EfficientSAMRefinementProvider(backend, device)

    # Empty base mask
    rgb = torch.rand(1, 3, 64, 64, device=device)
    base_mask = torch.zeros(1, 1, 64, 64, device=device)

    # Should return None (no refinement possible)
    refined = provider.get_refined_mask(rgb, base_mask, "water")
    assert refined is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
