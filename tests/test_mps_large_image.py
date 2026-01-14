#!/usr/bin/env python3
"""
Regression test for MPS large image 4× upscaling fix.

Critical regression: MPS buffer allocation failure with 3.86 GB tensor
(3600×6000 → 14400×24000 upscale).

Test ensures:
1. Upscaling succeeds without MPS allocation errors
2. Output dimensions are correct (4× input dimensions)
3. No silent failure (same-size output)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v2.config import PipelineConfig
from lux_depth_v2.pipeline import LuxPipelineV2

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_large_image_4x_upscale():
    """
    Regression test for MPS 3.86 GB buffer overflow fix.

    Scenario: 3600×6000 → 14400×24000 upscale (4×)
    - Input: 3600×6000 RGB (61.6 MP)
    - Output: 14400×24000 RGB (984.6 MP)
    - Buffer: 3 × 14400 × 24000 × 4 bytes = 3.86 GB (float32)

    Expected:
    - No MPS allocation errors
    - Output dimensions = 14400×24000
    - File size significantly larger than master

    Previous bug: torch_ops.Tiler created same-size output buffer,
    silently truncating upscaled tiles → no upscaling occurred.
    """
    # Create test configuration
    cfg = PipelineConfig(
        preset="interior_luxury",
        device="mps",
        upscale=4,
        upscaler_backend="torch",
        write_outputs=False,  # Don't write files during test
        save_master=False,
        save_upscaled=False,
    )
    cfg.apply_preset()

    # Create pipeline
    pipeline = LuxPipelineV2(cfg=cfg)

    # Create test image (3600×6000 RGB)
    input_h, input_w = 3600, 6000
    test_image = np.random.rand(input_h, input_w, 3).astype(np.float32)

    # Convert to torch tensor
    import torch

    test_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).to("mps")
    assert test_tensor.shape == (1, 3, input_h, input_w), "Test tensor shape mismatch"

    # Test upscaling
    try:
        # Use upscaler directly (matches pipeline.py line 958)
        upscaled = pipeline.upscaler.upscale(test_tensor)

        # Verify dimensions
        expected_h = input_h * cfg.upscale  # 14400
        expected_w = input_w * cfg.upscale  # 24000

        assert upscaled.shape == (
            1,
            3,
            expected_h,
            expected_w,
        ), f"Upscale failed: expected {(1, 3, expected_h, expected_w)}, got {upscaled.shape}"

        # Verify output is not same as input (no silent truncation)
        assert upscaled.shape[2] == expected_h, f"Height mismatch: {upscaled.shape[2]} != {expected_h}"
        assert upscaled.shape[3] == expected_w, f"Width mismatch: {upscaled.shape[3]} != {expected_w}"

        # Verify no NaN/Inf in output
        assert not torch.isnan(upscaled).any(), "Output contains NaN"
        assert not torch.isinf(upscaled).any(), "Output contains Inf"

        # Verify values in valid range
        assert upscaled.min() >= 0.0, f"Output has negative values: {upscaled.min()}"
        assert upscaled.max() <= 1.0, f"Output exceeds 1.0: {upscaled.max()}"

        print(f"✅ MPS large image upscale SUCCESS: {input_h}×{input_w} → {expected_h}×{expected_w}")

    except RuntimeError as e:
        if "MPS" in str(e) or "buffer" in str(e).lower():
            pytest.fail(
                f"MPS allocation failure (regression): {e}\n"
                f"Expected: Tiled upscaling should handle 3.86 GB buffer safely.\n"
                f"Check: lux_depth_v2/pipeline.py upscaling logic (lines 916-951)"
            )
        else:
            raise


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_bicubic_fallback():
    """Verify bicubic → bilinear fallback on MPS."""
    from lux_depth_v2 import torch_ops

    # Create MPS tensor
    test_tensor = torch.rand(1, 3, 512, 512, device="mps")

    # Request bicubic (should auto-fallback to bilinear)
    result = torch_ops.resize(test_tensor, (1024, 1024), mode="bicubic", autocast=False)

    # Verify success
    assert result.shape == (1, 3, 1024, 1024), f"Resize failed: {result.shape}"
    assert result.device.type == "mps", "Result should stay on MPS"

    print("✅ MPS bicubic fallback working (bicubic → bilinear)")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
def test_torch_upscaler_tiled_method():
    """Verify TorchUpscaler._upscale_tiled() creates correct output buffer size."""
    from lux_depth_v2.upscaling import TorchUpscaler

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Minimal config with tiling enabled
    class MinimalConfig:
        upscale = 4

        class Phase2Config:
            tile_based_upscaling = True
            upscale_tile_size = 512
            upscale_overlap = 64

        phase2 = Phase2Config()

    cfg = MinimalConfig()
    upscaler = TorchUpscaler(cfg, device)

    # Create test image that triggers tiling
    test_h, test_w = 1024, 1024
    test_tensor = torch.rand(1, 3, test_h, test_w, device=device)

    # Upscale
    result = upscaler.upscale(test_tensor)

    # Verify dimensions
    expected_h, expected_w = test_h * 4, test_w * 4
    assert result.shape == (1, 3, expected_h, expected_w), f"Tiled upscale failed: {result.shape}"

    print(f"✅ TorchUpscaler tiled method working: {test_h}×{test_w} → {expected_h}×{expected_w}")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("MPS Large Image Upscaling Regression Tests")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available, skipping tests")
        sys.exit(0)

    if not torch.backends.mps.is_available():
        print("⚠️  MPS not available, skipping MPS-specific tests")
        sys.exit(0)

    try:
        test_mps_bicubic_fallback()
        test_torch_upscaler_tiled_method()
        test_mps_large_image_4x_upscale()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
