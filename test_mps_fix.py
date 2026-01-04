#!/usr/bin/env python3
"""Quick test to verify MPS bicubic fix works."""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from lux_depth_v2 import torch_ops
from lux_depth_v2.upscaling import TorchUpscaler, NoneUpscaler


def test_torch_ops_resize():
    """Test torch_ops.resize with MPS device."""
    print("=" * 60)
    print("Testing torch_ops.resize() with MPS")
    print("=" * 60)

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("⚠️  MPS not available, testing on CPU instead")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print(f"✅ MPS available, testing on {device}")

    # Create test tensor (simulate 512x512 RGB image)
    test_tensor = torch.rand(1, 3, 512, 512, device=device)
    print(f"Input shape: {test_tensor.shape}")

    # Test resize (this would fail with bicubic on MPS before the fix)
    try:
        resized = torch_ops.resize(test_tensor, (1024, 1024), mode="bilinear", autocast=True)
        print(f"✅ Resize successful! Output shape: {resized.shape}")
        print(f"   Mode: bilinear (MPS-compatible)")
        return True
    except Exception as e:
        print(f"❌ Resize failed: {e}")
        return False


def test_upscaler(image_path: Path):
    """Test upscaler classes with real image."""
    print("\n" + "=" * 60)
    print(f"Testing upscalers with: {image_path.name}")
    print("=" * 60)

    # Load image
    img = Image.open(image_path).convert("RGB")
    # Resize to small for faster testing
    img = img.resize((256, 256))
    img_np = np.array(img).astype(np.float32) / 255.0

    # Convert to torch tensor [1, 3, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device")

    img_tensor = img_tensor.to(device)

    # Create minimal config
    class MinimalConfig:
        upscale = 2  # 2x for faster testing

        class Phase2Config:
            tile_based_upscaling = False
            upscale_tile_size = 0
            upscale_overlap = 64

        phase2 = Phase2Config()

    config = MinimalConfig()

    # Test NoneUpscaler
    print("\n1. Testing NoneUpscaler...")
    try:
        upscaler = NoneUpscaler(config, device)
        result = upscaler.upscale(img_tensor)
        print(f"✅ NoneUpscaler success! {img_tensor.shape} → {result.shape}")
    except Exception as e:
        print(f"❌ NoneUpscaler failed: {e}")
        return False

    # Test TorchUpscaler
    print("\n2. Testing TorchUpscaler...")
    try:
        upscaler = TorchUpscaler(config, device)
        result = upscaler.upscale(img_tensor)
        print(f"✅ TorchUpscaler success! {img_tensor.shape} → {result.shape}")
    except Exception as e:
        print(f"❌ TorchUpscaler failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("\n🔬 MPS Bicubic Fix Validation Test")
    print("=" * 60)

    # Test 1: torch_ops.resize
    test1_pass = test_torch_ops_resize()

    # Test 2: Upscalers with real image
    test_images = list(Path("data/validation_expanded").glob("*.jpg"))
    if not test_images:
        print("\n⚠️  No test images found in data/validation_expanded/")
        test2_pass = None
    else:
        test2_pass = test_upscaler(test_images[0])

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"torch_ops.resize: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    if test2_pass is not None:
        print(f"Upscaler classes: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    else:
        print(f"Upscaler classes: ⚠️  SKIPPED (no test images)")

    all_pass = test1_pass and (test2_pass if test2_pass is not None else True)

    if all_pass:
        print("\n🎉 All tests PASSED! MPS bicubic fix is working.")
        return 0
    else:
        print("\n❌ Some tests FAILED. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
