"""Tests for memory-efficient tiled upscaling."""

import pytest
import torch
import numpy as np


def test_tiled_upscaler_basic():
    """Test that tiled upscaling produces correct output size."""
    from lux_depth_v2.upscaling import TorchUpscaler
    from types import SimpleNamespace
    
    # Create config with tiling enabled
    cfg = SimpleNamespace(
        upscale=2,
        upscale_tile_size=256,
        upscale_tile_overlap=32
    )
    
    device = torch.device("cpu")
    upscaler = TorchUpscaler(cfg, device)
    
    # Create small test image
    input_tensor = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    
    # Upscale
    output = upscaler.upscale(input_tensor)
    
    # Verify output size
    assert output.shape == (1, 3, 1024, 1024)
    assert output.dtype == torch.float32
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_tiled_vs_full_consistency():
    """Test that tiled upscaling produces similar results to full upscaling."""
    from lux_depth_v2.upscaling import TorchUpscaler
    from types import SimpleNamespace
    
    device = torch.device("cpu")
    
    # Create test image with some structure
    input_tensor = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    
    # Full upscaling (no tiling)
    cfg_full = SimpleNamespace(
        upscale=2,
        upscale_tile_size=0,  # Disable tiling
        upscale_tile_overlap=32
    )
    upscaler_full = TorchUpscaler(cfg_full, device)
    output_full = upscaler_full.upscale(input_tensor)
    
    # Tiled upscaling
    cfg_tiled = SimpleNamespace(
        upscale=2,
        upscale_tile_size=256,
        upscale_tile_overlap=32
    )
    upscaler_tiled = TorchUpscaler(cfg_tiled, device)
    output_tiled = upscaler_tiled.upscale(input_tensor)
    
    # Outputs should be very similar (allowing for minor numerical differences from blending)
    diff = torch.abs(output_full - output_tiled)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    # Acceptable thresholds for tiled upscaling with overlap blending
    assert mean_diff < 0.02, f"Mean difference too large: {mean_diff}"
    assert max_diff < 0.10, f"Max difference too large: {max_diff}"


def test_tiled_upscaling_memory_efficiency():
    """Test that tiled upscaling uses less memory for large images."""
    pytest.skip("Memory profiling test - run manually with memory_profiler")
    
    from lux_depth_v2.upscaling import TorchUpscaler
    from types import SimpleNamespace
    import tracemalloc
    
    device = torch.device("cpu")
    
    # Large image (simulating 8K)
    input_tensor = torch.rand(1, 3, 4096, 4096, dtype=torch.float32)
    
    # Tiled upscaling with memory tracking
    tracemalloc.start()
    
    cfg_tiled = SimpleNamespace(
        upscale=2,
        upscale_tile_size=1024,
        upscale_tile_overlap=64
    )
    upscaler_tiled = TorchUpscaler(cfg_tiled, device)
    output_tiled = upscaler_tiled.upscale(input_tensor)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Peak memory should be significantly less than full-image approach
    # Full image would require ~384MB input + ~1.5GB output = ~1.9GB
    # Tiled should use <800MB peak
    peak_gb = peak / 1e9
    assert peak_gb < 1.0, f"Peak memory too high: {peak_gb:.2f}GB"


def test_blend_mask_creation():
    """Test positional blend mask for seamless tile merging."""
    from lux_depth_v2.upscaling import TorchUpscaler
    from types import SimpleNamespace
    
    cfg = SimpleNamespace(
        upscale=2,
        upscale_tile_size=256,
        upscale_tile_overlap=32
    )
    
    device = torch.device("cpu")
    upscaler = TorchUpscaler(cfg, device)
    
    # Test different tile positions
    shape = (1, 3, 256, 256)
    overlap = 32
    
    # First tile (top-left): no fading on top/left edges
    mask_tl = upscaler._create_positional_blend_mask(
        shape, overlap, fade_top=False, fade_bottom=True, fade_left=False, fade_right=True
    )
    assert mask_tl[:, :, 0, 0].item() == 1.0  # Top-left corner should be 1.0
    assert mask_tl[:, :, 0, -1].mean() < 1.0  # Right edge should be faded
    assert mask_tl[:, :, -1, 0].mean() < 1.0  # Bottom edge should be faded
    
    # Interior tile: all edges faded
    mask_interior = upscaler._create_positional_blend_mask(
        shape, overlap, fade_top=True, fade_bottom=True, fade_left=True, fade_right=True
    )
    assert mask_interior[:, :, 0, 0].item() < 0.1  # Top-left corner should be faded
    assert mask_interior[:, :, 128, 128].item() == 1.0  # Center should be 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
