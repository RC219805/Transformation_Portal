"""Unit tests for torch_ops module."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

from lux_depth_v2 import torch_ops


class TestDeviceSelection:
    """Test device selection utilities."""

    def test_require_torch(self):
        """Test torch requirement check."""
        torch_ops.require_torch()  # Should not raise

    def test_pick_device_auto(self):
        """Test automatic device selection."""
        device = torch_ops.pick_device("auto")
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "cpu", "mps")

    def test_pick_device_cpu(self):
        """Test CPU device selection."""
        device = torch_ops.pick_device("cpu")
        assert device.type == "cpu"

    def test_pick_device_cuda(self):
        """Test CUDA device selection."""
        device = torch_ops.pick_device("cuda")
        assert device.type == "cuda"

    def test_configure_torch(self):
        """Test torch configuration."""
        torch_ops.configure_torch(cudnn_benchmark=True)
        # Should not raise


class TestTensorConversion:
    """Test tensor conversion utilities."""

    def test_to_torch_rgb(self, torch_device, sample_rgb_array):
        """Test numpy to torch RGB conversion."""
        rgb_t = torch_ops.to_torch_rgb(sample_rgb_array, torch_device)
        assert rgb_t.shape == (1, 3, 64, 64)
        assert rgb_t.dtype == torch.float32
        assert rgb_t.device.type == torch_device.type

    def test_to_torch_rgb_invalid_shape(self, torch_device):
        """Test conversion with invalid shape raises error."""
        invalid = np.zeros((64, 64), dtype=np.float32)  # Missing channel dim
        with pytest.raises(ValueError, match="Expected HxWx3"):
            torch_ops.to_torch_rgb(invalid, torch_device)

    def test_from_torch_rgb(self, torch_device, sample_rgb_array):
        """Test torch to numpy RGB conversion."""
        rgb_t = torch_ops.to_torch_rgb(sample_rgb_array, torch_device)
        rgb_np = torch_ops.from_torch_rgb(rgb_t)
        assert rgb_np.shape == (64, 64, 3)
        assert rgb_np.dtype == np.float32
        assert np.allclose(rgb_np, sample_rgb_array, atol=1e-6)

    def test_from_torch_rgb_clamps(self, torch_device):
        """Test from_torch_rgb clamps values to [0, 1]."""
        rgb_t = torch.tensor([[[[2.0], [-1.0], [0.5]]]], device=torch_device)
        rgb_np = torch_ops.from_torch_rgb(rgb_t)
        assert np.all(rgb_np >= 0.0)
        assert np.all(rgb_np <= 1.0)


class TestImageOperations:
    """Test image processing operations."""

    def test_luma_calculation(self, torch_device, sample_rgb_array):
        """Test luma calculation."""
        rgb_t = torch_ops.to_torch_rgb(sample_rgb_array, torch_device)
        luma_t = torch_ops.luma(rgb_t)
        assert luma_t.shape == (1, 1, 64, 64)
        assert luma_t.dtype == torch.float32

    def test_luma_weights(self, torch_device):
        """Test luma uses correct RGB weights."""
        # Pure red, green, blue
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        rgb[0:3, :, 0] = 1.0  # Red
        rgb[3:6, :, 1] = 1.0  # Green
        rgb[6:9, :, 2] = 1.0  # Blue

        rgb_t = torch_ops.to_torch_rgb(rgb, torch_device)
        luma_t = torch_ops.luma(rgb_t)
        luma_np = luma_t[0, 0].cpu().numpy()

        # Check approximate luma values (Rec. 709 weights)
        assert np.allclose(luma_np[1, 0], 0.2126, atol=0.01)  # Red
        assert np.allclose(luma_np[4, 0], 0.7152, atol=0.01)  # Green
        assert np.allclose(luma_np[7, 0], 0.0722, atol=0.01)  # Blue

    def test_smoothstep(self, torch_device):
        """Test smoothstep interpolation."""
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=torch_device)
        result = torch_ops.smoothstep(0.0, 1.0, x)
        assert result.shape == x.shape
        assert result[0].item() == pytest.approx(0.0)
        assert result[-1].item() == pytest.approx(1.0)
        assert 0 < result[2].item() < 1.0

    def test_midtone_map(self, torch_device):
        """Test midtone mapping function."""
        luma = torch.linspace(0, 1, 100, device=torch_device).view(1, 1, 10, 10)
        mid_map = torch_ops.midtone_map(luma)
        assert mid_map.shape == luma.shape
        # Midtones should have higher values than shadows/highlights
        mid_idx = 50  # Middle of range
        shadow_idx = 10
        highlight_idx = 90
        mid_val = mid_map.view(-1)[mid_idx].item()
        shadow_val = mid_map.view(-1)[shadow_idx].item()
        highlight_val = mid_map.view(-1)[highlight_idx].item()
        assert mid_val > shadow_val
        assert mid_val > highlight_val


class TestBlurOperations:
    """Test Gaussian blur operations."""

    def test_gaussian_blur_basic(self, torch_device):
        """Test basic Gaussian blur."""
        img = torch.ones((1, 3, 32, 32), device=torch_device, dtype=torch.float32)
        img[:, :, 15:17, 15:17] = 0.0  # Add a dark square

        blurred = torch_ops.gaussian_blur(img, sigma=2.0)
        assert blurred.shape == img.shape
        assert blurred.dtype == torch.float32

    def test_gaussian_blur_zero_sigma(self, torch_device):
        """Test blur with zero sigma returns original."""
        img = torch.rand((1, 3, 32, 32), device=torch_device)
        blurred = torch_ops.gaussian_blur(img, sigma=0.0)
        assert torch.allclose(blurred, img)

    def test_gaussian_blur_channels(self, torch_device):
        """Test blur works with multiple channels."""
        for n_channels in [1, 3, 4]:
            img = torch.rand((1, n_channels, 32, 32), device=torch_device)
            blurred = torch_ops.gaussian_blur(img, sigma=1.5)
            assert blurred.shape == img.shape


class TestResizeOperations:
    """Test resize operations."""

    def test_resize_upscale(self, torch_device):
        """Test image upscaling."""
        img = torch.rand((1, 3, 32, 32), device=torch_device)
        resized = torch_ops.resize(img, (64, 64), mode="bicubic")
        assert resized.shape == (1, 3, 64, 64)

    def test_resize_downscale(self, torch_device):
        """Test image downscaling."""
        img = torch.rand((1, 3, 64, 64), device=torch_device)
        resized = torch_ops.resize(img, (32, 32), mode="bilinear")
        assert resized.shape == (1, 3, 32, 32)

    def test_resize_modes(self, torch_device):
        """Test different resize modes."""
        img = torch.rand((1, 3, 32, 32), device=torch_device)
        for mode in ["bilinear", "bicubic"]:
            resized = torch_ops.resize(img, (48, 48), mode=mode)
            assert resized.shape == (1, 3, 48, 48)


class TestColorOperations:
    """Test color adjustment operations."""

    def test_soft_clip01(self, torch_device):
        """Test soft clipping to [0, 1]."""
        img = torch.tensor([[[[2.0, -1.0, 0.5, 0.95]]]], device=torch_device)
        clipped = torch_ops.soft_clip01(img, knee=0.9)
        assert torch.all(clipped >= 0.0)
        assert torch.all(clipped <= 1.0)

    def test_apply_temperature(self, torch_device):
        """Test temperature adjustment."""
        rgb = torch.rand((1, 3, 32, 32), device=torch_device)
        temp = torch.full((1, 1, 32, 32), 0.05, device=torch_device)
        result = torch_ops.apply_temperature(rgb, temp)
        assert result.shape == rgb.shape
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_apply_saturation(self, torch_device):
        """Test saturation adjustment."""
        rgb = torch.rand((1, 3, 32, 32), device=torch_device)
        sat = torch.full((1, 1, 32, 32), 1.2, device=torch_device)
        result = torch_ops.apply_saturation(rgb, sat)
        assert result.shape == rgb.shape
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_apply_exp_con(self, torch_device):
        """Test exposure and contrast adjustment."""
        rgb = torch.rand((1, 3, 32, 32), device=torch_device)
        exp = torch.full((1, 1, 32, 32), 1.05, device=torch_device)
        con = torch.full((1, 1, 32, 32), 1.1, device=torch_device)
        result = torch_ops.apply_exp_con(rgb, exp, con)
        assert result.shape == rgb.shape
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)


class TestEdgeDetection:
    """Test edge detection operations."""

    def test_edge_map(self, torch_device):
        """Test edge map computation."""
        # Create image with clear edge
        luma = torch.zeros((1, 1, 64, 64), device=torch_device)
        luma[:, :, :32, :] = 1.0  # Half white, half black

        edges = torch_ops.edge_map(luma)
        assert edges.shape == luma.shape
        assert torch.all(edges >= 0.0)
        assert torch.all(edges <= 1.0)
        # Edge should be detected near row 32
        assert edges[:, :, 31:33, :].mean() > edges.mean()


class TestParameterMapping:
    """Test parameter mapping utilities."""

    def test_param_map(self, torch_device):
        """Test parameter map creation."""
        wfg = torch.ones((1, 1, 32, 32), device=torch_device) * 0.3
        wmid = torch.ones((1, 1, 32, 32), device=torch_device) * 0.4
        wbg = torch.ones((1, 1, 32, 32), device=torch_device) * 0.3

        result = torch_ops.param_map(wfg, wmid, wbg, 1.0, 0.8, 0.6)
        assert result.shape == (1, 1, 32, 32)
        expected = 0.3 * 1.0 + 0.4 * 0.8 + 0.3 * 0.6
        assert result.mean().item() == pytest.approx(expected, rel=0.01)


class TestTiler:
    """Test tiling utility for large images."""

    def test_tiler_no_tiling(self, torch_device):
        """Test tiler with tile=0 (no tiling)."""
        tiler = torch_ops.Tiler(tile=0, overlap=0)
        img = torch.rand((1, 3, 64, 64), device=torch_device)

        def identity_fn(tile, ya0, xa0, ya1, xa1, y0, x0, y1, x1):
            return tile

        result = tiler.run(img, identity_fn)
        assert torch.allclose(result, img)

    def test_tiler_with_tiles(self, torch_device):
        """Test tiler with actual tiling."""
        tiler = torch_ops.Tiler(tile=32, overlap=8)
        img = torch.rand((1, 3, 64, 64), device=torch_device)

        def add_one_fn(tile, ya0, xa0, ya1, xa1, y0, x0, y1, x1):
            return tile + 1.0

        result = tiler.run(img, add_one_fn)
        assert result.shape == img.shape
        # All pixels should be incremented by 1
        assert torch.allclose(result, img + 1.0, atol=1e-5)


class TestValidationMetrics:
    """Test validation metric computations."""

    def test_mean_abs_rgb(self, torch_device):
        """Test RGB difference metric."""
        a = torch.rand((1, 3, 64, 64), device=torch_device)
        b = a + 0.1

        diff = torch_ops.mean_abs_rgb(a, b)
        assert isinstance(diff, float)
        assert diff == pytest.approx(0.1, abs=0.01)

    def test_mean_abs_luma(self, torch_device):
        """Test luma difference metric."""
        a = torch.rand((1, 3, 64, 64), device=torch_device)
        b = a * 1.2

        diff = torch_ops.mean_abs_luma(a, b)
        assert isinstance(diff, float)
        assert diff > 0


class TestMaybeAutocast:
    """Test autocast context manager."""

    def test_autocast_enabled_cuda(self, torch_device):
        """Test autocast with CUDA device."""
        if torch_device.type != "cuda":
            pytest.skip("CUDA not available")

        with torch_ops.maybe_autocast(True, torch_device):
            x = torch.rand((1, 3, 32, 32), device=torch_device)
            # Operations inside should work

    def test_autocast_disabled(self, torch_device):
        """Test autocast disabled."""
        with torch_ops.maybe_autocast(False, torch_device):
            x = torch.rand((1, 3, 32, 32), device=torch_device)
            # Should work without autocast
