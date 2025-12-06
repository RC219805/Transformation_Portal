"""Pytest configuration and shared fixtures for lux_depth_v2 tests."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    tifffile = None


@pytest.fixture
def torch_device():
    """Get available torch device."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_rgb_array():
    """Create a sample RGB float32 array (HxWx3)."""
    # 64x64 test image with gradient
    h, w = 64, 64
    arr = np.zeros((h, w, 3), dtype=np.float32)
    arr[:, :, 0] = np.linspace(0, 1, h)[:, None]  # Red gradient vertical
    arr[:, :, 1] = np.linspace(0, 1, w)[None, :]  # Green gradient horizontal
    arr[:, :, 2] = 0.5  # Constant blue
    return arr


@pytest.fixture
def sample_depth_array():
    """Create a sample depth array (HxW)."""
    h, w = 64, 64
    y, x = np.mgrid[0:h, 0:w]
    # Radial depth gradient (darker center = closer)
    cy, cx = h // 2, w // 2
    depth = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    depth = depth / depth.max()
    return depth.astype(np.float32)


@pytest.fixture
def sample_mask_array():
    """Create a sample binary mask (HxW)."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=np.float32)
    mask[16:48, 16:48] = 1.0  # Square mask in center
    return mask


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp(prefix="lux_depth_v2_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_image_file(temp_dir, sample_rgb_array):
    """Create a sample image file (PNG)."""
    if not CV2_AVAILABLE:
        pytest.skip("OpenCV not available")
    img_path = temp_dir / "test_image.png"
    rgb8 = (sample_rgb_array * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_path), bgr)
    return img_path


@pytest.fixture
def sample_tiff_file(temp_dir, sample_rgb_array):
    """Create a sample 16-bit TIFF file."""
    if not TIFFFILE_AVAILABLE:
        pytest.skip("tifffile not available")
    tiff_path = temp_dir / "test_image.tif"
    rgb16 = (sample_rgb_array * 65535).astype(np.uint16)
    tifffile.imwrite(str(tiff_path), rgb16, photometric="rgb")
    return tiff_path


@pytest.fixture
def sample_depth_file(temp_dir, sample_depth_array):
    """Create a sample 16-bit depth TIFF."""
    if not TIFFFILE_AVAILABLE:
        pytest.skip("tifffile not available")
    depth_path = temp_dir / "test_depth.tif"
    depth16 = (sample_depth_array * 65535).astype(np.uint16)
    tifffile.imwrite(str(depth_path), depth16)
    return depth_path


@pytest.fixture
def mock_config():
    """Create a mock PipelineConfig for testing."""
    from lux_depth_v2.config import PipelineConfig, Preset
    return PipelineConfig(
        preset=Preset.PHOTO_REALISTIC,
        upscale=4,
        device="cpu",
        precision="fp32",
        save_master=True,
        save_upscaled=True,
        save_marketing_png=True,
        save_preview_jpg=False,
        enable_material=False,  # Disable for faster tests
        upscaler_backend="none",  # Use bicubic for tests
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark test as integration test")
