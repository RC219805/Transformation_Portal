"""Unit tests for io_utils module."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (CV2_AVAILABLE and TIFFFILE_AVAILABLE),
    reason="opencv-python and tifffile required"
)

from lux_depth_v2 import io_utils


class TestReadRgbAny:
    """Test RGB image reading."""

    def test_read_png(self, sample_image_file, sample_rgb_array):
        """Test reading PNG file."""
        rgb, info = io_utils.read_rgb_any(sample_image_file)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.float32
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)
        assert info.width == 64
        assert info.height == 64
        assert info.dtype == "uint8"
        assert info.bit_depth == 8

    def test_read_tiff_16bit(self, sample_tiff_file):
        """Test reading 16-bit TIFF file."""
        rgb, info = io_utils.read_rgb_any(sample_tiff_file)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.float32
        assert info.dtype == "uint16"
        assert info.bit_depth == 16

    def test_read_nonexistent_file(self, temp_dir):
        """Test reading nonexistent file raises error."""
        fake_path = temp_dir / "nonexistent.png"
        with pytest.raises(FileNotFoundError):
            io_utils.read_rgb_any(fake_path)

    def test_read_grayscale_converted(self, temp_dir):
        """Test grayscale image is converted to RGB."""
        gray_path = temp_dir / "gray.png"
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(gray_path), gray)
        
        rgb, info = io_utils.read_rgb_any(gray_path)
        assert rgb.shape == (64, 64, 3)
        # All channels should be identical
        assert np.allclose(rgb[:, :, 0], rgb[:, :, 1])
        assert np.allclose(rgb[:, :, 1], rgb[:, :, 2])


class TestReadDepthU16:
    """Test depth map reading."""

    def test_read_depth(self, sample_depth_file):
        """Test reading depth TIFF."""
        depth = io_utils.read_depth_u16(sample_depth_file)
        assert depth.shape == (64, 64)
        assert depth.dtype == np.float32
        assert np.all(depth >= 0.0)
        assert np.all(depth <= 1.0)

    def test_read_depth_nonexistent(self, temp_dir):
        """Test reading nonexistent depth file raises error."""
        fake_path = temp_dir / "nonexistent_depth.tif"
        with pytest.raises(FileNotFoundError):
            io_utils.read_depth_u16(fake_path)

    def test_read_depth_percentile_normalization(self, temp_dir):
        """Test depth is normalized using percentiles."""
        # Create depth with outliers
        depth = np.random.randint(1000, 60000, (64, 64), dtype=np.uint16)
        depth[0, 0] = 100  # Outlier low
        depth[0, 1] = 65000  # Outlier high
        
        depth_path = temp_dir / "depth_outliers.tif"
        tifffile.imwrite(str(depth_path), depth)
        
        depth_norm = io_utils.read_depth_u16(depth_path)
        # Should be normalized and outliers handled
        assert depth_norm.min() >= 0.0
        assert depth_norm.max() <= 1.0


class TestReadMaskAny:
    """Test mask reading."""

    def test_read_mask_png(self, temp_dir, sample_mask_array):
        """Test reading mask from PNG."""
        mask_path = temp_dir / "mask.png"
        mask8 = (sample_mask_array * 255).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask8)
        
        mask = io_utils.read_mask_any(mask_path)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

    def test_read_mask_tiff_16bit(self, temp_dir, sample_mask_array):
        """Test reading mask from 16-bit TIFF."""
        mask_path = temp_dir / "mask.tif"
        mask16 = (sample_mask_array * 65535).astype(np.uint16)
        tifffile.imwrite(str(mask_path), mask16)
        
        mask = io_utils.read_mask_any(mask_path)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_read_mask_multichannel(self, temp_dir):
        """Test reading multichannel image as mask (uses first channel)."""
        mask_path = temp_dir / "mask_rgb.png"
        rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(mask_path), rgb)
        
        mask = io_utils.read_mask_any(mask_path)
        assert mask.shape == (64, 64)  # Should extract single channel


class TestAtomicWriteRgb16Tiff:
    """Test atomic TIFF writing."""

    def test_write_rgb16_tiff(self, temp_dir, sample_rgb_array):
        """Test writing 16-bit RGB TIFF."""
        out_path = temp_dir / "output.tif"
        io_utils.atomic_write_rgb16_tiff(out_path, sample_rgb_array)
        
        assert out_path.exists()
        # Verify can read back
        rgb, info = io_utils.read_rgb_any(out_path)
        assert rgb.shape == sample_rgb_array.shape
        assert info.bit_depth == 16

    def test_write_creates_parent_dirs(self, temp_dir, sample_rgb_array):
        """Test writing creates parent directories."""
        out_path = temp_dir / "subdir" / "output.tif"
        io_utils.atomic_write_rgb16_tiff(out_path, sample_rgb_array)
        assert out_path.exists()

    def test_write_clamps_values(self, temp_dir):
        """Test writing clamps out-of-range values."""
        rgb = np.array([[[2.0, -0.5, 0.5]]], dtype=np.float32)
        out_path = temp_dir / "clamped.tif"
        io_utils.atomic_write_rgb16_tiff(out_path, rgb)
        
        rgb_read, _ = io_utils.read_rgb_any(out_path)
        assert np.all(rgb_read >= 0.0)
        assert np.all(rgb_read <= 1.0)

    def test_atomic_write_removes_tmp(self, temp_dir, sample_rgb_array):
        """Test atomic write removes temporary file."""
        out_path = temp_dir / "atomic.tif"
        tmp_path = out_path.with_suffix(".tif.tmp")
        
        io_utils.atomic_write_rgb16_tiff(out_path, sample_rgb_array)
        
        assert out_path.exists()
        assert not tmp_path.exists()


class TestAtomicWritePng8:
    """Test atomic PNG writing."""

    def test_write_png8(self, temp_dir, sample_rgb_array):
        """Test writing 8-bit PNG."""
        out_path = temp_dir / "output.png"
        io_utils.atomic_write_png8(out_path, sample_rgb_array)
        
        assert out_path.exists()
        rgb, info = io_utils.read_rgb_any(out_path)
        assert rgb.shape == sample_rgb_array.shape
        assert info.bit_depth == 8

    def test_write_png_compression(self, temp_dir, sample_rgb_array):
        """Test PNG is compressed."""
        out_path = temp_dir / "compressed.png"
        io_utils.atomic_write_png8(out_path, sample_rgb_array)
        
        # File should exist and be reasonable size
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestAtomicWriteJpg8:
    """Test atomic JPG writing."""

    def test_write_jpg8(self, temp_dir, sample_rgb_array):
        """Test writing 8-bit JPG."""
        out_path = temp_dir / "output.jpg"
        io_utils.atomic_write_jpg8(out_path, sample_rgb_array, quality=95)
        
        assert out_path.exists()
        rgb, info = io_utils.read_rgb_any(out_path)
        assert rgb.shape == sample_rgb_array.shape

    def test_write_jpg_quality(self, temp_dir, sample_rgb_array):
        """Test JPG quality parameter."""
        low_path = temp_dir / "low_quality.jpg"
        high_path = temp_dir / "high_quality.jpg"
        
        io_utils.atomic_write_jpg8(low_path, sample_rgb_array, quality=50)
        io_utils.atomic_write_jpg8(high_path, sample_rgb_array, quality=95)
        
        # Higher quality should produce larger file
        assert high_path.stat().st_size > low_path.stat().st_size


class TestImageInfo:
    """Test ImageInfo dataclass."""

    def test_image_info_creation(self, temp_dir):
        """Test ImageInfo is properly populated."""
        path = temp_dir / "test.png"
        info = io_utils.ImageInfo(
            path=path,
            width=1920,
            height=1080,
            dtype="uint8",
            bit_depth=8,
        )
        assert info.path == path
        assert info.width == 1920
        assert info.height == 1080
        assert info.dtype == "uint8"
        assert info.bit_depth == 8
