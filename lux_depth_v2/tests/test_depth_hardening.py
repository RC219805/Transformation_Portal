"""
Tests for hardened Option B depth loader (eliminates silent footguns).

Covers:
1. Format gating (reject .jpg, etc.)
2. Multi-channel safety (reject RGB colormaps)
3. uint8 depth handling (warn but accept)
4. Channel-first (C,H,W) layout support
5. Dtype validation (reject floats, negative ints)
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from lux_depth_v2 import io_utils
from lux_depth_v2.io_utils import DepthInfo

try:
    import cv2  # type: ignore
    import tifffile  # type: ignore
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="opencv-python or tifffile not installed")
class TestFormatGating:
    """Test that only .tif/.tiff/.png are accepted."""

    def test_reject_jpg_depth(self, tmp_path: Path):
        """Reject .jpg files (common mistake)."""
        jpg_path = tmp_path / "depth.jpg"
        img = np.zeros((16, 16), dtype=np.uint8)
        cv2.imwrite(str(jpg_path), img)

        with pytest.raises(ValueError, match="Unsupported depth file extension"):
            io_utils.read_depth_u16_with_info(jpg_path)

    def test_reject_webp_depth(self, tmp_path: Path):
        """Reject .webp files."""
        webp_path = tmp_path / "depth.webp"
        webp_path.write_bytes(b"fake webp")

        with pytest.raises(ValueError, match="Unsupported depth file extension"):
            io_utils.read_depth_u16_with_info(webp_path)

    def test_reject_bmp_depth(self, tmp_path: Path):
        """Reject .bmp files."""
        bmp_path = tmp_path / "depth.bmp"
        bmp_path.write_bytes(b"fake bmp")

        with pytest.raises(ValueError, match="Unsupported depth file extension"):
            io_utils.read_depth_u16_with_info(bmp_path)

    def test_accept_tiff_depth(self, tmp_path: Path):
        """Accept .tif and .tiff files."""
        for ext in [".tif", ".tiff"]:
            depth_path = tmp_path / f"depth{ext}"
            depth = np.random.randint(0, 65536, (50, 50), dtype=np.uint16)
            tifffile.imwrite(str(depth_path), depth)

            depth01, info = io_utils.read_depth_u16_with_info(depth_path)
            assert depth01.dtype == np.float32
            assert info.file_format in ("tif", "tiff")

    def test_accept_png_depth(self, tmp_path: Path):
        """Accept .png files."""
        depth_path = tmp_path / "depth.png"
        depth = np.random.randint(0, 65536, (50, 50), dtype=np.uint16)
        cv2.imwrite(str(depth_path), depth)

        depth01, info = io_utils.read_depth_u16_with_info(depth_path)
        assert depth01.dtype == np.float32
        assert info.file_format == "png"


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="opencv-python or tifffile not installed")
class TestMultiChannelSafety:
    """Test that RGB/RGBA files are only accepted if channels are identical."""

    def test_reject_rgb_with_different_channels(self, tmp_path: Path):
        """Reject RGB PNG where R != G != B (colormap or RGB image)."""
        r = np.random.randint(0, 65536, (50, 50), dtype=np.uint16)
        g = np.random.randint(0, 65536, (50, 50), dtype=np.uint16)
        b = np.random.randint(0, 65536, (50, 50), dtype=np.uint16)
        rgb = np.stack([r, g, b], axis=-1)

        png_path = tmp_path / "colormap.png"
        cv2.imwrite(str(png_path), rgb)

        with pytest.raises(ValueError, match="single-channel.*differing channels"):
            io_utils.read_depth_u16_with_info(png_path)

    def test_reject_rgba_with_different_channels(self, tmp_path: Path):
        """Reject RGBA PNG where R != G != B."""
        r = np.full((50, 50), 10000, dtype=np.uint16)
        g = np.full((50, 50), 20000, dtype=np.uint16)
        b = np.full((50, 50), 30000, dtype=np.uint16)
        a = np.full((50, 50), 65535, dtype=np.uint16)
        rgba = np.stack([r, g, b, a], axis=-1)

        png_path = tmp_path / "rgba_colormap.png"
        cv2.imwrite(str(png_path), rgba)

        with pytest.raises(ValueError, match="single-channel.*differing channels"):
            io_utils.read_depth_u16_with_info(png_path)

    def test_accept_grayscale_as_rgb(self, tmp_path: Path):
        """Accept RGB PNG where R == G == B (grayscale saved as RGB)."""
        gray = np.random.randint(10000, 50000, (50, 50), dtype=np.uint16)
        rgb = np.stack([gray, gray, gray], axis=-1)

        png_path = tmp_path / "gray_as_rgb.png"
        cv2.imwrite(str(png_path), rgb)

        depth01, info = io_utils.read_depth_u16_with_info(png_path)

        assert depth01.dtype == np.float32
        assert depth01.shape == (50, 50)
        assert info.channels == 3
        assert info.channel_collapsed is True

    def test_accept_grayscale_as_rgba(self, tmp_path: Path):
        """Accept RGBA PNG where R == G == B (alpha ignored)."""
        gray = np.random.randint(10000, 50000, (50, 50), dtype=np.uint16)
        alpha = np.full((50, 50), 65535, dtype=np.uint16)
        rgba = np.stack([gray, gray, gray, alpha], axis=-1)

        png_path = tmp_path / "gray_as_rgba.png"
        cv2.imwrite(str(png_path), rgba)

        depth01, info = io_utils.read_depth_u16_with_info(png_path)

        assert depth01.dtype == np.float32
        assert depth01.shape == (50, 50)
        assert info.channels == 4
        assert info.channel_collapsed is True


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="opencv-python or tifffile not installed")
class TestChannelFirstLayout:
    """Test channel-first (C,H,W) layout support (seen in some TIFFs)."""

    def test_channel_first_single_channel(self, tmp_path: Path):
        """Accept (1, H, W) layout."""
        depth = np.random.randint(0, 65536, (1, 50, 50), dtype=np.uint16)
        tiff_path = tmp_path / "depth_chw.tif"
        tifffile.imwrite(str(tiff_path), depth)

        depth01, info = io_utils.read_depth_u16_with_info(tiff_path)

        assert depth01.dtype == np.float32
        assert depth01.shape == (50, 50)
        assert info.channels == 1
        assert info.channel_collapsed is True

    def test_channel_first_rgb_identical(self, tmp_path: Path):
        """Accept (3, H, W) layout where all channels are identical."""
        gray = np.random.randint(10000, 50000, (50, 50), dtype=np.uint16)
        chw = np.stack([gray, gray, gray], axis=0)  # (3, 50, 50)

        tiff_path = tmp_path / "gray_as_rgb_chw.tif"
        tifffile.imwrite(str(tiff_path), chw)

        depth01, info = io_utils.read_depth_u16_with_info(tiff_path)

        assert depth01.dtype == np.float32
        assert depth01.shape == (50, 50)
        assert info.channels == 3
        assert info.channel_collapsed is True

    def test_channel_first_rgb_different_rejected(self, tmp_path: Path):
        """Reject (3, H, W) layout where channels differ."""
        r = np.full((50, 50), 10000, dtype=np.uint16)
        g = np.full((50, 50), 20000, dtype=np.uint16)
        b = np.full((50, 50), 30000, dtype=np.uint16)
        chw = np.stack([r, g, b], axis=0)  # (3, 50, 50)

        tiff_path = tmp_path / "rgb_chw.tif"
        tifffile.imwrite(str(tiff_path), chw)

        with pytest.raises(ValueError, match="single-channel.*differing channels"):
            io_utils.read_depth_u16_with_info(tiff_path)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="opencv-python or tifffile not installed")
class TestUint8Handling:
    """Test uint8 depth handling (warn but accept)."""

    def test_uint8_png_warns(self, tmp_path: Path):
        """uint8 PNG should trigger RuntimeWarning."""
        depth8 = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64)
        png_path = tmp_path / "depth8.png"
        cv2.imwrite(str(png_path), depth8)

        with pytest.warns(RuntimeWarning, match="8-bit.*Upscaling"):
            depth01, info = io_utils.read_depth_u16_with_info(png_path)

        assert depth01.dtype == np.float32
        assert info.source_dtype == "uint8"
        assert info.dtype == "uint16"  # Coerced

    def test_uint8_png_upscales_correctly(self, tmp_path: Path):
        """uint8 should be upscaled to uint16 via *257."""
        depth8 = np.array([[0, 128, 255]], dtype=np.uint8)
        png_path = tmp_path / "depth8_ramp.png"
        cv2.imwrite(str(png_path), depth8)

        with pytest.warns(RuntimeWarning):
            depth01, info = io_utils.read_depth_u16_with_info(png_path)

        # After *257: 0 -> 0, 128 -> 32896, 255 -> 65535
        # Then percentile normalization happens
        assert depth01.dtype == np.float32
        assert info.u16_max == 65535

    def test_uint16_png_no_warning(self, tmp_path: Path):
        """uint16 PNG should not trigger warning."""
        depth16 = np.random.randint(0, 65536, (50, 50), dtype=np.uint16)
        png_path = tmp_path / "depth16.png"
        cv2.imwrite(str(png_path), depth16)

        # Should NOT warn
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            try:
                depth01, info = io_utils.read_depth_u16_with_info(png_path)
                assert info.source_dtype == "uint16"
            except RuntimeWarning:
                pytest.fail("uint16 depth should not trigger warning")


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="opencv-python or tifffile not installed")
class TestDtypeValidation:
    """Test dtype validation (reject floats, negative ints)."""

    def test_reject_float32_depth(self, tmp_path: Path):
        """Reject float32 TIFF depth maps."""
        depth_f32 = np.random.rand(50, 50).astype(np.float32)
        tiff_path = tmp_path / "depth_float.tif"
        tifffile.imwrite(str(tiff_path), depth_f32)

        with pytest.raises(TypeError, match="must be uint16/uint8 integer.*floating point"):
            io_utils.read_depth_u16_with_info(tiff_path)

    def test_reject_float64_depth(self, tmp_path: Path):
        """Reject float64 TIFF depth maps."""
        depth_f64 = np.random.rand(50, 50).astype(np.float64)
        tiff_path = tmp_path / "depth_float64.tif"
        tifffile.imwrite(str(tiff_path), depth_f64)

        with pytest.raises(TypeError, match="must be uint16/uint8 integer.*floating point"):
            io_utils.read_depth_u16_with_info(tiff_path)

    def test_reject_negative_int16_depth(self, tmp_path: Path):
        """Reject int16 TIFF with negative values."""
        depth_i16 = np.random.randint(-1000, 1000, (50, 50), dtype=np.int16)
        tiff_path = tmp_path / "depth_int16.tif"
        tifffile.imwrite(str(tiff_path), depth_i16)

        with pytest.raises(ValueError, match="negative values"):
            io_utils.read_depth_u16_with_info(tiff_path)

    def test_accept_positive_int16_depth(self, tmp_path: Path):
        """Accept int16 TIFF if all values are non-negative."""
        depth_i16 = np.random.randint(0, 32767, (50, 50), dtype=np.int16)
        tiff_path = tmp_path / "depth_int16_pos.tif"
        tifffile.imwrite(str(tiff_path), depth_i16)

        depth01, info = io_utils.read_depth_u16_with_info(tiff_path)

        assert depth01.dtype == np.float32
        assert info.source_dtype == "int16"
        assert info.dtype == "uint16"  # Coerced

    def test_accept_uint32_depth(self, tmp_path: Path):
        """Accept uint32 TIFF (coerce to uint16) if values fit."""
        depth_u32 = np.random.randint(0, 65536, (50, 50), dtype=np.uint32)
        tiff_path = tmp_path / "depth_uint32.tif"
        tifffile.imwrite(str(tiff_path), depth_u32)

        depth01, info = io_utils.read_depth_u16_with_info(tiff_path)

        assert depth01.dtype == np.float32
        assert info.source_dtype == "uint32"
        assert info.dtype == "uint16"  # Coerced

    def test_reject_uint32_overflow(self, tmp_path: Path):
        """Reject uint32 TIFF with values > 65535 (overflow protection)."""
        depth_u32 = np.array([[100000, 200000]], dtype=np.uint32)
        tiff_path = tmp_path / "depth_uint32_overflow.tif"
        tifffile.imwrite(str(tiff_path), depth_u32)

        with pytest.raises(ValueError, match="exceed uint16 range.*max=200000"):
            io_utils.read_depth_u16_with_info(tiff_path)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="opencv-python or tifffile not installed")
class TestShapeValidation:
    """Test expected_hw shape validation."""

    def test_shape_match_passes(self, tmp_path: Path):
        """Correct shape should pass validation."""
        depth = np.random.randint(0, 65536, (100, 200), dtype=np.uint16)
        tiff_path = tmp_path / "depth.tif"
        tifffile.imwrite(str(tiff_path), depth)

        depth01, _ = io_utils.read_depth_u16_with_info(tiff_path, expected_hw=(100, 200))
        assert depth01.shape == (100, 200)

    def test_shape_mismatch_fails(self, tmp_path: Path):
        """Wrong shape should raise ValueError."""
        depth = np.random.randint(0, 65536, (100, 200), dtype=np.uint16)
        tiff_path = tmp_path / "depth.tif"
        tifffile.imwrite(str(tiff_path), depth)

        with pytest.raises(ValueError, match="shape mismatch"):
            io_utils.read_depth_u16_with_info(tiff_path, expected_hw=(200, 100))

    def test_shape_mismatch_clear_error(self, tmp_path: Path):
        """Shape mismatch error should be clear and actionable."""
        depth = np.random.randint(0, 65536, (480, 640), dtype=np.uint16)
        tiff_path = tmp_path / "depth.tif"
        tifffile.imwrite(str(tiff_path), depth)

        with pytest.raises(ValueError) as exc_info:
            io_utils.read_depth_u16_with_info(tiff_path, expected_hw=(1080, 1920))

        assert "got (480, 640)" in str(exc_info.value)
        assert "expected (1080, 1920)" in str(exc_info.value)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="opencv-python or tifffile not installed")
class TestDepthInfoProvenance:
    """Verify DepthInfo still provides complete provenance."""

    def test_depth_info_has_all_fields(self, tmp_path: Path):
        """DepthInfo should have all diagnostic fields."""
        depth = np.random.randint(5000, 60000, (100, 100), dtype=np.uint16)
        tiff_path = tmp_path / "depth.tif"
        tifffile.imwrite(str(tiff_path), depth)

        _, info = io_utils.read_depth_u16_with_info(tiff_path)

        # All provenance fields present
        assert hasattr(info, "file_format")
        assert hasattr(info, "source_dtype")
        assert hasattr(info, "dtype")
        assert hasattr(info, "shape")
        assert hasattr(info, "channels")
        assert hasattr(info, "channel_collapsed")
        assert hasattr(info, "u16_min")
        assert hasattr(info, "u16_max")
        assert hasattr(info, "p1")
        assert hasattr(info, "p99")

    def test_depth_info_serializable_to_dict(self, tmp_path: Path):
        """DepthInfo should be serializable for report JSON."""
        from dataclasses import asdict
        import json

        depth = np.random.randint(0, 65536, (50, 50), dtype=np.uint16)
        png_path = tmp_path / "depth.png"
        cv2.imwrite(str(png_path), depth)

        _, info = io_utils.read_depth_u16_with_info(png_path)
        info_dict = asdict(info)

        # Should be JSON-serializable
        json_str = json.dumps(info_dict)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["file_format"] == "png"
        assert parsed["source_dtype"] == "uint16"
