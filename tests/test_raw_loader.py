"""Unit tests for RAW camera file loader.

Tests RAW file detection, error handling, and PIL/RAW boundary conditions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.raw_loader import RAW_EXTENSIONS, is_raw_file, load_raw_as_pil, load_raw_as_rgb


class TestRawExtensions:
    """Test RAW file extension detection."""

    def test_raw_extensions_exclude_tiff(self):
        """CRITICAL: TIFF must NOT be in RAW_EXTENSIONS."""
        assert ".tif" not in RAW_EXTENSIONS
        assert ".tiff" not in RAW_EXTENSIONS

    def test_raw_extensions_include_dng(self):
        """DNG (TIFF-based RAW) should be included."""
        assert ".dng" in RAW_EXTENSIONS

    def test_raw_extensions_are_lowercase(self):
        """All extensions should be lowercase for consistency."""
        for ext in RAW_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} should be lowercase"

    def test_raw_extensions_start_with_dot(self):
        """All extensions should start with dot."""
        for ext in RAW_EXTENSIONS:
            assert ext.startswith("."), f"Extension {ext} should start with '.'"


class TestIsRawFile:
    """Test is_raw_file() function."""

    def test_is_raw_file_canon_cr2(self):
        """Canon CR2 files are RAW."""
        assert is_raw_file(Path("test.cr2"))
        assert is_raw_file(Path("test.CR2"))  # Case-insensitive

    def test_is_raw_file_nikon_nef(self):
        """Nikon NEF files are RAW."""
        assert is_raw_file(Path("test.nef"))
        assert is_raw_file(Path("test.NEF"))

    def test_is_raw_file_sony_arw(self):
        """Sony ARW files are RAW."""
        assert is_raw_file(Path("test.arw"))
        assert is_raw_file(Path("test.ARW"))

    def test_is_raw_file_adobe_dng(self):
        """Adobe DNG files are RAW (TIFF-based)."""
        assert is_raw_file(Path("test.dng"))
        assert is_raw_file(Path("test.DNG"))

    def test_is_raw_file_tiff_not_raw(self):
        """CRITICAL: Standard TIFF files are NOT RAW."""
        assert not is_raw_file(Path("test.tif"))
        assert not is_raw_file(Path("test.tiff"))
        assert not is_raw_file(Path("test.TIF"))
        assert not is_raw_file(Path("test.TIFF"))

    def test_is_raw_file_standard_formats_not_raw(self):
        """Standard image formats are not RAW."""
        assert not is_raw_file(Path("test.jpg"))
        assert not is_raw_file(Path("test.jpeg"))
        assert not is_raw_file(Path("test.png"))
        assert not is_raw_file(Path("test.webp"))
        assert not is_raw_file(Path("test.bmp"))


class TestRawpyNotInstalled:
    """Test error handling when rawpy is not installed."""

    def test_load_raw_as_rgb_clear_error_message(self, tmp_path):
        """Clear error when rawpy missing."""
        # Create dummy RAW file
        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        # Mock the import to fail inside load_raw_as_rgb
        with patch.dict("sys.modules", {"rawpy": None}):
            with pytest.raises(ImportError) as exc_info:
                load_raw_as_rgb(raw_file)

            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert "rawpy required" in error_msg or "pip install rawpy" in error_msg

    def test_load_raw_as_pil_missing_rawpy(self, tmp_path):
        """load_raw_as_pil should also fail gracefully when rawpy missing."""
        raw_file = tmp_path / "test.nef"
        raw_file.write_bytes(b"fake raw data")

        with patch.dict("sys.modules", {"rawpy": None}):
            with pytest.raises(ImportError) as exc_info:
                load_raw_as_pil(raw_file)

            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert "rawpy required" in error_msg or "pip install rawpy" in error_msg


class TestTiffStillWorksWithoutRawpy:
    """CRITICAL: TIFF should work via PIL even without rawpy."""

    def test_tiff_works_without_rawpy(self, tmp_path):
        """TIFF should NOT require rawpy (routes through PIL)."""
        # Create test TIFF
        tiff_path = tmp_path / "test.tiff"
        test_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        test_img.save(tiff_path)

        # TIFF should NOT be detected as RAW
        assert not is_raw_file(tiff_path)

        # PIL should handle TIFF without rawpy
        # (This test validates the architectural decision)
        loaded_img = Image.open(tiff_path).convert("RGB")
        assert loaded_img.size == (64, 64)

    def test_tif_extension_not_raw(self, tmp_path):
        """Both .tif and .tiff should NOT be RAW."""
        tif_path = tmp_path / "test.tif"
        tiff_path = tmp_path / "test.tiff"

        # Create dummy files
        tif_path.write_bytes(b"dummy")
        tiff_path.write_bytes(b"dummy")

        # Neither should be detected as RAW
        assert not is_raw_file(tif_path)
        assert not is_raw_file(tiff_path)


@pytest.mark.ml
class TestRawToRgbConversion:
    """Test RAW to RGB conversion (requires rawpy)."""

    def test_rawpy_not_available_skip(self):
        """Skip if rawpy not available."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

    def test_raw_to_rgb_conversion_linear_output(self, tmp_path):
        """Test linear output (default, APEX compliant)."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        # Create fake 16-bit linear RGB array (new default)
        fake_rgb = np.random.randint(0, 65536, (4000, 6000, 3), dtype=np.uint16)

        # Mock rawpy.imread to return a context manager
        mock_raw_context = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.raw_image.shape = (4000, 6000)
        mock_raw_obj.camera_iso_speed = 400
        mock_raw_obj.postprocess.return_value = fake_rgb

        # Setup __enter__ and __exit__ for context manager
        mock_raw_context.__enter__.return_value = mock_raw_obj
        mock_raw_context.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw_context):
            rgb = load_raw_as_rgb(raw_file, output_linear=True, output_bps=16)

            # Verify output shape and dtype (16-bit linear)
            assert rgb.shape == (4000, 6000, 3)
            assert rgb.dtype == np.uint16
            assert np.array_equal(rgb, fake_rgb)
            
            # Verify postprocess was called with linear settings
            mock_raw_obj.postprocess.assert_called_once()
            call_kwargs = mock_raw_obj.postprocess.call_args[1]
            assert call_kwargs["output_bps"] == 16
            assert call_kwargs["gamma"] == (1, 1)  # Linear gamma

    def test_raw_to_rgb_gamma_output_blocked(self, tmp_path):
        """Gamma-encoded output should be blocked for APEX."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        # Attempting gamma output should raise error
        with pytest.raises(ValueError, match="Gamma-encoded.*not allowed.*APEX"):
            load_raw_as_rgb(raw_file, output_linear=False)

    def test_raw_to_rgb_conversion_mocked(self, tmp_path):
        """Mock rawpy conversion to test interface (legacy test)."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        # Create fake RGB array (16-bit for new default)
        fake_rgb = np.random.randint(0, 65536, (4000, 6000, 3), dtype=np.uint16)

        # Mock rawpy.imread to return a context manager
        mock_raw_context = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.raw_image.shape = (4000, 6000)
        mock_raw_obj.camera_iso_speed = 400
        mock_raw_obj.postprocess.return_value = fake_rgb

        # Setup __enter__ and __exit__ for context manager
        mock_raw_context.__enter__.return_value = mock_raw_obj
        mock_raw_context.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw_context):
            rgb = load_raw_as_rgb(raw_file)

            # Verify output shape and dtype (default is now 16-bit linear)
            assert rgb.shape == (4000, 6000, 3)
            assert rgb.dtype == np.uint16
            assert np.array_equal(rgb, fake_rgb)

    def test_load_raw_as_pil_returns_pil_image(self, tmp_path):
        """load_raw_as_pil should return PIL Image."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        raw_file = tmp_path / "test.nef"
        raw_file.write_bytes(b"fake raw data")

        # Create fake 16-bit RGB array (new default for linear)
        fake_rgb = np.random.randint(0, 65536, (100, 150, 3), dtype=np.uint16)

        # Mock rawpy.imread
        mock_raw_context = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.postprocess.return_value = fake_rgb

        mock_raw_context.__enter__.return_value = mock_raw_obj
        mock_raw_context.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw_context):
            pil_img = load_raw_as_pil(raw_file)

            # Verify output is PIL Image
            assert isinstance(pil_img, Image.Image)
            assert pil_img.mode == "RGB"
            assert pil_img.size == (150, 100)  # PIL uses (W, H)

    def test_file_not_found_error(self):
        """FileNotFoundError for missing RAW file."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        with pytest.raises(FileNotFoundError):
            load_raw_as_rgb(Path("/nonexistent/file.cr2"))


# Integration smoke tests (optional, only if rawpy available)


@pytest.mark.ml
@pytest.mark.slow
class TestRawLoaderIntegration:
    """Integration tests with actual rawpy (optional, slow)."""

    def test_rawpy_available(self):
        """Check if rawpy is available for integration tests."""
        try:
            import rawpy  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("rawpy not installed - integration tests skipped")
