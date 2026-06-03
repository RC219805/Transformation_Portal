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

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.raw_loader import (
    RAW_EXTENSIONS,
    is_raw_file,
    is_valid_demosaic_name,
    load_raw_as_pil,
    load_raw_as_rgb,
    resolve_demosaic_algorithm,
)


class TestRawExtensions:
    """Test RAW file extension detection."""

    def test_raw_extensions_exclude_tiff(self):
        """CRITICAL: TIFF must NOT be in RAW_EXTENSIONS."""
        assert ".tif" not in RAW_EXTENSIONS
        assert ".tiff" not in RAW_EXTENSIONS

    def test_raw_extensions_include_dng(self):
        """DNG (TIFF-based RAW) should be included."""
        assert ".dng" in RAW_EXTENSIONS

    def test_raw_extensions_include_cr3(self):
        """Canon CR3 (modern Canon bodies) should be included."""
        assert ".cr3" in RAW_EXTENSIONS

    def test_raw_extensions_match_canonical_source(self):
        """RAW_EXTENSIONS must be identical across every module that classifies
        RAW inputs. The canonical definition lives in
        ``transformation_portal.core.raw_formats``; rendering, sidecar, and
        input-manager paths must all re-export the same set.

        Regression guard: prior to convergence three different whitelists
        drifted (only one of them included CR3), so the same file could be
        accepted by sidecar generation and rejected by depth ingest.
        """
        from transformation_portal.core.raw_formats import RAW_EXTENSIONS as CANONICAL
        from transformation_portal.ingest.raw_sidecar import RAW_EXTENSIONS as SIDECAR_EXTS
        from transformation_portal.lux_depth_v3.input_manager import _RAW_EXTENSIONS as INPUT_MGR_EXTS

        assert RAW_EXTENSIONS == CANONICAL
        assert SIDECAR_EXTS == CANONICAL
        assert INPUT_MGR_EXTS == CANONICAL

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
            assert "rawpy required" in error_msg
            assert "./scripts/setup/install_raw_runtime.sh" in error_msg
            assert "pip install rawpy" not in error_msg

    def test_load_raw_as_rgb_uses_dedicated_runtime_when_configured(self, tmp_path, monkeypatch):
        """Configured RAW runtime should dispatch through the subprocess worker."""
        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        fake_rgb = np.full((8, 8, 3), 1024, dtype=np.uint16)
        captured: dict[str, object] = {}

        def fake_run_raw_worker(*, python_executable, command_name, input_path, payload, start):
            captured["python_executable"] = python_executable
            captured["command_name"] = command_name
            captured["input_path"] = input_path
            captured["payload"] = payload
            captured["start"] = start
            return fake_rgb, {"dtype": "uint16", "shape": [8, 8, 3]}

        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.raw_loader.run_raw_worker",
            fake_run_raw_worker,
        )

        rgb = load_raw_as_rgb(
            raw_file,
            use_camera_wb=False,
            half_size=True,
            output_bps=16,
            output_linear=True,
            python_executable="./.venv-raw/bin/python",
            demosaic="DCB",
        )

        assert np.array_equal(rgb, fake_rgb)
        assert captured["python_executable"] == "./.venv-raw/bin/python"
        assert captured["command_name"] == "load_rgb"
        assert captured["input_path"] == raw_file
        assert captured["payload"] == {
            "use_camera_wb": False,
            "half_size": True,
            "output_bps": 16,
            "output_linear": True,
            "demosaic": "DCB",
        }


class TestDemosaicAlgorithm:
    """Test demosaic algorithm parameterization."""

    def test_is_valid_demosaic_name_accepts_known_members(self):
        """The syntactic gate accepts every name documented as a rawpy member,
        including builds-specific ones like AFD/VCD/VCD_MODIFIED_AHD that
        used to be artificially blocked by the curated allowlist."""
        for name in (
            "AHD",
            "AAHD",
            "AMAZE",
            "DCB",
            "DHT",
            "LINEAR",
            "LMMSE",
            "MODIFIED_AHD",
            "PPG",
            "VNG",
            "AFD",
            "VCD",
            "VCD_MODIFIED_AHD",
        ):
            assert is_valid_demosaic_name(name), name

    def test_is_valid_demosaic_name_normalizes_case_and_whitespace(self):
        assert is_valid_demosaic_name("  amaze  ")
        assert is_valid_demosaic_name("Dcb")

    def test_is_valid_demosaic_name_rejects_garbage(self):
        for bad in ("", "  ", "amaze!", "amaze;rm -rf /", "amaze bar", "_AMAZE", "1AHD", None, 42):
            assert not is_valid_demosaic_name(bad), bad

    def test_resolve_demosaic_algorithm_unknown_raises(self):
        """Unknown demosaic names must fail closed with a clear ValueError that
        lists the actual installed members (not raw dir() noise)."""
        import sys
        import types

        # Use a real Enum so __members__ works as it would in production.
        from enum import IntEnum

        class _FakeDemosaic(IntEnum):
            AHD = 1
            AMAZE = 2

        fake_rawpy = types.SimpleNamespace(DemosaicAlgorithm=_FakeDemosaic)
        original = sys.modules.get("rawpy")
        sys.modules["rawpy"] = fake_rawpy
        try:
            with pytest.raises(ValueError, match="Unknown demosaic algorithm") as exc_info:
                resolve_demosaic_algorithm("DEFINITELY_NOT_REAL")
            msg = str(exc_info.value)
            # Real member names appear; the noise that bare dir() would
            # surface (dunder attrs, IntEnum protocol methods like 'value',
            # 'name', 'mro') must not.
            assert "'AHD'" in msg and "'AMAZE'" in msg
            for noise in ("__class__", "__module__", "mro", "'value'", "'name'"):
                assert noise not in msg, f"unexpected noise {noise!r} in error: {msg}"
        finally:
            if original is None:
                sys.modules.pop("rawpy", None)
            else:
                sys.modules["rawpy"] = original

    def test_load_raw_as_rgb_passes_demosaic_to_postprocess(self, tmp_path):
        """load_raw_as_rgb must forward the demosaic name into rawpy.postprocess."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        fake_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        sentinel = object()

        mock_raw_context = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.raw_image.shape = (4, 4)
        mock_raw_obj.camera_iso_speed = 100
        mock_raw_obj.postprocess.return_value = fake_rgb
        mock_raw_context.__enter__.return_value = mock_raw_obj
        mock_raw_context.__exit__.return_value = None

        with (
            patch("rawpy.imread", return_value=mock_raw_context),
            patch(
                "transformation_portal.lux_depth_v3.raw_loader.resolve_demosaic_algorithm",
                return_value=sentinel,
            ) as resolve_mock,
        ):
            load_raw_as_rgb(raw_file, demosaic="DCB")

        resolve_mock.assert_called_once_with("DCB")
        call_kwargs = mock_raw_obj.postprocess.call_args[1]
        assert call_kwargs["demosaic_algorithm"] is sentinel

    def test_load_raw_as_pil_missing_rawpy(self, tmp_path):
        """load_raw_as_pil should also fail gracefully when rawpy missing."""
        raw_file = tmp_path / "test.nef"
        raw_file.write_bytes(b"fake raw data")

        with patch.dict("sys.modules", {"rawpy": None}):
            with pytest.raises(ImportError) as exc_info:
                load_raw_as_pil(raw_file)

            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert "rawpy required" in error_msg
            assert "./scripts/setup/install_raw_runtime.sh" in error_msg
            assert "pip install rawpy" not in error_msg


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

        # Create fake 16-bit linear RGB array (small size for tests)
        fake_rgb = np.random.randint(0, 65536, (16, 16, 3), dtype=np.uint16)

        # Mock rawpy.imread to return a context manager
        mock_raw_context = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.raw_image.shape = (16, 16)
        mock_raw_obj.camera_iso_speed = 400
        mock_raw_obj.postprocess.return_value = fake_rgb

        # Setup __enter__ and __exit__ for context manager
        mock_raw_context.__enter__.return_value = mock_raw_obj
        mock_raw_context.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw_context):
            rgb = load_raw_as_rgb(raw_file, output_linear=True, output_bps=16)

            # Verify output shape and dtype (16-bit linear)
            assert rgb.shape == (16, 16, 3)
            assert rgb.dtype == np.uint16
            assert np.array_equal(rgb, fake_rgb)

            # Verify postprocess was called with linear settings
            mock_raw_obj.postprocess.assert_called_once()
            call_kwargs = mock_raw_obj.postprocess.call_args[1]
            assert call_kwargs["output_bps"] == 16
            assert call_kwargs["gamma"] == (1, 1)  # Linear gamma

    def test_raw_to_rgb_gamma_output_allowed(self, tmp_path):
        """Gamma-encoded output is allowed for legacy compatibility."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        # Create fake 8-bit gamma RGB array
        fake_rgb = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

        # Mock rawpy.imread
        mock_raw_context = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.postprocess.return_value = fake_rgb
        mock_raw_context.__enter__.return_value = mock_raw_obj
        mock_raw_context.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw_context):
            # Gamma output should work (legacy compatibility)
            rgb = load_raw_as_rgb(raw_file, output_linear=False, output_bps=8)
            assert rgb.dtype == np.uint8

    def test_raw_to_rgb_conversion_mocked(self, tmp_path):
        """Mock rawpy conversion to test interface (legacy test)."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        # Create fake RGB array (8-bit gamma for legacy default, small size for tests)
        fake_rgb = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

        # Mock rawpy.imread to return a context manager
        mock_raw_context = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.raw_image.shape = (16, 16)
        mock_raw_obj.camera_iso_speed = 400
        mock_raw_obj.postprocess.return_value = fake_rgb

        # Setup __enter__ and __exit__ for context manager
        mock_raw_context.__enter__.return_value = mock_raw_obj
        mock_raw_context.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw_context):
            rgb = load_raw_as_rgb(raw_file)

            # Verify output shape and dtype (legacy default is 8-bit gamma)
            assert rgb.shape == (16, 16, 3)
            assert rgb.dtype == np.uint8
            assert np.array_equal(rgb, fake_rgb)

    def test_load_raw_as_pil_returns_pil_image(self, tmp_path):
        """load_raw_as_pil should return PIL Image."""
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        raw_file = tmp_path / "test.nef"
        raw_file.write_bytes(b"fake raw data")

        # Create fake 8-bit RGB array (legacy default, small size)
        fake_rgb = np.random.randint(0, 256, (16, 24, 3), dtype=np.uint8)

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
            assert pil_img.size == (24, 16)  # PIL uses (W, H)

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
            import rawpy
        except ImportError:
            pytest.skip("rawpy not installed - integration tests skipped")
        # Verify the imported module is actually rawpy (defends against a
        # stub/shim being injected ahead of the real package).
        assert hasattr(rawpy, "imread"), "rawpy is importable but missing imread()"
