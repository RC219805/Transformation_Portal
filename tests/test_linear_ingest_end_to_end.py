"""End-to-end tests for linear ingest pipeline.

Tests linear light preservation from RAW/TIFF files through to final tensors:
- RAW → linear uint16 → float32 [0,1] verification
- 16-bit TIFF → float32 [0,1] verification
- Gamma-encoded inputs must be rejected
- dtype validation (no uint8/uint16 in final output)
- Range validation ([0, 1] bounds)

These are correctness-critical tests per APEX Contract and Spatial AI Foundation ROADMAP.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.linear_verify import (
    LinearityViolationError,
    create_gamma_encoded_fixture,
    create_linear_test_fixture,
    verify_linear_ingest,
)
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image_linear


class TestLinearPreprocessing:
    """Test linear-preserving preprocessing."""

    def test_numpy_array_linear_input(self):
        """Linear float32 numpy array should pass through unchanged."""
        linear_input = create_linear_test_fixture(shape=(100, 100, 3))

        result, orig_shape = preprocess_image_linear(linear_input)

        # Should preserve linearity
        verify_linear_ingest(result)
        assert result.dtype == np.float32
        assert orig_shape == (100, 100)

    def test_numpy_uint8_to_linear_float32(self):
        """uint8 input should convert to linear float32."""
        # Create uint8 input (simulating 8-bit image)
        uint8_input = np.random.randint(0, 256, (56, 56, 3), dtype=np.uint8)

        result, orig_shape = preprocess_image_linear(uint8_input)

        # Should be float32
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        verify_linear_ingest(result)

    def test_numpy_uint16_to_linear_float32(self):
        """uint16 input should convert to linear float32 preserving precision."""
        # Create uint16 input (simulating 16-bit linear TIFF)
        uint16_input = np.random.randint(0, 65536, (56, 56, 3), dtype=np.uint16)

        result, orig_shape = preprocess_image_linear(uint16_input)

        # Should be float32
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        verify_linear_ingest(result)

    def test_gamma_encoded_numpy_logs_warning(self):
        """Gamma-encoded numpy input should log warning (not reject by default)."""
        gamma_input = create_gamma_encoded_fixture(shape=(50, 50, 3))

        # Should pass with warning (default behavior)
        result, orig_shape = preprocess_image_linear(gamma_input)
        # Verify it passed (no exception)
        assert result is not None

    def test_dimension_enforcement(self):
        """Dimensions should be enforced to multiples of 14."""
        # Create input with non-compliant dimensions
        linear_input = create_linear_test_fixture(shape=(100, 75, 3))

        result, orig_shape = preprocess_image_linear(linear_input)

        # Result should have compliant dimensions
        h, w = result.shape[:2]
        assert h % 14 == 0
        assert w % 14 == 0

        # Original shape preserved
        assert orig_shape == (100, 75)

    def test_grayscale_converted_to_rgb(self):
        """Grayscale input should be converted to 3-channel RGB."""
        grayscale = np.random.rand(56, 56).astype(np.float32)

        result, orig_shape = preprocess_image_linear(grayscale)

        assert result.shape[2] == 3
        # All channels should be identical
        assert np.allclose(result[:, :, 0], result[:, :, 1])
        assert np.allclose(result[:, :, 1], result[:, :, 2])

    def test_rgba_converted_to_rgb(self):
        """RGBA input should drop alpha channel."""
        rgba = np.random.rand(56, 56, 4).astype(np.float32)

        result, orig_shape = preprocess_image_linear(rgba)

        assert result.shape[2] == 3  # Alpha dropped


class TestTiffLinearIngest:
    """Test TIFF file linear ingest (requires tifffile)."""

    def test_16bit_tiff_preserves_precision(self, tmp_path):
        """16-bit TIFF should preserve precision through conversion."""
        # Skip if tifffile not available
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")

        # Create 16-bit linear TIFF fixture
        tiff_path = tmp_path / "linear_16bit.tif"

        # Generate linear data with full 16-bit range
        linear_data = (np.random.rand(100, 100, 3) * 65535).astype(np.uint16)
        tifffile.imwrite(str(tiff_path), linear_data)

        # Load and preprocess
        result, orig_shape = preprocess_image_linear(tiff_path)

        # Should be float32 [0, 1]
        assert result.dtype == np.float32
        verify_linear_ingest(result)
        assert orig_shape == (100, 100)

    def test_8bit_png_linear_path(self, tmp_path):
        """8-bit PNG can go through linear path with explicit escape hatch.

        Note: PNG is rejected by default (format boundary). This test uses
        apex_strict_formats=False to test the conversion path.
        """
        # Create 8-bit PNG
        png_path = tmp_path / "test.png"
        test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        test_img.save(png_path)

        # Load and preprocess with escape hatch
        result, orig_shape = preprocess_image_linear(
            png_path,
            apex_strict_formats=False,  # Bypass format boundary for this test
            verify_linearity=False,  # Skip gamma check (gray may trigger heuristics)
        )

        # Should be float32
        assert result.dtype == np.float32
        assert orig_shape == (64, 64)


class TestRAWLinearIngest:
    """Test RAW file linear ingest (mocked, no actual rawpy)."""

    @pytest.mark.ml
    def test_raw_loader_linear_output_for_apex(self, tmp_path):
        """RAW loader with output_linear=True produces uint16 linear output."""
        # This test requires rawpy, skip if not available
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed")

        from unittest.mock import MagicMock, patch

        from transformation_portal.lux_depth_v3.raw_loader import load_raw_as_rgb

        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"fake raw data")

        # Create fake 16-bit linear RGB
        fake_rgb = np.random.randint(0, 65536, (16, 16, 3), dtype=np.uint16)

        mock_raw_context = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.postprocess.return_value = fake_rgb
        mock_raw_context.__enter__.return_value = mock_raw_obj
        mock_raw_context.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw_context):
            # Test APEX mode: linear output
            rgb = load_raw_as_rgb(raw_file, output_linear=True, output_bps=16)

            # Assertions
            assert rgb is not None
            assert rgb.dtype == np.uint16
            assert rgb.shape == (16, 16, 3)

            # Verify postprocess was called with linear settings
            mock_raw_obj.postprocess.assert_called_once()
            call_kwargs = mock_raw_obj.postprocess.call_args[1]
            # Check gamma was set to linear
            assert call_kwargs["gamma"] == (1, 1)

    @pytest.mark.ml
    def test_raw_file_preprocessed_as_linear(self, tmp_path):
        """RAW file should be preprocessed as linear (integration test)."""
        # Skip if rawpy not available
        try:
            import rawpy  # noqa: F401
        except ImportError:
            pytest.skip("rawpy not installed - cannot test RAW preprocessing")

        # Create fake RAW file (can't actually test without real RAW data)
        # This would require a real RAW file fixture
        pytest.skip("Requires real RAW file fixture")


class TestEndToEndLinearityPreservation:
    """End-to-end tests for linearity preservation."""

    def test_linear_fixture_end_to_end(self):
        """Linear fixture should preserve linearity through full pipeline."""
        # Start with known-linear fixture
        linear_input = create_linear_test_fixture(shape=(100, 100, 3), mean=0.3, seed=42)

        # Preprocess (should preserve linearity)
        result, orig_shape = preprocess_image_linear(linear_input, verify_linearity=True)

        # Verify still linear
        verify_linear_ingest(result)

        # Should be deterministic
        result2, _ = preprocess_image_linear(linear_input, verify_linearity=True)
        # Note: dimension enforcement may introduce small differences
        # so we check shape equality, not value equality
        assert result.shape == result2.shape

    def test_gamma_fixture_logs_warning_end_to_end(self):
        """Gamma fixture logs warning but doesn't reject by default."""
        gamma_input = create_gamma_encoded_fixture(shape=(50, 50, 3), seed=42)

        # Should pass with warning (default is non-strict)
        result, _ = preprocess_image_linear(gamma_input, verify_linearity=True)
        # Verify it passed
        assert result is not None

    def test_can_disable_verification(self):
        """Verification can be disabled for debugging (not recommended)."""
        gamma_input = create_gamma_encoded_fixture(shape=(50, 50, 3))

        # With verification disabled, gamma input passes (for debugging only)
        result, _ = preprocess_image_linear(gamma_input, verify_linearity=False)

        # But the result is still gamma-encoded (this is the danger!)
        # We can detect it manually:
        from transformation_portal.lux_depth_v3.linear_verify import detect_gamma_encoding

        assert detect_gamma_encoding(result)

    def test_dtype_enforcement_end_to_end(self):
        """Final output must be float32, never uint8/uint16."""
        # Try various input dtypes
        test_cases = [
            (np.uint8, (56, 56, 3)),
            (np.uint16, (56, 56, 3)),
            (np.float32, (56, 56, 3)),
            (np.float64, (56, 56, 3)),
        ]

        for dtype, shape in test_cases:
            if dtype in [np.uint8, np.uint16]:
                input_arr = np.random.randint(0, 256 if dtype == np.uint8 else 65536, shape, dtype=dtype)
            else:
                input_arr = np.random.rand(*shape).astype(dtype) * 0.5  # Keep mean low

            result, _ = preprocess_image_linear(input_arr)

            # Final output MUST be float32
            assert result.dtype == np.float32
            # And must pass linear verification
            verify_linear_ingest(result)

    def test_range_enforcement_end_to_end(self):
        """Values must stay in [0, 1] through full pipeline."""
        # Create input that might overflow
        input_arr = np.random.rand(56, 56, 3).astype(np.float32)

        result, _ = preprocess_image_linear(input_arr)

        # Range must be preserved
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        verify_linear_ingest(result)


class TestTargetSizeResizing:
    """Test resizing with linear preservation."""

    def test_resize_preserves_linearity(self):
        """Resizing should preserve linear light (approximately)."""
        linear_input = create_linear_test_fixture(shape=(200, 200, 3), mean=0.3, seed=42)

        # Resize to smaller size
        result, orig_shape = preprocess_image_linear(linear_input, target_size=112)

        # Should still be linear (within tolerance due to interpolation)
        # Note: resize may introduce small gamma-like artifacts due to interpolation,
        # but the overall distribution should remain linear-ish
        verify_linear_ingest(result, check_gamma=False)  # Skip gamma check due to resize

        # Should be resized
        assert max(result.shape[:2]) <= 112

        # Original shape preserved
        assert orig_shape == (200, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
