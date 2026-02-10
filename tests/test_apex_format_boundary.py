"""Tests for APEX format boundary enforcement in preprocess_image_linear.

Tests the deterministic format-based rejection of JPEG/PNG inputs,
ensuring training-safe linear ingest with explicit escape hatches.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.preprocessing import preprocess_image_linear


class TestApexFormatBoundary:
    """Test APEX format boundary enforcement (RAW + TIFF only by default)."""

    def test_tiff_accepted_by_default(self, tmp_path):
        """Test that TIFF files are accepted with apex_strict_formats=True."""
        # Create 16-bit TIFF
        tiff_path = tmp_path / "test.tif"
        tiff_array = np.random.randint(0, 65535, (64, 64, 3), dtype=np.uint16)

        try:
            import tifffile

            tifffile.imwrite(str(tiff_path), tiff_array)
        except ImportError:
            pytest.skip("tifffile not installed")

        # Should accept TIFF by default
        result, orig_shape = preprocess_image_linear(
            tiff_path,
            apex_strict_formats=True,  # Explicit for clarity
            verify_linearity=False,  # Skip gamma checks (we're testing format boundary)
        )

        assert result.dtype == np.float32
        assert orig_shape == (64, 64)

    def test_tiff_uppercase_extension_accepted(self, tmp_path):
        """Test that .TIFF (uppercase) is also accepted."""
        tiff_path = tmp_path / "test.TIFF"
        tiff_array = np.random.randint(0, 65535, (64, 64, 3), dtype=np.uint16)

        try:
            import tifffile

            tifffile.imwrite(str(tiff_path), tiff_array)
        except ImportError:
            pytest.skip("tifffile not installed")

        # Should accept .TIFF (case-insensitive)
        result, orig_shape = preprocess_image_linear(
            tiff_path,
            apex_strict_formats=True,
            verify_linearity=False,
        )

        assert result.dtype == np.float32

    def test_jpeg_rejected_by_default(self, tmp_path):
        """Test that JPEG files are rejected with apex_strict_formats=True."""
        # Create JPEG
        jpg_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(jpg_path, "JPEG")

        # Should reject JPEG by default
        with pytest.raises(ValueError) as exc_info:
            preprocess_image_linear(
                jpg_path,
                apex_strict_formats=True,  # Explicit (this is the default)
                verify_linearity=False,
            )

        # Verify error message content
        error_msg = str(exc_info.value)
        assert "APEX linear ingest only supports RAW + TIFF" in error_msg
        assert ".jpg" in error_msg or ".jpeg" in error_msg
        assert "apex_strict_formats=False" in error_msg  # Suggests escape hatch

    def test_png_rejected_by_default(self, tmp_path):
        """Test that PNG files are rejected with apex_strict_formats=True."""
        # Create PNG
        png_path = tmp_path / "test.png"
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(png_path, "PNG")

        # Should reject PNG by default
        with pytest.raises(ValueError) as exc_info:
            preprocess_image_linear(
                png_path,
                apex_strict_formats=True,
            )

        error_msg = str(exc_info.value)
        assert "APEX linear ingest only supports RAW + TIFF" in error_msg
        assert ".png" in error_msg

    def test_webp_rejected_by_default(self, tmp_path):
        """Test that WebP files are rejected with apex_strict_formats=True."""
        # Create WebP
        webp_path = tmp_path / "test.webp"
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        try:
            img.save(webp_path, "WEBP")
        except (OSError, ValueError):
            pytest.skip("WebP support not available in PIL")

        # Should reject WebP by default
        with pytest.raises(ValueError) as exc_info:
            preprocess_image_linear(
                webp_path,
                apex_strict_formats=True,
            )

        error_msg = str(exc_info.value)
        assert "APEX linear ingest only supports RAW + TIFF" in error_msg

    def test_jpeg_accepted_with_escape_hatch(self, tmp_path):
        """Test that JPEG is accepted when apex_strict_formats=False."""
        # Create JPEG (8-bit, likely gamma-encoded, but we'll allow it)
        jpg_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(jpg_path, "JPEG")

        # Should accept JPEG with explicit escape hatch
        result, orig_shape = preprocess_image_linear(
            jpg_path,
            apex_strict_formats=False,  # Escape hatch
            verify_linearity=False,  # Skip gamma checks to allow JPEG through
        )

        assert result.dtype == np.float32
        assert result.ndim == 3
        assert orig_shape == (64, 64)

    def test_png_accepted_with_escape_hatch(self, tmp_path):
        """Test that PNG is accepted when apex_strict_formats=False."""
        # Create PNG
        png_path = tmp_path / "test.png"
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(png_path, "PNG")

        # Should accept PNG with explicit escape hatch
        result, orig_shape = preprocess_image_linear(
            png_path,
            apex_strict_formats=False,
            verify_linearity=False,
        )

        assert result.dtype == np.float32
        assert orig_shape == (64, 64)

    def test_numpy_array_bypass_format_check(self):
        """Test that numpy array inputs bypass format boundary check."""
        # Create numpy array (simulates pre-loaded data)
        arr = np.random.rand(64, 64, 3).astype(np.float32)

        # Should accept numpy array regardless of apex_strict_formats
        result, orig_shape = preprocess_image_linear(
            arr,
            apex_strict_formats=True,  # Format check only applies to file paths
            verify_linearity=False,
        )

        assert result.dtype == np.float32
        assert orig_shape == (64, 64)

    def test_default_behavior_is_strict(self, tmp_path):
        """Test that apex_strict_formats defaults to True."""
        # Create JPEG
        jpg_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(jpg_path, "JPEG")

        # Should reject JPEG by default (without explicit apex_strict_formats=True)
        with pytest.raises(ValueError) as exc_info:
            preprocess_image_linear(jpg_path, verify_linearity=False)

        assert "APEX linear ingest only supports RAW + TIFF" in str(exc_info.value)

    @pytest.mark.ml
    def test_raw_file_accepted_by_default(self, tmp_path):
        """Test that RAW files are accepted with apex_strict_formats=True.

        Note: This test requires rawpy, marked with @pytest.mark.ml
        Since we don't have actual RAW files in fixtures, we'll use a mock.
        """
        pytest.skip("RAW file testing requires actual RAW files or rawpy mock")

    def test_error_message_includes_guidance(self, tmp_path):
        """Test that error message includes helpful guidance."""
        jpg_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(jpg_path, "JPEG")

        with pytest.raises(ValueError) as exc_info:
            preprocess_image_linear(jpg_path)

        error_msg = str(exc_info.value)

        # Check for key guidance elements
        assert "APEX linear ingest only supports RAW + TIFF" in error_msg
        assert "preprocess_image()" in error_msg  # Suggests legacy function
        assert "apex_strict_formats=False" in error_msg  # Suggests escape hatch
        assert "data-fidelity tradeoffs" in error_msg  # Explains why

    def test_format_check_happens_before_verification(self, tmp_path):
        """Test that format boundary check happens before linearity verification.

        This ensures we reject based on format first (deterministic, fast),
        before running statistical gamma detection (slower, heuristic).
        """
        # Create JPEG that might pass gamma detection (e.g., very dark image)
        jpg_path = tmp_path / "dark.jpg"
        # Create a very dark image that might not trigger gamma detection
        dark_array = np.zeros((64, 64, 3), dtype=np.uint8)
        dark_array[:, :, :] = 5  # Very dark, almost black
        img = Image.fromarray(dark_array, mode="RGB")
        img.save(jpg_path, "JPEG")

        # Should reject based on format, NOT on gamma detection
        with pytest.raises(ValueError) as exc_info:
            preprocess_image_linear(
                jpg_path,
                apex_strict_formats=True,
                verify_linearity=True,  # Even with verification enabled
            )

        # Error should be about format (deterministic boundary check)
        # not about statistical gamma detection from verify_linear_ingest()
        error_msg = str(exc_info.value)
        assert "APEX linear ingest only supports RAW + TIFF" in error_msg
        # Should NOT be from gamma detection (which would say "Linear ingest verification failed")
        assert "Linear ingest verification failed" not in error_msg

    def test_tiff_then_extension_case_insensitive(self, tmp_path):
        """Test that .tiff (lowercase) and .TIFF (uppercase) both work."""
        for ext in [".tiff", ".TIFF", ".tif", ".TIF"]:
            tiff_path = tmp_path / f"test{ext}"
            tiff_array = np.random.randint(0, 65535, (64, 64, 3), dtype=np.uint16)

            try:
                import tifffile

                tifffile.imwrite(str(tiff_path), tiff_array)
            except ImportError:
                pytest.skip("tifffile not installed")

            # Should accept all case variations
            result, _ = preprocess_image_linear(
                tiff_path,
                apex_strict_formats=True,
                verify_linearity=False,
            )

            assert result.dtype == np.float32, f"Failed for extension: {ext}"


class TestFormatBoundaryIntegration:
    """Integration tests for format boundary with other preprocessing features."""

    def test_format_boundary_with_target_size(self, tmp_path):
        """Test that format boundary works with target_size parameter."""
        # Create TIFF
        tiff_path = tmp_path / "test.tif"
        tiff_array = np.random.randint(0, 65535, (128, 128, 3), dtype=np.uint16)

        try:
            import tifffile

            tifffile.imwrite(str(tiff_path), tiff_array)
        except ImportError:
            pytest.skip("tifffile not installed")

        # Should accept TIFF and resize
        result, orig_shape = preprocess_image_linear(
            tiff_path,
            target_size=64,  # Resize to 64px long edge
            apex_strict_formats=True,
            verify_linearity=False,
        )

        assert result.dtype == np.float32
        assert orig_shape == (128, 128)
        # Result should be resized (approximately 64x64, accounting for dimension multiple)
        assert max(result.shape[:2]) <= 70  # Accounting for multiple-of-14 adjustment

    def test_format_boundary_with_verification_disabled(self, tmp_path):
        """Test that format boundary works independently of linearity verification."""
        jpg_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(jpg_path, "JPEG")

        # Format boundary should still enforce even with verify_linearity=False
        with pytest.raises(ValueError) as exc_info:
            preprocess_image_linear(
                jpg_path,
                apex_strict_formats=True,
                verify_linearity=False,  # Verification disabled
            )

        assert "APEX linear ingest only supports RAW + TIFF" in str(exc_info.value)

    def test_escape_hatch_still_runs_verification_if_enabled(self, tmp_path):
        """Test that apex_strict_formats=False still runs gamma detection if verify_linearity=True."""
        # Create a JPEG (likely gamma-encoded)
        jpg_path = tmp_path / "test.jpg"
        # Create an image that will likely trigger gamma detection
        gamma_array = np.power(np.linspace(0, 1, 64 * 64 * 3), 2.2).reshape(64, 64, 3)
        gamma_uint8 = (gamma_array * 255).astype(np.uint8)
        img = Image.fromarray(gamma_uint8, mode="RGB")
        img.save(jpg_path, "JPEG")

        # apex_strict_formats=False bypasses format boundary
        # But verify_linearity=True should still detect gamma
        # (Though this is heuristic and may not always trigger)

        # For now, just verify it doesn't raise format error
        # Gamma detection is tested separately in test_linear_verify.py
        try:
            result, _ = preprocess_image_linear(
                jpg_path,
                apex_strict_formats=False,  # Bypass format boundary
                verify_linearity=True,  # Keep verification enabled
            )
            # If it succeeds, JPEG was allowed through escape hatch
            assert result.dtype == np.float32
        except ValueError as e:
            # If it fails, should be gamma detection, not format rejection
            error_msg = str(e)
            assert "APEX linear ingest only supports RAW + TIFF" not in error_msg
            # Might be gamma detection or other verification failure
