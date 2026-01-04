#!/usr/bin/env python3
"""
Tests for Depth Writer Statistics Module
=========================================

Tests for lux_depth_v3/enhance/depth_writer.py focusing on:
- Invalid fraction tracking (NaN, Inf values)
- Clipping fraction computation (actual saturation at 0 and 65535)
- All-invalid depth array handling
- Atomic write produces final file
- Pre-quantized uint16 passthrough
- Zero-size depth array edge cases
"""

import pytest
import numpy as np
import math
from pathlib import Path
from PIL import Image

from lux_depth_v3.enhance.depth_writer import (
    write_depth_u16_png,
    write_depth_u16_png_with_stats,
    atomic_write_depth_u16_png,
    atomic_write_depth_u16_png_with_stats,
    read_depth_u16_png,
    DepthScalingStats,
)


class TestDepthWriterInvalidFractions:
    """Test invalid value fraction tracking."""

    def test_invalid_fraction_with_nan(self, tmp_path):
        """Test invalid fraction tracking with NaN values."""
        out = tmp_path / "depth_nan.png"

        # Create depth with 10% NaN values (10 out of 100 pixels)
        depth = np.ones((10, 10), dtype=np.float32) * 100.0
        depth[0, :] = np.nan  # First row is NaN (10 pixels)

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Should detect 10% invalid (10/100)
        assert math.isclose(stats.invalid_frac, 0.10, abs_tol=1e-12)

        # Output should exist and be valid uint16
        assert out.exists()
        depth_read = read_depth_u16_png(out)
        assert depth_read.dtype == np.uint16
        assert not np.isnan(depth_read).any()  # NaN replaced with median

    def test_invalid_fraction_with_inf(self, tmp_path):
        """Test invalid fraction tracking with Inf values."""
        out = tmp_path / "depth_inf.png"

        # Create depth with 5% Inf values (5 out of 100 pixels)
        depth = np.ones((10, 10), dtype=np.float32) * 50.0
        depth[0, :5] = np.inf  # 5 pixels are +Inf

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Should detect 5% invalid (5/100)
        assert math.isclose(stats.invalid_frac, 0.05, abs_tol=1e-12)

        # Output should be valid
        assert out.exists()
        depth_read = read_depth_u16_png(out)
        assert np.isfinite(depth_read).all()

    def test_invalid_fraction_mixed_nan_inf(self, tmp_path):
        """Test invalid fraction with both NaN and Inf."""
        out = tmp_path / "depth_mixed.png"

        # 100 pixels: 10 NaN, 5 +Inf, 5 -Inf, 80 valid
        depth = np.ones((10, 10), dtype=np.float32) * 25.0
        depth[0, :] = np.nan  # 10 NaN
        depth[1, :5] = np.inf  # 5 +Inf
        depth[1, 5:] = -np.inf  # 5 -Inf

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Should detect 20% invalid (20/100)
        assert math.isclose(stats.invalid_frac, 0.20, abs_tol=1e-12)

    def test_all_invalid_depth_array(self, tmp_path):
        """Test all-invalid depth array (should produce zeros with warning)."""
        out = tmp_path / "depth_all_invalid.png"

        # All NaN
        depth = np.full((10, 10), np.nan, dtype=np.float32)

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Should detect 100% invalid
        assert math.isclose(stats.invalid_frac, 1.0, abs_tol=1e-12)

        # Output should be all zeros (fallback when no valid data)
        depth_read = read_depth_u16_png(out)
        assert (depth_read == 0).all()


class TestDepthWriterClippingFractions:
    """Test clipping fraction computation for actual saturation."""

    def test_clipping_fractions_measure_actual_saturation(self, tmp_path):
        """Verify clipping fractions count actual uint16 extremes (0, 65535)."""
        out = tmp_path / "depth_clip.png"

        # 100 pixels: 1 extreme low, 1 extreme high, 98 in range
        depth = np.full((10, 10), 50.0, dtype=np.float32)
        depth[0, 0] = -100.0  # Will clip to 0 after p1p99 normalization
        depth[0, 1] = 200.0  # Will clip to 65535 after p1p99 normalization

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Verify actual clipping in quantized output
        depth_read = read_depth_u16_png(out)
        low_count = np.count_nonzero(depth_read == 0)
        high_count = np.count_nonzero(depth_read == 65535)

        # Both should be at least 1 pixel each
        assert low_count >= 1
        assert high_count >= 1

        # Clipping fractions should match actual counts
        assert math.isclose(stats.clipped_low_frac, low_count / 100.0, abs_tol=1e-12)
        assert math.isclose(stats.clipped_high_frac, high_count / 100.0, abs_tol=1e-12)

    def test_no_clipping_with_narrow_range(self, tmp_path):
        """Test that no clipping occurs when values are within percentile range."""
        out = tmp_path / "depth_no_clip.png"

        # All values between 10 and 90 (p1p99 will be ~10 and ~90)
        depth = np.linspace(10, 90, 100, dtype=np.float32).reshape(10, 10)

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # With narrow range and linear distribution, minimal/no clipping at extremes
        # Actually, with p1p99, the 1st and 99th percentiles will cause some clipping
        # But the clipping should be minimal
        assert stats.clipped_low_frac <= 0.02  # At most 2% clipped low (1st percentile)
        assert stats.clipped_high_frac <= 0.02  # At most 2% clipped high (99th percentile)

    def test_minmax_prevents_clipping(self, tmp_path):
        """Test that minmax method prevents clipping."""
        out = tmp_path / "depth_minmax.png"

        # Values from 0 to 100
        depth = np.linspace(0, 100, 100, dtype=np.float32).reshape(10, 10)

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="minmax")

        # minmax should map min to 0 and max to 65535 with no additional clipping
        # However, edge values will still saturate at 0 and 65535
        depth_read = read_depth_u16_png(out)

        # First pixel (min) should be 0, last pixel (max) should be 65535
        assert depth_read.flat[0] == 0
        assert depth_read.flat[-1] == 65535

        # Clipping fractions should be minimal (just the min/max values)
        # With 100 pixels, min and max are each 1 pixel → 1/100 = 1%
        assert math.isclose(stats.clipped_low_frac, 0.01, abs_tol=0.005)
        assert math.isclose(stats.clipped_high_frac, 0.01, abs_tol=0.005)

    def test_high_clipping_fraction_with_outliers(self, tmp_path):
        """Test high clipping fraction when many outliers exist."""
        out = tmp_path / "depth_outliers.png"

        # 100 pixels: 20 very low, 20 very high, 60 normal
        depth = np.full((10, 10), 50.0, dtype=np.float32)
        depth[:2, :] = 0.0  # 20 pixels at low extreme
        depth[-2:, :] = 150.0  # 20 pixels at high extreme

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # With p1p99, the outliers should be clipped
        # At least 19% should be clipped at each end (20% outliers, minus percentile tolerance)
        assert stats.clipped_low_frac >= 0.15
        assert stats.clipped_high_frac >= 0.15


class TestDepthWriterUint16Passthrough:
    """Test pre-quantized uint16 passthrough (new in PR #651)."""

    def test_uint16_passthrough_preserves_values(self, tmp_path):
        """Test that uint16 input is passed through without requantization."""
        out = tmp_path / "depth_u16_passthrough.png"

        # Create uint16 depth with specific values
        depth = np.array(
            [
                [0, 1000, 32767],
                [32768, 65535, 50000],
            ],
            dtype=np.uint16,
        )

        p1, p99 = write_depth_u16_png(out, depth, method="p1p99")

        # Read back and verify exact match
        depth_read = read_depth_u16_png(out)
        assert np.array_equal(depth_read, depth)

        # Percentiles should be computed from uint16 values
        assert p1 == pytest.approx(np.percentile(depth.astype(np.float32), 1.0), abs=1e-6)
        assert p99 == pytest.approx(np.percentile(depth.astype(np.float32), 99.0), abs=1e-6)

    def test_uint16_passthrough_with_stats(self, tmp_path):
        """Test uint16 passthrough with statistics computation (note: still requantizes)."""
        out = tmp_path / "depth_u16_stats.png"

        # Create uint16 depth
        depth = np.random.randint(1000, 60000, size=(50, 50), dtype=np.uint16)

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Read back - NOTE: write_depth_u16_png_with_stats ALWAYS requantizes,
        # unlike write_depth_u16_png which preserves uint16 input
        depth_read = read_depth_u16_png(out)
        assert depth_read.shape == depth.shape

        # Invalid fraction should be 0 (no NaN/Inf in uint16)
        assert stats.invalid_frac == 0.0

        # Verify stats are computed correctly
        assert stats.method == "p1p99"
        assert 0.0 <= stats.clipped_low_frac <= 1.0
        assert 0.0 <= stats.clipped_high_frac <= 1.0


class TestDepthWriterAtomicWrite:
    """Test atomic write functionality."""

    def test_atomic_write_produces_final_file(self, tmp_path):
        """Test that atomic write produces the final file."""
        out = tmp_path / "depth_atomic.png"

        depth = np.random.rand(20, 20).astype(np.float32) * 100.0

        p1, p99 = atomic_write_depth_u16_png(out, depth, method="p1p99")

        # Final file should exist
        assert out.exists()

        # Temporary file should NOT exist
        tmp = out.with_suffix(".tmp.png")
        assert not tmp.exists()

        # Read back and verify
        depth_read = read_depth_u16_png(out)
        assert depth_read.shape == depth.shape
        assert depth_read.dtype == np.uint16

    def test_atomic_write_with_stats(self, tmp_path):
        """Test atomic write with statistics."""
        out = tmp_path / "depth_atomic_stats.png"

        depth = np.random.rand(30, 30).astype(np.float32) * 50.0

        _p1, _p99, stats = atomic_write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Final file should exist
        assert out.exists()

        # Temporary file should NOT exist
        tmp = out.with_suffix(".tmp.png")
        assert not tmp.exists()

        # Stats should be valid
        assert isinstance(stats, DepthScalingStats)
        assert stats.method == "p1p99"
        assert 0.0 <= stats.invalid_frac <= 1.0

    def test_atomic_write_cleanup_on_error(self, tmp_path):
        """Test that temporary file is cleaned up on error."""
        out = tmp_path / "depth_error.png"

        # Invalid depth (wrong shape)
        depth = np.random.rand(5, 5, 5, 5).astype(np.float32)

        with pytest.raises(IOError):
            atomic_write_depth_u16_png(out, depth, method="p1p99")

        # Final file should NOT exist
        assert not out.exists()

        # Temporary file should be cleaned up
        tmp = out.with_suffix(".tmp.png")
        assert not tmp.exists()


class TestDepthWriterEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_size_depth_array(self, tmp_path):
        """Test zero-size depth array (edge case that may fail in np.percentile)."""
        out = tmp_path / "depth_empty.png"

        # Empty array triggers IndexError in numpy percentile
        # This is an edge case that the implementation doesn't handle
        depth = np.array([], dtype=np.float32).reshape(0, 0)

        # This will raise an error - document as known limitation
        with pytest.raises(IndexError):
            write_depth_u16_png_with_stats(out, depth, method="p1p99")

    def test_single_pixel_depth(self, tmp_path):
        """Test single-pixel depth array."""
        out = tmp_path / "depth_single.png"

        depth = np.array([[42.0]], dtype=np.float32)

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Should complete without error
        assert out.exists()

        # Single pixel: all percentiles are the same value
        assert math.isclose(stats.v_low_value, 42.0, abs_tol=1e-6)
        assert math.isclose(stats.v_high_value, 42.0, abs_tol=1e-6)

    def test_depth_range_too_small(self, tmp_path):
        """Test depth with range too small (triggers zero output)."""
        out = tmp_path / "depth_small_range.png"

        # All pixels same value (p1 ≈ p99)
        depth = np.full((10, 10), 50.0, dtype=np.float32)

        _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method="p1p99")

        # Should produce zeros (fallback for degenerate range)
        depth_read = read_depth_u16_png(out)
        assert (depth_read == 0).all()

        # All pixels clipped to 0
        assert math.isclose(stats.clipped_low_frac, 1.0, abs_tol=1e-12)

    def test_3d_depth_takes_first_channel(self, tmp_path):
        """Test that 3D depth array takes first channel."""
        out = tmp_path / "depth_3d.png"

        # 3D depth (H, W, C) with C=3
        depth_3d = np.random.rand(10, 10, 3).astype(np.float32) * 100.0

        _p1, _p99 = write_depth_u16_png(out, depth_3d, method="p1p99")

        # Should take first channel and write as 2D
        depth_read = read_depth_u16_png(out)
        assert depth_read.ndim == 2
        assert depth_read.shape == (10, 10)

    def test_invalid_depth_shape_raises_error(self, tmp_path):
        """Test that invalid depth shape raises ValueError."""
        out = tmp_path / "depth_invalid.png"

        # 4D depth is invalid
        depth = np.random.rand(5, 5, 5, 5).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 2D or 3D depth"):
            write_depth_u16_png(out, depth, method="p1p99")

    def test_debug_verify_option(self, tmp_path):
        """Test debug verification option."""
        out = tmp_path / "depth_verify.png"

        depth = np.random.rand(15, 15).astype(np.float32) * 80.0

        # Should not raise with valid depth
        _p1, _p99 = write_depth_u16_png(out, depth, method="p1p99", debug_verify=True)

        # Read back and verify
        depth_read = read_depth_u16_png(out)
        assert depth_read.shape == depth.shape

    def test_different_quantization_methods(self, tmp_path):
        """Test all quantization methods produce valid output."""
        depth = np.random.rand(20, 20).astype(np.float32) * 60.0

        for method in ["p1p99", "p0.5p99.5", "minmax"]:
            out = tmp_path / f"depth_{method}.png"
            _p1, _p99, stats = write_depth_u16_png_with_stats(out, depth, method=method)

            assert out.exists()
            assert stats.method == method

            # Verify percentiles match method
            if method == "p1p99":
                assert stats.p_low_percentile == 1.0
                assert stats.p_high_percentile == 99.0
            elif method == "p0.5p99.5":
                assert stats.p_low_percentile == 0.5
                assert stats.p_high_percentile == 99.5
            elif method == "minmax":
                assert stats.p_low_percentile == 0.0
                assert stats.p_high_percentile == 100.0

    def test_read_nonexistent_file_raises_error(self, tmp_path):
        """Test that reading nonexistent file raises FileNotFoundError."""
        out = tmp_path / "nonexistent.png"

        with pytest.raises(FileNotFoundError):
            read_depth_u16_png(out)
