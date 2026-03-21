"""Unit tests for _canonical_mask helper (Phase A.1).

Tests the 3D mask bug fix for SAM2 integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.pixel_ops_executor import _bounding_box, _canonical_mask

pytestmark = pytest.mark.unit


class TestCanonicalMask:
    """Test _canonical_mask helper function."""

    def test_canonical_mask_2d_float32(self):
        """Test that 2D float32 masks pass through unchanged."""
        mask = np.random.rand(100, 100).astype(np.float32)
        result = _canonical_mask(mask)

        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, mask)

    def test_canonical_mask_2d_uint8(self):
        """Test that 2D uint8 masks are converted to float32."""
        mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        result = _canonical_mask(mask)

        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, mask.astype(np.float32))

    def test_canonical_mask_3d_single_channel_hwc(self):
        """Test that (H, W, 1) masks are squeezed correctly."""
        # This is the SAM2 case that was crashing
        mask_2d = np.random.rand(100, 100).astype(np.float32)
        mask_3d = mask_2d[..., np.newaxis]  # Add channel dimension

        assert mask_3d.shape == (100, 100, 1)

        result = _canonical_mask(mask_3d)

        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, mask_2d)

    def test_canonical_mask_3d_single_channel_chw(self):
        """Test that (1, H, W) masks are squeezed correctly."""
        mask_2d = np.random.rand(100, 100).astype(np.float32)
        mask_3d = mask_2d[np.newaxis, ...]  # Add batch dimension

        assert mask_3d.shape == (1, 100, 100)

        result = _canonical_mask(mask_3d)

        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, mask_2d)

    def test_canonical_mask_3d_multi_channel_raises(self):
        """Test that (H, W, C) with C > 1 raises ValueError."""
        mask_rgb = np.random.rand(100, 100, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Cannot canonicalize 3D mask"):
            _canonical_mask(mask_rgb)

    def test_canonical_mask_4d_raises(self):
        """Test that 4D masks raise ValueError."""
        mask_4d = np.random.rand(1, 100, 100, 1).astype(np.float32)

        with pytest.raises(ValueError, match="Cannot canonicalize mask with 4 dimensions"):
            _canonical_mask(mask_4d)

    def test_canonical_mask_1d_raises(self):
        """Test that 1D masks raise ValueError."""
        mask_1d = np.random.rand(100).astype(np.float32)

        with pytest.raises(ValueError, match="Cannot canonicalize mask with 1 dimensions"):
            _canonical_mask(mask_1d)

    def test_canonical_mask_preserves_values(self):
        """Test that canonical mask preserves actual mask values."""
        # Create binary mask
        mask_2d = np.zeros((50, 50), dtype=np.float32)
        mask_2d[10:40, 10:40] = 1.0

        # Test 2D
        result_2d = _canonical_mask(mask_2d)
        np.testing.assert_array_equal(result_2d, mask_2d)

        # Test 3D (H, W, 1)
        mask_3d = mask_2d[..., np.newaxis]
        result_3d = _canonical_mask(mask_3d)
        np.testing.assert_array_equal(result_3d, mask_2d)

    def test_canonical_mask_with_boolean_input(self):
        """Test that boolean masks are converted to float32."""
        mask_bool = np.random.rand(100, 100) > 0.5

        result = _canonical_mask(mask_bool)

        assert result.dtype == np.float32
        assert result.shape == (100, 100)
        # Boolean True should become 1.0, False should become 0.0
        assert set(np.unique(result)) <= {0.0, 1.0}


class TestBoundingBoxWith3DMasks:
    """Test _bounding_box with 3D masks (integration test for A.1)."""

    def test_bounding_box_with_2d_mask(self):
        """Test bounding box computation with standard 2D mask."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 30:70] = 1.0

        bbox = _bounding_box(mask)

        assert bbox is not None
        x0, y0, x1, y1 = bbox
        assert x0 == 30
        assert y0 == 20
        assert x1 == 70
        assert y1 == 80

    def test_bounding_box_with_3d_mask_hwc(self):
        """Test bounding box computation with (H, W, 1) mask (SAM2 case)."""
        mask_2d = np.zeros((100, 100), dtype=np.float32)
        mask_2d[20:80, 30:70] = 1.0

        # Create 3D mask (H, W, 1) like SAM2 returns
        mask_3d = mask_2d[..., np.newaxis]

        # This should NOT crash anymore after canonicalization
        bbox = _bounding_box(mask_3d)

        assert bbox is not None
        x0, y0, x1, y1 = bbox
        assert x0 == 30
        assert y0 == 20
        assert x1 == 70
        assert y1 == 80

    def test_bounding_box_with_empty_mask(self):
        """Test that empty masks return None."""
        mask = np.zeros((100, 100), dtype=np.float32)

        bbox = _bounding_box(mask)

        assert bbox is None

    def test_bounding_box_with_single_pixel(self):
        """Test bounding box with single pixel mask."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[50, 60] = 1.0

        bbox = _bounding_box(mask)

        assert bbox is not None
        x0, y0, x1, y1 = bbox
        assert x0 == 60
        assert y0 == 50
        assert x1 == 61
        assert y1 == 51

    def test_bounding_box_with_threshold(self):
        """Test that bounding box uses 0.5 threshold correctly."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 30:70] = 0.6  # Above threshold
        mask[10:15, 10:15] = 0.4  # Below threshold (should be ignored)

        bbox = _bounding_box(mask)

        assert bbox is not None
        x0, y0, x1, y1 = bbox
        # Should only include the 0.6 region, not the 0.4 region
        assert x0 == 30
        assert y0 == 20
        assert x1 == 70
        assert y1 == 80


class TestCanonicalMaskEdgeCases:
    """Test edge cases for canonical mask."""

    def test_canonical_mask_with_zero_mask(self):
        """Test canonical mask with all-zero mask."""
        mask = np.zeros((100, 100), dtype=np.float32)

        result = _canonical_mask(mask)

        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        assert np.all(result == 0.0)

    def test_canonical_mask_with_ones_mask(self):
        """Test canonical mask with all-ones mask."""
        mask = np.ones((100, 100), dtype=np.float32)

        result = _canonical_mask(mask)

        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        assert np.all(result == 1.0)

    def test_canonical_mask_with_fractional_values(self):
        """Test canonical mask preserves fractional values."""
        mask = np.random.rand(50, 50).astype(np.float32)

        result = _canonical_mask(mask)

        np.testing.assert_array_equal(result, mask)

    def test_canonical_mask_minimal_size(self):
        """Test canonical mask with minimal 1x1 mask."""
        mask = np.array([[0.5]], dtype=np.float32)

        result = _canonical_mask(mask)

        assert result.shape == (1, 1)
        assert result[0, 0] == 0.5
