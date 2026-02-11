"""
Tests for MaterialSegmentation _segment_materials output format normalization.

Validates that the internal _segment_materials method correctly handles:
- Tuple outputs: (mask, confidence)
- Mask-only outputs: mask
- Mixed outputs: some tuples, some masks
- Invalid outputs: logged and skipped gracefully

These tests verify the defensive normalization logic that prevents production
failures when segmentation backends return different output formats.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from transformation_portal.stage_graph.stages.materials import MaterialSegmentationStage


@pytest.fixture
def stage():
    """Create a MaterialSegmentationStage for testing."""
    return MaterialSegmentationStage()


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.zeros((128, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample mask."""
    return np.ones((128, 128), dtype=np.float32)


class TestSegmentMaterialsOutputNormalization:
    """Test output normalization in _segment_materials."""

    def test_handles_tuple_outputs(self, stage, sample_image, sample_mask):
        """Should handle (mask, confidence) tuple outputs from modern backends."""
        # Mock segmenter that returns tuples
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask * 0.8, 0.87),
            "water": (sample_mask * 0.5, 0.64),
            "foliage": (sample_mask * 0.3, 0.45),
        }
        stage._segmenter = mock_segmenter

        # Process
        result = stage._segment_materials(sample_image, None, "cpu")

        # Should extract masks successfully
        assert len(result) == 3
        assert all(mat in result for mat in ["glass", "water", "foliage"])

        # All values should be np.float32 masks
        for material, mask in result.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.float32
            assert mask.shape == sample_image.shape[:2]

    def test_handles_mask_only_outputs(self, stage, sample_image, sample_mask):
        """Should handle mask-only outputs from legacy backends."""
        # Mock segmenter that returns masks only (no confidence)
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": sample_mask * 0.8,
            "water": sample_mask * 0.5,
            "stone": sample_mask * 0.3,
        }
        stage._segmenter = mock_segmenter

        # Process
        result = stage._segment_materials(sample_image, None, "cpu")

        # Should extract masks successfully
        assert len(result) == 3
        assert all(mat in result for mat in ["glass", "water", "stone"])

        # All values should be np.float32 masks
        for material, mask in result.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.float32

    def test_handles_mixed_outputs(self, stage, sample_image, sample_mask):
        """Should handle mixed tuple and mask-only outputs."""
        # Mock segmenter with mixed outputs
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask * 0.8, 0.87),  # Tuple (modern backend)
            "water": sample_mask * 0.5,  # Mask only (legacy backend)
            "foliage": (sample_mask * 0.3, 0.55),  # Tuple
            "stone": sample_mask * 0.2,  # Mask only
        }
        stage._segmenter = mock_segmenter

        # Process
        result = stage._segment_materials(sample_image, None, "cpu")

        # Should extract all masks successfully
        assert len(result) == 4
        assert all(mat in result for mat in ["glass", "water", "foliage", "stone"])

        # All values should be np.float32 masks
        for material, mask in result.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.float32

    def test_skips_invalid_outputs_gracefully(self, stage, sample_image, sample_mask, caplog):
        """Should skip invalid outputs and log warnings without crashing."""
        # Mock segmenter with invalid outputs mixed in
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask * 0.8, 0.87),  # Valid tuple
            "invalid_string": "not a mask",  # Invalid: string
            "water": sample_mask * 0.5,  # Valid mask
            "invalid_tuple": (sample_mask, "bad_conf", "extra"),  # Invalid: wrong tuple size
            "invalid_none": None,  # Invalid: None
        }
        stage._segmenter = mock_segmenter

        # Process
        import logging

        with caplog.at_level(logging.WARNING):
            result = stage._segment_materials(sample_image, None, "cpu")

        # Should extract only valid masks
        assert len(result) == 2
        assert "glass" in result
        assert "water" in result

        # Invalid entries should be skipped
        assert "invalid_string" not in result
        assert "invalid_tuple" not in result
        assert "invalid_none" not in result

        # Should have logged warnings
        assert any("Unexpected segmentation output format" in msg or "Failed to process" in msg for msg in caplog.messages)

    def test_logs_confidence_when_available(self, stage, sample_image, sample_mask, caplog):
        """Should log confidence scores when available."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask * 0.8, 0.87),
            "water": (sample_mask * 0.5, 0.64),
        }
        stage._segmenter = mock_segmenter

        import logging

        with caplog.at_level(logging.DEBUG):
            stage._segment_materials(sample_image, None, "cpu")

        # Should log confidence percentages
        log_text = " ".join(caplog.messages)
        assert "87%" in log_text or "0.87" in log_text
        assert "64%" in log_text or "0.64" in log_text

    def test_logs_no_confidence_for_mask_only(self, stage, sample_image, sample_mask, caplog):
        """Should log when no confidence is available."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": sample_mask * 0.8,
        }
        stage._segmenter = mock_segmenter

        import logging

        with caplog.at_level(logging.DEBUG):
            stage._segment_materials(sample_image, None, "cpu")

        # Should log that no confidence was provided
        log_text = " ".join(caplog.messages).lower()
        assert "no confidence" in log_text

    def test_fails_fast_on_segmenter_exception(self, stage, sample_image):
        """Should fail-fast when segmenter raises an exception (no silent failures)."""
        # Mock segmenter that raises an exception
        mock_segmenter = Mock()
        mock_segmenter.segment.side_effect = RuntimeError("Segmentation backend crashed")
        stage._segmenter = mock_segmenter

        # Should propagate the exception instead of silently returning empty dict
        with pytest.raises(RuntimeError, match="Segmentation backend crashed"):
            stage._segment_materials(sample_image, None, "cpu")

    def test_coerces_confidence_to_float(self, stage, sample_image, sample_mask):
        """Should coerce confidence values to float."""
        # Mock segmenter with various numeric types for confidence
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask * 0.8, 0.87),  # float
            "water": (sample_mask * 0.5, np.float32(0.64)),  # np.float32
            "stone": (sample_mask * 0.3, 55),  # int (will be coerced to 55.0)
        }
        stage._segmenter = mock_segmenter

        # Should not crash on type coercion
        result = stage._segment_materials(sample_image, None, "cpu")

        # All masks should be present
        assert len(result) == 3
        assert all(isinstance(mask, np.ndarray) for mask in result.values())

    def test_handles_empty_results(self, stage, sample_image):
        """Should handle empty segmentation results gracefully."""
        # Mock segmenter that finds no materials
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {}
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, "cpu")

        # Should return empty dict (not crash)
        assert result == {}
        assert isinstance(result, dict)

    def test_validates_mask_dtype_conversion(self, stage, sample_image):
        """Should convert masks to float32 regardless of input dtype."""
        # Mock segmenter with various mask dtypes
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (np.ones((128, 128), dtype=np.uint8), 0.87),
            "water": (np.ones((128, 128), dtype=np.float64), 0.64),
            "stone": np.ones((128, 128), dtype=bool),  # Mask only, bool type
        }
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, "cpu")

        # All masks should be converted to float32
        for material, mask in result.items():
            assert mask.dtype == np.float32, f"{material} mask dtype is {mask.dtype}, expected float32"
