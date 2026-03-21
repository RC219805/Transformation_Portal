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



pytestmark = pytest.mark.unit

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


@pytest.fixture
def context():
    """Create a StageContext for testing."""
    from transformation_portal.stage_graph.stage import StageContext

    return StageContext(artifacts={}, config={}, device="cpu")


class TestSegmentMaterialsOutputNormalization:
    """Test output normalization in _segment_materials."""

    def test_handles_tuple_outputs(self, stage, sample_image, sample_mask, context):
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
        result = stage._segment_materials(sample_image, None, context)

        # Should extract masks successfully
        assert len(result) == 3
        assert all(mat in result for mat in ["glass", "water", "foliage"])

        # All values should be np.float32 masks
        for material, mask in result.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.float32
            assert mask.shape == sample_image.shape[:2]

    def test_handles_mask_only_outputs(self, stage, sample_image, sample_mask, context):
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
        result = stage._segment_materials(sample_image, None, context)

        # Should extract masks successfully
        assert len(result) == 3
        assert all(mat in result for mat in ["glass", "water", "stone"])

        # All values should be np.float32 masks
        for material, mask in result.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.float32

    def test_handles_mixed_outputs(self, stage, sample_image, sample_mask, context):
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
        result = stage._segment_materials(sample_image, None, context)

        # Should extract all masks successfully
        assert len(result) == 4
        assert all(mat in result for mat in ["glass", "water", "foliage", "stone"])

        # All values should be np.float32 masks
        for material, mask in result.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.float32

    def test_skips_invalid_outputs_gracefully(self, stage, sample_image, sample_mask, context, caplog):
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
            result = stage._segment_materials(sample_image, None, context)

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

    def test_logs_confidence_when_available(self, stage, sample_image, sample_mask, context, caplog):
        """Should log confidence scores when available."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask * 0.8, 0.87),
            "water": (sample_mask * 0.5, 0.64),
        }
        stage._segmenter = mock_segmenter

        import logging

        with caplog.at_level(logging.DEBUG):
            stage._segment_materials(sample_image, None, context)

        # Should log confidence percentages
        log_text = " ".join(caplog.messages)
        assert "87%" in log_text or "0.87" in log_text
        assert "64%" in log_text or "0.64" in log_text

    def test_logs_no_confidence_for_mask_only(self, stage, sample_image, sample_mask, context, caplog):
        """Should log when no confidence is available."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": sample_mask * 0.8,
        }
        stage._segmenter = mock_segmenter

        import logging

        with caplog.at_level(logging.DEBUG):
            stage._segment_materials(sample_image, None, context)

        # Should log that no confidence was provided
        log_text = " ".join(caplog.messages).lower()
        assert "no confidence" in log_text

    def test_default_soft_failure_on_exception(self, stage, sample_image, context):
        """Should return empty dict by default when segmenter raises an exception (soft failure)."""
        # Mock segmenter that raises an exception
        mock_segmenter = Mock()
        mock_segmenter.segment.side_effect = RuntimeError("Segmentation backend crashed")
        stage._segmenter = mock_segmenter

        # Should return empty dict (soft failure) instead of propagating exception
        result = stage._segment_materials(sample_image, None, context)
        assert result == {}

    def test_coerces_confidence_to_float(self, stage, sample_image, sample_mask, context):
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
        result = stage._segment_materials(sample_image, None, context)

        # All masks should be present
        assert len(result) == 3
        assert all(isinstance(mask, np.ndarray) for mask in result.values())

    def test_handles_empty_results(self, stage, sample_image, context):
        """Should handle empty segmentation results gracefully."""
        # Mock segmenter that finds no materials
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {}
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, context)

        # Should return empty dict (not crash)
        assert result == {}
        assert isinstance(result, dict)

    def test_validates_mask_dtype_conversion(self, stage, sample_image, context):
        """Should convert masks to float32 regardless of input dtype."""
        # Mock segmenter with various mask dtypes
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (np.ones((128, 128), dtype=np.uint8), 0.87),
            "water": (np.ones((128, 128), dtype=np.float64), 0.64),
            "stone": np.ones((128, 128), dtype=bool),  # Mask only, bool type
        }
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, context)

        # All masks should be converted to float32
        for material, mask in result.items():
            assert mask.dtype == np.float32, f"{material} mask dtype is {mask.dtype}, expected float32"


class TestSoftFailureBehavior:
    """Test soft failure vs strict failure modes (Issue 1)."""

    def test_soft_failure_default_behavior(self, stage, sample_image, caplog):
        """Should return empty dict and continue pipeline when segmenter fails (default)."""
        from transformation_portal.stage_graph.stage import StageContext

        # Mock segmenter that raises an exception
        mock_segmenter = Mock()
        mock_segmenter.segment.side_effect = RuntimeError("Backend crashed")
        stage._segmenter = mock_segmenter

        # Create context WITHOUT strict mode (default)
        context = StageContext(artifacts={"image": sample_image}, config={}, device="cpu")  # No strict mode

        import logging

        with caplog.at_level(logging.WARNING):
            result = stage._segment_materials(sample_image, None, context)

        # Should return empty dict (soft failure)
        assert result == {}
        assert isinstance(result, dict)

        # Should log the soft failure warning
        log_text = " ".join(caplog.messages)
        assert "soft failure" in log_text.lower()
        assert "materials_segmentation_strict" in log_text

    def test_strict_mode_raises(self, stage, sample_image):
        """Should propagate exception when strict mode is enabled."""
        from transformation_portal.stage_graph.stage import StageContext

        # Mock segmenter that raises an exception
        mock_segmenter = Mock()
        mock_segmenter.segment.side_effect = RuntimeError("Backend crashed")
        stage._segmenter = mock_segmenter

        # Create context WITH strict mode
        context = StageContext(artifacts={"image": sample_image}, config={"materials_segmentation_strict": True}, device="cpu")

        # Should raise exception (hard failure)
        with pytest.raises(RuntimeError, match="Backend crashed"):
            stage._segment_materials(sample_image, None, context)

    def test_soft_failure_preserves_pipeline_execution(self, stage):
        """Should allow pipeline to continue after soft failure."""
        from transformation_portal.stage_graph.stage import StageContext

        sample_image = np.zeros((128, 128, 3), dtype=np.uint8)

        # Mock segmenter that fails
        mock_segmenter = Mock()
        mock_segmenter.segment.side_effect = RuntimeError("Backend error")
        stage._segmenter = mock_segmenter

        # Create context without strict mode
        context = StageContext(artifacts={"image": sample_image}, config={}, device="cpu")

        # Call compute() which should complete with empty materials
        result = stage.compute(context)

        # Stage should complete successfully (not FAILED)
        # Note: compute() wraps _segment_materials, so we test the full flow
        from transformation_portal.stage_graph.stage import StageStatus

        assert result.status == StageStatus.COMPLETED
        assert result.artifacts["material_masks"] == {}


class TestConfidenceValidation:
    """Test confidence validation logic (Issue 2)."""

    def test_confidence_validation_invalid_range_above(self, stage, sample_image, sample_mask, context):
        """Should discard confidence values above 1.0."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask, 2.5),  # 250% confidence - invalid
        }
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, context)

        # Should keep mask, discard invalid confidence
        assert "glass" in result
        assert isinstance(result["glass"], np.ndarray)

    def test_confidence_validation_invalid_range_below(self, stage, sample_image, sample_mask, context):
        """Should discard confidence values below 0.0."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "water": (sample_mask, -0.5),  # Negative confidence - invalid
        }
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, context)

        # Should keep mask, discard invalid confidence
        assert "water" in result
        assert isinstance(result["water"], np.ndarray)

    def test_confidence_validation_nan(self, stage, sample_image, sample_mask, context):
        """Should discard NaN confidence values."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "metal": (sample_mask, float("nan")),  # NaN - invalid
        }
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, context)

        # Should keep mask, discard invalid confidence
        assert "metal" in result
        assert isinstance(result["metal"], np.ndarray)

    def test_confidence_validation_inf(self, stage, sample_image, sample_mask, context):
        """Should discard Inf confidence values."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "wood": (sample_mask, float("inf")),  # Inf - invalid
        }
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, context)

        # Should keep mask, discard invalid confidence
        assert "wood" in result
        assert isinstance(result["wood"], np.ndarray)

    def test_confidence_validation_valid(self, stage, sample_image, sample_mask, context, caplog):
        """Should accept valid confidence values in [0.0, 1.0]."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask, 0.0),  # Lower bound - valid
            "water": (sample_mask, 0.5),  # Mid-range - valid
            "stone": (sample_mask, 1.0),  # Upper bound - valid
            "wood": (sample_mask, 0.87),  # Normal value - valid
        }
        stage._segmenter = mock_segmenter

        import logging

        with caplog.at_level(logging.DEBUG):
            result = stage._segment_materials(sample_image, None, context)

        # All masks should be present
        assert len(result) == 4
        assert all(mat in result for mat in ["glass", "water", "stone", "wood"])

        # Should log confidence values
        log_text = " ".join(caplog.messages)
        assert "87%" in log_text or "0.87" in log_text

    def test_confidence_validation_edge_case_exactly_zero(self, stage, sample_image, sample_mask, context):
        """Should accept exactly 0.0 as valid confidence."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask, 0.0),
        }
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, context)

        assert "glass" in result
        assert isinstance(result["glass"], np.ndarray)

    def test_confidence_validation_edge_case_exactly_one(self, stage, sample_image, sample_mask, context):
        """Should accept exactly 1.0 as valid confidence."""
        mock_segmenter = Mock()
        mock_segmenter.segment.return_value = {
            "glass": (sample_mask, 1.0),
        }
        stage._segmenter = mock_segmenter

        result = stage._segment_materials(sample_image, None, context)

        assert "glass" in result
        assert isinstance(result["glass"], np.ndarray)
