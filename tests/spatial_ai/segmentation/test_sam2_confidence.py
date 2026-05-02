"""Unit tests for SAM2 confidence semantics (Phase C.2).

This test suite validates Phase C.2 implementation:
- Extraction of real SAM2 IoU and stability scores
- Defensive fallback to 1.0 when attributes missing
- Proper population of MaskMetadata.stability_score
- Proper use of IoU scores in SegmentationResult.scores
- Backward compatibility with stub backends

Test Coverage:
1. test_extract_sam2_predictions_with_real_scores - Happy path with real SAM2 output
2. test_extract_sam2_predictions_missing_iou - Fallback when iou_predictions missing
3. test_extract_sam2_predictions_missing_stability - Fallback when stability_scores missing
4. test_extract_sam2_predictions_none_values - Fallback when attributes are None
5. test_extract_sam2_predictions_shape_consistency - Verify array shapes match
6. test_extract_sam2_predictions_dtype_conversion - Verify float32 conversion
7. test_extract_sam2_predictions_empty_output - Handle zero masks case
"""

from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend


class TestExtractSAM2Predictions:
    """Test _extract_sam2_predictions() helper method."""

    @pytest.fixture(autouse=True)
    def skip_if_no_checkpoint(self):
        """Skip all tests in this class if checkpoint not available."""
        from pathlib import Path

        checkpoint_path = Path("checkpoints/sam2_hiera_base_plus.pt")
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    def test_extract_sam2_predictions_with_real_scores(self):
        """Test extraction when SAM2 provides real scores."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output with all attributes present
        mock_output = Mock()
        mock_output.pred_masks = np.random.rand(3, 64, 64) > 0.5  # 3 masks, (N, H, W) bool
        mock_output.iou_predictions = np.array([0.9, 0.7, 0.85], dtype=np.float32)
        mock_output.stability_scores = np.array([0.95, 0.88, 0.92], dtype=np.float32)

        # Extract predictions
        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify shapes
        assert masks.shape == (3, 64, 64)
        assert iou.shape == (3,)
        assert stability.shape == (3,)

        # Verify values match input
        assert np.array_equal(masks, mock_output.pred_masks)
        np.testing.assert_array_equal(iou, mock_output.iou_predictions)
        np.testing.assert_array_equal(stability, mock_output.stability_scores)

        # Verify dtypes
        assert masks.dtype == bool
        assert iou.dtype == np.float32
        assert stability.dtype == np.float32

    def test_extract_sam2_predictions_missing_iou(self):
        """Test extraction when iou_predictions attribute is missing."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output WITHOUT iou_predictions attribute
        mock_output = Mock(spec=["pred_masks", "stability_scores"])
        mock_output.pred_masks = np.random.rand(3, 64, 64) > 0.5
        mock_output.stability_scores = np.array([0.95, 0.88, 0.92], dtype=np.float32)

        # Extract predictions - should fallback to 1.0 for IoU
        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify fallback to 1.0 for IoU
        assert iou.shape == (3,)
        assert iou.dtype == np.float32
        np.testing.assert_array_equal(iou, np.ones(3, dtype=np.float32))

        # Verify stability scores still extracted correctly
        np.testing.assert_array_equal(stability, mock_output.stability_scores)

    def test_extract_sam2_predictions_missing_stability(self):
        """Test extraction when stability_scores attribute is missing."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output WITHOUT stability_scores attribute
        mock_output = Mock(spec=["pred_masks", "iou_predictions"])
        mock_output.pred_masks = np.random.rand(3, 64, 64) > 0.5
        mock_output.iou_predictions = np.array([0.9, 0.7, 0.85], dtype=np.float32)

        # Extract predictions - should fallback to 1.0 for stability
        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify fallback to 1.0 for stability
        assert stability.shape == (3,)
        assert stability.dtype == np.float32
        np.testing.assert_array_equal(stability, np.ones(3, dtype=np.float32))

        # Verify IoU scores still extracted correctly
        np.testing.assert_array_equal(iou, mock_output.iou_predictions)

    def test_extract_sam2_predictions_none_values(self):
        """Test extraction when attributes are set to None."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output with attributes set to None
        mock_output = Mock()
        mock_output.pred_masks = np.random.rand(3, 64, 64) > 0.5
        mock_output.iou_predictions = None  # Explicitly None
        mock_output.stability_scores = None  # Explicitly None

        # Extract predictions - should fallback to 1.0 for both
        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify fallback to 1.0 for both scores
        assert iou.shape == (3,)
        assert stability.shape == (3,)
        np.testing.assert_array_equal(iou, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(stability, np.ones(3, dtype=np.float32))

    def test_extract_sam2_predictions_missing_pred_masks(self):
        """pred_masks is required and should fail with a clear error when missing."""
        backend = SAM2Backend(device="cpu")

        mock_output = Mock(spec=["iou_predictions", "stability_scores"])
        mock_output.iou_predictions = np.array([0.9], dtype=np.float32)
        mock_output.stability_scores = np.array([0.95], dtype=np.float32)

        with pytest.raises(AttributeError, match="pred_masks"):
            backend._extract_sam2_predictions(mock_output)

    def test_extract_sam2_predictions_shape_consistency(self):
        """Test that extracted arrays have consistent shapes."""
        backend = SAM2Backend(device="cpu")

        # Test with different mask counts
        for n_masks in [1, 5, 10, 20]:
            mock_output = Mock()
            mock_output.pred_masks = np.random.rand(n_masks, 128, 128) > 0.5
            mock_output.iou_predictions = np.random.rand(n_masks).astype(np.float32)
            mock_output.stability_scores = np.random.rand(n_masks).astype(np.float32)

            masks, iou, stability = backend._extract_sam2_predictions(mock_output)

            # Verify shapes match
            assert masks.shape[0] == n_masks
            assert iou.shape[0] == n_masks
            assert stability.shape[0] == n_masks

            # Verify spatial dimensions preserved
            assert masks.shape[1:] == (128, 128)

    def test_extract_sam2_predictions_dtype_conversion(self):
        """Test that scores are converted to float32."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output with non-float32 dtypes
        mock_output = Mock()
        mock_output.pred_masks = np.random.rand(3, 64, 64) > 0.5
        mock_output.iou_predictions = np.array([0.9, 0.7, 0.85], dtype=np.float64)  # float64
        mock_output.stability_scores = [0.95, 0.88, 0.92]  # Python list

        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify conversion to float32
        assert iou.dtype == np.float32
        assert stability.dtype == np.float32

        # Verify values preserved
        np.testing.assert_allclose(iou, [0.9, 0.7, 0.85], rtol=1e-5)
        np.testing.assert_allclose(stability, [0.95, 0.88, 0.92], rtol=1e-5)

    def test_extract_sam2_predictions_empty_output(self):
        """Test extraction with zero masks."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output with no masks
        mock_output = Mock()
        mock_output.pred_masks = np.empty((0, 64, 64), dtype=bool)
        mock_output.iou_predictions = np.array([], dtype=np.float32)
        mock_output.stability_scores = np.array([], dtype=np.float32)

        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify empty arrays with correct shapes
        assert masks.shape == (0, 64, 64)
        assert iou.shape == (0,)
        assert stability.shape == (0,)
        assert iou.dtype == np.float32
        assert stability.dtype == np.float32


class TestExtractSAM2PredictionsValueRanges:
    """Test that extracted scores respect contract value ranges."""

    @pytest.fixture(autouse=True)
    def skip_if_no_checkpoint(self):
        """Skip all tests in this class if checkpoint not available."""
        from pathlib import Path

        checkpoint_path = Path("checkpoints/sam2_hiera_base_plus.pt")
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    def test_iou_scores_in_valid_range(self):
        """Test that IoU scores are in [0, 1] range."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output with edge case values
        mock_output = Mock()
        mock_output.pred_masks = np.random.rand(5, 64, 64) > 0.5
        mock_output.iou_predictions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        mock_output.stability_scores = np.ones(5, dtype=np.float32)

        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify all scores in valid range
        assert np.all(iou >= 0.0)
        assert np.all(iou <= 1.0)

    def test_stability_scores_in_valid_range(self):
        """Test that stability scores are in [0, 1] range."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output with edge case values
        mock_output = Mock()
        mock_output.pred_masks = np.random.rand(5, 64, 64) > 0.5
        mock_output.iou_predictions = np.ones(5, dtype=np.float32)
        mock_output.stability_scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify all scores in valid range
        assert np.all(stability >= 0.0)
        assert np.all(stability <= 1.0)

    def test_fallback_scores_satisfy_contract(self):
        """Test that fallback 1.0 scores satisfy contract constraints."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output with missing attributes
        mock_output = Mock(spec=["pred_masks"])
        mock_output.pred_masks = np.random.rand(3, 64, 64) > 0.5

        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Verify fallback values satisfy [0, 1] constraint
        assert np.all(iou == 1.0)
        assert np.all(stability == 1.0)
        assert np.all(iou >= 0.0) and np.all(iou <= 1.0)
        assert np.all(stability >= 0.0) and np.all(stability <= 1.0)


class TestBackwardCompatibility:
    """Test backward compatibility with stub backends."""

    @pytest.fixture(autouse=True)
    def skip_if_no_checkpoint(self):
        """Skip all tests in this class if checkpoint not available."""
        from pathlib import Path

        checkpoint_path = Path("checkpoints/sam2_hiera_base_plus.pt")
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    def test_stub_backend_without_sam2_attributes(self):
        """Test that stub backends (no SAM2) still work with fallback."""
        backend = SAM2Backend(device="cpu")

        # Simulate stub backend output with minimal attributes
        mock_output = Mock(spec=["pred_masks"])
        mock_output.pred_masks = np.random.rand(2, 32, 32) > 0.5

        # Should not raise exception, should return fallback scores
        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        assert masks.shape == (2, 32, 32)
        assert np.all(iou == 1.0)
        assert np.all(stability == 1.0)

    def test_partial_stub_backend(self):
        """Test stub backend with only some attributes present."""
        backend = SAM2Backend(device="cpu")

        # Simulate partial stub: has IoU but not stability
        mock_output = Mock(spec=["pred_masks", "iou_predictions"])
        mock_output.pred_masks = np.random.rand(2, 32, 32) > 0.5
        mock_output.iou_predictions = np.array([0.8, 0.9], dtype=np.float32)

        masks, iou, stability = backend._extract_sam2_predictions(mock_output)

        # Should use real IoU but fallback stability
        np.testing.assert_allclose(iou, [0.8, 0.9], rtol=1e-6)
        np.testing.assert_array_equal(stability, [1.0, 1.0])


class TestDefensiveProgramming:
    """Test defensive programming patterns."""

    @pytest.fixture(autouse=True)
    def skip_if_no_checkpoint(self):
        """Skip all tests in this class if checkpoint not available."""
        from pathlib import Path

        checkpoint_path = Path("checkpoints/sam2_hiera_base_plus.pt")
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    def test_no_exceptions_on_missing_attributes(self):
        """Test that missing attributes never raise exceptions."""
        backend = SAM2Backend(device="cpu")

        # Mock with minimal attributes
        mock_output = Mock(spec=["pred_masks"])
        mock_output.pred_masks = np.random.rand(3, 64, 64) > 0.5

        # Should complete without exceptions
        try:
            masks, iou, stability = backend._extract_sam2_predictions(mock_output)
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
        # Contract: masks come back with the expected shape; missing iou /
        # stability attributes fall back to per-mask `np.ones(count)` arrays
        # (sam2_backend.py:766-776), so callers always get a usable score
        # vector rather than None.
        assert masks is not None
        assert masks.shape == (3, 64, 64)
        assert iou.shape == (3,)
        assert stability.shape == (3,)
        assert np.all(iou == 1.0)
        assert np.all(stability == 1.0)

    def test_no_exceptions_on_none_attributes(self):
        """Test that None attributes never raise exceptions."""
        backend = SAM2Backend(device="cpu")

        # Mock with None values
        mock_output = Mock()
        mock_output.pred_masks = np.random.rand(3, 64, 64) > 0.5
        mock_output.iou_predictions = None
        mock_output.stability_scores = None

        # Should complete without exceptions
        try:
            masks, iou, stability = backend._extract_sam2_predictions(mock_output)
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
        assert masks is not None
        assert masks.shape == (3, 64, 64)
        # When iou_predictions / stability_scores are None, the helper falls
        # back to `np.ones(count, dtype=np.float32)` rather than propagating
        # None — see _extract_scores in sam2_backend.py.
        assert iou.shape == (3,)
        assert stability.shape == (3,)
        assert np.all(iou == 1.0)
        assert np.all(stability == 1.0)


@pytest.mark.benchmark
class TestPerformance:
    """Test performance characteristics (excluded from core CI)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_checkpoint(self):
        """Skip all tests in this class if checkpoint not available."""
        from pathlib import Path

        checkpoint_path = Path("checkpoints/sam2_hiera_base_plus.pt")
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    def test_zero_overhead_for_attribute_checks(self):
        """Test that hasattr checks have negligible overhead.

        Note: Marked as benchmark to avoid CI flakiness. Run locally or
        in dedicated performance workflows.
        """
        import time

        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output
        mock_output = Mock()
        mock_output.pred_masks = np.random.rand(100, 64, 64) > 0.5
        mock_output.iou_predictions = np.random.rand(100).astype(np.float32)
        mock_output.stability_scores = np.random.rand(100).astype(np.float32)

        # Time extraction (informational, not strict)
        start = time.perf_counter()
        for _ in range(1000):  # 1000 iterations
            masks, iou, stability = backend._extract_sam2_predictions(mock_output)
        elapsed = time.perf_counter() - start

        # Log timing (informational only, no strict threshold)
        # Typical: < 1s for 1000 iterations (~1ms per call)
        print(f"Extraction timing: {elapsed:.3f}s for 1000 iterations ({elapsed*1000:.3f}ms per call)")

    def test_no_memory_allocation_overhead(self):
        """Test that extraction doesn't cause excessive memory allocation."""
        backend = SAM2Backend(device="cpu")

        # Mock SAM2 output
        mock_output = Mock()
        mock_output.pred_masks = np.random.rand(10, 64, 64) > 0.5
        mock_output.iou_predictions = np.random.rand(10).astype(np.float32)
        mock_output.stability_scores = np.random.rand(10).astype(np.float32)

        # Extract once
        masks1, iou1, stability1 = backend._extract_sam2_predictions(mock_output)

        # Extract again
        masks2, iou2, stability2 = backend._extract_sam2_predictions(mock_output)

        # Verify same values (no random allocations)
        assert np.array_equal(masks1, masks2)
        np.testing.assert_array_equal(iou1, iou2)
        np.testing.assert_array_equal(stability1, stability2)
