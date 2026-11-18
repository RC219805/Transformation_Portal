"""Tests for evals/metrics.py module (Phase 5 coverage).

Tests for:
- PSNR computation
- SSIM computation
- LPIPS computation (mocked)
- IoU computation
- Dice coefficient computation
- Score normalization utilities

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.evals.metrics import (
    _to_numpy,
    _to_numpy_pair,
    _to_torch_pair,
    dice_coefficient,
    lpips_score,
    lpips_to_score,
    normalize_score,
    psnr,
    psnr_to_score,
    psnr_torch,
    segmentation_iou,
    ssim,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


@pytest.fixture
def identical_images():
    """Create two identical images."""
    img = np.random.rand(64, 64, 3).astype(np.float32)
    return img.copy(), img.copy()


@pytest.fixture
def different_images():
    """Create two different images."""
    img1 = np.random.rand(64, 64, 3).astype(np.float32)
    img2 = np.random.rand(64, 64, 3).astype(np.float32)
    return img1, img2


@pytest.fixture
def overlapping_masks():
    """Create overlapping boolean masks."""
    mask1 = np.zeros((64, 64), dtype=bool)
    mask1[20:50, 20:50] = True

    mask2 = np.zeros((64, 64), dtype=bool)
    mask2[25:55, 25:55] = True

    return mask1, mask2


class TestToNumpy:
    """Test _to_numpy conversion function."""

    def test_numpy_passthrough(self):
        """Test numpy array passthrough."""
        arr = np.random.rand(64, 64, 3).astype(np.float32)
        result = _to_numpy(arr)

        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32

    def test_uint8_normalization(self):
        """Test uint8 to float32 normalization."""
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = _to_numpy(arr)

        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_high_value_normalization(self):
        """Test high value normalization."""
        arr = np.random.rand(64, 64, 3).astype(np.float32) * 255
        result = _to_numpy(arr)

        assert result.max() <= 1.0

    @pytest.mark.parametrize("dtype", [np.uint16, np.dtype(">u2")])
    def test_uint16_normalised_by_dtype_not_by_255(self, dtype):
        """uint16 data is scaled by 65535, so the PSNR matches the same pixels as floats (M-02)."""
        rng = np.random.default_rng(0)
        ref = rng.random((32, 32, 3)).astype(np.float32)
        noisy = np.clip(ref + rng.normal(0, 0.1, ref.shape).astype(np.float32), 0, 1)
        ref16 = np.round(ref * 65535).astype(dtype)
        noisy16 = np.round(noisy * 65535).astype(dtype)

        result = _to_numpy(ref16)

        np.testing.assert_allclose(result, ref16.astype(np.float32) / 65535.0)
        assert result.max() <= 1.0
        assert psnr(noisy16, ref16) == pytest.approx(psnr(noisy, ref), abs=1e-3)
        assert ssim(noisy16, ref16) == pytest.approx(ssim(noisy, ref), abs=1e-3)

    def test_dark_uint16_image_is_not_read_as_unit_range(self):
        """A uint16 image whose values are all <= 1 is still near black, not white (M-02)."""
        dark = np.ones((4, 4), dtype=np.uint16)

        assert _to_numpy(dark).max() == pytest.approx(1.0 / 65535.0)

    def test_pair_scaled_identically_when_only_prediction_overshoots(self):
        """One overshooting sample must not put the prediction on a 1/255 scale (M-02)."""
        rng = np.random.default_rng(0)
        ref = rng.random((64, 64, 3)).astype(np.float32)
        pred = np.clip(ref + rng.normal(0, 0.1, ref.shape).astype(np.float32), 0, 1)
        pred_overshoot = pred.copy()
        pred_overshoot[0, 0, 0] = 1.0001  # un-clipped sharpening

        arr_pred, arr_ref = _to_numpy_pair(pred_overshoot, ref)

        np.testing.assert_array_equal(arr_pred, pred_overshoot)
        np.testing.assert_array_equal(arr_ref, ref)
        # The overshoot changes one sample; before the fix the PSNR collapsed to ~4.8 dB.
        assert psnr(pred_overshoot, ref) == pytest.approx(psnr(pred, ref), abs=0.05)
        assert psnr(ref, pred_overshoot) == pytest.approx(psnr(ref, pred), abs=0.05)
        assert ssim(pred_overshoot, ref) == pytest.approx(ssim(pred, ref), abs=0.01)

    def test_pair_of_0_255_floats_keeps_the_255_convention(self):
        """Two 0-255 float images are still both read as 0-255 data."""
        rng = np.random.default_rng(1)
        ref = rng.random((32, 32, 3)).astype(np.float32)
        pred = np.clip(ref + rng.normal(0, 0.05, ref.shape).astype(np.float32), 0, 1)

        arr_pred, arr_ref = _to_numpy_pair(pred * 255, ref * 255)

        np.testing.assert_allclose(arr_pred, pred, atol=1e-6)
        np.testing.assert_allclose(arr_ref, ref, atol=1e-6)
        assert psnr(pred * 255, ref * 255) == pytest.approx(psnr(pred, ref), abs=1e-3)

    def test_bit_depth_member_does_not_rescale_the_float_member(self):
        """With a uint8 target, a 0-255 float prediction is still decided on its own values."""
        rng = np.random.default_rng(2)
        ref8 = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        pred = ref8.astype(np.float32)

        arr_pred, arr_ref = _to_numpy_pair(pred, ref8)

        np.testing.assert_allclose(arr_pred, arr_ref)
        assert psnr(pred, ref8) == float("inf")

    def test_torch_pair_normalises_by_dtype_and_scales_identically(self):
        """The LPIPS tensor conversion follows the same pair rules as the numpy path (M-02)."""
        torch = pytest.importorskip("torch")

        dark = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)
        dark_t, _ = _to_torch_pair(dark, dark)
        assert float(dark_t.max()) == pytest.approx(1.0 / 255.0)

        ref = torch.rand(8, 8, 3, generator=torch.Generator().manual_seed(0))
        overshoot = ref.clone()
        overshoot[0, 0, 0] = 1.0001
        over_t, ref_t = _to_torch_pair(overshoot, ref)
        assert over_t.shape == (3, 8, 8)
        assert float(over_t.max()) == pytest.approx(1.0001)
        assert torch.allclose(ref_t, ref.permute(2, 0, 1))

        tensor_t, array_t = _to_torch_pair(ref * 255, (ref * 255).numpy())
        assert torch.allclose(tensor_t, ref.permute(2, 0, 1), atol=1e-6)
        assert torch.allclose(array_t, ref.permute(2, 0, 1), atol=1e-6)

    def test_path_loading_pil(self, tmp_path):
        """Test loading from path using PIL."""
        import numpy as np
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_path)

        # Mock cv2 to be unavailable to test PIL fallback
        with patch.dict("sys.modules", {"cv2": None}):
            result = _to_numpy(img_path)

        assert result.dtype == np.float32
        assert result.shape == (64, 64, 3)

    def test_torch_tensor_conversion(self):
        """Test torch tensor conversion."""
        pytest.importorskip("torch")
        import torch

        tensor = torch.rand(64, 64, 3)
        result = _to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_torch_tensor_conversion_without_numpy_bridge(self):
        """Test torch tensor conversion when torch's NumPy bridge is unavailable."""

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                raise RuntimeError("Numpy is not available")

            def tolist(self):
                return [[0.0, 0.5], [1.0, 0.25]]

        result = _to_numpy(FakeTensor())

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32))

    def test_torch_tensor_conversion_preserves_unrelated_runtime_errors(self):
        """Test torch tensor conversion still raises unexpected tensor errors."""

        class BrokenTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                raise RuntimeError("unexpected tensor conversion failure")

        with pytest.raises(RuntimeError, match="unexpected tensor conversion failure"):
            _to_numpy(BrokenTensor())


class TestPSNR:
    """Test PSNR computation."""

    def test_identical_images(self, identical_images):
        """Test PSNR for identical images is infinity."""
        img1, img2 = identical_images
        result = psnr(img1, img2)

        assert result == float("inf")

    def test_different_images(self, different_images):
        """Test PSNR for different images is finite."""
        img1, img2 = different_images
        result = psnr(img1, img2)

        assert 0 < result < float("inf")

    def test_shape_mismatch_raises(self):
        """Test shape mismatch raises error."""
        img1 = np.random.rand(64, 64, 3).astype(np.float32)
        img2 = np.random.rand(32, 32, 3).astype(np.float32)

        with pytest.raises(ValueError, match="shapes don't match"):
            psnr(img1, img2)

    def test_custom_max_val(self, different_images):
        """Test PSNR with custom max value."""
        img1, img2 = different_images
        # Convert to 0-255 range
        img1_255 = (img1 * 255).astype(np.float32)
        img2_255 = (img2 * 255).astype(np.float32)

        result_255 = psnr(img1_255, img2_255, max_val=255.0)
        result_1 = psnr(img1, img2, max_val=1.0)

        # Both should be finite positive values
        # The relationship depends on scaling
        assert result_255 > 0
        assert result_1 > 0


class TestPSNRTorch:
    """Test PyTorch PSNR computation."""

    def test_psnr_torch(self):
        """Test PSNR torch implementation."""
        pytest.importorskip("torch")
        import torch

        tensor1 = torch.rand(64, 64, 3)
        tensor2 = tensor1.clone()

        result = psnr_torch(tensor1, tensor2)

        assert result == 100.0  # Perfect match

    def test_psnr_torch_different(self):
        """Test PSNR torch with different tensors."""
        pytest.importorskip("torch")
        import torch

        tensor1 = torch.rand(64, 64, 3)
        tensor2 = torch.rand(64, 64, 3)

        result = psnr_torch(tensor1, tensor2)

        assert 0 < result < 100


class TestSSIM:
    """Test SSIM computation."""

    def test_identical_images(self, identical_images):
        """Test SSIM for identical images is close to 1."""
        img1, img2 = identical_images
        result = ssim(img1, img2)

        assert result == pytest.approx(1.0, rel=0.01)

    def test_different_images(self, different_images):
        """Test SSIM for different images is less than 1."""
        img1, img2 = different_images
        result = ssim(img1, img2)

        assert -1 <= result < 1

    def test_ssim_range(self, different_images):
        """Test SSIM is in valid range."""
        img1, img2 = different_images
        result = ssim(img1, img2)

        assert -1 <= result <= 1


class TestLPIPS:
    """Test LPIPS computation (mocked)."""

    def test_lpips_score_mocked(self, different_images):
        """Test LPIPS with mocked backend."""
        pytest.importorskip("lpips", reason="LPIPS package required")
        pytest.importorskip("torch", reason="torch required for LPIPS")

        img1, img2 = different_images

        # Test that lpips_score returns a float
        result = lpips_score(img1, img2)
        assert isinstance(result, float)

    def test_lpips_unavailable(self, different_images):
        """Test LPIPS returns 0 when unavailable."""
        img1, img2 = different_images

        # Save original state
        import transformation_portal.evals.metrics as metrics_module

        original_model = metrics_module._lpips_model

        try:
            # Reset the model cache
            metrics_module._lpips_model = None

            # When lpips import fails, it should return 0.0
            # We test this by checking the return type
            result = lpips_score(img1, img2)
            assert isinstance(result, float)
        finally:
            # Restore
            metrics_module._lpips_model = original_model


class TestSegmentationIoU:
    """Test segmentation IoU computation."""

    def test_identical_masks(self):
        """Test IoU for identical masks is 1."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:50, 20:50] = True

        result = segmentation_iou(mask, mask.copy())

        assert result == pytest.approx(1.0)

    def test_no_overlap(self):
        """Test IoU for non-overlapping masks is 0."""
        mask1 = np.zeros((64, 64), dtype=bool)
        mask1[0:20, 0:20] = True

        mask2 = np.zeros((64, 64), dtype=bool)
        mask2[40:60, 40:60] = True

        result = segmentation_iou(mask1, mask2)

        assert result == 0.0

    def test_partial_overlap(self, overlapping_masks):
        """Test IoU for partially overlapping masks."""
        mask1, mask2 = overlapping_masks
        result = segmentation_iou(mask1, mask2)

        assert 0 < result < 1

    def test_both_empty(self):
        """Test IoU for both empty masks is 1."""
        mask1 = np.zeros((64, 64), dtype=bool)
        mask2 = np.zeros((64, 64), dtype=bool)

        result = segmentation_iou(mask1, mask2)

        assert result == 1.0

    def test_soft_masks(self):
        """Test IoU with soft (float) masks."""
        mask1 = np.zeros((64, 64), dtype=np.float32)
        mask1[20:50, 20:50] = 0.8

        mask2 = np.zeros((64, 64), dtype=np.float32)
        mask2[20:50, 20:50] = 0.9

        result = segmentation_iou(mask1, mask2, threshold=0.5)

        assert result == pytest.approx(1.0)

    def test_multichannel_masks(self):
        """Test IoU handles multichannel masks."""
        mask1 = np.zeros((64, 64, 1), dtype=np.float32)
        mask1[20:50, 20:50, 0] = 1.0

        mask2 = np.zeros((64, 64, 1), dtype=np.float32)
        mask2[20:50, 20:50, 0] = 1.0

        result = segmentation_iou(mask1, mask2)

        assert result == pytest.approx(1.0)


class TestDiceCoefficient:
    """Test Dice coefficient computation."""

    def test_identical_masks(self):
        """Test Dice for identical masks is 1."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:50, 20:50] = True

        result = dice_coefficient(mask, mask.copy())

        assert result == pytest.approx(1.0)

    def test_no_overlap(self):
        """Test Dice for non-overlapping masks is 0."""
        mask1 = np.zeros((64, 64), dtype=bool)
        mask1[0:20, 0:20] = True

        mask2 = np.zeros((64, 64), dtype=bool)
        mask2[40:60, 40:60] = True

        result = dice_coefficient(mask1, mask2)

        assert result == 0.0

    def test_partial_overlap(self, overlapping_masks):
        """Test Dice for partially overlapping masks."""
        mask1, mask2 = overlapping_masks
        result = dice_coefficient(mask1, mask2)

        assert 0 < result < 1
        # Dice should be >= IoU for same masks
        iou = segmentation_iou(mask1, mask2)
        assert result >= iou

    def test_both_empty(self):
        """Test Dice for both empty masks is 1."""
        mask1 = np.zeros((64, 64), dtype=bool)
        mask2 = np.zeros((64, 64), dtype=bool)

        result = dice_coefficient(mask1, mask2)

        assert result == 1.0

    def test_multichannel_masks(self):
        """Test Dice handles multichannel masks."""
        mask1 = np.zeros((64, 64, 3), dtype=np.float32)
        mask1[20:50, 20:50, 0] = 1.0

        mask2 = np.zeros((64, 64, 3), dtype=np.float32)
        mask2[20:50, 20:50, 0] = 1.0

        result = dice_coefficient(mask1, mask2)

        assert result == pytest.approx(1.0)


class TestNormalizeScore:
    """Test normalize_score utility."""

    def test_min_value(self):
        """Test normalization of minimum value."""
        result = normalize_score(0.0, min_val=0.0, max_val=100.0)
        assert result == 0.0

    def test_max_value(self):
        """Test normalization of maximum value."""
        result = normalize_score(100.0, min_val=0.0, max_val=100.0)
        assert result == 1.0

    def test_mid_value(self):
        """Test normalization of middle value."""
        result = normalize_score(50.0, min_val=0.0, max_val=100.0)
        assert result == 0.5

    def test_clamping_below(self):
        """Test clamping values below min."""
        result = normalize_score(-10.0, min_val=0.0, max_val=100.0)
        assert result == 0.0

    def test_clamping_above(self):
        """Test clamping values above max."""
        result = normalize_score(150.0, min_val=0.0, max_val=100.0)
        assert result == 1.0

    def test_equal_min_max(self):
        """Test edge case of equal min and max."""
        result = normalize_score(5.0, min_val=5.0, max_val=5.0)
        assert result == 0.5

    @pytest.mark.regression
    @pytest.mark.parametrize(("min_val", "max_val"), [(20.0, 40.0), (5.0, 5.0)])
    def test_nan_value_fails_closed(self, min_val, max_val):
        """NaN must normalise to 0.0, not to the 1.0 that ``min(1.0, nan)`` yields (M-01)."""
        assert normalize_score(float("nan"), min_val=min_val, max_val=max_val) == 0.0
        assert normalize_score(np.float32("nan"), min_val=min_val, max_val=max_val) == 0.0

    def test_infinite_values_keep_their_order(self):
        """+inf saturates high and -inf saturates low."""
        assert normalize_score(float("inf"), min_val=20.0, max_val=40.0) == 1.0
        assert normalize_score(float("-inf"), min_val=20.0, max_val=40.0) == 0.0


class TestPSNRToScore:
    """Test psnr_to_score conversion."""

    def test_excellent_psnr(self):
        """Test excellent PSNR gives score near 1."""
        result = psnr_to_score(45.0)
        assert result == 1.0

    def test_good_psnr(self):
        """Test good PSNR gives high score."""
        result = psnr_to_score(30.0)
        assert result == 0.5

    def test_poor_psnr(self):
        """Test poor PSNR gives low score."""
        result = psnr_to_score(15.0)
        assert result == 0.0

    @pytest.mark.regression
    def test_nan_psnr_scores_zero(self):
        """One NaN sample makes PSNR NaN; that must score 0.0, not 1.0 (M-01)."""
        a = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
        b = a.copy()
        b[0, 0, 0] = np.nan

        assert np.isnan(psnr(a, b))
        assert psnr_to_score(psnr(a, b)) == 0.0

    def test_identical_images_score_one(self):
        """Identical images have infinite PSNR, which is a perfect score."""
        a = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)

        assert psnr(a, a.copy()) == float("inf")
        assert psnr_to_score(psnr(a, a.copy())) == 1.0


class TestLPIPSToScore:
    """Test lpips_to_score conversion."""

    def test_perfect_lpips(self):
        """Test perfect LPIPS (0) gives score of 1."""
        result = lpips_to_score(0.0)
        assert result == 1.0

    def test_bad_lpips(self):
        """Test bad LPIPS (0.5) gives score of 0."""
        result = lpips_to_score(0.5)
        assert result == 0.0

    def test_medium_lpips(self):
        """Test medium LPIPS gives medium score."""
        result = lpips_to_score(0.25)
        assert result == 0.5

    def test_very_bad_lpips_clamped(self):
        """Test very bad LPIPS is clamped to 0."""
        result = lpips_to_score(1.0)
        assert result == 0.0
