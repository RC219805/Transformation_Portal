#!/usr/bin/env python3
"""
Tests for Depth Estimation Loss Functions

Tests covering:
- Scale-Invariant Loss
- Gradient Loss
- SSIM Loss
- L1 Loss
- Combined Loss

Author: Transformation Portal Team
Version: 1.0.0
"""

import pytest

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch required for loss function tests"
)


class TestScaleInvariantLoss:
    """Test Scale-Invariant Loss."""

    def test_loss_init(self):
        """Test ScaleInvariantLoss initialization."""
        from src.training.losses import ScaleInvariantLoss

        loss_fn = ScaleInvariantLoss()

        assert loss_fn.variance_focus == 0.85
        assert loss_fn.eps == 1e-6

    def test_loss_shape(self):
        """Test loss returns scalar."""
        from src.training.losses import ScaleInvariantLoss

        loss_fn = ScaleInvariantLoss()

        pred = torch.rand(4, 1, 64, 64) + 0.1
        target = torch.rand(4, 1, 64, 64) + 0.1

        loss = loss_fn(pred, target)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_identical_zero_loss(self):
        """Test identical inputs give near-zero loss."""
        from src.training.losses import ScaleInvariantLoss

        loss_fn = ScaleInvariantLoss()

        data = torch.rand(2, 1, 32, 32) + 0.1
        loss = loss_fn(data, data.clone())

        assert loss.item() < 1e-5

    def test_different_scales_invariance(self):
        """Test loss is invariant to global scale."""
        from src.training.losses import ScaleInvariantLoss

        loss_fn = ScaleInvariantLoss()

        target = torch.rand(2, 1, 32, 32) + 0.5

        # Prediction at different global scales
        pred_1x = target.clone()
        pred_2x = target * 2

        loss_1x = loss_fn(pred_1x, target)
        loss_2x = loss_fn(pred_2x, target)

        # Scale-invariant loss should be similar for uniform scaling
        # (not exactly equal due to numerical precision)
        assert abs(loss_1x.item() - loss_2x.item()) < 0.1

    def test_mask_support(self):
        """Test loss supports validity mask."""
        from src.training.losses import ScaleInvariantLoss

        loss_fn = ScaleInvariantLoss()

        pred = torch.rand(2, 1, 32, 32) + 0.1
        target = torch.rand(2, 1, 32, 32) + 0.1

        # Create mask (only half valid)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, :, :16, :] = True

        loss = loss_fn(pred, target, mask)

        assert loss.ndim == 0
        assert loss.item() >= 0


class TestGradientLoss:
    """Test Gradient Loss."""

    def test_loss_init(self):
        """Test GradientLoss initialization."""
        from src.training.losses import GradientLoss

        loss_fn = GradientLoss(scales=4)

        assert loss_fn.scales == 4
        assert hasattr(loss_fn, "sobel_x")
        assert hasattr(loss_fn, "sobel_y")

    def test_loss_shape(self):
        """Test loss returns scalar."""
        from src.training.losses import GradientLoss

        loss_fn = GradientLoss()

        pred = torch.rand(4, 1, 64, 64)
        target = torch.rand(4, 1, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_identical_zero_loss(self):
        """Test identical inputs give near-zero loss."""
        from src.training.losses import GradientLoss

        loss_fn = GradientLoss()

        data = torch.rand(2, 1, 32, 32)
        loss = loss_fn(data, data.clone())

        assert loss.item() < 1e-5

    def test_edge_sensitivity(self):
        """Test loss is sensitive to edge differences."""
        from src.training.losses import GradientLoss

        loss_fn = GradientLoss()

        # Target with edges
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, :, 16:] = 1.0  # Vertical edge

        # Prediction without edges
        pred_no_edge = torch.ones(1, 1, 32, 32) * 0.5

        # Prediction with matching edges
        pred_with_edge = target.clone()

        loss_no_edge = loss_fn(pred_no_edge, target)
        loss_with_edge = loss_fn(pred_with_edge, target)

        # Matching edges should have lower loss
        assert loss_with_edge.item() < loss_no_edge.item()


class TestSSIMLoss:
    """Test SSIM Loss."""

    def test_loss_init(self):
        """Test SSIMLoss initialization."""
        from src.training.losses import SSIMLoss

        loss_fn = SSIMLoss(window_size=11)

        assert loss_fn.window_size == 11
        assert hasattr(loss_fn, "window")

    def test_loss_shape(self):
        """Test loss returns scalar."""
        from src.training.losses import SSIMLoss

        loss_fn = SSIMLoss()

        pred = torch.rand(4, 1, 64, 64)
        target = torch.rand(4, 1, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert 0 <= loss.item() <= 2  # 1 - SSIM, SSIM in [-1, 1]

    def test_identical_zero_loss(self):
        """Test identical inputs give near-zero loss (SSIM=1)."""
        from src.training.losses import SSIMLoss

        loss_fn = SSIMLoss()

        data = torch.rand(2, 1, 32, 32)
        loss = loss_fn(data, data.clone())

        # 1 - SSIM(identical) should be close to 0
        assert loss.item() < 0.01

    def test_structural_sensitivity(self):
        """Test loss is sensitive to structural differences."""
        from src.training.losses import SSIMLoss

        loss_fn = SSIMLoss()

        target = torch.rand(1, 1, 64, 64)

        # Shifted version (structural change)
        pred_shifted = torch.roll(target, shifts=5, dims=2)

        # Same mean/variance but different structure
        loss = loss_fn(pred_shifted, target)

        # Should have higher loss for structural mismatch
        assert loss.item() > 0.01


class TestL1Loss:
    """Test L1 Loss."""

    def test_loss_shape(self):
        """Test loss returns scalar."""
        from src.training.losses import L1Loss

        loss_fn = L1Loss()

        pred = torch.rand(4, 1, 64, 64)
        target = torch.rand(4, 1, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_identical_zero_loss(self):
        """Test identical inputs give zero loss."""
        from src.training.losses import L1Loss

        loss_fn = L1Loss()

        data = torch.rand(2, 1, 32, 32)
        loss = loss_fn(data, data.clone())

        assert loss.item() < 1e-6

    def test_log_transform(self):
        """Test log transform option."""
        from src.training.losses import L1Loss

        loss_fn = L1Loss(use_log=True)

        pred = torch.rand(2, 1, 32, 32) + 0.5
        target = torch.rand(2, 1, 32, 32) + 0.5

        loss = loss_fn(pred, target)

        assert loss.item() >= 0


class TestCombinedDepthLoss:
    """Test Combined Depth Loss."""

    def test_loss_init(self):
        """Test CombinedDepthLoss initialization."""
        from src.training.losses import CombinedDepthLoss

        loss_fn = CombinedDepthLoss()

        assert "scale_invariant" in loss_fn.weights
        assert len(loss_fn.losses) > 0

    def test_custom_weights(self):
        """Test custom weight initialization."""
        from src.training.losses import CombinedDepthLoss

        weights = {
            "scale_invariant": 2.0,
            "gradient": 1.0,
            "ssim": 0.5,
        }

        loss_fn = CombinedDepthLoss(weights=weights)

        assert loss_fn.weights["scale_invariant"] == 2.0
        assert loss_fn.weights["gradient"] == 1.0

    def test_loss_output(self):
        """Test loss returns total and dictionary."""
        from src.training.losses import CombinedDepthLoss

        loss_fn = CombinedDepthLoss()

        pred = torch.rand(4, 1, 64, 64) + 0.1
        target = torch.rand(4, 1, 64, 64) + 0.1

        total_loss, loss_dict = loss_fn(pred, target)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict

    def test_backward_pass(self):
        """Test loss supports backward pass."""
        from src.training.losses import CombinedDepthLoss

        loss_fn = CombinedDepthLoss()

        pred = torch.rand(2, 1, 32, 32, requires_grad=True) + 0.1
        target = torch.rand(2, 1, 32, 32) + 0.1

        total_loss, _ = loss_fn(pred, target)
        total_loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_get_weights(self):
        """Test weight retrieval."""
        from src.training.losses import CombinedDepthLoss

        weights = {"scale_invariant": 1.5, "gradient": 0.5}
        loss_fn = CombinedDepthLoss(weights=weights)

        retrieved = loss_fn.get_weights()

        assert retrieved["scale_invariant"] == 1.5

    def test_set_weights(self):
        """Test weight updating."""
        from src.training.losses import CombinedDepthLoss

        loss_fn = CombinedDepthLoss()

        loss_fn.set_weights({"scale_invariant": 2.0})

        assert loss_fn.weights["scale_invariant"] == 2.0


class TestLossGradients:
    """Test that losses have well-behaved gradients."""

    @pytest.fixture
    def sample_data(self):
        """Create sample prediction and target."""
        pred = torch.rand(2, 1, 32, 32, requires_grad=True) + 0.1
        target = torch.rand(2, 1, 32, 32) + 0.1
        return pred, target

    def test_scale_invariant_gradient(self, sample_data):
        """Test ScaleInvariantLoss has finite gradients."""
        from src.training.losses import ScaleInvariantLoss

        pred, target = sample_data
        loss_fn = ScaleInvariantLoss()

        loss = loss_fn(pred, target)
        loss.backward()

        assert torch.isfinite(pred.grad).all()

    def test_gradient_loss_gradient(self, sample_data):
        """Test GradientLoss has finite gradients."""
        from src.training.losses import GradientLoss

        pred, target = sample_data
        loss_fn = GradientLoss()

        loss = loss_fn(pred, target)
        loss.backward()

        assert torch.isfinite(pred.grad).all()

    def test_ssim_loss_gradient(self, sample_data):
        """Test SSIMLoss has finite gradients."""
        from src.training.losses import SSIMLoss

        pred, target = sample_data
        loss_fn = SSIMLoss()

        loss = loss_fn(pred, target)
        loss.backward()

        assert torch.isfinite(pred.grad).all()

    def test_combined_loss_gradient(self, sample_data):
        """Test CombinedDepthLoss has finite gradients."""
        from src.training.losses import CombinedDepthLoss

        pred, target = sample_data
        loss_fn = CombinedDepthLoss()

        total_loss, _ = loss_fn(pred, target)
        total_loss.backward()

        assert torch.isfinite(pred.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
