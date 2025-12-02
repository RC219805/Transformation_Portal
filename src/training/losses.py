#!/usr/bin/env python3
"""
Loss Functions for Depth Estimation Training

Implements depth-specific loss functions:
- Scale-Invariant Loss (primary for depth)
- Gradient Loss (edge preservation)
- SSIM Loss (structural similarity)
- Combined Loss with configurable weights

References:
- Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
- Ranftl et al., "Vision Transformers for Dense Prediction"

Author: Transformation Portal Team
Version: 1.0.0
"""

import logging
from typing import Optional, Tuple

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    _BaseModule = nn.Module
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _BaseModule = object  # type: ignore

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    """Check if PyTorch is available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch required for loss functions. "
            "Install with: pip install torch"
        )


class ScaleInvariantLoss(_BaseModule):
    """Scale-Invariant Loss for depth estimation.

    This loss is invariant to the global scale of the depth prediction,
    making it suitable for monocular depth estimation where absolute scale
    is ambiguous.

    The loss is computed as:
        L = sqrt(1/n * sum(d_i^2) - λ/n^2 * (sum(d_i))^2)

    where d_i = log(pred_i) - log(gt_i) and λ controls the scale invariance.

    Reference:
        Eigen et al., "Depth Map Prediction from a Single Image using a
        Multi-Scale Deep Network", NeurIPS 2014

    Args:
        variance_focus: Weight for variance term (default: 0.85)
            Higher values emphasize relative depth ordering
        eps: Small value for numerical stability
    """

    def __init__(
        self,
        variance_focus: float = 0.85,
        eps: float = 1e-6,
    ):
        _check_torch()
        super().__init__()
        self.variance_focus = variance_focus
        self.eps = eps

    def forward(
        self,
        pred: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute scale-invariant loss.

        Args:
            pred: Predicted depth (B, 1, H, W) or (B, H, W)
            target: Ground truth depth (B, 1, H, W) or (B, H, W)
            mask: Optional valid pixel mask

        Returns:
            Scalar loss value
        """
        # Ensure same shape
        pred = pred.squeeze(1) if pred.dim() == 4 else pred
        target = target.squeeze(1) if target.dim() == 4 else target

        # Apply mask if provided
        if mask is None:
            mask = (target > self.eps) & (pred > self.eps)

        # Log difference
        log_pred = torch.log(pred.clamp(min=self.eps))
        log_target = torch.log(target.clamp(min=self.eps))
        diff = log_pred - log_target

        # Apply mask
        diff = diff[mask]

        if diff.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Scale-invariant loss
        diff_sq = (diff ** 2).mean()
        diff_mean_sq = (diff.mean()) ** 2

        loss = torch.sqrt(diff_sq - self.variance_focus * diff_mean_sq + self.eps)

        return loss


class GradientLoss(_BaseModule):
    """Gradient Loss for edge-aware depth estimation.

    Encourages the predicted depth map to have similar gradients
    as the ground truth, preserving depth discontinuities at edges.

    The loss computes L1 difference of horizontal and vertical gradients:
        L = |∂pred/∂x - ∂gt/∂x| + |∂pred/∂y - ∂gt/∂y|

    Args:
        scales: Number of scales for multi-scale gradient loss
        weight_decay: Weight decay per scale (0.5 means half weight at each scale)
    """

    def __init__(
        self,
        scales: int = 4,
        weight_decay: float = 0.5,
    ):
        _check_torch()
        super().__init__()
        self.scales = scales
        self.weight_decay = weight_decay

        # Sobel filters for gradient computation
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
        )

    def _compute_gradient(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Compute image gradients using Sobel filters.

        Args:
            x: Input tensor (B, 1, H, W)

        Returns:
            Tuple of (gradient_x, gradient_y)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        return grad_x, grad_y

    def forward(
        self,
        pred: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute multi-scale gradient loss.

        Args:
            pred: Predicted depth (B, 1, H, W) or (B, H, W)
            target: Ground truth depth (B, 1, H, W) or (B, H, W)
            mask: Optional valid pixel mask

        Returns:
            Scalar loss value
        """
        # Ensure 4D tensors
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        total_loss = 0.0
        weight = 1.0

        for scale in range(self.scales):
            # Compute gradients
            pred_grad_x, pred_grad_y = self._compute_gradient(pred)
            target_grad_x, target_grad_y = self._compute_gradient(target)

            # L1 gradient loss
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)

            total_loss += weight * (loss_x + loss_y)

            # Downsample for next scale
            if scale < self.scales - 1:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
                weight *= self.weight_decay

        return total_loss


class SSIMLoss(_BaseModule):
    """Structural Similarity (SSIM) Loss for depth estimation.

    SSIM compares local patterns of pixel intensities that have been
    normalized for luminance and contrast. It's particularly useful for
    preserving structural information in depth maps.

    Reference:
        Wang et al., "Image Quality Assessment: From Error Visibility to
        Structural Similarity", IEEE TIP 2004

    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of Gaussian window
        channels: Number of input channels
        reduction: Reduction method ('mean' or 'none')
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
        reduction: str = "mean",
    ):
        _check_torch()
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.reduction = reduction

        # Create Gaussian window
        window = self._create_gaussian_window(window_size, sigma)
        self.register_buffer("window", window.expand(channels, 1, window_size, window_size))

        # Constants for stability
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _create_gaussian_window(
        self,
        window_size: int,
        sigma: float,
    ) -> "torch.Tensor":
        """Create 2D Gaussian window.

        Args:
            window_size: Size of window
            sigma: Standard deviation

        Returns:
            Gaussian window tensor
        """
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2

        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()

        return gauss_2d.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        pred: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute SSIM loss.

        Args:
            pred: Predicted depth (B, 1, H, W) or (B, H, W)
            target: Ground truth depth (B, 1, H, W) or (B, H, W)
            mask: Optional valid pixel mask (unused, for API compatibility)

        Returns:
            1 - SSIM (so lower is better)
        """
        # Ensure 4D tensors
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Normalize to [0, 1] range for SSIM computation
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)

        # Compute local means
        mu_pred = F.conv2d(pred_norm, self.window, padding=self.window_size // 2, groups=self.channels)
        mu_target = F.conv2d(target_norm, self.window, padding=self.window_size // 2, groups=self.channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        # Compute local variances and covariance
        sigma_pred_sq = F.conv2d(
            pred_norm ** 2, self.window, padding=self.window_size // 2, groups=self.channels
        ) - mu_pred_sq
        sigma_target_sq = F.conv2d(
            target_norm ** 2, self.window, padding=self.window_size // 2, groups=self.channels
        ) - mu_target_sq
        sigma_pred_target = F.conv2d(
            pred_norm * target_norm, self.window, padding=self.window_size // 2, groups=self.channels
        ) - mu_pred_target

        # SSIM formula
        numerator = (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)
        denominator = (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2)
        ssim_map = numerator / (denominator + 1e-8)

        if self.reduction == "mean":
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map

    def _normalize(self, x: "torch.Tensor") -> "torch.Tensor":
        """Normalize tensor to [0, 1] range.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)


class L1Loss(_BaseModule):
    """Simple L1 Loss for depth estimation.

    Computes mean absolute error between prediction and target.
    Can be applied to log-depth or linear depth.

    Args:
        use_log: Whether to apply log transform before computing loss
        eps: Small value for numerical stability in log transform
    """

    def __init__(
        self,
        use_log: bool = False,
        eps: float = 1e-6,
    ):
        _check_torch()
        super().__init__()
        self.use_log = use_log
        self.eps = eps

    def forward(
        self,
        pred: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute L1 loss.

        Args:
            pred: Predicted depth
            target: Ground truth depth
            mask: Optional valid pixel mask

        Returns:
            Scalar loss value
        """
        if self.use_log:
            pred = torch.log(pred.clamp(min=self.eps))
            target = torch.log(target.clamp(min=self.eps))

        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        return F.l1_loss(pred, target)


class CombinedDepthLoss(_BaseModule):
    """Combined loss for depth estimation with configurable weights.

    Combines multiple loss functions with learned or fixed weights:
    - Scale-Invariant Loss (primary)
    - Gradient Loss (edge preservation)
    - SSIM Loss (structural similarity)
    - L1 Loss (baseline)

    Example:
        >>> loss_fn = CombinedDepthLoss(
        ...     weights={
        ...         "scale_invariant": 1.0,
        ...         "gradient": 0.5,
        ...         "ssim": 0.3,
        ...     }
        ... )
        >>> loss = loss_fn(pred_depth, gt_depth)

    Args:
        weights: Dictionary of loss weights
        use_mask: Whether to use validity mask
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        use_mask: bool = True,
    ):
        _check_torch()
        super().__init__()

        # Default weights
        self.weights = weights or {
            "scale_invariant": 1.0,
            "gradient": 0.5,
            "ssim": 0.3,
            "l1": 0.0,
        }
        self.use_mask = use_mask

        # Initialize loss functions
        self.losses = {}
        if TORCH_AVAILABLE:
            self.losses = nn.ModuleDict()

        if self.weights.get("scale_invariant", 0) > 0:
            self.losses["scale_invariant"] = ScaleInvariantLoss()

        if self.weights.get("gradient", 0) > 0:
            self.losses["gradient"] = GradientLoss()

        if self.weights.get("ssim", 0) > 0:
            self.losses["ssim"] = SSIMLoss()

        if self.weights.get("l1", 0) > 0:
            self.losses["l1"] = L1Loss()

        logger.info(f"Initialized CombinedDepthLoss with weights: {self.weights}")

    def forward(
        self,
        pred: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", dict]:
        """Compute combined loss.

        Args:
            pred: Predicted depth (B, 1, H, W) or (B, H, W)
            target: Ground truth depth (B, 1, H, W) or (B, H, W)
            mask: Optional valid pixel mask

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Create mask if needed
        if self.use_mask and mask is None:
            mask = target > 1e-6

        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 0)
            if weight > 0:
                loss_value = loss_fn(pred, target, mask)
                weighted_loss = weight * loss_value
                total_loss = total_loss + weighted_loss
                loss_dict[name] = loss_value.detach()

        loss_dict["total"] = total_loss.detach()

        return total_loss, loss_dict

    def get_weights(self) -> dict:
        """Get current loss weights.

        Returns:
            Dictionary of loss weights
        """
        return self.weights.copy()

    def set_weights(self, weights: dict) -> None:
        """Update loss weights.

        Args:
            weights: New weight dictionary
        """
        self.weights.update(weights)
        logger.info(f"Updated loss weights: {self.weights}")
