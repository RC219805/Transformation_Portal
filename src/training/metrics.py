#!/usr/bin/env python3
"""
Depth Estimation Metrics

Implements standard depth estimation evaluation metrics:
- Absolute Relative Error (Abs Rel)
- Squared Relative Error (Sq Rel)
- Root Mean Squared Error (RMSE)
- RMSE log
- Threshold accuracy (δ < 1.25, δ < 1.25², δ < 1.25³)

Reference:
    Eigen et al., "Depth Map Prediction from a Single Image using a
    Multi-Scale Deep Network", NeurIPS 2014

Author: Transformation Portal Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DepthMetrics:
    """Container for depth estimation metrics.

    Attributes:
        abs_rel: Absolute Relative Error
        sq_rel: Squared Relative Error
        rmse: Root Mean Squared Error
        rmse_log: RMSE in log space
        delta1: Threshold accuracy δ < 1.25
        delta2: Threshold accuracy δ < 1.25²
        delta3: Threshold accuracy δ < 1.25³
        num_samples: Number of samples evaluated
    """
    abs_rel: float = 0.0
    sq_rel: float = 0.0
    rmse: float = 0.0
    rmse_log: float = 0.0
    delta1: float = 0.0
    delta2: float = 0.0
    delta3: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary.

        Returns:
            Dictionary of metric values
        """
        return {
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "rmse": self.rmse,
            "rmse_log": self.rmse_log,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "delta3": self.delta3,
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"DepthMetrics(\n"
            f"  abs_rel={self.abs_rel:.4f},\n"
            f"  sq_rel={self.sq_rel:.4f},\n"
            f"  rmse={self.rmse:.4f},\n"
            f"  rmse_log={self.rmse_log:.4f},\n"
            f"  δ<1.25={self.delta1:.4f},\n"
            f"  δ<1.25²={self.delta2:.4f},\n"
            f"  δ<1.25³={self.delta3:.4f}\n"
            f")"
        )


class DepthMetricCalculator:
    """Calculator for depth estimation metrics.

    Computes standard metrics used to evaluate depth estimation models.
    Supports both numpy arrays and PyTorch tensors.

    Example:
        >>> calculator = DepthMetricCalculator()
        >>> for pred, target in dataloader:
        ...     calculator.update(pred, target)
        >>> metrics = calculator.compute()
        >>> print(metrics)
    """

    def __init__(
        self,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        use_median_scaling: bool = False,
    ):
        """Initialize metric calculator.

        Args:
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            use_median_scaling: Whether to apply median scaling alignment
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_median_scaling = use_median_scaling

        # Accumulators
        self._reset()

    def _reset(self) -> None:
        """Reset accumulators."""
        self.abs_rel_sum = 0.0
        self.sq_rel_sum = 0.0
        self.rmse_sum = 0.0
        self.rmse_log_sum = 0.0
        self.delta1_sum = 0.0
        self.delta2_sum = 0.0
        self.delta3_sum = 0.0
        self.num_pixels = 0
        self.num_samples = 0

    def update(
        self,
        pred: Union[np.ndarray, "torch.Tensor"],
        target: Union[np.ndarray, "torch.Tensor"],
        mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    ) -> None:
        """Update metrics with a batch of predictions.

        Args:
            pred: Predicted depth maps (B, H, W) or (B, 1, H, W)
            target: Ground truth depth maps (B, H, W) or (B, 1, H, W)
            mask: Optional validity mask
        """
        # Convert to numpy if needed
        if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if TORCH_AVAILABLE and isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        if mask is not None and TORCH_AVAILABLE and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # Squeeze channel dimension if present
        pred = np.squeeze(pred)
        target = np.squeeze(target)

        # Handle batch dimension
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]

        batch_size = pred.shape[0]

        for i in range(batch_size):
            p = pred[i]
            t = target[i]

            # Create validity mask
            if mask is not None:
                m = mask[i] if mask.ndim == 3 else mask
            else:
                m = (t > self.min_depth) & (t < self.max_depth)

            # Apply median scaling if requested
            if self.use_median_scaling:
                scale = np.median(t[m]) / (np.median(p[m]) + 1e-8)
                p = p * scale

            # Clamp predictions
            p = np.clip(p, self.min_depth, self.max_depth)

            # Extract valid pixels
            p_valid = p[m]
            t_valid = t[m]

            if len(p_valid) == 0:
                continue

            # Compute metrics
            self._update_metrics(p_valid, t_valid)
            self.num_samples += 1

    def _update_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Update metric accumulators.

        Args:
            pred: Flattened valid predictions
            target: Flattened valid ground truth
        """
        n = len(pred)

        # Absolute relative error
        abs_rel = np.abs(pred - target) / target
        self.abs_rel_sum += np.sum(abs_rel)

        # Squared relative error
        sq_rel = ((pred - target) ** 2) / target
        self.sq_rel_sum += np.sum(sq_rel)

        # RMSE
        rmse_sq = (pred - target) ** 2
        self.rmse_sum += np.sum(rmse_sq)

        # RMSE log
        log_pred = np.log(pred + 1e-8)
        log_target = np.log(target + 1e-8)
        rmse_log_sq = (log_pred - log_target) ** 2
        self.rmse_log_sum += np.sum(rmse_log_sq)

        # Threshold accuracy
        ratio = np.maximum(pred / target, target / pred)
        self.delta1_sum += np.sum(ratio < 1.25)
        self.delta2_sum += np.sum(ratio < 1.25 ** 2)
        self.delta3_sum += np.sum(ratio < 1.25 ** 3)

        self.num_pixels += n

    def compute(self) -> DepthMetrics:
        """Compute final metrics.

        Returns:
            DepthMetrics object with computed values
        """
        if self.num_pixels == 0:
            logger.warning("No valid pixels for metric computation")
            return DepthMetrics()

        n = self.num_pixels

        return DepthMetrics(
            abs_rel=self.abs_rel_sum / n,
            sq_rel=self.sq_rel_sum / n,
            rmse=np.sqrt(self.rmse_sum / n),
            rmse_log=np.sqrt(self.rmse_log_sum / n),
            delta1=self.delta1_sum / n,
            delta2=self.delta2_sum / n,
            delta3=self.delta3_sum / n,
            num_samples=self.num_samples,
        )

    def reset(self) -> None:
        """Reset all accumulators."""
        self._reset()


def compute_depth_metrics(
    pred: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    use_median_scaling: bool = False,
) -> DepthMetrics:
    """Compute depth estimation metrics.

    Convenience function for single-batch metric computation.

    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        mask: Optional validity mask
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        use_median_scaling: Whether to apply median scaling

    Returns:
        DepthMetrics object

    Example:
        >>> metrics = compute_depth_metrics(pred_depth, gt_depth)
        >>> print(f"RMSE: {metrics.rmse:.4f}")
    """
    calculator = DepthMetricCalculator(
        min_depth=min_depth,
        max_depth=max_depth,
        use_median_scaling=use_median_scaling,
    )
    calculator.update(pred, target, mask)
    return calculator.compute()


def visualize_depth_comparison(
    pred: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    image: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    cmap: str = "turbo",
) -> Optional["plt.Figure"]:
    """Visualize depth prediction comparison.

    Creates a figure with RGB image (optional), predicted depth,
    ground truth depth, and error map.

    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        image: Optional RGB image
        save_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap for depth visualization

    Returns:
        matplotlib Figure object (if not saving)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for visualization")
        return None

    # Convert to numpy if needed
    if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if image is not None and TORCH_AVAILABLE and isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # Squeeze and handle dimensions
    pred = np.squeeze(pred)
    target = np.squeeze(target)

    # Handle image dimensions (C, H, W) -> (H, W, C)
    if image is not None:
        image = np.squeeze(image)
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        if image.max() > 1.0:
            image = image / 255.0

    # Determine number of subplots
    n_plots = 4 if image is not None else 3

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    idx = 0

    # RGB image
    if image is not None:
        axes[idx].imshow(image)
        axes[idx].set_title("RGB Image")
        axes[idx].axis("off")
        idx += 1

    # Predicted depth
    vmin = min(pred.min(), target.min())
    vmax = max(pred.max(), target.max())

    im0 = axes[idx].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[idx].set_title("Predicted Depth")
    axes[idx].axis("off")
    plt.colorbar(im0, ax=axes[idx], fraction=0.046, pad=0.04)
    idx += 1

    # Ground truth depth
    im1 = axes[idx].imshow(target, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[idx].set_title("Ground Truth")
    axes[idx].axis("off")
    plt.colorbar(im1, ax=axes[idx], fraction=0.046, pad=0.04)
    idx += 1

    # Error map
    error = np.abs(pred - target)
    im2 = axes[idx].imshow(error, cmap="hot")
    axes[idx].set_title("Absolute Error")
    axes[idx].axis("off")
    plt.colorbar(im2, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved depth comparison to {save_path}")
        return None

    return fig


def log_metrics_to_tensorboard(
    writer: "torch.utils.tensorboard.SummaryWriter",
    metrics: DepthMetrics,
    step: int,
    prefix: str = "val",
) -> None:
    """Log metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        metrics: DepthMetrics object
        step: Global step
        prefix: Metric prefix (e.g., 'train', 'val')
    """
    for name, value in metrics.to_dict().items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def create_metric_table(
    metrics_list: List[DepthMetrics],
    names: Optional[List[str]] = None,
) -> str:
    """Create formatted table of metrics.

    Args:
        metrics_list: List of DepthMetrics objects
        names: Optional list of names for each row

    Returns:
        Formatted table string
    """
    if names is None:
        names = [f"Model {i+1}" for i in range(len(metrics_list))]

    # Header
    header = f"{'Model':<20} {'Abs Rel':>10} {'Sq Rel':>10} {'RMSE':>10} {'δ<1.25':>10} {'δ<1.25²':>10} {'δ<1.25³':>10}"
    separator = "-" * len(header)

    rows = [header, separator]

    for name, m in zip(names, metrics_list):
        row = (
            f"{name:<20} {m.abs_rel:>10.4f} {m.sq_rel:>10.4f} {m.rmse:>10.4f} "
            f"{m.delta1:>10.4f} {m.delta2:>10.4f} {m.delta3:>10.4f}"
        )
        rows.append(row)

    return "\n".join(rows)
