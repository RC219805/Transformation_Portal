"""Validation and quality metrics for depth estimation.

Provides depth quality metrics compatible with existing validation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from lux_depth_v3.inference import DepthResult


@dataclass
class DepthQualityMetrics:
    """Quality metrics for depth estimation."""

    # Standard depth metrics
    rmse: float = 0.0  # Root Mean Square Error
    mae: float = 0.0  # Mean Absolute Error
    absrel: float = 0.0  # Absolute Relative error
    sqrel: float = 0.0  # Squared Relative error

    # Threshold accuracies (δ < threshold)
    delta_1: float = 0.0  # δ < 1.25
    delta_2: float = 0.0  # δ < 1.25²
    delta_3: float = 0.0  # δ < 1.25³

    # Edge completeness
    edge_completeness: float = 0.0
    edge_accuracy: float = 0.0

    # Additional metrics
    valid_pixels: int = 0
    total_pixels: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def passes_quality_gate(
        self,
        min_delta_1: float = 0.8,
        max_rmse: float = 0.5,
    ) -> bool:
        """Check if metrics pass quality gate.

        Args:
            min_delta_1: Minimum δ < 1.25 threshold
            max_rmse: Maximum RMSE

        Returns:
            True if metrics pass
        """
        return self.delta_1 >= min_delta_1 and self.rmse <= max_rmse

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "absrel": float(self.absrel),
            "sqrel": float(self.sqrel),
            "delta_1": float(self.delta_1),
            "delta_2": float(self.delta_2),
            "delta_3": float(self.delta_3),
            "edge_completeness": float(self.edge_completeness),
            "edge_accuracy": float(self.edge_accuracy),
            "valid_pixels": int(self.valid_pixels),
            "total_pixels": int(self.total_pixels),
            **self.metadata,
        }


class DepthValidator:
    """Validator for depth estimation quality."""

    def __init__(
        self,
        ground_truth_dir: Optional[Path] = None,
    ):
        """Initialize validator.

        Args:
            ground_truth_dir: Directory with ground truth depth maps
        """
        self.ground_truth_dir = ground_truth_dir

    def validate(
        self,
        result: DepthResult,
        ground_truth: Optional[np.ndarray] = None,
    ) -> DepthQualityMetrics:
        """Validate depth estimation result.

        Args:
            result: Depth estimation result
            ground_truth: Ground truth depth map (H, W)

        Returns:
            Quality metrics
        """
        # Try to load ground truth if not provided
        if ground_truth is None and self.ground_truth_dir is not None:
            ground_truth = self._load_ground_truth(result)

        # If no ground truth, return empty metrics
        if ground_truth is None:
            return DepthQualityMetrics(metadata={"has_ground_truth": False})

        # Compute metrics
        metrics = self._compute_metrics(
            result.depth_map,
            ground_truth,
        )

        # Add metadata
        metrics.metadata.update(
            {
                "has_ground_truth": True,
                "model_variant": result.metadata.get("model_variant"),
                "inference_mode": result.metadata.get("inference_mode"),
            }
        )

        return metrics

    def _load_ground_truth(
        self,
        result: DepthResult,
    ) -> Optional[np.ndarray]:
        """Load ground truth depth map.

        Args:
            result: Depth result with input path

        Returns:
            Ground truth depth map or None
        """
        if self.ground_truth_dir is None:
            return None

        input_path = result.metadata.get("input_path")
        if input_path is None:
            return None

        # Construct ground truth path
        input_name = Path(input_path).stem
        gt_path = self.ground_truth_dir / f"{input_name}_depth.npy"

        if not gt_path.exists():
            gt_path = self.ground_truth_dir / f"{input_name}.npy"

        if not gt_path.exists():
            return None

        # Load ground truth
        try:
            ground_truth = np.load(gt_path)
            return ground_truth
        except Exception:
            return None

    def _compute_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> DepthQualityMetrics:
        """Compute depth quality metrics.

        Args:
            pred: Predicted depth map (H, W)
            gt: Ground truth depth map (H, W)

        Returns:
            Quality metrics
        """
        # Ensure same shape
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom

            scale_y = gt.shape[0] / pred.shape[0]
            scale_x = gt.shape[1] / pred.shape[1]
            pred = zoom(pred, (scale_y, scale_x), order=1)

        # Create valid mask (non-zero ground truth)
        valid_mask = gt > 0

        if not valid_mask.any():
            return DepthQualityMetrics()

        # Extract valid pixels
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        # Compute errors
        abs_diff = np.abs(pred_valid - gt_valid)
        sq_diff = (pred_valid - gt_valid) ** 2

        # RMSE and MAE
        rmse = np.sqrt(np.mean(sq_diff))
        mae = np.mean(abs_diff)

        # Relative errors
        abs_rel = np.mean(abs_diff / (gt_valid + 1e-8))
        sq_rel = np.mean(sq_diff / (gt_valid + 1e-8))

        # Threshold accuracies
        threshold = np.maximum(pred_valid / (gt_valid + 1e-8), gt_valid / (pred_valid + 1e-8))

        delta_1 = np.mean(threshold < 1.25)
        delta_2 = np.mean(threshold < 1.25**2)
        delta_3 = np.mean(threshold < 1.25**3)

        # Edge completeness (simplified)
        edge_completeness, edge_accuracy = self._compute_edge_metrics(
            pred,
            gt,
            valid_mask,
        )

        return DepthQualityMetrics(
            rmse=float(rmse),
            mae=float(mae),
            absrel=float(abs_rel),
            sqrel=float(sq_rel),
            delta_1=float(delta_1),
            delta_2=float(delta_2),
            delta_3=float(delta_3),
            edge_completeness=float(edge_completeness),
            edge_accuracy=float(edge_accuracy),
            valid_pixels=int(valid_mask.sum()),
            total_pixels=int(valid_mask.size),
        )

    def _compute_edge_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute edge completeness and accuracy.

        Args:
            pred: Predicted depth map
            gt: Ground truth depth map
            valid_mask: Valid pixel mask

        Returns:
            Tuple of (completeness, accuracy)
        """
        from scipy.ndimage import sobel

        # Detect edges in ground truth
        gt_edge_x = sobel(gt, axis=0)
        gt_edge_y = sobel(gt, axis=1)
        gt_edge = np.sqrt(gt_edge_x**2 + gt_edge_y**2)
        gt_edge_mask = (gt_edge > np.percentile(gt_edge, 90)) & valid_mask

        # Detect edges in prediction
        pred_edge_x = sobel(pred, axis=0)
        pred_edge_y = sobel(pred, axis=1)
        pred_edge = np.sqrt(pred_edge_x**2 + pred_edge_y**2)
        pred_edge_mask = (pred_edge > np.percentile(pred_edge, 90)) & valid_mask

        # Compute metrics
        if gt_edge_mask.sum() == 0:
            return 0.0, 0.0

        # Completeness: how many GT edges are detected
        completeness = np.sum(gt_edge_mask & pred_edge_mask) / np.sum(gt_edge_mask)

        # Accuracy: how many predicted edges are correct
        if pred_edge_mask.sum() == 0:
            accuracy = 0.0
        else:
            accuracy = np.sum(gt_edge_mask & pred_edge_mask) / np.sum(pred_edge_mask)

        return completeness, accuracy


class ValidationReport:
    """Generate validation reports."""

    def __init__(self):
        """Initialize validation report."""
        self.metrics: List[DepthQualityMetrics] = []

    def add_result(self, metrics: DepthQualityMetrics):
        """Add metrics to report."""
        self.metrics.append(metrics)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics.

        Returns:
            Summary dictionary
        """
        if not self.metrics:
            return {}

        # Aggregate metrics
        summary = {
            "num_images": len(self.metrics),
            "mean_rmse": np.mean([m.rmse for m in self.metrics]),
            "mean_mae": np.mean([m.mae for m in self.metrics]),
            "mean_delta_1": np.mean([m.delta_1 for m in self.metrics]),
            "mean_delta_2": np.mean([m.delta_2 for m in self.metrics]),
            "mean_delta_3": np.mean([m.delta_3 for m in self.metrics]),
            "mean_edge_completeness": np.mean([m.edge_completeness for m in self.metrics]),
        }

        return summary

    def save(self, output_path: Path):
        """Save validation report to JSON.

        Args:
            output_path: Output file path
        """
        import json

        report = {
            "summary": self.compute_summary(),
            "metrics": [m.to_dict() for m in self.metrics],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Validation report saved to: {output_path}")
