"""Full benchmark suite for image quality evaluation.

This module provides a comprehensive benchmark suite combining:
- PSNR (reconstruction fidelity)
- SSIM (structural similarity)
- LPIPS (perceptual similarity)
- IoU (segmentation accuracy)
- LLaVA (semantic quality assessment)

Example:
    >>> suite = BenchmarkSuite(llava_backend=llava_backend)
    >>> result = suite.run(
    ...     prediction=pred_img,
    ...     ground_truth=gt_img,
    ...     pred_mask=pred_mask,
    ...     gt_mask=gt_mask,
    ... )
    >>> print(f"PSNR: {result.psnr:.2f}, IoU: {result.iou:.3f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from transformation_portal.evals.metrics import (
    ImageLike,
    dice_coefficient,
    lpips_score,
    lpips_to_score,
    psnr,
    psnr_to_score,
    segmentation_iou,
    ssim,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from benchmark suite evaluation.

    Attributes:
        psnr: Peak Signal-to-Noise Ratio (dB)
        ssim: Structural Similarity Index
        lpips: LPIPS perceptual distance
        iou: Segmentation IoU (if masks provided)
        dice: Dice coefficient (if masks provided)
        llava_score: LLaVA quality score (if backend provided)
        llava_issues: Issues detected by LLaVA
        combined_score: Weighted combined score [0, 1]
        metadata: Additional benchmark metadata
    """

    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    iou: float = 0.0
    dice: float = 0.0
    llava_score: float = 0.0
    llava_issues: list[dict[str, Any]] = field(default_factory=list)
    combined_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "iou": self.iou,
            "dice": self.dice,
            "llava_score": self.llava_score,
            "llava_issues": self.llava_issues,
            "combined_score": self.combined_score,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkWeights:
    """Weights for combining metrics into final score.

    All weights should sum to 1.0.
    """

    psnr: float = 0.2
    ssim: float = 0.15
    lpips: float = 0.15
    iou: float = 0.2
    llava: float = 0.3


class BenchmarkSuite:
    """Comprehensive benchmark suite for image quality evaluation.

    Combines traditional metrics (PSNR, SSIM, LPIPS, IoU) with
    VLM-based quality assessment (LLaVA).

    Example:
        >>> from transformation_portal.evals.vision_language import LlavaQualityBackend
        >>>
        >>> suite = BenchmarkSuite(
        ...     llava_backend=llava_backend,
        ...     weights=BenchmarkWeights(psnr=0.3, llava=0.2),
        ... )
        >>>
        >>> result = suite.run(
        ...     prediction=output_image,
        ...     ground_truth=reference_image,
        ... )
        >>> print(f"Combined score: {result.combined_score:.3f}")
    """

    def __init__(
        self,
        *,
        llava_backend: Optional[Any] = None,
        weights: Optional[BenchmarkWeights] = None,
    ) -> None:
        """Initialize benchmark suite.

        Args:
            llava_backend: Optional LLaVA quality backend
            weights: Metric weights for combined score
        """
        self.llava_backend = llava_backend
        self.weights = weights or BenchmarkWeights()

    def run(
        self,
        *,
        prediction: ImageLike,
        ground_truth: Optional[ImageLike] = None,
        pred_mask: Optional[ImageLike] = None,
        gt_mask: Optional[ImageLike] = None,
        prediction_path: Optional[Path] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run full benchmark suite.

        Args:
            prediction: Predicted/generated image
            ground_truth: Ground truth image (for PSNR/SSIM/LPIPS)
            pred_mask: Predicted segmentation mask
            gt_mask: Ground truth segmentation mask
            prediction_path: Path to prediction image (for LLaVA)
            context: Context for LLaVA evaluation

        Returns:
            BenchmarkResult with all metrics
        """
        result = BenchmarkResult(
            metadata={
                "has_ground_truth": ground_truth is not None,
                "has_masks": pred_mask is not None and gt_mask is not None,
                "has_llava": self.llava_backend is not None,
            }
        )

        # Image quality metrics (require ground truth)
        if ground_truth is not None:
            result.psnr = self._compute_psnr(prediction, ground_truth)
            result.ssim = self._compute_ssim(prediction, ground_truth)
            result.lpips = self._compute_lpips(prediction, ground_truth)

        # Segmentation metrics (require masks)
        if pred_mask is not None and gt_mask is not None:
            result.iou = self._compute_iou(pred_mask, gt_mask)
            result.dice = self._compute_dice(pred_mask, gt_mask)

        # LLaVA quality assessment
        if self.llava_backend is not None and prediction_path is not None:
            llava_result = self._run_llava(prediction_path, context)
            result.llava_score = llava_result.get("score", 0.0)
            result.llava_issues = llava_result.get("issues", [])

        # Compute combined score
        result.combined_score = self._compute_combined_score(result)

        return result

    def _compute_psnr(
        self,
        pred: ImageLike,
        gt: ImageLike,
    ) -> float:
        """Compute PSNR metric."""
        try:
            return psnr(pred, gt)
        except Exception as exc:
            logger.warning("PSNR computation failed: %s", exc)
            return 0.0

    def _compute_ssim(
        self,
        pred: ImageLike,
        gt: ImageLike,
    ) -> float:
        """Compute SSIM metric."""
        try:
            return ssim(pred, gt)
        except Exception as exc:
            logger.warning("SSIM computation failed: %s", exc)
            return 0.0

    def _compute_lpips(
        self,
        pred: ImageLike,
        gt: ImageLike,
    ) -> float:
        """Compute LPIPS metric."""
        try:
            return lpips_score(pred, gt)
        except Exception as exc:
            logger.warning("LPIPS computation failed: %s", exc)
            return 0.0

    def _compute_iou(
        self,
        pred: ImageLike,
        gt: ImageLike,
    ) -> float:
        """Compute IoU metric."""
        try:
            return segmentation_iou(pred, gt)
        except Exception as exc:
            logger.warning("IoU computation failed: %s", exc)
            return 0.0

    def _compute_dice(
        self,
        pred: ImageLike,
        gt: ImageLike,
    ) -> float:
        """Compute Dice coefficient."""
        try:
            return dice_coefficient(pred, gt)
        except Exception as exc:
            logger.warning("Dice computation failed: %s", exc)
            return 0.0

    def _run_llava(
        self,
        image_path: Path,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run LLaVA quality assessment."""
        try:
            result = self.llava_backend.evaluate_images(
                image_paths=[image_path],
                context=context,
            )
            return {
                "score": result.summary_score,
                "issues": [
                    {
                        "issue_type": i.issue_type,
                        "severity": i.severity,
                        "evidence": i.evidence,
                    }
                    for i in result.issues
                ],
            }
        except Exception as exc:
            logger.warning("LLaVA evaluation failed: %s", exc)
            return {"score": 0.0, "issues": []}

    def _compute_combined_score(
        self,
        result: BenchmarkResult,
    ) -> float:
        """Compute weighted combined score.

        Normalizes all metrics to [0, 1] and applies weights.
        """
        scores = []
        total_weight = 0.0

        # PSNR (higher = better)
        if result.metadata.get("has_ground_truth"):
            psnr_score = psnr_to_score(result.psnr)
            scores.append(psnr_score * self.weights.psnr)
            total_weight += self.weights.psnr

            # SSIM (already [0, 1], higher = better)
            scores.append(max(0.0, result.ssim) * self.weights.ssim)
            total_weight += self.weights.ssim

            # LPIPS (lower = better, convert to [0, 1] score)
            lpips_norm = lpips_to_score(result.lpips)
            scores.append(lpips_norm * self.weights.lpips)
            total_weight += self.weights.lpips

        # IoU (already [0, 1])
        if result.metadata.get("has_masks"):
            scores.append(result.iou * self.weights.iou)
            total_weight += self.weights.iou

        # LLaVA (already [0, 1])
        if result.metadata.get("has_llava"):
            scores.append(result.llava_score * self.weights.llava)
            total_weight += self.weights.llava

        if total_weight == 0:
            return 0.0

        # Normalize by actual weights used
        return sum(scores) / total_weight


def run_benchmark_batch(
    suite: BenchmarkSuite,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run benchmark suite on a batch of samples.

    Args:
        suite: BenchmarkSuite instance
        samples: List of sample dicts with keys matching suite.run() args

    Returns:
        Aggregated results
    """
    results = []

    for sample in samples:
        result = suite.run(**sample)
        results.append(result)

    # Aggregate
    n = len(results)
    if n == 0:
        return {"count": 0}

    return {
        "count": n,
        "mean_psnr": sum(r.psnr for r in results) / n,
        "mean_ssim": sum(r.ssim for r in results) / n,
        "mean_lpips": sum(r.lpips for r in results) / n,
        "mean_iou": sum(r.iou for r in results) / n,
        "mean_llava": sum(r.llava_score for r in results) / n,
        "mean_combined": sum(r.combined_score for r in results) / n,
    }
