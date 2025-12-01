"""
Perceptual Analyzer for Image Quality Assessment

Performs comprehensive perceptual analysis of images using multiple metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import time

import torch
from torch import Tensor
import numpy as np

from .metrics import QualityMetrics, PerceptualScore, MetricType
from .image_loader import ImageMetadata

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of perceptual analysis."""
    image_path: Path
    image_metadata: ImageMetadata
    quality_scores: Dict[MetricType, PerceptualScore]
    overall_quality: float  # Weighted average of normalized scores
    analysis_time: float
    timestamp: float

    # Derived metrics
    sharpness: float = 0.0
    contrast: float = 0.0
    colorfulness: float = 0.0
    naturalness: float = 0.0

    # Comparison to reference (if available)
    comparison_scores: Dict[MetricType, PerceptualScore] = field(default_factory=dict)

    def get_score(self, metric_type: MetricType) -> Optional[PerceptualScore]:
        """Get score for specific metric type."""
        return self.quality_scores.get(metric_type)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of analysis."""
        return {
            "path": str(self.image_path),
            "image_type": self.image_metadata.image_type.value if self.image_metadata.image_type else None,
            "dimensions": f"{self.image_metadata.width}x{self.image_metadata.height}",
            "overall_quality": round(self.overall_quality, 3),
            "sharpness": round(self.sharpness, 3),
            "contrast": round(self.contrast, 3),
            "colorfulness": round(self.colorfulness, 3),
            "scores": {
                metric.value: round(score.normalized_score, 3)
                for metric, score in self.quality_scores.items()
            }
        }


class PerceptualAnalyzer:
    """
    Perceptual analyzer for comprehensive image quality assessment.

    Combines multiple quality metrics and perceptual features to
    establish baseline quality measurements.
    """

    def __init__(
        self,
        substrate,
        metric_weights: Optional[Dict[MetricType, float]] = None
    ):
        """
        Initialize perceptual analyzer.

        Args:
            substrate: Computational substrate
            metric_weights: Weights for combining metrics (defaults to equal)
        """
        self.substrate = substrate
        self.quality_metrics = QualityMetrics(substrate)

        # Default metric weights for overall quality
        self.metric_weights = metric_weights or {
            MetricType.BRISQUE: 0.25,
            MetricType.NIQE: 0.25,
            MetricType.PSNR: 0.20,
            MetricType.SSIM: 0.15,
            MetricType.LPIPS: 0.15,
        }

        logger.info("Initialized PerceptualAnalyzer")

    def analyze(
        self,
        image: Tensor,
        metadata: ImageMetadata,
        reference: Optional[Tensor] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive perceptual analysis.

        Args:
            image: Input image tensor
            metadata: Image metadata
            reference: Optional reference image for comparison

        Returns:
            Analysis result with all metrics
        """
        start_time = time.time()

        logger.info(f"Analyzing image: {metadata.path.name}")

        # Compute quality metrics
        quality_scores = self.quality_metrics.compute_all(image, reference)

        # Compute derived metrics
        sharpness = self._compute_sharpness(image)
        contrast = self._compute_contrast(image)
        colorfulness = self._compute_colorfulness(image)
        naturalness = self._compute_naturalness(image)

        # Compute overall quality (weighted average of normalized scores)
        overall_quality = self._compute_overall_quality(quality_scores)

        # Comparison scores if reference provided
        comparison_scores = {}
        if reference is not None:
            comparison_scores = {
                k: v for k, v in quality_scores.items()
                if k in [MetricType.LPIPS, MetricType.PSNR, MetricType.SSIM, MetricType.MSE]
            }

        analysis_time = time.time() - start_time

        result = AnalysisResult(
            image_path=metadata.path,
            image_metadata=metadata,
            quality_scores=quality_scores,
            overall_quality=overall_quality,
            analysis_time=analysis_time,
            timestamp=time.time(),
            sharpness=sharpness,
            contrast=contrast,
            colorfulness=colorfulness,
            naturalness=naturalness,
            comparison_scores=comparison_scores,
        )

        logger.info(
            f"Analysis complete: overall_quality={overall_quality:.3f}, "
            f"time={analysis_time:.2f}s"
        )

        return result

    def analyze_batch(
        self,
        images: List[Tensor],
        metadatas: List[ImageMetadata],
        references: Optional[List[Tensor]] = None
    ) -> List[AnalysisResult]:
        """
        Analyze batch of images.

        Args:
            images: List of image tensors
            metadatas: List of image metadatas
            references: Optional list of reference images

        Returns:
            List of analysis results
        """
        if references is None:
            references = [None] * len(images)

        results = []
        for image, metadata, reference in zip(images, metadatas, references):
            result = self.analyze(image, metadata, reference)
            results.append(result)

        return results

    def compare(
        self,
        image1: Tensor,
        image2: Tensor,
        metadata1: ImageMetadata,
        metadata2: ImageMetadata
    ) -> Dict[str, Any]:
        """
        Compare two images and determine which is better.

        Args:
            image1: First image
            image2: Second image
            metadata1: Metadata for first image
            metadata2: Metadata for second image

        Returns:
            Comparison result with winner and scores
        """
        # Analyze both images
        result1 = self.analyze(image1, metadata1)
        result2 = self.analyze(image2, metadata2)

        # Compare overall quality
        winner = 1 if result1.overall_quality > result2.overall_quality else 2
        quality_diff = abs(result1.overall_quality - result2.overall_quality)

        # Metric-by-metric comparison
        metric_comparisons = {}
        for metric_type in MetricType:
            score1 = result1.get_score(metric_type)
            score2 = result2.get_score(metric_type)

            if score1 and score2:
                better = 1 if score1.is_better_than(score2) else 2
                metric_comparisons[metric_type.value] = {
                    "winner": better,
                    "score1": score1.score,
                    "score2": score2.score,
                    "difference": abs(score1.score - score2.score),
                }

        return {
            "winner": winner,
            "quality_difference": quality_diff,
            "image1_quality": result1.overall_quality,
            "image2_quality": result2.overall_quality,
            "metric_comparisons": metric_comparisons,
            "summary": f"Image {winner} is better (quality diff: {quality_diff:.3f})"
        }

    def _compute_overall_quality(
        self,
        quality_scores: Dict[MetricType, PerceptualScore]
    ) -> float:
        """Compute weighted average overall quality."""
        total_weight = 0.0
        weighted_sum = 0.0

        for metric_type, weight in self.metric_weights.items():
            if metric_type in quality_scores:
                score = quality_scores[metric_type]
                weighted_sum += score.normalized_score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _compute_sharpness(self, image: Tensor) -> float:
        """Compute image sharpness using Laplacian variance."""
        if image.ndim == 4:
            image = image[0]

        # Laplacian kernel
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32,
            device=image.device
        ).view(1, 1, 3, 3)

        # Apply to each channel and average
        sharpness_per_channel = []
        for c in range(image.shape[0]):
            channel = image[c:c + 1].unsqueeze(0)
            laplacian_response = torch.nn.functional.conv2d(channel, laplacian, padding=1)
            variance = laplacian_response.var().item()
            sharpness_per_channel.append(variance)

        return np.mean(sharpness_per_channel)

    def _compute_contrast(self, image: Tensor) -> float:
        """Compute image contrast (RMS contrast)."""
        if image.ndim == 4:
            image = image[0]

        # RMS contrast
        mean = image.mean()
        contrast = torch.sqrt(((image - mean) ** 2).mean()).item()

        return contrast

    def _compute_colorfulness(self, image: Tensor) -> float:
        """
        Compute colorfulness metric.

        Based on: "Measuring colourfulness in natural images" by Hasler and Süsstrunk.
        """
        if image.ndim == 4:
            image = image[0]

        if image.shape[0] != 3:
            return 0.0  # Grayscale

        # Move to CPU for numpy operations
        image_np = image.cpu().numpy()

        r, g, b = image_np[0], image_np[1], image_np[2]

        # Compute rg and yb
        rg = r - g
        yb = 0.5 * (r + g) - b

        # Compute statistics
        rg_std = np.std(rg)
        yb_std = np.std(yb)
        rg_mean = np.mean(rg)
        yb_mean = np.mean(yb)

        # Colorfulness metric
        colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)

        return float(colorfulness)

    def _compute_naturalness(self, image: Tensor) -> float:
        """Compute naturalness score based on color and intensity distributions."""
        if image.ndim == 4:
            image = image[0]

        # Compute histogram entropy (natural images have high entropy)
        image_flat = image.flatten()
        hist = torch.histc(image_flat, bins=256, min=0, max=1)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        entropy = -(hist * torch.log2(hist)).sum().item()

        # Normalize entropy (max is log2(256) = 8)
        normalized_entropy = entropy / 8.0

        # Check color balance (natural images have balanced colors)
        if image.shape[0] == 3:
            channel_means = image.mean(dim=(1, 2))
            color_balance = 1.0 - channel_means.std().item()
        else:
            color_balance = 1.0

        # Combine metrics
        naturalness = (normalized_entropy + color_balance) / 2.0

        return naturalness

    def generate_report(
        self,
        results: List[AnalysisResult],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable analysis report.

        Args:
            results: List of analysis results
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        lines = [
            "=" * 80,
            "PERCEPTUAL QUALITY ANALYSIS REPORT",
            "=" * 80,
            f"Total Images Analyzed: {len(results)}",
            f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Summary statistics
        avg_quality = np.mean([r.overall_quality for r in results])
        avg_sharpness = np.mean([r.sharpness for r in results])
        avg_contrast = np.mean([r.contrast for r in results])
        avg_colorfulness = np.mean([r.colorfulness for r in results])

        lines.extend([
            "OVERALL STATISTICS",
            "-" * 80,
            f"Average Overall Quality: {avg_quality:.3f}",
            f"Average Sharpness: {avg_sharpness:.3f}",
            f"Average Contrast: {avg_contrast:.3f}",
            f"Average Colorfulness: {avg_colorfulness:.3f}",
            "",
        ])

        # Individual results
        lines.append("INDIVIDUAL RESULTS")
        lines.append("-" * 80)

        for i, result in enumerate(results, 1):
            summary = result.get_summary()
            lines.extend([
                f"\n{i}. {summary['path']}",
                f"   Type: {summary['image_type'] or 'Unknown'}",
                f"   Dimensions: {summary['dimensions']}",
                f"   Overall Quality: {summary['overall_quality']}",
                f"   Sharpness: {result.sharpness:.3f}",
                f"   Contrast: {result.contrast:.3f}",
                f"   Colorfulness: {result.colorfulness:.3f}",
                "   Metric Scores:",
            ])

            for metric_name, score in summary['scores'].items():
                lines.append(f"     {metric_name}: {score}")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report
