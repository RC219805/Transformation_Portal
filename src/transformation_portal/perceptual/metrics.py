"""
Quality Metrics for Perceptual Assessment

Implements various perceptual and statistical quality metrics:
- LPIPS: Learned Perceptual Image Patch Similarity
- FID: Fréchet Inception Distance
- BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
- NIQE: Natural Image Quality Evaluator
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import logging

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of quality metrics."""
    LPIPS = "lpips"  # Perceptual similarity (lower is better)
    FID = "fid"  # Distribution distance (lower is better)
    BRISQUE = "brisque"  # No-reference quality (lower is better)
    NIQE = "niqe"  # Natural image quality (lower is better)
    PSNR = "psnr"  # Peak SNR (higher is better)
    SSIM = "ssim"  # Structural similarity (higher is better)
    MSE = "mse"  # Mean squared error (lower is better)


@dataclass
class PerceptualScore:
    """Result of perceptual quality assessment."""
    metric_type: MetricType
    score: float
    higher_is_better: bool
    normalized_score: float  # Normalized to [0, 1] where 1 is best
    metadata: Dict[str, Any]

    def is_better_than(self, other: "PerceptualScore") -> bool:
        """Check if this score is better than another."""
        if self.metric_type != other.metric_type:
            raise ValueError("Cannot compare different metric types")

        if self.higher_is_better:
            return self.score > other.score
        else:
            return self.score < other.score


class QualityMetrics:
    """
    Quality metrics calculator for perceptual assessment.

    Computes various perceptual and statistical quality metrics
    with caching and batch processing support.
    """

    def __init__(self, substrate, cache_models: bool = True):
        """
        Initialize quality metrics calculator.

        Args:
            substrate: Computational substrate
            cache_models: Whether to cache loaded models
        """
        self.substrate = substrate
        self.cache_models = cache_models
        self.device = substrate.get_device()

        # Model cache
        self._lpips_model = None
        self._inception_model = None

        logger.info("Initialized QualityMetrics")

    def compute_all(
        self,
        image: Tensor,
        reference: Optional[Tensor] = None
    ) -> Dict[MetricType, PerceptualScore]:
        """
        Compute all available metrics.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            reference: Reference image for full-reference metrics

        Returns:
            Dictionary of metric scores
        """
        scores = {}

        # No-reference metrics (don't need reference)
        scores[MetricType.BRISQUE] = self.compute_brisque(image)
        scores[MetricType.NIQE] = self.compute_niqe(image)

        # Full-reference metrics (need reference)
        if reference is not None:
            scores[MetricType.LPIPS] = self.compute_lpips(image, reference)
            scores[MetricType.PSNR] = self.compute_psnr(image, reference)
            scores[MetricType.SSIM] = self.compute_ssim(image, reference)
            scores[MetricType.MSE] = self.compute_mse(image, reference)

        return scores

    def compute_lpips(
        self,
        image: Tensor,
        reference: Tensor,
        network: str = "alex"
    ) -> PerceptualScore:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity).

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            reference: Reference image
            network: Network to use ('alex', 'vgg', 'squeeze')

        Returns:
            LPIPS score (lower is better, 0 = identical)
        """
        # Ensure 4D tensors
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)

        # Lazy load LPIPS model
        if self._lpips_model is None or not self.cache_models:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net=network).to(self.device)
                self._lpips_model.eval()
            except ImportError:
                logger.warning("lpips package not installed, using MSE fallback")
                return self.compute_mse(image, reference)

        # Compute LPIPS
        with torch.no_grad():
            # LPIPS expects images in [-1, 1]
            img_norm = image * 2.0 - 1.0
            ref_norm = reference * 2.0 - 1.0

            distance = self._lpips_model(img_norm, ref_norm)
            score = distance.mean().item()

        # Normalize (LPIPS typically ranges [0, 1], already normalized)
        normalized = 1.0 - min(score, 1.0)

        return PerceptualScore(
            metric_type=MetricType.LPIPS,
            score=score,
            higher_is_better=False,
            normalized_score=normalized,
            metadata={"network": network}
        )

    def compute_psnr(
        self,
        image: Tensor,
        reference: Tensor,
        max_value: float = 1.0
    ) -> PerceptualScore:
        """
        Compute PSNR (Peak Signal-to-Noise Ratio).

        Args:
            image: Input image
            reference: Reference image
            max_value: Maximum pixel value (1.0 for normalized images)

        Returns:
            PSNR score (higher is better, typically 20-50 dB)
        """
        mse = F.mse_loss(image, reference).item()

        if mse == 0:
            psnr = float('inf')
            normalized = 1.0
        else:
            psnr = 20 * np.log10(max_value / np.sqrt(mse))
            # Normalize to [0, 1] assuming typical range [20, 50]
            normalized = min(max(psnr - 20, 0) / 30, 1.0)

        return PerceptualScore(
            metric_type=MetricType.PSNR,
            score=psnr,
            higher_is_better=True,
            normalized_score=normalized,
            metadata={"mse": mse}
        )

    def compute_ssim(
        self,
        image: Tensor,
        reference: Tensor,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03
    ) -> PerceptualScore:
        """
        Compute SSIM (Structural Similarity Index).

        Args:
            image: Input image
            reference: Reference image
            window_size: Size of Gaussian window
            k1, k2: SSIM constants

        Returns:
            SSIM score (higher is better, range [0, 1])
        """
        # Ensure 4D
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)

        # Create Gaussian window
        window = self._create_window(window_size, image.shape[1])
        window = window.to(image.device)

        # Compute SSIM
        ssim_val = self._ssim(
            image, reference, window, window_size,
            k1=k1, k2=k2
        )

        score = ssim_val.item()

        return PerceptualScore(
            metric_type=MetricType.SSIM,
            score=score,
            higher_is_better=True,
            normalized_score=score,  # Already in [0, 1]
            metadata={"window_size": window_size}
        )

    def compute_mse(
        self,
        image: Tensor,
        reference: Tensor
    ) -> PerceptualScore:
        """
        Compute MSE (Mean Squared Error).

        Args:
            image: Input image
            reference: Reference image

        Returns:
            MSE score (lower is better)
        """
        mse = F.mse_loss(image, reference).item()

        # Normalize (assuming max MSE = 1.0 for [0,1] images)
        normalized = 1.0 - min(mse, 1.0)

        return PerceptualScore(
            metric_type=MetricType.MSE,
            score=mse,
            higher_is_better=False,
            normalized_score=normalized,
            metadata={}
        )

    def compute_brisque(self, image: Tensor) -> PerceptualScore:
        """
        Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).

        No-reference quality metric based on natural scene statistics.

        Args:
            image: Input image

        Returns:
            BRISQUE score (lower is better, typically 0-100)
        """
        try:
            import cv2

            # Convert to numpy
            if image.ndim == 4:
                image = image[0]  # Take first image

            img_np = image.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # Compute BRISQUE (requires opencv-contrib-python)
            score = cv2.quality.QualityBRISQUE_compute(gray, "")[0]

            # Normalize (BRISQUE typically [0, 100], lower is better)
            normalized = 1.0 - min(score / 100.0, 1.0)

        except (ImportError, AttributeError):
            logger.warning("BRISQUE not available, using simplified version")
            # Fallback: use simple variance-based quality
            score = self._simple_sharpness_metric(image)
            normalized = score

        return PerceptualScore(
            metric_type=MetricType.BRISQUE,
            score=score,
            higher_is_better=False,
            normalized_score=normalized,
            metadata={}
        )

    def compute_niqe(self, image: Tensor) -> PerceptualScore:
        """
        Compute NIQE (Natural Image Quality Evaluator).

        No-reference quality metric based on natural scene statistics.

        Args:
            image: Input image

        Returns:
            NIQE score (lower is better)
        """
        # Simplified NIQE implementation
        # Full NIQE requires training on pristine images
        score = self._simple_naturalness_metric(image)

        # Normalize (lower is better, typical range [0, 10])
        normalized = 1.0 - min(score / 10.0, 1.0)

        return PerceptualScore(
            metric_type=MetricType.NIQE,
            score=score,
            higher_is_better=False,
            normalized_score=normalized,
            metadata={"simplified": True}
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _create_window(self, window_size: int, n_channels: int) -> Tensor:
        """Create Gaussian window for SSIM."""
        # Create 1D Gaussian
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()

        # Create 2D window
        window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window_2d.unsqueeze(0).unsqueeze(0)

        # Expand for all channels
        window = window.expand(n_channels, 1, window_size, window_size).contiguous()

        return window

    def _ssim(
        self,
        img1: Tensor,
        img2: Tensor,
        window: Tensor,
        window_size: int,
        k1: float = 0.01,
        k2: float = 0.03
    ) -> Tensor:
        """Compute SSIM between two images."""
        L = 1.0  # Dynamic range

        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2

        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2

        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return ssim_map.mean()

    def _simple_sharpness_metric(self, image: Tensor) -> float:
        """Simple sharpness metric based on gradient magnitude."""
        if image.ndim == 4:
            image = image[0]

        # Compute gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device)
        sobel_y = sobel_x.t()

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(image.shape[0], 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(image.shape[0], 1, 1, 1)

        # Apply to each channel
        grad_x = F.conv2d(image.unsqueeze(0), sobel_x, padding=1, groups=image.shape[0])
        grad_y = F.conv2d(image.unsqueeze(0), sobel_y, padding=1, groups=image.shape[0])

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Average sharpness
        sharpness = grad_mag.mean().item()

        return sharpness

    def _simple_naturalness_metric(self, image: Tensor) -> float:
        """Simple naturalness metric based on color distribution."""
        if image.ndim == 4:
            image = image[0]

        # Compute color distribution statistics
        mean = image.mean(dim=(1, 2))
        std = image.std(dim=(1, 2))

        # Natural images typically have balanced colors
        # Measure deviation from ideal distribution
        ideal_mean = torch.tensor([0.5, 0.5, 0.5], device=image.device)
        mean_dev = (mean - ideal_mean).abs().mean().item()

        # Natural images have moderate variance
        ideal_std = 0.2
        std_dev = (std.mean() - ideal_std).abs().item()

        # Combine deviations
        naturalness = (mean_dev + std_dev) * 5  # Scale to [0, 10] range

        return naturalness


# Convenience functions for direct metric computation

def compute_lpips(image: Tensor, reference: Tensor, device: torch.device) -> float:
    """Compute LPIPS score."""
    substrate = type('obj', (object,), {
        'get_device': lambda: device,
        'to_device': lambda x: x.to(device)
    })()
    metrics = QualityMetrics(substrate, cache_models=True)
    score = metrics.compute_lpips(image, reference)
    return score.score


def compute_psnr(image: Tensor, reference: Tensor) -> float:
    """Compute PSNR score."""
    mse = F.mse_loss(image, reference).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(image: Tensor, reference: Tensor) -> float:
    """Compute SSIM score."""
    device = image.device
    substrate = type('obj', (object,), {'get_device': lambda: device})()
    metrics = QualityMetrics(substrate)
    score = metrics.compute_ssim(image, reference)
    return score.score


def compute_brisque(image: Tensor) -> float:
    """Compute BRISQUE score."""
    device = image.device
    substrate = type('obj', (object,), {'get_device': lambda: device})()
    metrics = QualityMetrics(substrate)
    score = metrics.compute_brisque(image)
    return score.score


def compute_niqe(image: Tensor) -> float:
    """Compute NIQE score."""
    device = image.device
    substrate = type('obj', (object,), {'get_device': lambda: device})()
    metrics = QualityMetrics(substrate)
    score = metrics.compute_niqe(image)
    return score.score


def compute_fid(real_features: Tensor, fake_features: Tensor) -> float:
    """
    Compute FID (Fréchet Inception Distance).

    Args:
        real_features: Features from real images
        fake_features: Features from generated images

    Returns:
        FID score (lower is better)
    """
    # Compute mean and covariance
    mu1 = real_features.mean(dim=0)
    mu2 = fake_features.mean(dim=0)

    sigma1 = torch.cov(real_features.T)
    sigma2 = torch.cov(fake_features.T)

    # Compute FID
    diff = mu1 - mu2
    covmean = torch.linalg.eigvals(sigma1 @ sigma2).sqrt()

    if torch.is_complex(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)

    return fid.item()
