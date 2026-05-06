"""Quality assessment and GPU memory helpers for the 4K rendering pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from PIL import Image

# Optional: scipy for advanced image processing
try:
    from scipy.ndimage import convolve, median_filter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    convolve = None
    median_filter = None

# Optional: LPIPS for perceptual quality scoring
try:
    import torch  # noqa: F401 - used for availability check

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # noqa: F841 - placeholder for optional import

# Optional: PerceptualQualityAssessor for advanced quality metrics
try:
    from ...enhancements.perceptual_quality_assessment import PerceptualQualityAssessor

    HAS_PERCEPTUAL_ASSESSOR = True
except ImportError:
    HAS_PERCEPTUAL_ASSESSOR = False
    PerceptualQualityAssessor = None

from .types import DeviceType, QualityFeedbackConfig, QualityMetrics

__all__ = [
    "GPUMemoryManager",
    "QualityAssessor",
]

logger = logging.getLogger("transformation_portal.pipelines.rendering_4k_pipeline")


class GPUMemoryManager:
    """GPU memory monitoring and management for preventing OOM errors."""

    def __init__(self, device: DeviceType):
        """Initialize GPU memory manager.

        Args:
            device: The compute device type (CPU, CUDA, MPS)
        """
        self.device = device
        self._torch_available = HAS_TORCH

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics.

        Returns:
            Dictionary with memory stats (allocated_gb, reserved_gb, total_gb, usage_percent)
        """
        if not self._torch_available or self.device == DeviceType.CPU:
            return {}

        stats: Dict[str, float] = {}
        try:
            if self.device == DeviceType.CUDA:
                import torch

                stats["allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                stats["reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                stats["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                stats["usage_percent"] = (stats["allocated_gb"] / stats["total_gb"]) * 100
            elif self.device == DeviceType.MPS:
                import torch

                stats["allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
                stats["total_gb"] = 16.0  # Conservative estimate for Apple Silicon
                stats["usage_percent"] = (stats["allocated_gb"] / stats["total_gb"]) * 100
        except Exception as e:
            logger.debug(f"Failed to get memory stats: {e}")
        return stats

    def clear_cache(self):
        """Clear GPU memory cache."""
        if not self._torch_available or self.device == DeviceType.CPU:
            return
        try:
            import torch

            if self.device == DeviceType.CUDA:
                torch.cuda.empty_cache()
            elif self.device == DeviceType.MPS:
                torch.mps.empty_cache()
        except Exception as e:
            logger.debug(f"Failed to clear GPU cache: {e}")

    def check_memory_threshold(self, threshold: float = 0.85) -> bool:
        """Check if memory usage is below threshold.

        Args:
            threshold: Maximum acceptable memory usage ratio (0.0-1.0)

        Returns:
            True if memory usage is below threshold, False otherwise
        """
        stats = self.get_memory_stats()
        if not stats:
            return True
        usage = stats.get("usage_percent", 0) / 100.0
        return usage < threshold

    def log_memory_status(self):
        """Log current memory status."""
        stats = self.get_memory_stats()
        if stats:
            logger.info(
                f"  GPU Memory: {stats['allocated_gb']:.2f}GB / " f"{stats['total_gb']:.2f}GB ({stats['usage_percent']:.1f}%)"
            )


class QualityAssessor:
    """
    RAG-based quality assessment system.

    Evaluates image quality using multiple metrics and provides
    feedback for iterative refinement in the quality feedback loop.

    Supports two modes:
    1. Heuristic-based: Fast, lightweight quality metrics (sharpness, contrast, etc.)
    2. LPIPS-based: Perceptual quality scoring aligned with human perception

    When use_lpips=True and reference image is provided, uses LPIPS perceptual
    distance for quality scoring, targeting 95th percentile perceptual quality.
    """

    def __init__(self, config: QualityFeedbackConfig):
        """Initialize quality assessor."""
        self.config = config
        self._metric_weights = {
            "sharpness": 0.25,
            "contrast": 0.20,
            "colorfulness": 0.20,
            "exposure": 0.20,
            "noise": 0.15,
        }
        self._perceptual_assessor = None

    def _get_perceptual_assessor(self) -> Optional[PerceptualQualityAssessor]:
        """Get or initialize the perceptual quality assessor (lazy loading)."""
        if not self.config.use_lpips:
            return None

        if not HAS_PERCEPTUAL_ASSESSOR:
            logger.warning(
                "LPIPS requested but perceptual assessor not available. "
                "Install torch and lpips for perceptual quality scoring."
            )
            return None

        if self._perceptual_assessor is None:
            try:
                self._perceptual_assessor = PerceptualQualityAssessor(use_lpips_package=True)
                logger.info("Initialized LPIPS-based perceptual quality assessor")
            except Exception as e:
                logger.warning(f"Failed to initialize perceptual assessor: {e}")
                return None

        return self._perceptual_assessor

    def assess(
        self,
        image: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> QualityMetrics:
        """
        Assess image quality using multiple metrics.

        Args:
            image: RGB image as float32 array [0, 1]
            reference: Optional reference image for LPIPS comparison

        Returns:
            QualityMetrics object with all scores
        """
        metrics = QualityMetrics()

        if "sharpness" in self.config.metrics:
            metrics.sharpness = self._compute_sharpness(image)

        if "contrast" in self.config.metrics:
            metrics.contrast = self._compute_contrast(image)

        if "colorfulness" in self.config.metrics:
            metrics.colorfulness = self._compute_colorfulness(image)

        if "exposure" in self.config.metrics:
            metrics.exposure_balance = self._compute_exposure_balance(image)

        metrics.noise_level = self._estimate_noise(image)

        if self.config.use_lpips and reference is not None:
            perceptual_metrics = self._compute_lpips_metrics(image, reference)
            metrics.lpips_score = perceptual_metrics.get("lpips_score", 0.0)
            metrics.lpips_percentile = perceptual_metrics.get("lpips_percentile", 0.0)
            metrics.material_fidelity = perceptual_metrics.get("material_fidelity", 0.0)
            metrics.perceptual_quality = perceptual_metrics.get("composite_score", 0.0)

        metrics.overall_score = self._compute_overall_score(metrics)

        return metrics

    def _compute_lpips_metrics(
        self,
        enhanced: np.ndarray,
        reference: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute LPIPS-based perceptual quality metrics.

        Args:
            enhanced: Enhanced image as float32 array [0, 1]
            reference: Reference image as float32 array [0, 1]

        Returns:
            Dictionary with perceptual metrics
        """
        assessor = self._get_perceptual_assessor()
        if assessor is None:
            return {}

        try:
            enhanced_pil = Image.fromarray((np.clip(enhanced, 0, 1) * 255).astype(np.uint8), mode="RGB")
            reference_pil = Image.fromarray((np.clip(reference, 0, 1) * 255).astype(np.uint8), mode="RGB")

            report = assessor.assess(
                enhanced=enhanced_pil,
                reference=reference_pil,
                compute_material_fidelity=True,
            )

            return {
                "lpips_score": report.lpips_score,
                "lpips_percentile": report.lpips_percentile,
                "material_fidelity": report.overall_material_fidelity,
                "composite_score": report.composite_score,
                "ssim_score": report.ssim_score,
                "niqe_score": report.niqe_score,
            }

        except Exception as e:
            logger.warning(f"LPIPS assessment failed: {e}")
            return {}

    def _compute_sharpness(self, image: np.ndarray) -> float:
        """Compute sharpness using Laplacian variance."""
        gray = np.mean(image, axis=2)

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        if HAS_SCIPY and convolve is not None:
            laplacian = convolve(gray, kernel)
        else:
            laplacian = self._simple_convolve(gray, kernel)

        variance = np.var(laplacian)

        return float(np.clip(variance * 50, 0, 1))

    def _simple_convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution without scipy. WARNING: Slow for large images."""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        if h * w > 1_000_000:  # ~1MP
            logger.warning("Large image without scipy: convolution will be slow. " "Install scipy for better performance.")

        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

        result = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i : i + kh, j : j + kw] * kernel)

        return result

    def _compute_contrast(self, image: np.ndarray) -> float:
        """Compute contrast using standard deviation of luminance."""
        lum = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]

        std = np.std(lum)

        return float(np.clip(std * 3, 0, 1))

    def _compute_colorfulness(self, image: np.ndarray) -> float:
        """
        Compute colorfulness metric (Hasler and Susstrunk 2003).

        Higher values indicate more colorful images.
        """
        r, g, b = image[..., 0], image[..., 1], image[..., 2]

        rg = r - g
        yb = 0.5 * (r + g) - b

        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)

        std_root = np.sqrt(std_rg**2 + std_yb**2)
        mean_root = np.sqrt(mean_rg**2 + mean_yb**2)

        colorfulness = std_root + 0.3 * mean_root

        return float(np.clip(colorfulness * 2, 0, 1))

    def _compute_exposure_balance(self, image: np.ndarray) -> float:
        """
        Compute exposure balance score.

        Returns higher scores for well-exposed images (mean luminance ~0.4-0.6).
        """
        lum = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        mean_lum = np.mean(lum)

        optimal = 0.45
        deviation = abs(mean_lum - optimal)

        return float(np.clip(1.0 - deviation * 2, 0, 1))

    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation.

        Returns noise level (lower is better).
        """
        gray = np.mean(image, axis=2)

        if HAS_SCIPY and median_filter is not None:
            smoothed = median_filter(gray, size=3)
        else:
            smoothed = self._simple_smooth(gray, size=3)
        noise = np.abs(gray - smoothed)

        mad = np.median(noise)

        return float(np.clip(mad * 20, 0, 1))

    def _simple_smooth(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        """Simple smoothing filter without scipy. WARNING: Slow for large images."""
        h, w = image.shape
        pad = size // 2

        if h * w > 1_000_000:  # ~1MP
            logger.warning("Large image without scipy: smoothing will be slow. " "Install scipy for better performance.")

        padded = np.pad(image, pad, mode="reflect")
        result = np.zeros_like(image)

        for i in range(h):
            for j in range(w):
                result[i, j] = np.mean(padded[i : i + size, j : j + size])

        return result

    def _compute_overall_score(self, metrics: QualityMetrics) -> float:
        """Compute weighted overall quality score."""
        score = 0.0
        total_weight = 0.0

        if "sharpness" in self.config.metrics:
            score += metrics.sharpness * self._metric_weights["sharpness"]
            total_weight += self._metric_weights["sharpness"]

        if "contrast" in self.config.metrics:
            score += metrics.contrast * self._metric_weights["contrast"]
            total_weight += self._metric_weights["contrast"]

        if "colorfulness" in self.config.metrics:
            score += metrics.colorfulness * self._metric_weights["colorfulness"]
            total_weight += self._metric_weights["colorfulness"]

        if "exposure" in self.config.metrics:
            score += metrics.exposure_balance * self._metric_weights["exposure"]
            total_weight += self._metric_weights["exposure"]

        noise_penalty = metrics.noise_level * self._metric_weights["noise"]
        score -= noise_penalty
        total_weight += self._metric_weights["noise"]

        if total_weight > 0:
            score = max(0, score / total_weight)

        return float(np.clip(score, 0, 1))

    def suggest_adjustments(self, metrics: QualityMetrics) -> Dict[str, float]:
        """
        Suggest parameter adjustments based on quality metrics.

        Returns dictionary of parameter adjustments for the feedback loop.
        """
        adjustments = {}

        if metrics.sharpness < 0.5:
            adjustments["clarity_boost"] = 0.2

        if metrics.contrast < 0.4:
            adjustments["contrast_increase"] = 0.1

        if metrics.colorfulness < 0.4:
            adjustments["saturation_boost"] = 0.05

        if metrics.exposure_balance < 0.5:
            adjustments["exposure_adjust"] = 0.1 if metrics.exposure_balance < 0.4 else -0.1

        if metrics.noise_level > 0.3:
            adjustments["denoise_strength"] = 0.2

        return adjustments
