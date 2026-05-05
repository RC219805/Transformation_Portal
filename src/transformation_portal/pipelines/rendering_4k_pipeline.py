#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End 4K Rendering Enhancement Pipeline

Integrates the best features from the Transformation Portal ecosystem:
- Depth Anything V2 with CoreML/MPS acceleration
- Material Response Technology for surface realism
- Intelligent tone mapping (AgX, Filmic, Reinhard)
- AI-powered enhancement via ControlNet guidance
- Real-ESRGAN 4x upscaling to 4K resolution
- RAG-based quality feedback loop for iterative refinement
- Professional color grading with LUT stacks
- Complete metadata preservation

Designed for:
- Luxury real estate rendering
- Architectural visualization
- Editorial post-production

Optimized for:
- Apple Silicon (M-series) with Metal Performance Shaders
- NVIDIA CUDA GPUs
- CPU fallback for compatibility

Example:
    >>> from transformation_portal.pipelines.rendering_4k_pipeline import Rendering4KPipeline
    >>> pipeline = Rendering4KPipeline.from_preset("luxury_estate")
    >>> result = pipeline.process("input.jpg", output_dir="output/")
    >>> print(f"Quality Score: {result.quality_score}")

Author: Transformation Portal Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# Optional: scipy for advanced image processing
try:
    from scipy.ndimage import convolve, gaussian_filter, median_filter, uniform_filter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    convolve = None
    gaussian_filter = None
    median_filter = None
    uniform_filter = None

# Optional: PyYAML for configuration loading
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Optional: LPIPS for perceptual quality scoring
try:
    import torch  # noqa: F401 - used for availability check

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # noqa: F841 - placeholder for optional import

# Optional: ControlNet auxiliary processors
try:
    from controlnet_aux import CannyDetector

    HAS_CONTROLNET_AUX = True
except ImportError:
    HAS_CONTROLNET_AUX = False
    CannyDetector = None

# Optional: PerceptualQualityAssessor for advanced quality metrics
try:
    from ...enhancements.perceptual_quality_assessment import PerceptualQualityAssessor
    from ...enhancements.perceptual_quality_assessment import QualityReport as PerceptualQualityReport

    HAS_PERCEPTUAL_ASSESSOR = True
except ImportError:
    HAS_PERCEPTUAL_ASSESSOR = False
    PerceptualQualityAssessor = None
    PerceptualQualityReport = None

# Optional: QualityFeedbackBridge for unified quality assessment
try:
    from .quality_feedback_bridge import QualityFeedbackBridge, UnifiedQualityMetrics, create_rag_indexing_callback

    HAS_QUALITY_BRIDGE = True
except ImportError:
    HAS_QUALITY_BRIDGE = False
    QualityFeedbackBridge = None
    UnifiedQualityMetrics = None
    create_rag_indexing_callback = None

# Import internal utilities
from ..core.security.model_lock import resolve_model_lock_revision
from ..utils.image_utils import load_image, np_to_pil, pil_to_np
from .rendering_4k.types import (
    STAGE_NAMES,
    AIEnhancementConfig,
    ColorGradingConfig,
    DepthConfig,
    DeviceType,
    MaterialResponseConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingResult,
    QualityFeedbackConfig,
    QualityLevel,
    QualityMetrics,
    StageMetrics,
    ToneMappingConfig,
    ToneMappingMethod,
    UpscalingConfig,
)

logger = logging.getLogger(__name__)


def _json_default(obj: object) -> object:
    """`json.dump` fallback serializer for common ML / pathlib types."""
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Path):
        return str(obj)

    if HAS_TORCH and torch is not None:
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.ndim == 0 else obj.detach().cpu().tolist()

    return str(obj)


# =============================================================================
# GPU Memory Management
# =============================================================================


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


# =============================================================================
# Quality Assessment Module
# =============================================================================


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
        # Lazy-loaded perceptual assessor
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

        # Compute individual metrics
        if "sharpness" in self.config.metrics:
            metrics.sharpness = self._compute_sharpness(image)

        if "contrast" in self.config.metrics:
            metrics.contrast = self._compute_contrast(image)

        if "colorfulness" in self.config.metrics:
            metrics.colorfulness = self._compute_colorfulness(image)

        if "exposure" in self.config.metrics:
            metrics.exposure_balance = self._compute_exposure_balance(image)

        metrics.noise_level = self._estimate_noise(image)

        # LPIPS perceptual scoring (when enabled and reference available)
        if self.config.use_lpips and reference is not None:
            perceptual_metrics = self._compute_lpips_metrics(image, reference)
            metrics.lpips_score = perceptual_metrics.get("lpips_score", 0.0)
            metrics.lpips_percentile = perceptual_metrics.get("lpips_percentile", 0.0)
            metrics.material_fidelity = perceptual_metrics.get("material_fidelity", 0.0)
            metrics.perceptual_quality = perceptual_metrics.get("composite_score", 0.0)

        # Compute weighted overall score
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
            # Convert numpy arrays to PIL Images for the assessor
            enhanced_pil = Image.fromarray((np.clip(enhanced, 0, 1) * 255).astype(np.uint8), mode="RGB")
            reference_pil = Image.fromarray((np.clip(reference, 0, 1) * 255).astype(np.uint8), mode="RGB")

            # Run perceptual assessment
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
        # Convert to grayscale
        gray = np.mean(image, axis=2)

        # Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        # Convolve (use scipy if available, else simple fallback)
        if HAS_SCIPY and convolve is not None:
            laplacian = convolve(gray, kernel)
        else:
            # Simple numpy-based convolution fallback
            laplacian = self._simple_convolve(gray, kernel)

        # Variance of Laplacian as sharpness measure
        variance = np.var(laplacian)

        # Normalize to 0-1 (empirical scaling)
        return float(np.clip(variance * 50, 0, 1))

    def _simple_convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution without scipy. WARNING: Slow for large images."""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Warn about performance for large images
        if h * w > 1_000_000:  # ~1MP
            logger.warning("Large image without scipy: convolution will be slow. " "Install scipy for better performance.")

        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

        # Convolve
        result = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i : i + kh, j : j + kw] * kernel)

        return result

    def _compute_contrast(self, image: np.ndarray) -> float:
        """Compute contrast using standard deviation of luminance."""
        # Compute luminance
        lum = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]

        # Standard deviation as contrast measure
        std = np.std(lum)

        # Normalize to 0-1 (optimal range around 0.15-0.25)
        return float(np.clip(std * 3, 0, 1))

    def _compute_colorfulness(self, image: np.ndarray) -> float:
        """
        Compute colorfulness metric (Hasler & Süsstrunk 2003).

        Higher values indicate more colorful images.
        """
        r, g, b = image[..., 0], image[..., 1], image[..., 2]

        rg = r - g
        yb = 0.5 * (r + g) - b

        # Standard deviation and mean of color opponent channels
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)

        std_root = np.sqrt(std_rg**2 + std_yb**2)
        mean_root = np.sqrt(mean_rg**2 + mean_yb**2)

        colorfulness = std_root + 0.3 * mean_root

        # Normalize to 0-1 (empirical scaling)
        return float(np.clip(colorfulness * 2, 0, 1))

    def _compute_exposure_balance(self, image: np.ndarray) -> float:
        """
        Compute exposure balance score.

        Returns higher scores for well-exposed images (mean luminance ~0.4-0.6).
        """
        lum = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        mean_lum = np.mean(lum)

        # Optimal mean luminance is around 0.4-0.5
        # Penalize both over and under exposure
        optimal = 0.45
        deviation = abs(mean_lum - optimal)

        # Score decreases as deviation increases
        return float(np.clip(1.0 - deviation * 2, 0, 1))

    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation.

        Returns noise level (lower is better).
        """
        gray = np.mean(image, axis=2)

        # High-pass filter to isolate noise
        if HAS_SCIPY and median_filter is not None:
            smoothed = median_filter(gray, size=3)
        else:
            # Simple fallback: use local mean
            smoothed = self._simple_smooth(gray, size=3)
        noise = np.abs(gray - smoothed)

        # Median absolute deviation
        mad = np.median(noise)

        # Normalize (empirical scaling)
        return float(np.clip(mad * 20, 0, 1))

    def _simple_smooth(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        """Simple smoothing filter without scipy. WARNING: Slow for large images."""
        h, w = image.shape
        pad = size // 2

        # Warn about performance for large images
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

        # Noise penalty (inverse - lower noise is better)
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

        # Low sharpness -> increase clarity/sharpening
        if metrics.sharpness < 0.5:
            adjustments["clarity_boost"] = 0.2

        # Low contrast -> increase contrast
        if metrics.contrast < 0.4:
            adjustments["contrast_increase"] = 0.1

        # Low colorfulness -> increase saturation
        if metrics.colorfulness < 0.4:
            adjustments["saturation_boost"] = 0.05

        # Poor exposure -> adjust exposure
        if metrics.exposure_balance < 0.5:
            # Determine direction from luminance
            adjustments["exposure_adjust"] = 0.1 if metrics.exposure_balance < 0.4 else -0.1

        # High noise -> increase denoising
        if metrics.noise_level > 0.3:
            adjustments["denoise_strength"] = 0.2

        return adjustments


# =============================================================================
# Image Processing Functions
# =============================================================================


def apply_tone_mapping(
    image: np.ndarray,
    config: ToneMappingConfig,
) -> np.ndarray:
    """
    Apply HDR tone mapping to image.

    Args:
        image: HDR image as float32 array
        config: Tone mapping configuration

    Returns:
        Tone-mapped image in [0, 1] range
    """
    if not config.enabled:
        return np.clip(image, 0, 1)

    # Apply exposure adjustment first
    if config.exposure != 0:
        image = image * (2.0**config.exposure)

    # Select tone mapping operator
    if config.method == ToneMappingMethod.REINHARD:
        # Simple Reinhard global operator
        mapped = image / (1.0 + image)

    elif config.method == ToneMappingMethod.FILMIC:
        # Hable/Uncharted 2 filmic curve
        mapped = _filmic_hable(image, config.white_point)

    elif config.method == ToneMappingMethod.ACES:
        # ACES approximation
        mapped = _aces_approximation(image)

    else:  # AGX (default)
        # AgX-inspired sigmoid curve
        mapped = _agx_sigmoid(image)

    # Apply contrast adjustment
    if config.contrast != 1.0:
        mean = np.mean(mapped)
        mapped = (mapped - mean) * config.contrast + mean

    return np.clip(mapped, 0, 1).astype(np.float32)


def _filmic_hable(x: np.ndarray, white_point: float = 11.2) -> np.ndarray:
    """Hable/Uncharted 2 filmic tone mapping curve."""

    def hable_curve(v: np.ndarray) -> np.ndarray:
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

    curr = hable_curve(x)
    white = hable_curve(np.array([white_point]))
    return curr / white


def _aces_approximation(x: np.ndarray) -> np.ndarray:
    """Simple ACES approximation (Krzysztof Narkowicz)."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _agx_sigmoid(x: np.ndarray) -> np.ndarray:
    """AgX-inspired sigmoid tone mapping."""
    # Apply log-space compression
    x = np.maximum(x, 1e-10)
    log_x = np.log2(x + 0.001)

    # Sigmoid in log space
    sigmoid = 1.0 / (1.0 + np.exp(-log_x * 0.5))

    # Scale to output range
    return np.clip(sigmoid, 0, 1)


def apply_material_response(
    image: np.ndarray,
    depth_map: Optional[np.ndarray],
    config: MaterialResponseConfig,
) -> np.ndarray:
    """
    Apply Material Response Technology enhancement.

    Enhances surface textures and material properties using depth information.

    Args:
        image: Input image as float32 array [0, 1]
        depth_map: Depth map (optional, improves results)
        config: Material Response configuration

    Returns:
        Enhanced image
    """
    if not config.enabled:
        return image

    enhanced = image.copy()

    # Texture enhancement via high-frequency boost
    if config.texture_boost > 0:
        if HAS_SCIPY and gaussian_filter is not None:
            blurred = gaussian_filter(enhanced, sigma=(1.2, 1.2, 0))
        else:
            # Fallback: use PIL-based blur
            blurred = _simple_gaussian_blur(enhanced, sigma=1.2)
        detail = enhanced - blurred
        enhanced = np.clip(enhanced + config.texture_boost * detail, 0, 1)

    # Micro-contrast enhancement
    if config.micro_contrast > 0:
        enhanced = _apply_local_contrast(enhanced, config.micro_contrast)

    # Apply strength blending
    enhanced = image * (1 - config.strength) + enhanced * config.strength

    return np.clip(enhanced, 0, 1).astype(np.float32)


def _simple_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Simple Gaussian blur using PIL as fallback."""
    from PIL import ImageFilter

    # Convert to PIL, blur, convert back
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred).astype(np.float32) / 255.0


def _apply_local_contrast(image: np.ndarray, strength: float) -> np.ndarray:
    """Apply local contrast enhancement (CLAHE-like)."""
    if HAS_SCIPY and uniform_filter is not None:
        # Local mean using scipy
        local_mean = uniform_filter(image, size=(32, 32, 1))
    else:
        # Fallback: use simple box blur
        local_mean = _simple_box_blur(image, size=32)

    # Local contrast enhancement
    enhanced = image + strength * (image - local_mean)

    return np.clip(enhanced, 0, 1)


def _simple_box_blur(image: np.ndarray, size: int) -> np.ndarray:
    """Simple box blur as fallback for uniform_filter."""
    from PIL import ImageFilter

    # Handle each channel
    h, w, c = image.shape
    result = np.zeros_like(image)
    for ch in range(c):
        img_uint8 = (np.clip(image[..., ch], 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="L")
        blurred = pil_img.filter(ImageFilter.BoxBlur(size // 2))
        result[..., ch] = np.array(blurred).astype(np.float32) / 255.0
    return result


def apply_color_grading(
    image: np.ndarray,
    config: ColorGradingConfig,
) -> np.ndarray:
    """
    Apply color grading adjustments including LUT stacks.

    Supports:
    - Temperature shift (RGB multipliers)
    - Saturation and vibrance adjustments
    - LUT (Look-Up Table) application with configurable strengths

    Args:
        image: Input image as float32 array [0, 1]
        config: Color grading configuration

    Returns:
        Color-graded image
    """
    if not config.enabled:
        return image

    graded = image.copy()

    # Apply LUTs first (before other adjustments)
    if config.lut_paths and config.lut_strengths:
        for lut_path, strength in zip(config.lut_paths, config.lut_strengths):
            if strength > 0:
                lut_result = _apply_lut(graded, lut_path, strength)
                if lut_result is not None:
                    graded = lut_result
                    logger.debug(f"Applied LUT: {Path(lut_path).name} @ {strength:.0%}")
                else:
                    logger.warning(f"Failed to apply LUT: {Path(lut_path).name} (strength={strength:.0%})")

    # Apply temperature shift (RGB multipliers)
    r_mult, g_mult, b_mult = config.temperature_shift
    graded[..., 0] *= r_mult
    graded[..., 1] *= g_mult
    graded[..., 2] *= b_mult

    # Apply saturation adjustment
    if config.saturation != 1.0:
        # Convert to HSV-like representation
        lum = 0.2126 * graded[..., 0] + 0.7152 * graded[..., 1] + 0.0722 * graded[..., 2]
        graded = lum[..., np.newaxis] + config.saturation * (graded - lum[..., np.newaxis])

    # Apply vibrance (saturation that targets less saturated colors)
    if config.vibrance != 1.0:
        graded = _apply_vibrance(graded, config.vibrance)

    return np.clip(graded, 0, 1).astype(np.float32)


def _load_cube_lut(lut_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load a .cube LUT file.

    Args:
        lut_path: Path to .cube LUT file

    Returns:
        3D LUT array (size, size, size, 3) or None if loading fails
    """
    lut_path = Path(lut_path)
    if not lut_path.exists():
        logger.warning(f"LUT file not found: {lut_path}")
        return None

    try:
        lut_size = 0
        lut_data = []

        with open(lut_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("LUT_3D_SIZE"):
                    lut_size = int(line.split()[-1])
                elif line and not line.startswith("#") and not line.startswith("TITLE"):
                    # Skip comments, titles, and domain specifications
                    if line.startswith(("DOMAIN_", "LUT_")):
                        continue
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            r, g, b = map(float, parts)
                            lut_data.append([r, g, b])
                        except ValueError:
                            continue

        if lut_size > 0 and len(lut_data) == lut_size**3:
            return np.array(lut_data, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3)
        else:
            logger.warning(f"Invalid LUT data: size={lut_size}, expected {lut_size**3} entries, got {len(lut_data)}")
            return None

    except Exception as e:
        logger.warning(f"Failed to load LUT {lut_path}: {e}")
        return None


def _apply_lut(
    image: np.ndarray,
    lut_path: Union[str, Path],
    strength: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Apply a .cube LUT to an image using trilinear interpolation.

    Args:
        image: Input image as float32 array [0, 1] with shape (H, W, 3)
        lut_path: Path to .cube LUT file
        strength: LUT application strength (0.0-1.0)

    Returns:
        LUT-processed image, or None if LUT could not be applied
    """
    lut = _load_cube_lut(lut_path)
    if lut is None:
        return None

    lut_size = lut.shape[0]

    # Normalize image to LUT index space
    array = np.clip(image, 0, 1).astype(np.float32)
    indices = array * (lut_size - 1)
    indices = np.clip(indices, 0, lut_size - 1.001)

    # Get floor and ceiling indices for trilinear interpolation
    idx0 = np.floor(indices).astype(np.int32)
    idx1 = np.minimum(idx0 + 1, lut_size - 1)
    frac = indices - idx0

    # Extract RGB indices
    r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
    r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]
    fr, fg, fb = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]

    # Trilinear interpolation (8 corner lookups)
    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    # Interpolate along each axis
    c00 = c000 * (1 - fr) + c100 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c11 = c011 * (1 - fr) + c111 * fr

    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg

    graded = c0 * (1 - fb) + c1 * fb

    # Blend with original based on strength
    result = array * (1 - strength) + graded * strength

    return np.clip(result, 0, 1).astype(np.float32)


def _apply_vibrance(image: np.ndarray, vibrance: float) -> np.ndarray:
    """Apply vibrance (smart saturation targeting less saturated colors)."""
    # Compute current saturation
    max_rgb = np.max(image, axis=2, keepdims=True)
    min_rgb = np.min(image, axis=2, keepdims=True)
    sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)

    # Low saturation areas get more boost
    boost = 1.0 + (vibrance - 1.0) * (1.0 - sat)

    # Apply saturation boost
    lum = 0.2126 * image[..., 0:1] + 0.7152 * image[..., 1:2] + 0.0722 * image[..., 2:3]
    boosted = lum + boost * (image - lum)

    return np.clip(boosted, 0, 1)


def apply_upscaling(
    image: Image.Image,
    config: UpscalingConfig,
) -> Image.Image:
    """
    Upscale image to target resolution.

    Args:
        image: PIL Image to upscale
        config: Upscaling configuration

    Returns:
        Upscaled PIL Image
    """
    if not config.enabled:
        return image

    current_w, current_h = image.size
    target_w, target_h = config.target_resolution

    # Check if upscaling is needed
    if current_w >= target_w and current_h >= target_h:
        logger.info(f"Image already at or above target resolution ({current_w}x{current_h})")
        return image

    # Calculate scale to fit within target while maintaining aspect ratio
    scale_w = target_w / current_w
    scale_h = target_h / current_h
    scale = min(scale_w, scale_h)

    new_w = int(current_w * scale)
    new_h = int(current_h * scale)

    # Apply upscaling method
    if config.method == "esrgan":
        # Real-ESRGAN upscaling (requires optional dependencies)
        logger.warning("ESRGAN upscaling requires optional ML dependencies. Using Lanczos fallback.")
        upscaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        # Default Lanczos
        upscaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Optional sharpening after upscale
    if config.preserve_sharpness:
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=50, threshold=0))

    logger.info(f"Upscaled from {current_w}x{current_h} to {new_w}x{new_h}")

    return upscaled


def estimate_depth_simple(image: np.ndarray) -> np.ndarray:
    """
    Simple depth estimation using luminance gradient.

    This is a lightweight fallback when Depth Anything V2 is not available.
    For production use, the full depth model should be used.

    Args:
        image: RGB image as float32 array [0, 1]

    Returns:
        Depth map as float32 array [0, 1]
    """
    # Compute luminance
    lum = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]

    # Simple depth proxy using luminance + spatial gradient
    if HAS_SCIPY and gaussian_filter is not None:
        # Blur for depth approximation (distant objects blur more)
        blurred = gaussian_filter(lum, sigma=15)
    else:
        # Fallback: use PIL-based blur
        blurred = _simple_gaussian_blur_2d(lum, sigma=15)

    # Vertical gradient (sky typically brighter at top)
    h, w = lum.shape
    y_gradient = np.linspace(0, 1, h)[:, np.newaxis]
    y_gradient = np.tile(y_gradient, (1, w))

    # Combine luminance inversion with spatial cues
    depth = 0.5 * (1 - blurred) + 0.5 * y_gradient

    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    return depth.astype(np.float32)


def _simple_gaussian_blur_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    """Simple 2D Gaussian blur using PIL as fallback."""
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred).astype(np.float32) / 255.0


# =============================================================================
# Main Pipeline Class
# =============================================================================


class Rendering4KPipeline:
    """
    End-to-End 4K Rendering Enhancement Pipeline.

    Combines depth estimation, tone mapping, material response, color grading,
    AI enhancement, and upscaling with a RAG-based quality feedback loop.

    Example:
        >>> pipeline = Rendering4KPipeline.from_preset("luxury_estate")
        >>> result = pipeline.process("input.jpg", output_dir="output/")
        >>> print(f"Quality: {result.quality_score:.2%}")
    """

    # Built-in presets
    PRESETS = {
        "default": PipelineConfig(
            name="default",
            description="Balanced settings for general use",
        ),
        "luxury_estate": PipelineConfig(
            name="luxury_estate",
            description="Optimized for luxury real estate interiors",
            material_response=MaterialResponseConfig(
                strength=0.75,
                texture_boost=0.3,
                micro_contrast=0.2,
            ),
            color_grading=ColorGradingConfig(
                saturation=1.08,
                vibrance=1.12,
                temperature_shift=(1.0, 0.98, 0.95),  # Warm
            ),
            quality_feedback=QualityFeedbackConfig(
                use_lpips=True,  # Enable LPIPS for luxury workflows
                hybrid_mode=True,
                rag_indexing_enabled=True,
            ),
        ),
        "aerial_exterior": PipelineConfig(
            name="aerial_exterior",
            description="Optimized for aerial and exterior shots",
            depth=DepthConfig(
                num_zones=3,
            ),
            tone_mapping=ToneMappingConfig(
                method=ToneMappingMethod.FILMIC,
                contrast=1.1,
            ),
            color_grading=ColorGradingConfig(
                saturation=1.12,
                vibrance=1.15,
                temperature_shift=(1.05, 1.0, 0.95),  # Golden hour warmth
            ),
            quality_feedback=QualityFeedbackConfig(
                use_lpips=True,  # Enable LPIPS for luxury workflows
                hybrid_mode=True,
            ),
        ),
        "editorial": PipelineConfig(
            name="editorial",
            description="High-end editorial/magazine quality",
            quality_level=QualityLevel.ULTRA,
            tone_mapping=ToneMappingConfig(
                method=ToneMappingMethod.ACES,
                contrast=1.05,
            ),
            material_response=MaterialResponseConfig(
                strength=0.8,
                texture_boost=0.35,
            ),
            quality_feedback=QualityFeedbackConfig(
                use_lpips=True,  # Enable LPIPS for editorial workflows
                hybrid_mode=True,
                rag_indexing_enabled=True,
            ),
        ),
        "750_picacho": PipelineConfig(
            name="750_picacho",
            description="Optimized preset for 750 Picacho Lane estate images",
            quality_level=QualityLevel.ULTRA,
            material_response=MaterialResponseConfig(
                strength=0.80,
                texture_boost=0.35,
                micro_contrast=0.25,
                surface_types=["quartzite", "oak", "metal", "glass", "stucco"],
            ),
            color_grading=ColorGradingConfig(
                saturation=1.10,
                vibrance=1.15,
                temperature_shift=(1.02, 0.99, 0.96),  # Warm Montecito tones
            ),
            quality_feedback=QualityFeedbackConfig(
                use_lpips=True,
                hybrid_mode=True,
                perceptual_percentile_target=95.0,
                material_fidelity_target=0.98,
                rag_indexing_enabled=True,
            ),
        ),
        "preview": PipelineConfig(
            name="preview",
            description="Fast preview with reduced quality",
            quality_level=QualityLevel.PREVIEW,
            depth=DepthConfig(enabled=False),
            material_response=MaterialResponseConfig(strength=0.5),
            upscaling=UpscalingConfig(enabled=False),
            quality_feedback=QualityFeedbackConfig(enabled=False),
            output=OutputConfig(
                master_tiff_16bit=False,
                save_intermediate=False,
                save_depth_visualization=False,
            ),
        ),
    }

    def __init__(
        self,
        config: PipelineConfig,
        *,
        strict_model_lock: Optional[bool] = None,
    ):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
            strict_model_lock: Enforce pinned revisions for remote model loads.
                If None, uses ``TP_STRICT_MODEL_LOCK`` environment variable.
        """
        self.config = config
        self.strict_model_lock = strict_model_lock
        self.quality_assessor = QualityAssessor(config.quality_feedback)
        # Use OrderedDict for true LRU cache behavior
        self._depth_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Initialize QualityFeedbackBridge if available and LPIPS requested
        self._quality_bridge: Optional[QualityFeedbackBridge] = None
        if HAS_QUALITY_BRIDGE and config.quality_feedback.use_lpips:
            rag_callback = None
            if config.quality_feedback.rag_indexing_enabled and config.quality_feedback.rag_index_path:
                rag_callback = create_rag_indexing_callback(config.quality_feedback.rag_index_path)
            self._quality_bridge = QualityFeedbackBridge(
                hybrid_mode=config.quality_feedback.hybrid_mode,
                lpips_network=config.quality_feedback.lpips_network,
                enable_material_fidelity=config.quality_feedback.enable_material_fidelity,
                rag_callback=rag_callback,
            )
            logger.info("QualityFeedbackBridge initialized for LPIPS scoring")

        # Track original input for quality comparison
        self._current_original: Optional[np.ndarray] = None
        self._current_image_id: str = ""

        # Detect compute device
        self.device = self._detect_device()

        # Initialize GPU memory manager
        self.memory_manager = GPUMemoryManager(self.device)

        # Initialize ML models (lazy loading)
        self._depth_model = None
        self._depth_model_initialized = False
        self._controlnet_pipe = None
        self._controlnet_initialized = False

        logger.info(f"Initialized Rendering4KPipeline: {config.name}")
        logger.info(f"Device: {self.device.value}")
        logger.info(f"Quality Level: {config.quality_level.value}")

        # Log GPU status if available
        if self.device != DeviceType.CPU:
            self.memory_manager.log_memory_status()

    @classmethod
    def from_preset(cls, preset_name: str) -> "Rendering4KPipeline":
        """
        Create pipeline from built-in preset.

        Args:
            preset_name: Name of preset (default, luxury_estate, aerial_exterior, editorial, preview)

        Returns:
            Initialized pipeline

        Raises:
            ValueError: If preset not found
        """
        if preset_name not in cls.PRESETS:
            available = ", ".join(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        config = cls.PRESETS[preset_name]
        return cls(config)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Rendering4KPipeline":
        """
        Create pipeline from YAML configuration file.

        Args:
            config_path: Path to YAML config

        Returns:
            Initialized pipeline

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If config file does not exist
        """
        if not HAS_YAML or yaml is None:
            raise ImportError("PyYAML is required for loading YAML configs. " "Install with: pip install pyyaml")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open(encoding="utf-8") as f:
            # YAML_GOVERNANCE_EXEMPT: legacy render pipeline config, not a preset/governance entrypoint.
            data = yaml.safe_load(f)

        # Build config from YAML data
        config = cls._build_config_from_dict(data)

        return cls(config)

    @staticmethod
    def _build_config_from_dict(data: Dict) -> PipelineConfig:
        """Build PipelineConfig from dictionary with proper enum conversion."""
        # Parse nested configs (most use strings, no enum conversion needed)
        depth = DepthConfig(**data.get("depth", {}))

        # Parse tone mapping config with ToneMappingMethod enum conversion
        tone_mapping_data = data.get("tone_mapping", {})
        if "method" in tone_mapping_data and isinstance(tone_mapping_data["method"], str):
            try:
                tone_mapping_data["method"] = ToneMappingMethod(tone_mapping_data["method"])
            except ValueError:
                logger.warning(f"Invalid tone_mapping method '{tone_mapping_data['method']}', using 'agx'")
                tone_mapping_data["method"] = ToneMappingMethod.AGX
        tone_mapping = ToneMappingConfig(**tone_mapping_data)

        # Parse remaining configs (all use strings, no enum conversion needed)
        material_response = MaterialResponseConfig(**data.get("material_response", {}))
        color_grading = ColorGradingConfig(**data.get("color_grading", {}))
        ai_enhancement = AIEnhancementConfig(**data.get("ai_enhancement", {}))
        upscaling = UpscalingConfig(**data.get("upscaling", {}))
        quality_feedback = QualityFeedbackConfig(**data.get("quality_feedback", {}))
        output = OutputConfig(**data.get("output", {}))

        # Parse quality level with validation
        quality_level_value = data.get("quality_level", "high")
        try:
            quality_level = QualityLevel(quality_level_value)
        except ValueError:
            logger.warning(f"Invalid quality_level '{quality_level_value}', using 'high'")
            quality_level = QualityLevel.HIGH

        return PipelineConfig(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            quality_level=quality_level,
            depth=depth,
            tone_mapping=tone_mapping,
            material_response=material_response,
            color_grading=color_grading,
            ai_enhancement=ai_enhancement,
            upscaling=upscaling,
            quality_feedback=quality_feedback,
            output=output,
        )

    def _detect_device(self) -> DeviceType:
        """Detect best available compute device."""
        try:
            import torch

            # Check for MPS (Apple Silicon) support
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
                if torch.backends.mps.is_available():
                    return DeviceType.MPS
            # Check for CUDA support
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return DeviceType.CUDA
        except (ImportError, AttributeError):
            # torch is not installed or has unexpected structure; fall back to CPU processing
            pass
        return DeviceType.CPU

    def process(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> ProcessingResult:
        """
        Process single image through complete pipeline.

        Args:
            input_path: Path to input image
            output_dir: Output directory (optional)

        Returns:
            ProcessingResult with enhanced image and metadata
        """
        start_time = time.time()
        input_path = Path(input_path)
        stage_metrics: List[StageMetrics] = []

        # Store image ID and original for RAG provenance and LPIPS comparison
        self._current_image_id = input_path.stem
        self._current_original = None

        logger.info("=" * 70)
        logger.info(f"Processing: {input_path.name}")
        logger.info(f"Preset: {self.config.name}")
        logger.info("=" * 70)

        # Stage 1: Input Validation
        stage_start = time.time()
        logger.info("[1/9] Input Validation")
        try:
            image_pil = load_image(input_path)
            image_np = pil_to_np(image_pil, to_float=True)
            # Store original for quality comparison
            self._current_original = image_np.copy()
            logger.info(f"  Size: {image_pil.size}, Shape: {image_np.shape}")
            stage_metrics.append(
                StageMetrics(
                    "input_validation",
                    (time.time() - stage_start) * 1000,
                    True,
                )
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")
            raise

        # Stage 2: Depth Estimation
        stage_start = time.time()
        logger.info("[2/9] Depth Estimation")
        depth_map = None
        if self.config.depth.enabled:
            depth_map = self._estimate_depth(image_np, input_path)
            logger.info(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "depth_estimation",
                (time.time() - stage_start) * 1000,
                True if depth_map is not None else False,
            )
        )

        # Stage 3: Tone Mapping
        stage_start = time.time()
        logger.info("[3/9] Tone Mapping")
        if self.config.tone_mapping.enabled:
            processed = apply_tone_mapping(image_np, self.config.tone_mapping)
            logger.info(f"  Method: {self.config.tone_mapping.method.value}")
        else:
            processed = image_np
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "tone_mapping",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Stage 4: Material Response
        stage_start = time.time()
        logger.info("[4/9] Material Response")
        if self.config.material_response.enabled:
            processed = apply_material_response(processed, depth_map, self.config.material_response)
            logger.info(f"  Strength: {self.config.material_response.strength}")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "material_response",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Stage 5: Color Grading
        stage_start = time.time()
        logger.info("[5/9] Color Grading")
        if self.config.color_grading.enabled:
            processed = apply_color_grading(processed, self.config.color_grading)
            sat = self.config.color_grading.saturation
            vib = self.config.color_grading.vibrance
            logger.info(f"  Saturation: {sat}, Vibrance: {vib}")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "color_grading",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Convert to PIL for AI enhancement and upscaling
        result_pil = np_to_pil(processed)

        # Stage 6: AI Enhancement (optional, requires ML deps)
        stage_start = time.time()
        logger.info("[6/9] AI Enhancement")
        if self.config.ai_enhancement.enabled:
            try:
                result_pil = self._apply_ai_enhancement(result_pil, depth_map)
                logger.info("  ✓ ControlNet enhancement complete")
                stage_metrics.append(
                    StageMetrics(
                        "ai_enhancement",
                        (time.time() - stage_start) * 1000,
                        True,
                    )
                )
            except Exception as e:
                logger.warning(f"  AI enhancement failed: {e}")
                stage_metrics.append(
                    StageMetrics(
                        "ai_enhancement",
                        (time.time() - stage_start) * 1000,
                        False,
                        notes=str(e),
                    )
                )
        else:
            logger.info("  Skipped (disabled)")
            stage_metrics.append(
                StageMetrics(
                    "ai_enhancement",
                    (time.time() - stage_start) * 1000,
                    False,
                )
            )

        # Stage 7: Upscaling to 4K
        stage_start = time.time()
        logger.info("[7/9] Upscaling")
        if self.config.upscaling.enabled:
            result_pil = apply_upscaling(result_pil, self.config.upscaling)
            logger.info(f"  Target: {self.config.upscaling.target_resolution}")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "upscaling",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Stage 8: Quality Assessment & Feedback Loop
        stage_start = time.time()
        logger.info("[8/9] Quality Assessment")
        quality_metrics = None
        unified_metrics = None
        iterations = 1
        if self.config.quality_feedback.enabled:
            enhanced_np = pil_to_np(result_pil, to_float=True)

            # Use QualityFeedbackBridge if available (LPIPS-based scoring)
            if self._quality_bridge is not None:
                unified_metrics = self._quality_bridge.assess(
                    enhanced=enhanced_np,
                    original=self._current_original,
                    image_id=self._current_image_id,
                    pipeline_config_name=self.config.name,
                )
                # Translate unified metrics to QualityMetrics for backward compatibility
                quality_metrics = QualityMetrics(
                    sharpness=unified_metrics.heuristic.sharpness,
                    contrast=unified_metrics.heuristic.contrast,
                    colorfulness=unified_metrics.heuristic.colorfulness,
                    exposure_balance=unified_metrics.heuristic.exposure_balance,
                    noise_level=unified_metrics.heuristic.noise_level,
                    overall_score=unified_metrics.hybrid_score / 100.0,  # Normalize to 0-1
                    lpips_score=unified_metrics.perceptual.lpips_score,
                    lpips_percentile=unified_metrics.perceptual.lpips_percentile,
                    material_fidelity=unified_metrics.material_fidelity.overall_fidelity,
                    perceptual_quality=unified_metrics.perceptual_composite,
                )
                logger.info(f"  Hybrid Score: {unified_metrics.hybrid_score:.1f}/100")
                logger.info(f"  Perceptual: {unified_metrics.perceptual_composite:.1f}/100")
                logger.info(f"  Heuristic: {unified_metrics.heuristic_composite:.1f}/100")
                if unified_metrics.lpips_available:
                    logger.info(f"  LPIPS: {unified_metrics.perceptual.lpips_score:.4f}")
                    logger.info(f"  Material Fidelity: {unified_metrics.material_fidelity.overall_fidelity:.1%}")
                logger.info(f"  {unified_metrics.targets_summary}")
            else:
                # Fallback to heuristic-only QualityAssessor
                quality_metrics = self.quality_assessor.assess(enhanced_np)
                logger.info(f"  Overall Score: {quality_metrics.overall_score:.2%}")
                logger.info(f"  Sharpness: {quality_metrics.sharpness:.2%}")
                logger.info(f"  Contrast: {quality_metrics.contrast:.2%}")
                logger.info(f"  Colorfulness: {quality_metrics.colorfulness:.2%}")

            # Feedback loop for quality refinement
            auto_adjust = self.config.quality_feedback.auto_adjust
            threshold = self.config.quality_feedback.min_quality_threshold
            if auto_adjust and quality_metrics.overall_score < threshold:
                logger.info("  Quality below threshold, suggesting adjustments...")
                adjustments = self.quality_assessor.suggest_adjustments(quality_metrics)
                logger.info(f"  Suggested: {adjustments}")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "quality_assessment",
                (time.time() - stage_start) * 1000,
                True if quality_metrics else False,
            )
        )

        # Stage 9: Output Generation
        stage_start = time.time()
        logger.info("[9/9] Output Generation")
        output_paths = {}
        if output_dir:
            output_paths = self._save_outputs(
                result_pil,
                depth_map,
                quality_metrics,
                input_path,
                Path(output_dir),
                unified_metrics=unified_metrics,
            )
            logger.info(f"  Saved {len(output_paths)} files")
        stage_metrics.append(
            StageMetrics(
                "output_generation",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Build result
        total_duration = (time.time() - start_time) * 1000

        logger.info("=" * 70)
        logger.info("✅ Processing Complete")
        logger.info(f"   Total Time: {total_duration:.0f}ms")
        if quality_metrics:
            logger.info(f"   Quality Score: {quality_metrics.overall_score:.2%}")
        logger.info("=" * 70)

        return ProcessingResult(
            image=result_pil,
            depth_map=depth_map,
            quality_metrics=quality_metrics,
            stage_metrics=stage_metrics,
            total_duration_ms=total_duration,
            iterations=iterations,
            output_paths=output_paths,
            config_used=self.config,
        )

    def _get_or_load_depth_model(self):
        """Lazy-load Depth Anything V2 model.

        Returns:
            Hugging Face depth estimation pipeline, or None if unavailable
        """
        if self._depth_model_initialized:
            return self._depth_model

        try:
            from transformers import pipeline as hf_pipeline

            model_map = {
                "small": "depth-anything/Depth-Anything-V2-Small-hf",
                "base": "depth-anything/Depth-Anything-V2-Base-hf",
                "large": "depth-anything/Depth-Anything-V2-Large-hf",
            }
            model_id = model_map.get(self.config.depth.model_variant, model_map["small"])
            model_revision = resolve_model_lock_revision(
                model_id,
                requested_revision=None,
                strict=self.strict_model_lock,
                context="Rendering4KPipeline(depth_estimation)",
            )
            device_id = 0 if self.device != DeviceType.CPU else -1

            logger.info(f"Loading Depth Anything V2 ({self.config.depth.model_variant})...")
            self._depth_model = hf_pipeline(
                "depth-estimation",
                model=model_id,
                revision=model_revision,
                device=device_id,
            )
            logger.info("✓ Depth Anything V2 loaded")
        except Exception as e:
            logger.warning(f"Depth Anything V2 unavailable: {e}. Using fallback.")
            self._depth_model = None

        self._depth_model_initialized = True
        return self._depth_model

    def _estimate_depth(
        self,
        image: np.ndarray,
        input_path: Path,
    ) -> np.ndarray:
        """
        Estimate depth map using Depth Anything V2 or fallback with caching.

        Args:
            image: RGB image as float32 array
            input_path: Path for cache key

        Returns:
            Depth map as float32 array [0, 1]
        """
        # Check cache
        cache_key = None
        if self.config.depth.cache_enabled:
            cache_key = self._compute_cache_key(image)
            if cache_key in self._depth_cache:
                logger.debug("  Using cached depth map")
                # Move to end to mark as recently used (LRU behavior)
                self._depth_cache.move_to_end(cache_key)
                return self._depth_cache[cache_key]

        # Try Depth Anything V2
        depth_model = self._get_or_load_depth_model()
        if depth_model is not None:
            try:
                image_pil = np_to_pil(image)
                result = depth_model(image_pil)
                # Validate result structure
                if "depth" not in result:
                    logger.warning("Depth model returned unexpected format, using fallback")
                    depth_map = estimate_depth_simple(image)
                else:
                    depth_map = np.array(result["depth"]).astype(np.float32)
                    # Normalize to [0, 1], handling edge case of constant depth
                    depth_range = depth_map.max() - depth_map.min()
                    if depth_range > 1e-8:
                        depth_map = (depth_map - depth_map.min()) / depth_range
                    else:
                        # Constant depth map - set to mid-range
                        depth_map = np.full_like(depth_map, 0.5)
            except Exception as e:
                logger.warning(f"Depth inference failed: {e}")
                depth_map = estimate_depth_simple(image)
        else:
            depth_map = estimate_depth_simple(image)

        # Cache result
        if self.config.depth.cache_enabled and cache_key is not None:
            if len(self._depth_cache) >= self.config.depth.cache_max_size:
                # Remove oldest (least recently used) entry
                self._depth_cache.popitem(last=False)
            self._depth_cache[cache_key] = depth_map

        return depth_map

    def _compute_cache_key(self, image: np.ndarray) -> str:
        """Compute cache key from image content (non-security, non-cryptographic)."""
        # Use SHA-256 for cache key (non-cryptographic; safe for content hashing)
        data = image.tobytes()[:4096]  # First 4KB for speed
        return hashlib.sha256(data).hexdigest()

    def _get_or_load_controlnet_pipe(self):
        """Lazy-load ControlNet pipeline for AI enhancement.

        Returns:
            StableDiffusionControlNetImg2ImgPipeline, or None if unavailable
        """
        if self._controlnet_initialized:
            return self._controlnet_pipe

        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

            logger.info("Loading ControlNet pipeline...")
            controlnets = []
            dtype = torch.float16 if self.device != DeviceType.CPU else torch.float32
            strict_model_lock = self.strict_model_lock

            if self.config.ai_enhancement.use_controlnet:
                canny_model_id = "lllyasviel/sd-controlnet-canny"
                canny_revision = resolve_model_lock_revision(
                    canny_model_id,
                    requested_revision=None,
                    strict=strict_model_lock,
                    context="Rendering4KPipeline(controlnet_canny)",
                )
                controlnets.append(
                    ControlNetModel.from_pretrained(  # nosec B615
                        canny_model_id,
                        revision=canny_revision,
                        torch_dtype=dtype,
                    )
                )
            if self.config.ai_enhancement.use_depth_guidance:
                depth_model_id = "lllyasviel/sd-controlnet-depth"
                depth_revision = resolve_model_lock_revision(
                    depth_model_id,
                    requested_revision=None,
                    strict=strict_model_lock,
                    context="Rendering4KPipeline(controlnet_depth)",
                )
                controlnets.append(
                    ControlNetModel.from_pretrained(  # nosec B615
                        depth_model_id,
                        revision=depth_revision,
                        torch_dtype=dtype,
                    )
                )

            if not controlnets:
                logger.warning("No ControlNet models configured")
                self._controlnet_pipe = None
                self._controlnet_initialized = True
                return None

            base_model_id = "runwayml/stable-diffusion-v1-5"
            base_model_revision = resolve_model_lock_revision(
                base_model_id,
                requested_revision=None,
                strict=strict_model_lock,
                context="Rendering4KPipeline(base_model)",
            )
            self._controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(  # nosec B615
                base_model_id,
                revision=base_model_revision,
                controlnet=controlnets if len(controlnets) > 1 else controlnets[0],
                torch_dtype=dtype,
                safety_checker=None,
            )
            self._controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(self._controlnet_pipe.scheduler.config)

            device_str = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}.get(self.device.value, "cpu")
            self._controlnet_pipe.to(device_str)

            if self.device == DeviceType.CUDA:
                self._controlnet_pipe.enable_attention_slicing()
                self._controlnet_pipe.enable_vae_slicing()

            logger.info("✓ ControlNet pipeline loaded")
        except Exception as e:
            logger.warning(f"ControlNet unavailable: {e}")
            self._controlnet_pipe = None

        self._controlnet_initialized = True
        return self._controlnet_pipe

    def _apply_ai_enhancement(
        self,
        image: Image.Image,
        depth_map: Optional[np.ndarray],
    ) -> Image.Image:
        """Apply ControlNet AI enhancement.

        Args:
            image: PIL Image to enhance
            depth_map: Optional depth map for depth-guided enhancement

        Returns:
            Enhanced PIL Image
        """
        pipe = self._get_or_load_controlnet_pipe()
        if pipe is None or not HAS_CONTROLNET_AUX:
            return image

        try:
            import torch

            control_images = []

            # Generate Canny edge map if ControlNet is enabled
            if self.config.ai_enhancement.use_controlnet and CannyDetector is not None:
                canny = CannyDetector()
                control_images.append(canny(image))

            # Use depth map if depth guidance is enabled
            if self.config.ai_enhancement.use_depth_guidance and depth_map is not None:
                # Convert depth map directly to RGB image for ControlNet
                # Use squeeze() to handle both (H, W) and (H, W, 1) shapes
                depth_uint8 = (depth_map.squeeze() * 255).astype(np.uint8)
                # Stack grayscale to RGB channels directly
                depth_rgb = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=-1)
                depth_pil = Image.fromarray(depth_rgb, mode="RGB").resize(image.size)
                control_images.append(depth_pil)

            if not control_images:
                logger.warning("No control images available for ControlNet")
                return image

            # Use CPU generator for MPS as PyTorch's Generator doesn't support MPS directly
            device_for_gen = "cpu" if self.device == DeviceType.MPS else pipe.device
            generator = torch.Generator(device=device_for_gen).manual_seed(self.config.ai_enhancement.seed)

            result = pipe(
                prompt=self.config.ai_enhancement.prompt,
                negative_prompt=self.config.ai_enhancement.negative_prompt,
                image=image,
                control_image=(control_images if len(control_images) > 1 else control_images[0]),
                num_inference_steps=self.config.ai_enhancement.num_steps,
                guidance_scale=self.config.ai_enhancement.guidance_scale,
                strength=self.config.ai_enhancement.strength,
                generator=generator,
            ).images[0]

            return result
        except Exception as e:
            logger.error(f"ControlNet failed: {e}")
            return image

    def _save_outputs(
        self,
        image: Image.Image,
        depth_map: Optional[np.ndarray],
        quality_metrics: Optional[QualityMetrics],
        input_path: Path,
        output_dir: Path,
        unified_metrics: Optional[UnifiedQualityMetrics] = None,
    ) -> Dict[str, Path]:
        """Save all output files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        outputs = {}

        # Master TIFF (16-bit)
        if self.config.output.master_tiff_16bit and HAS_TIFFFILE:
            tiff_path = output_dir / f"{stem}_MASTER.tiff"
            img_np = pil_to_np(image, to_float=True)
            img_16bit = (np.clip(img_np, 0, 1) * 65535).astype(np.uint16)
            tifffile.imwrite(str(tiff_path), img_16bit, photometric="rgb")
            outputs["master_tiff"] = tiff_path
            logger.info(f"  Master TIFF: {tiff_path.name}")

        # Delivery JPEG
        if self.config.output.delivery_jpeg:
            jpeg_path = output_dir / f"{stem}_DELIVERY.jpg"
            image.save(
                jpeg_path,
                quality=self.config.output.jpeg_quality,
                progressive=self.config.output.jpeg_progressive,
                optimize=True,
            )
            outputs["delivery_jpeg"] = jpeg_path
            logger.info(f"  Delivery JPEG: {jpeg_path.name}")

        # Depth visualization
        if self.config.output.save_depth_visualization and depth_map is not None:
            depth_path = output_dir / f"{stem}_depth.png"
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_pil = Image.fromarray(depth_vis, mode="L")
            depth_pil.save(depth_path)
            outputs["depth_visualization"] = depth_path
            logger.info(f"  Depth Map: {depth_path.name}")

        # Quality report
        if self.config.output.save_quality_report and quality_metrics:
            report_path = output_dir / f"{stem}_quality_report.json"
            report = {
                "input": str(input_path),
                "preset": self.config.name,
                "quality_metrics": quality_metrics.to_dict(),
                "config": asdict(self.config),
            }
            # Include unified metrics if available (RAG-indexable)
            if unified_metrics is not None:
                report["unified_metrics"] = unified_metrics.to_dict()
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)
            outputs["quality_report"] = report_path
            logger.info(f"  Quality Report: {report_path.name}")

        # Save unified metrics as separate RAG document if enabled
        if unified_metrics is not None and self.config.quality_feedback.rag_indexing_enabled:
            rag_path = output_dir / f"{stem}_unified_quality.json"
            with open(rag_path, "w") as f:
                json.dump(
                    unified_metrics.to_rag_document(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                )
            outputs["unified_quality_doc"] = rag_path
            logger.info(f"  Unified Quality Doc: {rag_path.name}")

        return outputs

    def batch_process(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        show_progress: bool = True,
    ) -> List[ProcessingResult]:
        """
        Process multiple images in batch with GPU memory management.

        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            show_progress: Show progress bar

        Returns:
            List of ProcessingResults
        """
        results = []

        # Use tqdm if available, otherwise simple iteration
        if show_progress and HAS_TQDM and tqdm is not None:
            iterator = tqdm(input_paths, desc="Processing")
        else:
            iterator = input_paths
            if show_progress and not HAS_TQDM:
                logger.info(f"Processing {len(input_paths)} images...")

        for i, path in enumerate(iterator):
            try:
                # Check GPU memory before processing (75% threshold is conservative
                # to allow headroom for spikes during inference)
                if not self.memory_manager.check_memory_threshold(0.75):
                    logger.warning("High GPU memory usage, clearing cache...")
                    self.memory_manager.clear_cache()
                    # Clear depth cache when over half full to prevent memory accumulation
                    # while preserving recent entries for potential cache hits
                    if len(self._depth_cache) > self.config.depth.cache_max_size // 2:
                        self._depth_cache.clear()

                if show_progress and not HAS_TQDM:
                    logger.info(f"Processing {i+1}/{len(input_paths)}: {Path(path).name}")
                result = self.process(path, output_dir)
                results.append(result)

                # Periodic cleanup every 5 images to prevent memory fragmentation
                # and ensure consistent performance across batch
                if (i + 1) % 5 == 0:
                    self.memory_manager.clear_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM on {path}, clearing cache and skipping")
                    self.memory_manager.clear_cache()
                    self.clear_cache()
                else:
                    logger.error(f"Failed to process {path}: {e}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")

        # Final cleanup
        self.memory_manager.clear_cache()

        # Print summary
        self._print_batch_summary(results)

        return results

    def _print_batch_summary(self, results: List[ProcessingResult]):
        """Print batch processing summary."""
        if not results:
            logger.warning("No images processed successfully")
            return

        total_time = sum(r.total_duration_ms for r in results)
        avg_time = total_time / len(results)

        quality_scores = [r.quality_score for r in results if r.quality_score > 0]
        avg_quality = np.mean(quality_scores) if quality_scores else 0

        logger.info("\n" + "=" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Images processed: {len(results)}")
        logger.info(f"Total time: {total_time / 1000:.1f}s")
        logger.info(f"Average time per image: {avg_time:.0f}ms")
        logger.info(f"Average quality score: {avg_quality:.2%}")
        logger.info(f"Throughput: {len(results) / (total_time / 3600000):.0f} images/hour")
        logger.info("=" * 60)

    def clear_cache(self):
        """Clear depth cache."""
        self._depth_cache.clear()
        logger.info("Depth cache cleared")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point for the 4K rendering pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-End 4K Rendering Enhancement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with default preset
  python -m transformation_portal.pipelines.rendering_4k_pipeline -i input.jpg -o output/

  # Use luxury estate preset
  python -m transformation_portal.pipelines.rendering_4k_pipeline -i input.jpg -o output/ --preset luxury_estate

  # Batch process directory
  python -m transformation_portal.pipelines.rendering_4k_pipeline -d inputs/ -o outputs/ --preset editorial

  # Custom config from YAML
  python -m transformation_portal.pipelines.rendering_4k_pipeline -i input.jpg --config custom.yaml
        """,
    )

    # Input/Output
    parser.add_argument("-i", "--input", type=Path, help="Input image path")
    parser.add_argument("-d", "--directory", type=Path, help="Batch process directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output_4k"),
        help="Output directory (default: output_4k)",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg,*.png,*.tif,*.tiff",
        help="Glob pattern for batch (default: *.jpg,*.png,*.tif,*.tiff)",
    )

    # Preset selection
    parser.add_argument(
        "--preset",
        choices=list(Rendering4KPipeline.PRESETS.keys()),
        default="default",
        help="Processing preset (default: default)",
    )
    parser.add_argument("--config", type=Path, help="Custom YAML config file")

    # Processing options
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation")
    parser.add_argument("--no-upscale", action="store_true", help="Disable 4K upscaling")
    parser.add_argument(
        "--no-quality-feedback",
        action="store_true",
        help="Disable quality feedback loop",
    )

    # Utility
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show config without processing")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate inputs
    if not args.input and not args.directory:
        parser.error("Must specify either --input or --directory")

    # Create pipeline
    if args.config:
        pipeline = Rendering4KPipeline.from_yaml(args.config)
    else:
        pipeline = Rendering4KPipeline.from_preset(args.preset)

    # Apply CLI overrides
    if args.no_depth:
        pipeline.config.depth.enabled = False
    if args.no_upscale:
        pipeline.config.upscaling.enabled = False
    if args.no_quality_feedback:
        pipeline.config.quality_feedback.enabled = False

    # Dry run
    if args.dry_run:
        logger.info("Configuration:")
        logger.info(json.dumps(asdict(pipeline.config), indent=2, default=str))
        return 0

    # Process
    try:
        if args.directory:
            # Batch processing
            patterns = args.pattern.split(",")
            input_paths = []
            for pattern in patterns:
                input_paths.extend(args.directory.glob(pattern.strip()))
            input_paths = sorted(set(input_paths))

            if not input_paths:
                logger.error(f"No files found matching pattern in {args.directory}")
                return 1

            logger.info(f"Found {len(input_paths)} images to process")
            pipeline.batch_process(input_paths, args.output)
        else:
            # Single image
            pipeline.process(args.input, args.output)

        logger.info("✅ Processing complete!")
        return 0

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
