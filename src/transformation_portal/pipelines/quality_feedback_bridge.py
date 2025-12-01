#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality Feedback Bridge for Rendering4KPipeline

Provides the architectural seam between Rendering4KPipeline._assess_quality()
and the PerceptualQualityScorer. Enables LPIPS-based perceptual quality
scoring while maintaining backward compatibility with heuristic metrics.

Key Features:
- Lazy Loading: LPIPS dependencies load on first use, avoiding import-time failures
- Hybrid Mode: Computes both LPIPS and heuristic metrics simultaneously
- Unified Output: UnifiedQualityMetrics for RAG-indexable document structure
- Callback Architecture: Generates closures for pipeline hook injection

Design Goals:
- Bridge gap between current scoring (~78/100) and target metrics
  (95th percentile perceptual, 98% material fidelity)
- Enable transitional validation before deprecating heuristics
- Support RAG feedback loop for iterative quality improvement

Example:
    >>> from quality_feedback_bridge import QualityFeedbackBridge
    >>> bridge = QualityFeedbackBridge()
    >>> metrics = bridge.assess(enhanced_image, original_image, image_id="img001")
    >>> print(f"Perceptual Score: {metrics.perceptual_composite:.1f}/100")
    >>> print(f"Targets Met: {metrics.targets_met}")

Author: Transformation Portal Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# Lazy Loading Flags
# =============================================================================

# These are set on first use to avoid import-time failures
_LPIPS_AVAILABLE: Optional[bool] = None
_PERCEPTUAL_ASSESSOR_AVAILABLE: Optional[bool] = None
_TORCH_AVAILABLE: Optional[bool] = None


def _check_torch_available() -> bool:
    """Check if PyTorch is available (lazy check)."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _check_lpips_available() -> bool:
    """Check if LPIPS package is available (lazy check)."""
    global _LPIPS_AVAILABLE
    if _LPIPS_AVAILABLE is None:
        try:
            import lpips  # noqa: F401
            _LPIPS_AVAILABLE = True
        except ImportError:
            _LPIPS_AVAILABLE = False
    return _LPIPS_AVAILABLE


def _check_perceptual_assessor_available() -> bool:
    """Check if PerceptualQualityAssessor is available (lazy check)."""
    global _PERCEPTUAL_ASSESSOR_AVAILABLE
    if _PERCEPTUAL_ASSESSOR_AVAILABLE is None:
        try:
            from ..enhancements.perceptual_quality_assessment import PerceptualQualityAssessor  # noqa: F401
            _PERCEPTUAL_ASSESSOR_AVAILABLE = True
        except ImportError:
            _PERCEPTUAL_ASSESSOR_AVAILABLE = False
    return _PERCEPTUAL_ASSESSOR_AVAILABLE


# =============================================================================
# Quality Target Constants
# =============================================================================

@dataclass
class QualityTargets:
    """Target thresholds for luxury real estate visualization."""

    # Perceptual targets
    perceptual_percentile_target: float = 95.0  # 95th percentile
    lpips_threshold_excellent: float = 0.10  # LPIPS < 0.10 = excellent
    lpips_threshold_good: float = 0.20  # LPIPS < 0.20 = good
    lpips_threshold_acceptable: float = 0.30  # LPIPS < 0.30 = acceptable

    # Material fidelity targets
    material_fidelity_target: float = 0.98  # 98% material fidelity

    # Per-material thresholds
    material_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'quartzite': 0.96,
        'oak': 0.95,
        'metal': 0.97,
        'glass': 0.94,
        'stucco': 0.95,
        'water': 0.92,
        'vegetation': 0.90,
        'sky': 0.88,
    })

    # Structural thresholds
    ssim_target: float = 0.92
    ms_ssim_target: float = 0.94

    # Heuristic thresholds (legacy)
    sharpness_target: float = 0.70
    contrast_target: float = 0.60
    colorfulness_target: float = 0.50
    exposure_target: float = 0.70


# =============================================================================
# Unified Quality Metrics
# =============================================================================

@dataclass
class HeuristicMetrics:
    """Traditional heuristic-based quality metrics."""
    sharpness: float = 0.0  # 0-1 (Laplacian variance based)
    contrast: float = 0.0  # 0-1 (luminance std based)
    colorfulness: float = 0.0  # 0-1 (Hasler & Süsstrunk)
    exposure_balance: float = 0.0  # 0-1 (mean luminance based)
    noise_level: float = 0.0  # 0-1 (lower is better)
    overall_score: float = 0.0  # 0-1 (weighted combination)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerceptualMetrics:
    """LPIPS-based perceptual quality metrics."""
    lpips_score: float = 0.0  # 0-1 (lower is better)
    lpips_percentile: float = 0.0  # 0-100 (higher is better)
    ssim_score: float = 0.0  # 0-1 (higher is better)
    ms_ssim_score: float = 0.0  # 0-1 (higher is better)
    niqe_score: float = 0.0  # No-reference quality
    brisque_score: float = 0.0  # No-reference quality
    naturalness_score: float = 0.0  # 0-100
    composite_score: float = 0.0  # 0-100+ (can exceed 100)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MaterialFidelityMetrics:
    """Per-material fidelity metrics."""
    per_material: Dict[str, float] = field(default_factory=dict)
    overall_fidelity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'per_material': self.per_material,
            'overall_fidelity': self.overall_fidelity,
        }


@dataclass
class UnifiedQualityMetrics:
    """
    Unified quality metrics aggregating both LPIPS and heuristic paradigms.

    This structure is designed for RAG indexing, enabling queries that
    correlate code changes with perceptual outcomes.
    """

    # Identification
    image_id: str = ""
    pipeline_config_name: str = ""
    timestamp: str = ""

    # Heuristic metrics (legacy, for transitional validation)
    heuristic: HeuristicMetrics = field(default_factory=HeuristicMetrics)

    # Perceptual metrics (LPIPS-based)
    perceptual: PerceptualMetrics = field(default_factory=PerceptualMetrics)

    # Material-specific fidelity
    material_fidelity: MaterialFidelityMetrics = field(default_factory=MaterialFidelityMetrics)

    # Composite scores
    perceptual_composite: float = 0.0  # 0-100+ primary score
    heuristic_composite: float = 0.0  # 0-100 legacy score
    hybrid_score: float = 0.0  # Weighted combination during transition

    # Target achievement
    targets_met: Dict[str, bool] = field(default_factory=dict)
    targets_summary: str = ""

    # Processing metadata
    processing_time_ms: float = 0.0
    lpips_available: bool = False
    hybrid_mode: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'image_id': self.image_id,
            'pipeline_config_name': self.pipeline_config_name,
            'timestamp': self.timestamp,
            'heuristic': self.heuristic.to_dict(),
            'perceptual': self.perceptual.to_dict(),
            'material_fidelity': self.material_fidelity.to_dict(),
            'scores': {
                'perceptual_composite': round(self.perceptual_composite, 2),
                'heuristic_composite': round(self.heuristic_composite, 2),
                'hybrid_score': round(self.hybrid_score, 2),
            },
            'targets_met': self.targets_met,
            'targets_summary': self.targets_summary,
            'metadata': {
                'processing_time_ms': round(self.processing_time_ms, 1),
                'lpips_available': self.lpips_available,
                'hybrid_mode': self.hybrid_mode,
                'warnings': self.warnings,
            },
        }

    def to_rag_document(self) -> Dict[str, Any]:
        """Convert to RAG-indexable document format."""
        doc = self.to_dict()
        doc['_type'] = 'unified_quality_metrics'
        doc['_version'] = '1.0.0'
        doc['_indexed_at'] = datetime.utcnow().isoformat()
        return doc


# =============================================================================
# Quality Feedback Bridge
# =============================================================================

class QualityFeedbackBridge:
    """
    Bridge between Rendering4KPipeline and PerceptualQualityScorer.

    Provides unified quality assessment that:
    - Computes both LPIPS and heuristic metrics
    - Supports hybrid mode for transitional validation
    - Generates RAG-indexable quality documents
    - Handles graceful fallback when LPIPS unavailable

    Example:
        >>> bridge = QualityFeedbackBridge(hybrid_mode=True)
        >>> metrics = bridge.assess(enhanced, original, "image_001")
        >>> print(f"Score: {metrics.perceptual_composite:.1f}/100")
    """

    def __init__(
        self,
        targets: Optional[QualityTargets] = None,
        hybrid_mode: bool = True,
        lpips_network: str = 'alex',
        enable_material_fidelity: bool = True,
        rag_callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize Quality Feedback Bridge.

        Args:
            targets: Quality target thresholds
            hybrid_mode: Compute both LPIPS and heuristic metrics
            lpips_network: LPIPS network ('alex', 'vgg', 'squeeze')
            enable_material_fidelity: Compute per-material fidelity
            rag_callback: Callback for RAG indexing (receives document dict)
        """
        self.targets = targets or QualityTargets()
        self.hybrid_mode = hybrid_mode
        self.lpips_network = lpips_network
        self.enable_material_fidelity = enable_material_fidelity
        self.rag_callback = rag_callback

        # Lazy-loaded components
        self._perceptual_assessor = None
        self._lpips_checked = False
        self._lpips_available = False

        logger.info(
            f"Initialized QualityFeedbackBridge "
            f"(hybrid_mode={hybrid_mode}, network={lpips_network})"
        )

    def _ensure_perceptual_assessor(self) -> bool:
        """
        Ensure perceptual assessor is loaded (lazy loading).

        Returns:
            True if assessor is available, False otherwise
        """
        if self._perceptual_assessor is not None:
            return True

        if self._lpips_checked:
            return self._lpips_available

        self._lpips_checked = True

        # Check dependencies
        if not _check_torch_available():
            logger.info("PyTorch not available, using heuristic-only mode")
            self._lpips_available = False
            return False

        if not _check_perceptual_assessor_available():
            logger.info(
                "PerceptualQualityAssessor not available, using heuristic-only mode"
            )
            self._lpips_available = False
            return False

        # Try to initialize assessor
        try:
            from ..enhancements.perceptual_quality_assessment import (
                PerceptualQualityAssessor,
            )
            self._perceptual_assessor = PerceptualQualityAssessor(
                use_lpips_package=_check_lpips_available()
            )
            self._lpips_available = True
            logger.info("LPIPS-based perceptual assessor initialized successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize perceptual assessor: {e}")
            self._lpips_available = False
            return False

    def assess(
        self,
        enhanced: Union[np.ndarray, Image.Image],
        original: Optional[Union[np.ndarray, Image.Image]] = None,
        image_id: str = "",
        pipeline_config_name: str = "",
    ) -> UnifiedQualityMetrics:
        """
        Assess image quality using unified metrics.

        Args:
            enhanced: Enhanced image (array or PIL Image)
            original: Original image for comparison (optional)
            image_id: Unique identifier for the image
            pipeline_config_name: Name of pipeline config used

        Returns:
            UnifiedQualityMetrics with all scores
        """
        start_time = time.time()

        metrics = UnifiedQualityMetrics(
            image_id=image_id,
            pipeline_config_name=pipeline_config_name,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Convert inputs to numpy arrays
        enhanced_np = self._to_numpy(enhanced)
        original_np = self._to_numpy(original) if original is not None else None

        # Always compute heuristic metrics (fast, no dependencies)
        metrics.heuristic = self._compute_heuristic_metrics(enhanced_np)
        metrics.heuristic_composite = metrics.heuristic.overall_score * 100

        # Compute LPIPS perceptual metrics if available
        metrics.lpips_available = self._ensure_perceptual_assessor()

        if metrics.lpips_available and original_np is not None:
            try:
                perceptual_result = self._compute_perceptual_metrics(
                    enhanced_np, original_np
                )
                metrics.perceptual = perceptual_result['perceptual']
                metrics.material_fidelity = perceptual_result['material_fidelity']
                metrics.perceptual_composite = metrics.perceptual.composite_score
            except Exception as e:
                logger.warning(f"Perceptual assessment failed: {e}")
                metrics.warnings.append(f"Perceptual assessment failed: {str(e)}")
                metrics.lpips_available = False

        # Compute hybrid score
        metrics.hybrid_mode = self.hybrid_mode
        if self.hybrid_mode:
            metrics.hybrid_score = self._compute_hybrid_score(metrics)
        else:
            metrics.hybrid_score = (
                metrics.perceptual_composite if metrics.lpips_available
                else metrics.heuristic_composite
            )

        # Check targets
        metrics.targets_met = self._check_targets(metrics)
        metrics.targets_summary = self._summarize_targets(metrics.targets_met)

        # Record timing
        metrics.processing_time_ms = (time.time() - start_time) * 1000

        # Invoke RAG callback if configured
        if self.rag_callback is not None:
            try:
                self.rag_callback(metrics.to_rag_document())
            except Exception as e:
                logger.warning(f"RAG callback failed: {e}")
                metrics.warnings.append(f"RAG callback failed: {str(e)}")

        return metrics

    def _to_numpy(self, image: Union[np.ndarray, Image.Image, None]) -> Optional[np.ndarray]:
        """Convert image to numpy array [0, 1] float32."""
        if image is None:
            return None

        if isinstance(image, np.ndarray):
            arr = image.astype(np.float32)
            if arr.size > 0 and arr.max() > 1.0:
                arr = arr / 255.0
            return arr

        if isinstance(image, Image.Image):
            arr = np.array(image.convert('RGB')).astype(np.float32) / 255.0
            return arr

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _compute_heuristic_metrics(self, image: np.ndarray) -> HeuristicMetrics:
        """
        Compute traditional heuristic-based quality metrics.

        These metrics are fast to compute and don't require ML dependencies.
        """
        metrics = HeuristicMetrics()

        # Sharpness (Laplacian variance)
        metrics.sharpness = self._compute_sharpness(image)

        # Contrast (luminance std)
        metrics.contrast = self._compute_contrast(image)

        # Colorfulness (Hasler & Süsstrunk)
        metrics.colorfulness = self._compute_colorfulness(image)

        # Exposure balance
        metrics.exposure_balance = self._compute_exposure_balance(image)

        # Noise estimation
        metrics.noise_level = self._estimate_noise(image)

        # Overall weighted score
        weights = {
            'sharpness': 0.25,
            'contrast': 0.20,
            'colorfulness': 0.20,
            'exposure': 0.20,
            'noise': 0.15,
        }

        score = (
            weights['sharpness'] * metrics.sharpness +
            weights['contrast'] * metrics.contrast +
            weights['colorfulness'] * metrics.colorfulness +
            weights['exposure'] * metrics.exposure_balance -
            weights['noise'] * metrics.noise_level
        )
        metrics.overall_score = float(np.clip(score, 0, 1))

        return metrics

    def _compute_sharpness(self, image: np.ndarray) -> float:
        """Compute sharpness using Laplacian variance."""
        gray = np.mean(image, axis=2)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        # Simple convolution fallback
        # Try scipy for convolution, fall back to manual implementation
        try:
            from scipy.ndimage import convolve
            laplacian = convolve(gray, kernel, mode='reflect')
        except ImportError:
            # Vectorized convolution fallback (faster than loops, but slower than scipy)
            # For 3x3 Laplacian kernel, use slicing and weighted sums
            # WARNING: For large 4K+ images, consider installing scipy for 10-100x speedup
            h, w = gray.shape
            pad_h, pad_w = 1, 1
            padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

            # Vectorized 3x3 Laplacian convolution using slicing
            laplacian = (
                kernel[0, 1] * padded[0:h, 1:w+1] +
                kernel[1, 0] * padded[1:h+1, 0:w] +
                kernel[1, 1] * padded[1:h+1, 1:w+1] +
                kernel[1, 2] * padded[1:h+1, 2:w+2] +
                kernel[2, 1] * padded[2:h+2, 1:w+1]
            )

        variance = float(np.var(laplacian))
        return float(np.clip(variance * 50, 0, 1))

    def _compute_contrast(self, image: np.ndarray) -> float:
        """Compute contrast using standard deviation of luminance."""
        lum = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        std = float(np.std(lum))
        return float(np.clip(std * 3, 0, 1))

    def _compute_colorfulness(self, image: np.ndarray) -> float:
        """Compute colorfulness (Hasler & Süsstrunk 2003)."""
        r, g, b = image[..., 0], image[..., 1], image[..., 2]

        rg = r - g
        yb = 0.5 * (r + g) - b

        std_rg = float(np.std(rg))
        std_yb = float(np.std(yb))
        mean_rg = float(np.mean(rg))
        mean_yb = float(np.mean(yb))

        std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)

        colorfulness = std_root + 0.3 * mean_root
        return float(np.clip(colorfulness * 2, 0, 1))

    def _compute_exposure_balance(self, image: np.ndarray) -> float:
        """Compute exposure balance score."""
        lum = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        mean_lum = float(np.mean(lum))

        optimal = 0.45
        deviation = abs(mean_lum - optimal)

        return float(np.clip(1.0 - deviation * 2, 0, 1))

    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level using median absolute deviation."""
        gray = np.mean(image, axis=2)

        # Try scipy median filter first, fall back to PIL
        try:
            from scipy.ndimage import median_filter
            smoothed = median_filter(gray, size=3)
        except ImportError:
            # Fallback to PIL-based smoothing
            from PIL import ImageFilter
            img_uint8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='L')
            smoothed_pil = pil_img.filter(ImageFilter.MedianFilter(3))
            smoothed = np.array(smoothed_pil).astype(np.float32) / 255.0

        noise = np.abs(gray - smoothed)
        mad = float(np.median(noise))

        return float(np.clip(mad * 20, 0, 1))

    def _compute_perceptual_metrics(
        self,
        enhanced: np.ndarray,
        original: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute LPIPS-based perceptual metrics using PerceptualQualityAssessor.
        """
        if self._perceptual_assessor is None:
            return {
                'perceptual': PerceptualMetrics(),
                'material_fidelity': MaterialFidelityMetrics(),
            }

        # Convert to PIL for assessor
        enhanced_pil = Image.fromarray(
            (np.clip(enhanced, 0, 1) * 255).astype(np.uint8), mode='RGB'
        )
        original_pil = Image.fromarray(
            (np.clip(original, 0, 1) * 255).astype(np.uint8), mode='RGB'
        )

        # Run assessment
        report = self._perceptual_assessor.assess(
            enhanced=enhanced_pil,
            reference=original_pil,
            compute_material_fidelity=self.enable_material_fidelity,
        )

        perceptual = PerceptualMetrics(
            lpips_score=report.lpips_score,
            lpips_percentile=report.lpips_percentile,
            ssim_score=report.ssim_score,
            ms_ssim_score=report.ms_ssim_score,
            niqe_score=report.niqe_score,
            brisque_score=report.brisque_score,
            naturalness_score=report.naturalness_score,
            composite_score=report.composite_score,
        )

        material_fidelity = MaterialFidelityMetrics(
            per_material=report.material_fidelity,
            overall_fidelity=report.overall_material_fidelity,
        )

        return {
            'perceptual': perceptual,
            'material_fidelity': material_fidelity,
        }

    def _compute_hybrid_score(self, metrics: UnifiedQualityMetrics) -> float:
        """
        Compute hybrid score combining LPIPS and heuristic metrics.

        During transition period, this enables validation of LPIPS
        correlation with heuristics before full migration.
        """
        if not metrics.lpips_available:
            return metrics.heuristic_composite

        # Weighted combination (favor perceptual when available)
        perceptual_weight = 0.7
        heuristic_weight = 0.3

        return (
            perceptual_weight * metrics.perceptual_composite +
            heuristic_weight * metrics.heuristic_composite
        )

    def _check_targets(self, metrics: UnifiedQualityMetrics) -> Dict[str, bool]:
        """Check which quality targets are met."""
        targets_met = {}

        # Heuristic targets
        targets_met['heuristic_sharpness'] = (
            metrics.heuristic.sharpness >= self.targets.sharpness_target
        )
        targets_met['heuristic_contrast'] = (
            metrics.heuristic.contrast >= self.targets.contrast_target
        )
        targets_met['heuristic_colorfulness'] = (
            metrics.heuristic.colorfulness >= self.targets.colorfulness_target
        )
        targets_met['heuristic_exposure'] = (
            metrics.heuristic.exposure_balance >= self.targets.exposure_target
        )

        # Perceptual targets (if available)
        if metrics.lpips_available:
            targets_met['perceptual_95th'] = (
                metrics.perceptual.lpips_percentile >= self.targets.perceptual_percentile_target
            )
            targets_met['lpips_excellent'] = (
                metrics.perceptual.lpips_score <= self.targets.lpips_threshold_excellent
            )
            targets_met['ssim'] = (
                metrics.perceptual.ssim_score >= self.targets.ssim_target
            )

            # Material fidelity target
            targets_met['material_98pct'] = (
                metrics.material_fidelity.overall_fidelity >= self.targets.material_fidelity_target
            )

        return targets_met

    def _summarize_targets(self, targets_met: Dict[str, bool]) -> str:
        """Generate human-readable summary of target achievement."""
        met_count = sum(1 for v in targets_met.values() if v)
        total_count = len(targets_met)

        if met_count == total_count:
            return f"✓ All {total_count} targets met"
        elif met_count >= total_count * 0.7:
            return f"○ {met_count}/{total_count} targets met (good)"
        elif met_count >= total_count * 0.5:
            return f"○ {met_count}/{total_count} targets met (acceptable)"
        else:
            return f"✗ {met_count}/{total_count} targets met (needs improvement)"


# =============================================================================
# Pipeline Integration Helpers
# =============================================================================

def create_quality_callback_for_pipeline(
    pipeline_config_name: str,
    rag_index_path: Optional[str] = None,
) -> Callable[[UnifiedQualityMetrics], None]:
    """
    Create a quality callback closure for pipeline integration.

    The callback logs quality metrics and optionally indexes them
    to the RAG system for feedback loop analysis.

    Args:
        pipeline_config_name: Name of pipeline configuration
        rag_index_path: Optional path to RAG index

    Returns:
        Callback function that receives UnifiedQualityMetrics
    """
    def callback(metrics: UnifiedQualityMetrics) -> None:
        # Log quality summary
        logger.info(
            f"Quality Assessment [{metrics.image_id}]: "
            f"Perceptual={metrics.perceptual_composite:.1f}/100, "
            f"Heuristic={metrics.heuristic_composite:.1f}/100, "
            f"Hybrid={metrics.hybrid_score:.1f}/100"
        )
        logger.info(f"  {metrics.targets_summary}")

        # RAG indexing (if configured)
        if rag_index_path:
            try:
                index_quality_metrics_to_rag(metrics, rag_index_path)
            except Exception as e:
                logger.warning(f"Failed to index quality metrics: {e}")

    return callback


def index_quality_metrics_to_rag(
    metrics: UnifiedQualityMetrics,
    rag_index_path: str,
) -> bool:
    """
    Index quality metrics to RAG system.

    Args:
        metrics: Unified quality metrics to index
        rag_index_path: Path to RAG index directory

    Returns:
        True if indexing succeeded, False otherwise
    """
    try:
        index_dir = Path(rag_index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Create document filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"quality_{metrics.image_id}_{timestamp}.json"
        filepath = index_dir / filename

        # Write document
        document = metrics.to_rag_document()
        with open(filepath, 'w') as f:
            json.dump(document, f, indent=2)

        logger.debug(f"Indexed quality metrics to {filepath}")
        return True

    except Exception as e:
        logger.warning(f"Failed to index quality metrics: {e}")
        return False


def create_rag_indexing_callback(
    index_path: Optional[str] = None,
) -> Callable[[Dict], None]:
    """
    Create a RAG indexing callback for the QualityFeedbackBridge.

    Args:
        index_path: Path to RAG index directory

    Returns:
        Callback function for RAG indexing
    """
    def callback(document: Dict) -> None:
        if index_path is None:
            logger.debug("RAG indexing disabled (no index path)")
            return

        try:
            index_dir = Path(index_path)
            index_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            image_id = document.get('image_id', 'unknown')
            filename = f"unified_quality_{image_id}_{timestamp}.json"
            filepath = index_dir / filename

            with open(filepath, 'w') as f:
                json.dump(document, f, indent=2)

            logger.debug(f"RAG indexed: {filepath}")

        except Exception as e:
            logger.warning(f"RAG indexing failed: {e}")

    return callback
