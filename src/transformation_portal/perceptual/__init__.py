"""
Phase 2: Perceptual Baseline Calibration

This module provides perceptual quality assessment and baseline calibration
for the six source images (pool, bedrooms, bathroom, aerial, kitchen, great room).

Establishes empirical foundation for measuring enhancement trajectories beyond
conventional photorealistic limitations.

Key Components:
- Image Loader: Load and preprocess source images
- Quality Metrics: LPIPS, FID, BRISQUE, NIQE, PSNR, SSIM
- Baseline Assessment: Establish quality baselines
- Enhancement Tracking: Measure improvement trajectories
- Visualization: Report generation and analysis

Usage:
    from transformation_portal.perceptual import PerceptualBaseline

    # Initialize with substrate
    baseline = PerceptualBaseline(substrate)

    # Calibrate with source images
    results = baseline.calibrate(image_paths)

    # Get baseline metrics
    metrics = baseline.get_baseline_metrics()
"""

from .analyzer import AnalysisResult, PerceptualAnalyzer
from .baseline import BaselineConfig, PerceptualBaseline
from .image_loader import ImageLoader, ImageMetadata
from .metrics import (
    MetricType,
    PerceptualScore,
    QualityMetrics,
    compute_brisque,
    compute_fid,
    compute_lpips,
    compute_niqe,
    compute_psnr,
    compute_ssim,
)
from .synthetic_viewer import ACUScore, JourneyMoment, SyntheticViewer
from .tracker import EnhancementTracker, TrajectoryPoint

__all__ = [
    "PerceptualBaseline",
    "BaselineConfig",
    "QualityMetrics",
    "MetricType",
    "PerceptualScore",
    "ImageLoader",
    "ImageMetadata",
    "PerceptualAnalyzer",
    "AnalysisResult",
    "SyntheticViewer",
    "ACUScore",
    "JourneyMoment",
    "EnhancementTracker",
    "TrajectoryPoint",
    "compute_lpips",
    "compute_fid",
    "compute_brisque",
    "compute_niqe",
    "compute_psnr",
    "compute_ssim",
]

__version__ = "1.0.0"
