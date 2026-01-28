"""
Baseline Comparison Logic.

Compares current execution results against established baselines
to detect regression or drift.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .metrics import MetricsComputer, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    passed: bool
    drift_score: float  # How much it deviates
    metrics: QualityMetrics
    message: str


class BaselineComparator:
    """
    Validates results against a 'Golden Master' baseline.
    """
    
    def __init__(
        self, 
        min_psnr: float = 30.0, 
        min_ssim: float = 0.90
    ):
        self.min_psnr = min_psnr
        self.min_ssim = min_ssim

    def compare(
        self, 
        current_image, 
        baseline_image
    ) -> ComparisonResult:
        """
        Check if current image matches baseline standards.
        """
        metrics = MetricsComputer.compute(current_image, baseline_image)
        
        passed = True
        failures = []
        
        if metrics.psnr < self.min_psnr:
            passed = False
            failures.append(f"PSNR {metrics.psnr:.2f} < {self.min_psnr}")
            
        if metrics.ssim > 0 and metrics.ssim < self.min_ssim:
            passed = False
            failures.append(f"SSIM {metrics.ssim:.3f} < {self.min_ssim}")
            
        msg = "Regression detected: " + ", ".join(failures) if not passed else "Within tolerance."
        
        # Drift score (inverse of SSIM, 0 is perfect match)
        drift = 1.0 - metrics.ssim
        
        return ComparisonResult(passed, drift, metrics, msg)
