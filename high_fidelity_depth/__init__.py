"""
High-Fidelity Depth Pipeline
=============================

Production-grade tile-based depth inference with:
- Scale reconciliation (fixes tiling seams)
- Edge-preserving fusion
- Isolation testing
- A/B validation

This module addresses critical bugs identified in TILING_BUG_IDENTIFIED.md
"""

from .depth_estimator import HighFidelityDepthEstimator, DepthConfig
from .isolation_tests import run_isolation_tests
from .validation import validate_depth_quality, EdgeMetrics

__version__ = "1.0.0"
__all__ = [
    "HighFidelityDepthEstimator",
    "DepthConfig",
    "run_isolation_tests",
    "validate_depth_quality",
    "EdgeMetrics",
]
