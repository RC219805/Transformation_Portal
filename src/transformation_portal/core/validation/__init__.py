"""
Validation and reproducibility tracking.

Provides comprehensive reporting for all processing runs.
"""

from .comparison import BaselineComparator, ComparisonResult
from .metrics import MetricsComputer, QualityMetrics
from .report import DeviceInfo, GitInfo, ModelInfo, ProcessingReport

__all__ = [
    "ProcessingReport",
    "GitInfo",
    "DeviceInfo",
    "ModelInfo",
    "MetricsComputer",
    "QualityMetrics",
    "BaselineComparator",
    "ComparisonResult",
]
