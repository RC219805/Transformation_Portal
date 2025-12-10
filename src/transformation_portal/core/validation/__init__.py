"""
Validation and reproducibility tracking.

Provides comprehensive reporting for all processing runs.
"""

from .report import ProcessingReport, GitInfo, DeviceInfo, ModelInfo
from .metrics import MetricsComputer, QualityMetrics
from .comparison import BaselineComparator, ComparisonResult

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
