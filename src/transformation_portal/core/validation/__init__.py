"""
Validation and reproducibility tracking.

Provides comprehensive reporting for all processing runs.
Compatibility note: retained as an internal/shared helper surface with
direct smoke coverage, but it currently has no production imports.
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
