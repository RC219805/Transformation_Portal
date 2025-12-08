"""Quality validation framework for Lux Depth V2.

This module provides production-grade quality validation capabilities:
- Synthetic reference mode (degrade + compare)
- Real-world mode (no-reference metrics)
- Baseline comparison (vs Topaz/Adobe/etc.)
- Multiple metric categories (fidelity, perceptual, aesthetic)
- Batch validation and regression testing support
"""

from .quality_validator import QualityValidator, ValidationReport, ComparisonReport

__all__ = [
    "QualityValidator",
    "ValidationReport",
    "ComparisonReport",
]
