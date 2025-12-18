#!/usr/bin/env python3
"""
Depth Quality Validation
=========================

DEPRECATED: Use quality_metrics.py instead.
This module is maintained for backward compatibility only.

All new code should use:
    from high_fidelity_depth.quality_metrics import validate_depth_quality, EdgeMetrics
"""

import logging
import warnings

warnings.warn(
    "high_fidelity_depth.validation is deprecated. "
    "Use high_fidelity_depth.quality_metrics instead.",
    DeprecationWarning,
    stacklevel=2
)

from .quality_metrics import (
    EdgeMetrics,
    validate_depth_quality,
    detect_edges,
    compute_edge_overlap,
    detect_halos
)

logger = logging.getLogger(__name__)

# Re-export for compatibility
__all__ = [
    'EdgeMetrics',
    'validate_depth_quality',
    'detect_edges',
    'compute_edge_overlap',
    'detect_halos'
]
