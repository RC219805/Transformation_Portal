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
    detect_halos,
    compute_edge_alignment_corr,
)


def compute_edge_alignment(edges_pred, edges_ref):
    """
    Computes edge alignment score between predicted and reference edges.
    
    Wrapper for backward compatibility - delegates to compute_edge_alignment_corr.
    
    Args:
        edges_pred: Binary edge map from prediction
        edges_ref: Binary edge map from reference
        
    Returns:
        Edge alignment score (0-1, higher is better)
    """
    import numpy as np
    
    if edges_pred.shape != edges_ref.shape:
        raise ValueError("Edge maps must have same shape")
    
    intersection = (edges_pred & edges_ref).sum()
    union = (edges_pred | edges_ref).sum()
    
    return float(intersection / max(union, 1))


logger = logging.getLogger(__name__)

# Re-export for compatibility
__all__ = [
    'EdgeMetrics',
    'validate_depth_quality',
    'detect_edges',
    'compute_edge_overlap',
    'detect_halos',
    'compute_edge_alignment',
    'compute_edge_alignment_corr',
]
