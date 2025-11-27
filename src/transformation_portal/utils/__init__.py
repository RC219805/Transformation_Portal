"""Shared utilities and helper functions.

This package provides common utilities used across the Transformation Portal:
- performance: Performance monitoring, caching, and profiling utilities
- error_handling: Robust error handling and validation utilities
"""

# Lazy imports to avoid loading heavy modules on package import
__all__ = [
    'performance',
    'error_handling',
]
