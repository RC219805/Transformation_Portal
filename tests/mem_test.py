#!/usr/bin/env python3
"""Memory profiling test for CI performance monitoring.

This module provides basic memory profiling tests to track memory usage
during typical operations. Used by the Performance Monitor CI workflow.

Usage:
    python -m memory_profiler tests/mem_test.py
"""
from memory_profiler import profile


@profile
def test_import_core():
    """Test that core imports don't leak memory."""
    import numpy as np  # noqa: F401
    from PIL import Image  # noqa: F401
    return True


@profile
def test_basic_array_operations():
    """Test basic array operations memory usage."""
    import numpy as np
    # Create a moderate-sized array
    arr = np.random.rand(1000, 1000)
    # Perform some operations
    result = arr.mean()
    del arr
    return result


if __name__ == "__main__":
    test_import_core()
    test_basic_array_operations()
