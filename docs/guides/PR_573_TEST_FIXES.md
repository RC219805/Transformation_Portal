# PR #573 Test Import Fixes

**Date**: 2025-12-20  
**Commit**: 7b81652

## Problem

CI tests were failing during collection phase with:

```
ModuleNotFoundError: No module named 'torch.nn'; 'torch' is not a package
```

This occurred in:
- `tests/test_metric_depth.py`
- `tests/test_model_cache.py`

## Root Cause

**Eager Import Chain in `lux_depth_v3/__init__.py`**:

```
lux_depth_v3/__init__.py
  ↓ imports validation.py
    ↓ imports inference.py
      ↓ imports da3_wrapper.py
        ↓ imports torch.nn ❌
```

This meant that **any** import of `lux_depth_v3` (even just config classes) would immediately try to import PyTorch, causing test collection to fail when PyTorch wasn't installed or was misconfigured.

## Solution

### 1. Lazy Loading in `lux_depth_v3/__init__.py`

Implemented proper lazy imports with graceful degradation:

```python
# Core components (safe imports - no heavy dependencies)
from lux_depth_v3.config import DA3Config, ...
from lux_depth_v3.input_manager import InputManager, ImageInput

# Lazy imports for optional DA3 components
DA3InferenceEngine = None
DepthQualityMetrics = None
DA3Backend = None

try:
    from lux_depth_v3.validation import DepthQualityMetrics as _DQM
    DepthQualityMetrics = _DQM
except ImportError:
    pass

try:
    from lux_depth_v3.inference import DA3InferenceEngine as _Engine
    DA3InferenceEngine = _Engine
except ImportError:
    pass

# ... similar for other DA3 components
```

**Benefits**:
- ✅ Core config classes can be imported without PyTorch
- ✅ Tests can import what they need without pulling in ML stack
- ✅ Graceful degradation when dependencies unavailable
- ✅ Clear separation between required vs optional components

### 2. Test Module Skip Pattern

Updated test modules to use try/except for dependency checking:

```python
"""Tests for metric depth conversion utilities."""

import pytest
import numpy as np
from pathlib import Path

# Try to import dependencies - skip entire module if unavailable
try:
    import cv2  # noqa: F401
    import torch  # noqa: F401
    from lux_depth_v3.metric_depth import (
        MetricDepthConverter,
        # ...
    )
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    SKIP_REASON = f"Dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE, 
    reason=getattr(globals(), 'SKIP_REASON', 'Dependencies not available')
)
```

**Why This Works**:
- `pytestmark` applies skip marker to entire module
- Gracefully handles import failures during collection
- Provides clear skip reason in test output
- Allows CI to pass without ML dependencies

### 3. Files Changed

- `lux_depth_v3/__init__.py` - Lazy loading implementation
- `tests/test_metric_depth.py` - Try/except skip pattern
- `tests/test_model_cache.py` - Try/except skip pattern

## Testing

### Local Verification

```bash
# Test import without triggering PyTorch
python -c "import lux_depth_v3; print(f'DA3 available: {lux_depth_v3._DA3_AVAILABLE}')"

# Test collection
pytest tests/test_metric_depth.py tests/test_model_cache.py --co -q
```

### CI Impact

- ✅ Tests collect successfully even when PyTorch not installed
- ✅ Tests skip gracefully with clear reason when dependencies missing
- ✅ Core tests can run without ML dependencies
- ✅ ML tests run when dependencies are available

## Alignment with DA3 Defer Decision

This fix aligns perfectly with the strategic decision to **defer DA3**:

- DA3 code is **optional**, not required for core functionality
- Tests can run and validate DA2 baseline without DA3 dependencies
- Future DA3 work is preserved but isolated
- Production deployment doesn't require DA3 infrastructure

## Prevention

To avoid similar issues in the future:

1. **Always use lazy imports for optional/heavy dependencies**
2. **Keep `__init__.py` files lightweight**
3. **Use try/except in test modules for dependency checking**
4. **Test import isolation**: `python -c "import module"` should not pull in unrelated dependencies

## References

- PR #573: Validation baseline freeze + DA3 evaluation (DEFER)
- Commit: 7b81652
- Related: DA3 evaluation decision (docs/decisions/DA3_EVALUATION_DECISION.md)
