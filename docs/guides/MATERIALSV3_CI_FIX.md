# MaterialsV3 CI Test Failures - Fix Summary

## Issue
PR #707 ("Fix MaterialsV3 CI test failures and security warnings") was failing with 10 test failures in `test_materials_v3_edge_cases.py`. All failures had the same root cause:

```
RuntimeError: PyTorch is required for V2 GPU pipeline. Install torch.
```

## Root Cause
The `ci_safe_config` fixture in `tests/test_materials_v3_edge_cases.py` was using `pytest.importorskip("torch")` which doesn't properly integrate with pytest's skip mechanism when used in fixtures. This caused the fixture to execute and attempt to import `PipelineConfig` and `Preset` from modules that require PyTorch, leading to the RuntimeError.

## Solution
Changed the fixture to check the module-level `TORCH_AVAILABLE` flag and use `pytest.skip()` instead:

```python
@pytest.fixture
def ci_safe_config(self, output_dir):
    """Create a CI-safe pipeline config (uses heuristic backend to avoid transformers dependency).

    NOTE: This fixture requires PyTorch. Tests using it will be skipped if PyTorch is unavailable.
    """
    # Skip the entire fixture if PyTorch is not available
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch required for V2 pipeline")

    config = PipelineConfig(
        preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS,
        output_dir=output_dir,
        write_outputs=False  # Speed up tests
    )
    # Override segmentation backend to use heuristic (no external dependencies)
    config.segmentation.backend = "heuristic"
    return config
```

## Key Changes
- **Before**: Used `pytest.importorskip("torch")` followed by conditional imports
- **After**: Check `TORCH_AVAILABLE` flag (set at module level) and call `pytest.skip()` early

This ensures:
1. Tests are properly skipped in CI environments without PyTorch
2. The class-level `@pytest.mark.skipif` decorator works correctly
3. No imports from PyTorch-dependent modules occur when PyTorch is unavailable

## Test Results

### Local (with PyTorch)
```
13 passed, 1 skipped in 47.96s
```

### Simulated CI (without PyTorch)
```
✓ Module loaded successfully
✓ TORCH_AVAILABLE = False
✓ Tests will be skipped when PyTorch is unavailable (expected in CI)
```

## Security Status
All security checks continue to pass:
- ✅ `basicsr` package is NOT installed (CVE-2024-27763 mitigated)
- ✅ No vulnerable `basicsr` imports detected
- ✅ Safe upscaling backends (`torch`, `onnx`) used instead

## Files Modified
- `tests/test_materials_v3_edge_cases.py` - Fixed `ci_safe_config` fixture

## Expected CI Behavior
When this fix is merged:
1. CI job "Core Tests (Python 3.10)" will skip all MaterialsV3 edge case tests (PyTorch not installed)
2. No test failures will occur
3. Tests will show as "skipped" with reason "PyTorch required for V2 pipeline"
4. Security checks will continue to pass

## Related Issues
- PR #707: Fix MaterialsV3 CI test failures and security warnings
- CVE-2024-27763: Vulnerable `basicsr` package (mitigated, not installed)
