# Fix MaterialsV3 CI Test Failures (PyTorch Dependency)

## Problem
The MaterialsV3 edge case tests were failing in CI with:
```
RuntimeError: PyTorch is required for V2 GPU pipeline. Install torch.
```

All 10 tests in `test_materials_v3_edge_cases.py` failed because the `ci_safe_config` fixture was trying to instantiate `LuxPipelineV2` even when PyTorch wasn't available in the CI environment.

## Root Cause
The fixture was using `pytest.importorskip("torch")` which doesn't properly integrate with pytest's skip mechanism when used in fixtures. The fixture would execute and attempt to import PyTorch-dependent classes before the skip could take effect.

## Solution
✅ Replace `pytest.importorskip()` with a check of the module-level `TORCH_AVAILABLE` flag
✅ Call `pytest.skip()` early in the fixture, before any imports
✅ Ensures the class-level `@pytest.mark.skipif` decorator works correctly

### Changed Code
```python
@pytest.fixture
def ci_safe_config(self, output_dir):
    # Skip the entire fixture if PyTorch is not available
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch required for V2 pipeline")

    config = PipelineConfig(...)
    return config
```

## Test Results

### ✅ Local (with PyTorch installed)
```
13 passed, 1 skipped in 47.96s
```

### ✅ Simulated CI (without PyTorch)
```
✓ Tests skip properly with reason "PyTorch required for V2 pipeline"
✓ No RuntimeError exceptions
```

## Security Status
All security checks continue to pass:
- ✅ `basicsr` package is NOT installed (CVE-2024-27763 mitigated)
- ✅ No vulnerable `basicsr` imports detected
- ✅ Safe upscaling backends used (`torch`, `onnx`)

## Files Changed
- `tests/test_materials_v3_edge_cases.py` - Fixed fixture skip behavior (7 lines changed)
- `docs/guides/MATERIALSV3_CI_FIX.md` - Documentation (78 lines added)

## Expected CI Behavior
When merged, the CI job "Core Tests (Python 3.10)" will:
1. Skip all MaterialsV3 edge case tests gracefully
2. Report tests as "skipped" with proper reason
3. Pass all other tests
4. Pass all security checks

## Checklist
- [x] Tests pass locally
- [x] Skip behavior verified without PyTorch
- [x] Documentation added
- [x] Security checks verified
- [x] No changes to production code
- [x] Minimal, surgical fix

## Related
- Fixes test failures from commit `b6d8684`
- Maintains CVE-2024-27763 mitigation
