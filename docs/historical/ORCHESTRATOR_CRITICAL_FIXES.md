# Orchestrator Critical Bug Fixes - Implementation Report

**Date:** 2026-02-05
**Status:** COMPLETED
**Priority:** P1 - Critical Pipeline Correctness

## Summary

Fixed critical orchestrator bugs that broke trust in automation, CI dashboards, and governance gates. Implemented structured result handling, startup dependency reporting, stage-aware concurrency, and made CoreML optional.

## Problems Fixed

### 1. ✅ CRITICAL: Summary Counters Bug (P1)

**Problem:** Orchestrator summary showed "Successful: 0 / Skipped: 0 / Failed: 0" despite processing images.

**Root Cause:** Status value mismatch between orchestrator and CLI
- Orchestrator returns `status: "ok"` for successful processing
- CLI was checking for `status: "success"` (wrong value)
- All successful results were uncounted

**Fix:**
- Updated `src/transformation_portal/lux_depth_v3/__main__.py` line 344
- Changed counter to check for `status: "ok"` instead of `status: "success"`
- Added comment documenting the correct status value

**Impact:** Summary counters now accurately reflect processing results.

**Tests:** Added `tests/test_orchestrator_counters.py` with comprehensive counter verification:
- Test orchestrator returns `status: "ok"` for success
- Test orchestrator returns `status: "skipped"` for skipped images
- Test orchestrator returns `status: "error"` for failures
- Test batch processing accumulates counters correctly
- Test CLI counts `status: "ok"` as successful

### 2. ✅ CoreML Optional Dependencies

**Problem:** CoreML import warnings at startup for optional acceleration
- `scikit-learn 1.8.0 vs coremltools max 1.5.1` warnings
- `torch 2.10.0 untested with coremltools` warnings
- Import-time warnings for optional features

**Fix:**
- Moved `coremltools` from `ml` extra to new `coreml` optional extra in `pyproject.toml`
- Removed import-time warnings from `inference.py`
- Lazy import pattern already in place - warnings were the only issue
- Users can now install CoreML acceleration separately: `pip install "transformation_portal[coreml]"`

**Impact:** Clean startup, no warnings for optional dependencies.

### 3. ✅ Startup Dependency Report

**Problem:** Vague warnings about missing dependencies
- "scikit-image not available" without context
- "using NumPy fallback (30-50% slower)" without explanation
- No visibility into HF_TOKEN status

**Fix:**
- Added `_log_dependency_status()` function in `orchestrator.py`
- Reports status of: torch, transformers, coremltools, scikit-image, numba, HF_TOKEN
- Actionable guidance for each dependency
- DEBUG level for optional features, INFO for essential features
- Called once during orchestrator initialization

**Example Output:**
```
DEBUG: torch 2.10.0 available
DEBUG: transformers 4.35.0 available
DEBUG: coremltools not available (optional). Install: pip install coremltools
DEBUG: scikit-image 0.21.0 available
DEBUG: numba 0.58.0 available - performance optimizations enabled
DEBUG: HF_TOKEN not set - using unauthenticated downloads (rate limits apply, slower warm starts)
DEBUG:   Set HF_TOKEN for faster downloads: export HF_TOKEN=<your_token>
```

**Impact:** Clear visibility into pipeline capabilities at startup.

### 4. ✅ Stage-Aware Concurrency Policy

**Problem:** 19 images, 15 workers, MPS bottleneck
- DA3/MPS inference + PBR + V2 subprocesses contending for GPU memory
- No concurrency limits for GPU/MPS backends
- Conservative parallelism needed for VRAM management

**Fix:**
- Implemented stage-aware concurrency in orchestrator initialization
- GPU/MPS devices: limit to `min(2, cpu_count())` workers
- CPU devices: use `max(1, cpu_count() - 1)` workers (original behavior)
- Logged concurrency strategy for transparency

**Code:**
```python
if config.depth_device in ("mps", "cuda"):
    # GPU backends: conservative concurrency to avoid VRAM contention
    self.max_workers = min(2, cpu_count())
    logger.debug(f"GPU/MPS device detected - limiting workers to {self.max_workers} for VRAM management")
else:
    # CPU backend: moderate parallelism for I/O-bound operations
    self.max_workers = config.max_parallel_workers or max(1, cpu_count() - 1)
```

**Impact:** Prevents VRAM contention on GPU devices, maintains parallelism on CPU.

## Files Changed

### Core Pipeline Files
1. `src/transformation_portal/lux_depth_v3/__main__.py`
   - Fixed summary counter to check for `status: "ok"`

2. `src/transformation_portal/lux_depth_v3/orchestrator.py`
   - Added `_log_dependency_status()` function
   - Integrated dependency report into `__init__`
   - Implemented stage-aware concurrency policy

3. `src/transformation_portal/lux_depth_v3/inference.py`
   - Removed import-time warnings for torch, transformers, coremltools
   - Kept lazy import pattern, added `None` assignment on ImportError

### Configuration
4. `pyproject.toml`
   - Moved `coremltools>=7.0` from `ml` extra to new `coreml` extra
   - Users can install: `pip install "transformation_portal[coreml]"`

### Tests
5. `tests/test_orchestrator_counters.py` (NEW)
   - Comprehensive counter correctness tests
   - Status value verification (ok/skipped/error)
   - Batch accumulation tests
   - CLI counter logic tests
   - Dependency report tests

## Test Results

All tests passing:
```
tests/test_orchestrator_counters.py::TestOrchestratorCounters::test_enhance_image_returns_ok_status PASSED
tests/test_orchestrator_counters.py::TestOrchestratorCounters::test_enhance_image_returns_skipped_status PASSED
tests/test_orchestrator_counters.py::TestOrchestratorCounters::test_enhance_image_returns_error_status PASSED
tests/test_orchestrator_counters.py::TestOrchestratorCounters::test_batch_processing_accumulates_counters PASSED
tests/test_orchestrator_counters.py::TestCLICounters::test_cli_counts_ok_as_successful PASSED
tests/test_orchestrator_counters.py::TestDependencyReport::test_dependency_status_logged PASSED
tests/test_orchestrator_counters.py::TestDependencyReport::test_dependency_report_returns_status PASSED

7 passed in 2.86s
```

Existing orchestrator tests still pass:
```
tests/test_orchestrator_improvements.py - 29 passed in 2.94s
```

## Architecture Compliance

### Meets Governance Requirements
- ✅ Mechanical enforcement: Status values verified by tests
- ✅ Deterministic behavior: Counters accurately reflect processing
- ✅ Fail-fast validation: Tests catch status value mismatches
- ✅ Backward compatibility: Orchestrator API unchanged

### Follows Codebase Philosophy
- ✅ Minimal changes: Surgical fixes to specific bugs
- ✅ Contract stability: No changes to public interfaces
- ✅ Enforced by tests: Counter correctness verified mechanically
- ✅ Clear documentation: Comments explain status values

### Security Posture
- ✅ No new dependencies added
- ✅ CoreML made optional (reduces supply chain surface)
- ✅ Import-time warnings removed (cleaner startup, less noise)
- ✅ HF_TOKEN status reported (security awareness)

## Production Impact

### Before
```
Processing complete:
  Successful: 0
  Skipped: 0
  Failed: 0
```
(19 images processed successfully, counters broken)

### After
```
Processing complete:
  Successful: 19
  Skipped: 0
  Failed: 0
```
(Counters accurately reflect reality)

### Startup Improvements
- No more CoreML version warnings (unless explicitly installed)
- Clear dependency status report at DEBUG level
- Actionable guidance for missing optional features
- HF_TOKEN status visibility

### Concurrency Improvements
- MPS/CUDA: max 2 workers (prevents VRAM contention)
- CPU: original behavior (cpu_count - 1)
- Logged strategy for transparency

## Migration Notes

### For Users
- **No action required** for existing workflows
- Optional: Install CoreML acceleration with `pip install "transformation_portal[coreml]"`
- Optional: Set `HF_TOKEN` for faster model downloads

### For CI/CD
- Summary counters now accurate - can trust automation gates
- Tests verify counter correctness - regression protected
- Dependency report helps debug missing features

### For Developers
- Status values documented: use `"ok"`, `"skipped"`, `"error"`
- Dependency status function available for debugging
- Stage-aware concurrency policy for new backends

## Outstanding Items

None. All critical bugs addressed.

## Conclusion

This fix restores trust in pipeline automation by ensuring summary counters accurately reflect processing results. The structured result pattern, startup dependency report, and stage-aware concurrency policy make the pipeline "auditable, quiet when optional stuff is absent, and honest in its summaries—the holy trinity of production pipelines."

**Status:** Ready for production deployment.
