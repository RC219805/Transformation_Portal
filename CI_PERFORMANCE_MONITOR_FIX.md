# Performance Monitor Workflow Fix

## Issue
The Performance Monitor workflow failed with:
```
ImportError: Hypothesis is required for these tests. Install via `pip install hypothesis`.
```

## Root Cause
The workflow was only installing minimal dependencies (`memory-profiler` and `pytest-benchmark`) but not the full CI test dependencies from `requirements-ci.txt`, which includes:
- `hypothesis>=6,<7` (required by `tests/__init__.py`)
- `pytest>=8,<10`
- `pytest-cov>=4.0,<8`
- Other test infrastructure

Additionally, the workflow was using `--benchmark-only` flag expecting pytest-benchmark style tests, but the actual performance tests are regular pytest functions (not using the `benchmark` fixture).

## Fix Applied

### 1. Install Full Test Dependencies
**Before:**
```yaml
pip install memory-profiler pytest-benchmark
```

**After:**
```yaml
pip install -r requirements-ci.txt
pip install memory-profiler pytest-benchmark
```

This ensures all test infrastructure (including hypothesis) is available.

### 2. Remove `--benchmark-only` Flag
**Before:**
```yaml
python -m pytest tests/ -k performance \
  --benchmark-only \
  --benchmark-json=benchmark-results.json \
  --benchmark-sort=mean
```

**After:**
```yaml
python -m pytest tests/ -k performance -v \
  --tb=short \
  --maxfail=5
```

This runs all performance tests (not just benchmark-style tests). ML-heavy tests are automatically skipped via `pytest.importorskip("torch")`.

### 3. Simplified Artifact Upload
Removed references to non-existent `benchmark-results.json` since we're not using pytest-benchmark fixtures.

## Verification

The fix ensures:
1. ✅ `hypothesis` is installed before running tests
2. ✅ Performance tests run without requiring pytest-benchmark fixtures
3. ✅ ML tests are gracefully skipped when torch is not available
4. ✅ No false failures from missing benchmark results

## Impact
- **Zero regression risk**: Only affects the Performance Monitor workflow
- **Backward compatible**: Existing performance tests continue to work
- **Clear failure messages**: `--maxfail=5` prevents cascading failures

## Testing
Run locally to verify:
```bash
pip install -r requirements-ci.txt
pytest tests/ -k performance -v --tb=short
```

Expected: Performance tests pass (ML tests may be skipped if torch not installed).
