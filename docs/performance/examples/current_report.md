# Performance Comparison Report

**Baseline:** v2.0.0-post-pr841 (da3, standard)
**Current:** (20 images)
**Environment:** Darwin-25.2.0-arm64, Python 3.11.14, torch 2.10.0, device=mps

## Statistics

| Metric | Baseline | Current | Change | Status |
|--------|----------|---------|--------|--------|
| Mean | 13.89s | 13.89s | +0.0% | ✅ OK |
| Median | 11.82s | 11.82s | +0.0% | ✅ OK |
| p90 | 22.05s | 22.05s | +0.0% | ✅ OK |
| p95 | 30.43s | 30.43s | +0.0% | ✅ OK |
| Min | 8.20s | 8.20s | +0.0% | ✅ OK |
| Max | 30.83s | 30.83s | +0.0% | ✅ OK |
| Success Rate | 100.0% | 100.0% | +0.0% | ✅ OK |

## Recommendation

✅ **OK TO MERGE** - No performance regressions detected.
