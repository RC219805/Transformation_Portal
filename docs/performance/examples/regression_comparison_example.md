# Performance Comparison Report

**Baseline:** v2.0.0-post-pr841 (da3, standard)
**Current:** (1 images)
**Environment:** Darwin-25.2.0-arm64, Python 3.11.14, torch 2.10.0, device=mps

## Statistics

| Metric | Baseline | Current | Change | Status |
|--------|----------|---------|--------|--------|
| Mean | 13.89s | 12.43s | -10.5% | ✅ OK |
| Median | 11.82s | 12.43s | +5.2% | ✅ OK |
| p90 | 22.05s | 12.43s | -43.6% | ✅ OK |
| p95 | 30.43s | 12.43s | -59.2% | ✅ OK |
| Min | 8.20s | 12.43s | +51.5% | ✅ OK |
| Max | 30.83s | 12.43s | -59.7% | ✅ OK |
| Success Rate | 100.0% | 100.0% | +0.0% | ✅ OK |

## Recommendation

✅ **OK TO MERGE** - No performance regressions detected.