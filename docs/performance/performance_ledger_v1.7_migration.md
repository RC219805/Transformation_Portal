# Performance Ledger v1.7 Migration Guide

**Document Version:** 1.0
**Date:** 2026-02-05
**Target Audience:** Users migrating from v1.0 to v1.7

---

## Executive Summary

Performance Ledger v1.7 introduces significant enhancements while maintaining backward compatibility through deprecation warnings and shims. This guide helps you migrate smoothly.

**Key Changes:**
- Optional NumPy dependency (pure Python fallback)
- Bootstrap confidence intervals for mean
- Expanded exit codes (4 codes instead of 2)
- Enhanced input validation
- Backend mismatch detection

---

## Breaking Changes and Migration Path

### 1. CLI Flag Rename: `--version` → `--baseline-version`

**v1.0 Usage:**
```bash
python tools/performance_ledger.py \
    --manifests-dir ./manifests \
    --output ./baseline.json \
    --version v2.0.0
```

**v1.7 Usage (Preferred):**
```bash
python tools/performance_ledger.py \
    --manifests-dir ./manifests \
    --output ./baseline.json \
    --baseline-version v2.0.0
```

**Migration Strategy:**
- **Phase 1 (Now - Month 2):** Both flags work, `--version` logs deprecation warning
- **Phase 2 (Month 2-6):** Update all scripts to use `--baseline-version`
- **Phase 3 (v2.0+):** `--version` removed entirely

**Action Required:**
- Update CI/CD scripts to use `--baseline-version`
- Update documentation and examples
- Search codebase for `--version` usage: `grep -r "performance_ledger.*--version"`

---

### 2. Exit Code Expansion

**v1.0 Exit Codes:**
```
0 = Success (no regression)
1 = Regression detected
```

**v1.7 Exit Codes:**
```
0 = Success (no regression or potential regression without --strict)
1 = Significant regression detected
2 = Backend mismatch between baseline and current
3 = Insufficient data for comparison
```

**Impact on CI/CD:**

❌ **Old CI Script (May Miss Regressions):**
```bash
if python tools/performance_ledger.py --baseline ... --compare ...; then
    echo "Performance OK"
else
    echo "Performance degraded"
fi
```

✅ **New CI Script (Recommended):**
```bash
python tools/performance_ledger.py \
    --baseline baseline.json \
    --compare ./current_manifests \
    --output report.md \
    --strict

exit_code=$?

case $exit_code in
    0)
        echo "✅ Performance OK"
        ;;
    1)
        echo "❌ Performance regression detected"
        exit 1
        ;;
    2)
        echo "⚠️  Backend mismatch - results not comparable"
        exit 1
        ;;
    3)
        echo "⚠️  Insufficient data for comparison"
        exit 1
        ;;
esac
```

**Key Decision: Use `--strict` in CI**

Without `--strict`:
- Exit 0 if change is below threshold (even if slightly slower)
- Good for exploratory/local development

With `--strict` (recommended for CI):
- Fail on any potential regression
- Prevents gradual performance erosion

---

### 3. JSON Schema Extensions

**v1.0 Baseline Schema:**
```json
{
  "version": "v2.0.0",
  "backend": "da3",
  "quality_tier": "standard",
  "environment": { ... },
  "statistics": {
    "count": 100,
    "mean_sec": 10.0,
    "median_sec": 9.5,
    "p90_sec": 12.0,
    "p95_sec": 13.0,
    "min_sec": 8.0,
    "max_sec": 15.0,
    "success_rate": 1.0
  },
  "captured_at": "2026-01-01T00:00:00Z"
}
```

**v1.7 Baseline Schema (Additive):**
```json
{
  "version": "v2.0.0",
  "backend": "da3",
  "quality_tier": "standard",
  "environment": { ... },
  "statistics": {
    "count": 100,
    "mean_sec": 10.0,
    "median_sec": 9.5,
    "p90_sec": 12.0,
    "p95_sec": 13.0,
    "min_sec": 8.0,
    "max_sec": 15.0,
    "success_rate": 1.0,
    "std_sec": 1.5,                      // NEW
    "bootstrap_ci_95_lower": 9.7,        // NEW
    "bootstrap_ci_95_upper": 10.3        // NEW
  },
  "captured_at": "2026-01-01T00:00:00Z",
  "backend_compliance": {                 // NEW (only if mismatches exist)
    "expected": "da3",
    "actual": ["da3"],
    "mismatch_count": 0
  }
}
```

**Compatibility:**
- ✅ v1.7 can load v1.0 baselines (backward compatible)
- ✅ v1.0 can load v1.7 baselines (forward compatible, ignores unknown fields)
- ⚠️ v1.7 baselines are larger (2-3x) due to bootstrap CI

**Migration Actions:**
- No action required for existing baselines
- Optionally re-capture baselines to include v1.7 features
- Monitor baseline file sizes in version control

---

## New Features and Usage

### 4. Bootstrap Confidence Intervals

**What It Does:**
Computes 95% confidence interval for mean using bootstrap resampling.

**Usage:**
```bash
# Enabled by default (1000 iterations)
python tools/performance_ledger.py \
    --manifests-dir ./manifests \
    --output ./baseline.json

# Custom iterations
python tools/performance_ledger.py \
    --manifests-dir ./manifests \
    --output ./baseline.json \
    --bootstrap-iterations 5000

# Disable bootstrap (faster)
python tools/performance_ledger.py \
    --manifests-dir ./manifests \
    --output ./baseline.json \
    --no-bootstrap
```

**Performance Impact:**
- 100 iterations: ~10ms overhead
- 1000 iterations (default): ~50-100ms overhead
- 5000 iterations: ~500ms overhead

**When to Use:**
- ✅ Capturing production baselines: Use default (1000)
- ✅ Quick experiments: Use `--no-bootstrap`
- ❌ Large datasets (>1000 samples): Consider lower iterations

---

### 5. Backend Mismatch Detection

**What It Does:**
Detects when comparison manifests use a different backend than baseline.

**Example:**
```bash
# Baseline captured with da3
python tools/performance_ledger.py \
    --manifests-dir ./manifests_da3 \
    --output ./baseline_da3.json \
    --backend da3

# Compare with depth-pro manifests -> Exit code 2
python tools/performance_ledger.py \
    --baseline ./baseline_da3.json \
    --compare ./manifests_depthpro \
    --output ./report.md

# Output:
# ERROR: Backend mismatch: 100.0% samples differ from baseline 'da3'
# Exit code: 2
```

**Backend Aliases (Normalized):**
- `da3`, `depth-anything-v3`, `depth_anything_v3` → `da3`
- `depth-pro`, `depth_pro`, `depthpro` → `depth-pro`

---

### 6. Input Validation Bounds

**v1.7 enforces resource limits to prevent DoS:**

| Parameter | Limit | Reason |
|-----------|-------|--------|
| `--bootstrap-iterations` | max 10,000 | Prevent excessive CPU usage |
| `--bootstrap-iterations` | min 0 | Logical constraint |
| Comparison samples | min 3 | Statistical validity |

**Examples:**
```bash
# ❌ Rejected (exceeds max)
python tools/performance_ledger.py \
    --manifests-dir ./manifests \
    --output ./baseline.json \
    --bootstrap-iterations 50000

# ERROR: Bootstrap iterations exceeds maximum (10000)
# Exit code: 3

# ✅ Accepted
python tools/performance_ledger.py \
    --manifests-dir ./manifests \
    --output ./baseline.json \
    --bootstrap-iterations 5000
```

---

## NumPy Optional Dependency

### Pure Python Fallback

**v1.0:** NumPy required (hard dependency)

**v1.7:** NumPy optional, pure Python fallback

**Performance Impact:**

| Operation | NumPy | Pure Python | Slowdown |
|-----------|-------|-------------|----------|
| Mean | ~0.1ms | ~5ms | 50x |
| Percentile | ~0.5ms | ~20ms | 40x |
| Std Dev | ~0.2ms | ~10ms | 50x |
| Full Stats (n=50) | ~2ms | ~100ms | 50x |

**Recommendation:**
- ✅ **Keep NumPy installed** for production use
- ⚠️ Pure Python acceptable for:
  - CI environments without ML dependencies
  - Small datasets (< 20 samples)
  - Quick prototyping

**Detection:**
```python
from tools.performance_ledger import HAS_NUMPY

if HAS_NUMPY:
    print("Using NumPy (fast)")
else:
    print("Using pure Python (50x slower)")
```

---

## Migration Checklist

### Pre-Migration

- [ ] Audit current usage: `grep -r "performance_ledger" .`
- [ ] Identify all CI/CD scripts using the tool
- [ ] Back up existing baseline files
- [ ] Review custom integrations (e.g., dashboards parsing JSON)

### Phase 1: Update Scripts (Week 1-2)

- [ ] Replace `--version` with `--baseline-version`
- [ ] Add `--strict` to CI regression checks
- [ ] Update exit code handling in CI scripts
- [ ] Test in staging environment

### Phase 2: Validate (Week 3-4)

- [ ] Run side-by-side comparison (v1.0 vs v1.7)
- [ ] Verify exit codes match expectations
- [ ] Check baseline file sizes (should grow 2-3x)
- [ ] Confirm NumPy is installed in CI (or accept pure Python slowdown)

### Phase 3: Production Rollout (Month 2)

- [ ] Deploy to production CI/CD
- [ ] Monitor for unexpected failures
- [ ] Update documentation and examples
- [ ] Train team on new features

### Phase 4: Cleanup (Month 3-6)

- [ ] Remove `--version` usage from all scripts
- [ ] Re-capture all baselines with v1.7 features
- [ ] Archive v1.0 backup

---

## Troubleshooting

### Issue: Deprecation warnings in CI logs

**Symptom:**
```
WARNING: --version flag will be removed in v2.0. Use --baseline-version instead.
```

**Solution:**
Replace `--version` with `--baseline-version` in scripts.

---

### Issue: Exit code 2 (Backend mismatch)

**Symptom:**
```
ERROR: Backend mismatch: 100.0% samples differ from baseline 'da3'
```

**Solution:**
- Ensure baseline and comparison use same backend
- Check manifest `depth.model` field
- Verify `--backend` flag matches actual backend

---

### Issue: Exit code 3 (Insufficient data)

**Symptom:**
```
ERROR: Insufficient data: need at least 3 samples, got 2
```

**Solution:**
- Increase manifest count (min 3 samples)
- Check for failed processing (empty manifests)
- Verify `--manifests-dir` contains valid JSONs

---

### Issue: Slow performance without NumPy

**Symptom:**
Tool takes 50x longer than expected.

**Solution:**
```bash
# Check if NumPy is available
python -c "import numpy; print('NumPy available')"

# If not installed
pip install numpy

# Or accept slowdown for small datasets
```

---

### Issue: Bootstrap CI takes too long

**Symptom:**
Baseline capture takes minutes instead of seconds.

**Solution:**
```bash
# Reduce iterations
--bootstrap-iterations 500

# Or disable entirely
--no-bootstrap
```

---

## Rollback Plan

If v1.7 causes issues:

1. **Restore v1.0:**
   ```bash
   cp archive/scripts/performance_ledger_v1_0_backup.py tools/performance_ledger.py
   ```

2. **Revert CI scripts:** Remove `--strict`, restore `--version` flag

3. **Report issue:** Include exit code, error message, sample manifests

---

## FAQ

**Q: Do I need to re-capture all baselines?**
A: No, v1.7 loads v1.0 baselines. Re-capture only if you want bootstrap CI.

**Q: Will v1.0 load v1.7 baselines?**
A: Yes, it ignores unknown fields. But you lose v1.7 features.

**Q: Should I use `--strict` in CI?**
A: Yes, recommended. Prevents gradual performance erosion.

**Q: Can I use v1.7 without NumPy?**
A: Yes, but expect 50x slowdown. Install NumPy for production.

**Q: What if I need > 10,000 bootstrap iterations?**
A: Not supported (DoS prevention). Use offline analysis instead.

**Q: How do I test backend mismatch detection?**
A: Create test manifests with different `depth.model` values.

---

## Support

For issues or questions:
1. Check this migration guide
2. Review [ADR-023](../architecture/ADR-023-performance-ledger.md)
3. Run tests: `pytest tests/test_performance_ledger*.py -v`
4. Escalate to Transformation Portal Architect

---

**Document Status:** ✅ Complete
**Last Updated:** 2026-02-05
**Next Review:** v2.0.0 release
