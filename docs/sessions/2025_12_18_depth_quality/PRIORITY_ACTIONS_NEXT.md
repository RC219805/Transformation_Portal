# Priority Actions - Next Steps

**Date**: December 17, 2025, 7:35 PM
**Status**: Quality system fixed, ready for validation

## ✅ COMPLETED

1. **Atomic JSON write system** - prevents truncated output
2. **Unified metric implementation** - single source of truth
3. **Shift-tolerant primary metric** - Edge F1 instead of correlation
4. **Calibrated thresholds** - based on empirical data
5. **Seam detection** - catches tile artifacts
6. **Edge artifact gates** - prevents false positives
7. **Regression test suite** - all tests passing

## 🎯 IMMEDIATE NEXT STEPS (in order)

### Step 1: Update Isolation Tests (30 min)

**File**: `high_fidelity_depth/isolation_tests.py`

**Change required**: Replace old metric computation with unified `validate_depth_quality()`

**Before**:
```python
# Old inconsistent metrics
metrics = old_compute_metrics(...)
```

**After**:
```python
from quality_metrics import validate_depth_quality

metrics = validate_depth_quality(rgb, depth, dilation=3)
```

**Verification**: Run isolation tests and confirm JSON is parseable and metrics are consistent

---

### Step 2: Run Comprehensive A/B on Real Images (5 min)

**Kitchen**:
```bash
python high_fidelity_depth/comprehensive_validation.py \
  --rgb input_images/750_Picacho/Source_TIFFs/750Picacho_Kitchen_16bit.tiff \
  --depth outputs/high_fidelity_validation/kitchen_depth.tiff \
  --output-dir validation_kitchen/
```

**Pool**:
```bash
python high_fidelity_depth/comprehensive_validation.py \
  --rgb input_images/750_Picacho/Source_TIFFs/750Picacho_Pool_16bit.tiff \
  --depth outputs/high_fidelity_validation/pool_depth.tiff \
  --output-dir validation_pool/
```

**Expected output**:
- `validation_report.json` (atomic, guaranteed parseable)
- `edge_overlay.png` (visual diagnostic)
- Clear pass/fail with root cause if failed

---

### Step 3: Root Cause Analysis Based on Validation Results (2-4 hrs)

**If seam validation fails** → Scale reconciliation bug
- Check affine matching in overlap regions
- Verify per-tile normalization is NOT happening

**If edge F1 < baseline** → Refinement bug
- Check guided filter eps (too high?)
- Check edge snapping mask (inverted?)
- Check bypass mode tensor selection

**If edge count ratio > 3.0** → Artifact generation
- Check for global sharpening (remove it)
- Check edge snapping is AND-gated (RGB ∧ depth)
- Check CLAHE is only on low-frequency

---

## 📊 Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Metric system | ✅ FIXED | Atomic write, unified, calibrated |
| Regression tests | ✅ PASS | All tests green |
| Isolation tests | ⚠️ NEEDS UPDATE | Use new unified metrics |
| A/B validation | ⚠️ READY TO RUN | Tools ready, awaiting execution |
| Root cause fix | ⏸️ PENDING | Depends on A/B results |

---

## 🎪 Key Insight from Review

> "The terminal update shows the pipeline is no longer fundamentally broken.
> But you still have a quality-system integrity problem: invalid/truncated JSON
> and mismatched metrics are undermining every decision.
> Fix reporting + metric coherence first; only then will edge snapping +
> guided filter tuning be meaningful."

**Status**: ✅ Reporting + metric coherence FIXED

**Next**: Run trustworthy A/B validation to identify remaining issues

---

## 📝 Documentation Updates Needed

- [ ] Update `HIGH_FIDELITY_DEPTH_IMPLEMENTATION_PLAN.md` - reference new metric system
- [ ] Update `DEPTH_PIPELINE_FINAL_DIAGNOSIS.md` - remove "edge gradient 0.09" narrative
- [ ] Archive contradictory "research-grade" documentation

---

## ⚡ Quick Reference

**Run regression tests**:
```bash
cd high_fidelity_depth/
python test_metrics_atomic.py
python test_comprehensive_validation.py
```

**Run comprehensive validation**:
```bash
python high_fidelity_depth/comprehensive_validation.py \
  --rgb <input_rgb> \
  --depth <depth_output> \
  --output-dir <results_dir>
```

**Check JSON is valid**:
```bash
python -m json.tool <results_dir>/validation_report.json
```

---

**Bottom line**: Quality system is now trustworthy. Proceed with validation on real images.
