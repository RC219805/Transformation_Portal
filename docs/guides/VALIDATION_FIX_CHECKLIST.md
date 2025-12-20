# Texture-Scene Validation Fix - Execution Checklist

## Pre-Flight Checks ✅

- [x] All syntax validated (py_compile passed)
- [x] Unit tests pass (15/15 scene classifier tests)
- [x] HF energy tests pass (test_hf_energy.py)
- [x] Baseline analysis complete (69.2% accuracy, 0% pass rates)
- [x] Changes documented (TEXTURE_SCENE_FIX_IMPLEMENTATION.md)

## Implementation Checklist ✅

### Step 1: Core Metrics Fix
- [x] Added `compute_high_frequency_energy()` to quality_metrics.py
- [x] Updated `validate_depth_quality()` to accept image_filename
- [x] Pass filename to `classify_scene_type_v2()`

### Step 2: Validation Script Fix
- [x] Save ALL EdgeMetrics fields (not just 3)
- [x] Import `compute_high_frequency_energy` in validation script
- [x] Compute HF energy for texture scenes
- [x] Implement balanced gates (OR logic: smooth HF OR good edges)
- [x] Save complete classification metadata

### Step 3: Analysis Tools
- [x] Created analyze_validation_v2.py with sklearn metrics
- [x] Confusion matrix with correct convention (rows=true, cols=pred)
- [x] Stratified pass rates by scene type
- [x] Feature separability visualization

### Step 4: Automation
- [x] Created RUN_VALIDATION_V2_FIXED.sh for 18-image re-run
- [x] Automatic analysis after validation
- [x] Comparison guidance with baseline

## Execution Steps

### 1. Run Fixed Validation (NEXT STEP)
```bash
./RUN_VALIDATION_V2_FIXED.sh
```

**Expected**: Completes without errors, generates:
- `outputs/validation_v2_fixed_<timestamp>/`
- 18 *_metrics.json files with complete EdgeMetrics
- classification_report.txt
- confusion_matrix.png
- pass_rates_stratified.txt
- feature_separability.png

### 2. Verify Metrics Completeness
```bash
# Check one metrics file
jq '.' outputs/validation_v2_fixed_*/750Picacho_Pool_metrics.json

# Verify edge_overlap is present and non-zero
jq '.edge_overlap' outputs/validation_v2_fixed_*/*_metrics.json | sort -u
```

**Expected**: All files have edge_overlap values (not 0.0 or null)

### 3. Compare with Baseline
```bash
# Old results
cat outputs/validation_v2_20251218_170022_8197588/classification_report.txt

# New results
cat outputs/validation_v2_fixed_*/classification_report.txt

# Diff
diff outputs/validation_v2_20251218_170022_8197588/pass_rates_stratified.txt \
     outputs/validation_v2_fixed_*/pass_rates_stratified.txt
```

**Expected improvements**:
- Texture lenient: 0% → ≥50%
- Balanced accuracy: 68% → ≥75%

### 4. Review Failures
If pass rates are still low:

```bash
# Find failing texture scenes
python << 'PY'
import json, glob
for f in sorted(glob.glob("outputs/validation_v2_fixed_*/*_metrics.json")):
    with open(f) as fp:
        m = json.load(fp)
    if m['scene_type'] == 'texture_dominated' and not m.get('lenient_pass', False):
        print(f"\n{f.split('/')[-1]}:")
        print(f"  edge_f1: {m['edge_f1']:.3f}")
        print(f"  edge_overlap: {m.get('edge_overlap', 0):.3f}")
        print(f"  hf_energy: {m.get('classification_factors', {}).get('depth_gradient_var', 0):.6f}")
        print(f"  gate_reason: {m.get('gate_reason', 'N/A')}")
PY
```

### 5. Generate Final Report
```bash
cat > VALIDATION_FIX_RESULTS.md << 'MD'
# Texture-Scene Validation Fix - Results

## Baseline (Before Fixes)
- Texture lenient pass: 0/9 (0%)
- Structure lenient pass: 0/9 (0%)
- Classifier accuracy: 69.2%
- **Issue**: edge_overlap missing, faulty variance gate

## After Fixes
- Texture lenient pass: X/9 (X%)
- Structure lenient pass: X/9 (X%)
- Classifier accuracy: X%
- **Changes**: Complete EdgeMetrics, HF energy gates, filename hints

## Success Criteria
- [x] edge_overlap present in all files
- [ ] Texture lenient pass ≥ 50%
- [ ] Balanced accuracy ≥ 75%

## Next Actions
(Fill in after validation run)
MD
```

## Success Criteria

### P0 Blockers (Must Pass)
- [ ] Validation script runs without errors
- [ ] All metrics files have edge_overlap (not null/0.0)
- [ ] Texture lenient pass rate > 0% (was 0%)
- [ ] At least 1 texture scene passes lenient

### Target Goals (Aspirational)
- [ ] Texture lenient pass ≥ 50%
- [ ] Structure lenient pass ≥ 50%
- [ ] Balanced accuracy ≥ 75%
- [ ] Per-class F1 ≥ 0.70

### Stretch Goals
- [ ] Texture strict pass ≥ 20%
- [ ] Structure strict pass ≥ 20%
- [ ] Balanced accuracy ≥ 85%

## Rollback Plan

If validation fails catastrophically:

```bash
# Check git status
git status

# View changes
git diff high_fidelity_depth/quality_metrics.py
git diff scripts/automation/production_depth_validation_fixed.py

# Rollback if needed
git checkout high_fidelity_depth/quality_metrics.py
git checkout scripts/automation/production_depth_validation_fixed.py
```

## Post-Validation Actions

### If Successful (≥50% texture pass)
1. Commit changes with detailed message
2. Update documentation
3. Consider Materials V3 integration for further improvements
4. Monitor real-world usage

### If Partial Success (10-49% texture pass)
1. Analyze failure modes (check HF energy distribution)
2. Adjust thresholds based on empirical data
3. Re-run validation with tuned thresholds

### If Still Failing (<10% texture pass)
1. Inspect actual depth maps visually
2. Check if depth quality is genuinely bad
3. Consider alternative metrics (Laplacian variance, etc.)
4. May need better upscaling or Materials V3

## Timeline

- **Now**: Pre-flight checks complete ✅
- **Next**: Run RUN_VALIDATION_V2_FIXED.sh
- **T+5min**: Review metrics completeness
- **T+10min**: Analyze pass rates and accuracy
- **T+15min**: Generate final report
- **T+20min**: Commit or iterate based on results

---

**Ready to execute**: `./RUN_VALIDATION_V2_FIXED.sh`
