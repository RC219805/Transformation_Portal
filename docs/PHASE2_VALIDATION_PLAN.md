# Phase 2 Validation Plan

**Status**: ✅ Infrastructure Complete  
**Date**: December 13, 2025  
**Next Step**: Execute validation runs

---

## Executive Summary

Phase 2 validation infrastructure is now in place to test all Phase 2 features (CLIP classification, lighting detection, auto-preset selection) as a cohesive whole before proceeding to EfficientSAM V3.

This document outlines the validation approach, benchmark dataset, and success criteria.

---

## Validation Infrastructure

### 1. Benchmark Dataset

**Location**: `assets/phase2_bench/`

**Images** (symlinked from `input_images/750_Picacho/Ultimate_TIFFs_Base/`):

| Image | Scene Type | Expected Classification | Quality Tiers | Materials |
|-------|------------|-------------------------|---------------|-----------|
| `750Picacho_Kitchen_Ultimate.tif` | Interior Kitchen | interior_luxury_apex_quality | Standard/Max/APEX | Wood, stone, metal |
| `750Picacho_PrimaryBedroom_Ultimate.tif` | Interior Bedroom | interior_luxury_apex_quality | APEX | Textiles, wood, glass |
| `750Picacho_PrimaryBathroom_Ultimate.tif` | Interior Bathroom | interior_luxury_max_quality | APEX | Tile, glass, metal |
| `750Picacho_Pool_Ultimate.tif` | Exterior Pool | exterior_pool_apex_quality | Standard/APEX | Water, stone, vegetation |
| `750Picacho_Aerial_Ultimate.tif` | Exterior Facade | exterior_showcase | Standard | Roofing, hardscape |

**Rationale**: These 5 images stress different aspects of Phase 2:
- Multiple interior scene types (kitchen, bedroom, bathroom)
- Exterior water/pool handling
- Various lighting conditions (daylight, twilight, mixed)
- Different material combinations

### 2. Integration Tests

**Location**: `tests/integration/test_phase2_end_to_end.py`

**Tests** (7 total, 6 passing + 1 manual):
1. ✅ `test_auto_preset_interior_classification` - CLIP scene detection
2. ✅ `test_auto_preset_selection_returns_valid_preset` - Preset mapping
3. ✅ `test_preset_selector_quality_tier_mapping` - Tier differentiation
4. ✅ `test_scene_classification_confidence_structure` - Output structure
5. ✅ `test_preset_recommendation_structure` - Recommendation format
6. ⏸️ `test_full_pipeline_with_auto_preset` - Manual validation only
7. ✅ `test_phase2_imports` - Module import validation

**Run command**:
```bash
pytest tests/integration/test_phase2_end_to_end.py -v
```

### 3. Benchmark Matrix Runner

**Location**: `scripts/run_phase2_bench_matrix.sh`

**Test Matrix**:

#### Interior Scenes
- Kitchen: Standard, Max, APEX presets (explicit)
- Bedroom: APEX preset
- Bathroom: APEX preset
- Kitchen: Auto-preset (APEX tier)
- Bedroom: Auto-preset (Max tier)

#### Exterior Scenes
- Pool: APEX preset
- Pool: Interior preset (control - should work but suboptimal)
- Aerial: Standard preset
- Pool: Auto-preset (APEX tier)
- Aerial: Auto-preset (Standard tier)

**Total**: 11-14 test cases (depending on `--quick` mode)

**Run commands**:
```bash
# Full matrix (all tiers)
./scripts/run_phase2_bench_matrix.sh

# Quick mode (APEX only, faster)
./scripts/run_phase2_bench_matrix.sh --quick

# Custom output directory
./scripts/run_phase2_bench_matrix.sh --output-dir /path/to/output
```

---

## Validation Procedure

### Step 1: Run Integration Tests ✅

**Status**: Complete  
**Result**: 6/6 tests passing (1 manual test skipped)

```bash
pytest tests/integration/test_phase2_end_to_end.py -v
```

**Expected**: All 6 active tests pass

### Step 2: Run Benchmark Matrix (Recommended)

**Status**: Ready to execute  
**Estimated Time**: 
- Quick mode: ~30-45 minutes (5 APEX runs)
- Full mode: ~60-90 minutes (11-14 runs)

```bash
# Recommended: Quick mode first
./scripts/run_phase2_bench_matrix.sh --quick
```

**What to validate**:

1. **Logs Check** (in each `outputs/phase2_bench_matrix/*/pipeline.log`):
   - CLIP logs scene classification with >0.5 confidence
   - Lighting detector reports plausible time of day
   - No CLIP download warnings (models should be cached)
   - No unexpected error fallbacks

2. **Outputs Present**:
   - Each run generates output images (PNG/TIFF)
   - APEX runs produce high-bit masters
   - Marketing versions created where appropriate

3. **Auto-Preset Accuracy**:
   - Kitchen → `interior_luxury_apex_quality` (APEX tier)
   - Bedroom → `interior_luxury_max_quality` (Max tier)
   - Pool → `exterior_pool_apex_quality` (APEX tier)
   - Aerial → `exterior_showcase` (Standard tier)

### Step 3: Spot-Check Quality Metrics

For **2-3 key images** (Kitchen APEX, Pool APEX, one interior):

```bash
# Use existing metrics pipeline
python scripts/compare_depth_quality.py \
  --master input_images/750_Picacho/Ultimate_TIFFs_Base/750Picacho_Kitchen_Ultimate.tif \
  --upscaled outputs/phase2_bench_matrix/750Picacho_Kitchen_Ultimate_interior_luxury_apex_quality/output.tif \
  --output-dir validation_metrics/
```

**Check**:
- Color accuracy within APEX thresholds (< 0.06)
- Luma accuracy within APEX thresholds (< 0.06)
- Phase 2 overhead < 500ms (CLIP + lighting)

### Step 4: CI Feature Flag Validation

**Status**: Not yet run (requires GitHub Actions trigger)

**Test 3 modes**:

1. **Default (Phase 2 enabled, no benchmarks)**:
   ```bash
   gh workflow run ci-consolidated.yml
   ```
   - Verify: Phase 2 tests run, benchmark job skipped

2. **Phase 2 disabled**:
   ```bash
   gh workflow run ci-consolidated.yml -f enable_phase2_features=false
   ```
   - Verify: Phase 2 tests filtered, no CLIP downloads

3. **Benchmark regression enabled**:
   ```bash
   gh workflow run ci-consolidated.yml -f run_benchmark_regression=true
   ```
   - Verify: Benchmark job runs, artifacts uploaded

### Step 5: Document Results

After validation runs complete, create:

**File**: `lux_depth_v2/PHASE2_VALIDATION_RESULTS.md`

**Content**:
- Benchmark dataset description
- Test matrix results (pass/fail for each case)
- Phase 2 overhead measurements (CLIP + lighting times)
- Any edge cases discovered
- Recommendations for EfficientSAM priorities

---

## Success Criteria

### Minimum Bar (Required to Proceed)

- [ ] Integration tests: 6/6 passing
- [ ] Benchmark quick mode: 5/5 passing
- [ ] Auto-preset accuracy: 100% correct (4/4 matches expected)
- [ ] No Phase 2 crashes or exceptions
- [ ] CLIP overhead < 500ms per image
- [ ] At least 1 APEX run meets quality thresholds

### Ideal State (Desired)

- [ ] Full benchmark matrix: 11+/11+ passing
- [ ] All APEX runs meet quality thresholds
- [ ] Lighting detection: 100% plausible (no "unknown" fallbacks)
- [ ] CI feature flags: All 3 modes green
- [ ] Phase 2 overhead < 400ms average
- [ ] Zero warnings in logs

---

## Next Steps After Validation

### If Validation Passes

1. **Document baseline**: Create `PHASE2_VALIDATION_RESULTS.md`
2. **Proceed to EfficientSAM V3**: Start multi-session implementation
3. **Use this baseline** for before/after comparisons

### If Issues Found

1. **Triage**: Categorize issues (bugs vs edge cases vs config)
2. **Fix critical bugs**: Anything that breaks auto-preset or crashes
3. **Document edge cases**: Known limitations to address in V3
4. **Re-run validation**: Verify fixes don't introduce regressions

---

## Maintenance

### Adding New Benchmark Images

1. Add symlink to `assets/phase2_bench/`
2. Update `README.md` in that directory
3. Add test case to `run_phase2_bench_matrix.sh`
4. Document expected classification and preset

### Updating Integration Tests

1. Edit `tests/integration/test_phase2_end_to_end.py`
2. Run `pytest tests/integration/ -v` to verify
3. Update this document with new test descriptions

---

## Timeline Estimate

| Task | Time | When |
|------|------|------|
| Integration tests (already run) | 10 min | ✅ Complete |
| Benchmark quick mode | 30-45 min | Next session |
| Spot-check quality metrics | 20-30 min | Next session |
| Document results | 30 min | After benchmarks |
| CI feature flag testing | 15 min | After merge |
| **Total** | **~2 hours** | **1 session** |

---

## Notes

- Benchmark images are **symlinked**, not copied (saves space)
- Integration tests use **synthetic images** (fast, no large files in repo)
- Benchmark matrix script is **idempotent** (can re-run safely)
- Results are timestamped for historical comparison

---

**Status**: Infrastructure ready, validation runs pending  
**Blocking**: None (can execute anytime)  
**Risk**: Low (validation only, no code changes)
