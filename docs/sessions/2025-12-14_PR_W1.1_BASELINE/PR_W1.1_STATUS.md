# PR-W1.1: Water Baseline Infrastructure - STATUS COMPLETE ✅

## Critical Fix Applied

**Root Cause Identified & Resolved:**
- Harness had `water_detection_enabled=True` but `enabled=False` (Materials V3 master gate)
- Result: 0% recall (detector never executed)
- **Fix:** Added `enabled=True` to harness config
- **Outcome:** 100% pool/ocean recall achieved

## Baseline Report (Pinned: `data/water_v0/baseline_ci_v0.json`)

**Metrics (seed=42, deterministic):**
```
Pool recall:     100.0% (6/6 detected)
  - Avg coverage:    85.4%, median: 100.0%
  - Avg confidence:  0.688
  - Avg edge align:  0.243

Ocean recall:    100.0% (6/6 detected)
  - Avg coverage:    100.0%, median: 100.0%
  - Avg confidence:  0.759
  - Avg edge align:  0.224

False trigger:   100.0% (2/2)
  - Both hard negatives detected (blue_wall, glass)
  - Confidence: 0.446-0.448 (threshold=0.4)

Processing time: 97.6ms avg
```

## Analysis

**What's Correct ✅:**
1. Detector is exercised (non-zero recall)
2. Materials V3 integration working
3. Deterministic (stable across runs with seed=42)
4. Contract stable (v0 schema validated)

**Expected Behavior ⚠️:**
- **100% FT rate is expected** for uncalibrated threshold (0.4)
- Hard negatives have blue chromaticity but lack water-specific texture/specular cues
- Detector confidence (0.446-0.448) is just above threshold
- **This documents baseline for threshold calibration (PR-W1.2)**

**Not a Bug:**
- FT rate will be addressed via threshold tuning (PR-W1.2)
- Real dataset will provide calibration data
- Current baseline proves infrastructure works

## Changes Committed

**Files Modified:**
- `scripts/prw_water_validation.py` - Added `enabled=True` to config
- `scripts/gen_water_ci_fixture.py` - Improved hard negatives (desaturated)
- `data/water_v0/baseline_ci_v0.json` - Pinned baseline report

**Safety Confirmed:**
- ✅ No images committed (`.gitignore` active)
- ✅ Only metadata tracked
- ✅ CI-safe (deterministic synthetic generation)

## Next Steps (Post-Merge)

**Immediate (PR-W1.2 - Calibration):**
1. Collect private real-world dataset (pool/ocean/negatives)
2. Run calibration sweep: threshold ∈ [0.45, 0.75], step 0.05
3. Optimize for F1 score (balance recall vs FT rate)
4. Update default `water_candidate_confidence_threshold`
5. Re-pin baseline with calibrated threshold

**Then (PR-4E - Wood Pixel Ops):**
- Water is now observable and protected by regression baseline
- Safe to expand Materials V3 to wood

## PR-W1.1 Acceptance Criteria

✅ **Baseline Infrastructure Complete:**
- [x] Harness exercises detector (100% recall proves execution)
- [x] Synthetic fixtures deterministic (seed=42)
- [x] Baseline pinned (regression anchor)
- [x] Contract stable (v0 schema)
- [x] CI-safe (no model downloads, CPU-only)

✅ **Known Limitations Documented:**
- [x] FT rate 100% (uncalibrated threshold)
- [x] Calibration deferred to PR-W1.2
- [x] Real dataset required for tuning

**Status:** READY TO OPEN PR #559
