# PR-W1.1 Final Assessment (Dec 15, 2025)

## Current Status

**PR #559: READY TO MERGE** (with caveats documented below)

### What's Complete ✅
- Deterministic CI fixture generation (`scripts/gen_water_ci_fixture.py`)
- Ground truth v0 schema + validator
- Baseline pinned (`data/water_v0/baseline_ci_v0.json`)
- Warn-only CI regression job
- Path resolution fix (root relative to ground_truth.json)
- Pre-commit guard + .gitignore coverage
- README accurately describes detector status

### Baseline Metrics (Uncalibrated)
```json
{
  "pool_recall": 1.0,         // ✅ Detector is working
  "ocean_recall": 1.0,        // ✅ Detector is working
  "false_trigger_rate": 1.0,  // ❌ 100% FT (2/2 negatives misclassified)
  "false_trigger_count": 2,   // Both hard negatives triggered
  "total_images": 14          // 6 pool + 6 ocean + 2 negatives
}
```

## Critical Understanding

### This is NOT a "stub detector" baseline
- Post-PR #558, `WaterCandidateDetector` is a **real CPU heuristic** (chromaticity + specular + texture + planarity)
- The 100% false trigger rate reflects **uncalibrated thresholds**, not broken plumbing
- The baseline proves: detector executes, SegFormer-first logic works, contract is stable

### This IS an "uncalibrated detector" baseline
- Current `confidence_threshold=0.4` is too permissive
- Hard negatives (blue wall, reflective glass) satisfy enough cues to pass
- PR-W1.2 calibration will tune thresholds based on real-world dataset

### Regression Checker is Correct
`scripts/check_regression.py` enforces **monotonic constraints**:
- ✅ Recall drops >10% → warn/error
- ✅ False trigger increases >15% → warn/error
- ✅ Coverage drift >2x or <0.5x → warn
- ✅ Edge alignment drops >0.1 → warn

This means:
- **Calibration improvements pass cleanly** (lowering FT from 100% → 20% is not flagged)
- **Real regressions are caught** (recall dropping or FT rising)

## What PR-W1.1 Actually Validates

### ✅ Execution Contract
- MaterialsV3 engine runs with `enabled=True`
- Water detection executes when `water_detection_enabled=True`
- Detector returns typed `WaterCandidateResult` (not dict)
- Report is JSON-serializable

### ✅ Determinism
- Same seed → same fixtures → same results
- CI can reproduce baseline locally

### ✅ Schema Stability
- Ground truth v0 validated
- Baseline report structure stable
- No silent field renames

### ⚠️  NOT Validated Yet
- **Threshold quality** (uncalibrated)
- **Real-world performance** (synthetic fixtures only)
- **Negative precision** (100% FT shows threshold too low)

## Merge Decision

### MERGE NOW if you accept this framing:
✅ PR-W1.1 is **infrastructure + contract guardrail**  
✅ Current baseline is **uncalibrated but functional**  
✅ 100% FT is **documented and expected** until PR-W1.2  
✅ Regression checker **won't punish calibration improvements**  

### Required PR Body Language
Replace any "stub detector" references with:
- "Detector: WaterCandidateDetector heuristic (merged in PR #558)"
- "Baseline: uncalibrated thresholds (100% FT expected)"
- "Next: threshold calibration + real-world dataset (PR-W1.2)"

Add to PR body:
```markdown
## Baseline Quality Note

Current baseline metrics:
- Pool/ocean recall: 100% (detector is functional)
- False trigger rate: 100% (uncalibrated thresholds, expected)

The 100% false trigger rate reflects **uncalibrated default thresholds** (confidence=0.4), 
not broken detection logic. PR-W1.2 will calibrate thresholds using a real-world dataset.

The regression checker uses monotonic constraints, so calibration improvements 
(reducing FT from 100% → target <20%) will pass cleanly.
```

## Next Steps (Post-Merge)

### 1. PR-W1.2: Threshold Calibration
- Collect real pool/ocean images (private, not committed)
- Run harness on real dataset
- Tune thresholds to achieve:
  - Pool recall ≥ 85%
  - Ocean recall ≥ 80%
  - **False trigger rate < 20%** (primary goal)
- Update defaults in `WaterDetectionParams`
- Re-pin baseline after calibration

### 2. Update CI Job (Optional)
- Current: warn-only (correct for now)
- After PR-W1.2: consider flipping to error mode once FT < 20%

### 3. Materials V3 Next
- With water baseline stable, proceed to:
  - **PR-4E (Wood)** - highest ROI next material
  - Materials V3 stone canary observability
  - Expand heuristics based on measured failure modes

## Files Changed (PR #559)

### Added
- `data/water_v0/ground_truth.json` (v0 schema metadata)
- `data/water_v0/ground_truth.schema.json` (JSON Schema validator)
- `data/water_v0/baseline_ci_v0.json` (pinned baseline)
- `data/water_v0/ci_subset.txt` (14 images for CI)
- `data/water_v0/README.md` (dataset documentation)
- `scripts/gen_water_ci_fixture.py` (deterministic fixture generator)
- `scripts/validate_ground_truth.py` (schema validator)
- `scripts/check_regression.py` (warn-only regression checker)
- `.github/workflows/water-regression.yml` (CI job)

### Modified
- `scripts/prw_water_validation.py` (path resolution + enabled=True)
- `.gitignore` (recursive patterns for `data/water_*/images/**`)
- `.pre-commit-config.yaml` (pre-commit guard for images)

## Verification Commands

```bash
# 1. Generate CI fixtures
python scripts/gen_water_ci_fixture.py --seed 42 --output data/water_v0/images/

# 2. Validate ground truth schema
python scripts/validate_ground_truth.py data/water_v0/ground_truth.json

# 3. Run harness (should match baseline)
python scripts/prw_water_validation.py \
  --ground-truth data/water_v0/ground_truth.json \
  --subset-file data/water_v0/ci_subset.txt \
  --output outputs/prw_test/validation_report.json \
  --seed 42

# 4. Check regression (should pass)
python scripts/check_regression.py \
  --baseline data/water_v0/baseline_ci_v0.json \
  --current outputs/prw_test/validation_report.json \
  --mode warning

# 5. Verify no images staged
git diff --cached --name-only | grep -E '^data/water_v0/images/' \
  && echo "ERROR: images staged" || echo "OK: no images staged"
```

## Bottom Line

**MERGE PR #559** after adding the "Baseline Quality Note" to the PR body.

The 100% false trigger rate is:
- ✅ **Expected** (uncalibrated thresholds)
- ✅ **Documented** (in README + this assessment)
- ✅ **Not a blocker** (regression checker won't punish improvements)
- ✅ **Fixed in PR-W1.2** (calibration)

This baseline is a **contract + determinism guardrail**, not a quality benchmark.  
That's sufficient for PR-W1.1 scope.

---

**Assessment Date:** 2025-12-15  
**Reviewer:** Context analysis (automated)  
**Status:** READY TO MERGE (with documentation update)
