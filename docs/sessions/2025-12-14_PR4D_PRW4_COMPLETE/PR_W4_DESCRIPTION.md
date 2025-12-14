# PR-W4: Water Validation Harness (Pool + Ocean)

## Summary

Added a standalone validation harness and test suite to measure water detection quality on labeled datasets.

## What's Included

### Validation Harness
- **File:** `scripts/prw_water_validation.py`
- **Capabilities:**
  - Coverage sanity checks (mean + median per class)
  - Boundary pixel counting
  - Edge alignment vs image gradients (primary metric)
  - Stability across perturbations (deterministic with `--seed`)
  - False trigger rate tracking (`should_detect=false` logic)
  - JSON report generation with summary statistics
  - CI subset support via `--subset-file`

### Test Suite
- **Files:** `tests/test_prw_water_validation.py`, `tests/test_prw_water_validation_deterministic.py`
- **Coverage:** 16 tests (schema, metrics, determinism, end-to-end validation)
- **Status:** 16/16 passing

### Regression Checker
- **File:** `scripts/check_regression.py`
- **Features:** Compares baseline vs current reports; tracks recall, edge alignment, stability, false trigger rate, coverage drift
- **Mode:** Warning-only (safe for early iterations)

### Documentation
- `docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md` - Schema specification
- `docs/PR_W4_COMPLETION_ACCURATE.md` - Detailed completion status
- This file - PR description

### Infrastructure
- `.gitignore` patterns for water datasets (prevents accidental image commits)
- `data/water_v0/` scaffold (ci_subset.txt, README placeholders)

## Current Limitations

### 1. Detector Is a Stub
- `lux_depth_v2/water_candidate.py` exists but returns empty mask
- Edge alignment and boundary pixel metrics default to 0.0 / 0 when mask unavailable
- **Resolution:** PR-W1 (real heuristic detector)

### 2. Thresholds Not Calibrated
- Target values documented (edge ≥ 0.6, stability ≥ 0.8, FT ≤ 5%)
- Must be derived from dataset v0 statistics
- **Resolution:** Dataset v0 collection + calibration

### 3. No Production Dataset
- Schema and tools exist
- No labeled images committed yet
- **Resolution:** 72-hour data collection plan

## Quality Gates

✅ **Linting:** flake8 clean (0 errors, max-line-length=127)  
✅ **Tests:** 16/16 passing (0.19s)  
✅ **Determinism:** Stability scores identical across runs with same seed  
✅ **Schema Compliance:** Matches finalized ground truth schema  
✅ **CI-Ready:** `--subset-file` + regression checker executable  

## Key Design Decisions

### Two Labels Only (pool, ocean)
- Both are water; no "non_water" class
- Negative controls use `should_detect: false` instead of third label
- False trigger rate replaces false positive rate

### Per-Image Deterministic Seeding
- Seed derived from `(seed ^ hash(img_path)) & 0xFFFFFFFF`
- Each image gets stable but unique perturbations
- Enables reproducible CI comparisons

### Mask-Based Metrics
- Edge alignment computed when detector provides mask
- Defaults to 0.0 when mask unavailable (stub detector)
- Harness correctly structured to consume mask from PR-W1

## What This PR Does NOT Include

❌ Real water detector (PR-W1 scope)  
❌ Labeled dataset (manual curation needed)  
❌ Calibrated thresholds (requires dataset v0)  
❌ CI job integration (ready, but waiting for baseline)  

## Next Steps

1. **PR-W1:** Implement heuristic detector (HSV gating, component filtering, texture sanity)
2. **Dataset v0:** Collect 60+ labeled images (pool, ocean, hard negatives)
3. **Calibration:** Run harness on dataset v0; lock thresholds
4. **CI Integration:** Add GitHub Actions job with warning mode

## Testing

```bash
# Run all tests
pytest tests/test_prw_water_validation.py tests/test_prw_water_validation_deterministic.py -v

# Run harness (example)
python scripts/prw_water_validation.py \
  --ground-truth data/water_v0/ground_truth.json \
  --output water_validation_report.json \
  --seed 42

# Check regression
python scripts/check_regression.py \
  data/water_v0/baseline.json \
  water_validation_report.json \
  --mode warn
```

## Status

**Validation Harness:** ✅ Complete and tested  
**Detector:** ⏳ Stub (PR-W1 pending)  
**Dataset:** ⏳ Schema ready, no images yet  
**CI Integration:** ✅ Ready for setup  

**Merge Status:** ✅ Ready (harness infrastructure complete; detector is explicit stub)

---

**Reviewer Note:** This PR delivers validation infrastructure only. Edge alignment and boundary metrics will populate with real values once PR-W1 detector lands. Thresholds will be calibrated once dataset v0 is populated. Current stub behavior is intentional and documented.
