# PR-W4: Water Validation Harness + CI Regression Checker (Baseline Detector)

## One-Line Scope Statement

**Validation harness is complete and integration-ready; the detector remains a minimal baseline pending PR-W1.**

---

## What's Included

### Core Implementation
1. **`scripts/prw_water_validation.py`** - Validation harness with v0 schema support
2. **`lux_depth_v2/water_candidate.py`** - Baseline detector (explicitly labeled as stub)
3. **`tests/test_prw_water_validation_deterministic.py`** - 3 deterministic stability tests (all passing)

### Tooling
4. **`scripts/check_regression.py`** - CI regression checker with epsilon guard
5. **`.gitignore`** - Water dataset patterns (recursive for class subfolders)

### Documentation
6. **`docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md`** - Complete v0 schema specification
7. **`docs/PR_W4_FINAL_SUMMARY.md`** - Implementation summary and usage guide
8. **`docs/PR_W4_CORRECTNESS_FIXES.md`** - Semantic drift fixes applied

### Dataset Scaffolding
9. **`data/water_v0/ci_subset.txt`** - 14 image paths for fast CI (<10s validation)
10. **`data/water_v0/README.md`** - Dataset documentation

---

## Key Features

### Validation Harness
- ✅ Loads v0 ground truth schema (pool/ocean labels + negative controls)
- ✅ Computes per-class metrics (pool recall, ocean recall)
- ✅ Computes false trigger rate (should_detect=false but detected)
- ✅ Computes median coverage (drift detection with epsilon guard)
- ✅ Deterministic stability via `--seed` (3 tests passing)
- ✅ Subset support via `--subset-file` (for fast CI)
- ✅ Backward-compatible report schema

### CI Regression Checker
- ✅ 4 regression gates (recall, edge, coverage drift, false trigger)
- ✅ Epsilon guard for coverage drift (prevents divide-by-zero)
- ✅ Absolute delta semantics (not relative %)
- ✅ Warning mode (exit 0) or error mode (exit 1)

### Correctness Fixes (No Silent Lies)
- ✅ `is_false_positive` = `is_false_trigger` (legacy alias, same value)
- ✅ `detected` field uses explicit `present` flag (not inferred from coverage)
- ✅ `.gitignore` recursive patterns catch class subfolders (`pool/`, `ocean/`)

---

## Test Results

```
============================= test session starts ===============================
tests/test_prw_water_validation_deterministic.py::test_stability_deterministic_with_seed PASSED
tests/test_prw_water_validation_deterministic.py::test_stability_different_with_different_seed PASSED
tests/test_prw_water_validation_deterministic.py::test_full_validation_deterministic PASSED
============================== 3 passed in 0.16s =================================
```

---

## Usage

### Full Validation
```bash
python scripts/prw_water_validation.py \
    --ground-truth data/water_v0/ground_truth.json \
    --output water_validation_report.json \
    --seed 42
```

### CI Subset Only
```bash
python scripts/prw_water_validation.py \
    --ground-truth data/water_v0/ground_truth.json \
    --subset-file data/water_v0/ci_subset.txt \
    --output ci_report.json \
    --seed 42
```

### CI Regression Check
```bash
python scripts/check_regression.py \
    --baseline data/water_v0/baseline_v0.json \
    --current water_validation_ci.json \
    --mode warning
```

---

## Report Schema (Per-Image)

```json
{
  "image_path": "pool/pool_0001.jpg",
  "scene_type": "pool",
  "should_detect": true,
  "difficulty": "easy",
  "tags": [],
  "detected": true,
  "coverage": 0.85,
  "coverage_px": 55296,
  "confidence": 0.72,
  "source": "heuristic",
  "implementation": "stub_v0_blue_threshold",
  "edge_alignment_score": 0.68,
  "boundary_px": 1024,
  "stability_score": 0.82,
  "is_false_positive": false,
  "is_false_trigger": false,
  "processing_time_ms": 28.5
}
```

**New fields**:
- `detected` - Explicit boolean from detector (not inferred)
- `should_detect` - From ground truth (enables false trigger metric)
- `difficulty`, `tags` - From ground truth
- `implementation` - Detector version for apples-to-apples comparison

**Legacy fields** (backward-compatible):
- `scene_type` - Alias for `label`
- `is_false_positive` - Equals `is_false_trigger`

---

## Metrics Computed

| Metric | Target | Status |
|--------|--------|--------|
| Pool recall | ≥85% | Computed from `detected` flag |
| Ocean recall | ≥85% | Computed from `detected` flag |
| False trigger rate | ≤10% (v1), ≤5% (prod) | Computed for should_detect=false |
| Edge alignment | ≥0.6 per class | Computed when mask available |
| Stability | ≥0.8 per class | Deterministic with `--seed` |
| Median coverage drift | <2x, >0.5x | Epsilon guard prevents false alarms |

---

## Acceptance Criteria (from PR-W4 Spec)

✅ **Coverage sanity**: Pool/ocean recall, median coverage computed  
✅ **Boundary pixels**: Computed from mask when available  
✅ **Edge alignment** (primary): Computed from mask + image gradients  
✅ **Stability**: Deterministic with `--seed`, computed across perturbations  
✅ **False trigger checks**: Computed for should_detect=false  
✅ **JSON reporting**: Clean schema, backward-compatible  
✅ **CLI**: Supports ground-truth, output, subset-file, seed  
✅ **Deterministic**: Tests prove stability is reproducible  
✅ **CI regression gates**: 4 gates with epsilon guard, absolute deltas  

---

## Known Limitations

1. **Detector is stub** (PR-W1 pending):
   - Current: Simple blue threshold (always present=true, full coverage)
   - Needed: Multi-cue heuristic detector (HSV, texture, component filtering)

2. **Edge alignment requires mask**:
   - Currently computed when detector provides mask
   - Falls back to 0.0 if mask unavailable

3. **Thresholds not calibrated**:
   - Targets (85% recall, 0.6 edge, 0.8 stability) are goals
   - Need labeled dataset v0 to calibrate

---

## Breaking Changes

**None** - Backward-compatible schema with legacy aliases preserved.

---

## Merge-Ready Statement

**Status**: Validation harness complete; integration-ready; detector remains stub pending PR-W1

**Tests**: All 3 deterministic stability tests passing

**Linting**: Clean (no new linting issues)

**Reviewer notes**:
- Schema avoids "silent lies" (no fields that claim things they don't compute)
- Epsilon guard prevents noisy CI warnings
- Implementation field enables future detector comparison
- Deterministic stability enables meaningful CI regression checks
- Correctness fixes ensure `is_false_positive == is_false_trigger` and `detected` uses explicit boolean

---

## Files Changed (10 Total)

**Core**:
- `scripts/prw_water_validation.py` (harness)
- `lux_depth_v2/water_candidate.py` (baseline detector)
- `tests/test_prw_water_validation_deterministic.py` (3 tests)

**Tooling**:
- `scripts/check_regression.py` (CI checker)
- `.gitignore` (water dataset patterns)

**Documentation**:
- `docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md` (schema spec)
- `docs/PR_W4_FINAL_SUMMARY.md` (implementation summary)
- `docs/PR_W4_CORRECTNESS_FIXES.md` (semantic fixes)

**Dataset**:
- `data/water_v0/ci_subset.txt` (CI subset)
- `data/water_v0/README.md` (dataset docs)

---

## Next Steps (72-Hour Plan)

**Hour 0-6**: Collect dataset v0 (44 images) + establish baseline  
**Hour 6-12**: Implement PR-W1 detector (HSV, component filtering, texture)  
**Hour 12-18**: Calibrate thresholds + wire regression checker into CI  
**Hour 18-72**: Expand dataset + second detector iteration + CI hardening
