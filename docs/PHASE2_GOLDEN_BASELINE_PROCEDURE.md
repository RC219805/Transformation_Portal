# ✅ **PHASE 2 — GOLDEN BASELINE TEST RUN PROCEDURE**

**Version:** 1.0  
**Date:** December 13, 2025  
**Status:** Ready for execution  
**Purpose:** Establish a reproducible, reference-quality baseline for *performance, correctness, segmentation accuracy, lighting detection, CLIP classification, preset selection, and output quality* across all Phase 2 components.

---

## 1. **Purpose of the Golden Baseline**

This procedure validates:

* APEX quality outputs are stable and reproducible
* Auto-preset selection behaves deterministically
* CLIP material classifier and lighting detector perform correctly
* Depth maps remain consistent across runs
* Performance remains within thresholds
* No regressions are present before EfficientSAM V3 integration

This baseline becomes the *ground truth* for all future enhancements, including EfficientSAM V3, temporal/video processing, Materials V3, and distributed batch execution.

---

## 2. **Prerequisites**

### 2.1 Environment

* Python environment activated
* Dependencies installed (`ml`, `lux-depth-v2`, `bench`, etc.)
* Models cached OR allow CI to download CLIP automatically
* Phase 2 features enabled:

```bash
export ENABLE_PHASE2_FEATURES=true
```

### 2.2 Directory Structure

Required benchmark set (already created):

```
assets/phase2_bench/
  750Picacho_Kitchen_Ultimate.tif
  750Picacho_PrimaryBedroom_Ultimate.tif
  750Picacho_PrimaryBathroom_Ultimate.tif
  750Picacho_Pool_Ultimate.tif
  750Picacho_Aerial_Ultimate.tif
```

### 2.3 Scripts

* `scripts/run_phase2_bench_matrix.sh`
* `bench/bench_phase2.py`
* `scripts/compare_depth_quality.py`

---

## 3. **Test Overview**

### **Golden Baseline Test Suite Consists of 4 Pillars**

| Pillar | What It Validates | Artifact Produced |
|--------|-------------------|-------------------|
| **P1. Functional Correctness** | Auto-preset, CLIP, lighting, APEX pipeline | 11–14 processed outputs |
| **P2. Metrics Consistency** | Color/luma accuracy, depth map structure | QA JSON reports |
| **P3. Performance Stability** | Runtime, memory, CLIP overhead | Bench result JSON |
| **P4. Structural Integrity** | File formats, metadata, segmentation structure | Validation logs |

**Total expected time: ~90 minutes.**

---

## 4. **Execution Steps**

---

### **Step 1 — Run the Benchmark Matrix (Functional Coverage)**

#### Quick mode (recommended first):

```bash
bash scripts/run_phase2_bench_matrix.sh --quick
```

This runs:

* APEX on Kitchen
* APEX on Pool
* APEX on Bedroom
* APEX on Bathroom
* Auto-preset on 3 scenes

Outputs will appear under:

```
outputs/phase2_bench_matrix/
```

#### Full mode (establishing the golden standard):

```bash
bash scripts/run_phase2_bench_matrix.sh
```

This runs:

* **All tiers**: Standard, Max, APEX
* **All benchmarks**: Kitchen, Bedroom, Bath, Pool, Aerial
* **Auto preset**: all 5 scenes

**Total runs:** 11–14 depending on configuration.

**Expected outputs per run:**
- Processed output images (PNG/TIFF)
- `pipeline.log` with Phase 2 logging
- Quality metrics (if configured)

---

### **Step 2 — Validate Reports and Metrics**

For each output path, inspect logs for Phase 2 behavior:

```bash
# Check for CLIP classification logs
grep -r "CLIP" outputs/phase2_bench_matrix/*/pipeline.log

# Check for lighting detection
grep -r "Lighting" outputs/phase2_bench_matrix/*/pipeline.log

# Check for auto-preset selection
grep -r "Auto-selected preset" outputs/phase2_bench_matrix/*/pipeline.log
```

#### **Pass Criteria**

| Metric | Threshold |
|--------|-----------|
| Color Accuracy | `< 0.06` |
| Luma Accuracy | `< 0.06` |
| Segmentation Quality | `>= 0.55` (for APEX) |
| Phase 2 Overhead | `< 500ms` |

#### **Special Case Validation**

* **Pool twilight scenes:** check sky gradient and water specular behavior
* **Bathroom scenes:** check high specular surfaces (mirrors, chrome)
* **Aerial scene:** check depth stability at far distances

---

### **Step 3 — Validate Depth Map Consistency**

Use the depth comparison utility:

```bash
python scripts/compare_depth_quality.py \
  --master assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif \
  --upscaled outputs/phase2_bench_matrix/750Picacho_Kitchen_Ultimate_interior_luxury_apex_quality/output_upscaled.tif \
  --output-dir outputs/phase2_depth_compare/kitchen/
```

Examine:

* Δ-depth heatmap
* Edge gradient consistency
* Noise profile differences

#### Pass Criteria

* Δ-depth mean < **0.03**
* No structural distortion along edges
* No stair-stepping or posterization

---

### **Step 4 — Evaluate CLIP + Lighting Detection Outputs**

For each pipeline log, extract Phase 2 decisions:

```bash
# Scene classification
grep "Scene type:" outputs/phase2_bench_matrix/*/pipeline.log

# Lighting detection
grep "Lighting condition:" outputs/phase2_bench_matrix/*/pipeline.log
```

#### Expected Values by Scene

| Scene | Expected Scene Label | Expected Lighting |
|-------|---------------------|-------------------|
| Kitchen | `interior_kitchen` | `afternoon` or `mixed` |
| Bedroom | `interior_bedroom` | `soft_morning` or `diffuse` |
| Bath | `interior_bathroom` | `mixed` |
| Pool | `exterior_pool` | `twilight` or `golden_hour` |
| Aerial | `exterior_landscape` or `exterior_facade` | `midday` |

#### Pass Criteria

* Scene classification matches expected category (or reasonable alternative)
* Lighting detection not wildly inconsistent
* Confidence > 0.20 for auto-preset selection

---

### **Step 5 — Confirm Auto-Preset Behavior**

Check auto-preset selection logs:

```bash
grep "Auto-selected preset" outputs/phase2_bench_matrix/*_auto_*/pipeline.log
```

#### Expected:

* Kitchen APEX tier → `interior_luxury_apex_quality`
* Bedroom Max tier → `interior_luxury_max_quality`
* Pool APEX tier → `exterior_pool_apex_quality`
* Aerial Standard tier → `exterior_showcase` or appropriate exterior preset

#### Pass Criteria:

* Auto-preset chooses the **correct preset** (matching mapping table in code)
* Confidence > 0.20
* Fallback behavior documented if used

---

### **Step 6 — Performance Benchmark Run**

```bash
python bench/bench_phase2.py
```

Inspect:

```
bench/results/phase2_benchmark_results.json
```

#### Pass Criteria (as enforced by CI)

| Metric | Threshold |
|--------|-----------|
| CLIP Classification Avg Time | `< 500 ms` |
| Init Time (Standard/Max/APEX) | `< 2.0 s` |
| Peak Memory | `< 1200 MB` |

---

## 5. **Golden Baseline Certification Summary**

Once all steps pass, create certification document:

**File:** `docs/SESSIONS/PHASE2_GOLDEN_BASELINE_RESULTS_<YYYY-MM-DD>.md`

### Include:

* Benchmark matrix summary (pass/fail for each test case)
* Performance results from bench_phase2.py
* Depth consistency results
* Auto-preset decisions with confidence scores
* Any anomalies (even if acceptable)
* Screenshots or crops of representative frames
* SHA commit hash of validated code state

This file becomes your *frozen reference* before EfficientSAM V3.

### Template:

```markdown
# Phase 2 Golden Baseline Results

**Date:** YYYY-MM-DD  
**Commit:** <git SHA>  
**Tester:** <name>  
**Status:** ✅ CERTIFIED / ⚠️ CONDITIONAL / ❌ FAILED

## Benchmark Matrix Results

| Test Case | Preset | Status | Notes |
|-----------|--------|--------|-------|
| Kitchen | APEX | ✅ | Color: 0.0019, Luma: 0.0018 |
| ... | ... | ... | ... |

## Performance Metrics

- CLIP Overhead: XXX ms
- Init Time (APEX): X.XX s
- Peak Memory: XXX MB

## Auto-Preset Accuracy

- Kitchen → interior_luxury_apex_quality (confidence: 0.XX)
- ...

## Known Issues

- None / List any edge cases

## Certification

This baseline is CERTIFIED for use as reference for EfficientSAM V3 development.

Signed: <name>
Date: <date>
```

---

## 6. **Failure Handling**

### If a test fails:

1. Add annotation to the Baseline Results doc
2. Compare metrics vs previous commit
3. Identify regression source via:
   * Test logs
   * Git diff
   * Benchmark result deltas
4. Resolve issues before proceeding to EfficientSAM

This ensures EfficientSAM is built on a stable, validated Phase 2 foundation.

---

## 7. **Completion Criteria (Go/No-Go for EfficientSAM V3)**

### **🟢 GREEN (Proceed)** when all are true:

* All APEX outputs meet color/luma thresholds
* CLIP + lighting stable across runs
* Auto-preset correct for 4/5 benchmark scenes
* Depth map consistency passes threshold
* Performance is within CI-enforced bounds

### **🟡 YELLOW (Proceed with caution)** when:

* Minor deviations present but explained
* Slight fluctuation in lighting classification
* Variation in exterior scenes under complex lighting

### **🔴 RED (Do NOT proceed)** when:

* Color/luma metrics exceed thresholds
* Depth map structural inconsistency
* Auto-preset selects wrong class consistently
* CLIP fails to load or misclassifies
* Runtime exceeds benchmark thresholds

**EfficientSAM V3 should *only* begin once the baseline is GREEN.**

---

## 8. **Maintenance and Updates**

### Re-running the Baseline

Run Golden Baseline validation:

1. **After major Phase 2 changes** (new CLIP models, lighting detector updates)
2. **Before each Phase 3 milestone** (EfficientSAM, video, distributed)
3. **Monthly** for regression detection
4. **Before production releases**

### Updating Thresholds

If thresholds need adjustment:

1. Document rationale in Git commit
2. Update this procedure
3. Re-certify with new thresholds
4. Archive old baseline results

### Adding New Test Cases

1. Add scene to `assets/phase2_bench/`
2. Update `run_phase2_bench_matrix.sh`
3. Add expected values to Step 4 table
4. Re-run certification

---

## 9. **Quick Reference**

```bash
# Full validation (90 minutes)
bash scripts/run_phase2_bench_matrix.sh
python bench/bench_phase2.py
python scripts/compare_depth_quality.py --master ... --upscaled ...

# Quick validation (30 minutes)
bash scripts/run_phase2_bench_matrix.sh --quick
python bench/bench_phase2.py

# Check results
grep -r "Auto-selected" outputs/phase2_bench_matrix/*/pipeline.log
jq '.performance' bench/results/phase2_benchmark_results.json
```

---

## 10. **Appendix: Troubleshooting**

### CLIP Model Download Issues

```bash
# Pre-download CLIP model
python -c "from transformers import CLIPProcessor, CLIPModel; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Memory Issues

```bash
# Reduce batch sizes in bench_phase2.py
# Or run quick mode only
bash scripts/run_phase2_bench_matrix.sh --quick
```

### Pipeline Failures

```bash
# Check detailed logs
tail -100 outputs/phase2_bench_matrix/*/pipeline.log

# Run single test case manually
python -m lux_depth_v2.cli --input assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif \
    --output-dir test_output --preset interior_luxury_apex_quality --verbose
```

---

**Version History:**
- v1.0 (2025-12-13): Initial Golden Baseline procedure

**Next Review:** Before EfficientSAM V3 implementation
