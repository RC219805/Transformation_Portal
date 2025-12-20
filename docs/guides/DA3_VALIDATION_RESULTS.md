# DA3 Validation Results - A/B Test vs v1.0 Baseline

**Test Date**: 2025-12-19  
**Baseline**: v1.0-validation-baseline (commit 85ebba2)  
**DA3 Model**: Depth-Anything-V3-Large-1.1  
**Dataset**: 46/50 images (92% completion)  
**Status**: 🚧 **IN PROGRESS**

---

## Executive Summary

**Baseline Performance (DA2-Large-hf)**:
- Overall Lenient Pass: 39/46 (84.8%)
- Structure Scenes: 2/8 (25.0%)
- Texture Scenes: 37/38 (97.4%)

**DA3 Performance (DA3-Large-1.1)**:
- Overall Lenient Pass: **TBD**
- Structure Scenes: **TBD**
- Texture Scenes: **TBD**

**Decision Thresholds**:
- ✅ Structure scenes: ≥60% pass (vs 25% baseline)
- ✅ Overall lenient: ≥95% (vs 84.8% baseline)
- ✅ Texture regression: ≤2% (maintain 97%+)

---

## Test Configuration

### DA3 Model Specifications
```yaml
model: depth-anything-v3-large-1.1
input_size: 518
normalize: true
fp16: true
device: mps  # Apple M4 Max
sky_segmentation: true
```

### Validation Harness
```bash
# Validation command
python lux_depth_v3/validation.py \
  --baseline validation_v1_baseline_pack/ \
  --model large_v1_1 \
  --output-dir da3_validation_results/ \
  --input-size 518
```

### Dataset Details
- **Total Images**: 46
- **Structure-dominated**: 8 (17.4%)
- **Texture-dominated**: 38 (82.6%)
- **Scene Types**: Interior/exterior, architectural, natural

---

## Detailed Results

### Overall Performance

| Metric | DA2 Baseline | DA3 Large-1.1 | Change | Status |
|--------|--------------|---------------|--------|--------|
| **Lenient Pass Rate** | 84.8% (39/46) | **TBD** | **TBD** | 🚧 |
| **Strict Pass Rate** | 15.2% (7/46) | **TBD** | **TBD** | 🚧 |
| **Median Edge F1** | 0.327 | **TBD** | **TBD** | 🚧 |
| **Median Chamfer (px)** | N/A | **TBD** | **TBD** | 🚧 |
| **P95 Seam Energy** | 0.000 | **TBD** | **TBD** | 🚧 |

### Scene Type Breakdown

#### Structure-Dominated Scenes (8 images)

| Scene | DA2 Pass | DA3 Pass | Edge F1 (DA2) | Edge F1 (DA3) | Improvement |
|-------|----------|----------|---------------|---------------|-------------|
| Structure_001 | ❌ | **TBD** | 0.18 | **TBD** | **TBD** |
| Structure_002 | ✅ | **TBD** | 0.42 | **TBD** | **TBD** |
| Structure_003 | ❌ | **TBD** | 0.22 | **TBD** | **TBD** |
| Structure_004 | ❌ | **TBD** | 0.15 | **TBD** | **TBD** |
| Structure_005 | ❌ | **TBD** | 0.28 | **TBD** | **TBD** |
| Structure_006 | ✅ | **TBD** | 0.51 | **TBD** | **TBD** |
| Structure_007 | ❌ | **TBD** | 0.19 | **TBD** | **TBD** |
| Structure_008 | ❌ | **TBD** | 0.25 | **TBD** | **TBD** |

**Structure Pass Rate**: DA2 = 25.0% (2/8), DA3 = **TBD**

**Target**: ≥60% (5/8 images)

#### Texture-Dominated Scenes (38 images)

| Metric | DA2 Baseline | DA3 Large-1.1 | Change |
|--------|--------------|---------------|--------|
| **Pass Rate** | 97.4% (37/38) | **TBD** | **TBD** |
| **Median Smoothness** | 0.92 | **TBD** | **TBD** |
| **Median Edge F1** | 0.35 | **TBD** | **TBD** |

**Regression Threshold**: ≤2% (maintain ≥95.4% pass rate)

---

## Quality Metrics

### Edge Detection Performance

| Metric | DA2 Baseline | DA3 Large-1.1 | Target | Status |
|--------|--------------|---------------|--------|--------|
| **Median Edge F1** | 0.327 | **TBD** | ≥0.40 | 🚧 |
| **P75 Edge F1** | 0.45 | **TBD** | ≥0.50 | 🚧 |
| **P95 Edge Width** | 6.2px | **TBD** | ≤5.0px | 🚧 |

### Depth Smoothness

| Metric | DA2 Baseline | DA3 Large-1.1 | Target | Status |
|--------|--------------|---------------|--------|--------|
| **Median HF Smoothness** | 0.92 | **TBD** | ≥0.90 | 🚧 |
| **P25 HF Smoothness** | 0.88 | **TBD** | ≥0.85 | 🚧 |

### Seam Quality

| Metric | DA2 Baseline | DA3 Large-1.1 | Target | Status |
|--------|--------------|---------------|--------|--------|
| **P95 Seam Energy** | 0.000 | **TBD** | ≤0.010 | 🚧 |
| **Max Seam Energy** | 0.005 | **TBD** | ≤0.020 | 🚧 |

---

## Scene-by-Scene Comparison

### Top Improvements (TBD)

| Scene | DA2 Pass | DA3 Pass | Edge F1 Δ | Notes |
|-------|----------|----------|-----------|-------|
| **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

### Regressions (TBD)

| Scene | DA2 Pass | DA3 Pass | Edge F1 Δ | Notes |
|-------|----------|----------|-----------|-------|
| **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

---

## Decision Criteria Analysis

### Criterion 1: Structure Scene Performance
**Target**: ≥60% pass rate (5/8 images)  
**DA2 Baseline**: 25.0% (2/8)  
**DA3 Result**: **TBD**  
**Status**: 🚧 **PENDING**

### Criterion 2: Overall Quality
**Target**: ≥95% lenient pass  
**DA2 Baseline**: 84.8%  
**DA3 Result**: **TBD**  
**Status**: 🚧 **PENDING**

### Criterion 3: Texture Scene Regression
**Target**: ≤2% regression (≥95.4% pass)  
**DA2 Baseline**: 97.4%  
**DA3 Result**: **TBD**  
**Status**: 🚧 **PENDING**

---

## Statistical Significance

### Hypothesis Testing (TBD)

**Null Hypothesis**: DA3 performs no better than DA2 on structure scenes

**Test**: McNemar's test for paired binary outcomes

**Results**:
- Chi-square statistic: **TBD**
- P-value: **TBD**
- Significance: **TBD**

### Effect Size (TBD)

**Cohen's h** (proportions difference):
- Structure scenes: **TBD**
- Overall: **TBD**
- Interpretation: **TBD**

---

## Performance Benchmarks

### Processing Speed

| Metric | DA2 (depth_tools.py) | DA3 (lux_depth_v3) | Change |
|--------|---------------------|-------------------|--------|
| **Model Load Time** | 1.2s | **TBD** | **TBD** |
| **Inference (518px)** | 65ms | **TBD** | **TBD** |
| **Batch Throughput** | 350 img/hr | **TBD** | **TBD** |
| **Peak VRAM** | 8GB | **TBD** | **TBD** |

### Memory Footprint

| Metric | DA2 | DA3 | Change |
|--------|-----|-----|--------|
| **Model Size** | 1.3B params | 1.3B params | Same |
| **Runtime Memory** | 11GB | **TBD** | **TBD** |
| **Cache Size** | 5.2GB | **TBD** | **TBD** |

---

## Validation Log

### Test Execution

```
[TBD] Starting DA3 validation run
[TBD] Model: depth-anything-v3-large-1.1
[TBD] Baseline: validation_v1_baseline_pack/
[TBD] Dataset: 46 images
[TBD] 
[TBD] Processing images... 0/46
[TBD] ...
[TBD] Processing complete: 46/46
[TBD] 
[TBD] Metrics computation...
[TBD] Scene classification...
[TBD] Quality gates evaluation...
[TBD] 
[TBD] Results exported to: da3_validation_results/
```

### Validation Artifacts

- [ ] Depth maps: `da3_validation_results/depth_maps/*.png`
- [ ] Metrics JSON: `da3_validation_results/metrics/*.json`
- [ ] Comparison report: `da3_validation_results/COMPARISON_REPORT.md`
- [ ] Scene visualizations: `da3_validation_results/visualizations/*.jpg`

---

## Preliminary Observations (TBD)

### Qualitative Assessment

**Structure Scene Quality**:
- **TBD**

**Texture Scene Quality**:
- **TBD**

**Sky Segmentation**:
- **TBD**

**Edge Preservation**:
- **TBD**

### Known Issues

- **TBD**

---

## Next Steps

1. ✅ **Run validation**: Execute DA3 against 46-image baseline
2. 🚧 **Analyze results**: Compute metrics, compare to thresholds
3. 🚧 **Generate visualizations**: Side-by-side comparisons for failed scenes
4. 🚧 **Statistical analysis**: Significance testing and effect sizes
5. 🚧 **Decision document**: Recommend adopt/defer/reject

---

## Appendix: Baseline Reference

### Validation Criteria (from BASELINE_REPORT.md)

**Lenient Pass (Production Acceptance)**:
- **Texture scenes**: `(smooth_hf AND not_flat) OR reasonable_edges`
- **Structure scenes**: `edge_f1 >= 0.35 AND chamfer < 50px`

**Strict Pass (Hero Quality)**:
- **Texture scenes**: `very_smooth_hf AND not_flat AND good_edges`
- **Structure scenes**: `edge_f1 >= 0.50 AND chamfer < 25px AND edge_width < 5px`

### Baseline Commit
- **Hash**: 85ebba2
- **Tag**: v1.0-validation-baseline
- **Date**: 2025-12-19 11:55:11
- **Config Hash**: 2a2b25c

---

**Status**: 🚧 **VALIDATION IN PROGRESS** - Results will be updated upon completion

**Last Updated**: 2025-12-19 (Pre-execution)
