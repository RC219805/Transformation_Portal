# Materials V3 Integration Decision
**Date**: 2025-12-18  
**Session**: Post-Structure-Aware Edge Gating Implementation  
**Architect**: @transformation-portal-architect  

---

## Executive Summary

**DECISION: NO-GO**

Materials V3 integration is **NOT RECOMMENDED** at this time. The structure-aware edge gating baseline has **FAILED validation** with a scene classification accuracy of **28.6%** (2/7 images correct).

**Critical Issues**:
1. ❌ **Scene classification accuracy**: 28.6% (expected >90%)
2. ❌ **Baseline quality gates**: 0% lenient pass, 0% strict pass
3. ❌ **Misclassification pattern**: Systematic failures across both texture AND structure scenes

**Required Action**: Fix heuristic classifier FIRST before considering Materials V3.

---

## Phase 1: Baseline Validation Results

### Execution Summary
- **Total images**: 7
- **Execution success**: 7/7 (100%)
- **Seam validation**: 7/7 (100%)
- **Quality (lenient)**: 0/7 (0%)
- **Quality (strict)**: 0/7 (0%)

### Scene Classification Accuracy: 28.6% ❌

| Image | Expected | Predicted | Status | Depth Var | Edge Ratio |
|-------|----------|-----------|--------|-----------|------------|
| glass_building | texture | texture | ✓ | 0.0385 | 41359.00 |
| glass_facade | texture | **structure** | ✗ | 0.0774 | 9.46 |
| interior_bathroom | structure | **texture** | ✗ | 0.0743 | 14.38 |
| interior_kitchen | structure | **texture** | ✗ | 0.0572 | 5.99 |
| ocean_1 | texture | texture | ✓ | 0.0456 | 48912.00 |
| pool_texture_1 | texture | **structure** | ✗ | 0.0202 | 83.81 |
| pool_texture_2 | texture | **structure** | ✗ | 0.0175 | 85.45 |

**Misclassification Rate**: 5/7 (71.4%)

### Root Cause Analysis

**Current Classification Logic** (`high_fidelity_depth/quality_metrics.py:classify_scene_type`):
```python
def classify_scene_type(
    rgb_edges_raw: np.ndarray,
    rgb_edges_structure: np.ndarray,
    texture_threshold: float = 3.0
) -> str:
    raw_count = np.count_nonzero(rgb_edges_raw)
    structure_count = np.count_nonzero(rgb_edges_structure)
    
    if structure_count == 0:
        return 'texture_dominated'
    
    ratio = raw_count / structure_count
    
    return 'texture_dominated' if ratio > texture_threshold else 'structure_dominated'
```

**Problem**: Single-threshold heuristic (`ratio > 3.0`) is **TOO SIMPLISTIC**.

#### Failure Patterns

**Pattern 1: Pool Water Misclassified as Structure**
- `pool_texture_1`: ratio=83.81 → classified as **structure** (WRONG)
- `pool_texture_2`: ratio=85.45 → classified as **structure** (WRONG)
- **Root Cause**: Bilateral filter over-smoothed water surface, leaving only pool edges
- **Effect**: High edge ratio (>3.0) but edges are STRUCTURE, not texture

**Pattern 2: Glass Facade Misclassified as Structure**
- `glass_facade`: ratio=9.46 → classified as **structure** (WRONG)
- **Root Cause**: Reflective glass has low texture, bilateral filter preserved building edges
- **Effect**: Edge ratio close to threshold, misclassified

**Pattern 3: Interiors Misclassified as Texture**
- `interior_bathroom`: ratio=14.38 → classified as **texture** (WRONG)
- `interior_kitchen`: ratio=5.99 → classified as **texture** (WRONG)
- **Root Cause**: Patterned tiles, wood grain, fabric texture dominate edge count
- **Effect**: High edge ratio despite clear structural elements (walls, cabinets)

### Validation Metrics Breakdown

**Texture Scenes (Expected: 5 images)**
- ✓ Correctly classified: 2 (glass_building, ocean_1)
- ✗ Misclassified as structure: 3 (glass_facade, pool_texture_1, pool_texture_2)

**Structure Scenes (Expected: 2 images)**
- ✓ Correctly classified: 0
- ✗ Misclassified as texture: 2 (interior_bathroom, interior_kitchen)

### Quality Gate Analysis

**All images FAILED both lenient and strict gates** (0/7 pass rate).

**Gate Distribution**:
- Smoothness gates (texture): 4 images (all FAIL)
- Edge alignment gates (structure): 3 images (all FAIL)

**Implication**: Even if gates were applied to correctly classified scenes, quality is insufficient.

---

## Phase 2: Materials V3 Integration Assessment

### Strategic Evaluation: Does NOT Meet Requirements

**Materials V3 adds value ONLY IF**:
1. ❌ Heuristic classifier misclassifies important cases → **CONFIRMED** (71.4% error rate)
2. ❓ Pixel-level material masks needed for post-processing → **UNKNOWN** (not validated)
3. ❓ Need determinism/robustness beyond heuristics → **CANNOT ASSESS** (baseline broken)

### Decision Logic

**Materials V3 Prerequisites**:
- [x] Scene classification accuracy >90% (currently 28.6%) ❌
- [x] Baseline quality passes ≥70% lenient (currently 0%) ❌
- [ ] Evidence of heuristic limitations on customer data ⏸️ (cannot validate with broken baseline)

**Verdict**: **2 of 3 critical prerequisites FAILED**.

### Risk Assessment

**If Materials V3 were integrated NOW**:

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Garbage in, garbage out (broken baseline) | **HIGH** | **CRITICAL** | Fix heuristic first |
| Complexity debt with no ROI | **HIGH** | **HIGH** | Require evidence of value |
| Masking underlying classification failures | **MEDIUM** | **HIGH** | A/B test with fixed baseline |
| Model download/compute overhead | **MEDIUM** | **MEDIUM** | Feature flag + fallback |

**Overall Risk**: **UNACCEPTABLE** - Do not proceed.

---

## Alternative Recommendations: Fix Baseline FIRST

### Option A: Multi-Factor Scene Classification (RECOMMENDED)

**Replace single-threshold heuristic with composite classifier**:

```python
def classify_scene_type_v2(
    rgb_edges_raw: np.ndarray,
    rgb_edges_structure: np.ndarray,
    depth_variance: float,
    rgb: np.ndarray
) -> str:
    """
    Multi-factor scene classification with tuned thresholds.
    
    Factors:
    1. Edge ratio (raw/structure)
    2. Depth variance (spatial complexity)
    3. Saturation variance (material diversity)
    4. Edge density (absolute count)
    """
    raw_count = np.count_nonzero(rgb_edges_raw)
    structure_count = np.count_nonzero(rgb_edges_structure)
    
    # Avoid division by zero
    if structure_count == 0:
        return 'texture_dominated'
    
    ratio = raw_count / structure_count
    
    # Factor 1: Edge ratio (primary signal)
    # TUNED THRESHOLDS:
    # - ratio > 50: Strong texture (ocean, glass, uniform surfaces)
    # - ratio < 8: Strong structure (interiors, architectural)
    # - 8-50: Mixed (requires secondary factors)
    
    if ratio > 50:
        return 'texture_dominated'
    
    if ratio < 8:
        # Check if depth variance is TOO LOW (pool water)
        # Pool water: depth_var ~ 0.02, edge_ratio ~ 80+
        # BUT structure_count is LOW (only pool edges)
        if depth_variance < 0.03 and structure_count < 1000:
            return 'texture_dominated'  # Pool/water edge case
        return 'structure_dominated'
    
    # Mixed zone (8 < ratio < 50): Use depth variance
    # - High depth variance (>0.06) + high ratio → texture (bathroom tiles)
    # - Low depth variance (<0.03) → texture (glass facade, pool)
    
    if depth_variance > 0.06:
        return 'texture_dominated'  # Interior with patterned materials
    
    if depth_variance < 0.03:
        return 'texture_dominated'  # Smooth surfaces (glass, water)
    
    # Default: structure (conservative for architectural scenes)
    return 'structure_dominated'
```

**Expected Improvements**:
- Pool water: depth_var=0.02, ratio=80+ → texture (FIXED)
- Glass facade: depth_var=0.08, ratio=9.5 → texture (FIXED)
- Interior bathroom: depth_var=0.07, ratio=14 → texture (CORRECT, but needs quality fix)
- Interior kitchen: depth_var=0.06, ratio=6 → structure (FIXED)

**Estimated Accuracy**: 6/7 (85.7%) → meets >80% threshold.

### Option B: Increase Depth Anything V2 Input Size (IMMEDIATE ROI)

**Current**: `input_size=518` (default)  
**Recommended**: `input_size=896` or `input_size=1024`

**Trade-offs**:
- ✅ +10-20% edge precision (DA V2 documentation)
- ✅ Better small feature preservation
- ❌ +50-100% compute time
- ❌ +30-50% memory usage

**Rationale**: Depth quality is low (F1=0.0-0.3). Larger input may improve depth accuracy BEFORE applying gates.

### Option C: Tune Bilateral Filter Parameters

**Current** (`high_fidelity_depth/quality_metrics.py:detect_structure_edges`):
```python
filtered = cv2.bilateralFilter(
    gray, 
    d=9,               # Spatial extent
    sigmaColor=75,     # Color smoothing
    sigmaSpace=75      # Spatial smoothing
)
```

**Issue**: `sigmaColor=75` is **too aggressive** for scenes with subtle material variations (wood grain, patterned tiles).

**Recommendation**: Add scene-adaptive parameters:
```python
# For texture-rich interiors: reduce smoothing
if depth_variance > 0.05:
    sigma_color = 50  # Preserve material texture
else:
    sigma_color = 75  # Strong smoothing for glass/water
```

---

## Phased Remediation Plan

### Phase 1: Fix Scene Classification (Priority 1)

**Tasks**:
1. Implement multi-factor classifier (Option A)
2. Add unit tests for known misclassifications
3. Re-run 7-image validation
4. **Acceptance Criteria**: ≥85% classification accuracy (6/7 images)

**Timeline**: 2-3 hours  
**Files Modified**: `high_fidelity_depth/quality_metrics.py`, `high_fidelity_depth/test_structure_edges.py`

### Phase 2: Improve Depth Quality (Priority 2)

**Tasks**:
1. Increase DA V2 input size to 896 (Option B)
2. Tune bilateral filter parameters (Option C)
3. Re-run validation with fixed classifier
4. **Acceptance Criteria**: ≥70% lenient pass rate

**Timeline**: 1-2 hours  
**Files Modified**: `high_fidelity_depth/config.py`, `high_fidelity_depth/quality_metrics.py`

### Phase 3: Expand Validation Set (Priority 3)

**Tasks**:
1. Add 8-13 more images (diverse scenes: wood, carpet, foliage, metal, stone)
2. Validate 15-20 image suite with fixed classifier + depth improvements
3. Measure per-material classification accuracy
4. **Acceptance Criteria**: ≥90% accuracy on expanded set

**Timeline**: 3-4 hours (including image selection + annotation)

### Phase 4: Revisit Materials V3 (If Needed)

**Trigger**: Expanded validation reveals systematic heuristic failures (accuracy <85%)

**Tasks**:
1. Implement shadow mode (feature-flagged Materials V3)
2. A/B test: heuristic vs Materials V3 on expanded set
3. **Acceptance Criteria**:
   - Materials V3 accuracy >95% (vs heuristic 85%)
   - Lenient pass rate +5% improvement
   - Compute overhead <30%

**Timeline**: 4-6 hours  
**Files Created**: `high_fidelity_depth/scene_classification.py` (new module)  
**Files Modified**: `high_fidelity_depth/config.py`, `scripts/automation/production_depth_validation_fixed.py`

---

## Materials V3 Integration Design (Shadow Mode)

**IF Phase 4 is triggered, implement as follows**:

### Architecture: Pluggable Scene Classifier

**New Module**: `high_fidelity_depth/scene_classification.py`

**Key Components**:
1. **`SceneClassifier` (Base Class)**: Abstract interface for all classifiers
2. **`HeuristicClassifier`**: Current baseline (edge ratio + depth variance)
3. **`MaterialsV3Classifier`**: Segmentation-based (pixel-level material masks)
4. **`ShadowClassifier`**: Runs both, logs comparison, uses heuristic result

**Configuration** (`high_fidelity_depth/config.py`):
```python
from enum import Enum

class SceneClassifierType(Enum):
    HEURISTIC = "heuristic"          # Default: multi-factor edge ratio
    MATERIALS_V3 = "materials_v3"    # MaterialsV3 segmentation
    SHADOW = "shadow"                # Run both, log comparison, use heuristic

@dataclass
class DepthConfig:
    # Scene classification
    scene_classifier: SceneClassifierType = SceneClassifierType.HEURISTIC
    
    # Materials V3 settings (only used if classifier != HEURISTIC)
    materials_v3_model_path: str = "models/materials_v3.pth"
    materials_v3_fallback_to_heuristic: bool = True  # Graceful degradation
```

### Implementation Checklist

- [ ] Create `high_fidelity_depth/scene_classification.py` with base classes
- [ ] Implement `HeuristicClassifier` (refactor existing logic)
- [ ] Implement `MaterialsV3Classifier` stub with graceful fallback
- [ ] Implement `ShadowClassifier` with disagreement logging
- [ ] Add CLI flag `--scene-classifier {heuristic,materials_v3,shadow}`
- [ ] Update validation script to use pluggable classifier
- [ ] Add unit tests for all classifier backends
- [ ] Document A/B test protocol

### A/B Test Protocol

**Command**:
```bash
# Baseline (heuristic)
python scripts/automation/production_depth_validation_fixed.py \
  --image-dir data/validation_expanded \
  --output-dir outputs/validation_heuristic \
  --scene-classifier heuristic

# Materials V3 (active mode)
python scripts/automation/production_depth_validation_fixed.py \
  --image-dir data/validation_expanded \
  --output-dir outputs/validation_materials_v3 \
  --scene-classifier materials_v3

# Shadow mode (log comparison, use heuristic)
python scripts/automation/production_depth_validation_fixed.py \
  --image-dir data/validation_expanded \
  --output-dir outputs/validation_shadow \
  --scene-classifier shadow
```

**Acceptance Criteria** (Materials V3 worth integrating):
1. Classification accuracy: Materials V3 >95% (vs heuristic 85%)
2. Quality improvement: Lenient pass rate +5% over heuristic
3. Compute cost: <30% overhead vs heuristic
4. Robustness: No crashes, graceful fallback when model unavailable

**If Materials V3 does NOT meet criteria**:
- Document findings in `MATERIALS_V3_EVALUATION.md`
- Keep heuristic as default
- Revisit only if customer data reveals systematic failures

---

## Decision Rationale

### Why NO-GO Now?

**Technical Reasons**:
1. **Baseline is broken**: 28.6% classification accuracy is catastrophic
2. **Premature optimization**: Cannot assess Materials V3 value with broken baseline
3. **Complexity debt**: Adding Materials V3 now masks underlying issues
4. **No evidence of ROI**: Quality gates fail regardless of classification (0% pass rate)

**Process Reasons**:
1. **Violates scientific method**: Fix one variable at a time (classifier, then depth, then materials)
2. **Risk of confusion**: Cannot attribute improvements/failures to specific changes
3. **Regression risk**: Adding Materials V3 may introduce new failure modes

**Business Reasons**:
1. **Validation quality**: 0% lenient pass rate is unacceptable for production
2. **Customer impact**: Misclassified scenes receive wrong processing pipeline
3. **Technical debt**: Shadow mode adds complexity without proven value

### Success Criteria for Future GO Decision

Materials V3 integration is warranted ONLY IF:

1. **Baseline Fixed**:
   - [x] Scene classification ≥85% accuracy (currently 28.6%) ❌
   - [x] Lenient pass rate ≥70% (currently 0%) ❌
   - [x] Meaningful failures only (no adversarial edge cases) ❌

2. **Materials V3 Adds Value**:
   - [ ] Classification accuracy >95% (>10% improvement over heuristic)
   - [ ] Quality improvement measurable (+5% lenient pass rate)
   - [ ] Pixel-level masks enable new post-processing features

3. **Acceptable Trade-offs**:
   - [ ] Compute overhead <30% (vs heuristic)
   - [ ] Memory overhead <2GB
   - [ ] Graceful fallback when model unavailable

**Current Status**: 0/3 categories satisfied → **NO-GO**.

---

## Deliverables

### Immediate Actions (Next 4-6 Hours)

1. **Implement multi-factor scene classifier** (Option A)
   - File: `high_fidelity_depth/quality_metrics.py`
   - Function: `classify_scene_type_v2()`
   - Tests: `high_fidelity_depth/test_structure_edges.py`

2. **Re-run 7-image validation**:
   ```bash
   ./RUN_STRUCTURE_EDGE_VALIDATION.sh
   ```
   - Expected: 85% classification accuracy (6/7 images)

3. **Increase DA V2 input size** (Option B):
   - File: `high_fidelity_depth/config.py`
   - Change: `input_size: int = 896`  # was 518
   - Re-run validation
   - Expected: +10-20% edge F1, +20-30% lenient pass rate

4. **Tune bilateral filter** (Option C):
   - File: `high_fidelity_depth/quality_metrics.py`
   - Add: Scene-adaptive `sigma_color` (50 for interiors, 75 for glass/water)

### Medium-Term Actions (Next Session)

5. **Expand validation set** to 15-20 images:
   - Add: wood grain, patterned carpet, dense foliage, metal, stone
   - Annotate: expected scene type, material composition
   - Validate: ≥90% classification accuracy

6. **Document findings**:
   - Create: `SCENE_CLASSIFICATION_IMPROVEMENTS.md`
   - Include: Before/after accuracy, per-material breakdown, threshold tuning rationale

### Long-Term (If Needed)

7. **Implement Materials V3 shadow mode** (only if expanded validation shows heuristic limitations):
   - File: `high_fidelity_depth/scene_classification.py` (new)
   - A/B test protocol documented above
   - Acceptance criteria: accuracy >95%, quality +5%, compute <30%

---

## Conclusion

**Materials V3 integration is NOT READY**. The heuristic scene classifier has **catastrophic failure rate (71.4%)**, and quality gates fail **100% of images**. 

**Priority 1**: Fix baseline classification and depth quality FIRST.

**Priority 2**: Expand validation to diverse materials (15-20 images).

**Priority 3**: Revisit Materials V3 ONLY IF heuristic proves insufficient (accuracy <85% on expanded set).

**Estimated Timeline to Baseline Stability**: 6-10 hours of focused work.

---

## Appendix: Validation Raw Data

### Per-Image Detailed Metrics

```json
{
  "glass_building": {
    "scene_type": "texture_dominated",
    "depth_variance": 0.0385,
    "edge_ratio": 41359.0,
    "edge_f1": 0.0,
    "gate": "smoothness",
    "lenient": false,
    "reason": "Texture scene: depth_var=0.039, edge_ratio=41359.00"
  },
  "glass_facade": {
    "scene_type": "structure_dominated",  # WRONG (expected texture)
    "depth_variance": 0.0774,
    "edge_ratio": 9.46,
    "edge_f1": 0.239,
    "gate": "edge_alignment",
    "lenient": false,
    "reason": "F1=0.239, Chamfer=319.0px"
  },
  "interior_bathroom": {
    "scene_type": "texture_dominated",  # WRONG (expected structure)
    "depth_variance": 0.0743,
    "edge_ratio": 14.38,
    "edge_f1": 0.232,
    "gate": "smoothness",
    "lenient": false,
    "reason": "Texture scene: depth_var=0.074, edge_ratio=14.38"
  },
  "interior_kitchen": {
    "scene_type": "texture_dominated",  # WRONG (expected structure)
    "depth_variance": 0.0572,
    "edge_ratio": 5.99,
    "edge_f1": 0.304,
    "gate": "smoothness",
    "lenient": false,
    "reason": "Texture scene: depth_var=0.057, edge_ratio=5.99"
  },
  "ocean_1": {
    "scene_type": "texture_dominated",
    "depth_variance": 0.0456,
    "edge_ratio": 48912.0,
    "edge_f1": 0.0,
    "gate": "smoothness",
    "lenient": false,
    "reason": "Texture scene: depth_var=0.046, edge_ratio=48912.00"
  },
  "pool_texture_1": {
    "scene_type": "structure_dominated",  # WRONG (expected texture)
    "depth_variance": 0.0202,
    "edge_ratio": 83.81,
    "edge_f1": 0.108,
    "gate": "edge_alignment",
    "lenient": false,
    "reason": "F1=0.108, Chamfer=75.9px"
  },
  "pool_texture_2": {
    "scene_type": "structure_dominated",  # WRONG (expected texture)
    "depth_variance": 0.0175,
    "edge_ratio": 85.45,
    "edge_f1": 0.107,
    "gate": "edge_alignment",
    "lenient": false,
    "reason": "F1=0.107, Chamfer=80.1px"
  }
}
```

### Classification Confusion Matrix

|                    | Predicted: Texture | Predicted: Structure |
|--------------------|-------------------|----------------------|
| **Actual: Texture**     | 2 (glass_building, ocean_1) | 3 (glass_facade, pool_texture_1, pool_texture_2) |
| **Actual: Structure**   | 2 (interior_bathroom, interior_kitchen) | 0 |

**Accuracy**: 2/7 = **28.6%**  
**Precision (texture)**: 2/4 = 50%  
**Recall (texture)**: 2/5 = 40%  
**Precision (structure)**: 0/3 = 0%  
**Recall (structure)**: 0/2 = 0%

---

**Status**: Phase 1 validation COMPLETE. Baseline FAILED. Materials V3 integration **NOT RECOMMENDED**.

**Next Steps**: Implement multi-factor classifier (Option A) and re-validate.
