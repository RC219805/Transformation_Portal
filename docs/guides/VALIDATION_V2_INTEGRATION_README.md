# Production Depth Validation - V2 Classifier Integration

## Critical Fix: Silent Failure Prevention

**What was broken**: 18-image validation completed "successfully" with all metrics showing `scene_type: null`, `edge_f1: null`, `lenient_pass: null`. The script didn't call the V2 classifier and wrote placeholder nulls instead of failing fast.

**What was fixed**:
1. ✅ **V2 Classifier Integration**: Script now explicitly calls `classify_scene_type_v2()` before quality validation
2. ✅ **Fail-Fast Validation**: `validate_metrics_complete()` checks for null values and exits non-zero
3. ✅ **CLI Ergonomics**: Both `--input-dir` and `--image-dir` are accepted
4. ✅ **Metrics Serialization**: Proper dataclass serialization with V2 classifier metadata
5. ✅ **Integration Test**: pytest test that catches silent failures

---

## Two-Run Recovery Plan

### Run 0: Smoke Test (2 images, ~5 min)

**Purpose**: Verify V2 classifier integration before running full validation.

```bash
# Run smoke test
./run_smoke_test.sh
```

**Pass Criteria**:
- ✅ `scene_type` is a string (not null)
- ✅ `edge_f1` is a number (not null)
- ✅ `lenient_pass` is a boolean (not null)
- ✅ `classification_factors` populated with ratio, depth_variance, decision_rule

**If any are null → STOP, fix integration, don't proceed to Run 1**

---

### Run 1: Full 18-Image Validation (~30 min)

**Only run after smoke test passes!**

```bash
# Full validation
python scripts/automation/production_depth_validation_fixed.py \
  --input-dir data/validation_expanded \
  --output-dir outputs/validation_v2_$(date +%Y%m%d_%H%M%S) \
  --tile-size 1024 \
  --overlap 192
```

**Generate Confusion Matrix**:

```bash
# After validation completes
python scripts/validation/generate_confusion_matrix.py \
  --output-dir outputs/validation_v2_YYYYMMDD_HHMMSS
```

**Expected Results**:
- Classification accuracy ≥85% (15-16/18 correct)
- Lenient pass rate ≥70% (13/18)
- No null placeholders in any metrics file

---

## Integration Test

**Run before any validation**:

```bash
# Integration test (catches silent failures)
pytest tests/integration/test_validation_script.py -v
```

This test:
- Creates 2 synthetic test images
- Runs validation script
- Verifies all required fields are populated (not null)
- Checks field types (str, float, bool)
- Validates V2 classifier metadata presence

---

## What Changed in the Script

### Before (BROKEN)
```python
# OLD - doesn't call V2 classifier
metrics = validate_depth_quality(depth_map, rgb_image)
# Returns EdgeMetrics with scene_type=None (never classified)
```

### After (FIXED)
```python
# NEW - calls V2 classifier stack
from high_fidelity_depth.quality_metrics import (
    classify_scene_type_v2,
    extract_structure_edges,
    detect_edges
)

# 1. Extract edges (both raw and structure)
rgb_edges_raw = detect_edges(gray)
rgb_edges_structure = extract_structure_edges(gray)

# 2. Classify scene using V2 multi-factor classifier
scene_type, scene_metadata = classify_scene_type_v2(
    rgb_edges_raw=rgb_edges_raw,
    rgb_edges_structure=rgb_edges_structure,
    depth_map=depth
)

# 3. Validate depth quality
metrics = validate_depth_quality(
    depth_map=depth,
    rgb_image=rgb,
    use_structure_edges=True
)

# 4. Build metrics dict with V2 classifier data
metrics_dict = {
    'edge_f1': float(metrics.edge_f1),
    'scene_type': scene_type,
    'classification_factors': {
        'ratio': float(scene_metadata['ratio']),
        'depth_variance': float(scene_metadata['depth_variance']),
        'decision_rule': scene_metadata['decision']
    },
    'lenient_pass': bool(lenient_pass),
    'strict_pass': bool(strict_pass),
}

# 5. VALIDATE BEFORE WRITING (fail fast)
validate_metrics_complete(metrics_dict, image_name)

# 6. Save atomic
save_metrics_atomic(metrics_dict, output_path)
```

---

## Fail-Fast Validation

```python
def validate_metrics_complete(metrics_dict: dict, image_name: str) -> None:
    """
    Fail fast if metrics are incomplete.

    Production validators MUST NOT write null placeholders.
    """
    required_fields = [
        'scene_type',
        'edge_f1',
        'lenient_pass',
        'strict_pass',
        'classification_factors'
    ]

    missing = [f for f in required_fields if metrics_dict.get(f) is None]

    if missing:
        raise ValueError(
            f"Incomplete metrics for {image_name}: missing {missing}. "
            "This is a P0 integration bug."
        )
```

**Script exits non-zero if validation fails → no more silent failures**

---

## Scene Classification Decision Tree (V2)

The V2 classifier uses **3 factors**:

1. **Edge ratio** (raw/structure): texture indicator
2. **Depth variance**: smoothness indicator
3. **Edge density**: structural complexity

**Decision rules** (tuned on validation data):

```
Rule 1: edge_density < 0.005 → texture_dominated
  (smooth surfaces: water, glass, ocean)

Rule 2: ratio > 10.0 → texture_dominated
  (patterned interiors with high texture)

Rule 3: edge_density > 0.02 AND 3 <= ratio <= 10 → structure_dominated
  (structured interiors: kitchens, bathrooms)

Rule 4: ratio < 2.0 AND depth_var < 0.025 → texture_dominated
  (smooth texture: pools)

Rule 5: ratio < 2.0 AND edge_density > 0.008 → texture_dominated
  (glass/reflective surfaces)

Rule 6: 2 <= ratio <= 5 AND edge_density > 0.015 → structure_dominated

Rule 7: Fallback → ratio <= 3.0 ? structure : texture
```

---

## Expected Scene Types (Ground Truth)

Update `scripts/validation/generate_confusion_matrix.py` with actual scene types:

```python
EXPECTED_SCENE_TYPES = {
    # Structure-dominated (interiors with strong lines)
    '750Picacho_Kitchen': 'structure_dominated',
    '750Picacho_GreatRoom': 'structure_dominated',
    '750Picacho_MasterBath': 'structure_dominated',

    # Texture-dominated (water, glass, reflective)
    '750Picacho_Pool': 'texture_dominated',
    '750Picacho_Ocean': 'texture_dominated',
    'Montecito-Shores-10': 'texture_dominated',

    # Add all 18 images...
}
```

---

## Acceptance Criteria

**Smoke Test (Run 0)**:
- [ ] 2/2 images: scene_type populated
- [ ] 2/2 images: edge_f1 numeric
- [ ] 2/2 images: lenient_pass boolean
- [ ] No null placeholders

**Full Validation (Run 1)**:
- [ ] 18/18 images: scene_type populated
- [ ] Classification accuracy ≥85% (15-16/18 correct)
- [ ] Lenient pass rate ≥70% (13/18)
- [ ] Confusion matrix generated
- [ ] Integration test passing

**Materials V3 Go/No-Go**:
- [ ] ✅ Smoke test passes (Run 0)
- [ ] ✅ Full validation passes (Run 1)
- [ ] ✅ Classification accuracy ≥90%
- [ ] ✅ Baseline stable and frozen

---

## Troubleshooting

**"scene_type is null"**:
- V2 classifier not called
- Check imports: `classify_scene_type_v2`, `extract_structure_edges`
- Verify `use_structure_edges=True` in `validate_depth_quality()`

**"Script exits with ValueError"**:
- ✅ **This is expected behavior** if metrics are incomplete
- Fail-fast prevents silent failures
- Fix the integration issue, don't disable the check

**"Integration test fails"**:
- Run `pytest tests/integration/test_validation_script.py -v -s` for debug output
- Check that script is using updated version (not cached)
- Verify imports resolve correctly

---

## Files Changed

1. `scripts/automation/production_depth_validation_fixed.py`
   - Added V2 classifier integration
   - Added `validate_metrics_complete()` fail-fast check
   - Updated CLI to accept `--image-dir` alias
   - Improved metrics serialization

2. `tests/integration/test_validation_script.py` (NEW)
   - Integration test for V2 classifier
   - Catches silent failures
   - Validates metrics structure

3. `run_smoke_test.sh` (NEW)
   - 2-image smoke test
   - Pre-flight check before full validation
   - Validates metrics content

4. `scripts/validation/generate_confusion_matrix.py` (NEW)
   - Confusion matrix generation
   - Classification accuracy reporting
   - Quality gate summary by scene type

---

## Next Steps

After successful validation (≥85% accuracy):

1. ✅ Freeze baseline configuration
2. ✅ Document V2 classifier performance
3. ✅ Proceed to Materials V3 integration
4. ✅ Update integration tests for Materials V3

**DO NOT proceed to Materials V3 until smoke test + full validation pass!**
