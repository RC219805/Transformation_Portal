# CRITICAL BLOCKER: Validation Script Integration

**Status**: BLOCKED
**Priority**: P0 (Production-blocking)
**Date**: 2025-12-18

## Problem

The `production_depth_validation_fixed.py` script is NOT calling the V2 classifier implemented in `quality_metrics.py`.

**Evidence**:
- All 18 images: `scene_type: null`
- All metrics: `edge_f1: null`, `lenient_pass: null`, `strict_pass: null`
- Script completed without errors (silent failure)

## Root Cause

The validation script is using an OLD version of `validate_depth_quality()` that doesn't include:
1. V2 classifier call (`classify_scene_type_v2()`)
2. Structure-aware edge extraction (`extract_structure_edges()`)
3. Conditional quality gates (`evaluate_quality_gates()`)

## Fix Required

Update `scripts/automation/production_depth_validation_fixed.py` to call the NEW functions:

```python
# OLD (current - broken):
from high_fidelity_depth.quality_metrics import validate_depth_quality

metrics = validate_depth_quality(depth_map, rgb_image)

# NEW (required):
from high_fidelity_depth.quality_metrics import (
    validate_depth_quality,
    classify_scene_type_v2,
    extract_structure_edges,
    detect_edges
)

# Extract edges
rgb_edges_raw = detect_edges(rgb_image, mode='rgb')
rgb_edges_structure = extract_structure_edges(rgb_image)

# Classify scene
scene_type, scene_meta = classify_scene_type_v2(
    rgb_edges_raw, rgb_edges_structure, depth_map
)

# Validate with structure-aware edges
metrics = validate_depth_quality(
    depth_map,
    rgb_image,
    use_structure_edges=True  # Enable V2 classifier
)

# Apply conditional gates
from scripts.automation.production_depth_validation_fixed import evaluate_quality_gates
gates = evaluate_quality_gates(metrics.to_dict(), scene_type)
```

## Impact

**BLOCKS**:
- ✗ Classifier V2 validation (can't test 85.7% → 90%+ accuracy claim)
- ✗ Confusion matrix generation
- ✗ Materials V3 readiness assessment
- ✗ Production deployment

**Current State**:
- 18-image dataset curated ✓
- Classifier V2 implemented ✓ (6/6 tests passing)
- Validation script integration: ✗ BROKEN

## Timeline

**Estimated Fix**: 30-60 minutes
1. Update imports (5 min)
2. Add classifier calls (15 min)
3. Update metrics serialization (10 min)
4. Rerun validation (30 min)

## Acceptance Criteria

After fix:
- [ ] scene_type populated for all images
- [ ] Classification accuracy ≥85% (15-16/18 correct)
- [ ] Lenient pass rate ≥70% (13/18)
- [ ] Confusion matrix generated

## Next Session Entry Point

1. Fix `production_depth_validation_fixed.py` integration
2. Rerun validation on 18 images
3. Generate confusion matrix
4. If accuracy ≥90%: Lock thresholds, proceed to DA V2 input_size sweep
5. If accuracy <90%: Analyze misclassifications, tune thresholds

---

**This is the ONLY blocker preventing Phase 1B completion.**
