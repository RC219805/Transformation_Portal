# PR-4C Planning: Structural Separation of Decisions

## Date: 2025-12-14

## Current State (PR-4B)

**What Works**:
- Core/edge band extraction (deterministic, pixel-based)
- Per-class stats: coverage, mean_conf, edge_conf, core_conf
- Response strength planning with attenuation
- Canary-based refinement decisions

**What's Problematic**:
- `should_refine` does double duty:
  - "Should we run EfficientSAM edge refinement?"
  - "Should we apply pixel ops?"
- No edge signals (boundary pixels, gradient alignment)
- Conflates two independent decisions

**Risk**: Repeating EfficientSAM/foliage mistake if we expand without structural fix.

## PR-4C Goal

Expand response planning to additional materials (report-first) with **explicit, auditable decision fields** without changing pixels.

### Non-Goals
- ❌ No new pixel operations
- ❌ No expanding EfficientSAM beyond canary
- ❌ No training / model swaps

## Proposed Schema (v3.1)

```json
{
  "materials_v3_response_plan": {
    "version": "v3.1",
    "config": {...},
    "per_class": {
      "glass": {
        "present": true,
        "coverage_px": 874123,
        "coverage": 0.0432,
        "mean_conf": 0.806,
        "edge_conf": 0.525,
        "core_conf": 0.834,
        
        "strengths": {
          "core": 0.90,
          "edge": 0.70
        },
        
        "refinement": {
          "should_refine_edges": false,
          "reason": "confidence_already_high",
          "strategy": "canary"
        },
        
        "pixel_ops": {
          "should_apply": false,
          "reason": "confidence_already_high",
          "eligible": true
        },
        
        "edge_signals": {
          "boundary_pixels": 18250,
          "edge_alignment": 0.214,
          "notes": []
        },
        
        "risk_flags": ["high_confidence_skip"]
      },
      
      "wood": {
        "present": true,
        "pixel_ops": {
          "should_apply": false,
          "eligible": false,
          "reason": "no_implementation",
          "recommended_ops": ["microcontrast"]
        }
      }
    },
    
    "summary": {
      "present_classes": ["glass", "wood", "stone"],
      "eligible_for_pixel_ops": ["glass"],
      "eligible_for_refinement": ["glass"],
      "skipped_reasons_histogram": {
        "confidence_already_high": 1,
        "no_implementation": 2
      }
    }
  }
}
```

## Key Changes

### 1. Two Independent Decisions

**Refinement** (EfficientSAM):
- `refinement.should_refine_edges`
- Only for canary classes: {glass, foliage, water}
- Gated by edge signals (boundary_pixels, edge_alignment)

**Pixel Ops**:
- `pixel_ops.should_apply`
- Independent of refinement
- Glass has implementation (eligible=true)
- Others are report-only (eligible=false, recommended_ops=[...])

### 2. Edge Signals (First-Class)

**New Fields**:
- `boundary_pixels`: Count of edge pixels
- `edge_alignment`: Alignment with image gradients (0-1)
- Computed via `scipy.ndimage.sobel` (no new deps)

**Purpose**:
- Avoid foliage EfficientSAM regression
- Objective signal without ground truth
- Report-only in PR-4C

### 3. Expanded Material Coverage

**Strengths for**:
- wood, stone, metal, fabric, stucco (neutral defaults)

**Report for all**, apply pixel ops only where implemented.

## Decision Rules (Report-Only)

### A) Refinement Decisions (EfficientSAM)

**Eligible if**:
- class in canary set {glass, foliage, water}
- coverage_px ≥ min_coverage_px
- mean_conf ≥ min_mean_conf
- boundary_pixels ≥ 250 (learned from foliage)
- edge_alignment ≥ 0.10 (gradient-based)

**Recommend refine if**:
- eligible AND mean_conf < refine_conf_ambiguity_threshold

### B) Pixel Ops Decisions

**Glass Rule** (only implemented class):
- eligible = present AND coverage_px ≥ 1000
- should_apply = eligible AND (mean_conf < 0.80 OR edge_conf < 0.55)

**Reasons**:
- `confidence_already_high` (skip)
- `low_edge_confidence` (apply candidate)
- `low_mean_confidence` (apply candidate)
- `below_coverage_threshold` (skip)

**Other Materials**:
- eligible = false
- recommended_ops = ["microcontrast"] (or appropriate op type)
- No should_apply (report-only)

## Implementation Plan

### File Changes

1. **`materials_v3_response.py`**:
   - Add `compute_edge_signals()` function
   - Split `decide_should_refine()` → `decide_refinement()` + `decide_pixel_ops()`
   - Update `generate_response_plan()` schema to v3.1
   - Add material strength defaults for wood/stone/metal/fabric/stucco

2. **`materials_v3.py`**:
   - Pass RGB image to response planner (for edge signals)
   - Emit new schema version

3. **Tests**:
   - Unit tests for new schema fields
   - `pixel_ops.should_apply` toggles based on mean_conf/edge_conf
   - boundary_pixels guard prevents degenerate masks
   - Edge signals computed correctly

### No Pipeline Changes
- Keep tests torch-free (use synthetic masks)
- No ML-stage integration required for PR-4C

## Why This Prevents EfficientSAM Repeat

1. **Explicit separation**: Can't conflate "refine edges" with "apply pixel ops"
2. **Edge signals**: Objective boundary/gradient data prevents blind decisions
3. **Report-first**: Expand coverage without pixel risk
4. **Audit trail**: Clear reasons for every decision

## Success Criteria for PR-4C

- ✅ Schema v3.1 emitted for all present materials
- ✅ Two decision blocks independent
- ✅ Edge signals computed and reported
- ✅ Tests pass (no torch required)
- ✅ No pixel changes (report-only)
- ✅ CI green

## Next PRs After PR-4C

**PR-4D**: Apply pixel ops to second material (data-driven choice from PR-4C reports)
**PR-5**: Expand EfficientSAM refinement with edge-signal gates

---

**Status**: Ready to implement
**Risk**: Low (report-only, no pixel changes)
**ROI**: High (prevents architectural debt, enables data-driven expansion)
