# Materials V3 PR-3A: Metrics + Taxonomy Foundation

**Status:** Complete ✅  
**Date:** December 13, 2025  
**Purpose:** Foundation for objective edge-quality evaluation and material identity normalization

---

## Overview

PR-3A establishes the foundation for Materials V3 by introducing:

1. **Boundary metrics** — objective edge quality measurement (replaces "mean IoU" with boundary-focused scores)
2. **Taxonomy normalization** — canonical material keys + semantic mapping (fixes Stage 6 "water/pool/foliage" identity issues)

These are the **prerequisites** for:

* PR-3B: Materials V3 gating engine (decides when/where to refine)
* PR-3C: Stage 6 A/B rerun with boundary metrics (the real promotion decision)

---

## What Was Implemented

### 1. Boundary Metrics Module (`lux_depth_v2/metrics/boundary_metrics.py`)

**Key metrics:**

* **Boundary F1 (BF1)**: precision/recall on edge-band pixels only
* **Trimap IoU**: separate IoU for core / boundary / background regions
* **Edge alignment**: correlation with image gradients

**Why this matters:**

* Mean IoU treats all pixels equally; BF1 focuses on the edge band where EfficientSAM refinement actually matters
* Trimap IoU distinguishes interior quality from boundary quality
* Edge alignment uses image gradients as a proxy for "true" edges (no manual ground truth needed)

**Usage example:**

```python
from lux_depth_v2.metrics.boundary_metrics import compute_full_boundary_metrics

metrics = compute_full_boundary_metrics(
    pred_mask=refined_mask,
    ref_mask=base_mask,
    image_gradients=sobel_magnitude,
    band_width_px=5
)

print(f"Boundary F1: {metrics.boundary_f1:.3f}")
print(f"Edge alignment: {metrics.edge_alignment:.3f}")
```

---

### 2. Taxonomy Normalization (`lux_depth_v2/materials_v3_taxonomy.py`)

**Problem it solves:**

Stage 6 showed inconsistent material identity:

* Kitchen: "water" missing or inconsistent
* Pool: sometimes "pool_water", sometimes "water_surface"
* Aerial: "tree" vs "foliage" vs "vegetation"

This made refinement logic brittle and unpredictable.

**Solution:**

* **Canonical keys** (e.g., `water`, `glass`, `foliage`) — stable, lowercase, snake_case
* **Semantic → canonical mapping** — many-to-one (e.g., `pool_water`, `ocean`, `sea` → `water`)
* **Per-material metadata** — thresholds, refinement priority, specular sensitivity

**Usage example:**

```python
from lux_depth_v2.materials_v3_taxonomy import (
    normalize_material_name,
    get_material_metadata,
    should_refine_material,
)

# Normalize semantic names
assert normalize_material_name("pool_water") == "water"
assert normalize_material_name("window") == "glass"
assert normalize_material_name("tree") == "foliage"

# Get metadata (with thresholds + refinement hints)
meta = get_material_metadata("glass")
# meta.confidence_threshold == 0.40
# meta.refinement_priority == 10
# meta.benefits_from_effsam == True

# Decide refinement strategy
should_refine_material("glass", refinement_strategy="canary")  # → True
should_refine_material("wood", refinement_strategy="canary")   # → False
```

---

## Test Coverage

**40 tests total, all passing:**

* 13 boundary metrics tests (F1, trimap IoU, edge alignment, edge cases)
* 27 taxonomy tests (normalization, metadata, refinement decisions, consistency)

**Run tests:**

```bash
pytest lux_depth_v2/tests/test_boundary_metrics.py -v
pytest lux_depth_v2/tests/test_materials_v3_taxonomy.py -v
```

---

## Design Decisions

### 1. Why boundary metrics instead of mean IoU?

Mean IoU is dominated by large interior regions. A mask can have 0.95 IoU and still have terrible edges.

**BF1 focuses on the perimeter** where refinement actually matters, giving you a metric that:

* correlates with visual edge quality,
* is sensitive to haloing and edge spill,
* can detect when EfficientSAM improves edges even if SegFormer core is correct.

### 2. Why canonical keys + mapping instead of "just fix the names"?

Fixing names upstream (in SegFormer output) isn't possible — the model outputs what it outputs.

Canonical keys let you:

* handle variant names without duplicating logic,
* update mappings as new segmentation models appear,
* maintain stable configuration/presets even when model taxonomy changes.

### 3. Why metadata per material?

Different materials have different characteristics:

* **Glass**: low confidence, high refinement priority, specular-sensitive
* **Water**: very low confidence, high priority, variable
* **Wood**: high confidence, low priority, not specular
* **Sky**: high confidence, **never refine** (priority 0)

Metadata encodes these differences so Materials V3 gating can make intelligent per-class decisions.

---

## What This Enables (PR-3B + PR-3C)

### PR-3B: Materials V3 Gating Engine

Now that you have:

* canonical material keys,
* per-material metadata (thresholds, priorities),
* refinement decision logic (`should_refine_material`),

…you can implement a **gating engine** that decides:

* which classes to refine per scene,
* where to apply refinement (core vs boundary),
* when to skip refinement (size guards, coverage, complexity).

### PR-3C: Stage 6 A/B Rerun with Boundary Metrics

Replace the promotion gate from:

* ❌ "IoU vs SegFormer > 0.30" (treats SegFormer as truth)

to:

* ✅ "Boundary F1 improved AND no artifacts" (objective edge quality)

This lets you make the **real** EfficientSAM promotion decision based on edge improvement, not pixel-level agreement with SegFormer.

---

## Files Changed

### New modules

* `lux_depth_v2/metrics/__init__.py`
* `lux_depth_v2/metrics/boundary_metrics.py`
* `lux_depth_v2/materials_v3_taxonomy.py`

### New tests

* `lux_depth_v2/tests/test_boundary_metrics.py`
* `lux_depth_v2/tests/test_materials_v3_taxonomy.py`

### Documentation

* `docs/SESSIONS/materials-v3/2025-12-13_PR3A_COMPLETE.md` (this file)

---

## Next Steps

1. **Commit PR-3A** to `main` (safe, no behavior changes)
2. **PR-3B**: Implement Materials V3 gating engine
3. **PR-3C**: Update Stage 6 A/B script to emit boundary metrics
4. **Decision**: Promote EfficientSAM to default APEX **only if** BF1 improves consistently

---

## Acceptance Criteria (All Met ✅)

* ✅ Boundary metrics module complete with tests
* ✅ Taxonomy normalization module complete with tests
* ✅ All tests passing (40/40)
* ✅ No dependencies on ML models (pure numpy + scipy)
* ✅ No behavior changes to existing pipeline
* ✅ Documentation complete

---

**PR-3A is ready to merge.**
