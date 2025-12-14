# Water Detection: Ground Truth Schema (Final + Future-Proof)

## Schema Design

**Two labels only**: `pool` and `ocean` (both are water)

**Negative controls**: `should_detect: false` entries for hard negatives (no third label)

**Mask storage**: Derived stats only, not full mask arrays

---

## Ground Truth JSON Format

```json
{
  "version": "v0",
  "created": "2025-12-14",
  "root": "data/water_v0/images",
  "labels": ["pool", "ocean"],
  "images": {
    "pool/pool_0001.jpg": {
      "label": "pool",
      "should_detect": true,
      "difficulty": "easy",
      "notes": "clear water, direct sun",
      "tags": []
    },
    "pool/pool_0012.jpg": {
      "label": "pool",
      "should_detect": true,
      "difficulty": "hard",
      "notes": "dark water, evening lighting",
      "tags": ["low-light", "reflection"]
    },
    "pool/neg_blue_wall_0001.jpg": {
      "label": "pool",
      "should_detect": false,
      "difficulty": "hard",
      "notes": "HARD NEGATIVE: blue painted wall",
      "tags": ["hard_negative", "blue_paint"]
    },
    "ocean/ocean_0001.jpg": {
      "label": "ocean",
      "should_detect": true,
      "difficulty": "medium",
      "notes": "calm water, overcast",
      "tags": []
    },
    "ocean/neg_blue_sky_0001.jpg": {
      "label": "ocean",
      "should_detect": false,
      "difficulty": "hard",
      "notes": "HARD NEGATIVE: blue sky through window",
      "tags": ["hard_negative", "sky"]
    }
  }
}
```

**Key fields**:
- `label`: `pool` or `ocean` (which folder, for organization)
- `should_detect`: `true` for water, `false` for hard negatives
- `difficulty`: `easy` | `medium` | `hard`
- `tags`: Track failure modes without label explosion

**Why this works**:
- Two labels only (pool/ocean folders)
- `should_detect: false` enables false trigger rate without a third label
- Hard negatives live in pool/ocean folders (keeps structure clean)

---

## Folder Layout

```
data/water_v0/
├── images/
│   ├── pool/
│   │   ├── pool_0001.jpg          # should_detect: true
│   │   ├── pool_0002.jpg          # should_detect: true
│   │   ├── neg_blue_wall_0001.jpg # should_detect: false
│   │   └── neg_blue_sky_0001.jpg  # should_detect: false
│   └── ocean/
│       ├── ocean_0001.jpg         # should_detect: true
│       ├── ocean_0002.jpg         # should_detect: true
│       ├── neg_glass_0001.jpg     # should_detect: false
│       └── neg_tv_screen_0001.jpg # should_detect: false
├── ground_truth.json
├── LABELING_GUIDE.md
└── README.md
```

**Naming convention**: `neg_<descriptor>_<number>.jpg` for hard negatives

---

## Validation Metrics

### Per-Class Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Pool recall** | detected pool (should_detect=true) / total pool (should_detect=true) | ≥85% |
| **Ocean recall** | detected ocean (should_detect=true) / total ocean (should_detect=true) | ≥85% |
| **False trigger rate** | detected (should_detect=false) / total (should_detect=false) | ≤10% (v1), ≤5% (production) |
| **Pool avg coverage** | mean(coverage) for detected pools | Report only |
| **Ocean avg coverage** | mean(coverage) for detected oceans | Report only |
| **Pool median coverage** | median(coverage) for detected pools | Drift detection |
| **Ocean median coverage** | median(coverage) for detected oceans | Drift detection |
| **Pool avg edge alignment** | mean(edge_score) for pools | ≥0.6 |
| **Ocean avg edge alignment** | mean(edge_score) for oceans | ≥0.6 |
| **Pool avg stability** | mean(stability) for pools | ≥0.8 |
| **Ocean avg stability** | mean(stability) for oceans | ≥0.8 |

**Definitions**:
- **True Positive (TP)**: `should_detect=true` AND `detector.present=true`
- **False Negative (FN)**: `should_detect=true` AND `detector.present=false`
- **False Trigger (FT)**: `should_detect=false` AND `detector.present=true` — detector fired when it shouldn't
- **True Negative (TN)**: `should_detect=false` AND `detector.present=false`

**Naming semantics**:
- "False trigger" (not "false positive") — honest naming for should_detect=false firing
- `is_false_positive` kept as legacy alias, computed from `should_detect=false`, not from non_water label

---

## Updated Report Schema (Backward-Compatible)

### Per-Image Result

```json
{
  "image_path": "pool/pool_0001.jpg",
  "scene_type": "pool",
  "should_detect": true,
  "difficulty": "easy",
  "tags": [],
  
  "coverage": 0.85,
  "coverage_px": 55296,
  "confidence": 0.72,
  "source": "heuristic",
  "implementation": "baseline_blue_threshold_v0",
  
  "edge_alignment_score": 0.68,
  "boundary_px": 1024,
  "stability_score": 0.82,
  
  "is_false_positive": false,
  "is_false_trigger": false,
  "processing_time_ms": 28.5
}
```

**New fields** (backward-compatible):
- `should_detect`: boolean (from ground truth)
- `difficulty`: string (from ground truth)
- `tags`: list (from ground truth)
- `implementation`: string (detector version for apples-to-apples comparison)
- `is_false_trigger`: boolean (primary metric for should_detect=false cases)

**Legacy fields** (kept for compatibility):
- `scene_type`: Alias for `label` (kept for existing code)
- `is_false_positive`: Computed as `should_detect=false AND detector.present=true` (same as is_false_trigger)
- No references to "non_water" label (removed)

### Summary Statistics

```json
{
  "summary": {
    "dataset_version": "v0",
    "total_images": 44,
    "pool_images": 22,
    "ocean_images": 22,
    "should_detect_true": 40,
    "should_detect_false": 4,
    
    "pool_recall": 0.90,
    "pool_avg_coverage": 0.78,
    "pool_median_coverage": 0.75,
    "pool_avg_confidence": 0.68,
    "pool_avg_edge_alignment": 0.62,
    "pool_avg_stability": 0.80,
    
    "ocean_recall": 0.85,
    "ocean_avg_coverage": 0.65,
    "ocean_median_coverage": 0.62,
    "ocean_avg_confidence": 0.58,
    "ocean_avg_edge_alignment": 0.55,
    "ocean_avg_stability": 0.75,
    
    "false_trigger_count": 1,
    "false_trigger_rate": 0.25,
    
    "overall_avg_processing_time_ms": 32.1,
    
    "false_positive_count": 0,
    "false_positive_rate": 0.0
  },
  "results": [...]
}
```

**New summary fields**:
- `should_detect_true` / `should_detect_false`: Counts
- `pool_median_coverage` / `ocean_median_coverage`: Drift detection
- `false_trigger_count` / `false_trigger_rate`: Hard negative performance

**Kept for compatibility**:
- `false_positive_count` / `false_positive_rate`: Legacy (always 0 for v0)

---

## CI Regression Gates (Warning Mode + Coverage Drift)

**Baseline**: `data/water_v0/baseline_v0.json` (store initial run)

**Regression rules** (warn, don't fail):

**Delta semantics** (absolute, not relative):
- Recall drops: **Absolute** >10% (e.g., 0.90 → 0.79)
- Edge alignment drops: **Absolute** >0.1 (e.g., 0.65 → 0.54)
- False trigger rate increases: **Absolute** >15% (e.g., 0.05 → 0.20)
- Coverage drift: **Ratio** >2x or <0.5x (with epsilon guard)

```python
# scripts/check_regression.py

import json
import sys

baseline = json.load(open("data/water_v0/baseline_v0.json"))
current = json.load(open("water_validation_ci.json"))

warnings = []

# 1. Recall drops >10%
pool_recall_drop = baseline["summary"]["pool_recall"] - current["summary"]["pool_recall"]
if pool_recall_drop > 0.10:
    warnings.append(f"Pool recall dropped {pool_recall_drop:.1%}")

ocean_recall_drop = baseline["summary"]["ocean_recall"] - current["summary"]["ocean_recall"]
if ocean_recall_drop > 0.10:
    warnings.append(f"Ocean recall dropped {ocean_recall_drop:.1%}")

# 2. Edge alignment drops >0.1
pool_edge_drop = baseline["summary"]["pool_avg_edge_alignment"] - current["summary"]["pool_avg_edge_alignment"]
if pool_edge_drop > 0.1:
    warnings.append(f"Pool edge alignment dropped {pool_edge_drop:.2f}")

ocean_edge_drop = baseline["summary"]["ocean_avg_edge_alignment"] - current["summary"]["ocean_avg_edge_alignment"]
if ocean_edge_drop > 0.1:
    warnings.append(f"Ocean edge alignment dropped {ocean_edge_drop:.2f}")

# 3. Coverage drift (median changed by >2x or <0.5x, with epsilon guard)
EPSILON = 1e-6
ABSOLUTE_DRIFT_THRESHOLD = 0.05

pool_baseline_median = baseline["summary"]["pool_median_coverage"]
pool_current_median = current["summary"]["pool_median_coverage"]

if pool_baseline_median < EPSILON:
    # Baseline near zero: use absolute drift check
    if pool_current_median > ABSOLUTE_DRIFT_THRESHOLD:
        warnings.append(f"Pool median coverage jumped from ~0 to {pool_current_median:.2%}")
else:
    # Normal case: ratio test
    pool_cov_ratio = pool_current_median / pool_baseline_median
    if pool_cov_ratio > 2.0 or pool_cov_ratio < 0.5:
        warnings.append(f"Pool median coverage drifted {pool_cov_ratio:.2f}x")

ocean_baseline_median = baseline["summary"]["ocean_median_coverage"]
ocean_current_median = current["summary"]["ocean_median_coverage"]

if ocean_baseline_median < EPSILON:
    if ocean_current_median > ABSOLUTE_DRIFT_THRESHOLD:
        warnings.append(f"Ocean median coverage jumped from ~0 to {ocean_current_median:.2%}")
else:
    ocean_cov_ratio = ocean_current_median / ocean_baseline_median
    if ocean_cov_ratio > 2.0 or ocean_cov_ratio < 0.5:
        warnings.append(f"Ocean median coverage drifted {ocean_cov_ratio:.2f}x")

# 4. False trigger rate increased (absolute delta, not relative)
ft_increase = current["summary"]["false_trigger_rate"] - baseline["summary"]["false_trigger_rate"]
if ft_increase > 0.15:  # Absolute: +0.15 (e.g., 0.05 → 0.20)
    warnings.append(f"False trigger rate increased by {ft_increase:.1%}")

if warnings:
    print("⚠️ WARNING: Quality regression detected")
    for w in warnings:
        print(f"  - {w}")
    sys.exit(0)  # Warning mode: don't fail build
else:
    print("✅ No regression detected")
```

---

## Dataset v0 Acceptance Gate

Before calling v0 "ready," verify:

- [ ] **Each class has ≥5 tagged hard cases**
  - Pool: ≥5 with tags like `low-light`, `reflection`, `teal`, `tile`, `infinity-edge`
  - Ocean: ≥5 with tags like `waves`, `gray`, `distant`, `overcast`, `foam`

- [ ] **CI subset includes ≥2 hard cases per class**
  - Pool: 2+ with difficulty="hard"
  - Ocean: 2+ with difficulty="hard"

- [ ] **Harness produces stable JSON schema every run**
  - Run harness 3 times on same dataset → identical JSON structure
  - All summary fields present and correctly typed
  - No missing/null values in critical fields

- [ ] **Stability scoring is deterministic**
  - Harness accepts `--seed` parameter
  - Same seed + same dataset → identical stability scores
  - Per-image RNG derived from `seed + hash(image_relpath)`

- [ ] **Hard negatives included**
  - ≥2 hard negatives per class (should_detect=false)
  - Covers common false trigger modes (sky, paint, reflections)

---

## Example Dataset (44 images: 20 pool + 20 ocean + 2 pool negatives + 2 ocean negatives)

```json
{
  "version": "v0",
  "created": "2025-12-14",
  "root": "data/water_v0/images",
  "labels": ["pool", "ocean"],
  "images": {
    "pool/pool_0001.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "notes": "Clear blue, sun", "tags": []},
    "pool/pool_0002.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "notes": "Resort pool", "tags": []},
    "pool/pool_0003.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "notes": "Dark evening", "tags": ["low-light"]},
    "pool/pool_0004.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "notes": "Reflections", "tags": ["reflection"]},
    "pool/pool_0005.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "notes": "Teal water", "tags": ["teal"]},
    "pool/pool_0006.jpg": {"label": "pool", "should_detect": true, "difficulty": "medium", "notes": "Tile visible", "tags": ["tile"]},
    "pool/pool_0007.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "notes": "Infinity edge", "tags": ["infinity-edge"]},
    "pool/pool_0008.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "notes": "Lap pool", "tags": []},
    "pool/pool_0009.jpg": {"label": "pool", "should_detect": true, "difficulty": "medium", "notes": "Indoor pool", "tags": ["indoor"]},
    "pool/pool_0010.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "notes": "Overhead", "tags": []},
    "pool/pool_0011.jpg": {"label": "pool", "should_detect": true, "difficulty": "medium", "notes": "Overcast", "tags": []},
    "pool/pool_0012.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "notes": "Twilight", "tags": ["low-light"]},
    "pool/pool_0013.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "notes": "Clear sun", "tags": []},
    "pool/pool_0014.jpg": {"label": "pool", "should_detect": true, "difficulty": "medium", "notes": "Partial shade", "tags": []},
    "pool/pool_0015.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "notes": "Green tint", "tags": ["teal"]},
    "pool/pool_0016.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "notes": "Bright blue", "tags": []},
    "pool/pool_0017.jpg": {"label": "pool", "should_detect": true, "difficulty": "medium", "notes": "Stone edge", "tags": []},
    "pool/pool_0018.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "notes": "Mosaic", "tags": ["tile"]},
    "pool/pool_0019.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "notes": "Resort clear", "tags": []},
    "pool/pool_0020.jpg": {"label": "pool", "should_detect": true, "difficulty": "medium", "notes": "Curved edge", "tags": []},
    
    "pool/neg_blue_wall_0001.jpg": {"label": "pool", "should_detect": false, "difficulty": "hard", "notes": "HARD NEGATIVE: blue painted wall", "tags": ["hard_negative", "blue_paint"]},
    "pool/neg_blue_sky_0001.jpg": {"label": "pool", "should_detect": false, "difficulty": "hard", "notes": "HARD NEGATIVE: blue sky through window", "tags": ["hard_negative", "sky"]},
    
    "ocean/ocean_0001.jpg": {"label": "ocean", "should_detect": true, "difficulty": "easy", "notes": "Calm, sunny", "tags": []},
    "ocean/ocean_0002.jpg": {"label": "ocean", "should_detect": true, "difficulty": "medium", "notes": "Overcast", "tags": ["overcast"]},
    "ocean/ocean_0003.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "notes": "Gray water", "tags": ["gray"]},
    "ocean/ocean_0004.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "notes": "Choppy", "tags": ["waves"]},
    "ocean/ocean_0005.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "notes": "Distant", "tags": ["distant"]},
    "ocean/ocean_0006.jpg": {"label": "ocean", "should_detect": true, "difficulty": "medium", "notes": "Foam", "tags": ["foam"]},
    "ocean/ocean_0007.jpg": {"label": "ocean", "should_detect": true, "difficulty": "easy", "notes": "Clear blue", "tags": []},
    "ocean/ocean_0008.jpg": {"label": "ocean", "should_detect": true, "difficulty": "medium", "notes": "Sunset", "tags": []},
    "ocean/ocean_0009.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "notes": "Dusk", "tags": ["low-light"]},
    "ocean/ocean_0010.jpg": {"label": "ocean", "should_detect": true, "difficulty": "easy", "notes": "Beach", "tags": []},
    "ocean/ocean_0011.jpg": {"label": "ocean", "should_detect": true, "difficulty": "medium", "notes": "Horizon", "tags": []},
    "ocean/ocean_0012.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "notes": "Storm", "tags": ["gray", "overcast"]},
    "ocean/ocean_0013.jpg": {"label": "ocean", "should_detect": true, "difficulty": "easy", "notes": "Calm clear", "tags": []},
    "ocean/ocean_0014.jpg": {"label": "ocean", "should_detect": true, "difficulty": "medium", "notes": "Small waves", "tags": []},
    "ocean/ocean_0015.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "notes": "Rough sea", "tags": ["waves"]},
    "ocean/ocean_0016.jpg": {"label": "ocean", "should_detect": true, "difficulty": "easy", "notes": "Tropical", "tags": []},
    "ocean/ocean_0017.jpg": {"label": "ocean", "should_detect": true, "difficulty": "medium", "notes": "Rocky shore", "tags": []},
    "ocean/ocean_0018.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "notes": "Far horizon", "tags": ["distant"]},
    "ocean/ocean_0019.jpg": {"label": "ocean", "should_detect": true, "difficulty": "easy", "notes": "Clear water", "tags": []},
    "ocean/ocean_0020.jpg": {"label": "ocean", "should_detect": true, "difficulty": "medium", "notes": "Mid-distance", "tags": []},
    
    "ocean/neg_glass_building_0001.jpg": {"label": "ocean", "should_detect": false, "difficulty": "hard", "notes": "HARD NEGATIVE: reflective glass", "tags": ["hard_negative", "glass"]},
    "ocean/neg_blue_umbrella_0001.jpg": {"label": "ocean", "should_detect": false, "difficulty": "hard", "notes": "HARD NEGATIVE: blue fabric", "tags": ["hard_negative", "fabric"]}
  }
}
```

**Distribution**:
- Water: 40 images (20 pool + 20 ocean, should_detect=true)
- Hard negatives: 4 images (2 pool + 2 ocean, should_detect=false)
- Total: 44 images

---

## CI Subset (14 images including hard negatives)

```json
{
  "version": "ci_subset",
  "root": "data/water_v0/images",
  "labels": ["pool", "ocean"],
  "images": {
    "pool/pool_0001.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "tags": []},
    "pool/pool_0003.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "tags": ["low-light"]},
    "pool/pool_0005.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "tags": ["teal"]},
    "pool/pool_0007.jpg": {"label": "pool", "should_detect": true, "difficulty": "hard", "tags": ["infinity-edge"]},
    "pool/pool_0008.jpg": {"label": "pool", "should_detect": true, "difficulty": "easy", "tags": []},
    "pool/pool_0009.jpg": {"label": "pool", "should_detect": true, "difficulty": "medium", "tags": ["indoor"]},
    "pool/neg_blue_wall_0001.jpg": {"label": "pool", "should_detect": false, "difficulty": "hard", "tags": ["hard_negative", "blue_paint"]},
    
    "ocean/ocean_0001.jpg": {"label": "ocean", "should_detect": true, "difficulty": "easy", "tags": []},
    "ocean/ocean_0003.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "tags": ["gray"]},
    "ocean/ocean_0004.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "tags": ["waves"]},
    "ocean/ocean_0005.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "tags": ["distant"]},
    "ocean/ocean_0007.jpg": {"label": "ocean", "should_detect": true, "difficulty": "easy", "tags": []},
    "ocean/ocean_0009.jpg": {"label": "ocean", "should_detect": true, "difficulty": "hard", "tags": ["low-light"]},
    "ocean/neg_glass_building_0001.jpg": {"label": "ocean", "should_detect": false, "difficulty": "hard", "tags": ["hard_negative", "glass"]}
  }
}
```

**Balanced**: 6 pool + 6 ocean water (should_detect=true), 2 hard negatives (should_detect=false)

---

## Next Steps (Implementation Checklist)

### 1. Update `scripts/prw_water_validation.py`

- [ ] Load `should_detect`, `difficulty`, `tags` from ground truth
- [ ] Compute `false_trigger_rate` and `false_trigger_count`
- [ ] Compute `pool_median_coverage`, `ocean_median_coverage`
- [ ] Implement drift check with epsilon handling (EPSILON=1e-6, ABSOLUTE_DRIFT_THRESHOLD=0.05)
- [ ] Make stability deterministic via `--seed` parameter
- [ ] Add `implementation` field to per-image results (e.g., "baseline_blue_threshold_v0")
- [ ] Ensure `is_false_positive` = `is_false_trigger` for backward compatibility

### 2. Create CI Subset Manifest

**File**: `data/water_v0/ci_subset.txt` (simple text list of relative paths)

```
pool/pool_0001.jpg
pool/pool_0003.jpg
pool/pool_0005.jpg
pool/pool_0007.jpg
pool/pool_0008.jpg
pool/pool_0009.jpg
pool/neg_blue_wall_0001.jpg
ocean/ocean_0001.jpg
ocean/ocean_0003.jpg
ocean/ocean_0004.jpg
ocean/ocean_0005.jpg
ocean/ocean_0007.jpg
ocean/ocean_0009.jpg
ocean/neg_glass_building_0001.jpg
```

**Why**: Easy to change subset without rewriting JSON

### 3. Add `.gitignore` Entries

```gitignore
# Ignore full-res images (unless committing thumbnails deliberately)
data/water_*/images/*.jpg
data/water_*/images/*.png

# Keep ground truth and metadata
!data/water_*/ground_truth.json
!data/water_*/LABELING_GUIDE.md
!data/water_*/README.md
!data/water_*/ci_subset.txt
```

### 4. Create `scripts/check_regression.py`

Complete implementation with epsilon guard, absolute deltas, and clear warnings.

---

## Summary

**Two labels preserved**: Pool and ocean folders, no third label

**Negative controls**: `should_detect: false` enables false trigger rate without label explosion

**Mask storage**: Derived stats only (coverage_px, boundary_px, edge_score), no full arrays in JSON

**Coverage drift detection**: Median coverage tracks silent quality decay (with epsilon guard)

**Deterministic stability**: Seed-based RNG ensures reproducible results

**False trigger semantics**: Honest naming, not "false positive" for should_detect=false

**Backward compatible**: Existing PR-W4 fields preserved, new fields added

**Implementation field**: Track detector version for apples-to-apples comparison

**Absolute deltas**: CI gates use absolute thresholds (not relative %), less noisy for v0/v1

**Status**: Validation harness complete; integration-ready; detector remains a stub pending PR-W1

File: `docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md`
