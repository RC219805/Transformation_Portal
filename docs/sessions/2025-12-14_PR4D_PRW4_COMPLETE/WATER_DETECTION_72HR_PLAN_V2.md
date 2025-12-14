# Water Detection: 72-Hour Execution Plan (Revised)

**Goal**: Data-driven detector with defensible quality gates, not invented numbers.

**Philosophy**: Measure → Fix biggest buckets → Measure again → Lock gates

---

## Critical Decisions Before Starting

### Decision 1: Dataset Storage Strategy

**Problem**: 60-200 images will bloat repo, slow CI, create licensing issues.

**Chosen Strategy** (select one before Hour 0):

**Option A (RECOMMENDED): Git-based, Downscaled**
```bash
data/water_v0/
├── ground_truth.json          # Commit this (paths → labels)
├── LABELING_GUIDE.md           # Commit this
├── thumbnails/                 # Commit 256px downscaled versions (for PR review)
│   ├── pool/
│   ├── ocean/
│   └── non_water/
└── IMAGES_README.md            # Instructions to download full-res from storage
```

Full-resolution images live in: `s3://bucket/water_v0/` or local private storage.

CI runs on thumbnails only (fast, no license issues).

**Option B: Git LFS**
```bash
# Install Git LFS
git lfs install
git lfs track "data/water_v0/images/**/*.jpg"

# CI uses small subset
CI_SUBSET="data/water_v0/ci_subset.json"  # 12-20 images for fast validation
```

**Decision Required**: Pick A or B before creating dataset.

### Decision 2: Ground Truth Schema

**Proposed Schema** (cross-platform safe):
```json
{
  "version": "v0",
  "created": "2025-12-14",
  "total_images": 60,
  "images": {
    "pool/resort_pool_001.jpg": {
      "label": "pool",
      "notes": "Direct sun, clear blue, sharp edges"
    },
    "ocean/calm_ocean_overcast_002.jpg": {
      "label": "ocean",
      "notes": "Desaturated, gray tones, low contrast"
    },
    "non_water/blue_sky_through_window_003.jpg": {
      "label": "non_water",
      "notes": "HARD NEGATIVE: large blue region, high saturation"
    }
  }
}
```

**Path format**: Relative to `data/water_v0/images/`

**Validation script** (`scripts/validate_ground_truth.py`):
```python
#!/usr/bin/env python3
import json
from pathlib import Path

def validate_ground_truth(gt_path, images_dir):
    with open(gt_path) as f:
        gt = json.load(f)
    
    errors = []
    for rel_path, metadata in gt["images"].items():
        full_path = images_dir / rel_path
        if not full_path.exists():
            errors.append(f"Missing: {rel_path}")
        if metadata["label"] not in ["pool", "ocean", "non_water"]:
            errors.append(f"Invalid label: {rel_path} → {metadata['label']}")
    
    if errors:
        print("❌ Ground truth validation failed:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print(f"✅ Ground truth valid ({len(gt['images'])} images)")
    return True
```

---

## Hour 0-6: Dataset v0 + Baseline Measurement

### Deliverable 1: Dataset Structure (Hour 0-3)

**Target**: 60 images minimum (20/20/20 class balance)

**Hard Negatives Checklist** (non_water):
- [ ] Blue sky through windows (5+ examples)
- [ ] Blue painted walls (interior/exterior)
- [ ] Glass buildings with blue reflections
- [ ] Blue fabric/umbrellas/furniture
- [ ] TV/monitor screens showing blue content
- [ ] Glossy marble/stone with reflections
- [ ] Pool covers (not water surface)
- [ ] Shadows on wet surfaces (ambiguous)
- [ ] Blue cars/boats (not in water)
- [ ] Tiled pools with strong texture patterns

**Lighting/Angle Diversity**:
- Direct sun, overcast, indoor, dusk
- Overhead, horizon, oblique angles
- Close-up, mid-range, distant

**Dataset Layout**:
```bash
data/water_v0/
├── ground_truth.json           # Metadata + labels
├── LABELING_GUIDE.md           # Labeling rules + edge cases
├── images/                     # OR thumbnails/ (see Decision 1)
│   ├── pool/                   # 20+ examples
│   ├── ocean/                  # 20+ examples
│   └── non_water/              # 20+ examples (HARD NEGATIVES)
└── README.md                   # Dataset provenance, licensing, download instructions
```

**Time Estimate**: 
- Image collection/curation: 2 hours
- Labeling + ground_truth.json: 1 hour

### Deliverable 2: Baseline Measurement (Hour 3-6)

**Run validation harness with stub detector**:
```bash
cd /Users/rc/Transformation_Portal

# Validate ground truth schema first
python scripts/validate_ground_truth.py \
    data/water_v0/ground_truth.json \
    data/water_v0/images

# Run validation harness (stub detector)
python scripts/prw_water_validation.py \
    --input-dir data/water_v0/images \
    --ground-truth data/water_v0/ground_truth.json \
    --output data/water_v0/baseline_stub.json \
    --config water_detection_enabled=true
```

**Generate failure analysis** (auto-generated):
```bash
# Create failure analysis script
python scripts/analyze_failures.py \
    data/water_v0/baseline_stub.json \
    --output docs/WATER_V0_BASELINE.md
```

**docs/WATER_V0_BASELINE.md** (template):
```markdown
# Water v0 Baseline (Stub Detector)

**Date**: [AUTO]
**Dataset**: data/water_v0 (X pool, Y ocean, Z non-water)
**Detector**: Stub (simple blue threshold)

## Summary Statistics

| Metric | Stub Performance | Notes |
|--------|------------------|-------|
| Pool recall | [MEASURED]/20 ([%]) | Target: ≥85% (17/20) |
| Ocean recall | [MEASURED]/20 ([%]) | Target: ≥85% (17/20) |
| False positive count | [MEASURED]/20 | Target: ≤1/20 (5%) |
| False positive rate | [%] | Target: ≤5% |
| Avg edge alignment | [MEASURED] | Target: ≥0.6 |
| Avg stability | [MEASURED] | Target: ≥0.8 |

## Top Offenders by Category

### False Positives (by reason bucket)

**Sky (X failures)**
- `non_water/blue_sky_001.jpg` - Confidence: [X]
- `non_water/window_reflection_003.jpg` - Confidence: [X]
[... list all ...]

**Pattern**: Large blue regions, high saturation, low texture
**Root cause**: No chroma gating, no texture check
**Fix hypothesis**: HSV constraints + entropy check

**Paint/Surfaces (X failures)**
- `non_water/blue_wall_002.jpg` - Confidence: [X]
[... list all ...]

**Pattern**: Solid color, uniform
**Root cause**: Blue channel threshold too permissive
**Fix hypothesis**: Saturation range, component shape filtering

**Reflections (X failures)**
- `non_water/marble_reflection_005.jpg` - Confidence: [X]
[... list all ...]

**Pattern**: Irregular boundaries, high specular
**Root cause**: No edge validation
**Fix hypothesis**: Boundary-gradient alignment check

[... continue for each bucket ...]

### Missed Detections (by reason bucket)

**Low-light (X failures)**
- `pool/dark_pool_evening_004.jpg` - Confidence: [X]
[... list all ...]

**Pattern**: Value < 0.3, desaturated
**Root cause**: Value threshold too restrictive
**Fix hypothesis**: Scene-aware value ranges (pool vs ocean)

**Desaturated water (X failures)**
- `ocean/gray_ocean_overcast_007.jpg` - Confidence: [X]
[... list all ...]

**Pattern**: Saturation < 0.15, blue-gray
**Root cause**: Saturation threshold too high
**Fix hypothesis**: Lower saturation for ocean context

[... continue ...]

## Prioritized Fix List

Based on failure frequency and estimated ROI:

**P0: Chroma gating (HSV constraints)**
- **Impact**: Expected to reduce [X/Y] sky FPs, recover [A/B] desaturated misses
- **Complexity**: Low (30 mins)
- **Dependencies**: None

**P1: Component filtering (geometric constraints)**
- **Impact**: Expected to reduce [X/Y] solid-object FPs
- **Complexity**: Low (30 mins)
- **Dependencies**: None

**P2: Texture sanity check (entropy/Laplacian)**
- **Impact**: Expected to reduce [X/Y] texture-based FPs
- **Complexity**: Medium (45 mins)
- **Dependencies**: None

**P3: Scene-aware thresholds**
- **Impact**: Expected to recover [X/Y] low-light/desaturated misses
- **Complexity**: Low (20 mins)
- **Dependencies**: P0 (uses same HSV pipeline)

## Acceptance Gate for v1

Detector v1 must meet **ALL** of:
- [ ] Precision improves (FP count decreases OR FP rate decreases by ≥20%)
- [ ] Recall does not collapse (pool+ocean recall ≥ stub baseline - 10%)
- [ ] At least 2 of {edge alignment, stability, processing time} improve or hold steady

If gate fails, analyze v1 failures and iterate.
```

**Time Estimate**: 
- Harness run: 30 mins (for 60 images)
- Failure analysis generation: 1 hour
- Manual triage/categorization: 1.5 hours

---

## Hour 6-12: Detector v1 (Fix Top Buckets)

### Implementation Strategy

**DO NOT implement full PR-W1 spec.**

Implement **ONLY** the P0-P3 fixes identified in failure analysis.

### Deliverable 3: Detector v1 Implementation

**Time estimate**: 2-3 hours total (based on P0-P3 complexity)

**Example Implementation** (adapt based on actual failures):
```python
# lux_depth_v2/water_candidate.py

class WaterCandidateDetector:
    """
    Water Candidate Detector v1 - Failure-Driven
    
    Based on data/water_v0 baseline failure analysis.
    See: docs/WATER_V0_BASELINE.md
    
    Implements only P0-P3 fixes, not full PR-W1 spec.
    """
    
    def detect(self, rgb01, depth01=None, scene_context=SceneContext.UNKNOWN):
        h, w = rgb01.shape[:2]
        
        # P0: HSV chroma gating (tuned to failure buckets)
        hsv = rgb2hsv(rgb01)
        
        # Scene-aware ranges (derived from failure analysis, not guessed)
        if scene_context == SceneContext.POOL:
            hue_range = (170, 210)      # Derived from pool samples
            sat_range = (0.15, 0.8)     # Reject low-sat blues
            val_range = (0.2, 1.0)      # Allow darker pools
        elif scene_context == SceneContext.OCEAN:
            hue_range = (160, 220)      # Broader for ocean
            sat_range = (0.1, 0.7)      # Lower for desaturated ocean
            val_range = (0.15, 0.9)     # Lower for overcast
        else:
            hue_range = (165, 215)      # Conservative middle
            sat_range = (0.12, 0.75)
            val_range = (0.18, 0.95)
        
        chroma_mask = (
            (hsv[:,:,0]*360 >= hue_range[0]) & (hsv[:,:,0]*360 <= hue_range[1]) &
            (hsv[:,:,1] >= sat_range[0]) & (hsv[:,:,1] <= sat_range[1]) &
            (hsv[:,:,2] >= val_range[0]) & (hsv[:,:,2] <= val_range[1])
        )
        
        # P1: Component filtering (reject extreme shapes)
        filtered_mask = self._filter_components(
            chroma_mask,
            min_area=1000,           # Tuned to dataset
            max_aspect_ratio=5.0,    # Reject long thin regions
            min_fill_ratio=0.6       # Reject scattered regions
        )
        
        # P2: Texture sanity check (reject high-texture regions)
        texture_ok = self._texture_check(rgb01, filtered_mask)
        
        final_mask = filtered_mask & texture_ok
        
        # Compute metrics
        coverage_px = int(np.sum(final_mask))
        coverage = coverage_px / (h * w)
        confidence = self._compute_confidence(final_mask, chroma_mask, texture_ok)
        
        return {
            "present": coverage >= 0.04 and confidence >= 0.35,
            "coverage": coverage,
            "coverage_px": coverage_px,
            "confidence": confidence,
            "mask": final_mask.astype(np.float32)
        }
    
    def _filter_components(self, mask, min_area, max_aspect_ratio, min_fill_ratio):
        """P1: Geometric sanity checks."""
        from scipy.ndimage import label
        from skimage.measure import regionprops
        
        labeled, num = label(mask)
        filtered = np.zeros_like(mask)
        
        for region in regionprops(labeled):
            if region.area < min_area:
                continue
            
            bbox = region.bbox
            h = bbox[2] - bbox[0]
            w = bbox[3] - bbox[1]
            aspect = max(h, w) / max(min(h, w), 1)
            
            if aspect > max_aspect_ratio:
                continue
            
            fill = region.area / (h * w)
            if fill < min_fill_ratio:
                continue
            
            filtered[labeled == region.label] = 1
        
        return filtered.astype(bool)
    
    def _texture_check(self, rgb01, mask):
        """P2: Reject high-texture regions (simple Laplacian variance)."""
        from scipy.ndimage import laplace
        
        gray = rgb01.mean(axis=2)
        laplacian = laplace(gray)
        
        # Low Laplacian variance = smooth (like water)
        # Threshold derived from dataset analysis
        variance_threshold = 0.02  # Tuned to dataset
        
        smooth = np.abs(laplacian) < variance_threshold
        return smooth
```

### Deliverable 4: v1 Validation Report

**Re-run harness with detector v1**:
```bash
python scripts/prw_water_validation.py \
    --input-dir data/water_v0/images \
    --ground-truth data/water_v0/ground_truth.json \
    --output data/water_v0/detector_v1.json \
    --config water_detection_enabled=true
```

**Generate comparison report**:
```bash
python scripts/compare_detectors.py \
    --baseline data/water_v0/baseline_stub.json \
    --improved data/water_v0/detector_v1.json \
    --output docs/WATER_V1_RESULTS.md
```

**docs/WATER_V1_RESULTS.md** (template):
```markdown
# Detector v1 Results vs Baseline

## Summary

| Metric | Stub | v1 | Delta | Gate |
|--------|------|-----|-------|------|
| Pool recall | [X]/20 ([%]) | [Y]/20 ([%]) | [DELTA] | [PASS/FAIL] |
| Ocean recall | [X]/20 ([%]) | [Y]/20 ([%]) | [DELTA] | [PASS/FAIL] |
| FP count | [X]/20 | [Y]/20 | [DELTA] | [PASS/FAIL] |
| FP rate | [%] | [%] | [DELTA] | [PASS/FAIL] |
| Edge alignment | [X] | [Y] | [DELTA] | [PASS/FAIL] |
| Stability | [X] | [Y] | [DELTA] | [PASS/FAIL] |

**Acceptance Gate**: v1 must meet ALL of:
- [x/✓] Precision improves (FP ↓ by ≥20%)
- [x/✓] Recall holds (pool+ocean ≥ baseline - 10%)
- [x/✓] ≥2 other metrics improve/hold

**Overall**: [PASS/FAIL]

## Remaining Failures (v1)

[List failures that v1 still misses, categorized by bucket]

## Next Iteration Priorities

[Based on v1 failures, what are P4-P6 fixes?]
```

**Time estimate**: 
- v1 implementation: 2-3 hours
- Re-run harness: 30 mins
- Comparison analysis: 30 mins

---

## Hour 12-18: Calibration + CI Guardrail

### Deliverable 5: Threshold Calibration

**Derive thresholds from dataset statistics** (not aspirational):

```python
# scripts/calibrate_thresholds.py

import json
import numpy as np

def calibrate_thresholds(validation_report):
    """Derive thresholds from dataset v0 performance."""
    with open(validation_report) as f:
        report = json.load(f)
    
    results = report["results"]
    
    # Analyze confidence distribution for true positives
    tp_confidences = [
        r["confidence"] for r in results
        if r["scene_type"] in ["pool", "ocean"] and 
        r["source"] != "none"
    ]
    
    # Use 10th percentile as min confidence (allow 90% of TPs)
    confidence_threshold = np.percentile(tp_confidences, 10) if tp_confidences else 0.35
    
    # Analyze coverage distribution
    tp_coverages = [
        r["coverage"] for r in results
        if r["scene_type"] in ["pool", "ocean"] and
        r["coverage"] > 0
    ]
    
    # Use 5th percentile as min coverage
    coverage_threshold = np.percentile(tp_coverages, 5) if tp_coverages else 0.04
    
    return {
        "water_candidate_confidence_threshold": round(confidence_threshold, 2),
        "water_min_coverage": round(coverage_threshold, 3),
        "calibrated_on": "data/water_v0",
        "dataset_size": len(results),
        "notes": "Derived from 10th/5th percentiles to allow 90%/95% of true positives"
    }

# Run calibration
thresholds = calibrate_thresholds("data/water_v0/detector_v1.json")
print(json.dumps(thresholds, indent=2))
```

**Update config with calibrated values**:
```python
# lux_depth_v2/materials_v3.py

@dataclass
class MaterialsV3Config:
    # Water detection (calibrated on data/water_v0, detector v1)
    water_detection_enabled: bool = False
    water_candidate_confidence_threshold: float = 0.35  # Calibrated, not guessed
    water_min_coverage: float = 0.04  # Calibrated from dataset
```

### Deliverable 6: CI Regression Check (Warning Mode)

**Create CI workflow**:
```yaml
# .github/workflows/water_quality_gate.yml

name: Water Detection Quality Gate

on:
  pull_request:
    paths:
      - 'lux_depth_v2/water_candidate.py'
      - 'lux_depth_v2/materials_v3.py'
      - 'data/water_v0/**'
  workflow_dispatch:

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Validate ground truth schema
        run: |
          python scripts/validate_ground_truth.py \
            data/water_v0/ground_truth.json \
            data/water_v0/images
      
      - name: Run validation harness (CI subset)
        run: |
          # CI runs on subset for speed (12-20 images)
          python scripts/prw_water_validation.py \
            --input-dir data/water_v0/images \
            --ground-truth data/water_v0/ci_subset.json \
            --output water_validation_ci.json \
            --config water_detection_enabled=true
      
      - name: Check quality gates (WARNING MODE)
        run: |
          python scripts/check_quality_gates.py \
            water_validation_ci.json \
            --pool-recall-min 0.75 \
            --ocean-recall-min 0.70 \
            --fp-rate-max 0.20 \
            --mode warning
        # Warning mode: prints issues but doesn't fail build
      
      - name: Upload validation report
        uses: actions/upload-artifact@v3
        with:
          name: water-validation-report
          path: water_validation_ci.json
```

**CI Subset** (12-20 images for fast validation):
```json
// data/water_v0/ci_subset.json
{
  "version": "ci_subset",
  "parent": "data/water_v0/ground_truth.json",
  "images": {
    "pool/pool_001.jpg": {"label": "pool"},
    "pool/pool_005.jpg": {"label": "pool"},
    "ocean/ocean_002.jpg": {"label": "ocean"},
    "ocean/ocean_007.jpg": {"label": "ocean"},
    "non_water/blue_sky_001.jpg": {"label": "non_water"},
    "non_water/blue_wall_002.jpg": {"label": "non_water"},
    ... (12-20 total, balanced across classes and difficulty)
  }
}
```

**Quality Gate Script**:
```python
#!/usr/bin/env python3
# scripts/check_quality_gates.py

import argparse
import json
import sys

def check_gates(report_path, pool_min, ocean_min, fp_max, mode):
    with open(report_path) as f:
        report = json.load(f)
    
    summary = report.get("summary", {})
    
    pool_recall = summary.get("pool_detection_rate", 0.0)
    ocean_recall = summary.get("ocean_detection_rate", 0.0)
    fp_rate = summary.get("false_positive_rate", 1.0)
    
    failures = []
    
    if pool_recall < pool_min:
        failures.append(f"Pool recall {pool_recall:.1%} < {pool_min:.1%}")
    
    if ocean_recall < ocean_min:
        failures.append(f"Ocean recall {ocean_recall:.1%} < {ocean_min:.1%}")
    
    if fp_rate > fp_max:
        failures.append(f"FP rate {fp_rate:.1%} > {fp_max:.1%}")
    
    if failures:
        prefix = "❌ FAILED" if mode == "error" else "⚠️ WARNING"
        print(f"\n{prefix}: Quality gates")
        for f in failures:
            print(f"  - {f}")
        print(f"\nMetrics: pool={pool_recall:.1%}, ocean={ocean_recall:.1%}, fp={fp_rate:.1%}")
        
        if mode == "error":
            sys.exit(1)
        else:
            print("\n⚠️ Warning mode: build continues despite gate failures")
    else:
        print(f"✅ PASSED: Quality gates")
        print(f"Metrics: pool={pool_recall:.1%}, ocean={ocean_recall:.1%}, fp={fp_rate:.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="Validation report JSON")
    parser.add_argument("--pool-recall-min", type=float, default=0.75)
    parser.add_argument("--ocean-recall-min", type=float, default=0.70)
    parser.add_argument("--fp-rate-max", type=float, default=0.20)
    parser.add_argument("--mode", choices=["warning", "error"], default="warning")
    
    args = parser.parse_args()
    check_gates(args.report, args.pool_recall_min, args.ocean_recall_min, 
                args.fp_rate_max, args.mode)
```

**What "Regression" Means**:
- Pool recall drops >10% vs baseline
- Ocean recall drops >10% vs baseline
- FP rate increases >50% vs baseline
- Edge alignment drops >0.1 vs baseline

**Time estimate**: 2-3 hours for calibration + CI setup

---

## Hour 18-36: Dataset Expansion + Hard Negatives

### Deliverable 7: Dataset v1 (100+ images)

**Expand based on v1 failure analysis**:
- Add 20+ pool (target failure buckets: dark, teal, infinity edge, etc.)
- Add 20+ ocean (target failure buckets: choppy, gray, distant, etc.)
- Add 20+ non-water hard negatives (target remaining FP buckets)

**Re-run validation**:
```bash
python scripts/prw_water_validation.py \
    --input-dir data/water_v1/images \
    --ground-truth data/water_v1/ground_truth.json \
    --output data/water_v1/detector_v1.json
```

**Time estimate**: 12-18 hours (image collection, labeling, validation)

---

## Hour 36-54: Detector v2 (Second Iteration)

### Deliverable 8: Detector v2

Based on v1 failures on expanded dataset:
- Implement P4-P6 fixes from failure analysis
- Re-validate on full dataset
- Aim for 3/5 or 4/5 targets met

**Time estimate**: 8-12 hours (implementation, testing, validation)

---

## Hour 54-72: CI Hardening + Documentation

### Deliverable 9: Tighten CI Gates

Move from warning → soft error mode:
```yaml
# Update CI workflow
--mode error  # Now fails build if gates not met
--pool-recall-min 0.80  # Tighter threshold
--fp-rate-max 0.15      # Tighter threshold
```

### Deliverable 10: Final Documentation

**Update all status docs**:
- PR-W1 status (v1 vs v2, calibrated thresholds)
- PR-W4 status (validation harness used in anger)
- Dataset provenance and expansion roadmap
- Production deployment checklist

**Time estimate**: 6-8 hours

---

## Acceptance Gates Summary

### Gate for v1 (Hour 12)
v1 must meet **ALL** of:
- [ ] FP count/rate decreases by ≥20% vs stub
- [ ] Pool+ocean recall ≥ stub baseline - 10%
- [ ] ≥2 of {edge, stability, time} improve or hold

### Gate for v2 (Hour 48)
v2 should meet **≥3 of 5**:
- [ ] Pool recall ≥80% (relaxed from 85% final target)
- [ ] Ocean recall ≥75% (relaxed from 85% final target)
- [ ] FP rate ≤15% (relaxed from 5% final target)
- [ ] Edge alignment ≥0.55 (relaxed from 0.6 final target)
- [ ] Stability ≥0.75 (relaxed from 0.8 final target)

### Gate for Production (Week 4+)
Must meet **≥4 of 5 final targets**:
- Pool recall ≥85%
- Ocean recall ≥85%
- FP rate ≤5%
- Edge alignment ≥0.6
- Stability ≥0.8

---

## What Makes This Executable

**No invented numbers**: All metrics measured, deltas computed, not guessed

**Time realistic**: Full 72 hours scheduled, not just 18

**Dataset handled**: Storage strategy decided upfront (Option A recommended)

**CI defined**: Specific thresholds, specific metrics, specific artifact storage

**Gates clear**: Binary pass/fail with explicit acceptance criteria

**Iterative**: v0 → v1 → v1.1 → v2 with measurements at each step

---

## Checklist Before Starting

- [ ] **Decision 1**: Dataset storage (Option A or B)?
- [ ] **Decision 2**: Ground truth schema validated
- [ ] **Scripts ready**: validate_ground_truth.py, analyze_failures.py, compare_detectors.py, calibrate_thresholds.py, check_quality_gates.py
- [ ] **Validation harness**: prw_water_validation.py working with current stub
- [ ] **Time allocated**: Full 72 hours (not 18)
- [ ] **Image sources**: Know where to get 60-100 pool/ocean/non-water images
- [ ] **Licensing clear**: Can commit images or have alternate storage

---

## Ground Truth Schema (Final)

**Proposed**:
```json
{
  "version": "v0",
  "created": "2025-12-14T12:00:00Z",
  "dataset_name": "water_v0",
  "total_images": 60,
  "class_distribution": {
    "pool": 20,
    "ocean": 20,
    "non_water": 20
  },
  "images": {
    "pool/resort_pool_001.jpg": {
      "label": "pool",
      "notes": "Direct sun, clear blue",
      "difficulty": "easy"
    },
    "ocean/calm_ocean_002.jpg": {
      "label": "ocean",
      "notes": "Overcast, desaturated",
      "difficulty": "medium"
    },
    "non_water/blue_sky_003.jpg": {
      "label": "non_water",
      "notes": "HARD NEGATIVE: large blue region",
      "difficulty": "hard"
    }
  }
}
```

**Path format**: Relative to `data/water_v0/images/` for cross-platform compatibility

**Validation**: Run `scripts/validate_ground_truth.py` before any harness execution
