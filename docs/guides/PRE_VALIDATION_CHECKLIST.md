# Pre-Validation Checklist
**Date**: 2025-12-19
**Next Run**: 50-image expanded validation

## ✅ Ready to Run

### Core Implementation
- [x] Multi-factor classifier (V2) implemented
- [x] HF energy metric implemented
- [x] Not-flat safeguard implemented
- [x] Balanced quality gates implemented
- [x] Fail-fast on missing metrics implemented
- [x] Full metadata logging implemented

### Infrastructure
- [x] Dataset expanded (50 images stratified)
- [x] Labels.csv created with ground truth
- [x] Validation runner script ready (`RUN_VALIDATION_HF_FIXED.sh`)
- [x] Analysis scripts ready (`analyze_validation_v2.py`, `evaluate_classifier_balanced.py`)

## 🔧 Pre-Flight Checks

### 1. Model Availability
```bash
# Verify DA V2 Large model is cached locally
python3 -c "from transformers import pipeline; p = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf'); print('✓ Model cached')"
```

**Expected**: Model loads without network access
**If fails**: Run `scripts/download_depth_models.py` first

### 2. Dataset Integrity
```bash
# Verify all 50 images exist and labels match
python3 << 'PYEOF'
import pandas as pd
from pathlib import Path

labels = pd.read_csv('data/validation_full/labels.csv')
image_dir = Path('data/validation_full')

print(f"Labels: {len(labels)} rows")
print(f"Images: {len(list(image_dir.glob('*.[jp][pn]g')))} files")

missing = []
for fname in labels['filename']:
    if not (image_dir / fname).exists():
        missing.append(fname)

if missing:
    print(f"❌ Missing {len(missing)} images: {missing}")
else:
    print("✓ All images exist")

# Check stratification
print("\nStratification:")
print(labels['scene_type'].value_counts())
PYEOF
```

**Expected**:
- 50 labels, 50 images
- ~25 texture, ~25 structure
- No missing images

### 3. Output Directory Clean
```bash
# Ensure fresh output directory (no partial runs)
ls -lh outputs/validation_full_* 2>/dev/null | tail -5
```

**Expected**: No recent partial runs (or explicitly archive them)

### 4. Smoke Test (2 images)
```bash
# Quick 2-image smoke test to verify pipeline works
python3 scripts/automation/production_depth_validation_fixed.py \
  --input-dir data/validation_smoke \
  --output-dir outputs/validation_smoke_$(date +%Y%m%d_%H%M%S) \
  --tile-size 1024 \
  --overlap 128 \
  2>&1 | tee smoke_test.log
```

**Expected**:
- 2/2 images succeed
- All metrics populated (no nulls)
- `scene_type`, `hf_energy`, `depth_range`, `lenient_pass`, `strict_pass` present

**If smoke fails**: Stop. Fix integration before burning 50-image compute.

### 5. Disk Space
```bash
# Verify adequate disk space (depth maps + metrics = ~500MB for 50 images)
df -h . | tail -1 | awk '{print "Available: " $4}'
```

**Expected**: >2GB available

## 🚀 Run Command (After Checks Pass)

```bash
# Full 50-image validation
./RUN_VALIDATION_HF_FIXED.sh

# Monitor progress in another terminal
tail -f validation_run.log
```

## 📊 Post-Run Analysis Sequence

### 1. Verify Completion
```bash
# Check all 50 metrics files exist
ls outputs/validation_full_*/metrics/*_metrics.json | wc -l
# Expected: 50
```

### 2. Classifier Evaluation
```bash
python3 scripts/evaluate_classifier_balanced.py \
  --metrics-dir outputs/validation_full_* \
  --labels data/validation_full/labels.csv
```

**Decision Gate**:
- Balanced accuracy ≥ 75%: Proceed to threshold calibration
- Balanced accuracy < 75%: Fix classifier first (do not tune gates)

### 3. Stratified Report
```bash
python3 scripts/report_threshold_calibration.py \
  --metrics-dir outputs/validation_full_* \
  --labels data/validation_full/labels.csv
```

### 4. Visual Inspection
```bash
# Check top failures
python3 << 'PYEOF'
import json, glob
from pathlib import Path

results = []
for p in glob.glob('outputs/validation_full_*/metrics/*_metrics.json'):
    with open(p) as f:
        d = json.load(f)
    if not d.get('lenient_pass', True):
        results.append((d['image'], d.get('edge_f1', 0), d.get('chamfer_px', 999)))

results.sort(key=lambda x: x[1])  # Sort by edge_f1 ascending
print("\nTop 10 Failures (worst edge F1):")
for img, f1, chamfer in results[:10]:
    print(f"  {img:40s}  F1={f1:.3f}  Chamfer={chamfer:.1f}px")
PYEOF
```

## 🚫 Do NOT Proceed If

- Smoke test fails (fix integration first)
- Disk space <1GB
- Model requires network download (cache first)
- Previous run in progress (kill or wait)

## ✅ Success Criteria

**Baseline Health** (lenient gates):
- Overall lenient pass ≥ 70%
- Texture scenes lenient pass ≥ 80%
- Structure scenes lenient pass ≥ 40%
- Classifier balanced accuracy ≥ 75%

**If criteria met**: Freeze baseline, proceed to DA V2 input-size sweep
**If criteria not met**: Debug classifier or gates (do not integrate MaterialsV3 yet)

---

## Quick Reference: File Locations

- **Validation runner**: `scripts/automation/production_depth_validation_fixed.py`
- **Classifier logic**: `high_fidelity_depth/quality_metrics.py::classify_scene_type_v2()`
- **Quality gates**: `scripts/automation/production_depth_validation_fixed.py` (lines 412-457)
- **Analysis scripts**: `scripts/evaluate_classifier_balanced.py`, `scripts/report_threshold_calibration.py`
- **Dataset**: `data/validation_full/` (50 images + labels.csv)

---

**Last Updated**: 2025-12-19
**Next Session**: Run expanded validation → analyze → decide on DA V2 sweep vs classifier tuning
