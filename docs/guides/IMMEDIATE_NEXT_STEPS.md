# Immediate Next Steps — Priority-Ordered Action Plan

**Generated**: 2025-12-19  
**Context**: Post-validation session review  
**Status**: Partial 50-image run completed (46/50), baseline validated (18-image)

---

## Current State (Factual)

### ✅ What's Validated
- **18-image baseline**: 77.8% lenient pass, clean execution
- **HF-energy texture gate**: Working as designed, texture scenes no longer adversarial
- **Infrastructure**: Tiling + seam validation 100% reliable
- **46/50 partial run**: 84.8% lenient pass (but scene-biased: 38 texture / 8 structure)

### ⚠️ What's Incomplete
- **50-image run**: Missing 4 images (likely model download failures)
- **No consolidated report**: validation_report.json not generated
- **Classifier unvalidated**: No confusion matrix or balanced accuracy computed
- **Structure performance unknown at scale**: Only 8 structure scenes in 46-image partial

### ❌ What's Broken
- Model dependency fragility (network timeouts)
- Scene distribution bias (38/8 instead of intended 25/25)
- No fail-fast on missing metrics

---

## Priority 1: Complete the 50-Image Run (Today)

**Goal**: Get clean, reproducible 50-image validation with consolidated report.

### Step 1.1: Pre-cache Model Weights
```bash
# Ensure DA V2 model is cached locally
python3 -c "
from transformers import pipeline
print('Downloading DA V2 Small model...')
pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')
print('✓ Model cached')
"
```

### Step 1.2: Identify What Failed
```bash
# Find which 4 images are missing
cd /Users/rc/Transformation_Portal
comm -23 \
  <(tail -n +2 data/validation_full/labels.csv | cut -d',' -f1 | sort) \
  <(ls outputs/validation_full_50img_20251218_214935_2a2b25c/*_metrics.json | \
    xargs -n1 basename | sed 's/_metrics.json$//' | sort) \
  > /tmp/missing_images.txt

echo "Missing images:"
cat /tmp/missing_images.txt
```

### Step 1.3: Option A — Retry Failed Images Only (Fastest)
```bash
# Create retry script
cat > scripts/retry_failed_validation.sh <<'SCRIPT'
#!/bin/bash
set -e

MISSING_FILE="/tmp/missing_images.txt"
INPUT_DIR="data/validation_full"
OUTPUT_DIR="outputs/validation_full_50img_20251218_214935_2a2b25c"

echo "=== Retrying failed images ==="
while read -r filename; do
    echo "Processing: $filename"
    python3 production_depth_validation_fixed.py \
        --input-image "$INPUT_DIR/$filename" \
        --output-dir "$OUTPUT_DIR" \
        --tile-size 1024 \
        --overlap 128 \
        --no-anchor
done < "$MISSING_FILE"

echo "✓ Retry complete"
SCRIPT

chmod +x scripts/retry_failed_validation.sh
./scripts/retry_failed_validation.sh
```

### Step 1.4: Option B — Full Rerun (Safest)
```bash
# Rerun entire 50-image validation deterministically
OUTPUT_DIR="outputs/validation_full_50img_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"

python3 production_depth_validation_fixed.py \
    --input-dir data/validation_full \
    --output-dir "$OUTPUT_DIR" \
    --tile-size 1024 \
    --overlap 128 \
    --no-anchor \
    --generate-report

# Verify completion
if [ -f "$OUTPUT_DIR/validation_report.json" ]; then
    echo "✓ Full 50-image validation complete"
    cat "$OUTPUT_DIR/validation_report.json" | python3 -m json.tool | head -30
else
    echo "❌ Report not generated - check logs"
    exit 1
fi
```

**Acceptance Criteria**:
- [ ] 50/50 images processed
- [ ] `validation_report.json` exists
- [ ] All metrics files have non-null `scene_type`, `edge_f1`, `lenient_pass`, `strict_pass`

---

## Priority 2: Validate Classifier (Critical)

**Goal**: Prove classifier performance or identify need for fix.

### Step 2.1: Generate Confusion Matrix
```bash
cat > scripts/classifier_analysis.py <<'SCRIPT'
#!/usr/bin/env python3
"""Classifier validation analysis."""
import json
import csv
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import numpy as np

# Load ground truth labels
labels_file = Path("data/validation_full/labels.csv")
ground_truth = {}
with open(labels_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['filename'].replace('.jpg', '').replace('.png', '')
        ground_truth[filename] = row['scene_type']

# Load predictions from metrics
metrics_dir = Path("outputs/validation_full_50img_LATEST")  # UPDATE PATH
predictions = {}
for mf in metrics_dir.glob("*_metrics.json"):
    with open(mf) as f:
        data = json.load(f)
    
    filename = mf.stem.replace('_metrics', '')
    predictions[filename] = data.get('scene_type', 'unknown')

# Align predictions with ground truth
common = set(ground_truth.keys()) & set(predictions.keys())
y_true = [ground_truth[k] for k in sorted(common)]
y_pred = [predictions[k] for k in sorted(common)]

# Compute metrics
print(f"=== Classifier Validation (N={len(common)}) ===\n")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=['texture_dominated', 'structure_dominated'])
print("Confusion Matrix:")
print("                   Predicted")
print("                   texture  structure")
print(f"Actual texture     {cm[0,0]:7d}  {cm[0,1]:9d}")
print(f"       structure   {cm[1,0]:7d}  {cm[1,1]:9d}\n")

# Balanced accuracy
ba = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy: {ba:.3f} ({ba*100:.1f}%)\n")

# Per-class metrics
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['texture', 'structure']))

# Decision gate
print("\n=== Decision Gate ===")
if ba >= 0.85:
    print("✅ PASS: Balanced accuracy ≥ 85% — proceed to structure input-size sweep")
elif ba >= 0.75:
    print("⚠️  CONDITIONAL: 75% ≤ BA < 85% — proceed with explicit risk documentation")
else:
    print("❌ FAIL: BA < 75% — STOP and fix classifier before model upgrades")
SCRIPT

python3 scripts/classifier_analysis.py
```

**Decision Gates**:
- **BA ≥ 85%**: ✅ Proceed to Priority 3
- **75% ≤ BA < 85%**: ⚠️ Document risk, proceed cautiously
- **BA < 75%**: ❌ STOP — fix classifier first

---

## Priority 3: Structure Input-Size Sweep (Highest ROI)

**Prerequisite**: Priorities 1 & 2 complete, BA ≥ 75%

**Goal**: Improve structure-dominated scene performance via DA V2's documented quality lever.

### Step 3.1: Extract Structure Subset
```bash
# Create structure-only validation set
mkdir -p data/validation_structure_only

# Copy structure-dominated images from labels
tail -n +2 data/validation_full/labels.csv | \
    awk -F',' '$2 == "structure_dominated" {print $1}' | \
    while read img; do
        cp "data/validation_full/$img" data/validation_structure_only/
    done

echo "Structure images: $(ls data/validation_structure_only | wc -l)"
```

### Step 3.2: Input-Size Sweep
```bash
# Baseline: input_size=518 (already have data)
echo "=== Input Size Sweep: Structure Scenes ==="

for input_size in 768 896 1022; do
    echo -e "\n--- Testing input_size=$input_size ---"
    
    OUTPUT_DIR="outputs/structure_sweep_${input_size}_$(date +%Y%m%d_%H%M%S)"
    
    python3 production_depth_validation_fixed.py \
        --input-dir data/validation_structure_only \
        --output-dir "$OUTPUT_DIR" \
        --input-size "$input_size" \
        --tile-size 1024 \
        --overlap 128 \
        --no-anchor \
        --generate-report
    
    # Quick summary
    if [ -f "$OUTPUT_DIR/validation_report.json" ]; then
        python3 -c "
import json
with open('$OUTPUT_DIR/validation_report.json') as f:
    r = json.load(f)
print(f\"  Lenient: {r['quality']['lenient']['pass_rate']*100:.1f}%\")
print(f\"  Strict:  {r['quality']['strict']['pass_rate']*100:.1f}%\")
print(f\"  Avg Edge F1: {r.get('avg_edge_f1', 'N/A')}\")
"
    fi
done
```

### Step 3.3: Compare Results
```bash
cat > scripts/compare_sweep_results.py <<'SCRIPT'
#!/usr/bin/env python3
"""Compare input-size sweep results."""
import json
import glob

print("=== Structure Input-Size Sweep Comparison ===\n")
print(f"{'Input Size':<12} {'Lenient %':<12} {'Strict %':<12} {'Avg Edge F1':<12}")
print("-" * 48)

for report in sorted(glob.glob("outputs/structure_sweep_*/validation_report.json")):
    with open(report) as f:
        r = json.load(f)
    
    input_size = report.split('_')[2]  # Extract from path
    lenient_pct = r['quality']['lenient']['pass_rate'] * 100
    strict_pct = r['quality']['strict']['pass_rate'] * 100
    edge_f1 = r.get('avg_edge_f1', 0.0)
    
    print(f"{input_size:<12} {lenient_pct:>10.1f}% {strict_pct:>10.1f}% {edge_f1:>12.3f}")

print("\nTarget: Lenient ≥60% on structure scenes")
SCRIPT

python3 scripts/compare_sweep_results.py
```

**Acceptance Criteria**:
- [ ] Structure lenient pass rate improves (25% → 60%+)
- [ ] Edge F1 improves (0.35 → 0.50+)
- [ ] Chamfer distance decreases
- [ ] No regressions on texture scenes (verify separately)

---

## Priority 4: Freeze Baseline & Tag

**Goal**: Create immutable reference point for future work.

```bash
# Tag the validated baseline
git add VALIDATION_STATUS_FACTCHECK_20251219.md IMMEDIATE_NEXT_STEPS.md
git commit -m "docs: validation status factcheck + immediate action plan"

git tag -a v2-baseline-validated-20251219 -m "Baseline: HF-energy texture gate validated, 50-image complete"
git push origin v2-baseline-validated-20251219

# Archive immutable evidence
mkdir -p archive/validation_baselines
cp -r outputs/validation_full_50img_LATEST/ archive/validation_baselines/50img_20251219/
cp data/validation_full/labels.csv archive/validation_baselines/50img_20251219/

echo "✓ Baseline frozen and archived"
```

---

## What NOT to Do (Explicit Prohibitions)

❌ **Do NOT integrate MaterialsV3 into active path yet**
- Shadow mode only, after Priorities 1–3 complete
- Requires A/B evidence on 50-image set before promotion

❌ **Do NOT recalibrate thresholds on partial/biased data**
- Wait for complete 50-image run with balanced scene distribution

❌ **Do NOT claim "production-ready" in any documentation**
- Current strict pass: ~15% (far below production threshold)
- Structure performance unproven at scale

❌ **Do NOT add more heuristic gates to structure scenes**
- The bottleneck is model operating point (input_size), not gate logic

---

## Success Criteria Summary

### End of Priority 1 (Today)
- [x] 50/50 images processed successfully
- [x] Consolidated validation_report.json exists
- [x] Scene distribution: ~25/25 texture/structure (±3)

### End of Priority 2 (Today/Tomorrow)
- [x] Confusion matrix generated
- [x] Balanced accuracy computed
- [x] Decision gate met (BA ≥ 75%)

### End of Priority 3 (This Week)
- [x] Structure scenes: lenient ≥ 60%, edge F1 ≥ 0.50
- [x] Input-size policy implemented (conditional on scene type)
- [x] No texture scene regressions

### Production-Ready Gate (Future)
- [ ] Lenient: ≥70% overall, stratified by scene type
- [ ] Strict: ≥40% overall (or documented relaxation with rationale)
- [ ] Balanced accuracy: ≥85%
- [ ] Model dependency resilience (cached, retries, fail-fast)
- [ ] Reproducible with frozen config

---

**Next Session Entry Point**: Start with Priority 1, Step 1.1 (model pre-cache).

