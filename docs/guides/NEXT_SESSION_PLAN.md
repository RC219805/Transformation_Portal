# Next Session Plan: Validation Expansion & Structure Improvement

**Last Completed**: HF-Energy Texture Gate (77.8% lenient validated)  
**Commits**: 9fd2590 → b5927d1  
**Status**: ✅ Texture-healthy, ⚠️ Structure-limited

---

## 🎯 Current State

### Validated Results (18 images)
- **Overall**: 77.8% lenient (14/18), 16.7% strict (3/18)
- **Texture scenes**: 92.9% lenient (13/14) ✅ FIXED
- **Structure scenes**: 25.0% lenient (1/4) ⚠️ NEEDS WORK

### Key Insight
Texture branch is no longer adversarial. Remaining failures are concentrated in structure-dominated scenes that need higher inference detail.

---

## 🚀 Next Actions (Priority Order)

### 1️⃣ **Expand Validation Dataset** (40–60 images)
**Why**: 18 images is pilot-scale; need statistical confidence before production claims

**Action**:
```bash
# Select 40-60 images from input_images/
# Stratify: 50% texture, 50% structure
# Include: glass facades, foliage, water, interiors with strong geometry

python scripts/select_validation_images.py \
  --input-dir input_images \
  --output-dir data/validation_full \
  --count 50 \
  --stratify texture:25,structure:25
```

**Success Criteria**:
- [ ] 40–60 images selected and labeled
- [ ] Run validation: `scripts/automation/RUN_VALIDATION_HF_FIXED.sh`
- [ ] Lenient ≥70% overall, ≥85% texture, ≥55% structure

---

### 2️⃣ **Classifier Analysis** (confusion matrix)
**Why**: Unknown whether classifier is ≥85% balanced accuracy (target)

**Action**:
```bash
python scripts/analyze_validation_v2.py \
  outputs/validation_hf_fixed_20251218_211645_01fb79c \
  --output classifier_analysis.md
```

**Success Criteria**:
- [ ] Balanced accuracy ≥75% (18-image), ≥85% (40–60 image)
- [ ] Precision/recall ≥0.70 per class
- [ ] Confusion matrix shows diagonal dominance

**If <75%**: Improve classifier before proceeding (multi-factor tuning or learned model)

---

### 3️⃣ **Structure Input-Size Sweep** (DA V2 operating point)
**Why**: Structure scenes fail because edge_f1 ~0.37–0.51 (need higher detail)

**Action** (after 40–60 image baseline is stable):
```bash
# Test structure scenes only
python scripts/depth_input_size_sweep.py \
  --images data/validation_structure_subset \
  --input-sizes 518,768,896,1022 \
  --output outputs/input_size_sweep
```

**Success Criteria**:
- [ ] Edge F1 improves on structure scenes (target ≥0.60 for strict)
- [ ] Compute/memory cost acceptable (<2× vs baseline)
- [ ] No regressions on texture scenes

**Reference**: Depth Anything V2 docs explicitly state `--input-size` defaults to 518 and can be increased for more fine-grained results.

---

### 4️⃣ **MaterialsV3: Shadow Mode** (ONLY after 1–3 stable)
**Status**: Unblocked for shadow-mode integration (log-only, no active gating)

**Action**:
```bash
# Add flag: --scene-classifier {heuristic_v2, materials_v3}
# Default: heuristic_v2
# Shadow mode: run MaterialsV3, log outputs, do NOT change pass/fail
```

**Promotion to Active** (hard requirements):
- [ ] Classifier ≥85% balanced accuracy on 40–60 images (stable)
- [ ] MaterialsV3 improves classification accuracy by ≥10pp on hard categories
- [ ] Runtime overhead acceptable
- [ ] Graceful fallback if model download fails

---

## 📊 Empirical Data to Collect

### From 40–60 Image Run
1. **Classifier metrics**:
   - Confusion matrix
   - Balanced accuracy
   - Per-class precision/recall/F1
   
2. **HF energy distributions**:
   - Texture scenes (should cluster 0.0001–0.0003)
   - Structure scenes (0.0002–0.0008)
   - Validate thresholds empirically

3. **Failure analysis**:
   - Top 5 failures per scene type
   - Review overlays (RGB edges / depth edges / confusion)
   - Classify: real depth issue vs gate too strict

---

## ⚠️ What NOT to Do

❌ **Don't integrate MaterialsV3 into active path yet**  
   → Shadow mode first, prove incremental value

❌ **Don't tune thresholds before expanding dataset**  
   → 18 images insufficient to calibrate reliably

❌ **Don't chase strict gates on texture scenes**  
   → Texture "strict" may be conceptually undefined (no edges to align)

❌ **Don't add more heuristics to structure gate**  
   → Input-size sweep is the right lever, not more rules

---

## 📁 Key Files

- **Validation runner**: `scripts/automation/RUN_VALIDATION_HF_FIXED.sh`
- **Metrics**: `high_fidelity_depth/quality_metrics.py` (HF energy)
- **Gate logic**: `scripts/automation/production_depth_validation_fixed.py`
- **Latest output**: `outputs/validation_hf_fixed_20251218_211645_01fb79c/`

---

## 💡 Decision Points

**If classifier <75% on 40–60 images**:
→ Fix classifier first (tune multi-factor rules or use learned model)

**If structure lenient stays <50% after input-size sweep**:
→ Consider encoder upgrade (vitb → vitl) or accept as model limit

**If texture lenient drops below 85% on expanded set**:
→ Thresholds may be overfit; recalibrate HF energy empirically

---

## ✅ Session Handoff Checklist

- [x] HF-energy texture gate validated (92.9% lenient on 14 texture images)
- [x] Baseline frozen (commit b5927d1)
- [x] Session summary documented
- [x] Next actions prioritized (1 → 2 → 3 → 4)
- [x] Clear decision criteria defined
- [x] Known blockers identified (dataset size, classifier accuracy)

**Ready for next session pickup** 🚀
