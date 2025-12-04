# Production Policy: 750 Picacho Grade Selection

**Version:** 1.0  
**Date:** 2025-12-04  
**Status:** Clinical Framework - Ready for Production

---

## 📊 Current State: Full Comparison Matrix

### GreatRoom (Baseline: 57.77%, Best source shot)

| Recipe | Score | Delta | Status | Notes |
|--------|-------|-------|--------|-------|
| **Baseline** | 57.77% | - | 🥇 | High-quality source |
| Signature Estate (Original) | 52.54% | -5.23% | ❌ | Too aggressive |
| **Interior Warm Minimal** | 51.92% | -5.85% | ⚠️ | Lighter but still losing quality |

**Analysis:** Even minimal processing loses ~6%. This shot doesn't need enhancement.

### PrimaryBedroom (Baseline: 60.40%, Highest quality)

| Recipe | Score | Delta | Status | Notes |
|--------|-------|-------|--------|-------|
| **Baseline** | 60.40% | - | 🥇 | Pristine source |
| Signature Estate (Original) | 56.80% | -3.60% | ⚠️ | Acceptable loss |
| **Signature Estate (Gentle)** | 55.78% | -4.62% | ⚠️ | Gentler but worse |

**Analysis:** Original recipe actually performs better here. High-quality sources resist "gentle" adjustments.

### Kitchen (Baseline: 54.47%)

| Recipe | Score | Delta | Status | Notes |
|--------|-------|-------|--------|-------|
| **Baseline** | 54.47% | - | 🥇 | Good source |
| Signature Estate (Original) | 50.29% | -4.18% | ⚠️ | Moderate loss |

**Analysis:** Needs testing with gentle/minimal variants.

### Aerial (Baseline: 42.20%, Needs enhancement)

| Recipe | Score | Delta | Status | Notes |
|--------|-------|-------|--------|-------|
| **Baseline** | 42.20% | - | ⚠️ | Weak source |
| Signature Estate (Original) | 45.49% | +3.29% | ✅ | Good improvement |
| **Exterior Enhanced** | 48.04% | +5.84% | 🏆 | **Clear winner** |

**Analysis:** Low baseline benefits from aggressive enhancement. Exterior Enhanced is the correct choice.

### Pool (Baseline: 50.06%)

| Recipe | Score | Delta | Status | Notes |
|--------|-------|-------|--------|-------|
| **Baseline** | 50.06% | - | 🥇 | Moderate source |
| Signature Estate (Original) | 44.88% | -5.18% | ❌ | Significant loss |
| Pool Estate (Current) | 40.53% | -9.53% | ❌❌ | **Over-processed** |

**Analysis:** Pool-specific recipe is too aggressive. Temporarily shelve.

### PrimaryBathroom (Baseline: 49.42%)

| Recipe | Score | Delta | Status | Notes |
|--------|-------|-------|--------|-------|
| **Baseline** | 49.42% | - | 🥇 | Moderate source |
| Signature Estate (Original) | 45.03% | -4.39% | ⚠️ | Moderate loss |

**Analysis:** Needs testing with gentle variants.

---

## 🎯 Production Decision Framework

### Rule #1: Let the Baseline Guide You

**If Baseline ≥ 55% (Hero Shots):**
- Default: **Keep as baseline** or use absolute minimal processing
- These shots are already excellent
- Any "film look" will cost 4-6% 
- **Question to ask:** Is the brand look worth the quality loss?

**If Baseline 45-55% (Good Shots):**
- Use: **Signature Estate (Gentle)** or **Interior Warm Minimal**
- Accept 3-5% loss for film character
- Monitor results - if still losing > 5%, consider baseline

**If Baseline < 45% (Needs Enhancement):**
- Exteriors/Aerials: **Exterior Enhanced**
- Interiors: **Signature Estate (Original)**
- These shots benefit from aggressive processing

### Rule #2: Scene Type Matters

**Interiors (GreatRoom, Bedrooms, Bathrooms, Kitchen):**
```yaml
Priority:
  1. Baseline (if score > 55%)
  2. Interior Warm Minimal (if score 50-55%)
  3. Signature Estate Gentle (if score < 50%)

Accept: 3-5% loss maximum
```

**Exteriors/Aerials:**
```yaml
Priority:
  1. Exterior Enhanced (always for aerial)
  2. Signature Estate (for high-baseline exteriors)

Target: +3-6% improvement
```

**Pool/Water:**
```yaml
Status: High-risk - no reliable recipe yet

Priority:
  1. Baseline (safest)
  2. Signature Estate Gentle (experimental)
  3. DO NOT USE Pool Estate (currently over-processes)
```

---

## 📋 750 Picacho Specific Recommendations

### Batch 1: Keep as Baseline (No Processing)
```
- 750Picacho_GreatRoom.jpg     (57.77%)
- 750Picacho_PrimaryBedroom.jpg (60.40%)
- 750Picacho_Kitchen.jpg        (54.47%)
```

**Rationale:** These are already 54-60%. Any processing loses 4-6% for minimal visual gain. Ship these as-is or with *absolute minimal* adjustments if brand consistency requires it.

### Batch 2: Light Enhancement
```
- 750Picacho_Pool.jpg           (50.06%)
- 750Picacho_PrimaryBathroom.jpg (49.42%)
```

**Recipe:** Signature Estate (Gentle) - Test first  
**Expected:** 45-47% (3-5% loss acceptable)  
**Review Required:** Yes - visual comparison mandatory

### Batch 3: Full Enhancement
```
- 750Picacho_Aerial.jpg         (42.20%)
```

**Recipe:** Exterior Enhanced  
**Expected:** 48%+ (+6% gain)  
**Status:** Proven - use with confidence

---

## ⚠️ Critical Realizations

### 1. Your Source Photography is Excellent
- Average baseline: 52.39%
- Top 3 shots: 54-60%
- **This is not a "fix bad renders" situation**
- **This is a "preserve quality while adding subtle character" situation**

### 2. Film Emulation Has a Cost
- Kodak 2393 LUT @ 0.85 strength: ~5% quality loss
- Even @ 0.60 strength (gentle): ~4.5% loss
- Even @ 0.45 strength (minimal): ~6% loss on great shots

**Reality Check:** For hero shots, the "film look" might not be worth it.

### 3. The Metric is Telling You Something Important
When a 60% source drops to 52-56% across ALL recipe variants, it's saying:
- "This image doesn't need help"
- "Stop trying to improve perfection"
- "Any processing is degradation, not enhancement"

---

## 🚀 Recommended Workflow

### Step 1: Baseline Assessment (Always)
```bash
python -c "from transformation_portal.cli import app; app()" pipeline process \
  -i "input_images/batch/*.jpg" \
  -o "output_baseline" \
  -r config/recipes/baseline_quality.yaml
```

### Step 2: Route by Score
```python
# Pseudo-logic for automation
if baseline_score >= 55:
    decision = "KEEP_AS_BASELINE"
    action = "Ship without processing"

elif 45 <= baseline_score < 55:
    if scene_type == "interior":
        recipe = "signature_estate_gentle.yaml"
        action = "Process with gentle touch"
    elif scene_type == "exterior":
        recipe = "exterior_enhanced.yaml"
        action = "Full enhancement OK"

else:  # baseline_score < 45
    recipe = "exterior_enhanced.yaml"
    action = "Aggressive enhancement recommended"
```

### Step 3: Visual Review (Mandatory)
- Compare processed vs baseline side-by-side
- If processed looks "different but not better" → ship baseline
- Only use processed if it's clearly superior

---

## 📈 Success Metrics (Revised)

**Previous Target:**
- Interiors: < 3% loss
- Exteriors: > 3% gain

**Revised Reality-Based Target:**
- **Hero shots (≥55%):** 0-2% loss OR ship as baseline
- **Good shots (45-55%):** 3-5% loss acceptable if look is brand-appropriate
- **Weak shots (<45%):** +3-6% gain target

**Current Status:**
- ✅ Exteriors: Hitting target (+5.84%)
- ⚠️ Interiors: 4-6% loss even with gentle recipes
- ⚠️ Pool: Over-processed, needs revision

---

## 🔧 Next Iterations

### Priority 1: Create "barely there" interior recipe
```yaml
# interior_whisper.yaml (concept)
color_grading:
  lut_strength: 0.25    # Barely perceptible
  contrast: 1.00        # No change
  saturation: 1.01      # Almost imperceptible
  warmth: 0.02          # Tiny warmth hint
```

**Target:** <2% loss on hero shots while still adding "something"

### Priority 2: Conditional automation
Build script that:
1. Runs baseline on all images
2. Auto-routes based on score + filename/scene hints
3. Generates comparison report
4. Flags images for manual review

### Priority 3: Pool recipe revision
Test Pool with:
- LUT strength 0.30-0.40 (much lighter)
- Minimal saturation adjustments
- Target: 2-3% loss, not 10%

---

## 📌 Production Lock

**For 750 Picacho Final Delivery:**

**Use Baseline (No Processing):**
- GreatRoom (57.77%)
- PrimaryBedroom (60.40%)
- Kitchen (54.47%)

**Use Exterior Enhanced:**
- Aerial (42.20% → 48.04%)

**Client Review Required:**
- Pool (test Gentle vs Baseline)
- PrimaryBathroom (test Gentle vs Baseline)

**Estimated Quality:**
- 3 hero shots at 54-60% (pristine)
- 1 aerial at 48% (significantly improved)
- 2 shots TBD pending review (likely 45-50%)

**Average Expected:** ~52-54% (excellent for luxury real estate)

---

**Bottom Line:** You've built a proper grading framework. Now use it clinically: enhance what needs it, leave excellence alone.
