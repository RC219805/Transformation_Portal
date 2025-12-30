# Phase 1 Visual Verification Checklist
## Color/Tone Parameters — Kitchen & Great Room

**Date**: 2025-12-22
**Reviewer**: _________
**Duration**: ~15-20 minutes

---

## ⚙️ Setup Protocol

### Files to Review (master16.tif ONLY)

**Parameter 1: saturation_protection**
```
sweep_runs/color_tone_saturation_protection_delta0/outputs/750Picacho_Kitchen_UltraQuality_master16.tif
sweep_runs/color_tone_saturation_protection_delta1/outputs/750Picacho_Kitchen_UltraQuality_master16.tif
sweep_runs/color_tone_saturation_protection_delta2/outputs/750Picacho_Kitchen_UltraQuality_master16.tif

sweep_runs/color_tone_saturation_protection_delta0/outputs/750Picacho_GreatRoom_UltraQuality_master16.tif
sweep_runs/color_tone_saturation_protection_delta1/outputs/750Picacho_GreatRoom_UltraQuality_master16.tif
sweep_runs/color_tone_saturation_protection_delta2/outputs/750Picacho_GreatRoom_UltraQuality_master16.tif
```

**Parameter 2: local_contrast_gain**
```
sweep_runs/color_tone_local_contrast_gain_delta0/outputs/750Picacho_Kitchen_UltraQuality_master16.tif
sweep_runs/color_tone_local_contrast_gain_delta1/outputs/750Picacho_Kitchen_UltraQuality_master16.tif
sweep_runs/color_tone_local_contrast_gain_delta2/outputs/750Picacho_Kitchen_UltraQuality_master16.tif

sweep_runs/color_tone_local_contrast_gain_delta0/outputs/750Picacho_GreatRoom_UltraQuality_master16.tif
sweep_runs/color_tone_local_contrast_gain_delta1/outputs/750Picacho_GreatRoom_UltraQuality_master16.tif
sweep_runs/color_tone_local_contrast_gain_delta2/outputs/750Picacho_GreatRoom_UltraQuality_master16.tif
```

### Viewing Requirements
- ✅ Side-by-side, same zoom level (100% or fit-to-screen)
- ✅ Same scene across all 3 deltas simultaneously
- ✅ Color-accurate display (sRGB minimum)
- ❌ NO toggling/blinking between files
- ❌ NO relying on memory
- ❌ NO viewing marketing PNGs or previews

---

## 📋 Parameter 1: saturation_protection

**Values Tested:**
- Delta 0 (baseline): 1.0
- Delta 1: 0.8 (conservative, flatter)
- Delta 2: 0.85 (moderate suppression) ← **Predicted winner**

### Kitchen Scene Review

**Wood Cabinetry Warmth**
- [ ] Delta 0: Natural / Hot / Instagram-warm / Dead
- [ ] Delta 1: Natural / Hot / Instagram-warm / Dead
- [ ] Delta 2: Natural / Hot / Instagram-warm / Dead

**Stone/Countertop Neutrality**
- [ ] Delta 0: Clean neutral / Green creep / Magenta shift
- [ ] Delta 1: Clean neutral / Green creep / Magenta shift
- [ ] Delta 2: Clean neutral / Green creep / Magenta shift

**LED Highlights (if visible)**
- [ ] Delta 0: Clean / Chroma burn / Over-saturated
- [ ] Delta 1: Clean / Chroma burn / Over-saturated
- [ ] Delta 2: Clean / Chroma burn / Over-saturated

**Overall Kitchen Assessment**
- [ ] Delta 0: LOCK / HOLD / ARCHIVE
- [ ] Delta 1: LOCK / HOLD / ARCHIVE
- [ ] Delta 2: LOCK / HOLD / ARCHIVE

### Great Room Scene Review

**Color Separation**
- [ ] Delta 0: Distinct materials / Muddy / Over-saturated
- [ ] Delta 1: Distinct materials / Muddy / Over-saturated
- [ ] Delta 2: Distinct materials / Muddy / Over-saturated

**Shadow Chroma**
- [ ] Delta 0: Clean / Gray / Color-contaminated
- [ ] Delta 1: Clean / Gray / Color-contaminated
- [ ] Delta 2: Clean / Gray / Color-contaminated

**Overall Great Room Assessment**
- [ ] Delta 0: LOCK / HOLD / ARCHIVE
- [ ] Delta 1: LOCK / HOLD / ARCHIVE
- [ ] Delta 2: LOCK / HOLD / ARCHIVE

### **Final Decision: saturation_protection**
- [ ] **LOCK** Delta _____ (clear improvement, no artifacts)
- [ ] **HOLD** (scene-dependent, needs more data)
- [ ] **ARCHIVE ALL** (no clear winner or artifacts present)

**Rationale (1 sentence):**
_____________________________________________________________________________

---

## 📋 Parameter 2: local_contrast_gain

**Values Tested:**
- Delta 0 (baseline): 2.0
- Delta 1: 2.5 (increased local contrast)
- Delta 2: 2.2 (moderate increase)

### Kitchen Scene Review

**Edge Separation (cabinets, backsplash)**
- [ ] Delta 0: Clean separation / Halos present / Soft/muddy
- [ ] Delta 1: Clean separation / Halos present / Soft/muddy
- [ ] Delta 2: Clean separation / Halos present / Soft/muddy

**Shadow Depth (under cabinets, corners)**
- [ ] Delta 0: Natural depth / Crunchy / Lifted/flat
- [ ] Delta 1: Natural depth / Crunchy / Lifted/flat
- [ ] Delta 2: Natural depth / Crunchy / Lifted/flat

**White Surfaces (walls, ceilings)**
- [ ] Delta 0: Clean white / Chalky / Gray
- [ ] Delta 1: Clean white / Chalky / Gray
- [ ] Delta 2: Clean white / Chalky / Gray

**Texture Rendering**
- [ ] Delta 0: Natural / Exaggerated / Smoothed
- [ ] Delta 1: Natural / Exaggerated / Smoothed
- [ ] Delta 2: Natural / Exaggerated / Smoothed

**Overall Kitchen Assessment**
- [ ] Delta 0: LOCK / HOLD / ARCHIVE
- [ ] Delta 1: LOCK / HOLD / ARCHIVE
- [ ] Delta 2: LOCK / HOLD / ARCHIVE

### Great Room Scene Review

**Clarity vs. Artifacts**
- [ ] Delta 0: Balanced / HDR smell / Too flat
- [ ] Delta 1: Balanced / HDR smell / Too flat
- [ ] Delta 2: Balanced / HDR smell / Too flat

**Midtone Separation**
- [ ] Delta 0: Good / Excessive / Collapsed
- [ ] Delta 1: Good / Excessive / Collapsed
- [ ] Delta 2: Good / Excessive / Collapsed

**Overall Great Room Assessment**
- [ ] Delta 0: LOCK / HOLD / ARCHIVE
- [ ] Delta 1: LOCK / HOLD / ARCHIVE
- [ ] Delta 2: LOCK / HOLD / ARCHIVE

### **Final Decision: local_contrast_gain**
- [ ] **LOCK** Delta _____ (clear improvement, no artifacts)
- [ ] **HOLD** (scene-dependent, needs more data)
- [ ] **ARCHIVE ALL** (no clear winner or artifacts present)

**Rationale (1 sentence):**
_____________________________________________________________________________

---

## 🎯 Phase 1 Color/Tone Status

### Locked Parameters
- saturation_protection: _____ (value: _____)
- local_contrast_gain: _____ (value: _____)

### Next Action
- [ ] Proceed to depth.gamma sweep (parameters locked)
- [ ] Re-sweep with adjusted deltas (parameters held)
- [ ] Pause Phase 1 (artifacts detected)

---

## 🚨 Artifact Watch (Critical)

If you see ANY of these, ARCHIVE immediately:

**Saturation Protection:**
- [ ] "Instagram warmth" glow
- [ ] Dead/lifeless gray midtones
- [ ] Green or magenta color casts in neutrals
- [ ] Chroma clipping on bright surfaces

**Local Contrast Gain:**
- [ ] Halos around edges
- [ ] Crunchy/artificial shadows
- [ ] Chalky whites
- [ ] "HDR" over-processed look
- [ ] Unnatural texture exaggeration

---

## ⏱️ Time Check

- Kitchen scenes: ~7 minutes
- Great Room scenes: ~7 minutes
- Documentation: ~3 minutes

**Total: 15-20 minutes**

---

## 📝 Notes / Observations

_____________________________________________________________________________

_____________________________________________________________________________

_____________________________________________________________________________

---

**Verification Complete**: _____ (initial)
**Timestamp**: _____________________
**Ready for Depth Sweep**: [ ] YES / [ ] NO
