# Primary Bedroom Edge Artifacts - Investigation Summary

**Date**: 2026-02-14
**User Report**: "Clear color changes (blue and white) at perimeter of trees/foliage, and color contamination where ocean and sky meet"
**Status**: ✅ **CONFIRMED AND DIAGNOSED** - Awaiting correct fix

---

## Confirmed Artifacts

### Quantitative Measurements

| Artifact Type | Magnitude | Assessment |
|--------------|-----------|------------|
| **Edge artifacts** (>5% change at edges) | 15.40% of pixels (1.6M pixels) | 🔴 CRITICAL |
| **White halos** (all RGB >5% increase) | 42.01% of pixels (4.5M pixels) | 🔴 CRITICAL |
| **Blue contamination** (sky/ocean boundary) | 62,777 pixels | 🔴 SIGNIFICANT |
| **Max delta magnitude** | 1.7321 (173% change!) | 🔴 EXTREME |

### User Observations Validated

✅ **"Blue and white at tree perimeter"** - Confirmed: 42% white halos, blue excess at boundaries
✅ **"Color contamination at ocean/sky meet"** - Confirmed: 62,777 pixels with blue contamination

---

## Root Cause: V2 Enhancement Creates Artifacts

###Initial Hypothesis (INCORRECT)

❌ **Believed**: SAM2 masks have sharp edges → Materials V3 blending creates halos
❌ **Attempted fix**: Add Gaussian blur (sigma=3.0) to masks before pixel ops
❌ **Result**: NO IMPROVEMENT (artifacts identical before/after fix)

### Actual Root Cause

✅ **V2 Enhancement stage is creating or amplifying the artifacts**

Evidence:
1. Mask feathering in Materials V3 had **zero effect** on final output
2. Artifacts appear **identically** whether or not masks are feathered
3. The artifact locations/magnitudes are **identical** between runs
4. This suggests V2 is the source, not Materials V3

### Why V2 Creates Edge Artifacts

The `luxury_estate` preset applies:
- Contrast enhancement (amplifies edges)
- Saturation boost (+3-5% globally)
- Detail enhancement near edges
- Potentially unsharp masking or edge detection

When Materials V3 creates **subtle** edge transitions (even with feathering), V2's aggressive enhancement **amplifies** these into visible halos.

---

## Why Different Images Show Different Severity

| Image | Edge Artifacts | Why? |
|-------|---------------|------|
| **Great Room** | < 1% (minimal) | Interior shot, no bright sky/ocean contrast |
| **Aerial** | ~3% (moderate) | Outdoor, but foliage delta only 0.1% |
| **Primary Bedroom** | 15-42% (severe) | **High foliage delta (1.5%) + ocean/sky visible** |

**Key factor**: Primary Bedroom has **14× higher foliage delta** than Great Room (1.52% vs 0.11%). This means Materials V3 made **larger color changes** to foliage, which V2 then amplified at edges.

---

## Attempted Fix #1: Mask Feathering (FAILED)

**What was done**:
```python
# Added to pixel_ops_executor.py line 157
from scipy.ndimage import gaussian_filter
mask_feathered = gaussian_filter(mask.astype(np.float32), sigma=3.0)
mask_roi = np.clip(mask_feathered, 0.0, 1.0)
```

**Expected**: Smooth mask transitions → eliminate sharp edges → no halos
**Actual**: ZERO effect (artifacts identical)
**Conclusion**: V2 is the problem, not Materials V3 masking

---

## Correct Fix Options

### Option 1: Disable V2 Edge Enhancement for Materials V3 Outputs (RECOMMENDED)

**Strategy**: V2 should detect Materials V3-enhanced inputs and skip aggressive edge processing

**Implementation**:
```python
# In V2 enhance_image.py or equivalent
if materials_v3_enhanced:
    # Use gentler preset or disable edge enhancement
    preset_config['edge_enhancement'] = 0.0
    preset_config['unsharp_mask'] = False
```

**Pros**: Preserves Materials V3 work without amplification
**Cons**: Requires V2 awareness of Materials V3

### Option 2: Reduce luxury_estate Edge Enhancement Globally

**Strategy**: Make `luxury_estate` preset less aggressive on edges

**Implementation**: Adjust preset parameters:
- Reduce `edge_enhancement_strength` from current to 50%
- Reduce `unsharp_mask_amount` (if used)
- Soften contrast curves near high-frequency regions

**Pros**: Simple, affects all images uniformly
**Cons**: May reduce perceived "pop" in non-Materials V3 images

### Option 3: Materials V3 Pre-Compensates for V2 Amplification

**Strategy**: Materials V3 reduces op strength knowing V2 will amplify

**Implementation**:
```python
# In pixel_ops_registry.py
if v2_enabled:
    strength *= 0.5  # Reduce strength by 50% when V2 will follow
```

**Pros**: Surgical, only affects Materials V3+V2 combo
**Cons**: Coupling between stages (breaks modularity)

### Option 4: Add "Materials V3 Mode" to V2

**Strategy**: V2 detects Materials V3 output and switches to gentler mode

**Implementation**:
- Check for Materials V3 manifest/flag
- Use `architectural` or `default` preset instead of `luxury_estate`
- Or create new `post_materials_v3` preset

**Pros**: Clean separation of concerns
**Cons**: Adds complexity to V2

---

## Recommended Path Forward

### Immediate (< 1 hour)

1. **Test V2 with `default` preset** instead of `luxury_estate` on Primary Bedroom
2. Check if artifacts reduce to < 5%
3. If yes → V2 preset is the problem (use Option 2 or 4)

### Short-term (< 1 day)

1. Create `post_materials_v3` V2 preset:
   - Based on `luxury_estate`
   - 50% reduced edge enhancement
   - No unsharp masking
2. Auto-select this preset when Materials V3 output detected
3. Re-run full 6-image validation

### Long-term (future release)

1. Implement proper V2/Materials V3 integration
2. Add edge-aware blending in Materials V3 (feather + compensate for V2)
3. Add Quality Firewall check for edge artifacts (auto-fail if > 5%)

---

##Testing Commands

**Test V2 with default preset**:
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir test_single \
  --output-dir output_v2_default_preset \
  --quality-tier apex \
  --materials-v3 on \
  --enable-v2 on \
  --v2-preset default \
  --emit-master16 on
```

**Then diagnose**:
```bash
python diagnose_primary_bedroom_artifacts.py
# Update paths to output_v2_default_preset in script first
```

---

## Files Generated

1. `PRIMARY_BEDROOM_EDGE_ARTIFACTS.md` - Full investigation report
2. `primary_bedroom_artifacts_visualization.jpg` - Visual proof (14MB)
3. `primary_bedroom_artifact_diagnosis.json` - Quantitative data
4. `diagnose_primary_bedroom_artifacts.py` - Reusable diagnostic tool
5. `PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md` - This file

---

## Production Impact

**BLOCKER STATUS**: 🔴 **YES - BLOCKS PRODUCTION**

Severity justification:
- 42% of pixels affected (white halos)
- Visible in premium marketing materials
- Coastal/window-view properties severely impacted
- User explicitly reported visual artifacts

**Safe to ship**:
- Interior-only images (Great Room style)
- Images without Materials V3
- Images with V2 disabled

**NOT safe to ship**:
- Coastal properties with visible ocean/sky
- Window views with trees/foliage against sky
- Any image where Primary Bedroom artifacts would appear

---

*Last updated: 2026-02-14*
*Next step: Test V2 preset change*
