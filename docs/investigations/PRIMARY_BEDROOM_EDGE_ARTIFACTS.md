# Primary Bedroom Edge Artifact Investigation

**Date**: 2026-02-13
**Reporter**: User observation
**Status**: 🔴 **CONFIRMED - Real Materials V3 Masking Bug**

---

## User Report

"The primary bedroom image shows clear color changes (blue and white) at the perimeter of trees and foliage, as well as some small localized contamination of the color where the ocean and sky meet"

---

## Quantitative Diagnosis

### Artifact Severity

| Artifact Type | Pixels Affected | Severity |
|--------------|-----------------|----------|
| **Edge artifacts** (>5% change at edges) | 1,634,529 (15.4%) | 🔴 **CRITICAL** |
| **White halos** (all RGB >5% increase) | 4,458,684 (42.0%) | 🔴 **CRITICAL** |
| **Blue contamination** (sky/ocean boundary) | 62,777 pixels | 🔴 **SIGNIFICANT** |

### Sample Artifact Measurements

Real edge artifacts found at:
```
Location (y, x)    | Magnitude | RGB Delta
(1223, 1291)       | 0.7842    | (0.20, 0.41, 0.64)  ← High blue
(1186, 1327)       | 0.5282    | (0.26, 0.44, 0.15)  ← High green
(1496, 1361)       | 0.3846    | (0.26, 0.05, 0.28)  ← Red/Blue
(2507, 2249)       | 0.7730    | (0.41, 0.45, 0.48)  ← White halo
```

### Sky/Ocean Boundary Analysis

**Top 30% of image** (798 rows):
- Mean delta: R=7.5%, G=7.3%, B=7.0%
- Max delta: R/G/B all reached **100%** (complete white)
- Blue excess pixels (>5%): **62,777**

This is **NOT normal enhancement** - something is creating severe artifacts.

---

## Root Cause Analysis

### What's Happening

1. **SAM2 generates masks** with **sharp edges** (0/1 transitions)
2. **Materials V3 pixel ops** (foliage vibrance, glass brightness) are applied using these **unfeathered masks**
3. **Sharp mask blend** creates visible boundaries:
   ```python
   # Current code in pixel_ops_registry.py:
   def _apply_mask_blend(image, mask, modified):
       mask_3 = np.clip(mask, 0.0, 1.0)[..., None]
       return image * (1.0 - mask_3) + modified * mask_3
   ```
   This blends with **no feathering** - sharp transitions visible as halos

4. **V2 enhancement** (luxury_estate preset) **amplifies** these edge artifacts through contrast/saturation adjustments

### Why It's Worse in Primary Bedroom

- **Foliage coverage**: Higher delta (1.5% mean vs 0.1% in Great Room)
- **Ocean/sky visible**: Large blue/white regions where halos are obvious
- **Window views**: Trees framed against bright sky = maximum halo visibility

### Comparison to Great Room

| Image | Foliage Delta | Glass Delta | Edge Artifacts |
|-------|--------------|-------------|----------------|
| **Great Room** | 0.11% | 1.25% | ✅ Minimal (< 1%) |
| **Primary Bedroom** | **1.52%** | 0.64% | 🔴 Severe (15-42%) |

Primary Bedroom has **14× higher foliage delta**, suggesting:
- More aggressive pixel ops, OR
- Larger foliage masks with more edge perimeter

---

## Why This Wasn't Caught Earlier

1. **Great Room analysis showed < 1% change** (different scene composition)
2. **Aerial/Pool specialist analysis** focused on sky/water *regions*, not *edges*
3. **Delta stats in manifests** measure mean change, not edge artifacts
4. **Visual QA** required for edge artifact detection (metrics alone miss it)

---

## Evidence Files

1. **`primary_bedroom_artifacts_visualization.jpg`** (14MB)
   - Left: Input | Middle: Output | Right: Artifact heatmap
   - Red heatmap shows edge artifacts clearly visible

2. **`primary_bedroom_artifact_diagnosis.json`**
   - Full quantitative report
   - Sample artifact coordinates with RGB deltas

3. **`diagnose_primary_bedroom_artifacts.py`**
   - Reusable diagnostic script
   - Edge detection using Sobel gradients
   - Sky/ocean boundary analysis

---

## Recommended Fix (CRITICAL)

### Immediate Fix: Add Mask Feathering to Materials V3

**File**: `src/transformation_portal/lux_depth_v3/pixel_ops_executor.py`

**Add before line 157** (the op execution loop):

```python
# CRITICAL FIX: Feather mask edges to prevent visible halos
# SAM2 masks have sharp edges - blur them for smooth blending
from scipy.ndimage import gaussian_filter

def feather_mask(mask, sigma=3.0):
    """Apply Gaussian blur to mask edges for smooth transitions."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    feathered = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return np.clip(feathered, 0.0, 1.0)

# Apply feathering to mask_roi
mask_roi = feather_mask(mask_roi, sigma=3.0)
```

### Parameters

- **`sigma=3.0`**: Good balance (enough feathering without losing detail)
- **`sigma=2.0`**: Minimal feathering (conservative)
- **`sigma=5.0`**: Aggressive feathering (may blur too much)

### Expected Impact

✅ **Edge artifacts reduced** from 15% to < 1%
✅ **White halos eliminated** (smooth RGB transitions)
✅ **Blue contamination fixed** (gradual sky/foliage blend)
✅ **No quality loss** (masks already high-confidence in interiors)

---

## Alternative/Additional Fixes

### Option 2: Erode Masks (Conservative)

Apply pixel ops **only to interior** of masks (erode by N pixels):

```python
from scipy.ndimage import binary_erosion

def safe_mask(mask, erode_pixels=5):
    """Shrink mask to avoid edges."""
    binary = mask > 0.5
    eroded = binary_erosion(binary, iterations=erode_pixels)
    return eroded.astype(np.float32)
```

**Pros**: Safest (never touches edges)
**Cons**: Misses valid enhancement near edges

### Option 3: Reduce Pixel Op Strength Near Edges

Apply full strength in mask interior, fade to zero at edges:

```python
def gradient_mask(mask, edge_width=10):
    """Create gradient from mask edge inward."""
    from scipy.ndimage import distance_transform_edt

    binary = mask > 0.5
    distance = distance_transform_edt(binary)
    gradient = np.clip(distance / edge_width, 0.0, 1.0)
    return gradient * mask
```

**Pros**: Preserves edge enhancement with smooth falloff
**Cons**: More complex

---

## Testing Plan

### Quick Test (< 5 min)

1. Add mask feathering to `pixel_ops_executor.py`
2. Re-run **Primary Bedroom only**:
   ```bash
   python -m transformation_portal.lux_depth_v3 \
     --input-dir test_single \
     --output-dir output_edge_artifact_fix \
     --quality-tier apex \
     --materials-v3 on \
     --enable-segmentation on
   ```
3. Run diagnostic again:
   ```bash
   python diagnose_primary_bedroom_artifacts.py
   ```
4. Expect: Edge artifacts < 1%, white halos < 5%

### Full Validation (~ 30 min)

1. Re-run all 6 images with fix
2. Compare before/after visualizations
3. Check all manifests for delta stats
4. Visual QA on all outputs

---

## Governance Impact

### Contract Violations

🔴 **Quality Firewall**: Edge artifacts exceed 5% threshold (15.4% detected)
🔴 **Materials V3 Contract**: Pixel ops must not create visible halos
🔴 **16-bit Pipeline**: Artifacts suggest precision loss or blending issues

### Production Blocker?

**YES** - This is a **production blocker** for:
- Images with visible sky/ocean + foliage (coastal properties, window views)
- Close-up shots where edges are prominent
- Marketing materials (halos unacceptable for premium listings)

**NO** - Acceptable for:
- Interior-only shots (Great Room showed minimal artifacts)
- Lower-res outputs (artifacts less visible at web sizes)

### Recommendation

**BLOCK production deployment** until mask feathering implemented and validated.

---

## Priority

**P0 - CRITICAL**

- User-visible quality regression
- Affects core Materials V3 value proposition
- Simple fix available (1 function, ~10 lines of code)
- Validation can happen same day

---

## Next Steps

1. ✅ Root cause identified (no mask feathering)
2. ⏳ Implement mask feathering in `pixel_ops_executor.py`
3. ⏳ Test on Primary Bedroom (expect < 1% edge artifacts)
4. ⏳ Re-run full 6-image validation
5. ⏳ Update `CRITICAL_BUGFIXES_DEPTH_AND_MATERIALS.md` with Bug #3
6. ⏳ Merge fix before production deployment

---

*Diagnosed: 2026-02-13 by quantitative edge detection analysis*
*Validated: User visual observation + automated artifact detection*
*Status: Awaiting fix implementation*
