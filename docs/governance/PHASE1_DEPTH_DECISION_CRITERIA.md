# Phase 1 Depth Decision Criteria
## Visual Verification Guidelines for Depth Parameter Sweeps

**Date**: 2025-12-22
**Domain**: Depth Processing (4 parameters)
**Applicable After**: Color/Tone parameters locked
**Reference**: `PHASE1_VISUAL_VERIFICATION_CHECKLIST.md` (color/tone template)

---

## Purpose

This document defines **objective visual criteria** for evaluating depth parameter sweeps during Phase 1 parameter exploration.

**Critical Distinction**: Depth parameters affect **geometric structure** (edges, depth separation), NOT color appearance. Do not evaluate color quality here—that was locked in color/tone sweeps.

---

## Depth Parameter Decision Framework

### 1. `depth.gamma` - Depth Curve Shape

**What it controls:**
- Overall shape of depth curve (linear → S-curve)
- Foreground/background separation intensity
- Depth map dynamic range compression

**Visual Indicators:**

| Visual Cue | Gamma Too Low (<0.9) | Optimal (0.9-1.1) | Gamma Too High (>1.1) |
|------------|----------------------|-------------------|-----------------------|
| **Foreground Separation** | Weak, muddy boundaries | Clean, distinct layers | Over-separated, halos |
| **Background Detail** | Retained, less depth feel | Balanced | Lost, flattened |
| **Depth Transitions** | Gradual, soft | Natural | Abrupt, posterized |
| **Shadow Depth** | Visible, retained | Balanced | Crushed, lost |

**LOCK Criteria:**
- ✅ Foreground subject clearly separates from background
- ✅ Depth transitions feel natural (no abrupt jumps)
- ✅ Shadow detail preserved (not crushed black)
- ✅ Background retains some detail (not flattened)

**HOLD Criteria:**
- ⚠️ Kitchen and Great Room disagree on optimal gamma
- ⚠️ Marginal differences between deltas (within noise)

**ARCHIVE Criteria:**
- ❌ All deltas show posterization (abrupt depth jumps)
- ❌ All deltas lose shadow detail (crushed blacks)
- ❌ Halos around foreground subjects

---

### 2. `depth.percentile_clip_low` - Shadow Depth Retention

**What it controls:**
- How much shadow depth information is preserved
- Dark area depth map range clipping
- Prevents depth map noise in shadows

**Visual Indicators:**

| Visual Cue | Clip Too Aggressive (<0.3) | Optimal (0.3-0.7) | Clip Too Conservative (>0.7) |
|------------|----------------------------|-------------------|------------------------------|
| **Shadow Areas** | Clean, uniform depth | Natural gradation | Noisy, grainy |
| **Dark Corners** | Smooth, consistent | Subtle depth cues | Artifacts, mottling |
| **Under Cabinets (Kitchen)** | Flat, no depth | Gentle recession | Busy, texture noise |
| **Dark Furniture** | Simplified | Preserved form | Over-detailed, messy |

**LOCK Criteria:**
- ✅ Shadow areas show gentle depth gradation (not flat)
- ✅ Dark corners free of mottling artifacts
- ✅ Under-cabinet areas feel recessed (not dead black)
- ✅ Dark furniture retains form (not noisy texture)

**HOLD Criteria:**
- ⚠️ Scene-dependent (Kitchen shadows vs Great Room shadows differ)
- ⚠️ Trade-off between noise suppression and depth retention unclear

**ARCHIVE Criteria:**
- ❌ All deltas show grainy shadow artifacts
- ❌ All deltas flatten shadow depth (dead black)
- ❌ Depth map noise visible in output images

---

### 3. `depth.edge_filter_sigma_color` - Edge Smoothness

**What it controls:**
- Bilateral filter strength for depth map smoothing
- Edge preservation vs noise reduction trade-off
- Controls "crispness" of depth transitions

**Visual Indicators:**

| Visual Cue | Sigma Too Low (<50) | Optimal (50-100) | Sigma Too High (>100) |
|------------|---------------------|------------------|-----------------------|
| **Cabinet Edges** | Sharp, possibly harsh | Clean, defined | Soft, bleeding |
| **Architectural Lines** | Crisp, may have jaggies | Smooth, straight | Blurred, mushy |
| **Texture Surfaces** | Preserved, may be noisy | Balanced | Smoothed, lost detail |
| **Depth Halos** | Minimal risk | Controlled | Visible, glowing edges |

**LOCK Criteria:**
- ✅ Architectural edges (cabinets, walls) are clean and straight
- ✅ No visible halos around objects
- ✅ Texture detail preserved on surfaces (wood grain, stone)
- ✅ Depth transitions smooth (no jagged edges)

**HOLD Criteria:**
- ⚠️ Trade-off between edge crispness and halo suppression unclear
- ⚠️ Texture preservation vs smoothness depends on scene

**ARCHIVE Criteria:**
- ❌ All deltas show halos around edges
- ❌ All deltas blur architectural lines unacceptably
- ❌ All deltas lose critical texture detail

---

### 4. `depth.banding_suppression` - Posterization Control

**What it controls:**
- Depth map gradient smoothing to prevent "stair-stepping"
- Subtle depth transitions instead of discrete bands
- Histogram equalization strength

**Visual Indicators:**

| Visual Cue | Suppression Too Low (<0.003) | Optimal (0.003-0.007) | Suppression Too High (>0.007) |
|------------|------------------------------|----------------------|-------------------------------|
| **Smooth Walls** | Visible depth bands | Smooth gradient | Over-smoothed, flat |
| **Ceiling Gradients** | Stair-stepping | Continuous | Lost depth information |
| **Floor Planes** | Discrete bands | Subtle gradation | Uniform, no depth feel |
| **Curved Surfaces** | Jagged, faceted | Smooth curvature | Blob-like, detail loss |

**LOCK Criteria:**
- ✅ Smooth walls show continuous depth gradient (no bands)
- ✅ Ceiling and floor planes feel naturally receding
- ✅ Curved surfaces appear smooth (not faceted)
- ✅ Still retains depth information (not over-smoothed)

**HOLD Criteria:**
- ⚠️ Banding visible but tolerable
- ⚠️ Scene-dependent (Kitchen flat surfaces vs Great Room curves)

**ARCHIVE Criteria:**
- ❌ All deltas show visible banding on smooth surfaces
- ❌ All deltas over-smooth depth (lose 3D feel)
- ❌ Posterization artifacts obvious at normal viewing distance

---

## Decision Workflow (After Running Depth Sweep)

### Step 1: Load Files for Comparison (Example: depth.gamma)

**Open side-by-side:**
```
sweep_runs/depth_gamma_delta0/outputs/750Picacho_Kitchen_UltraQuality_master16.tif
sweep_runs/depth_gamma_delta1/outputs/750Picacho_Kitchen_UltraQuality_master16.tif
sweep_runs/depth_gamma_delta2/outputs/750Picacho_Kitchen_UltraQuality_master16.tif
```

**Viewing Setup:**
- Same zoom level (100% or fit-to-screen)
- Same scene (Kitchen OR Great Room)
- Color-accurate display (sRGB minimum)
- NO toggling between files (simultaneous viewing)

---

### Step 2: Apply Decision Criteria

**For each visual cue from table above:**

1. Identify which delta matches "Optimal" column
2. Check for artifacts in "Too Low" or "Too High" columns
3. Note if Kitchen and Great Room agree

**Example Assessment (depth.gamma):**

```
Kitchen Scene:
- Foreground Separation: Delta 1 (gamma=1.1) - Clean, distinct layers ✅
- Background Detail: Delta 1 - Balanced ✅
- Depth Transitions: Delta 1 - Natural ✅
- Shadow Depth: Delta 1 - Preserved ✅

Great Room Scene:
- Foreground Separation: Delta 1 - Clean ✅
- Background Detail: Delta 1 - Balanced ✅
- Depth Transitions: Delta 1 - Natural ✅
- Shadow Depth: Delta 1 - Preserved ✅

DECISION: LOCK Delta 1 (gamma=1.1) - Both scenes agree, all criteria met
```

---

### Step 3: Check Artifact Watch List

**Depth-Specific Artifacts** (Any of these → ARCHIVE immediately):

- [ ] **Halos**: Glowing edges around objects (depth edge bleeding)
- [ ] **Posterization**: Visible "stair-stepping" on smooth surfaces
- [ ] **Depth Inversion**: Far objects appear closer than near objects
- [ ] **Noise**: Grainy texture in shadow areas (depth map noise)
- [ ] **Over-smoothing**: Loss of architectural edges (blurred lines)
- [ ] **Crushed Blacks**: Shadow areas dead black, no depth information

**If ANY artifact is present in ALL deltas:**
- ARCHIVE all deltas
- Document artifact type
- Consider baseline has upstream issue (depth map quality)

---

### Step 4: Make Final Decision

**LOCK** - Clear winner, no artifacts, scenes agree:
- Record locked value in `params.json`
- Update `PHASE1_VISUAL_VERIFICATION_CHECKLIST.md`
- Proceed to next parameter sweep

**HOLD** - Scene-dependent or marginal differences:
- Expand sweep to 5 deltas (finer granularity)
- Test across all 6 scenes (not just Kitchen/Great Room)
- Consider combined sweep in Phase 2 (interaction with other parameters)

**ARCHIVE ALL** - Artifacts present or no improvement over baseline:
- Lock to baseline value (e.g., `depth.gamma=1.0`)
- Document rationale
- Investigate if baseline depth map has quality issues

---

## Scene-Specific Considerations

### Kitchen Scene (Best for evaluating)

**Why Kitchen is critical:**
- High edge density (cabinets, backsplash, counters)
- Shadow areas (under cabinets, corners)
- Mixed surface types (wood, stone, metal)
- Architectural lines (horizontal and vertical)

**Focus Areas:**
- Cabinet edge sharpness (`edge_filter_sigma_color`)
- Under-cabinet shadow depth (`percentile_clip_low`)
- Backsplash tile depth separation (`gamma`)
- Counter surface smoothness (`banding_suppression`)

---

### Great Room Scene (Best for evaluating)

**Why Great Room is critical:**
- Large smooth surfaces (walls, ceilings)
- Long-distance depth range (foreground to background)
- Fewer hard edges (more gradients)
- Curved surfaces (furniture, architectural features)

**Focus Areas:**
- Wall/ceiling banding (`banding_suppression`)
- Foreground/background separation (`gamma`)
- Large surface smoothness (`edge_filter_sigma_color`)
- Long-distance depth gradation (`percentile_clip_low`)

---

### Other Scenes (Secondary validation)

**Aerial**: High-altitude depth range (not representative for interiors)
**Pool**: Outdoor lighting, water reflections (edge case)
**Primary Bedroom**: Low edge density (less critical)
**Primary Bathroom**: Similar to Kitchen (good validation)

**Recommendation**: Use Kitchen + Great Room as primary decision scenes. If they disagree, test on Primary Bathroom as tiebreaker.

---

## Depth vs Color/Tone Interaction

**IMPORTANT**: After locking depth parameters, **re-verify color/tone quality** does not degrade.

**Potential Interaction Risks:**

1. **High `local_contrast_gain` + Low `edge_filter_sigma_color`**:
   - Risk: Compounding edge enhancement → halos
   - Check: Cabinet edges in Kitchen scene

2. **Low `saturation_protection` + High `depth.gamma`**:
   - Risk: Over-separated depth may amplify color saturation artifacts
   - Check: Wood warmth in Kitchen, Great Room color separation

**Verification Protocol:**
- After locking all 4 depth parameters
- Re-open locked color/tone deltas
- Compare to new outputs with locked depth parameters
- If color/tone quality degrades → HOLD depth parameters for Phase 2 interaction testing

---

## Metrics to Collect (Automated)

**Depth-Specific Metrics** (add to `metrics.json`):

```json
{
  "depth_metrics": {
    "dynamic_range": 0.85,           // Depth map min-max range (0-1)
    "edge_gradient_strength": 12.3,  // Average gradient at edges (higher = sharper)
    "banding_score": 0.15,           // Histogram entropy (lower = more banding)
    "shadow_depth_retention": 0.72   // Percentage of shadow area with depth info
  }
}
```

**Purpose**: Supplement visual verification with quantitative data for tie-breaking.

**Do NOT rely solely on metrics.** Human perception (halos, posterization) is ground truth.

---

## Time Allocation per Depth Parameter

| Parameter | Evaluation Time | Why |
|-----------|-----------------|-----|
| `depth.gamma` | 10 minutes | High impact, easy to see (foreground/background) |
| `depth.percentile_clip_low` | 8 minutes | Moderate impact, subtle (shadow areas only) |
| `depth.edge_filter_sigma_color` | 12 minutes | Critical, requires edge inspection at 100% zoom |
| `depth.banding_suppression` | 7 minutes | Easy to spot (smooth walls), quick decision |

**Total Depth Visual Verification**: ~40 minutes for all 4 parameters

---

## Example Decision Documentation

```markdown
## Depth Parameter Decisions

### depth.gamma
- **Locked Value**: 1.1 (delta 1)
- **Rationale**: Clean foreground/background separation in both Kitchen and Great Room, no posterization
- **Artifacts**: None detected
- **Scene Agreement**: Kitchen ✅, Great Room ✅
- **Confidence**: HIGH

### depth.percentile_clip_low
- **Locked Value**: 0.5 (baseline, delta 0)
- **Rationale**: Shadow areas clean, no improvement from other deltas
- **Artifacts**: None detected
- **Scene Agreement**: Kitchen ✅, Great Room ✅
- **Confidence**: MEDIUM (marginal differences)

### depth.edge_filter_sigma_color
- **Decision**: HOLD
- **Rationale**: Kitchen prefers 50 (crisp), Great Room prefers 75 (smooth) - scene-dependent
- **Artifacts**: Minor halos at sigma=50 in Great Room
- **Next Action**: Expand to 5 deltas (60, 65, 70, 75, 80) OR defer to Phase 2 combined sweep
- **Confidence**: LOW (requires more data)

### depth.banding_suppression
- **Decision**: ARCHIVE ALL
- **Rationale**: All deltas show banding on Great Room walls - baseline depth map issue
- **Artifacts**: Posterization visible at all tested values (0.003, 0.005, 0.007)
- **Next Action**: Investigate depth map pre-processing, accept baseline for now
- **Confidence**: N/A (baseline issue)
```

---

## Integration with Color/Tone Results

**After both Color/Tone and Depth sweeps complete:**

**Locked Parameters (Example):**
```json
{
  "color_tone": {
    "saturation_protection": 0.85,
    "local_contrast_gain": 2.2
  },
  "depth": {
    "depth_gamma": 1.1,
    "depth_percentile_clip_low": 0.5,
    "edge_filter_sigma_color": 75,
    "banding_suppression": 0.005
  }
}
```

**Next Phase**: Materials V3 sweeps (3 parameters) → Final Phase 1 production preset

---

**Document Status**: ✅ COMPLETE
**Usage**: Reference during depth parameter visual verification
**Next Review**: After first depth sweep completes (depth.gamma recommended)
