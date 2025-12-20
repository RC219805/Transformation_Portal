# Production Acceptance Criteria
## High-Fidelity Depth Pipeline - Go/No-Go Decision Framework

**Date:** 2025-12-17  
**Config Hash:** 4319f2d4  
**Reviewer Requirements:** Production validation on full 750_Picacho dataset

---

## Critical Quality Gates (MUST PASS)

### 1. Edge Fidelity (Primary Use Case: DOF Mattes)
- **Edge F1 Score**
  - Minimum: 0.30 (per-image)
  - Target Mean: ≥ 0.45
  - No image below 0.20
  
- **Chamfer Distance** (spatial alignment)
  - Target Mean: < 5.0 px
  - Maximum acceptable: < 15.0 px
  - Worst-case (any image): < 20.0 px

### 2. Artifact Prevention
- **Edge Count Ratio**
  - Target: 0.8 - 1.5×
  - Maximum: < 2.0× (prevents edge explosion)
  
- **Seam Energy**
  - Target Mean: < 1.1
  - Maximum (any tile boundary): < 1.2
  - Zero hard seams (energy > 2.0)

- **Halo/Overshoot Score**
  - Target: < 0.3
  - Maximum acceptable: < 0.5

### 3. Pass Rate
- **Overall**: ≥ 80% of images must pass all gates
- **Critical scenes** (Kitchen, GreatRoom, Pool): 100% must pass

---

## Scene-Specific Requirements

### Kitchen (Glass/Metal Complexity)
- Edge precision ≥ 0.35 (crisp cabinet edges)
- Chamfer < 8.0 px (tight boundaries)
- No glass-edge artifacts (halo < 0.4)

### GreatRoom (Large Planar Structure)
- Seam energy < 1.15 (wall/ceiling continuity)
- Edge recall ≥ 0.40 (preserve architectural lines)
- No low-frequency banding

### Pool (Exterior/Water)
- Edge F1 ≥ 0.30 (water boundaries)
- Overshoot < 0.45 (prevent edge halos)

### Aerial (Scale/Context Shift)
- Edge F1 ≥ 0.25 (different scale acceptable)
- Seam energy < 1.2

---

## Performance Requirements

### Runtime (Per Image, Warm Model)
- 4K (4000×6000): < 180 seconds
- 2K (2000×3000): < 60 seconds
- 8K (8000×12000): < 600 seconds

### Memory (Peak Usage)
- 4K: < 12 GB
- 8K: < 24 GB

---

## Validation Artifacts (Must Deliver)

### 1. Metrics Report (`validation_summary.json`)
```json
{
  "config_hash": "4319f2d4",
  "pass_rate": ≥ 0.80,
  "edge_f1_mean": ≥ 0.45,
  "chamfer_worst": < 20.0,
  "seam_energy_worst": < 1.2,
  "per_image": [...]
}
```

### 2. Visual Gallery
- 2×2 comparison grid for each image:
  - Top-left: RGB source
  - Top-right: Depth visualization (colormap)
  - Bottom-left: Edge overlay (RGB + depth edges in red)
  - Bottom-right: Metrics overlay
  
### 3. Failure Analysis
- For any failed images:
  - Root cause (seam/halo/alignment)
  - Visual proof (overlay)
  - Recommended fix

---

## Materials V3 Integration Gate

**Pre-Condition:** Depth pipeline must pass standalone validation

**Integration Test:**
1. Run Materials V3 with enhanced depth vs baseline
2. Measure:
   - Water mask boundary precision (Dice score improvement)
   - Material boundary error (reduction in misclassification)
   - Zone stability (reduction in flicker/jitter)

**Success Criteria:**
- Water mask Dice: +5% minimum improvement
- Material boundary error: -10% reduction
- Visual assessment: No new artifacts introduced

---

## Go/No-Go Decision Tree

### GO (Approve for Production Pilot)
✅ Pass rate ≥ 80%  
✅ All critical scenes pass  
✅ Chamfer worst-case < 20px  
✅ Seam energy worst-case < 1.2  
✅ Runtime within bounds  
✅ Visual gallery shows no severe artifacts  

**Action:** Deploy behind feature flag, monitor real-world usage

---

### CONDITIONAL GO (Limited Rollout)
⚠️ Pass rate 70-80%  
⚠️ One critical scene fails (but others pass)  
⚠️ Chamfer worst-case 20-25px  
⚠️ Minor halo artifacts (< 0.6) on <20% of images  

**Action:** Deploy to subset of images, gather feedback, iterate

---

### NO-GO (Block Production)
❌ Pass rate < 70%  
❌ Multiple critical scenes fail  
❌ Seam artifacts (energy > 1.5) present  
❌ Edge explosion (ratio > 2.5×) on any image  
❌ Chamfer > 30px on any critical scene  

**Action:** Return to development, fix root cause, re-validate

---

## Current Status (In Progress)

**Validation Run:** `outputs/full_validation_prod/`  
**Started:** 2025-12-17 20:39:54  
**Images:** 6 (Aerial, GreatRoom, Kitchen, Pool, Primary Bath, Primary Bedroom)  
**Config:**
- Tile size: 1024×1024
- Overlap: 128px
- Global anchor: OFF (per review recommendation)
- Edge snap: ON (strength 0.2, AND-gated)
- CLAHE: OFF (geometry preservation)

**To Be Updated:** Results pending completion

---

## Review Sign-Off Checklist

### Validation Completeness
- [ ] Full dataset processed at native resolution
- [ ] All critical scenes tested
- [ ] Metrics JSON validated (atomic write + readback)
- [ ] Visual gallery generated
- [ ] Config captured with all results

### Metric Integrity
- [ ] Float-based edge detection (not uint8)
- [ ] Shift-tolerant F1 (5px dilation)
- [ ] Precision/recall breakdown available
- [ ] Halo detection implemented
- [ ] Seam validation on tile boundaries

### Documentation
- [ ] Config hash traceable to exact parameters
- [ ] Failure modes documented
- [ ] Performance profiled
- [ ] Integration plan with Materials V3

---

## Next Steps After Validation

1. **If GO:** 
   - Update production config with validated parameters
   - Run Materials V3 integration A/B
   - Monitor first 100 production images
   - Document any edge cases

2. **If CONDITIONAL GO:**
   - Identify specific failure patterns
   - Implement targeted fixes
   - Re-validate failed scenes only
   - Decision point: GO or iterate

3. **If NO-GO:**
   - Prioritized fix list based on failure modes
   - Isolation tests for each failure type
   - Comprehensive re-validation required

---

**Acceptance Authority:** Technical review complete pending validation results  
**Signed Off By:** [Pending - requires validation data]

