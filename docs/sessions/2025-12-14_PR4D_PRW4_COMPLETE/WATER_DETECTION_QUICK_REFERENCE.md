# Water Detection: Quick Reference Card

**Purpose**: One-page reference for water detection advancement

---

## Current State (One Sentence)

Infrastructure complete, detector is stub, validation ready but primary metric blocked.

---

## Recommended Path (One Paragraph)

**Data-First Hybrid**: Create labeled dataset (Week 1), analyze stub failures, implement simplified heuristic detector (Week 2), validate and tune thresholds (Week 3), production deploy with monitoring (Week 4+). Total: 3-4 weeks to defensible, production-validated water detection.

---

## This Week Actions (Prioritized)

1. **Fix edge alignment metric** (2 hours, BLOCKING):
   ```python
   # Add to MaterialsV3Config
   water_validation_emit_mask: bool = False  # Debug mode for validation
   
   # Expose mask in report when enabled
   # Update validation harness to decode and use mask
   ```

2. **Start dataset collection** (1 week, CRITICAL):
   - 20-30 pool images (residential, resort, lap pools)
   - 20-30 ocean images (calm, waves, horizon)
   - 10-20 non-water (blue sky, glass, foliage)
   - Create `ground_truth.json` with scene labels

3. **Update docs** (1 hour, TRANSPARENCY):
   - Add "Known Limitations" to PR-W4
   - Mark PR-W1 as "stub only"
   - Document unblocking path

---

## Decision Matrix

| If... | Then Choose... | Timeline |
|-------|---------------|----------|
| Quality critical, have time | Data-First Hybrid | 3-4 weeks |
| Need production NOW | Fast Track (improve stub) | 1 day |
| Have labeled dataset | ML Detector | 2 weeks |
| Want best long-term | Data-First → ML | 4-5 weeks |

---

## Quality Targets

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Detection Rate (Pool) | ≥85% | % pool scenes with water detected |
| False Positive Rate | ≤5% | % non-water scenes with false detection |
| Edge Alignment | ≥0.6 | Boundary-gradient overlap score |
| Stability | ≥0.8 | Coverage variance < 0.04 |
| Processing Time | ≤50ms | p95 overhead per image |

---

## Stub vs Spec Comparison

| Feature | Stub (Current) | PR-W1 Spec | Gap |
|---------|---------------|-----------|-----|
| Chromaticity | Blue > Red/Green | HSV/Lab, pool vs ocean | ❌ Missing scene tuning |
| Specular | None | Highlights + low sat | ❌ Missing |
| Texture | None | Entropy/frequency | ❌ Missing |
| Planarity | None | Depth gradient | ❌ Missing |
| Combination | Boolean AND | Weighted | ❌ Missing weights |
| Post-processing | None | Morphology, holes | ❌ Missing |
| Component Filter | None | Top-K, min area | ❌ Missing |

**Conclusion**: Stub is ~10% of spec. Works for obvious cases, fails on complex scenes.

---

## Fast Track (If Urgent)

**Day 1** (6 hours):
```python
# Improve stub: HSV-based with scene context
hsv = rgb2hsv(rgb01)
hue = hsv[:, :, 0] * 360
sat = hsv[:, :, 1]
val = hsv[:, :, 2]

# Pool: cyan/blue (170-210°), ocean: blue-green (160-220°)
if scene_context == POOL:
    hue_match = (hue >= 170) & (hue <= 210)
else:
    hue_match = (hue >= 160) & (hue <= 220)

mask = hue_match & (sat > 0.2) & (val > 0.2)
mask = filter_components(mask, min_area=1000, top_k=3)
```

**Day 2**: Ship in experimental preset, monitor telemetry

**Week 2-3**: Build proper detector based on production data

---

## Data-First Hybrid (Recommended)

**Week 1**: Dataset creation
- Collect 50-100 images
- Label scene type, coverage
- Document characteristics

**Week 2**: Baseline + Detector
- Fix edge alignment (2h)
- Run baseline validation (1d)
- Implement simplified detector (2-3d):
  - Chromaticity cue (HSV, pool vs ocean)
  - Component filtering
  - Skip texture/planarity (not critical)

**Week 3**: Validation
- Run full validation
- Tune thresholds (detection ≥85%, FP ≤5%, edge ≥0.6)
- Document results

**Week 4+**: Production
- Canary deployment
- Monitor telemetry
- Gradual rollout

---

## Key Files

| Purpose | File | Status |
|---------|------|--------|
| Spec | `docs/PR_WATER_MASK_STRUCTURE.md` | Complete |
| Detector | `lux_depth_v2/water_candidate.py` | **Stub** |
| Integration | `lux_depth_v2/materials_v3.py` | Complete |
| Validation | `scripts/prw_water_validation.py` | Blocked (edge metric) |
| Tests | `tests/test_prw_water_validation.py` | Passing |

---

## Metrics Status

| Metric | Working? | Value | Issue |
|--------|----------|-------|-------|
| Coverage | ✅ Yes | Variable | - |
| Confidence | ✅ Yes | Variable | - |
| False Positives | ✅ Yes | Unknown | Need dataset |
| Performance | ✅ Yes | ~10ms | - |
| Stability | ✅ Yes | Unknown | Need dataset |
| Edge Alignment | ❌ **Blocked** | 0.0 | Mask not exposed |
| Boundary Pixels | ❌ **Blocked** | 0 | Mask not exposed |

**Fix Required**: Expose mask via debug flag (2 hours)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Heuristic insufficient | Data-First validates before production, can pivot to ML |
| Dataset not representative | Collect diverse scenes, iterate based on production |
| False positives in production | Conservative thresholds, canary deployment, instant rollback |
| Edge refinement overhead | Tune thresholds to trigger only when needed |

**Rollback**: `water_detection_enabled = False` (instant disable)

---

## Success Definition

**Meaningful advancement means**:
- ✅ Validated quality (data-proven, not guesswork)
- ✅ Production deployment (real workflows, measurable impact)
- ✅ Sustainable (can iterate over time)
- ✅ Defensible (quantified metrics, documented rationale)

**NOT meaningful**:
- ❌ Implementing spec without validation
- ❌ Shipping stub without knowing failure modes
- ❌ Perfect detector for synthetic tests
- ❌ Guessing thresholds and hoping

---

## Contact/Resources

- **Full Analysis**: `docs/WATER_DETECTION_STRATEGIC_ASSESSMENT.md`
- **Executive Summary**: `docs/WATER_DETECTION_EXECUTIVE_SUMMARY.md`
- **Spec**: `docs/PR_WATER_MASK_STRUCTURE.md`
- **Honest Status**: `PR_W4_HONEST_STATUS.md`

---

## Next Steps (Choose One Path)

**Path A: Data-First (Recommended)**
→ Fix edge metric (2h) → Dataset (1w) → Detector (1w) → Validation (1w) → Production (1-2w)

**Path B: Fast Track**
→ Improve stub (6h) → Ship experimental (1d) → Dataset (1w) → Proper detector (2w)

**Path C: ML-First**
→ Dataset (1w) → Train model (3d) → Integrate (1d) → Validate (1d) → Production (1-2w)

---

**Bottom Line**: Infrastructure ready. Detector stub. Dataset needed. 3-4 weeks to production-validated quality.

**Immediate Unblock**: Fix edge alignment metric (2 hours).

**Critical Path**: Dataset creation (Week 1).

**Recommended**: Data-First Hybrid (defensible, measurable, iterative).
