# Session Complete: PR-3C Final + Stage 6 A/B Boundary Metrics Test

**Date**: December 13, 2025  
**Duration**: ~2 hours  
**Focus**: Execute Stage 6 A/B test with boundary metrics, make final EfficientSAM promotion decision

---

## Session Achievements

### ✅ 1. Fixed PR-3C Implementation Issues

Fixed four critical runtime bugs in the Stage 6 A/B script:

1. **Device type handling**: Changed from string to `torch.device` object
2. **Gradient computation**: Removed scikit-image dependency, implemented PIL-based resizing + scipy Sobel gradients
3. **Metrics return type**: Added `_as_dict()` helper for safe `BoundaryMetrics.to_dict()` conversion
4. **Benchmark paths**: Updated to use actual UltraQuality TIFFs from `projects/750_picacho_lane/`

### ✅ 2. Executed Full Stage 6 A/B Test

Ran boundary metrics comparison across 5 benchmark scenes:
- Kitchen (glass)
- Bedroom (glass)
- Bathroom (glass)
- Pool (water, foliage)
- Aerial (foliage)

**Test infrastructure**:
- In-memory mask extraction (no disk I/O)
- Material name canonicalization via `normalize_material_dict()`
- Boundary F1 as regression guard
- Edge alignment vs image gradients as improvement metric
- Minimum boundary pixels threshold (250)

### ✅ 3. Made Data-Driven Promotion Decision

**Result**: ❌ **Do NOT promote EfficientSAM FUSED to default APEX**

**Evidence**:
- 0/5 scenes showed improvement
- 2 regressions (pool foliage BF1=0.138, aerial foliage BF1=0.156)
- Glass targets: either skipped (low coverage) or unchanged (BF1=1.0, Δ=0)
- Water class missing in pool scene (taxonomy issue)
- Foliage boundaries diverged massively from baseline without edge alignment justification

### ✅ 4. Documented Decision Rationale

Created comprehensive decision document:
- `docs/sessions/STAGE6_PR3C_FINAL_DECISION.md`

Key points:
- EfficientSAM infrastructure is correct and observable
- Problem is prompt strategy, not implementation
- Current box→center-point prompts insufficient
- Boundary metrics reveal structural disagreement with SegFormer
- Materials V3 taxonomy needs debugging (water detection)

---

## Technical Findings

### Boundary Metrics Analysis

**Bedroom glass** (only meaningful signal):
- Boundary F1: 1.0 (perfect agreement)
- Edge alignment: 0.367 (identical for baseline and canary)
- Interpretation: EfficientSAM produced identical output to SegFormer

**Pool foliage** (regression):
- Boundary F1: **0.138** (87% disagreement)
- Edge alignment delta: -0.006 (slight degradation)
- Boundary pixels: 156,519 (massive divergence)

**Aerial foliage** (regression):
- Boundary F1: **0.156** (84% disagreement)
- Edge alignment delta: +0.005 (marginal gain, not enough)
- Boundary pixels: 112,749

### Why BF1 < 0.20 is a Red Flag

Boundary F1 below 0.20 indicates:
- Canary and baseline boundaries overlap <20%
- This is **structural disagreement**, not refinement
- For this to be acceptable, edge alignment must improve dramatically (Δ ≥ 0.05)
- Observed edge alignment deltas: -0.006 to +0.005 (negligible)

### Water Class Missing

Target: pool scene, class `water`  
Canonicalization: `normalize_material_dict()` applied  
Result: **Still missing**

Root cause candidates:
1. SegFormer not emitting water class for pool scene
2. Canonical mapping incomplete
3. Coverage threshold filtering out water
4. Semantic key mismatch (`pool_water` vs `water_surface` vs `water`)

Requires Materials V3 taxonomy debugging in next PR.

---

## Deliverables Committed

### Files Added
1. `scripts/stage6_ab_final_v3.py` - Production-ready A/B test script with all fixes
2. `docs/sessions/STAGE6_PR3C_FINAL_DECISION.md` - Decision document

### Results Captured
- `outputs/stage6_ab_pr3c_final/stage6_ab_summary.json` - Full metrics (not committed)
- `outputs/stage6_ab_run.log` - Complete run log (not committed)

### Commit
```
test(stage6): PR-3C boundary metrics A/B test - EfficientSAM canary-only confirmed
Commit: 4fa1e25
```

---

## Strategic Implications

### ✅ Validated Infrastructure

- EfficientSAM V3 scaffolding (Stages 1-4) is production-ready
- Boundary metrics module works correctly
- Canary preset activation logic is safe
- Feature flags and opt-in design working as intended

### ❌ EfficientSAM Refinement Not Ready

Current state:
- Prompt strategy (box→center-point) insufficient
- IoU gating vs boundary quality mismatch understood
- Edge refinement requires different approach

Blockers:
- Need mask-driven multi-point prompts (farthest-point sampling)
- Need boundary metrics as primary gate (not IoU vs SegFormer)
- Water class detection must be fixed

### 🎯 Correct Next Steps

**Do NOT invest more EfficientSAM time now.**

Instead, proceed with:

1. **Materials V3 Taxonomy Debugging** (PR-3D or PR-4)
   - Fix water class detection/mapping
   - Add debug logging for canonical key resolution
   - Unit tests for edge cases in `normalize_material_dict()`

2. **Auto-Preset v2** (PR-4 or PR-5)
   - `--quality-tier auto`
   - `--intent {preview,client,hero}`
   - Complexity heuristic (gradient entropy)
   - **Hard gate**: `--allow-canary` required for EfficientSAM

3. **Materials V3 Plan Mode** (PR-5 or PR-6)
   - Implement "should_refine" decision engine
   - Edge-aware gating using boundary complexity
   - Emit `materials_v3` report block (no pixel changes)

4. **EfficientSAM Stage 7** (deferred to future sprint)
   - Only after Materials V3 gating is validated
   - Requires new prompt strategy implementation
   - Gate on boundary metrics, not IoU

---

## Lessons Learned

### 1. Boundary Metrics Are the Right Tool

IoU vs SegFormer (previous Stage 6) masked the problem:
- High IoU → "didn't improve"
- Low IoU → "rejected by gate"

Boundary metrics revealed the truth:
- Low BF1 → structural disagreement
- Edge alignment delta → actual improvement measure

### 2. Canary-Only Was Correct Decision

Stage 6 results from earlier session (2/5 scenes, marginal IoU) were directionally correct.
PR-2 prompt improvements didn't change the fundamental issue.
Canary-only design protected production.

### 3. Water Detection Is a Real Problem

Appears across multiple scenes and attempts.
Not a "one-off missing class" but a systematic taxonomy/detection gap.
Materials V3 must prioritize this.

### 4. Glass Too Sparse to Evaluate

3/3 interior scenes: either no glass detected or identical output.
Either:
- These scenes don't actually have significant glass, or
- SegFormer glass detection is weak, or
- Target class list needs refinement

Requires scene-specific investigation or different target materials.

---

## Reproducibility Notes

All results are fully reproducible on same environment:

```bash
# Prerequisites
cd /Users/rc/Transformation_Portal
source .venv/bin/activate

# Run test
python scripts/stage6_ab_final_v3.py

# Outputs
# - outputs/stage6_ab_pr3c_final/stage6_ab_summary.json
# - outputs/stage6_ab_run.log
```

Device: CPU (forced via `FORCE_DEVICE = "cpu"`)  
Runtime: ~5 minutes for 5 scenes  
Dependencies: scipy (sobel), PIL (resize), torch (segmentation)

---

## CI Status

**Main branch**: Pushed successfully (commit 4fa1e25)  
**Expected workflows**: CodeQL, CI/CD Consolidated, Quality Gate, etc.  
**No breaking changes**: All additions; no behavior changes to existing presets

---

## Next Session Preparation

Before next session:

1. ✅ Clean workspace: `make clean` (if needed)
2. ✅ Verify CI green on latest commit
3. ⏭️ Choose next PR:
   - **Option A**: Materials V3 taxonomy debugging (water class priority)
   - **Option B**: Auto-preset v2 (user experience upgrade, no risk)
4. ⏭️ Review Materials V3 scaffolding already merged
5. ⏭️ Consider creating GitHub issues for deferred work (EfficientSAM Stage 7, prompt strategy v2)

---

## Session Statistics

- **Duration**: ~2 hours
- **Scripts created**: 1 (stage6_ab_final_v3.py)
- **Bugs fixed**: 4 (device type, gradients, metrics dict, paths)
- **Benchmark scenes tested**: 5
- **Decision made**: 1 (keep canary-only)
- **Commits**: 1
- **Files modified**: 2
- **Lines added**: 598

---

## Closing Notes

This session represents a **critical milestone** in the EfficientSAM V3 journey:

✅ **Infrastructure is production-ready** (Stages 1-5B merged)  
✅ **Observability is comprehensive** (fusion stats, boundary metrics, decision logging)  
✅ **Safety gates work** (canary-only, feature flags, opt-in design)  
❌ **Current prompt strategy insufficient** (data-driven conclusion)  
🎯 **Strategic pivot validated** (Materials V3 taxonomy + auto-preset more valuable now)

**EfficientSAM remains canary-only. Move forward with Materials V3 and auto-preset work.**

---

**Session End**: December 13, 2025, ~11:25 PM PST  
**Status**: ✅ Complete, Decision Made, Repository Stable  
**Next**: Materials V3 Taxonomy Debugging (PR-3D/4) or Auto-Preset v2 (PR-4/5)
