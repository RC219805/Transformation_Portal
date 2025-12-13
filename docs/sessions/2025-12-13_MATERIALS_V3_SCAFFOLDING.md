# Session Complete: Materials V3 Scaffolding + Roadmap
**Date**: December 13, 2025  
**Session Focus**: Materials V3 Foundation (PR-1) + Implementation Roadmap

---

## Executive Summary

Successfully implemented **Materials V3 scaffolding** as a safe, disabled-by-default foundation following the proven EfficientSAM V3 pattern. Based on Stage 6 A/B results showing EfficientSAM fusion had marginal wins (2/5 scenes, IoU barely above threshold), **Materials V3 is designed to make refinement smarter, not just more frequent**.

### Key Decision: Keep EfficientSAM FUSED as Canary-Only

Stage 6 A/B testing validated the conservative approach:
- Fusion only applied in **2/5 benchmark scenes** (Bedroom glass + Aerial foliage)
- IoU values barely above gate threshold (0.431, 0.383)
- Visual diffs showed **no meaningful edge improvement** in the two "wins"
- Bathroom scene hit **OOM** (safety guards now in place)
- Kitchen/Pool showed severe divergence (IoU 0.089-0.297)

**Conclusion:** Do NOT promote FUSED to default APEX; keep as explicit canary preset.

---

## Materials V3 Architecture

### Design Philosophy

Materials V3 addresses the root causes of Stage 6 limitations:

1. **Better Targeting**: Not "refine everything," but "refine what matters, where it matters"
2. **Real Confidence**: Not placeholder masks, but genuine probability + quality scores
3. **Smarter Prompts**: From mask peaks (distance transform), not box centers
4. **Edge Awareness**: Different gating/response for core vs boundary
5. **Scene Awareness**: Optional lighting integration (when validated)

### Core Components (Scaffolded)

#### 1. Material Taxonomy

```python
class MaterialTaxonomy(str, Enum):
    BASE = "base"  # SegFormer buckets (8-12 classes)
    EXPANDED = "expanded"  # Semantic + material layers (18-24)
    FULL = "full"  # Future: full PBR taxonomy (40+)
```

**Expanded taxonomy** adds:
- **Semantic layer**: SegFormer ADE → buckets (sky, building, window, water, etc.)
- **Material layer**: What matters for response (wood_grain vs wood_smooth, glass_clear vs glass_frosted, water_surface vs water_volume, etc.)

#### 2. Refinement Strategy

```python
class RefinementStrategy(str, Enum):
    OFF = "off"  # No EfficientSAM
    CANARY = "canary"  # Glass/water/foliage only (Stage 6 validated)
    SELECTIVE = "selective"  # Auto-select based on confidence
    AGGRESSIVE = "aggressive"  # All materials (dev only)
```

**Default: OFF** (no behavior change until explicitly enabled).

#### 3. Real Confidence Semantics

```python
@dataclass
class ConfidenceSemantics:
    base_threshold: float = 0.50  # SegFormer
    refined_threshold: float = 0.45  # EfficientSAM (can be slightly lower)
    edge_threshold: float = 0.30  # Boundary-specific
    
    material_thresholds: Dict[str, float] = {
        'glass': 0.40,  # Inherently low confidence
        'water': 0.35,  # Highly variable
        'wood': 0.65,   # High confidence required
        ...
    }
    
    use_edge_confidence: bool = True
    edge_band_width: float = 0.20
```

**Key Improvement**: Distinguishes base/refined/edge/final confidence, not just "mask as confidence."

#### 4. Prompt Generation (from Mask Peaks)

```python
@dataclass
class PromptGenerationConfig:
    strategy: str = "mask_peaks"  # NOT box centers
    
    num_fg_points: int = 4
    fg_confidence_percentile: float = 80.0  # Top 20% of mask
    fg_spacing_min_px: int = 32  # Farthest-point sampling
    
    num_bg_points: int = 2  # Sparse negatives
    bg_margin_px: int = 16
    
    use_roi_crop: bool = True  # Crop-first refinement
    roi_padding_px: int = 32
    roi_max_side: int = 1024
```

**Key Improvement**: Sample from high-confidence regions with spatial distribution, not just box center.

#### 5. Edge-Aware Gating

```python
@dataclass
class EdgeAwareGating:
    core_threshold: float = 0.70
    edge_low: float = 0.20
    edge_high: float = 0.70
    
    core_strength: float = 1.0
    edge_strength: float = 0.8  # Conservative at edges
    
    edge_method: str = "confidence_gradient"
```

**Key Improvement**: Different response strength for core vs boundary regions.

#### 6. Safety Guards (from Stage 6 Learning)

```python
max_megapixels: float = 30.0  # OOM prevention (Bathroom fix)
max_dimension: int = 6000
```

---

## Implementation Status

### ✅ Completed (This Session)

1. **Materials V3 Module** (`lux_depth_v2/materials_v3.py`)
   - Complete config dataclasses
   - MaterialsV3Engine scaffolding
   - Pass-through when disabled
   - NotImplementedError when enabled (implementation in later PRs)
   - get_v3_report() for telemetry

2. **Pipeline Integration** (`lux_depth_v2/config.py`)
   - materials_v3: Optional['MaterialsV3Config'] field
   - Lazy import (avoid circular dependency)
   - No preset changes (default: disabled)

3. **Test Suite** (`lux_depth_v2/tests/test_materials_v3.py`)
   - 15 unit tests, all passing
   - Config validation
   - Enum values
   - Default behavior
   - Pass-through logic
   - Report structure

4. **CI/CD**
   - All workflows green
   - CodeQL scanning in progress
   - No regressions

---

## Next Steps: Implementation Roadmap

### PR-2: Prompt Strategy + ROI Refinement Provider (8-12h)

**Goal**: Fix the "low IoU" problem by improving prompts.

**Implementation**:
1. Add `_generate_prompts_from_mask()` in Materials V3:
   - Distance transform to find mask peaks
   - Farthest-point sampling for spatial distribution
   - Sparse negative points near boundary
2. Extend `EfficientSAMRefinementProvider`:
   - ROI cropping (pad around bbox, max 1024px side)
   - Use Materials V3 prompt generator
   - Fallback to box prompts on failure
3. Unit tests:
   - Synthetic masks → expected prompt positions
   - ROI cropping behavior
   - Fallback logic

**Acceptance Criteria**:
- Prompt generation deterministic + tested
- ROI refinement reduces runtime vs full-image
- No preset changes (still disabled by default)

---

### PR-3: Expanded Taxonomy + Gating (10-14h)

**Goal**: Implement semantic→material mapping and edge-aware gating.

**Implementation**:
1. Expand `ExpandedTaxonomyConfig`:
   - Implement semantic→material heuristic mapping
   - Add confidence fusion rules (SegFormer semantic + material-specific)
2. Implement `EdgeAwareGating`:
   - Compute core/edge bands from confidence gradient
   - Apply different response strength per region
3. Wire into Materials V3 Engine:
   - `process()` applies gating before final mask assembly
   - Emit per-material stats (core coverage, edge coverage, gating applied)
4. Unit tests:
   - Taxonomy mapping correctness
   - Core/edge band computation
   - Response strength application

**Acceptance Criteria**:
- Expanded taxonomy works on synthetic scenes
- Edge gating measurably reduces artifacts in test images
- Report includes per-material gating stats

---

### PR-4: Auto-Preset V2 (6-8h)

**Goal**: Improve auto-preset with quality-tier=auto + intent + complexity.

**Implementation**:
1. Extend CLI:
   - `--quality-tier auto` (in addition to standard/max/apex)
   - `--intent {preview,client,hero}`
   - Map: preview→STANDARD, client→MAX, hero→APEX
2. Add complexity heuristic:
   - Downscale image → compute Sobel gradient entropy
   - High complexity → suggest higher tier
3. Add lighting signal (basic):
   - Simple brightness + warmth analysis
   - Defer full lighting detector integration until validated
4. PresetSelector v2:
   - Incorporate complexity + lighting into decision
   - Never auto-select canary presets (explicit --allow-canary flag required)

**Acceptance Criteria**:
- `--quality-tier auto` selects sensible tier based on complexity
- Intent mapping works (preview/client/hero)
- Canary presets never selected without explicit flag

---

### PR-5: Benchmark Harness + Decision Report (4-6h)

**Goal**: Formalize Materials V3 A/B testing infrastructure.

**Implementation**:
1. Create `scripts/materials_v3_ab_benchmark.py`:
   - Baseline (Materials V2 or V3 disabled)
   - V3 Canary (refine_edges=CANARY)
   - V3 Selective (refine_edges=SELECTIVE)
2. Collect:
   - Runtime deltas
   - Refinement application rate (per-class)
   - Edge quality metrics (boundary F-score, trimap IoU)
   - Visual diff crops (auto-selected high-change regions)
3. Generate decision report:
   - Markdown summary
   - JSON structured data
   - PNG crops for visual inspection

**Acceptance Criteria**:
- A/B script runs without manual intervention
- Report includes objective edge metrics (not just visual inspection)
- Clear go/no-go decision criteria

---

## Repository State

### Files Modified
- `lux_depth_v2/config.py` (added materials_v3 field)

### Files Created
- `lux_depth_v2/materials_v3.py` (Materials V3 engine scaffolding)
- `lux_depth_v2/tests/test_materials_v3.py` (15 unit tests)

### Untracked (Experimental)
- `assets/phase2_bench/` (benchmark images)
- `scripts/run_phase2_bench_matrix.sh`
- `lux_depth_v2/tests/test_stage4_end_to_end.py`
- `lux_depth_v2/tests/test_tiled_upscaling.py`
- `tests/integration/test_phase2_end_to_end.py`

---

## Git State

### Committed to Main
- ✅ Materials V3 scaffolding (commit `34866e3`)

### Current Branch
- `main` (stable, CI green)

### Remote Status
- ✅ Pushed to `origin/main`
- ⏳ CodeQL scanning in progress

---

## Key Learnings

### From Stage 6 A/B Testing

1. **"More EfficientSAM" ≠ Better Output**
   - Fusion applied in only 2/5 scenes
   - IoU barely above threshold (marginal)
   - Visual diffs showed no meaningful improvement

2. **Prompt Strategy Matters More Than Model**
   - Box→center prompts produced low IoU (Kitchen 0.297, Pool 0.089)
   - Need mask-aware sampling (peaks, spatial distribution)

3. **OOM Risk is Real**
   - Bathroom scene (high MP) crashed EfficientSAM
   - Safety guards (max_megapixels=30.0) are mandatory

4. **IoU Gating vs SegFormer is Conservative**
   - Using SegFormer as "ground truth" rejects potentially-better EfficientSAM masks
   - Need edge-alignment metrics, not just pixel IoU

### Design Principles for Materials V3

1. **Disabled by Default** (proven safe merge pattern)
2. **Incremental Implementation** (PR-1 → PR-5 roadmap)
3. **Objective Metrics** (not just visual inspection)
4. **Safety First** (OOM guards, fallbacks, logging)
5. **No Silent Activation** (never auto-select canary features)

---

## Performance Metrics (Materials V3 Target)

### Refinement Efficiency Targets

- **Prompt Generation**: < 50ms per class (distance transform + sampling)
- **ROI Cropping**: Reduce EfficientSAM input from HxW → ~1024² max
- **Selective Strategy**: Refine only 2-4 classes per scene (not all)
- **Memory**: Respect 30 MP hard limit (OOM prevention)

### Quality Improvement Targets

- **Edge IoU**: ≥ 0.50 (vs 0.30-0.43 in Stage 6)
- **Boundary F-score**: ≥ 0.70 (new metric)
- **Refinement Application Rate**: 60-80% when selective (vs 40% in Stage 6)

---

## Decision Criteria for Materials V3 Promotion

**Do NOT promote to default APEX** until all criteria met:

### Hard Requirements
1. ✅ PR-1 through PR-5 implemented and tested
2. ✅ Benchmark A/B shows ≥ 3/5 scenes with meaningful edge improvement
3. ✅ Edge metrics (boundary F-score, trimap IoU) measurably better
4. ✅ No OOM regressions (all benchmark scenes complete safely)
5. ✅ Runtime delta acceptable for APEX tier (hero frames only)

### Soft Requirements (Recommended)
- Visual diff crops show cleaner edges (no halos, spill, artifacts)
- Refinement application rate ≥ 60% when selective
- Lighting detector validated and integrated (optional, can defer)

**If only 2-3 hard requirements met**: Keep V3 as canary-only (like EfficientSAM FUSED).

---

## Outstanding Work

### Immediate (Next Session)

1. **PR-2: Implement Prompt Strategy**
   - `_generate_prompts_from_mask()` with distance transform
   - ROI refinement provider
   - Unit tests

2. **Verify CI Status**
   - CodeQL scanning completion
   - All workflows green

### Short-Term (This Week)

1. **PR-3: Expanded Taxonomy + Gating**
2. **PR-4: Auto-Preset V2**
3. **PR-5: Benchmark Harness**

### Medium-Term (Next Sprint)

1. **Materials V3 A/B Validation**
   - Run full benchmark suite
   - Generate decision report
   - Make promotion decision (canary vs default APEX)

2. **Lighting Detector Validation**
   - Formal accuracy benchmarking
   - Optional integration into Materials V3

3. **Video Processing (Phase 3)**
   - Extend APEX to video inputs
   - Temporal consistency for materials/lighting

---

## Session Statistics

- **Duration**: ~2.5 hours
- **Commits**: 1 (Materials V3 scaffolding)
- **Files Created**: 2 (module + tests)
- **Files Modified**: 1 (config)
- **Tests Added**: 15 (all passing)
- **CI Status**: Green (CodeQL in progress)

---

## Recommendations for Next Session

### Start With

1. Verify CI is fully green (including CodeQL)
2. Review Stage 6 A/B summary (confirm canary-only decision)
3. Pull latest from origin (if working from another machine)

### Focus Areas

1. Implement PR-2 (prompt strategy + ROI refinement)
2. Unit test prompt generation determinism
3. Benchmark prompt strategy improvement (IoU comparison)

### Avoid

1. Enabling Materials V3 in any preset before PR-2 through PR-5 complete
2. Auto-selecting canary features (explicit opt-in only)
3. Changing default APEX behavior until full validation

---

## Closing Notes

This session establishes **Materials V3 as the right path forward** for advanced material understanding, but validates the conservative Stage 6 decision:

- ✅ EfficientSAM FUSED **stays canary-only** (marginal wins, OOM risk)
- ✅ Materials V3 **designed to fix root causes** (smarter targeting, better prompts, edge awareness)
- ✅ Safe scaffolding merge pattern proven again (no regressions, disabled by default)
- ✅ Clear roadmap (PR-1 → PR-5) with objective success criteria

**Ready for PR-2 implementation** when you return.

---

**Session End**: December 13, 2025, 12:59 PM PST  
**Status**: ✅ Complete, Repository Stable, All Tests Passing, CI Green
