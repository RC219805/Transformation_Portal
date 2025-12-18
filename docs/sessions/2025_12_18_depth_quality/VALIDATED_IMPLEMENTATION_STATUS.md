# High-Fidelity Depth: Validated Implementation Status
**Date**: 2025-12-18  
**Status**: PHASE 1 SCAFFOLDING COMPLETE - VALIDATION REQUIRED  
**Next**: Prove core claims, implement missing pieces

---

## What Was Actually Built (Honest Assessment)

### ✅ Implemented and Tested
1. **Tiled Inference Framework** (`lux_depth_v2/depth_inference.py`)
   - Tile extraction with overlap
   - Blending infrastructure (Hann/cosine windows)
   - Median fusion option (edge-preserving)
   - Scale reconciliation logic
   - **STATUS**: Code exists, architecture is sound

2. **Corrected Normal Map Generation** (`lux_depth_v2/normal_map.py`)
   - Fixed Z scale computation
   - Normalized depth input
   - Scharr/Sobel gradients
   - Tangent-space output
   - **STATUS**: ✅ VALIDATED - Math is correct, 15/15 tests passing

3. **Proper Quality Metrics** (`lux_depth_v2/quality_metrics.py`)
   - Edge alignment score (RGB-depth correlation)
   - Edge width measurement
   - Halo/ringing detection
   - Overall quality composite
   - **STATUS**: ✅ VALIDATED - Metrics are correct proxies for DOF/masking

### ⚠️ CRITICAL CLAIMS NOT YET PROVEN

#### Claim 1: "No Internal Resize" ❌ UNVALIDATED
**Documentation says**: *"Infer Each Tile at Model Native Resolution - KEY: No internal resize"*

**Reality**: We don't actually know if the HuggingFace pipeline is resizing tiles internally.

**Risk**: If the model processor resizes 1024×1024 tiles to 518×518 (common for Depth Anything),
the entire tiled approach delivers **ZERO** fidelity improvement—it's just marketing.

**Validation created**: `lux_depth_v2/tools/validate_tiled_inference.py`
- Instruments model forward pass to log actual tensor shapes
- Compares tile size vs model input size
- **MUST RUN** before claiming "high-res inference"

**Next action**: Execute validation script and update docs with **actual** tensor sizes

---

#### Claim 2: "5-10x Edge Fidelity Improvement" ❌ UNVALIDATED
**Documentation says**: *"Expected Impact: 5-10x edge fidelity improvement"*

**Reality**: No A/B comparison has been run. This is a forecast, not a measurement.

**Validation required**:
1. Run old pipeline on pool/kitchen images
2. Run new tiled pipeline on same images
3. Measure **actual** edge alignment scores
4. Compare: old vs new
5. Update docs with **measured** improvement (might be 2x, might be 8x, might be 1.1x)

**Deliverable**: `validation_report_ab_comparison.json` with concrete numbers

---

#### Claim 3: "20,000+ Unique Levels" ⚠️ MISLEADING METRIC
**Documentation says**: *"Target: 20,000+ unique levels"*

**Reality**: Unique value count is easy to game with stretching/quantization. 
The old pipeline already hit 65,536 unique values but still had smooth ramps.

**User feedback**: 
> "Hitting 65,536 unique values is easy—but it does not mean the map contains 
> 65K levels of meaningful scene depth. If the prediction is low-res and smooth, 
> those levels are mostly interpolated ramps."

**Fix**: De-emphasize unique levels in quality score. Primary metrics should be:
- Edge alignment ≥0.6 (actual usability for masking)
- Edge width ≤3px (sharp boundaries)
- Overshoot score ≥0.7 (clean, no halos)

**Status**: Metrics exist but weighting needs adjustment in `DepthQualityAnalyzer`

---

### ⚠️ MISSING IMPLEMENTATION (NOT OPTIONAL)

#### Missing 1: Global Anchor Pass ❌ NOT IMPLEMENTED IN PIPELINE
**Documentation says**: Tiled inference is "implemented"

**Reality**: Current `TiledDepthEstimator` only runs tiled passes. No global anchor.

**Risk**: Tiles lose global context → low-frequency banding, plane warps, seams

**User feedback**:
> "Tiles lose global context. Your doc lists scale reconciliation and Hann blending,
> but does not include a global anchor pass. That one addition often eliminates
> the most stubborn tiling artifacts."

**Implementation created**: `lux_depth_v2/global_anchor.py`
- Run low-res (512px) global pass for scene structure
- Run high-res tiled passes for detail
- Fuse as: `global_LF + tiled_HF` (frequency split)
- Optional edge-aware weighting

**Status**: Code exists, **not yet integrated** into `TiledDepthEstimator`

**Next action**: Add `use_global_anchor` flag to `TiledInferenceConfig`, integrate fusion

---

#### Missing 2: Edge Snapping ❌ PLANNED, NOT IMPLEMENTED
**Documentation says**: *"Planned for Phase 2"*

**Reality**: Edge snapping is **not optional** for luxury-grade mattes.

**Current state**: Tiled inference (if it works) provides better spatial structure,
but depth edges will still not perfectly align with RGB edges.

**User feedback**:
> "Given your current outputs (soft boundaries), edge snapping is not a luxury
> add-on. It's part of the minimum viable 'luxury-grade' result."

**Implementation needed**: Joint bilateral upsampling
```python
def snap_edges_to_rgb(depth, rgb, sigma_spatial=5, sigma_color=0.1):
    # Use RGB as guide to snap depth discontinuities to image edges
    return cv2.ximgproc.jointBilateralFilter(rgb, depth, ...)
```

**Status**: Not implemented, should be Phase 1 (not Phase 2)

**Next action**: Implement `lux_depth_v2/edge_snapping.py`, integrate into pipeline

---

## Contradictions That Must Be Resolved

### Contradiction 1: Edge Enhancement Policy
**Old docs** (`DEPTH_MAP_QUALITY_DIAGNOSIS_AND_FIX.md`):
> "Edge Enhancement: **skipped to preserve smoothness**"  
> "Smooth gradients are CORRECT for architectural scenes"

**New docs** (`HIGH_FIDELITY_DEPTH_SUMMARY.md`):
> "Soft boundaries are a CRITICAL failure"  
> "Target: edge width ≤3px for sharp masking"

**Resolution needed**: Update old docs to clarify:
- Smooth gradients are correct **within** surfaces (walls, floors)
- Sharp boundaries are correct **between** objects (furniture/background edges)
- The pipeline was preserving the wrong kind of smoothness

**Action**: Add errata to `DEPTH_MAP_QUALITY_DIAGNOSIS_AND_FIX.md` explaining the shift

---

### Contradiction 2: "Research-Grade" vs "Luxury-Grade"
**Old summary**: "Research-grade depth pipeline" with edge gradient 0.09 "CORRECT"

**New summary**: Edge gradient 0.09 is "MISLEADING" and "BLOCKER"

**Resolution**: The metrics changed because the **use case** clarified:
- Research-grade: Relative depth for academic study (smooth OK)
- Luxury-grade: Absolute depth for DOF/masking (sharp required)

**Action**: Rename old summary to `RESEARCH_GRADE_DEPTH_ARCHIVED.md` with deprecation notice

---

## Revised Implementation Plan

### Phase 1 (Actual) - Prove Core Claims
**Status**: IN PROGRESS

1. ✅ Normal map fix (DONE, validated)
2. ✅ Quality metrics (DONE, validated)
3. ⏳ **Tiled inference validation** (code exists, MUST RUN)
   - Execute `validate_tiled_inference.py`
   - Confirm no internal resize OR document actual resize factor
   - Update docs with measured tensor sizes
4. ⏳ **Global anchor integration** (code exists, NOT YET INTEGRATED)
   - Add to `TiledDepthEstimator.__init__`
   - Add config flag `use_global_anchor: bool = True`
5. ⏳ **A/B comparison** (REQUIRED)
   - Old pipeline vs new pipeline
   - Pool & kitchen images
   - Measure actual edge alignment improvement
   - Update docs with **measured** numbers (not forecasts)

---

### Phase 2 (Actual) - Close Quality Gaps
**Status**: NOT STARTED

6. ⏳ **Edge snapping implementation** (CRITICAL, not optional)
   - Implement joint bilateral upsampling
   - Integrate into pipeline after tiled+global fusion
7. ⏳ **Median ensemble fusion** (replaces weighted average)
   - Already coded in `depth_inference.py`
   - Just needs config flag to enable
8. ⏳ **Fix guided filter config** (stop washing out edges)
   - Reduce radius: r=10 → r=3-5
   - Reduce eps: 0.02 → 0.001-0.005

---

### Phase 3 (Validation) - Production Ready
**Status**: NOT STARTED

9. ⏳ **Benchmark performance** (actual throughput, not estimates)
   - 4K image: measure actual seconds (not "3-5s estimate")
   - Memory usage: measure actual GB
10. ⏳ **Client sample validation** (real luxury images)
11. ⏳ **CI integration** with quality gates

---

## What to Tell Users (Honest)

### What Works Now ✅
- Normal maps are **fixed** (validated with tests)
- Quality metrics are **correct** (edge alignment, width, overshoot)
- Tiled inference **architecture** is sound (code exists)

### What's Not Proven ⚠️
- Whether tiled inference **actually** preserves resolution (validation pending)
- Whether it delivers **measured** (not forecasted) quality improvement
- Whether global anchor is **needed** for your images (must test)

### What's Missing ❌
- Global anchor **not yet integrated** (code exists, needs hookup)
- Edge snapping **not implemented** (required for sharp mattes)
- A/B validation **not run** (required to claim improvement)

---

## Validation Checklist (Must Complete Before "Done")

- [ ] **Run tensor size validation** - Prove "no internal resize" OR document actual behavior
- [ ] **Integrate global anchor** - Add to TiledDepthEstimator with config flag
- [ ] **Implement edge snapping** - Joint bilateral upsampling after fusion
- [ ] **Run A/B comparison** - Old vs new on pool/kitchen, measure edge alignment
- [ ] **Update docs with measured numbers** - Replace forecasts with actual measurements
- [ ] **Resolve contradictions** - Update old docs explaining policy shift
- [ ] **De-emphasize unique levels** - Adjust quality score weighting
- [ ] **Benchmark actual performance** - Replace estimates with measurements

---

## Files Status

### Production-Ready ✅
- `lux_depth_v2/normal_map.py` - Correct math, validated
- `lux_depth_v2/quality_metrics.py` - Correct metrics, validated
- `lux_depth_v2/tests/test_high_fidelity_depth.py` - 15/15 passing

### Exists But Unvalidated ⚠️
- `lux_depth_v2/depth_inference.py` - Architecture sound, **validation pending**
- `lux_depth_v2/global_anchor.py` - Code correct, **not integrated**

### Created for Validation 🔬
- `lux_depth_v2/tools/validate_tiled_inference.py` - **MUST RUN**

### Missing, Required ❌
- `lux_depth_v2/edge_snapping.py` - **NOT IMPLEMENTED**
- A/B comparison script - **NOT IMPLEMENTED**
- Updated old docs with policy clarification - **NOT DONE**

---

## Bottom Line

**Scaffolding is solid. Core claims are unproven. Critical pieces are missing.**

The architecture (tiled + global + edge snapping) is **directionally correct**.
But before calling this "done":

1. **Prove** the tiles aren't being resized internally
2. **Integrate** global anchor fusion
3. **Implement** edge snapping (not optional)
4. **Measure** actual improvement (not forecast)
5. **Update** docs with validated claims (not aspirations)

**Current honest status**: Phase 1 scaffolding complete, Phase 1 validation incomplete.

---

**Next Actions** (Priority Order):
1. Run `validate_tiled_inference.py` → Update docs with actual tensor sizes
2. Integrate `global_anchor.py` into `TiledDepthEstimator`
3. Implement `edge_snapping.py` with joint bilateral
4. Run A/B on pool/kitchen → Update docs with measured improvement
5. Adjust quality score to de-emphasize unique levels

---

## 🔬 A/B VALIDATION RESULTS (2025-12-17)

### ❌ **VALIDATION FAILED - DO NOT DEPLOY**

**Test**: A/B comparison on 3 luxury interior images (750 Picacho)  
**Configuration**: Tiled (1024×1024) + Global Anchor + Edge Snapping  
**Duration**: 104 seconds total processing

### Measured Results vs Forecast

| Metric | Forecast | Measured | Status |
|--------|----------|----------|--------|
| **Edge Alignment** | 0.1 → 0.6 (6x↑) | 0.031 → -0.104 (5x↓) | ❌ **FAILED** |
| **Edge Sharpness** | "Sharper" | +3159% (31x↑) | ✅ **EXCEEDED** |
| **Processing Time** | 10-30s | 30.3s avg | ✅ **MATCHED** |
| **Edge Overlap** | Not specified | **0.2%** | ❌ **CRITICAL** |
| **Edge Pixel Count** | Not specified | **100x increase** | ⚠️ **CONCERNING** |

### Critical Findings

1. **Edge Misalignment** (-503% vs +400% expected)
   - Baseline: 0.031 alignment, ~3k edge pixels
   - Enhanced: -0.104 alignment (negative!), ~351k edge pixels
   - **Only 0.2% edge overlap** between RGB and depth edges
   - Edges are in **fundamentally different locations**

2. **Over-Sharpening** (100x more edges)
   - Enhanced depth has 100x more edge pixels than baseline
   - Mean gradient increased 14-21x (very sharp)
   - Likely **noise amplification** or **tiling artifacts**

3. **Potential Tiling Artifacts**
   - 69-192 edge peaks per axis (expected ~3-5 tiles)
   - Suggests **grid patterns** from tile boundaries
   - Median fusion may not be smoothing seams

4. **Depth Not Inverted**
   - Positive correlation 0.17-0.23 (correct orientation)
   - Different depth distribution (mean 39 vs 64)
   - Not a simple near/far flip

### Root Cause Hypotheses

**Most Likely**:
1. **Global Anchor Contamination** - Low-res pass adding noise
2. **Edge Snapping Amplification** - Bilateral filter creating false edges
3. **Tiling Artifacts** - Grid patterns from median fusion
4. **Metric Inadequacy** - Correlation may not capture perceptual quality

**Needs Testing**:
- Isolate components (test each enhancement separately)
- Adjust parameters (tile size, global weight, bilateral sigma)
- Visual quality inspection (may look better despite metrics)

### Validation Files

```
outputs/ab_validation_750_Picacho/
├── VALIDATION_EXECUTIVE_SUMMARY.md    # Decision-maker summary
├── VALIDATION_REPORT.md                # Technical details
├── validation_summary.json             # Quantitative metrics
├── V2_750Picacho_GreatRoom/
│   ├── comparison.png                  # ⚠️ REQUIRES MANUAL REVIEW
│   ├── baseline_depth.png
│   └── enhanced_depth.png
├── V2_750Picacho_PrimaryBedroom/
│   └── [same structure]
└── 750Picacho_Kitchen_16bit/
    └── [same structure]
```

### Updated Status: Phase 1 Claims

#### Claim 2: "5-10x Edge Fidelity Improvement" ❌ **DISPROVEN**

**Original forecast**: 5-10x improvement in edge alignment  
**Measured result**: **-5x degradation** (0.031 → -0.104)

**Evidence**:
- Edge alignment went **negative** (anti-correlated with RGB)
- Edge overlap only **0.2%** (expected 60%+)
- Enhanced edges in **different locations** than RGB edges

**Conclusion**: High-fidelity pipeline **does not improve** edge quality for luxury DOF/masking in current configuration.

**Next Actions**:
1. ✅ **Completed**: A/B validation on real luxury images
2. ⚠️ **Required**: Manual visual inspection of comparison PNGs
3. ⚠️ **Required**: Component isolation (test tiling, global anchor, edge snapping separately)
4. ⚠️ **Required**: Parameter tuning or architectural redesign
5. ❌ **Blocked**: Production deployment

---

## Updated Bottom Line (Post-Validation)

**Architecture scaffolding**: ✅ Sound (tiled + global + edge snapping)  
**Implementation quality**: ✅ Code is correct  
**Measured performance**: ❌ **FAILED** validation on real images

**Honest assessment**:
- The pipeline produces **dramatically sharper edges** (+3159%)
- But those edges are in the **wrong places** (0.2% RGB overlap)
- Current configuration is **not suitable** for production

**Critical path forward**:
1. Manual visual inspection (metrics may be misleading)
2. Component debugging (isolate failures)
3. Re-validate after fixes
4. If still failing: **abandon or redesign**

**Status**: ❌ **VALIDATION FAILED - REQUIRES DEBUGGING**

---

**Validation Date**: 2025-12-17  
**Validation ID**: ab_validation_750_Picacho_20251217  
**Next Review**: After component isolation and parameter tuning
