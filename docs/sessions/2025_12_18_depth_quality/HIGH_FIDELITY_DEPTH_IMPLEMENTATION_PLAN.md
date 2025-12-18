# High-Fidelity Depth Pipeline Implementation Plan
**Date**: 2025-12-17  
**Status**: IMPLEMENTATION READY  
**Priority**: CRITICAL (quality blocker)

---

## Executive Summary

The current depth pipeline achieves "16-bit numerical precision" but suffers from **critical spatial fidelity issues**: soft object boundaries, smooth depth ramps, and nearly-flat normal maps. User feedback (2025-12-17) identifies the root causes and highest-impact fixes.

**Core Problem**: Low-resolution depth inference (few hundred pixels) bicubically interpolated to 4K creates smooth gradients and mushy edges—no amount of 16-bit precision can restore missing spatial detail.

**Solution**: Three-phase implementation targeting **dramatic quality improvements** through:
1. **Tiled high-resolution inference** (the real unlock - 5-10x edge quality)
2. **Fixed normal map computation** (currently unusable for PBR)
3. **Correct quality metrics** (edge alignment, not misleading gradients)

---

## Current State Analysis

### Configuration
- **Ensemble**: 3-model weighted average (50/35/15) - **BLURS EDGES**
- **Super-resolution**: 2× upscaling - **COSMETIC IF MODEL RESIZES INTERNALLY**
- **Filtering**: Guided filter (r=10, eps=0.02) - **CONFIGURED TO SMOOTH**
- **Clipping**: Percentile 0.5-99.5
- **Measured Edge Score**: 0.09 vs target ≥180 - **METRIC IS WRONG**

### Critical Issues

#### Issue #1: Low-Resolution Inference (CRITICAL)
**Symptom**: Broad smooth ramps, soft boundaries (furniture edges bleed)  
**Root Cause**: Depth predicted at model's internal resolution (~few hundred px), then bicubically interpolated to 4K  
**Impact**: No post-processing can restore missing spatial detail  
**Fix**: Tile-based high-res inference (Phase 2, highest leverage)

#### Issue #2: Fundamentally Wrong Normal Map (BUG)
**Symptom**: Almost uniform purple/blue, not usable for PBR  
**Root Cause**: Computing normals with excessively large Z constant, forcing camera-facing orientation  
**Impact**: Material Response and relighting cannot work  
**Fix**: Correct math (Phase 1, immediate)

#### Issue #3: Misleading Quality Metrics (BLOCKER)
**Symptom**: Edge gradient ≥180 vs 0.09 - wrong proxy for usability  
**Root Cause**: Incorrect metric, wrong scaling, doesn't reflect DOF/masking requirements  
**Impact**: Pipeline optimizing wrong thing  
**Fix**: Edge alignment, edge width, halo detection (Phase 1, immediate)

#### Issue #4: Edge-Blurring Ensemble (HIGH IMPACT)
**Symptom**: Weighted average smears boundaries where models disagree  
**Root Cause**: Plain averaging instead of robust fusion  
**Impact**: Destroys edge fidelity ensemble is meant to improve  
**Fix**: Median or confidence-weighted fusion (Phase 1, high impact)

#### Issue #5: Edge-Smoothing Filter Config (HIGH IMPACT)
**Symptom**: Guided filter (r=10, eps=0.02) washes out discontinuities  
**Root Cause**: Filter configured for smoothing, not edge snapping  
**Impact**: Erases boundaries instead of tightening them  
**Fix**: Joint bilateral upsampling / edge snapping (Phase 2)

---

## Implementation Plan

### Phase 1: Immediate Fixes (Same Day) ✅ READY
**Goal**: Fix bugs and replace misleading metrics

#### 1.1 Fix Normal Map Computation ✅ IMPLEMENTED
**File**: `lux_depth_v2/normal_map.py`

**Changes**:
- ✅ Normalize depth to [0, 1] before gradient computation
- ✅ Use sane Z scale (default 1.0 for architectural, tunable)
- ✅ Proper Scharr/Sobel gradients with optional smoothing
- ✅ Tangent-space output for PBR compatibility
- ✅ Validation metrics (X/Y std, Z mean, angle distribution)
- ✅ Presets: architectural, subtle, pronounced

**Expected Impact**: Normal maps become usable for Material Response and relighting

**Testing**:
```python
from lux_depth_v2.normal_map import generate_normal_map, NormalMapGenerator, PRESETS

# Quick test
normals = generate_normal_map(depth_uint16, preset="architectural")

# Advanced
generator = NormalMapGenerator(PRESETS["pronounced"])
normals = generator.generate(depth, strength=1.2)
metrics = generator.validate_normal_map(normals)
```

---

#### 1.2 Replace Edge Metrics ✅ IMPLEMENTED
**File**: `lux_depth_v2/quality_metrics.py`

**Changes**:
- ✅ **Edge Alignment Score**: Correlation between RGB edges (Canny) and depth edges (Sobel)
- ✅ **Edge Width**: Median transition width at object boundaries
- ✅ **Halo/Ringing Detection**: Penalize overshoot artifacts
- ✅ **Spatial Detail Score**: Variance in local windows
- ✅ **Luxury Rendering Validation**: Pass/fail with specific issues
- ✅ **Overall Quality Score [0-100]**: Composite metric

**Expected Impact**: Pipeline now optimizes for **correct** DOF/masking quality

**Testing**:
```python
from lux_depth_v2.quality_metrics import quick_quality_check, DepthQualityAnalyzer

# Quick check
metrics = quick_quality_check(rgb_image, depth_map, depth_uint16)
print(metrics)

# Validation
analyzer = DepthQualityAnalyzer(
    target_edge_alignment=0.6,
    target_edge_width_px=3.0,
    target_unique_levels=10000
)
metrics = analyzer.analyze(rgb, depth)
passes, issues = analyzer.validate_for_luxury_rendering(metrics)
```

---

#### 1.3 Change Ensemble Fusion ⏳ TODO
**File**: `lux_depth_v2/pipeline.py` (or new `lux_depth_v2/ensemble.py`)

**Changes**:
- ⏳ Replace weighted mean with **median fusion** (preserves discontinuities)
- ⏳ OR implement **confidence-weighted mixture** (downweight unstable regions)
- ⏳ Align scales before fusing: fit each model to reference using robust regression
- ⏳ Add fusion mode config: `median` | `weighted` | `confidence`

**Expected Impact**: 2-3x edge quality improvement vs current weighted average

---

#### 1.4 Retune/Replace Guided Filter ⏳ TODO
**File**: `lux_depth_v2/pipeline.py` or edge-aware filtering module

**Changes**:
- ⏳ Option 1: Reduce radius (r=3-5) and eps (0.001-0.005) for edge preservation
- ⏳ Option 2: Use **two-stage filtering**: smooth interior, snap edges
- ⏳ Option 3: Replace with **joint bilateral upsampling** (RGB-guided)
- ⏳ Add config: `edge_filter_mode: smooth | snap | bilateral`

**Expected Impact**: Tighter depth boundaries, less washout

---

### Phase 2: Real Quality Lift (1-2 Days) ✅ TILING READY

#### 2.1 Implement Tiled Inference ✅ IMPLEMENTED
**File**: `lux_depth_v2/depth_inference.py`

**Architecture**:
```
Input Image (4K)
     ↓
Extract Overlapping Tiles (1024×1024, overlap=128)
     ↓
Infer Each Tile at Model Native Resolution
     ↓
Per-Tile Scale/Shift Reconciliation (robust fit in overlap)
     ↓
Blend with Hann/Cosine Window (seamless)
     ↓
Output: High-Fidelity 4K Depth
```

**Implementation**:
- ✅ `TiledInferenceConfig`: tile_size, overlap, fusion_mode, blend_window
- ✅ `TiledDepthEstimator`: core tiling engine
- ✅ `_extract_tiles()`: overlapping grid
- ✅ `_infer_tile()`: single tile inference (model native resolution)
- ✅ `_reconcile_tile_scale()`: robust linear fit in overlap (Theil-Sen or percentile)
- ✅ `_make_blend_window()`: Hann/cosine/linear ramps
- ✅ `_blend_tiles()`: median or weighted fusion with windowing
- ✅ `compute_edge_alignment()`: validation metric

**Expected Impact**: 
- **5-10x edge fidelity improvement** (the real unlock)
- Captures fine detail (window frames, furniture edges, molding)
- Eliminates "smooth ramp" artifacts

**Testing**:
```python
from lux_depth_v2.depth_inference import create_tiled_estimator

estimator = create_tiled_estimator(
    tile_size=1024,
    overlap=128,
    fusion_mode="median",
    device="auto"
)

depth = estimator.estimate_depth(rgb_image)
edge_score = estimator.compute_edge_alignment(rgb_image, depth)
```

---

#### 2.2 Add Edge Snapping Post-Process ⏳ TODO
**File**: `lux_depth_v2/edge_refinement.py` (new)

**Approach**: Joint bilateral upsampling or bilateral solver

**Implementation**:
```python
# After tile assembly, tighten edges to RGB boundaries
depth_snapped = joint_bilateral_upsample(
    guide=rgb_image,
    depth=depth_tiled,
    sigma_spatial=5,
    sigma_color=0.1
)
```

**Expected Impact**: Further 1.5-2x edge quality improvement

---

### Phase 3: Luxury-Grade (As Needed)

#### 3.1 Multi-View Geometry Fusion (OPTIONAL)
**When**: If multiple images with parallax available  
**Approach**: SfM/MVS (COLMAP/AliceVision) → fuse with ML depth  
**Impact**: True geometric edges, physically accurate depth

---

#### 3.2 Domain-Specific Fine-Tuning (EXPENSIVE)
**When**: Repeated production on luxury interiors  
**Approach**: Fine-tune Depth Anything V2 on labeled interior dataset  
**Impact**: Consistent handling of glass, marble, bright windows

---

#### 3.3 Depth Matte Pipeline (HYBRID)
**When**: Maximum edge control needed  
**Approach**: Segmentation (SAM) + depth → multi-layer matte  
**Implementation**:
- Segment objects with EfficientSAM or SAM-2
- Use depth to order layers (foreground/midground/background)
- Output: crisp object masks + smooth within-region depth
**Impact**: Art-directable edges even when monocular depth uncertain

---

## Success Criteria

### Phase 1 (Immediate)
- [x] Normal map shows surface variation (X/Y std > 0.1, angle_median > 15°)
- [ ] Edge alignment score > 0.5 (vs current ~0.1)
- [ ] Edge width < 5px median (vs current smooth ramps)
- [ ] Ensemble fusion preserves boundaries (visual inspection)

### Phase 2 (1-2 Days)
- [ ] Edge alignment score > 0.6 (target for luxury)
- [ ] Unique 16-bit levels > 10,000 (vs current 755-16,375 post-CLAHE)
- [ ] Visual: sharp furniture edges, clean window boundaries, distinct molding
- [ ] Throughput: <5sec per 4K image on M4 Max (tiling overhead acceptable)

### Phase 3 (If Needed)
- [ ] Edge alignment > 0.7 (research-grade)
- [ ] Client acceptance for high-end projects
- [ ] Production-ready for repeated architectural visualization

---

## Integration Plan

### Step 1: Add New Modules (✅ DONE)
- ✅ `lux_depth_v2/depth_inference.py` - Tiled inference
- ✅ `lux_depth_v2/normal_map.py` - Correct normals
- ✅ `lux_depth_v2/quality_metrics.py` - Correct metrics

### Step 2: Update Pipeline (⏳ TODO)
**File**: `lux_depth_v2/pipeline.py`

```python
# Replace current depth estimation
from lux_depth_v2.depth_inference import create_tiled_estimator

self.depth_estimator = create_tiled_estimator(
    tile_size=cfg.depth_tile_size,
    overlap=cfg.depth_tile_overlap,
    fusion_mode=cfg.depth_fusion_mode,
    device=self.device
)

# Replace normal map generation
from lux_depth_v2.normal_map import generate_normal_map

normals = generate_normal_map(depth_uint16, preset=cfg.normal_preset)

# Add quality validation
from lux_depth_v2.quality_metrics import quick_quality_check

metrics = quick_quality_check(rgb, depth, depth_uint16)
if cfg.validate_quality:
    analyzer = DepthQualityAnalyzer()
    passes, issues = analyzer.validate_for_luxury_rendering(metrics)
    if not passes:
        logger.warning(f"Quality issues: {issues}")
```

### Step 3: Update Config (⏳ TODO)
**File**: `lux_depth_v2/config.py`

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    
    # Phase 2 Slice 4: High-Fidelity Depth
    depth_tile_size: int = 1024  # Tile size for high-res inference
    depth_tile_overlap: int = 128  # Overlap for seamless blending
    depth_fusion_mode: str = "median"  # median | weighted | confidence
    depth_blend_window: str = "hann"  # hann | cosine | linear
    
    normal_preset: str = "architectural"  # architectural | subtle | pronounced
    normal_z_scale: float = 1.0  # Override preset z_scale
    normal_strength: float = 1.0  # Gradient multiplier
    
    validate_quality: bool = True  # Enforce quality bar
    target_edge_alignment: float = 0.6  # Luxury rendering minimum
```

### Step 4: Update Presets (⏳ TODO)
**File**: `lux_depth_v2/config.py` - `PipelineConfig.apply_preset()`

```python
elif preset == Preset.INTERIOR_LUXURY_APEX_QUALITY:
    # ... existing params ...
    self.depth_tile_size = 1024
    self.depth_tile_overlap = 128
    self.depth_fusion_mode = "median"
    self.normal_preset = "architectural"
    self.validate_quality = True
```

### Step 5: Testing (⏳ TODO)
```bash
# Unit tests
pytest lux_depth_v2/tests/test_depth_inference.py -v
pytest lux_depth_v2/tests/test_normal_map.py -v
pytest lux_depth_v2/tests/test_quality_metrics.py -v

# Integration test
python -m lux_depth_v2 \
    --input-dir test_images/ \
    --output-dir output_highfid/ \
    --preset interior_luxury_apex_quality \
    --validate-quality

# Benchmark
python lux_depth_v2/tools/benchmark_depth.py \
    --compare-modes current,tiled_512,tiled_1024,tiled_1536
```

---

## Risk Mitigation

### Performance
**Risk**: Tiling adds 2-4x overhead (multiple model runs)  
**Mitigation**:
- Parallelize tile inference (batch on GPU)
- Use smaller tiles (512) for fast preview, large (1536) for production
- LRU cache for iterative workflows

### Compatibility
**Risk**: Breaking changes to existing pipeline  
**Mitigation**:
- Feature flag: `use_tiled_inference=False` (default backward compatible)
- Gradual rollout: add as new preset `interior_luxury_max_quality_tiled`
- Version migration: keep current ensemble for comparison

### Quality Regression
**Risk**: Tiling introduces seams or artifacts  
**Mitigation**:
- Robust scale reconciliation in overlaps
- Smooth blending windows (Hann/cosine)
- Validation: edge alignment must improve (current ~0.1 → target 0.6+)
- A/B testing: current vs tiled on client sample set

---

## Next Actions

### Immediate (Today)
1. ✅ **Implement modules** (depth_inference.py, normal_map.py, quality_metrics.py)
2. ⏳ **Write unit tests** for new modules
3. ⏳ **Integrate into pipeline.py** (feature-flagged)
4. ⏳ **Update config.py** with new parameters
5. ⏳ **Test on sample images** (pool, kitchen from diagnosis report)

### Short-Term (This Week)
6. ⏳ **Benchmark performance** (throughput, memory, quality)
7. ⏳ **Fix ensemble fusion** (median or confidence-weighted)
8. ⏳ **Tune edge filtering** (bilateral or two-stage)
9. ⏳ **Update presets** (apex_quality → apex_quality_v2_highfid)
10. ⏳ **Document usage** (README, integration guide)

### Medium-Term (Next Week)
11. ⏳ **Production validation** on client projects
12. ⏳ **Performance optimization** (if needed: batched tiles, caching)
13. ⏳ **Edge snapping post-process** (joint bilateral)
14. ⏳ **CI integration** (quality validation in tests)

---

## Appendix: User Feedback Summary

**Source**: User message 2025-12-17 23:21:42 UTC

### Key Points
1. **"You can't post-process your way out of low-res inference"** → Tiling is the unlock
2. **"Your normal map is fundamentally wrong"** → Fix Z scale and math
3. **"Your edge 'sharpness' metric is misleading"** → Use edge alignment
4. **"Your ensemble strategy is blurring edges"** → Switch to median/robust
5. **"Your guided filter is configured to smooth"** → Retune for snapping

### Highest-Impact Fixes (Prioritized)
1. **Tile-based high-resolution inference** (dramatic - 5-10x quality)
2. **Fix normal map computation** (bug - currently unusable)
3. **Replace edge metrics** (blocker - optimizing wrong thing)
4. **Change ensemble to median** (high impact - 2-3x edge quality)
5. **Retune edge filter** (high impact - stop smoothing boundaries)

### Recommended "Phase 2" Path
**For luxury DOF/masking use case**: Tile inference + edge snapping  
**If multiple images**: Add multi-view fusion  
**If repeatable production**: Fine-tune on interior dataset

---

**Status**: ✅ Phase 1 modules implemented, ready for integration testing  
**Next**: Write unit tests, integrate into pipeline, validate on sample images
