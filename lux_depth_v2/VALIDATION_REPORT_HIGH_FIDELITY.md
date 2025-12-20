# High-Fidelity Depth Pipeline Validation Report
**Date**: 2025-12-18  
**Status**: ✅ VALIDATED

## Executive Summary

We have successfully **identified and fixed** the critical issue preventing high-fidelity depth inference. The HuggingFace transformers pipeline was **silently resizing all inputs to 518×518**, making tiled inference ineffective. We have implemented a bypass mechanism that preserves native resolution.

## Critical Findings

### Finding #1: HuggingFace Pipeline Internal Resize ⚠️

**Discovery**: The `transformers.pipeline("depth-estimation")` API **always resizes inputs to 518×518**, regardless of input size.

**Evidence**:
```
Test tile: 512×512  → Model input: 518×518 (resize factor: 1.01x)
Test tile: 1024×1024 → Model input: 518×518 (resize factor: 0.51x)
Test tile: 1536×1536 → Model input: 518×518 (resize factor: 0.34x)
```

**Impact**: This completely defeats the purpose of tiled inference. A 1024×1024 tile gets downsampled to 518×518 before processing, then bicubic-upsampled back to 1024×1024.

**Verdict**: ❌ **FAIL** - The original implementation was processing at ~0.5x resolution

### Finding #2: Bypass Solution ✅

**Solution**: Load model directly with `AutoModelForDepthEstimation` and use `AutoImageProcessor` with `do_resize=False`.

**Validation Results**:
```python
# Original (HF pipeline)
Input: 1024×1024 → Preprocessed: 518×518 ❌

# Fixed (bypass mode)
Input: 1024×1024 → Preprocessed: 1024×1024 ✅
```

**Evidence**: See `lux_depth_v2/validation_report_tiled_inference.json`

## Implemented Enhancements

### 1. Bypass Image Processor
**Module**: `lux_depth_v2/depth_inference.py`

**Key Changes**:
```python
# New config parameter
bypass_image_processor: bool = True  # CRITICAL: Bypass HF's 518px resize

# Direct model loading
self.image_processor = AutoImageProcessor.from_pretrained(model_name)
self.model = AutoModelForDepthEstimation.from_pretrained(model_name)

# Preprocessing with bypass
inputs = self.image_processor(
    images=tile_pil, 
    return_tensors="pt",
    do_resize=False  # CRITICAL: Disable resize
)
```

**Measured Improvement**: 
- **Resolution**: 518px → 1024px (2.0x linear, 4.0x pixels)
- **Expected quality gain**: 5-10x edge fidelity (pending empirical validation)

### 2. Global Anchor Fusion
**Module**: `lux_depth_v2/global_anchor.py`

**Integration**:
```python
# Added to TiledInferenceConfig
use_global_anchor: bool = True
global_anchor_config: Optional[GlobalAnchorConfig] = None

# In estimate_depth()
if self.config.use_global_anchor:
    global_depth = run_global_pass()  # Low-res full frame
    tiled_depth = run_tiled_pass()    # High-res tiles
    depth = fuse(global_depth, tiled_depth, rgb)
```

**Purpose**: Prevents tile artifacts (low-frequency banding, plane warps, global drift)

**Strategy**:
1. Global pass captures scene structure (walls, floors, ceiling planes)
2. Tiled passes capture fine detail (edges, furniture, fixtures)
3. Fusion: `global_LF + tiled_HF` (frequency-based decomposition)

### 3. Edge Snapping
**Module**: `lux_depth_v2/edge_snapping.py`

**Integration**:
```python
# Added to TiledInferenceConfig
use_edge_snapping: bool = True
edge_snap_config: Optional[EdgeSnappingConfig] = None

# In estimate_depth() - final step
if self.config.use_edge_snapping:
    depth = edge_snapper.snap(depth, rgb)
```

**Purpose**: Joint bilateral upsampling to snap depth discontinuities to RGB edges

**Importance**: "NOT OPTIONAL for luxury-grade DOF/masking" - User feedback

## Pipeline Architecture

```
Input RGB (e.g., 2048×2048)
    ↓
┌───────────────────────────────────────┐
│ Step 1: Global Anchor Pass            │
│ - Resize to 512px (context only)      │
│ - Single inference (fast)             │
│ - Captures scene-wide structure       │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Step 2: Tiled High-Res Inference      │
│ - Extract 1024×1024 tiles (128px overlap) │
│ - Process at NATIVE RESOLUTION        │
│ - Blend with Hann window              │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Step 3: Global Anchor Fusion          │
│ - Combine: global_LF + tiled_HF       │
│ - Edge-aware weighting                │
│ - Prevents tile seams                 │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Step 4: Edge Snapping                 │
│ - Joint bilateral filter              │
│ - Snap depth edges to RGB edges       │
│ - Sharp mattes for DOF/masking        │
└───────────────────────────────────────┘
    ↓
Output Depth (2048×2048, luxury-grade)
```

## Validation Tools

### 1. Tile Resolution Validation
**Script**: `lux_depth_v2/tools/validate_tiled_inference.py`

**Purpose**: Proves whether tiles are processed at native resolution

**Usage**:
```bash
python lux_depth_v2/tools/validate_tiled_inference.py \
    --output validation_report.json \
    --tile-sizes 512 1024 1536
```

**Results**: See `lux_depth_v2/validation_report_tiled_inference.json`

### 2. Bypass Mode Validation
**Script**: `lux_depth_v2/tools/validate_bypass_mode.py`

**Purpose**: Confirms `do_resize=False` preserves resolution

**Usage**:
```bash
python lux_depth_v2/tools/validate_bypass_mode.py
```

**Result**: ✅ PASS - 512×512 and 1024×1024 tiles preserved

### 3. A/B Comparison
**Script**: `lux_depth_v2/tools/ab_comparison.py`

**Purpose**: Empirical measurement of improvement

**Metrics**:
- Edge alignment score (correlation with RGB edges)
- Edge sharpness (gradient magnitude)
- Processing time

**Usage**:
```bash
# With synthetic test pattern
python lux_depth_v2/tools/ab_comparison.py

# With real image
python lux_depth_v2/tools/ab_comparison.py \
    --input path/to/image.jpg \
    --output-dir results/
```

**Status**: ⏳ Pending execution (requires model download ~5GB)

## Measured vs Expected Improvements

| Metric | Baseline (518px) | Enhanced (1024px) | Improvement | Status |
|--------|------------------|-------------------|-------------|--------|
| Resolution | 518×518 | 1024×1024 | 2.0x linear | ✅ Validated |
| Pixel count | 268k | 1049k | 3.9x | ✅ Validated |
| Edge alignment | TBD | TBD | **Expected: +30-50%** | ⏳ Pending A/B test |
| Edge sharpness | TBD | TBD | **Expected: +50-100%** | ⏳ Pending A/B test |
| Processing time | TBD | TBD | **Expected: 5-10x slower** | ⏳ Pending A/B test |

**Note**: Originally claimed "5-10x improvement" was **speculative**. We now label it as "expected" pending empirical validation.

## Documentation Status

### Completed ✅
- [x] Core implementation (depth_inference.py)
- [x] Global anchor integration (global_anchor.py)
- [x] Edge snapping integration (edge_snapping.py)
- [x] Validation scripts (validate_*.py)
- [x] A/B comparison tool (ab_comparison.py)
- [x] This validation report

### Pending ⏳
- [ ] Run A/B comparison with real images
- [ ] Measure actual edge alignment improvement
- [ ] Update docs with measured results (not expected)
- [ ] Performance benchmarks (throughput, memory)
- [ ] Integration tests in main pipeline

## Usage Examples

### Basic Tiled Inference
```python
from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig

# Configure
config = TiledInferenceConfig(
    tile_size=1024,
    overlap=128,
    bypass_image_processor=True,  # CRITICAL: No 518px resize
    use_global_anchor=True,
    use_edge_snapping=True
)

# Create estimator
estimator = TiledDepthEstimator(config)

# Process image
depth = estimator.estimate_depth(rgb_image)
```

### Custom Configuration
```python
from lux_depth_v2.global_anchor import GlobalAnchorConfig
from lux_depth_v2.edge_snapping import EdgeSnappingConfig

config = TiledInferenceConfig(
    tile_size=1536,
    overlap=256,
    bypass_image_processor=True,
    
    # Global anchor settings
    use_global_anchor=True,
    global_anchor_config=GlobalAnchorConfig(
        global_max_size=512,
        global_weight=0.3,
        use_frequency_split=True
    ),
    
    # Edge snapping settings
    use_edge_snapping=True,
    edge_snap_config=EdgeSnappingConfig(
        sigma_spatial=5.0,
        sigma_color=0.1,
        snap_strength=0.8
    )
)
```

## Next Steps

### Immediate (Critical Path)
1. **Run A/B comparison** with real architectural renders
2. **Measure actual improvements** (replace "expected" with "measured")
3. **Update all claims** to reflect empirical data

### Short-term (Integration)
4. Integrate into main `lux_depth_v2/pipeline.py`
5. Add CLI flags (`--use-tiled-inference`, `--tile-size`, etc.)
6. Create integration tests

### Long-term (Production)
7. Performance optimization (GPU tiling, async processing)
8. Memory optimization (tile streaming, checkpointing)
9. Multi-scale processing (pyramid approach)

## Conclusion

### What We Proved ✅
- HuggingFace pipeline **does** internally resize to 518px
- Bypass mode **successfully** preserves native resolution
- Integration of global anchor and edge snapping **completed**

### What We Haven't Proved Yet ⏳
- Actual edge alignment improvement (need A/B test)
- Actual edge sharpness improvement (need A/B test)
- Real-world performance on luxury renders (need testing)

### Honest Assessment
We have **fixed the critical flaw** (518px resize) and **integrated the enhancements** (global anchor, edge snapping). The architecture is **sound** and **validated at the component level**.

However, we **must run empirical tests** before claiming "5-10x improvement". The current status is:
- **Technical implementation**: ✅ Complete
- **Quality validation**: ⏳ Pending A/B test
- **Documentation**: ✅ Complete (with honest "expected" labels)

---

**Recommendation**: Run A/B comparison with 3-5 real architectural renders, measure improvements, and update this document with **measured** results.
