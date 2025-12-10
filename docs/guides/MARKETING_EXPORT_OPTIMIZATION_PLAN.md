# Marketing Export Optimization Plan

**Date**: 2025-12-10  
**Status**: 🎯 **ACTIVE PRIORITY**  
**Context**: Autotune validated, marketing export identified as primary bottleneck

---

## 🔍 Problem Statement

### Current State (Validated Data):

| Metric | Value | % of Total Export |
|--------|-------|-------------------|
| **export_marketing** | ~106s | **95.2%** |
| **export_upscaled** (TIFF) | ~5.1s | 4.6% |
| **export_master** (TIFF) | ~0.2s | 0.2% |
| **Total Export** | ~111s | 100% |

**Key Insight**: Even if we made TIFF writing **2x faster**, we'd only save ~2.7s (~2.4% overall).

Marketing export is where **90-96% of the time** is spent. This is the optimization target.

---

## 🎯 Objectives

### Primary:
1. Reduce `export_marketing` time by **≥30%** through encoding optimization
2. OR move marketing off critical path via async (perceived completion earlier)

### Secondary:
3. Maintain visual quality (acceptable for marketing use)
4. Keep file size within **≤+20%** of baseline

### Guardrails:
- No breaking changes to existing outputs
- Feature-flagged changes (safe rollback)
- Comprehensive benchmarking before production

---

## 📋 Implementation Slices

### **Slice M0: Instrument Marketing Export** ✅ NEXT

**Goal**: Get precise observability before changing behavior

#### Code Changes:

1. **Add marketing metadata to reports**:
```python
report["marketing_export"] = {
    "encoder": "png",  # or "webp", "jpeg"
    "compression_level": 6,
    "format_params": {...},
    "width": marketing_w,
    "height": marketing_h,
    "bytes_written": marketing_bytes,
    "write_time_s": export_marketing_time,
}
```

2. **Centralize marketing write path**:
   - Ensure all marketing writes go through single function: `write_marketing_image()`
   - Currently in: `src/transformation_portal/core/storage/export_manager.py`
   - Add timing wrapper and metadata capture

3. **Update analysis script**:
   - Extend `scripts/analyze_autotune_production.py`
   - Add marketing-specific analysis:
     - Group by encoder, compression level, preset
     - Compute mean/median `export_marketing`
     - File size distribution
     - Correlation with dimensions & scene type

#### Success Criteria:
- ✅ All marketing writes instrumented
- ✅ Metadata captured in reports
- ✅ Analysis script can parse and summarize

---

### **Slice M1: Encoding Strategy - Cheaper Export** ⏳ PLANNED

**Goal**: Find faster encoder settings without sacrificing quality

#### M1.1: PNG Compression Levels

**Current**: Likely using default (level 6)

**Test Matrix**:
| Level | Speed | Size | Trade-off |
|-------|-------|------|-----------|
| 1 | Fast | Larger | Quick write, acceptable size increase |
| 3-4 | Medium | Medium | Balanced |
| 6 | Default | Smaller | Current baseline |
| 9 | Slow | Smallest | Diminishing returns |

**Implementation**:
```python
# In config.py
@dataclass
class MarketingExportConfig:
    format: Literal["png", "webp", "jpeg"] = "png"
    png_compression_level: int = 6
    webp_quality: int = 90
    jpeg_quality: int = 95
```

**Benchmark Process**:
1. Run Pool, Aerial, GreatRoom with each level
2. Measure: `export_marketing` time, file size
3. Visual check: PNG is lossless, so quality unchanged
4. Choose optimal level per preset

**Expected Outcome**: 20-40s savings with level 1-3 vs level 6-9

---

#### M1.2: Alternative Formats (WebP, JPEG)

**Hypothesis**: Lossy formats acceptable for marketing, much faster encoding

**Test Matrix**:
| Format | Quality | Expected Speed | Expected Size | Quality |
|--------|---------|----------------|---------------|---------|
| PNG level 3 | Lossless | Baseline | Baseline | Perfect |
| WebP q=90 | Near-lossless | **2-3x faster** | 50-70% smaller | Excellent |
| WebP q=95 | Near-lossless | 2x faster | 60-80% smaller | Excellent |
| JPEG q=90 | Lossy | **3-4x faster** | 40-60% smaller | Good |
| JPEG q=95 | Lossy | 2-3x faster | 50-70% smaller | Very good |

**Implementation**:
```python
# In ExportManager
def write_marketing(self, img: np.ndarray, stem: str) -> Path:
    cfg = self.config.marketing
    
    if cfg.format == "webp":
        path = self._write_webp(img, stem, quality=cfg.webp_quality)
    elif cfg.format == "jpeg":
        path = self._write_jpeg(img, stem, quality=cfg.jpeg_quality)
    else:  # png
        path = self._write_png(img, stem, compression=cfg.png_compression_level)
    
    return path
```

**Benchmark Process**:
1. Run same 3 images with all formats
2. Visual comparison (side-by-side)
3. PSNR/SSIM metrics (optional)
4. Timing and size measurements

**Decision Matrix**:
- **Interiors**: PNG level 3 or WebP q=95 (high quality)
- **Exteriors**: WebP q=90 or JPEG q=92 (acceptable loss)

---

#### M1.3: Per-Preset Encoding

**Strategy**: Different presets can use different encoders

```python
# In config.py presets
PRESET_CONFIGS = {
    "interior_luxury": MarketingExportConfig(
        format="png",
        png_compression_level=3,  # Fast, lossless
    ),
    "exterior_showcase": MarketingExportConfig(
        format="webp",
        webp_quality=90,  # Fast, near-lossless
    ),
    "architectural": MarketingExportConfig(
        format="png",
        png_compression_level=3,
    ),
}
```

---

### **Slice M2: Execution Strategy - Async Marketing** 🚀 HIGH ROI

**Goal**: Move marketing off critical path (user gets master/upscaled immediately)

#### Architecture:

```python
# In ExportManager
class ExportManager:
    def __init__(self, config: ExportConfig):
        self.config = config
        self._marketing_async = config.marketing_async
        if self._marketing_async:
            self._executor = ThreadPoolExecutor(
                max_workers=config.marketing_async_workers or 2
            )
        else:
            self._executor = None
    
    def write_marketing(self, img: np.ndarray, stem: str):
        if self._marketing_async:
            # Queue for background processing
            future = self._executor.submit(
                self._do_write_marketing, img.copy(), stem
            )
            return future  # Don't wait
        else:
            # Synchronous (current behavior)
            return self._do_write_marketing(img, stem)
    
    def close(self):
        """Block until all async work completes."""
        if self._executor:
            self._executor.shutdown(wait=True)
```

#### Pipeline Integration:

```python
# In pipeline.py
def process_one(self, img_path: Path):
    # ... processing ...
    
    # Critical outputs (blocking)
    self.export_manager.write_master(master, stem)
    self.export_manager.write_upscaled(upscaled, stem)
    self.export_manager.write_report(report, stem)
    
    # Marketing (async if enabled)
    self.export_manager.write_marketing(marketing, stem)
    
    # User sees "done" here, marketing continues in background
    return report

def __del__(self):
    # Ensure marketing completes
    if self.export_manager:
        self.export_manager.close()
```

#### Config:

```python
@dataclass
class ExportConfig:
    # ... existing fields ...
    
    # Async marketing export
    marketing_async: bool = False  # Default OFF
    marketing_async_workers: int = 2
```

#### Trade-offs:

**Pros**:
- ✅ User-perceived completion **90s faster** (master/upscaled available immediately)
- ✅ Marketing still completes eventually
- ✅ No quality/size changes

**Cons**:
- ⚠️ More complexity in orchestrator/error handling
- ⚠️ Need "partial completion" state tracking
- ⚠️ Failures harder to detect/report

#### Phased Rollout:

1. **Phase 1**: Batch/offline mode only
   - Safe: call `export_manager.close()` at end
   - Validate correctness
2. **Phase 2**: Service mode (optional)
   - Return response before marketing completes
   - Track completion via separate endpoint

---

### **Slice M3: Marketing Autotune** 🤔 OPTIONAL

**Goal**: Automatically choose encoder based on scene characteristics

**Inputs**:
- Preset (interior vs exterior)
- Image dimensions / megapixels
- Scene complexity score
- Text/overlay presence (future)

**Outputs**:
- Encoder (png/webp/jpeg)
- Compression/quality settings

**Example Logic**:
```python
def autotune_marketing_config(
    preset: Preset,
    megapixels: float,
    scene_complexity: float
) -> MarketingExportConfig:
    
    # Exteriors with simple scenes: fast lossy OK
    if preset == Preset.EXTERIOR_SHOWCASE and scene_complexity < 0.5:
        return MarketingExportConfig(
            format="webp",
            webp_quality=88,  # Aggressive
        )
    
    # Interiors or complex scenes: lossless or near-lossless
    else:
        return MarketingExportConfig(
            format="png",
            png_compression_level=3,
        )
```

**Priority**: Low (M1+M2 likely sufficient)

---

## 📊 Measurement & Acceptance Criteria

### Success Metrics:

| Metric | Target | Measurement |
|--------|--------|-------------|
| **export_marketing reduction** | ≥30% | Median time across 20+ images |
| **OR async completion** | User sees "done" 90s earlier | Pipeline returns before marketing |
| **File size increase** | ≤+20% | Compare to PNG level 6 baseline |
| **Visual quality** | No regressions | PSNR/SSIM + manual review |
| **Overall pipeline speedup** | 20-30% | Total time reduction |

### Test Cases:

**Encoding Benchmark** (M1):
```bash
# PNG levels
lux-depth-v2 --input Pool.tif --output out_png1/ --marketing-png-compression 1
lux-depth-v2 --input Pool.tif --output out_png3/ --marketing-png-compression 3
lux-depth-v2 --input Pool.tif --output out_png6/ --marketing-png-compression 6

# WebP
lux-depth-v2 --input Pool.tif --output out_webp90/ --marketing-format webp --marketing-quality 90
lux-depth-v2 --input Pool.tif --output out_webp95/ --marketing-format webp --marketing-quality 95

# Analysis
python scripts/analyze_marketing_encoders.py out_png*/ out_webp*/
```

**Async Benchmark** (M2):
```bash
# Baseline (sync)
lux-depth-v2 --input-dir aerial_batch/ --output-dir out_sync/

# Async
lux-depth-v2 --input-dir aerial_batch/ --output-dir out_async/ --marketing-async

# Compare perceived completion times
```

---

## 🚀 Tactical Checklist (Next 2 Weeks)

### Week 1: M0 + M1.1
- [ ] **M0**: Instrument marketing export
  - [ ] Add metadata to reports (encoder, compression, size, time)
  - [ ] Centralize `write_marketing_image()` path
  - [ ] Update analysis script
- [ ] **M1.1**: PNG compression benchmark
  - [ ] Add `marketing_png_compression_level` config
  - [ ] CLI flag: `--marketing-png-compression`
  - [ ] Run Pool/Aerial/GreatRoom with levels 1, 3, 6, 9
  - [ ] Analyze and choose optimal level

### Week 2: M1.2 + M2 (if time)
- [ ] **M1.2**: Alternative formats
  - [ ] Add `marketing_format` config (png/webp/jpeg)
  - [ ] Implement WebP and JPEG writers
  - [ ] Benchmark same 3 images
  - [ ] Visual quality review
  - [ ] Document chosen formats per preset
- [ ] **M2** (optional): Async prototype
  - [ ] Add `marketing_async` flag
  - [ ] ThreadPoolExecutor in ExportManager
  - [ ] Test in batch mode
  - [ ] Validate correctness

---

## 📝 Documentation Requirements

After implementation:

1. **MARKETING_ENCODING_BENCHMARKS.md**:
   - Encoder comparison table (time, size, quality)
   - Recommended settings per preset
   - Visual examples (side-by-side)

2. **Update AUTOTUNE_ROLLOUT_PLAN.md**:
   - Add marketing optimization results
   - Update success metrics
   - Document overall performance gains

3. **Update README.md**:
   - Document marketing flags
   - Explain async mode trade-offs

---

## 🎯 Expected Outcomes

### Conservative Estimates:

| Optimization | Time Savings | Effort | Priority |
|--------------|-------------|--------|----------|
| PNG level 3 vs 6 | **20-30s** | Low | **HIGH** |
| WebP vs PNG | **40-60s** | Medium | **HIGH** |
| Async marketing | **Perceived: 90s** | Medium | Medium |
| **Total (M1+M2)** | **50-90s reduction** OR async | | |

### Overall Impact:

- **Baseline**: 119s total, 106s marketing
- **Optimized (M1)**: ~80-90s total (25-30% faster)
- **Optimized (M1+M2)**: User sees "done" at ~30s, marketing completes in background

---

## 🔄 Current Status

**Phase**: Planning → Implementation  
**Next Action**: Implement Slice M0 (instrumentation)  
**Blocker**: None  
**Dependencies**: None

**Ready to proceed** ✅

---

**Last Updated**: 2025-12-10  
**Owner**: Transformation Portal Team  
**Status**: 🎯 **ACTIVE - TOP PRIORITY**
