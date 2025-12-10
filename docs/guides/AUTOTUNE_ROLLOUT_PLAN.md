# Autotune Rollout & Next Steps

**Date**: 2025-12-10  
**Status**: 🚀 Controlled Production Trial  
**Phase**: Post-Integration Validation

---

## ✅ Completed: Heavy Lifting Done

- [x] Autotune integrated (commit 02aefa4)
- [x] Feature-flagged (default OFF)
- [x] 42 tests passing (unit + integration)
- [x] Comprehensive documentation (2,632 lines)
- [x] Benchmark corrections (5-10% gains for aerial)
- [x] Pipeline review (no blocking issues)

**Current State**: Autotune is in main, tested, documented, and ready for production validation.

---

## 🎯 Phase 1: Controlled Production Trial

### 1.1 Start with Aerial/Exterior Only

**Target Workloads**:
- ✅ Aerial/overhead content (validated in benchmarks)
- ✅ ~20-30 MP inputs
- ✅ Low-to-moderate scene complexity (sky, land, water)
- ❌ Interiors/pools (keep on baseline for now)

**CLI Usage**:
```bash
lux-depth-v2 \
  --input input_images/aerial_batch/ \
  --output output_autotune/aerial_batch/ \
  --autotune-export \
  --autotune-complexity
```

**Expected Gains**: ~5-10% total pipeline time reduction

---

### 1.2 Capture Real Metrics

**What to Collect** (already in reports):
- `timing_stages_s.export_upscaled`
- `timing_stages_s.export_marketing` (the 90-96% bottleneck)
- `timing_s.total_export`
- `timing_s.total_pipeline`
- `export_autotune.final_export_config`
- `export_autotune.image_stats` (MP, complexity)

**Analysis Script** (reuse benchmark tools):
```python
# scripts/analyze_autotune_production.py
import json
from pathlib import Path
import statistics

def analyze_autotune_batch(output_dir: Path):
    reports = list(output_dir.glob("*/*_report.json"))
    
    total_pipeline_times = []
    export_times = []
    marketing_times = []
    
    for report_path in reports:
        with open(report_path) as f:
            report = json.load(f)
        
        total_pipeline_times.append(report["timing_s"]["total_pipeline"])
        export_times.append(report["timing_s"]["total_export"])
        marketing_times.append(
            report["timing_stages_s"].get("export_marketing", 0)
        )
    
    print(f"Total Pipeline: {statistics.mean(total_pipeline_times):.1f}s "
          f"(median: {statistics.median(total_pipeline_times):.1f}s)")
    print(f"Export: {statistics.mean(export_times):.1f}s")
    print(f"Marketing: {statistics.mean(marketing_times):.1f}s "
          f"({100*statistics.mean(marketing_times)/statistics.mean(export_times):.1f}% of export)")
```

**Validation Criteria**:
- ✅ Mean total_pipeline within +0 to -10% vs baseline
- ✅ No individual images >15% slower
- ✅ Marketing export still ~90-96% of total export (no surprises)
- ✅ Scene complexity correlates with autotune decisions

---

### 1.3 Sanity Thresholds

**Red Flags** (investigate immediately):
- ⚠️ Any image >10-15% slower with autotune ON
- ⚠️ export_marketing timing anomalies
- ⚠️ File size differences (should be identical)
- ⚠️ Unexpected autotune decisions (complexity misclassification)

**Action if Red Flags**:
1. Log the image path and report
2. Inspect `export_autotune.image_stats` and `final_export_config`
3. Disable autotune for that preset/bucket
4. File issue with details for refinement

---

## 🔧 Phase 2: Refine Heuristics (Post-Validation)

### 2.1 Adjust Thresholds Based on Real Data

**Current Policy**:
```python
COMPLEXITY_THRESHOLD = 0.5   # Below = simple (aerial-like)
MEGAPIXEL_THRESHOLD = 20.0   # Above = large image

if megapixels > 20 and complexity < 0.5:
    enable tiled_atomic
```

**Potential Refinements** (data-driven):
- Bump complexity threshold to 0.6 if real aerials with 0.55 still benefit
- Tighten to 0.4 if borderline interiors get misclassified
- Add scene_label check (preset name) as secondary signal
- Adjust megapixel threshold based on observed file sizes

**Where to Change**: `autotune_export_config()` in `export_manager.py`

---

### 2.2 Optional: Dry-Run Logging Mode

**For Maximum Safety** (before full trust):
```python
# In PipelineConfig
autotune_export_log_only: bool = False

# In pipeline.py
if cfg.phase2.autotune_export_log_only:
    # Compute autotune decision
    tuned_cfg = autotune_export_config(...)
    
    # Log what would have happened
    logger.info(f"[DRY-RUN] Autotune would use: {tuned_cfg}")
    
    # Use baseline config anyway
    export_cfg = baseline_cfg
```

Compare "what autotune would have done" vs "what baseline did" before fully committing.

**Trade-off**: More safety, but delays validation (probably overkill given current test coverage).

---

## 🚀 Phase 3: Marketing Export Optimization (The Real Bottleneck)

### Key Finding from Benchmarks

| Image | Total Export | export_marketing | TIFF (master+upscaled) | TIFF % |
|-------|-------------|------------------|------------------------|---------|
| Pool | ~114s | ~110s | ~5.2s | **4.5%** |
| Aerial | ~119s | ~113s | ~6.0s | **5.0%** |
| GreatRoom | ~46s | ~44s | ~2.0s | **4.3%** |

**Insight**: Even 2x faster TIFF writes only save ~2-3% overall. Marketing export is 90-96% of the time.

---

### 3.1 Async / Background Marketing Flush

**Concept**: Move marketing PNG export off critical path

**Options**:
1. **Background Thread** (simplest):
   ```python
   # After master/upscaled/report written
   if cfg.phase2.async_marketing_export:
       executor.submit(self.export_manager.write_marketing, ...)
       # Pipeline returns immediately
   ```

2. **Separate Orchestrator Queue** (more robust):
   - Primary pipeline writes master/upscaled/report
   - Enqueue marketing export job
   - Separate worker processes handle marketing
   
3. **Separate Process** (max isolation):
   - Fork/spawn marketing export after critical outputs done

**Trade-offs**:
- ✅ User gets "ready to view" assets faster (master/upscaled)
- ✅ Marketing arrives seconds later (acceptable for most workflows)
- ⚠️ More complexity in orchestrator/error handling
- ⚠️ Need to track "partial completion" state

**Priority**: Phase 3 PR-3 candidate (already identified in architecture)

---

### 3.2 Cheaper Marketing Encoding

**Current**: PNG with default compression (likely 6-9)

**Options**:
1. **Lower PNG compression**:
   ```python
   # In PIL or imagecodecs
   compression_level=1  # vs 6-9 default
   # Trade: ~30-40s faster, ~10-20% larger files
   ```

2. **Alternative Formats**:
   - WebP (smaller, faster encode than PNG)
   - AVIF (best compression, slower encode)
   - JPEG XL (good middle ground)
   - JPG at high quality (fastest, lossy but acceptable for preview)

3. **Lower Resolution**:
   - Marketing at 2048px longest edge instead of full upscaled
   - Still looks good for web/preview
   - Encode time drops proportional to pixel count

**Benchmark Priority**: Quick side-by-side with StageProfiler will show cost/benefit immediately.

**Action**: Create `scripts/benchmark_marketing_encoders.py` using existing benchmark infrastructure.

---

### 3.3 Precomputed Marketing Inputs

**Concept**: Generate marketing from already-upscaled TIFF, avoid re-running expensive transforms

**Current Flow** (potential redundancy):
```
rgb → depth → materials → grade → MASTER → upscale → UPSCALED → [transforms] → MARKETING
```

**Optimized Flow**:
```
rgb → depth → materials → grade → MASTER → upscale → UPSCALED
                                                      ↓
                                            downscale/encode → MARKETING
```

**Benefit**: Skip re-applying transforms if marketing is just a different resolution/format of upscaled.

**Check**: Verify marketing stage doesn't re-do depth/materials/grading that's already in UPSCALED.

---

## 🔨 Phase 4: Close Small Gaps

### 4.1 Preflight Scratch Dir Validation

**Current**: ExportConfig validates scratch_dir, but only at ExportManager init  
**Better**: Preflight checks scratch_dir early (fail-fast)

```python
# In preflight.py
if cfg.phase2.enable_tiered_storage:
    if not cfg.phase2.scratch_dir:
        raise ValueError("enable_tiered_storage requires scratch_dir")
    if not cfg.phase2.scratch_dir.exists():
        raise ValueError(f"scratch_dir does not exist: {cfg.phase2.scratch_dir}")
```

---

### 4.2 Fallback I/O Logging

**Issue**: Rare paths bypass ExportManager (documented but not logged)

**Fix**:
```python
# In any fallback path
logger.warning(
    "[FALLBACK] ExportManager unavailable, using legacy io_utils path. "
    "This bypasses Phase 2 optimizations. Reason: %s", reason
)
```

---

### 4.3 Deprecate Unused Phase2 Config

**If**: `Phase2Config.tiff_compression` is superseded by `ExportConfig.tiff_compression`

**Then**: Mark deprecated in docstring:
```python
@dataclass
class Phase2Config:
    # DEPRECATED: Use ExportConfig.tiff_compression instead
    # Kept for backward compatibility, will be removed in v3.0
    tiff_compression: str | None = None
```

---

### 4.4 Checkpoint Metadata (Future)

**When checkpoints implemented**: Capture autotune decision

```python
checkpoint_record = {
    "image_path": str(input_path),
    "stage": "export/autotune",
    "export_config": {
        "tiff_tile_size": export_cfg.tiff_tile_size,
        "use_atomic_image_writes": export_cfg.use_atomic_image_writes,
        # ...
    },
    "autotune_decision": {
        "enabled": cfg.phase2.autotune_export,
        "image_stats": stats.__dict__,
        "rationale": "megapixels > 20 AND complexity < 0.5",
    },
}
```

Allows post-hoc debugging of "why did autotune choose X?"

---

## 📝 Phase 5: User-Facing Documentation

### Update README.md

Add to CLI section:
```markdown
### Autotune Export Optimization (Experimental)

Enable adaptive export configuration based on image characteristics:

```bash
lux-depth-v2 --input aerial.tif --output out/ --autotune-export
```

**Recommended for**:
- Aerial/exterior scenes
- Large images (>20 MP)
- Homogeneous content (sky, water, terrain)

**Not recommended for**:
- Interiors with complex textures
- Pools/water with reflections
- Small images (<20 MP)

**Status**: Opt-in, default OFF. Performance gains: ~5-10% for aerial-like scenes.

**Flags**:
- `--autotune-export`: Enable autotune
- `--autotune-complexity`: Use scene complexity heuristic (default: True)
```

### Update CLI Help

```python
# In cli.py
@click.option(
    "--autotune-export",
    is_flag=True,
    help="Enable adaptive export optimization (best for aerial/exterior scenes)."
)
@click.option(
    "--autotune-complexity",
    is_flag=True,
    default=True,
    help="Use scene complexity heuristic for autotune decisions."
)
```

---

## 📋 Priority Checklist (Next 2 Weeks)

### Week 1: Validation
- [ ] Run controlled trial on 50+ aerial images with `--autotune-export`
- [ ] Collect reports and analyze with `analyze_autotune_production.py`
- [ ] Validate ~5-10% gains hold in production
- [ ] Check for any anomalies or misclassifications
- [ ] Refine thresholds if needed

### Week 2: Marketing Optimization Planning
- [ ] Create `scripts/benchmark_marketing_encoders.py`
- [ ] Test PNG compression levels (1, 3, 6, 9)
- [ ] Test WebP/AVIF as alternatives
- [ ] Test lower resolution (2048px longest edge)
- [ ] Document findings and choose approach

### Optional (As Needed)
- [ ] Add preflight scratch_dir validation
- [ ] Add fallback I/O logging
- [ ] Deprecate unused Phase2Config fields
- [ ] Update README with autotune usage
- [ ] Update CLI help text

---

## 🎯 Success Metrics

### Short-Term (2 weeks)
- ✅ Autotune validated in production (50+ images)
- ✅ No regressions or anomalies
- ✅ Thresholds refined based on real data
- ✅ Marketing optimization benchmarks complete

### Medium-Term (1 month)
- ✅ Marketing export optimized (async or cheaper encoding)
- ✅ Total pipeline time reduced by 15-25% (5-10% autotune + 10-15% marketing)
- ✅ User documentation complete

### Long-Term (3 months)
- ✅ Autotune enabled by default for aerial presets
- ✅ Phase 3 async flush implemented
- ✅ Checkpoint system with autotune metadata

---

## 🔄 Current Mode: Controlled Rollout

**No longer in**: Design/architecture mode  
**Now in**: Controlled rollout and next-bottleneck targeting

**Focus**:
1. Use what we built to validate in production
2. Aim next development cycles at `export_marketing` (the real bottleneck)

**Next PR**: Marketing export optimization (Phase 3 PR-3)

---

**Last Updated**: 2025-12-10  
**Status**: 🟢 Ready for Production Trial
