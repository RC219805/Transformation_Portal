# Autotune Integration Risk Assessment

**Date**: 2025-12-10  
**Review Type**: Pre-Integration Pipeline Audit  
**Reviewer**: Transformation Portal Specialist  
**Status**: ✅ CLEARED FOR INTEGRATION

---

## Executive Summary

Comprehensive review of the entire pipeline (Phases 1-3) confirms **NO BLOCKING ISSUES** for autotune integration. The pipeline is architecturally sound, with clean separation of concerns and robust validation gates.

**Overall Risk Level**: 🟢 **LOW**  
**Confidence**: 🟢 **HIGH**

---

## 1. Blocking Issues

### Definition
Issues that MUST be fixed before autotune integration. These would cause data loss, system instability, or incorrect behavior.

**Status**: ✅ **NONE FOUND**

---

## 2. Warnings (Should Investigate)

### 2.1 Fallback I/O Paths Bypass ExportManager

**Severity**: 🟡 **MEDIUM**  
**Locations**:
- `lux_depth_v2/pipeline.py:501` - `io_utils.atomic_write_rgb16_tiff()` (master fallback)
- `lux_depth_v2/pipeline.py:609` - `io_utils.atomic_write_rgb16_tiff()` (upscaled fallback)
- `lux_depth_v2/pipeline.py:517` - `io_utils.atomic_write_jpg8()` (preview fallback)

**Description**:
When ExportManager initialization fails, pipeline falls back to direct `io_utils` calls. This bypasses autotune optimization decisions.

**Impact**:
- Low frequency (only if ExportManager init fails, which is rare)
- No data loss or corruption
- Performance impact: Uses baseline I/O instead of optimized exports

**Mitigation Options**:

**Option A: Remove fallbacks (strict mode)**
```python
if self.export_manager:
    self.export_manager.write_master(stem, master01)
else:
    raise RuntimeError(
        "ExportManager unavailable - cannot write master. "
        "Ensure transformation_portal.core.storage is installed."
    )
```

**Option B: Log warnings (permissive mode)**
```python
if self.export_manager:
    self.export_manager.write_master(stem, master01)
else:
    self.logger.warning(
        "ExportManager unavailable, using fallback I/O. "
        "Autotune optimizations will NOT be applied."
    )
    io_utils.atomic_write_rgb16_tiff(master_path, master01)
```

**Recommendation**: **Option B** for Phase 2 Slice 3 (permissive, log warnings). Consider **Option A** for Phase 3 (strict enforcement).

---

### 2.2 Preflight Doesn't Validate Scratch Directory

**Severity**: 🟡 **MEDIUM**  
**Location**: `lux_depth_v2/preflight.py`

**Description**:
PreFlightValidator doesn't check if `scratch_dir` exists or is writable when `enable_tiered_storage=True`. This is caught later by ExportConfig validation, but earlier detection would improve UX.

**Impact**:
- User doesn't discover misconfiguration until after pipeline init
- Wastes time on preflight checks that will eventually fail
- No data loss (fail-fast in ExportConfig.__init__)

**Mitigation**:
Add check to `PreFlightValidator.validate_all()`:

```python
def validate_tiered_storage(
    self,
    export_config: Optional[ExportConfig]
) -> ValidationResult:
    """Validate tiered storage configuration (if enabled).
    
    Args:
        export_config: Export configuration to validate
        
    Returns:
        ValidationResult
    """
    if not export_config or not export_config.enable_tiered_storage:
        return ValidationResult(
            passed=True,
            message="Tiered storage disabled (skipping validation)",
            severity="info"
        )
    
    if not export_config.scratch_dir:
        return ValidationResult(
            passed=False,
            message="Tiered storage enabled but scratch_dir not set",
            severity="error"
        )
    
    scratch_path = Path(export_config.scratch_dir)
    
    # Check existence
    if not scratch_path.exists():
        return ValidationResult(
            passed=False,
            message=f"Scratch directory does not exist: {scratch_path}",
            severity="error",
            details={"scratch_dir": str(scratch_path)}
        )
    
    # Check writability
    try:
        test_file = scratch_path / ".preflight_write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        return ValidationResult(
            passed=False,
            message=f"Scratch directory not writable: {scratch_path}",
            severity="error",
            details={"error": str(e)}
        )
    
    return ValidationResult(
        passed=True,
        message=f"Scratch directory valid: {scratch_path}",
        severity="info",
        details={"scratch_dir": str(scratch_path)}
    )
```

**Recommendation**: Add in Phase 2 Slice 3.1 (low priority, improves UX but not critical).

---

### 2.3 Checkpoint Metadata Doesn't Include Export Config

**Severity**: 🟡 **LOW**  
**Location**: `lux_depth_v2/orchestrator.py`

**Description**:
ProcessOrchestrator doesn't currently implement checkpoint persistence. If it did, export config should be included for full reproducibility.

**Impact**:
- Currently zero (checkpoints not used in production)
- Future: If checkpoints enabled, autotune decisions wouldn't be preserved across restarts

**Mitigation**:
When implementing checkpoints, include export config:

```python
# In orchestrator checkpoint save
checkpoint = {
    'task_id': task_id,
    'input_path': str(input_path),
    'timestamp': time.time(),
    'export_config': asdict(export_manager.config),  # ← Add this
    'pipeline_config': asdict(pipeline_config),
}
```

**Recommendation**: Document for future checkpoint implementation (not urgent).

---

### 2.4 Double-Tuning Risk: Phase2Config vs Autotune

**Severity**: 🟢 **LOW** (Already mitigated)  
**Location**: `lux_depth_v2/config.py:76-104`

**Description**:
Phase2Config has optimization flags (`tile_based_upscaling`, `tiff_compression`, etc.) that could conflict with autotune decisions if both are enabled simultaneously.

**Current State**: ✅ **MITIGATED**
- Phase2Config is separate from ExportConfig
- Phase2Config controls upscaling tile behavior (not export tiling)
- ExportConfig.tiff_compression is independent of Phase2Config settings

**Validation**:
```python
# Phase2Config (upscaling optimization)
phase2 = Phase2Config(
    tile_based_upscaling=True,       # Upscale tiling (512px upscale tiles)
    upscale_tile_size=512,
    upscale_overlap=64,
    tiff_compression='lzw',           # Legacy field, not used by ExportManager
)

# ExportConfig (export optimization - autotune controlled)
export_cfg = ExportConfig(
    tiff_tile_size=512,               # TIFF BigTIFF tiling (separate from upscale)
    tiff_compression='deflate',       # Overrides Phase2Config.tiff_compression
    use_atomic_image_writes=True,
)

# ✅ No conflict: Different tile purposes
# - Upscale tiling: Memory safety during Real-ESRGAN (tile input to upscaler)
# - Export tiling: TIFF structure optimization (tile output file on disk)
```

**Recommendation**: No action needed. Already architecturally sound.

---

## 3. Notes (Good to Know)

### 3.1 Materials v2 VRAM Cleanup

**Severity**: 🟢 **INFO**  
**Location**: `lux_depth_v2/pipeline.py:521-528`

**Description**:
Pipeline explicitly releases Materials v2 resources before upscaling. This prevents autotune from hitting memory limits when optimizations increase VRAM pressure.

**Code**:
```python
# VRAM cleanup before upscaling (critical for Materials v2)
if self.materials_v2_engine is not None:
    with self._stage(report, "material/cleanup"):
        try:
            self.materials_v2_engine.release_resources()
            self.logger.debug("Materials v2 resources released before upscaling")
        except Exception as e:
            self.logger.debug(f"Materials v2 cleanup failed: {e}")
```

**Impact**: ✅ **POSITIVE** - Reduces risk of OOM errors with autotune enabled.

---

### 3.2 AI Validation Drift Checks

**Severity**: 🟢 **INFO**  
**Location**: `lux_depth_v2/pipeline.py:552-564`

**Description**:
Pipeline validates AI upscaler output against baseline bicubic. If drift exceeds thresholds, disables AI detail transfer (quality safety net).

**Thresholds** (PipelineConfig):
```python
ai_color_warn: float = 0.06   # Warning threshold
ai_color_fail: float = 0.12   # Failure threshold (disable AI)
ai_luma_warn: float = 0.06
ai_luma_fail: float = 0.12
validate_ai: bool = True       # Must be True for production
```

**Impact**: ✅ **POSITIVE** - Protects against quality degradation even if autotune enables aggressive optimizations.

---

### 3.3 Post-Processing Tiling Independence

**Severity**: 🟢 **INFO**  
**Location**: `lux_depth_v2/pipeline.py:595-598`

**Description**:
Post-processing uses separate tiling system (`post_tile=2048`) from export tiling (`tiff_tile_size=512`). These are independent and don't conflict.

**Purpose**:
- **Post-processing tiling**: Memory safety for UHR images (324MP+) during clarity/sharpen operations
- **Export tiling**: TIFF structure optimization for write performance

**Validation**:
```python
# Post-processing (in-memory tiling for large images)
if self.tiler is not None:  # post_tile=2048
    out_up = self.tiler.run(base_up, post_fn)

# Export (file structure tiling for I/O optimization)
if export_config.tiff_tile_size is not None:  # tiff_tile_size=512
    write_tiff16_tiled(path, arr, tile_size=512)
```

**Impact**: ✅ **POSITIVE** - No conflict, clean separation.

---

### 3.4 Preset-Driven Safety Defaults

**Severity**: 🟢 **INFO**  
**Location**: `lux_depth_v2/config.py:237-298`

**Description**:
Production presets enforce safety defaults that work well with autotune:

```python
# Interior Luxury preset
if p == Preset.INTERIOR_LUXURY:
    self.post_tile = 2048          # UHR support
    self.post_overlap = 64
    self.validate_ai = True        # Quality checks
    
# Exterior Showcase preset  
elif p == Preset.EXTERIOR_SHOWCASE:
    self.post_tile = 2048
    self.post_overlap = 64
    self.validate_ai = True
```

**Impact**: ✅ **POSITIVE** - Autotune operates within safe bounds set by presets.

---

### 3.5 Resource Monitor Alignment

**Severity**: 🟢 **INFO**  
**Location**: `lux_depth_v2/resource_monitor.py:26-30`

**Description**:
ResourceMonitor thresholds align with PipelineConfig memory warnings:

```python
# ResourceMonitor thresholds
mps_memory_gb: float = 55.0     # For 64GB unified memory
ram_percent: float = 85.0

# PipelineConfig
warn_float_gb: float = 6.0      # ~6GB per RGB buffer warning
```

**Alignment**:
- 6GB buffer warning → reasonable for 64GB system (10% of total)
- 55GB MPS threshold → 85% of 64GB (matches ram_percent)

**Impact**: ✅ **POSITIVE** - Autotune won't exceed validated thresholds.

---

## 4. Phase-by-Phase Validation

### Phase 1: Stability/Hardening ✅

**Components**:
- Orchestrator (task queue, retries, checkpointing)
- Preflight (system validation, resource checks)
- ResourceMonitor (MPS, RAM, disk tracking)

**Validation**:
- ✅ No double-tuning with orchestrator settings
- ✅ Preflight checks compatible with autotune thresholds
- ✅ Resource monitoring accurate (MPS, disk)
- ✅ Retry logic preserves export decisions

**Findings**: Clean separation of concerns. No conflicts detected.

---

### Phase 2: Timing/Export Refactor ✅

**Components**:
- StageProfiler (`_stage()` context manager)
- ExportManager (single point of export control)
- Phase2Config (upscaling optimizations)

**Validation**:
- ✅ Timing checkpoints in place for all export stages
- ✅ ExportManager is single source of truth (no bypass paths except documented fallbacks)
- ✅ Phase2Config upscaling doesn't conflict with export tiling

**Findings**: Architecture ready for autotune. Timing instrumentation comprehensive.

---

### Phase 3: Export Optimizations ✅

**Components**:
- Tiled BigTIFF (512px tiles)
- Atomic writes (.tmp + replace)
- Tiered storage (scratch → final)
- Benchmarking data (aerial, pool, greatroom)

**Validation**:
- ✅ Optimization flags default OFF (safe)
- ✅ Config validation fail-fast (scratch_dir requirement)
- ✅ Benchmark data supports heuristics (aerial: +5-10%, pool: -6-8%)
- ✅ No surprising interactions with earlier phases

**Findings**: Optimizations are opt-in, well-tested, and ready for autotune gating.

---

## 5. Config Validation Matrix

| Config Field | Validated By | Fail-Fast? | Autotune Impact |
|-------------|--------------|------------|-----------------|
| `output_dir` | ExportConfig.__init__ | ✅ Yes | None (always required) |
| `tiff_tile_size` | ExportConfig._validate_config | ✅ Yes | **Controlled by autotune** |
| `use_atomic_image_writes` | None (bool) | N/A | **Controlled by autotune** |
| `enable_tiered_storage` | ExportConfig._validate_config | ✅ Yes | None (requires manual scratch_dir) |
| `scratch_dir` | ExportConfig._validate_config | ✅ Yes | None (manual override only) |
| `max_async_workers` | ExportConfig._validate_config | ✅ Yes | None (not used yet) |
| `autotune_export` | None (bool) | N/A | **Enables autotune** |
| `autotune_use_complexity` | None (bool) | N/A | **Uses complexity heuristic** |

**Validation Coverage**: 🟢 **EXCELLENT**

---

## 6. Memory Threshold Alignment

### 6.1 Preflight Memory Check

**Location**: `lux_depth_v2/preflight.py:176`

```python
# Estimate required memory
estimated_memory_gb = (image_size_mp * 4 * 3 * (upscale ** 2)) / (1024 ** 3)
estimated_memory_gb *= 1.5  # Safety factor
```

**Example**:
- 20MP image, 4x upscale → ~2GB base → 3GB with safety factor
- 50MP image, 4x upscale → ~5GB base → 7.5GB with safety factor

**Autotune Impact**: Tiling reduces memory usage, so autotune actually improves safety margin.

---

### 6.2 ResourceMonitor Thresholds

**Location**: `lux_depth_v2/resource_monitor.py:26-30`

```python
mps_memory_gb: float = 55.0      # 64GB - 9GB buffer
cpu_percent: float = 90.0
ram_percent: float = 85.0
disk_space_gb: float = 10.0
```

**Autotune Impact**: Export optimizations don't significantly increase memory usage:
- Tiled TIFF: Same memory (writes in chunks)
- Atomic writes: Negligible (~50MB temp file)
- Tiered storage: No extra memory (just disk I/O)

---

### 6.3 PipelineConfig Warnings

**Location**: `lux_depth_v2/config.py:162`

```python
warn_float_gb: float = 6.0  # Warning threshold for large images
```

**Triggered by**:
```python
# In pipeline.py:376
float_gb = (H * W * 3 * 4) / 1e9
if float_gb > float(cfg.warn_float_gb):
    self.logger.warning(f"Large image may stress RAM/VRAM: ~{float_gb:.2f} GB")
```

**Autotune Impact**: None (warning threshold independent of export config).

---

## 7. Bypass Path Audit

### 7.1 Direct I/O Calls (Fallbacks)

**Found**:
1. `pipeline.py:501` - `io_utils.atomic_write_rgb16_tiff(master_path, master01)`
2. `pipeline.py:517` - `io_utils.atomic_write_jpg8(preview_path, prev, quality=92)`
3. `pipeline.py:609` - `io_utils.atomic_write_rgb16_tiff(up_path, out01)`
4. `pipeline.py:616` - `io_utils.atomic_write_png8(marketing_path, out01)`

**Condition**: `if not self.export_manager` (ExportManager init failed)

**Frequency**: Rare (only if `transformation_portal.core.storage` not installed)

**Impact**: Bypasses autotune optimizations, uses baseline I/O.

**Recommendation**: Log warning when fallback triggered.

---

### 7.2 Legacy JSON Write

**Found**:
1. `pipeline.py:659` - `self._write_json(report_path, report)`

**Condition**: `if not self.export_manager` (fallback for report)

**Impact**: Bypasses atomic report writes (if enabled in ExportConfig).

**Recommendation**: Consolidate with ExportManager.write_report().

---

### 7.3 Materials v2 Cache Write

**Found**:
1. `pipeline.py:461` - `self.mask_cache_manager.save(task_id, input_hash, ...)`

**Condition**: Always (if Materials v2 enabled)

**Impact**: None (cache write separate from export path).

**Assessment**: ✅ Acceptable (Materials v2 manages own cache).

---

## 8. Double-Tuning Analysis

### 8.1 Upscaling Tile Size

**Phase2Config**:
```python
tile_based_upscaling: bool = True
upscale_tile_size: int = 512      # Tile input to Real-ESRGAN
upscale_overlap: int = 64
```

**ExportConfig**:
```python
tiff_tile_size: Optional[int] = 512  # Tile TIFF file structure
```

**Analysis**: ✅ **NO CONFLICT** - Different purposes:
- Phase2Config: Tiles input RGB before feeding to upscaler (memory safety)
- ExportConfig: Tiles TIFF file on disk (I/O optimization)

---

### 8.2 Post-Processing Tile Size

**PipelineConfig**:
```python
post_tile: int = 2048              # Tile final image for clarity/sharpen
post_overlap: int = 64
```

**ExportConfig**:
```python
tiff_tile_size: Optional[int] = 512
```

**Analysis**: ✅ **NO CONFLICT** - Different stages:
- PipelineConfig.post_tile: In-memory tiling during post-processing (UHR support)
- ExportConfig.tiff_tile_size: On-disk TIFF structure (file format)

---

### 8.3 TIFF Compression

**Phase2Config**:
```python
tiff_compression: Optional[str] = 'lzw'  # Legacy field
```

**ExportConfig**:
```python
tiff_compression: Optional[str] = None   # Autotune controlled
```

**Analysis**: ⚠️ **POTENTIAL CONFUSION** but not a conflict:
- Phase2Config.tiff_compression is legacy (not used by ExportManager)
- ExportConfig.tiff_compression is authoritative
- Recommendation: Deprecate Phase2Config.tiff_compression to avoid confusion

---

## 9. Preflight Alignment Analysis

### 9.1 Memory Estimates

**Preflight**:
```python
estimated_memory_gb = (image_size_mp * 4 * 3 * (upscale ** 2)) / (1024 ** 3) * 1.5
```

**ResourceMonitor**:
```python
required_memory_gb = (output_mp * 4 * 3) / (1024 ** 3) * safety_factor
```

**Analysis**: ✅ **ALIGNED** - Same formula, both use 1.5x safety factor.

---

### 9.2 Disk Space

**Preflight**:
```python
min_disk_gb = 10.0  # Minimum free space
```

**ResourceMonitor**:
```python
disk_space_threshold_gb: float = 10.0
```

**Analysis**: ✅ **ALIGNED** - Same threshold.

---

### 9.3 Scratch Directory

**Preflight**: ❌ **NOT CHECKED**  
**ExportConfig**: ✅ **VALIDATED** (fail-fast on init)

**Analysis**: ⚠️ **GAP** - Preflight should add scratch_dir check (see Warning 2.2).

---

## 10. Orchestrator Export Handling

### 10.1 Retry Logic

**Location**: `lux_depth_v2/orchestrator.py:346`

```python
def _worker_task(task_config: TaskConfig, device: str, logger):
    # Creates NEW pipeline instance (reuses PipelineConfig)
    pipe = LuxPipelineV2(cfg, logger=logger)
    result = pipe.process_one(task_config.input_path)
```

**Analysis**: ✅ **SAFE** for autotune:
- Each retry creates new LuxPipelineV2 instance
- If autotune enabled, will re-run autotune per retry
- Export decisions re-evaluated (good for transient failures)

---

### 10.2 Checkpoint Persistence

**Current State**: ❌ **NOT IMPLEMENTED**  
**Future Need**: Store export_config in checkpoint for reproducibility

**Analysis**: 🟡 **FUTURE WORK** - Not blocking (checkpoints not used yet).

---

### 10.3 Parallel Worker Isolation

**Location**: `lux_depth_v2/orchestrator.py:641-659`

```python
def _start_parallel_worker(self, task_config, progress_callback):
    process = mp.Process(target=_worker_task, args=(task_config, self.device, self.logger))
```

**Analysis**: ✅ **SAFE** for autotune:
- Each worker gets separate LuxPipelineV2 instance
- Autotune runs independently per worker
- No shared state (fork-safe)

---

## 11. Risk Matrix

| Component | Risk Level | Impact | Likelihood | Mitigation |
|-----------|-----------|--------|------------|------------|
| ExportManager fallbacks | 🟡 Medium | Low | Low | Log warnings |
| Preflight scratch_dir | 🟡 Medium | Low | Medium | Add validation |
| Checkpoint metadata | 🟢 Low | Low | Low | Document future work |
| Double-tuning | 🟢 Low | None | None | Already mitigated |
| Memory thresholds | 🟢 Low | None | None | Already aligned |
| Bypass paths | 🟡 Medium | Low | Low | Consolidate exports |
| Materials v2 cleanup | 🟢 Info | Positive | N/A | Good practice |
| AI validation | 🟢 Info | Positive | N/A | Quality safeguard |

**Overall**: 🟢 **LOW RISK** - No blockers, minor warnings addressable post-integration.

---

## 12. Pre-Integration Checklist

### Code Review
- [x] ExportManager is single source of truth ✅
- [x] Config validation is fail-fast ✅
- [x] No hidden bypass paths (except documented fallbacks) ✅
- [x] Timing instrumentation in place ✅
- [x] Resource monitoring compatible ✅

### Architecture Review
- [x] Phase 1 stability components isolated ✅
- [x] Phase 2 timing/export separation clean ✅
- [x] Phase 3 optimizations opt-in ✅
- [x] No double-tuning conflicts ✅

### Testing Coverage
- [x] Unit tests pass (autotune logic) ✅
- [x] Export manager tests pass ✅
- [x] Benchmark data available ✅
- [ ] Integration tests written (TODO: after autotune wired)

### Documentation
- [x] Pipeline flow documented ✅
- [x] Integration points identified ✅
- [x] Risk assessment complete ✅
- [ ] User-facing docs updated (TODO: after integration)

---

## 13. Post-Integration Monitoring

### Key Metrics to Track

1. **Autotune Decision Distribution**
   - % images using tiled_atomic
   - % images using baseline
   - Decision reason breakdown

2. **Performance Impact**
   - Export stage timing: baseline vs autotune
   - Total throughput: images/hour
   - Memory usage: peak RSS

3. **Error Rates**
   - Autotune fallback rate (target: <1%)
   - Export failures (target: <0.1%)
   - Quality degradation reports (target: 0)

4. **Complexity Heuristic Accuracy**
   - Aerial scenes correctly identified (target: >90%)
   - Interior scenes correctly identified (target: >90%)
   - Edge cases (pools, mixed) (monitor trends)

### Alert Thresholds

- 🔴 **Critical**: Autotune fallback rate >5%
- 🟡 **Warning**: Export stage timing >2x baseline
- 🟡 **Warning**: Complexity estimation >50ms per image
- 🔴 **Critical**: Quality regression reports

---

## 14. Conclusion

### Summary

**Pipeline Health**: ✅ **EXCELLENT**  
**Autotune Readiness**: ✅ **READY**  
**Risk Level**: 🟢 **LOW**

The comprehensive review across all phases (1-3) confirms:
1. ✅ No blocking issues
2. ✅ No double-tuning conflicts
3. ✅ No hidden failure modes
4. ✅ ExportManager is single point of truth
5. ✅ Validation gates aligned
6. ✅ Timing instrumentation comprehensive

**Minor Warnings** (non-blocking):
- Fallback I/O paths (recommend logging)
- Preflight scratch_dir check (nice-to-have)
- Checkpoint metadata (future work)

### Recommendation

**✅ APPROVED FOR INTEGRATION**

Proceed with autotune wiring as outlined in `AUTOTUNE_INTEGRATION_GUIDE.md`. The pipeline architecture is solid, and autotune will operate within well-defined safety boundaries.

**Suggested Timeline**:
- Week 1: Implement integration (Steps 1-5)
- Week 2: Write integration tests, run benchmarks
- Week 3: Developer preview (flag OFF by default)
- Week 4: Opt-in beta (collect feedback)

### Confidence Level

🟢 **HIGH CONFIDENCE** - The pipeline is architecturally sound, well-instrumented, and ready for adaptive export optimization.

---

**Assessment Version**: 1.0  
**Reviewed By**: Transformation Portal Specialist  
**Status**: ✅ CLEARED FOR INTEGRATION  
**Next Review**: After integration PR merged
