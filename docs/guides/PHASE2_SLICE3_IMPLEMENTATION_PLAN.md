# Phase 2 Slice 3: Export I/O Optimizations - Implementation Plan

**Status**: Planning  
**Target**: Reduce export latency, improve robustness, enable tiered storage  
**Approach**: Feature-flagged optimizations with strict backward compatibility

---

## 0. Baseline & Constraints

### Starting Point (Post Slice 1 & 2)
- ✅ **Slice 1**: StageProfiler wired into core and Lux Depth V2
  - `timing_s` (total float) + `timing_stages_s` (dict) in reports
- ✅ **Slice 2**: ExportManager introduced (`core/storage/export_manager.py`)
  - Lux Depth V2 uses ExportManager for all exports
  - Behavior strictly identical (direct writes), export stages timed

### Constraints for Slice 3
- **Only change what happens inside ExportManager**
- **Only when explicitly enabled via feature flags**
- **Default behavior must remain identical to Slice 2**

---

## 1. Goals & Non-Goals

### ✅ Goals
1. **Reduce export latency significantly** (especially 16-bit TIFF upscaling)
2. **Improve robustness**:
   - Atomic writes for large images and reports
   - Reduced risk of partial/corrupt outputs
3. **Tiered storage**:
   - Write to fast scratch dir, then finalize to output_dir
4. **Keep default behavior identical** until feature flag enabled

### ❌ Non-Goals (Future Work)
- No changes to upscaler, tiling of processing
- No changes to depth/material/grade logic
- No changes to report structure/fields (except optional `export_impl` metadata)
- No changes to batch semantics or checkpoints

---

## 2. Configuration & Feature Flags

### 2.1 Extended ExportConfig

**Location**: `src/transformation_portal/core/storage/export_manager.py`

```python
@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path
    
    # Existing fields (Slice 2)
    master_prefix: str = ""
    upscaled_prefix: str = ""
    preview_prefix: str = ""
    report_suffix: str = "_report.json"
    
    # NEW: Slice 3 fields (all default OFF for backward compatibility)
    enable_tiered_storage: bool = False
    scratch_dir: Optional[Path] = None  # if None, derive from output_dir or tempdir
    require_scratch_on_enable: bool = False  # If True, raise if tiered storage enabled without scratch_dir
    use_atomic_image_writes: bool = False
    use_atomic_report_writes: bool = False
    tiff_tile_size: Optional[int] = None  # e.g., 512 pixels; None = old behavior
    tiff_tile_size_min: int = 128  # Minimum tile size (prevents pathological tile counts)
    tiff_tile_size_max: int = 1024  # Maximum tile size (prevents 1x1-tile bugs)
    tiff_compression: Optional[str] = None  # e.g., "lzw", "zstd"; None = old behavior
    async_flush: bool = False
    max_async_workers: int = 2
```

### 2.2 Config Integration

**From Lux Depth V2 PipelineConfig**:
```python
# Map PipelineConfig → ExportConfig
export_cfg = ExportConfig(
    output_dir=cfg.output_dir,
    enable_tiered_storage=cfg.use_export_optimizations,
    use_atomic_image_writes=cfg.use_export_optimizations,
    use_atomic_report_writes=cfg.use_export_optimizations,
    tiff_tile_size=512 if cfg.use_export_optimizations else None,
    tiff_compression="lzw" if cfg.use_export_optimizations else None,
)
```

**Environment Variables** (for gradual rollout):
- `LUX_EXPORT_SCRATCH=/path/to/scratch`
- `LUX_EXPORT_ATOMIC=1`
- `LUX_EXPORT_TILED=1`

**Key Rule**: If all new flags are left at defaults, behavior stays identical to Slice 2.

---

## 3. Tiered Storage Design

### 3.1 Concept
- **Scratch dir**: Fast local filesystem (SSD, ephemeral)
  - Provided via config (`scratch_dir`)
  - OR derived: `scratch = output_dir / ".scratch"` or OS temp dir

### 3.2 Export Pipeline Flow
1. Write heavy artifacts (upscaled TIFF, master TIFF, preview, PNG, report) into **scratch**
2. When successful, **atomically move** to final `output_dir`

### 3.3 Implementation in ExportManager

**Helper Methods**:
```python
class ExportManager:
    def __init__(self, cfg: ExportConfig):
        self.cfg = cfg
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration on initialization."""
        if self.cfg.enable_tiered_storage and self.cfg.require_scratch_on_enable:
            if not self.cfg.scratch_dir:
                raise ValueError(
                    "require_scratch_on_enable=True but scratch_dir is None. "
                    "Provide scratch_dir or set require_scratch_on_enable=False."
                )
        
        if self.cfg.tiff_tile_size is not None:
            if not (self.cfg.tiff_tile_size_min <= self.cfg.tiff_tile_size <= self.cfg.tiff_tile_size_max):
                raise ValueError(
                    f"tiff_tile_size={self.cfg.tiff_tile_size} outside valid range "
                    f"[{self.cfg.tiff_tile_size_min}, {self.cfg.tiff_tile_size_max}]"
                )
    
    def _resolve_scratch_path(self, final_path: Path) -> Path:
        """Resolve scratch path if tiered storage is enabled."""
        if not self.cfg.enable_tiered_storage or not self.cfg.scratch_dir:
            return final_path
        
        # Map output_dir path to scratch_dir path
        rel = final_path.relative_to(self.cfg.output_dir)
        return self.cfg.scratch_dir / rel
    
    def _atomic_move(self, tmp_path: Path, final_path: Path) -> None:
        """Atomically move from scratch to final location."""
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.replace(final_path)  # atomic within same filesystem
    
    def cleanup_scratch(self) -> None:
        """Clean up scratch directory (operator/maintenance function)."""
        if self.cfg.scratch_dir and self.cfg.scratch_dir.exists():
            # Remove old files, preserve structure
            for item in self.cfg.scratch_dir.rglob("*.tmp"):
                item.unlink()
```

**Write Flow with Export Metadata**:
```python
def _write_image_tiff16(self, stem: str, arr: np.ndarray) -> tuple[Path, dict]:
    final_path = self._build_master_path(stem)
    path = self._resolve_scratch_path(final_path)
    
    # Track export implementation details for debugging/metrics
    export_impl = {
        "final_path": str(final_path),
        "scratch_path": str(path) if path != final_path else None,
        "finalized": False,
        "tiled": self.cfg.tiff_tile_size is not None,
        "tile_size": self.cfg.tiff_tile_size,
        "compression": self.cfg.tiff_compression,
        "atomic": self.cfg.use_atomic_image_writes,
    }
    
    # Heavy write to scratch (or direct if tiered storage disabled)
    self._io.write_tiff16(path, arr, 
                          tile_size=self.cfg.tiff_tile_size,
                          compression=self.cfg.tiff_compression)
    
    # Move to final location if using scratch
    if path != final_path:
        self._atomic_move(path, final_path)
        export_impl["finalized"] = True
    else:
        export_impl["finalized"] = True  # Direct write, no staging
    
    return final_path, export_impl
```

### 3.4 Failure & Cleanup Semantics
- **If write fails**: No final file present; scratch file remains for debugging
- **Cleanup**: Call `ExportManager.cleanup_scratch()` manually or via orchestrator
- **Resume**: Batch system can detect missing final files and retry

---

## 4. Tiled BigTIFF Writing

### 4.1 Goals
- Reduce peak memory & write time for large TIFFs
- Produce standard BigTIFF readable by any downstream tool

### 4.2 Design

**New I/O Helper** (or extend existing):
```python
def write_tiff16_tiled(
    path: Path,
    arr: np.ndarray,
    tile_size: int = 512,
    compression: Optional[str] = "lzw"
) -> None:
    """
    Write 16-bit TIFF with tiling for large images.
    
    Args:
        path: Output path
        arr: RGB uint16 array, shape (H, W, 3)
        tile_size: Tile dimension in pixels
        compression: Compression method ("lzw", "zstd", "deflate", None)
    """
    import tifffile
    
    # Ensure BigTIFF for large images (>4GB)
    bigtiff = arr.nbytes > 2**32
    
    tifffile.imwrite(
        path,
        arr,
        tile=(tile_size, tile_size),
        compression=compression,
        bigtiff=bigtiff,
        photometric='rgb',
        planarconfig='contig'
    )
```

**Integration in ExportManager**:
```python
def _write_tiff16(self, path: Path, arr: np.ndarray) -> None:
    if self.cfg.tiff_tile_size:
        write_tiff16_tiled(
            path, arr,
            tile_size=self.cfg.tiff_tile_size,
            compression=self.cfg.tiff_compression
        )
    else:
        write_tiff16_legacy(path, arr)  # existing behavior
```

### 4.3 Backward Compatibility
- **Default**: `tiff_tile_size = None`, `tiff_compression = None` → old path
- **When enabled**: Bitwise identity not guaranteed (different compression)
  - Documented as opt-in improvement
  - Outputs remain valid 16-bit TIFFs

---

## 5. Async Flush (Optional - Can Defer)

### 5.1 Concept
- Small thread pool (`concurrent.futures.ThreadPoolExecutor`)
- Write non-critical outputs (preview JPG, marketing PNG) asynchronously
- Move from scratch to final while pipeline continues

### 5.2 Implementation Sketch

**Initialization**:
```python
from concurrent.futures import ThreadPoolExecutor

class ExportManager:
    def __init__(self, cfg: ExportConfig):
        self.cfg = cfg
        self._executor = ThreadPoolExecutor(
            max_workers=cfg.max_async_workers
        ) if cfg.async_flush else None
```

**Write Method**:
```python
def write_preview(self, stem: str, arr: np.ndarray) -> Path:
    """Write preview JPG (optionally async)."""
    path = self._get_preview_path(stem)
    
    if self._executor:
        future = self._executor.submit(self._write_jpg, path, arr)
        # Track future for shutdown
        return path
    else:
        self._write_jpg(path, arr)
        return path
```

**Shutdown (Critical for Resource Cleanup)**:
```python
def close(self) -> None:
    """
    Shutdown async executor and cleanup resources.
    
    MUST be called by:
    - Pipeline shutdown
    - Batch job cleanup
    - Error paths (try/finally blocks)
    
    Prevents thread pool memory leaks.
    """
    if self._executor:
        self._executor.shutdown(wait=True)
        self._executor = None
```

**Integration Points**:
```python
# In lux_depth_v2/pipeline.py
class LuxPipelineV2:
    def __del__(self):
        """Cleanup on pipeline destruction."""
        if hasattr(self, 'export_manager'):
            self.export_manager.close()
    
    def process_directory(self):
        try:
            # ... processing ...
        finally:
            self.export_manager.close()

# In batch processor
def _process_item(self, item: JobItem) -> None:
    try:
        # ... processing ...
    finally:
        if hasattr(self, 'pipeline'):
            self.pipeline.export_manager.close()
```

---

## 6. Atomic Write Semantics

### 6.1 Images (with Tiered Storage)
When `enable_tiered_storage=True` + `use_atomic_image_writes=True`:
- Writes go to scratch
- `_atomic_move()` provides atomicity

**Without Tiered Storage**:
```python
if self.cfg.use_atomic_image_writes:
    tmp = path.with_suffix(path.suffix + ".tmp")
    self._io.write_tiff16(tmp, arr)
    tmp.replace(path)
else:
    self._io.write_tiff16(path, arr)
```

### 6.2 Reports
```python
def write_report(self, stem: str, report_dict: dict) -> Path:
    path = self._get_report_path(stem)
    
    if self.cfg.use_atomic_report_writes:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(report_dict, indent=2))
        tmp.replace(path)
    else:
        path.write_text(json.dumps(report_dict, indent=2))
    
    return path
```

**Documentation**:
- Atomic images + reports **only when flags are true**
- Default remains non-atomic for strict backward compatibility

---

## 7. Implementation Steps (Code Changes)

### Phase 1: Config & Infrastructure (PR-1)
1. **Extend ExportConfig** with new fields (all default OFF)
2. **Add helper methods**:
   - `_resolve_scratch_path()`
   - `_atomic_move()`
   - `cleanup_scratch()`
3. **Update write methods** to use scratch resolution
4. **Integration**: Map Lux Depth V2 `PipelineConfig` to `ExportConfig`

**Files**:
- `src/transformation_portal/core/storage/export_manager.py`
- `lux_depth_v2/pipeline.py` (config mapping)

### Phase 2: Tiled TIFF & Atomic Writes (PR-2)
1. **Implement tiled TIFF writer**:
   - `lux_depth_v2/io_utils.py`: `write_tiff16_tiled()`
2. **Add atomic write logic** for images and reports
3. **Wire into ExportManager** write methods

**Files**:
- `lux_depth_v2/io_utils.py`
- `src/transformation_portal/core/storage/export_manager.py`

### Phase 3: Async Flush (PR-3 - Optional)
1. **Add ThreadPoolExecutor** to ExportManager
2. **Async write methods** for preview/marketing
3. **Shutdown hook** in pipeline finalization

**Files**:
- `src/transformation_portal/core/storage/export_manager.py`
- `lux_depth_v2/pipeline.py`

---

## 8. Testing Plan

### 8.1 Unit Tests (tests/core/storage/test_export_manager_slice3.py)

**Config Validation**:
```python
def test_config_validation_scratch_required():
    """Verify config validation when scratch is required but missing."""
    with pytest.raises(ValueError, match="require_scratch_on_enable=True but scratch_dir is None"):
        cfg = ExportConfig(
            output_dir=Path("/output"),
            enable_tiered_storage=True,
            require_scratch_on_enable=True,
            scratch_dir=None
        )
        ExportManager(cfg, mock_io)

def test_config_validation_tile_size_bounds():
    """Verify tile size is validated against min/max bounds."""
    with pytest.raises(ValueError, match="tiff_tile_size=.* outside valid range"):
        cfg = ExportConfig(
            output_dir=Path("/output"),
            tiff_tile_size=64  # below minimum of 128
        )
        ExportManager(cfg, mock_io)

def test_config_validation_tile_size_max():
    """Verify tile size maximum is enforced."""
    with pytest.raises(ValueError, match="tiff_tile_size=.* outside valid range"):
        cfg = ExportConfig(
            output_dir=Path("/output"),
            tiff_tile_size=2048  # above maximum of 1024
        )
        ExportManager(cfg, mock_io)
```

**Tiered Storage**:
```python
def test_tiered_storage_path_resolution():
    """Verify scratch path resolution when tiered storage enabled."""
    cfg = ExportConfig(
        output_dir=Path("/output"),
        enable_tiered_storage=True,
        scratch_dir=Path("/scratch")
    )
    mgr = ExportManager(cfg, mock_io)
    
    final_path = Path("/output/test_master16.tif")
    scratch_path = mgr._resolve_scratch_path(final_path)
    
    assert scratch_path == Path("/scratch/test_master16.tif")

def test_atomic_move():
    """Verify atomic move from scratch to final."""
    # Test that tmp.replace() is called correctly

def test_write_master_with_tiered_storage(tmp_path):
    """Full integration: write → scratch → move to final."""
    # Verify file ends up in output_dir, not scratch_dir
```

**Tiled TIFF**:
```python
def test_tiled_tiff_selection(monkeypatch):
    """Verify tiled writer is used when tile_size is set."""
    cfg = ExportConfig(
        output_dir=Path("/output"),
        tiff_tile_size=512,
        tiff_compression="lzw"
    )
    
    called = {}
    def mock_tiled_write(path, arr, tile_size, compression):
        called['tile_size'] = tile_size
        called['compression'] = compression
    
    monkeypatch.setattr("lux_depth_v2.io_utils.write_tiff16_tiled", mock_tiled_write)
    
    mgr = ExportManager(cfg, ...)
    mgr.write_master("test", sample_image)
    
    assert called['tile_size'] == 512
    assert called['compression'] == "lzw"

def test_legacy_writer_when_tile_size_none():
    """Verify legacy writer is used when tile_size is None."""
```

**Atomic Writes**:
```python
def test_atomic_image_write(tmp_path):
    """Verify .tmp file is used and replaced for atomic writes."""

def test_atomic_report_write(tmp_path):
    """Verify atomic report write when flag enabled."""

def test_non_atomic_default(tmp_path):
    """Verify direct writes when atomic flags are False."""
```

### 8.2 Regression Tests (Behavior Parity)

**SHA256 Identity Check**:
```python
def test_export_behavior_parity_when_optimizations_disabled():
    """
    With all Slice 3 flags OFF, outputs must be bit-identical to Slice 2.
    
    Uses SHA256 hashes of master/upscaled/preview/report files.
    """
    # Run with old config (Slice 2)
    old_results = run_export_with_config(ExportConfig(output_dir=...))
    
    # Run with new config (Slice 3, all flags OFF)
    new_results = run_export_with_config(ExportConfig(
        output_dir=...,
        enable_tiered_storage=False,
        use_atomic_image_writes=False,
        # all defaults
    ))
    
    for filename in ["master16.tif", "upscaled16.tif", "report.json"]:
        assert sha256(old_results / filename) == sha256(new_results / filename)
```

### 8.3 Integration Tests (lux_depth_v2/tests/)

**Test File**: `lux_depth_v2/tests/test_pipeline_export_optimizations.py`

```python
def test_pipeline_with_optimizations_disabled():
    """Pipeline with use_export_optimizations=False → Slice 2 behavior."""
    cfg = PipelineConfig(
        ...,
        use_export_optimizations=False
    )
    pipeline = LuxPipelineV2(cfg)
    result = pipeline.process_one(test_image)
    
    # Assert outputs exist, timing_s populated
    assert result["status"] == "ok"

def test_pipeline_with_tiered_storage():
    """Pipeline with tiered storage enabled."""
    cfg = PipelineConfig(
        ...,
        use_export_optimizations=True,
        export_scratch_dir=tmp_path / "scratch"
    )
    pipeline = LuxPipelineV2(cfg)
    result = pipeline.process_one(test_image)
    
    # Assert outputs in output_dir (not scratch)
    assert (cfg.output_dir / "test_master16.tif").exists()
    
    # Assert scratch is clean or can be cleaned
    pipeline.export_manager.cleanup_scratch()
```

### 8.4 Performance Sanity (Manual/Local)

**Micro-benchmark**:
```python
@pytest.mark.perf
def test_export_performance_comparison(benchmark_large_image):
    """
    Compare export performance: old vs new (tiled + compressed).
    
    Run locally, log results to feed manual evaluation.
    """
    import time
    
    # Old path (Slice 2)
    t0 = time.perf_counter()
    export_old(benchmark_large_image)
    old_time = time.perf_counter() - t0
    
    # New path (Slice 3, optimizations ON)
    t0 = time.perf_counter()
    export_new(benchmark_large_image)
    new_time = time.perf_counter() - t0
    
    print(f"Old: {old_time:.2f}s, New: {new_time:.2f}s, Speedup: {old_time/new_time:.2f}x")
```

---

## 9. Rollout Strategy

### Stage 1: PR-1 (Config + Infrastructure)
- All logic behind flags, **nothing enabled by default**
- Full test coverage for new code paths
- Documentation: "Slice 3 optimizations available, opt-in only"

### Stage 2: PR-2 (Local Benchmarking)
- Run performance tests on dedicated host
- Measure:
  - Export latency reduction (target: 30-50% for large TIFFs)
  - Memory usage impact
  - Output file size with compression
- Document results in `docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md`

### Stage 3: PR-3 (Gradual Enablement)
- **Option A**: Enable for specific presets (e.g., `signature_estate`, `ultra_quality`)
- **Option B**: Enable based on input size threshold (e.g., >50MP images only)
- **Option C**: Environment variable opt-in for production validation

### Stage 4: Default Enablement (Post-Validation)
- After confidence is high (weeks/months of production use)
- Update defaults in `ExportConfig` or `PipelineConfig`
- Major version bump or clear release notes

---

## 10. Success Metrics

### Performance Goals
- ✅ **Export latency**: 30-50% reduction for 16-bit TIFFs >50MP
- ✅ **Memory peak**: No increase (or slight decrease with tiling)
- ✅ **File size**: 20-40% reduction with LZW/ZSTD compression

### Reliability Goals
- ✅ **Zero partial outputs**: Atomic writes prevent corruption
- ✅ **Clean failure recovery**: Scratch cleanup on retry
- ✅ **Backward compatibility**: 100% parity when flags OFF

### Observability
- ✅ **Timing breakdown**: `timing_stages_s` shows export substages
- ✅ **Export metadata**: Reports include `export_impl` (tiled/atomic/scratch status)
  - Example: `{"finalized": true, "tiled": true, "tile_size": 512, "compression": "lzw"}`

---

## 11. Documentation Updates

### User-Facing
- **README.md**: Add section on export optimizations
- **Configuration Guide**: Document new ExportConfig fields
- **Performance Guide**: Benchmark results, tuning recommendations

### Developer-Facing
- **ARCHITECTURE.md**: Update with tiered storage design
- **ExportManager API docs**: Document new methods and behavior
- **Testing Guide**: Document performance testing approach

---

## 12. Risk Mitigation

### Risk: Optimization breaks existing workflows
**Mitigation**: All optimizations behind feature flags, default OFF

### Risk: Scratch dir fills disk
**Mitigation**: 
- Automatic cleanup on success
- Manual cleanup command: `ExportManager.cleanup_scratch()`
- Monitoring: disk space alerts

### Risk: Performance regression in some cases
**Mitigation**:
- Benchmarking before default enablement
- Per-preset or per-size-threshold enablement
- Easy rollback via config change

### Risk: Compression artifacts
**Mitigation**:
- Use lossless compression (LZW, ZSTD, DEFLATE)
- Validate output quality in regression tests
- Document compression settings

---

## 13. Next Steps

1. **Review & Approve Plan**: Team review of this document
2. **Create PR-1 Branch**: `feature/phase2-slice3-pr1-config-infrastructure`
3. **Implement Config & Tests**: Follow Phase 1 plan
4. **Iterate**: PR-2 (tiled TIFF), PR-3 (async flush - optional)
5. **Benchmark & Validate**: Local performance testing
6. **Gradual Rollout**: Feature flag → preset → default

---

---

## 14. Expert Refinements Applied

### Configuration Hardening
✅ **Added**: `require_scratch_on_enable` - prevents misconfiguration  
✅ **Added**: `tiff_tile_size_min` (128) and `tiff_tile_size_max` (1024) - prevents pathological tile counts  
✅ **Added**: Config validation in `ExportManager.__init__()` with clear error messages

### Export Metadata Tracking
✅ **Enhanced**: Write methods return `(Path, dict)` tuple with export implementation details  
✅ **Added**: `export_impl` dict includes:
- `final_path`, `scratch_path`, `finalized` status
- `tiled`, `tile_size`, `compression`
- `atomic` write flag
✅ **Purpose**: Debugging, metrics, future audit trails

### Resource Management
✅ **Critical**: Explicit `close()` method for ExportManager  
✅ **Integration**: Pipeline `__del__` and `process_directory` finally blocks  
✅ **Purpose**: Prevents thread pool memory leaks in long-running processes

### Test Coverage Expansion
✅ **Added**: Config validation tests (3 new test cases)  
✅ **Coverage**: Scratch requirement, tile size bounds enforcement  
✅ **Purpose**: Fail-fast on misconfiguration, not at runtime

---

**Document Status**: Expert-Reviewed & Hardened  
**Last Updated**: 2025-12-10  
**Author**: GitHub Copilot CLI  
**Expert Review**: Complete  
**Implementation Status**: Ready for PR-1
