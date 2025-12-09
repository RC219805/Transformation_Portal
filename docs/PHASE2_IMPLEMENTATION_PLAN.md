# Phase 2 Performance Enhancements - Implementation Plan

**Date Created**: 2025-12-09  
**Status**: IN PROGRESS  
**Target Completion**: Week 2-3 (10-15 days)

---

## Executive Summary

Deploy Phase 2 performance optimizations targeting **10-20× performance improvement** while maintaining Phase 1's 100% success rate and <2% overhead.

### Current Baseline (Phase 1)
- ✅ 100% success rate (6/6 images)
- ✅ Fault isolation working
- ✅ Checkpoint/resume capability
- ⚠️ Performance: 14 min/image average, 34 min for Pool image
- ⚠️ Throughput: 20 images/hour (target: 60-120 img/hr)
- ⚠️ Sequential processing only (max_workers=1)

### Phase 2 Targets
- **Performance**: 14 min → 1-2 min per image (7-14× faster)
- **Pool Image**: 34 min → <5 min (7-10× faster)
- **Throughput**: 20 img/hr → 30-60 img/hr minimum
- **Success Rate**: Maintain 100%
- **Quality**: No degradation (AI diff <0.004)
- **Overhead**: <5% from Phase 2 optimizations

---

## Phase 2 Architecture Overview

Based on validation results, the key bottlenecks are:

### Identified Bottlenecks
1. **4× Upscaling I/O**: Pool image 34 minutes (disk writes dominate)
2. **Throughput**: Sequential processing, no parallelism
3. **Disk I/O**: Large TIFF writes (1.6GB files) causing severe bottlenecks
4. **Memory**: Not optimally utilizing 64GB unified memory

### Optimization Strategy

**Week 2 Focus: I/O Optimization (Highest ROI)**
1. Async TIFF writing (eliminate write bottleneck)
2. Streaming upscaling (tile-based, progressive write)
3. Storage manager (intelligent tiering internal/T9)
4. Expected: 5-7× improvement on write-heavy images

**Week 3 Focus: Parallel Processing + Upscaling**
1. Parallel orchestrator (2-4 concurrent workers)
2. Model caching (avoid repeated loading)
3. Depth map caching (reuse across runs)
4. Expected: Additional 2-3× improvement with parallelism

---

## Implementation Priorities

### Priority 1: I/O Optimization (Days 1-3) 🔥

**Module**: `lux_depth_v2/io_optimizer.py`

**Features**:
- `AsyncTIFFWriter`: Non-blocking TIFF writes with background threads
- `StreamingUpscaleWriter`: Write tiles as generated, never buffer full image
- Compression support (LZW) for space savings
- Progress callbacks for monitoring

**Expected Impact**: 5-7× faster on Pool image (34 min → 5-7 min)

### Priority 2: Storage Manager (Days 2-4)

**Module**: `lux_depth_v2/storage_manager.py`

**Features**:
- `StorageConfig`: Configure internal + T9 tiering
- `StorageManager`: Intelligent tier selection
- Auto-migration: Move large files (>2GB) to T9
- Space management: Pre-flight checks, cleanup

**Expected Impact**: Eliminate disk space bottlenecks, enable larger batches

### Priority 3: Upscaling Optimization (Days 3-5)

**Module**: `lux_depth_v2/upscale_optimizer.py`

**Features**:
- `TileBasedUpscaler`: Process in tiles, stream to output
- `UpscaleCache`: Keep model loaded across batch
- Progressive upscaling: 2×2 instead of 4× (memory safety)

**Expected Impact**: 2-3× faster upscaling, lower memory usage

### Priority 4: Model & Depth Caching (Days 4-6)

**Module**: `lux_depth_v2/cache_optimizer.py`

**Features**:
- `ModelCache`: Global cache across batch (singleton)
- `DepthMapCache`: Disk-based depth map caching
- Smart eviction: Free models before upscaling

**Expected Impact**: 1.5-2× faster by eliminating repeated loads

### Priority 5: Parallel Orchestrator (Days 5-7)

**Update**: `lux_depth_v2/orchestrator.py`

**Features**:
- `ParallelOrchestrator`: 2-4 concurrent workers
- Memory budget per worker (25GB default)
- Dynamic scheduling based on MPS availability
- Resource-aware queueing

**Expected Impact**: 1.8-2× additional throughput improvement

### Priority 6: Configuration & CLI (Days 6-7)

**Updates**:
- `lux_depth_v2/config.py`: Add `Phase2Config` dataclass
- `lux_depth_v2/cli.py`: Add Phase 2 CLI options
- YAML configuration support for all features

### Priority 7: Testing & Validation (Days 7-10)

**Test Suite**: `tests/test_phase2_performance.py`

**Benchmarks**:
- Single image I/O optimization (baseline vs Phase 2)
- Batch processing parallel vs sequential
- Pool image validation (<5 min target)
- Full 750 Picacho validation (6/6 success, performance)

---

## Implementation Timeline

### Week 2: I/O Optimization Focus

**Day 1-2: Async I/O Module**
- [ ] Create `lux_depth_v2/io_optimizer.py`
- [ ] Implement `AsyncTIFFWriter`
- [ ] Implement `StreamingUpscaleWriter`
- [ ] Unit tests for async writing

**Day 2-3: Storage Manager**
- [ ] Create `lux_depth_v2/storage_manager.py`
- [ ] Implement `StorageConfig` and `StorageManager`
- [ ] Auto-migration logic
- [ ] T9 tier integration
- [ ] Unit tests for storage tiering

**Day 3-4: Upscaling Optimizer**
- [ ] Create `lux_depth_v2/upscale_optimizer.py`
- [ ] Implement `TileBasedUpscaler`
- [ ] Implement `UpscaleCache`
- [ ] Progressive upscaling (2×2)
- [ ] Unit tests

**Day 4-5: Integration Testing**
- [ ] Test I/O optimization on Pool image
- [ ] Test storage manager with T9
- [ ] Measure performance improvements
- [ ] Fix issues, optimize

### Week 3: Parallel Processing + Full Integration

**Day 5-6: Caching Infrastructure**
- [ ] Create `lux_depth_v2/cache_optimizer.py`
- [ ] Implement `ModelCache` singleton
- [ ] Implement `DepthMapCache`
- [ ] Cache eviction strategies
- [ ] Unit tests

**Day 6-7: Parallel Orchestrator**
- [ ] Enhance `lux_depth_v2/orchestrator.py`
- [ ] Implement `ParallelOrchestrator`
- [ ] Memory budget management
- [ ] Dynamic worker scheduling
- [ ] Unit tests

**Day 7-8: Configuration & CLI**
- [ ] Add `Phase2Config` to `config.py`
- [ ] Update `cli.py` with Phase 2 options
- [ ] YAML configuration support
- [ ] Documentation updates

**Day 8-10: Validation & Benchmarking**
- [ ] Create `tests/test_phase2_performance.py`
- [ ] Run Test A: I/O optimization only
- [ ] Run Test B: Parallel processing
- [ ] Run Test C: Full Phase 2 stack
- [ ] Measure performance vs baseline
- [ ] Validate 750 Picacho batch

---

## Technical Specifications

### AsyncTIFFWriter

```python
class AsyncTIFFWriter:
    """Non-blocking TIFF writer with background threads."""
    
    def __init__(self, use_compression=True, compression='lzw'):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.compression = compression if use_compression else None
        
    async def write_tiff_async(self, image, path, metadata=None):
        """Async write, returns immediately."""
        future = self.executor.submit(
            self._write_tiff_sync, image, path, metadata
        )
        return await asyncio.wrap_future(future)
        
    def write_tiff_background(self, image, path, callback=None):
        """Background write with optional callback."""
        future = self.executor.submit(
            self._write_tiff_sync, image, path, None
        )
        if callback:
            future.add_done_callback(callback)
        return future
```

### StreamingUpscaleWriter

```python
class StreamingUpscaleWriter:
    """Write upscaled tiles progressively without buffering full image."""
    
    def __init__(self, output_path, final_dimensions):
        self.output_path = output_path
        self.width, self.height = final_dimensions
        self.buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
    def write_tile(self, tile, position):
        """Write tile to buffer at position."""
        x, y = position
        h, w = tile.shape[:2]
        self.buffer[y:y+h, x:x+w] = tile
        
    def finalize(self):
        """Flush buffer to disk."""
        io_utils.write_tiff(self.buffer, self.output_path)
```

### StorageManager

```python
class StorageManager:
    """Intelligent storage tiering (internal SSD + T9)."""
    
    def __init__(self, config: StorageConfig):
        self.internal = Path(config.internal_ssd_path)
        self.t9 = Path(config.external_t9_path) if config.external_t9_path else None
        self.auto_migrate = config.auto_migrate_threshold_gb
        
    def get_optimal_write_path(self, file_type, estimated_size_gb):
        """Select best storage tier for write."""
        # Large files (>2GB) → T9 if available
        if estimated_size_gb >= self.auto_migrate and self.t9:
            return self.t9 / file_type
        return self.internal / file_type
        
    def auto_migrate_if_needed(self, file_path):
        """Move large files to T9 after write."""
        size_gb = file_path.stat().st_size / 1e9
        if size_gb >= self.auto_migrate and self.t9:
            self._migrate_to_t9(file_path)
```

### ParallelOrchestrator

```python
class ParallelOrchestrator(ProcessOrchestrator):
    """Enhanced orchestrator with parallel processing."""
    
    def __init__(self, max_workers=2, memory_budget_per_worker=25.0):
        super().__init__()
        self.max_workers = max_workers
        self.memory_budget = memory_budget_per_worker
        
    def process_batch_parallel(self, tasks, max_concurrent=2):
        """Process multiple images concurrently."""
        with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            for task in tasks:
                # Check resource availability
                if self._has_available_resources():
                    future = executor.submit(self._process_image_isolated, task)
                    futures.append(future)
                else:
                    # Wait for resource availability
                    self._wait_for_resources()
            
            # Collect results
            results = [f.result() for f in futures]
        return results
```

---

## Configuration Schema

### Phase2Config

```python
@dataclass
class Phase2Config:
    # I/O Optimization
    async_io_enabled: bool = True
    tiff_compression: str = 'lzw'  # 'lzw' | 'deflate' | None
    streaming_upscale: bool = True
    
    # Storage Management
    storage_internal_path: str = "."
    storage_external_t9: Optional[str] = None
    auto_migrate_large_files: bool = True
    migrate_threshold_gb: float = 2.0
    
    # Parallel Processing
    max_concurrent_workers: int = 2
    memory_budget_per_worker_gb: float = 25.0
    
    # Caching
    model_cache_enabled: bool = True
    depth_map_cache_enabled: bool = True
    cache_dir: str = '.cache'
    
    # Upscaling Optimization
    tile_based_upscaling: bool = True
    upscale_tile_size: int = 512
    upscale_overlap: int = 64
```

---

## CLI Integration

### New Options

```bash
# Enable all Phase 2 optimizations
lux-depth-v2 --phase2-optimizations

# Individual features
lux-depth-v2 --async-io --streaming-upscale
lux-depth-v2 --parallel-workers 2 --model-cache
lux-depth-v2 --storage-external /Volumes/T9 --auto-migrate

# Full example
lux-depth-v2 \
  --input-dir input_images/750_Picacho/Optimized_TIFFs \
  --output-dir output_750_Picacho_Phase2 \
  --phase2-optimizations \
  --parallel-workers 2 \
  --storage-external /Volumes/T9/Transformation_Portal_Outputs \
  --model-cache \
  --depth-cache
```

---

## Testing Strategy

### Unit Tests

**Module**: `tests/test_phase2_performance.py`

```python
def test_async_tiff_writer():
    """Test non-blocking TIFF writes."""
    
def test_streaming_upscale_writer():
    """Test progressive tile writing."""
    
def test_storage_manager_tier_selection():
    """Test intelligent tier selection."""
    
def test_model_cache_singleton():
    """Test model cache across batch."""
    
def test_parallel_orchestrator_resource_budget():
    """Test memory budget enforcement."""
```

### Integration Tests

**Test A: I/O Optimization Only**
```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/750_Picacho/Optimized_TIFFs/750Picacho_Pool.tif \
  --output-dir output_Phase2_IO_Test \
  --async-io \
  --streaming-upscale \
  --tiff-compression lzw
```
**Expected**: Pool image 34 min → 5-7 min (5-7× faster)

**Test B: Parallel Processing**
```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/750_Picacho/Optimized_TIFFs \
  --output-dir output_Phase2_Parallel_Test \
  --parallel-workers 2 \
  --model-cache \
  --depth-cache
```
**Expected**: Batch throughput 20 img/hr → 35-40 img/hr (1.8-2× faster)

**Test C: Full Phase 2 Stack**
```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/750_Picacho/Optimized_TIFFs \
  --output-dir output_750_Picacho_Phase2_Full \
  --phase2-optimizations \
  --parallel-workers 2 \
  --storage-external /Volumes/T9/Transformation_Portal_Outputs \
  --auto-migrate \
  --model-cache \
  --depth-cache
```
**Expected**: 10-20× overall improvement, Pool <5 min, throughput 30-60 img/hr

### Performance Benchmarks

**Baseline (Phase 1)**:
- Single image: 14.2 min average
- Pool image: 34 min
- Batch throughput: 20 img/hr
- Success rate: 100%

**Phase 2 Targets**:
- Single image: 1-2 min (7-14× faster)
- Pool image: <5 min (7-10× faster)
- Batch throughput: 30-60 img/hr (2-3× faster)
- Success rate: 100% (maintained)

---

## Success Criteria

### Must Achieve ✅

- [ ] 10× minimum performance improvement (14 min → <90 sec)
- [ ] Pool image <5 minutes (vs 34 min baseline)
- [ ] 100% success rate maintained (no regressions)
- [ ] Quality maintained (AI diff <0.004)
- [ ] All Phase 1 tests still passing
- [ ] 750 Picacho validation: 6/6 success

### Should Achieve 🎯

- [ ] 15× performance improvement (<60 sec average)
- [ ] Pool image <3 minutes
- [ ] Batch throughput 60+ img/hr
- [ ] Memory usage same or lower
- [ ] Overhead <5% from Phase 2 optimizations

### Optional (Nice to Have) 💡

- [ ] 20× performance improvement with optimal parallelism
- [ ] Automatic performance tuning
- [ ] Cloud storage tier integration

---

## Risk Management

### Technical Risks

**Risk**: Async I/O adds complexity  
**Mitigation**: Feature-gated, fallback to sync writes  
**Fallback**: `--no-async-io` flag

**Risk**: Parallel processing causes MPS contention  
**Mitigation**: Memory budget enforcement, resource monitoring  
**Fallback**: `--parallel-workers 1` (sequential)

**Risk**: T9 external storage latency  
**Mitigation**: Only migrate large files, keep hot data internal  
**Fallback**: `--no-tiered-storage` (internal only)

### Quality Risks

**Risk**: Optimization changes output quality  
**Mitigation**: AI diff validation on every image  
**Fallback**: Revert to Phase 1 if AI diff >0.004

**Risk**: Tile blending artifacts  
**Mitigation**: Increased overlap, thorough testing  
**Fallback**: Disable tile-based upscaling

### Operational Risks

**Risk**: Configuration complexity  
**Mitigation**: Sensible defaults, `--phase2-optimizations` flag  
**Fallback**: Simplified config mode

**Risk**: Cache accumulation  
**Mitigation**: Auto-cleanup, configurable retention  
**Fallback**: Manual cleanup instructions

---

## Rollout Strategy

### Alpha (Internal Testing) - Days 8-9
- [ ] Enable Phase 2 with feature flags
- [ ] Test with 750 Picacho batch
- [ ] Validate performance improvements
- [ ] Verify stability maintained
- [ ] Collect metrics

### Beta (Gradual Rollout) - Days 9-10
- [ ] Default to Phase 1 stability
- [ ] Opt-in to Phase 2 with `--phase2-optimizations`
- [ ] Monitor performance and issues
- [ ] Collect user feedback
- [ ] Fine-tune parameters

### GA (Production) - Post Day 10
- [ ] Make Phase 2 default after validation
- [ ] Fallback to Phase 1 if issues detected
- [ ] Document migration path
- [ ] Provide performance tuning guide
- [ ] Release notes and changelog

---

## Documentation Deliverables

### User Documentation
- [ ] `docs/PHASE2_PERFORMANCE_GUIDE.md` - User guide for Phase 2 features
- [ ] `docs/STORAGE_TIERING_GUIDE.md` - Storage configuration guide
- [ ] `docs/PARALLEL_PROCESSING_GUIDE.md` - Parallel processing best practices
- [ ] Update `README.md` with Phase 2 features

### Technical Documentation
- [ ] `lux_depth_v2/PHASE2_IMPLEMENTATION.md` - Implementation details
- [ ] `docs/architecture/PHASE2_ARCHITECTURE.md` - Architecture decisions
- [ ] API documentation for new modules
- [ ] Performance benchmarking report

---

## Metrics & Monitoring

### Performance Metrics

**Track**:
- Processing time per image (avg, min, max, p95)
- Per-stage timing (load, depth, material, grade, upscale, export)
- Throughput (images/hour)
- Memory usage (peak, average)
- Disk I/O (read/write bandwidth)
- MPS utilization (percentage)

**Tools**:
- Built-in profiler (`lux_depth_v2/profiler.py`)
- System monitoring (`resource_monitor.py`)
- Custom metrics dashboard

### Quality Metrics

**Track**:
- AI diff scores (validate no degradation)
- Success rate (maintain 100%)
- Image quality checks (automated)

---

## Definition of Done

### Code Complete
- [ ] All modules implemented and tested
- [ ] Unit tests passing (90%+ coverage)
- [ ] Integration tests passing
- [ ] Code reviewed and approved

### Performance Validated
- [ ] Benchmark tests meet targets (10-20× improvement)
- [ ] Pool image <5 min validated
- [ ] 750 Picacho batch 6/6 success
- [ ] No performance regressions

### Quality Assured
- [ ] AI diff <0.004 maintained
- [ ] Phase 1 tests still passing
- [ ] No quality degradation observed
- [ ] Success rate 100% maintained

### Documentation Complete
- [ ] User guides published
- [ ] Technical docs updated
- [ ] CLI help updated
- [ ] Migration guide available

### Production Ready
- [ ] Feature-gated rollout complete
- [ ] Monitoring in place
- [ ] Rollback plan tested
- [ ] Team trained on new features

---

## Next Actions

**Immediate (Day 1)**:
1. ✅ Create feature branch: `feature/phase2-performance-enhancements`
2. ✅ Create this implementation plan
3. ⏳ Start implementing `io_optimizer.py`

**This Week (Days 1-7)**:
- Implement I/O optimization modules
- Implement storage manager
- Implement upscaling optimizer
- Initial integration testing

**Next Week (Days 8-15)**:
- Implement caching infrastructure
- Implement parallel orchestrator
- Full integration and validation
- Documentation and rollout

---

## Status Tracking

**Current Status**: IN PROGRESS - Day 1  
**Completion**: 0% (0/14 tasks)  
**On Track**: YES  
**Blockers**: None  
**Next Milestone**: I/O Optimizer Module (Day 1-2)

---

**Document Owner**: Transformation Portal Architect  
**Created**: 2025-12-09  
**Last Updated**: 2025-12-09  
**Version**: 1.0
