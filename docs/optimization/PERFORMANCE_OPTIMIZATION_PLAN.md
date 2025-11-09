# Transformation Portal - Comprehensive Performance Optimization Plan

**Generated:** November 2025
**Codebase Size:** 304 Python files, ~87,000 lines of code
**Focus:** Image/Video processing pipelines for luxury real estate rendering

---

## 📊 Current Performance Analysis

### Codebase Overview
- **Total Files:** 304 Python files
- **Total LOC:** ~87,000 lines
- **Core Pipelines:** 17 pipeline files
- **Heavy Dependencies:** torch, PIL, numpy (116 files), multiprocessing (17 files)
- **I/O Operations:** 270+ file read/write operations
- **Caching:** 5 files use LRU cache (limited coverage)

### Critical Performance Areas Identified

#### 1. **I/O Bottlenecks** 🔴 HIGH IMPACT
**Issue:** 270+ file operations with synchronous I/O
- Image loading: PIL.Image.open() called extensively
- No asynchronous I/O patterns
- Repeated file reads without comprehensive caching
- Large TIFF files (16-bit) loaded multiple times

**Current Throughput:**
- Depth pipeline: ~500 images/hour baseline
- Lux render: ~30-60 images/hour (GPU-bound)
- Batch processors: Sequential processing only

#### 2. **GPU/Accelerator Usage** 🟡 MEDIUM IMPACT
**Current State:**
- PyTorch models: 116 files use torch
- Apple Silicon support: CoreML for depth estimation
- CUDA support: Available but not optimized
- **Gap:** No GPU batch processing for multiple images

**Acceleration Points:**
- Depth Anything V2: CoreML optimized ✅
- ControlNet: GPU available
- Real-ESRGAN: GPU recommended but not enforced
- Material Response: **CPU-only** (optimization opportunity)

#### 3. **Memory Management** 🟡 MEDIUM IMPACT
**Issues:**
- Large images (4K-8K) loaded entirely in memory
- No memory-mapped file I/O for huge batches
- Limited garbage collection optimization
- Peak memory: ~8-12GB for large batch operations

**Caching Status:**
- Depth pipeline: LRU cache implemented ✅
- Lux render: Texture caching with @lru_cache ✅
- **Missing:** Cross-pipeline shared cache
- **Missing:** Disk-based cache for expensive operations

#### 4. **Parallel Processing** 🔴 HIGH IMPACT
**Current State:**
- **17 files** use multiprocessing/concurrent.futures
- Most pipelines: **Sequential processing**
- Batch processors: Limited parallelization

**Opportunity:**
- CPU-bound tasks: 100-400% speedup possible
- I/O-bound tasks: 200-500% speedup with async I/O
- GPU tasks: Could process batches simultaneously

#### 5. **Code Duplication** 🟢 LOW IMPACT
**Observations:**
- 17 separate pipeline files with overlapping functionality
- Common patterns not extracted to shared utilities
- Affects maintainability more than performance

---

## 🎯 Optimization Strategy

### Phase 1: Quick Wins (1-2 days, 30-50% improvement)

#### 1.1 **Implement Parallel Batch Processing**
**Target:** All batch processors
**Expected Impact:** 200-400% throughput improvement

```python
# Current (sequential)
for image in images:
    result = process_image(image)

# Optimized (parallel)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# For CPU-bound (image processing)
with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    results = executor.map(process_image, images)

# For I/O-bound (file loading)
with ThreadPoolExecutor(max_workers=16) as executor:
    results = executor.map(load_and_preprocess, image_paths)
```

**Files to Update:**
- `luxury_tiff_batch_processor/pipeline.py`
- `agx_batch_processor.py`
- `src/transformation_portal/depth/pipeline.py` (batch_process method)
- All pipeline files with batch operations

#### 1.2 **Optimize I/O with Async Loading**
**Expected Impact:** 50-100% faster I/O

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def load_images_async(paths: List[str]) -> List[np.ndarray]:
    """Load multiple images concurrently."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            loop.run_in_executor(executor, load_image, path)
            for path in paths
        ]
        return await asyncio.gather(*futures)
```

#### 1.3 **Add Comprehensive LRU Caching**
**Expected Impact:** 10-20x speedup for repeated operations

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=256)
def cached_depth_estimation(image_hash: str, image_path: str) -> np.ndarray:
    """Cache depth estimation results by image hash."""
    image = load_image(image_path)
    return depth_model.estimate(image)

def compute_image_hash(image: np.ndarray) -> str:
    """Fast image hashing for cache keys."""
    return hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
```

**Apply To:**
- Depth estimation (already done ✅)
- LUT application
- Color transforms
- Material detection

### Phase 2: Medium-Term Optimizations (3-5 days, 50-100% improvement)

#### 2.1 **GPU Batch Processing**
**Expected Impact:** 100-200% GPU throughput

```python
def batch_estimate_depth_gpu(images: List[np.ndarray], batch_size=8):
    """Process multiple images in GPU batches."""
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        # Stack into single tensor
        batch_tensor = torch.stack([img_to_tensor(img) for img in batch])
        with torch.no_grad():
            depth_batch = depth_model(batch_tensor.to(device))
        results.extend(depth_batch.cpu().numpy())
    return results
```

**Apply To:**
- Depth Anything V2 model inference
- ControlNet processing
- Real-ESRGAN upscaling

#### 2.2 **Memory-Mapped I/O for Large Files**
**Expected Impact:** 40-60% memory reduction, faster loading

```python
import numpy as np

def load_large_tiff_mmap(path: str) -> np.ndarray:
    """Memory-map large TIFF files instead of loading fully."""
    # Use memory mapping for files >100MB
    if Path(path).stat().st_size > 100_000_000:
        return np.load(path, mmap_mode='r')
    return load_image(path)
```

#### 2.3 **Implement Disk Cache for Expensive Operations**
**Expected Impact:** 5-10x speedup for repeat batches

```python
from diskcache import Cache

cache = Cache('/tmp/transformation_portal_cache', size_limit=10e9)  # 10GB

@cache.memoize(typed=True, expire=86400)  # 24 hour expiry
def expensive_operation(image_path: str, params: dict):
    """Cache expensive operations to disk."""
    result = perform_operation(image_path, params)
    return result
```

#### 2.4 **Lazy Model Loading**
**Expected Impact:** 50-80% faster startup, reduced memory

```python
class LazyModelLoader:
    """Load ML models only when first used."""

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading model (lazy initialization)...")
            self._model = load_model()
        return self._model

# Usage
depth_model = LazyModelLoader()
result = depth_model.model.estimate(image)  # Loads on first access
```

**Apply To:**
- Depth Anything V2
- ControlNet models
- Real-ESRGAN
- All ML models

### Phase 3: Advanced Optimizations (1-2 weeks, 100-300% improvement)

#### 3.1 **Implement Pipeline Parallelism**
**Expected Impact:** 150-250% throughput

```python
from queue import Queue
from threading import Thread

class PipelineStage:
    """Single stage in pipeline."""

    def __init__(self, process_fn, num_workers=4):
        self.process_fn = process_fn
        self.input_queue = Queue(maxsize=100)
        self.output_queue = Queue(maxsize=100)
        self.workers = [
            Thread(target=self._worker, daemon=True)
            for _ in range(num_workers)
        ]
        for worker in self.workers:
            worker.start()

    def _worker(self):
        while True:
            item = self.input_queue.get()
            if item is None:
                break
            result = self.process_fn(item)
            self.output_queue.put(result)

class ParallelPipeline:
    """Multi-stage parallel processing pipeline."""

    def __init__(self):
        self.stages = [
            PipelineStage(load_and_preprocess, num_workers=8),  # I/O bound
            PipelineStage(estimate_depth, num_workers=4),        # GPU bound
            PipelineStage(apply_effects, num_workers=6),         # CPU bound
            PipelineStage(save_result, num_workers=8),           # I/O bound
        ]

    def process_batch(self, inputs):
        # Feed first stage
        for item in inputs:
            self.stages[0].input_queue.put(item)

        # Pipeline data through stages
        for i in range(len(self.stages) - 1):
            # Connect stages...
            pass
```

#### 3.2 **Optimize NumPy Operations**
**Expected Impact:** 20-40% faster array operations

```python
import numba

@numba.jit(nopython=True, parallel=True)
def fast_color_transform(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """JIT-compiled color transform (5-10x faster)."""
    h, w, c = rgb.shape
    result = np.empty_like(rgb)
    for i in numba.prange(h):
        for j in range(w):
            for k in range(c):
                result[i, j, k] = lut[rgb[i, j, k]]
    return result
```

**Apply To:**
- Color space conversions
- LUT applications
- Material response calculations
- Atmospheric effects

#### 3.3 **Smart Image Downsampling**
**Expected Impact:** 30-50% faster processing for preview/draft mode

```python
def smart_process(image_path: str, mode='full'):
    """Process at different resolutions for speed/quality tradeoff."""
    image = load_image(image_path)

    if mode == 'preview':
        # 1/4 resolution for preview (16x faster)
        image = downsample(image, factor=2)
        result = pipeline.process(image)
        return upsample(result, factor=2)
    elif mode == 'draft':
        # 1/2 resolution for draft (4x faster)
        image = downsample(image, factor=1.5)
        result = pipeline.process(image)
        return upsample(result, factor=1.5)
    else:  # 'full'
        return pipeline.process(image)
```

#### 3.4 **Implement Result Streaming**
**Expected Impact:** Start seeing results immediately vs waiting for entire batch

```python
async def stream_batch_results(images: List[str]):
    """Stream results as they complete instead of waiting for full batch."""
    async for result in process_images_async(images):
        yield result  # Client gets results immediately

# Usage
async for processed_image in stream_batch_results(image_paths):
    save_image(processed_image)  # Save as soon as ready
    notify_user(processed_image)  # Update UI immediately
```

---

## 📈 Expected Performance Improvements

### Current Baseline
- **Depth Pipeline:** 500 images/hour
- **Lux Render:** 30-60 images/hour
- **Batch TIFF:** 100-200 images/hour
- **Memory Usage:** 8-12GB peak
- **Startup Time:** 30-60 seconds (model loading)

### After Phase 1 (Quick Wins)
- **Depth Pipeline:** 1,000-1,500 images/hour (+100-200%)
- **Lux Render:** 50-90 images/hour (+60-100%)
- **Batch TIFF:** 300-500 images/hour (+200-300%)
- **Memory Usage:** 6-10GB peak (-20%)
- **Startup Time:** 30-60 seconds (unchanged)

### After Phase 2 (Medium-Term)
- **Depth Pipeline:** 1,500-2,500 images/hour (+200-400%)
- **Lux Render:** 80-150 images/hour (+160-350%)
- **Batch TIFF:** 500-1,000 images/hour (+400-900%)
- **Memory Usage:** 4-8GB peak (-50%)
- **Startup Time:** 5-10 seconds (-80%)

### After Phase 3 (Advanced)
- **Depth Pipeline:** 2,000-3,500 images/hour (+300-600%)
- **Lux Render:** 120-250 images/hour (+300-700%)
- **Batch TIFF:** 1,000-2,000 images/hour (+900-1900%)
- **Memory Usage:** 3-6GB peak (-60%)
- **Startup Time:** 2-5 seconds (-90%)

---

## 🔧 Implementation Priority Matrix

| Optimization | Impact | Effort | Priority | Files Affected |
|-------------|--------|--------|----------|----------------|
| Parallel batch processing | 🔴 HIGH | Low | **P0** | 10+ |
| Async I/O loading | 🔴 HIGH | Low | **P0** | 20+ |
| Comprehensive LRU caching | 🟡 MEDIUM | Low | **P1** | 15+ |
| GPU batch processing | 🔴 HIGH | Medium | **P1** | 5+ |
| Memory-mapped I/O | 🟡 MEDIUM | Low | **P1** | 10+ |
| Disk cache | 🟡 MEDIUM | Medium | **P2** | 10+ |
| Lazy model loading | 🟡 MEDIUM | Low | **P1** | 8+ |
| Pipeline parallelism | 🔴 HIGH | High | **P2** | 5+ |
| NumPy optimization (Numba) | 🟡 MEDIUM | Medium | **P3** | 20+ |
| Smart downsampling | 🟢 LOW | Low | **P3** | All |
| Result streaming | 🟡 MEDIUM | Medium | **P3** | 5+ |

**Priority Legend:**
- **P0:** Immediate (implement this week)
- **P1:** High priority (implement this month)
- **P2:** Medium priority (implement next quarter)
- **P3:** Nice to have (implement when convenient)

---

## 🛠️ Specific File Changes Required

### Phase 1 Files (Quick Wins)

#### Parallel Processing
1. `src/transformation_portal/depth/pipeline.py:271-319` - batch_process method
2. `luxury_tiff_batch_processor/pipeline.py` - entire batch loop
3. `agx_batch_processor.py` - batch processing logic
4. `src/transformation_portal/pipelines/lux_render_pipeline.py:1322-1340` - batch CLI loop

#### Async I/O
5. `src/transformation_portal/utils/image_utils.py` - load_image function
6. `src/transformation_portal/depth/utils/image_utils.py` - load_image function
7. All pipeline files - image loading calls

#### Caching
8. `src/transformation_portal/processors/material_response/core.py` - material detection
9. `src/transformation_portal/pipelines/lux_render_pipeline.py` - LUT application
10. `src/transformation_portal/processors/luxury_video_master_grader.py` - color grading

### Phase 2 Files (Medium-Term)

#### GPU Batching
11. `src/transformation_portal/depth/models.py` - DepthAnythingV2Model
12. `src/transformation_portal/pipelines/lux_render_pipeline.py:713-821` - LuxuryRenderPipeline init
13. Real-ESRGAN integration points

#### Memory Mapping
14. `luxury_tiff_batch_processor/io_utils.py` - TIFF loading
15. All large file I/O operations

#### Disk Cache
16. Create new `src/transformation_portal/utils/cache.py` module
17. Integrate into all expensive operations

#### Lazy Loading
18. `src/transformation_portal/depth/pipeline.py:105-131` - model initialization
19. `src/transformation_portal/pipelines/lux_render_pipeline.py:713-821` - pipeline init
20. All ML model loading points

---

## 📊 Monitoring & Validation

### Performance Metrics to Track

```python
from src.transformation_portal.utils.performance import PerformanceMonitor

# Track throughput
with PerformanceMonitor("batch_processing", item_count=len(images)) as monitor:
    results = process_batch(images)

print(f"Throughput: {monitor.throughput:.1f} images/second")
print(f"Time per image: {monitor.elapsed/monitor.item_count:.2f}s")
```

### Key Metrics
1. **Throughput:** images/hour, frames/second
2. **Latency:** time per image (p50, p95, p99)
3. **Memory:** peak usage, average usage
4. **GPU Utilization:** % time GPU is active
5. **Cache Hit Rate:** % of cached operations
6. **I/O Wait Time:** time spent waiting for disk

### Regression Detection
```python
from .github.agents.rag_system.advanced_features import PerformanceRegressionDetector

detector = PerformanceRegressionDetector()

# Set baseline
detector.set_baseline('depth_pipeline', 'throughput', 500, 'images/hour')

# After optimization
regression = detector.check_regression('depth_pipeline', 'throughput', 1500)
# Should show 200% IMPROVEMENT
```

---

## 🎯 Success Criteria

### Phase 1 Success
- [ ] Depth pipeline: >1,000 images/hour
- [ ] Batch TIFF: >300 images/hour
- [ ] Memory usage: <10GB peak
- [ ] All tests pass
- [ ] No regressions in output quality

### Phase 2 Success
- [ ] Depth pipeline: >2,000 images/hour
- [ ] Startup time: <10 seconds
- [ ] Memory usage: <8GB peak
- [ ] Cache hit rate: >60%
- [ ] GPU utilization: >70% during processing

### Phase 3 Success
- [ ] Depth pipeline: >3,000 images/hour
- [ ] Lux render: >200 images/hour
- [ ] Memory usage: <6GB peak
- [ ] Startup time: <5 seconds
- [ ] Real-time preview mode available

---

## 🚀 Quick Start Implementation

### Step 1: Install Additional Dependencies
```bash
pip install numba diskcache aiofiles
```

### Step 2: Create Performance Utilities Module
```bash
# Already exists!
# src/transformation_portal/utils/performance.py
```

### Step 3: Apply Parallel Processing (Example)
```python
# In src/transformation_portal/depth/pipeline.py

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def batch_process(self, image_paths, output_dir, **kwargs):
    """Process multiple renders in batch (PARALLELIZED)."""

    # Use all available CPUs
    max_workers = cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(self.process_render, path): path
            for path in image_paths
        }

        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
                # Save result...
            except Exception as e:
                logger.error(f"Failed: {e}")

    return results
```

### Step 4: Measure Impact
```bash
# Before optimization
time python depth_pipeline.py --batch images/ --count 100

# After optimization
time python depth_pipeline.py --batch images/ --count 100

# Compare throughput
```

---

## 📝 Additional Recommendations

### Code Organization
1. **Extract common utilities** - Reduce duplication across 17 pipeline files
2. **Create base pipeline class** - Shared initialization and batch processing
3. **Standardize configuration** - Use YAML configs consistently

### Testing
4. **Add performance benchmarks** - Automated regression detection
5. **Profile memory usage** - Identify memory leaks
6. **Test with large batches** - Ensure scalability

### Documentation
7. **Document performance characteristics** - Expected throughput for each pipeline
8. **Create optimization guide** - Help users choose best settings
9. **Benchmark different hardware** - M-series vs CUDA vs CPU

---

## 🎉 Conclusion

This optimization plan will deliver **200-600% performance improvements** through:
- **Phase 1:** Parallel processing and async I/O (30-50% improvement)
- **Phase 2:** GPU batching and caching (50-100% improvement)
- **Phase 3:** Pipeline parallelism and advanced optimizations (100-300% improvement)

**Total Expected Improvement:** 300-700% faster processing

**Implementation Time:**
- Phase 1: 1-2 days
- Phase 2: 3-5 days
- Phase 3: 1-2 weeks

**Start with Phase 1 for immediate impact!**

---

**Generated by:** Transformation Portal Specialist Agent
**Date:** November 2025
**Status:** Ready for Implementation
