# Performance Optimization Template

**Use this template for**: Profiling, benchmarking, and optimizing processing pipelines for throughput, memory usage, and latency

---

## Performance Optimization Workflow

**Target Component**: `{COMPONENT_NAME}`

**Optimization Goals**:
- [ ] Reduce processing time (throughput)
- [ ] Reduce memory usage (peak RAM)
- [ ] Reduce latency (first result time)
- [ ] Improve GPU utilization
- [ ] Reduce batch processing time

**Current Performance** (baseline):
- Processing time: `{X}ms per image` (size: `{WxH}`)
- Throughput: `{Y} images/hour`
- Memory usage: `{Z}MB peak`
- GPU utilization: `{N}%`

**Target Performance** (goal):
- Processing time: `< {X}ms per image`
- Throughput: `> {Y} images/hour`
- Memory usage: `< {Z}MB peak`
- GPU utilization: `> {N}%`

---

## Step 1: Profiling

### 1.1 Time Profiling

**Python `cProfile`**:
```bash
# Profile script execution
python -m cProfile -s cumulative {script}.py > profile.txt

# Or use inline profiling
python -m cProfile -o profile.stats {script}.py

# Analyze with pstats
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

**Line-by-line profiling** (line_profiler):
```python
# Install: pip install line_profiler

# Add @profile decorator to functions to profile
from line_profiler import LineProfiler

@profile
def process_image(image_path):
    """Function to profile."""
    # Implementation
    pass

# Run with: kernprof -l -v script.py
```

**Example output**:
```
Timer unit: 1e-06 s

Total time: 0.245 s
Function: process_image at line 45

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    45                                           @profile
    46                                           def process_image(image_path):
    47         1         1200   1200.0      0.5      image = Image.open(image_path)
    48         1       180000 180000.0     73.5      depth_map = estimate_depth(image)  # BOTTLENECK
    49         1        45000  45000.0     18.4      result = apply_processing(image, depth_map)
    50         1        18800  18800.0      7.7      return result
```

### 1.2 Memory Profiling

**memory_profiler**:
```python
# Install: pip install memory_profiler

from memory_profiler import profile

@profile
def process_large_image(image_path):
    """Profile memory usage."""
    image = load_image(image_path)
    depth_map = estimate_depth(image)
    result = process(image, depth_map)
    return result

# Run with: python -m memory_profiler script.py
```

**Example output**:
```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    45     85.2 MiB     85.2 MiB           1   @profile
    46                                         def process_large_image(image_path):
    47    450.3 MiB    365.1 MiB           1       image = load_image(image_path)
    48   1250.8 MiB    800.5 MiB           1       depth_map = estimate_depth(image)  # MEMORY SPIKE
    49   1255.2 MiB      4.4 MiB           1       result = process(image, depth_map)
    50   1255.2 MiB      0.0 MiB           1       return result
```

### 1.3 GPU Profiling

**PyTorch Profiler** (for ML pipelines):
```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Code to profile
    model(input_tensor)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export Chrome trace
prof.export_chrome_trace("trace.json")
# View at chrome://tracing
```

**NVIDIA nvidia-smi** (monitor GPU utilization):
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU usage during processing
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 1 > gpu_usage.csv &
python {script}.py
kill %1  # Stop nvidia-smi logging
```

### 1.4 Identifying Bottlenecks

**Checklist**:
- [ ] Which function takes the most time? → Focus optimization there
- [ ] Which function allocates the most memory? → Consider streaming/tiling
- [ ] Is GPU underutilized (< 80%)? → Increase batch size or use async operations
- [ ] Are there repeated computations? → Add caching
- [ ] Are there I/O bottlenecks? → Use async I/O or prefetching

**Common Bottlenecks**:
1. **ML Model Inference** (60-80% of time)
   - Optimize: Use CoreML/TensorRT, reduce precision (fp16), batch inference
2. **Image I/O** (10-20% of time)
   - Optimize: Prefetch images, use memory mapping, parallel loading
3. **NumPy Operations** (5-15% of time)
   - Optimize: Vectorize loops, use in-place operations, avoid copies
4. **Image Conversions** (5-10% of time)
   - Optimize: Minimize conversions, use native formats

---

## Step 2: Optimization Strategies

### 2.1 Caching

**LRU Cache for Depth Estimation**:
```python
from functools import lru_cache
import hashlib

def image_hash(image_path):
    """Compute hash of image file for caching."""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

@lru_cache(maxsize=128)
def estimate_depth_cached(image_hash_str):
    """Cached depth estimation (10-20x speedup for repeated images)."""
    image = load_from_hash(image_hash_str)
    return depth_model.estimate(image)

# Usage
img_hash = image_hash(image_path)
depth_map = estimate_depth_cached(img_hash)
```

**Expected improvement**: 10-20x speedup for repeated images

### 2.2 Batch Processing

**Before** (sequential):
```python
# Slow: Process one image at a time
results = []
for image_path in image_paths:
    image = load_image(image_path)
    depth = estimate_depth(image)  # Model called once per image
    result = process(image, depth)
    results.append(result)
```

**After** (batched):
```python
# Fast: Batch inference
import torch

batch_size = 8
results = []

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    
    # Load batch
    images = [load_image(p) for p in batch_paths]
    image_tensors = torch.stack([to_tensor(img) for img in images])
    
    # Batch inference (much faster)
    with torch.no_grad():
        depth_maps = depth_model(image_tensors)
    
    # Process batch
    for img, depth in zip(images, depth_maps):
        result = process(img, depth)
        results.append(result)
```

**Expected improvement**: 2-4x speedup for batch processing

### 2.3 GPU Acceleration

**Enable MPS (Apple Silicon)**:
```python
import torch

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Neural Engine (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU (slower)")

# Move model to device
model = model.to(device)

# Process on GPU
input_tensor = input_tensor.to(device)
output = model(input_tensor)
```

**CoreML Optimization** (Apple Silicon):
```python
import coremltools as ct

# Convert PyTorch model to CoreML
traced_model = torch.jit.trace(model, example_input)
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_shape)],
    compute_precision=ct.precision.FLOAT16,  # Faster
    compute_units=ct.ComputeUnit.ALL  # Use Neural Engine + GPU
)

# Save CoreML model
coreml_model.save("model.mlpackage")
```

**Expected improvement**: 3-5x speedup on Apple Silicon with CoreML

### 2.4 Precision Reduction

**FP16 (Half Precision)**:
```python
# Convert model to fp16
model = model.half()

# Process with fp16
with torch.cuda.amp.autocast():  # Automatic mixed precision
    output = model(input_tensor.half())
```

**Expected improvement**: 1.5-2x speedup with minimal quality loss

### 2.5 Vectorization

**Before** (slow loops):
```python
# Slow: Python loops
result = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        result[i, j] = image[i, j] * depth_map[i, j]
```

**After** (vectorized):
```python
# Fast: NumPy vectorization
result = image * depth_map  # Element-wise multiplication
```

**Expected improvement**: 10-100x speedup for array operations

### 2.6 In-Place Operations

**Before** (creates copies):
```python
# Creates multiple intermediate arrays
image = image * 1.2
image = image + 10
image = np.clip(image, 0, 255)
```

**After** (in-place):
```python
# In-place operations (no copies)
image *= 1.2
image += 10
np.clip(image, 0, 255, out=image)
```

**Expected improvement**: 30-50% memory reduction

### 2.7 Lazy Loading

**Before** (eager loading):
```python
# Loads all models at import time (slow startup)
from depth_model import DepthModel
from upscaler import Upscaler

depth_model = DepthModel()
upscaler = Upscaler()
```

**After** (lazy loading):
```python
# Load models only when needed (fast startup)
_depth_model = None

def get_depth_model():
    global _depth_model
    if _depth_model is None:
        from depth_model import DepthModel
        _depth_model = DepthModel()
    return _depth_model

# Usage
model = get_depth_model()  # Loaded on first use
```

**Expected improvement**: 60-80% faster CLI startup time

### 2.8 Parallel Processing

**Multiprocessing for CPU-bound tasks**:
```python
from multiprocessing import Pool
from functools import partial

def process_single_image(image_path, config):
    """Process one image (CPU-intensive)."""
    # Implementation
    return result

# Parallel processing
def batch_process_parallel(image_paths, config, num_workers=4):
    """Process images in parallel."""
    process_fn = partial(process_single_image, config=config)
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_fn, image_paths)
    
    return results

# Usage
results = batch_process_parallel(image_paths, config, num_workers=4)
```

**Expected improvement**: Near-linear scaling with CPU cores (3-4x on 4-core)

### 2.9 Memory-Mapped Files

**For large TIFF files**:
```python
import numpy as np
import tifffile

# Memory-mapped (doesn't load entire file)
with tifffile.TiffFile(large_tiff_path) as tif:
    image = tif.asarray(out='memmap')  # Memory-mapped array
    
    # Process in tiles to avoid loading full image
    tile_size = 2048
    for y in range(0, image.shape[0], tile_size):
        for x in range(0, image.shape[1], tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            process_tile(tile)
```

**Expected improvement**: 80-90% memory reduction for large files

### 2.10 Tiling for Large Images

**Process large images in tiles**:
```python
def process_large_image_tiled(image_path, tile_size=2048, overlap=128):
    """
    Process large image in tiles to reduce memory usage.
    
    Args:
        image_path: Path to large image
        tile_size: Size of each tile (pixels)
        overlap: Overlap between tiles to avoid seams
    
    Returns:
        Processed image
    """
    from PIL import Image
    
    with Image.open(image_path) as img:
        width, height = img.size
        
        # Initialize output
        result = Image.new(img.mode, (width, height))
        
        # Process in tiles
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Extract tile with overlap
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)
                
                tile = img.crop((x, y, x + tile_width, y + tile_height))
                
                # Process tile
                processed_tile = process_image(tile)
                
                # Blend into result (center region to avoid overlap artifacts)
                paste_region = (
                    x + overlap // 2,
                    y + overlap // 2,
                    x + tile_width - overlap // 2,
                    y + tile_height - overlap // 2
                )
                result.paste(processed_tile.crop((
                    overlap // 2, overlap // 2,
                    tile_width - overlap // 2, tile_height - overlap // 2
                )), paste_region[:2])
        
        return result
```

**Expected improvement**: Process 8K+ images with < 2GB RAM

---

## Step 3: Benchmarking

### Benchmark Script

```python
# benchmarks/benchmark_{feature}.py
"""
Benchmark script for {FEATURE_NAME}.

Usage:
    python benchmarks/benchmark_{feature}.py --num-images 100
"""

import time
import argparse
from pathlib import Path
import psutil
import os
from PIL import Image
import numpy as np

from {module} import {FeatureClass}


def create_test_images(output_dir, num_images=10, size=(2048, 2048)):
    """Create synthetic test images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    image_paths = []
    for i in range(num_images):
        # Create random image
        img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_path = output_dir / f"test_{i:04d}.jpg"
        img.save(img_path, quality=90)
        image_paths.append(img_path)
    
    return image_paths


def benchmark_processing(image_paths, config):
    """Benchmark processing performance."""
    processor = {FeatureClass}(config)
    
    # Warmup (JIT compilation, model loading)
    print("Warming up...")
    _ = processor.process(image_paths[0])
    
    # Measure memory baseline
    process = psutil.Process(os.getpid())
    baseline_memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Benchmark
    print(f"\nProcessing {len(image_paths)} images...")
    results = []
    start_time = time.perf_counter()
    
    for i, image_path in enumerate(image_paths):
        iter_start = time.perf_counter()
        result = processor.process(image_path)
        iter_time = time.perf_counter() - iter_start
        
        results.append({
            'path': image_path,
            'time_ms': iter_time * 1000,
            'success': result is not None
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)}")
    
    total_time = time.perf_counter() - start_time
    
    # Memory usage
    peak_memory_mb = process.memory_info().rss / 1024 / 1024
    memory_increase_mb = peak_memory_mb - baseline_memory_mb
    
    # Statistics
    times_ms = [r['time_ms'] for r in results]
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total images: {len(image_paths)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {np.mean(times_ms):.2f}ms")
    print(f"Median time per image: {np.median(times_ms):.2f}ms")
    print(f"Min time: {np.min(times_ms):.2f}ms")
    print(f"Max time: {np.max(times_ms):.2f}ms")
    print(f"Std dev: {np.std(times_ms):.2f}ms")
    print(f"\nThroughput: {len(image_paths) / total_time:.2f} images/sec")
    print(f"Throughput (hourly): {len(image_paths) / total_time * 3600:.0f} images/hour")
    print(f"\nMemory usage:")
    print(f"  Baseline: {baseline_memory_mb:.1f}MB")
    print(f"  Peak: {peak_memory_mb:.1f}MB")
    print(f"  Increase: {memory_increase_mb:.1f}MB")
    print("="*60)
    
    return {
        'total_time_sec': total_time,
        'avg_time_ms': np.mean(times_ms),
        'throughput_per_hour': len(image_paths) / total_time * 3600,
        'memory_increase_mb': memory_increase_mb,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark {FEATURE_NAME}')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of test images to process')
    parser.add_argument('--size', type=int, nargs=2, default=[2048, 2048],
                       help='Image size (width height)')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration preset to use')
    
    args = parser.parse_args()
    
    # Create test images
    test_dir = Path('benchmark_temp')
    print(f"Creating {args.num_images} test images ({args.size[0]}x{args.size[1]})...")
    image_paths = create_test_images(test_dir, args.num_images, tuple(args.size))
    
    # Run benchmark
    config = load_config(args.config)
    results = benchmark_processing(image_paths, config)
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    return results


if __name__ == '__main__':
    main()
```

**Run benchmark**:
```bash
# Benchmark with 100 images
python benchmarks/benchmark_{feature}.py --num-images 100

# Different image size
python benchmarks/benchmark_{feature}.py --num-images 50 --size 4096 4096
```

---

## Step 4: Validation

### Before/After Comparison

**Create comparison report**:
```python
def compare_performance(before, after):
    """Compare performance metrics before and after optimization."""
    
    time_improvement = (before['avg_time_ms'] - after['avg_time_ms']) / before['avg_time_ms'] * 100
    throughput_improvement = (after['throughput_per_hour'] - before['throughput_per_hour']) / before['throughput_per_hour'] * 100
    memory_improvement = (before['memory_increase_mb'] - after['memory_increase_mb']) / before['memory_increase_mb'] * 100
    
    print("\n" + "="*60)
    print("OPTIMIZATION IMPACT")
    print("="*60)
    print(f"Processing time: {before['avg_time_ms']:.2f}ms → {after['avg_time_ms']:.2f}ms")
    print(f"  Improvement: {time_improvement:+.1f}% ({time_improvement/100*before['avg_time_ms']:.2f}ms faster)")
    print(f"\nThroughput: {before['throughput_per_hour']:.0f} → {after['throughput_per_hour']:.0f} images/hour")
    print(f"  Improvement: {throughput_improvement:+.1f}%")
    print(f"\nMemory: {before['memory_increase_mb']:.1f}MB → {after['memory_increase_mb']:.1f}MB")
    print(f"  Improvement: {memory_improvement:+.1f}% ({memory_improvement/100*before['memory_increase_mb']:.1f}MB saved)")
    print("="*60)
```

---

## Repository-Specific Optimizations

### Depth Pipeline Optimization

**Current**: Sequential depth estimation
**Optimized**: Batched with CoreML

```python
# Before
for image_path in image_paths:
    depth = estimate_depth(image_path)  # ~50ms each

# After (2-3x faster)
batch_size = 8
for i in range(0, len(image_paths), batch_size):
    batch = image_paths[i:i+batch_size]
    depths = estimate_depth_batch(batch)  # ~20ms per image
```

### Video Processing Optimization

**Use FFmpeg hardware acceleration**:

```bash
# Before (software encoding)
ffmpeg -i input.mp4 -vf "{filters}" output.mp4

# After (hardware encoding - 3-5x faster on compatible systems)
# macOS VideoToolbox
ffmpeg -i input.mp4 -vf "{filters}" -c:v h264_videotoolbox output.mp4

# NVIDIA NVENC
ffmpeg -i input.mp4 -vf "{filters}" -c:v h264_nvenc output.mp4

# Intel QuickSync
ffmpeg -i input.mp4 -vf "{filters}" -c:v h264_qsv output.mp4
```

---

**Template Version**: 1.0  
**Last Updated**: 2025-11-06  
**Maintained By**: Transformation Portal RAG System
