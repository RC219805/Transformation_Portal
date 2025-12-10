# Phase 2 Slice 3: Performance Validation Plan

**Status**: Ready to Execute  
**Date**: 2025-12-10  
**Owner**: Performance & Optimization Team

---

## Objective

Empirically validate the performance gains from Slice 3 PR-2 optimizations:
- Tiled BigTIFF writing
- LZW/ZSTD compression
- Atomic writes
- Tiered storage

**Goal**: Measure actual latency reduction, file size reduction, and memory usage on production-scale workloads.

---

## Test Matrix

| Test ID | Mode | Config | Expected Result |
|---------|------|--------|-----------------|
| **T1** | Baseline | All flags OFF | Ground truth (current behavior) |
| **T2** | Tiled | `tiff_tile_size=512`<br>`tiff_compression="lzw"` | 30-50% faster writes<br>20-40% smaller files |
| **T3** | Tiled+Atomic | T2 + `use_atomic_image_writes=True` | Same speed as T2<br>Zero partial outputs |
| **T4** | Full Optimized | T3 + `enable_tiered_storage=True`<br>`scratch_dir="/fast-ssd/scratch"` | Fastest path<br>Optimized I/O |

---

## Test Images

Use actual production-scale Picacho renders:

### High Priority (Core Test Set)
1. **Pool** - `input_images/750_Picacho/Pool.tif`
   - 50-80MP, high texture variation
   - Complex water reflections
   - Good test for compression gains

2. **Aerial** - `input_images/750_Picacho/Aerial.tif`
   - 80-100MP, large scale
   - Natural textures (landscape, vegetation)
   - Tests BigTIFF threshold

3. **Great Room** - `input_images/750_Picacho/GreatRoom.tif`
   - 60-80MP, architectural detail
   - Wood, metal, fabric materials
   - Tests material-aware rendering

4. **Kitchen** - `input_images/750_Picacho/Kitchen.tif`
   - 50-70MP, high reflectivity
   - Glass, metal, stone surfaces
   - Tests compression on specular content

### Secondary (Extended Validation)
5. **Primary Bedroom**
6. **Primary Bathroom**
7. **Exterior Front**
8. **Courtyard**

---

## Metrics to Capture

### 1. Export Timing (from `timing_stages_s`)
```python
{
    "export_master": float,      # Master 16-bit TIFF write time (seconds)
    "export_upscaled": float,    # Upscaled 16-bit TIFF write time (seconds)
    "export_preview": float,     # Preview JPG write time (seconds)
    "export_marketing": float,   # Marketing PNG write time (seconds)
    "export_report": float,      # JSON report write time (seconds)
}
```

**Key Metrics**:
- `export_master + export_upscaled` = total export time
- % reduction vs baseline

### 2. File Size
```python
{
    "master_size_mb": float,
    "upscaled_size_mb": float,
    "compression_ratio": float,  # (uncompressed_size / compressed_size)
}
```

**Key Metrics**:
- File size reduction (MB saved)
- Compression ratio (for compressed modes)
- Storage savings (%)

### 3. Memory Usage
```python
{
    "peak_rss_mb": float,        # Peak resident set size
    "baseline_rss_mb": float,    # Pre-export RSS
    "delta_rss_mb": float,       # Export-specific memory
}
```

**Key Metrics**:
- Memory-neutral or reduced with tiling
- Peak usage during export

### 4. Throughput
```python
{
    "images_per_hour": float,
    "mb_per_second": float,
}
```

**Key Metrics**:
- Overall batch processing speed
- I/O bandwidth utilization

---

## Test Execution Scripts

### Script 1: Single Image Benchmark
**Purpose**: Detailed per-image profiling

```bash
# scripts/benchmark_export.py
python scripts/benchmark_export.py \
    --input input_images/750_Picacho/Pool.tif \
    --output output_benchmark/pool \
    --mode baseline \
    --runs 3
```

**Output**: JSON report with timing, size, memory metrics

### Script 2: Batch Benchmark
**Purpose**: Realistic batch throughput

```bash
# scripts/benchmark_batch_export.py
python scripts/benchmark_batch_export.py \
    --input-dir input_images/750_Picacho \
    --output-dir output_benchmark/batch \
    --mode tiled \
    --images Pool Aerial GreatRoom Kitchen
```

**Output**: Aggregate statistics, per-image breakdown

### Script 3: Comparison Report
**Purpose**: Side-by-side comparison across modes

```bash
# scripts/compare_export_modes.py
python scripts/compare_export_modes.py \
    --results output_benchmark/*/results.json \
    --output docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md
```

**Output**: Markdown tables, charts, recommendations

---

## Data Collection Template

For each test run, capture:

```json
{
  "test_id": "T2_tiled_pool_run1",
  "mode": "tiled",
  "config": {
    "tiff_tile_size": 512,
    "tiff_compression": "lzw",
    "use_atomic_image_writes": false,
    "enable_tiered_storage": false
  },
  "input": {
    "path": "input_images/750_Picacho/Pool.tif",
    "size_mp": 72.4,
    "dimensions": [9600, 7544]
  },
  "timing": {
    "export_master": 2.31,
    "export_upscaled": 8.67,
    "total_export": 10.98
  },
  "file_size": {
    "master_mb": 164.2,
    "upscaled_mb": 656.8,
    "total_mb": 821.0,
    "compression_ratio": 2.8
  },
  "memory": {
    "peak_rss_mb": 3421,
    "delta_rss_mb": 487
  },
  "throughput": {
    "images_per_hour": 327,
    "mb_per_second": 74.8
  },
  "system": {
    "cpu": "Apple M4 Max",
    "cores": 16,
    "ram_gb": 128,
    "storage": "SSD (NVMe)"
  }
}
```

---

## Success Criteria

### Performance Targets (vs Baseline)
- ✅ Export latency: **30-50% reduction** on 50MP+ images
- ✅ File size: **20-40% reduction** with compression
- ✅ Memory usage: **Neutral or reduced** (tiling should help)
- ✅ Throughput: **50-100% increase** (images/hour)

### Quality Verification
- ✅ Bit-identical output (when compression OFF)
- ✅ No visual degradation (when compression ON)
- ✅ Metadata preserved (IPTC, XMP, GPS)
- ✅ Zero partial outputs (atomic writes)

### Stability
- ✅ No crashes or errors across 100+ images
- ✅ Consistent performance (low variance)
- ✅ Memory stable (no leaks)

---

## Results Documentation

Output will be captured in:
- **`docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md`** - Summary report
- **`output_benchmark/*/results.json`** - Raw JSON data
- **`output_benchmark/comparison.csv`** - Tabular comparison

Report structure:
1. Executive Summary (key findings, recommendations)
2. Test Environment (hardware, software, config)
3. Detailed Results (per-mode, per-image)
4. Comparison Tables (baseline vs optimized)
5. Visualizations (bar charts, latency graphs)
6. Recommendations (default rollout strategy)

---

## Rollout Decision Tree

After benchmarking, decide:

### Option A: Preset-Based Rollout
```python
# Enable for high-quality presets
OPTIMIZED_PRESETS = [
    "signature_estate",
    "ultra_quality",
    "editorial_premium"
]

if preset in OPTIMIZED_PRESETS:
    export_config.tiff_tile_size = 512
    export_config.tiff_compression = "lzw"
```

**Pros**: Controlled, predictable  
**Cons**: Misses large images in other presets

### Option B: Size-Based Rollout
```python
# Enable for large images
SIZE_THRESHOLD_MP = 50

if image_size_mp > SIZE_THRESHOLD_MP:
    export_config.tiff_tile_size = 512
    export_config.tiff_compression = "lzw"
```

**Pros**: Captures all large images  
**Cons**: Harder to A/B test

### Option C: Environment Variable Opt-In
```bash
export LUX_EXPORT_OPTIMIZATIONS=1
```

**Pros**: Easy to enable/disable for testing  
**Cons**: Not automatic

### Option D: Gradual Rollout (Recommended)
```python
# Phase 1: Enable for >80MP images (safest, biggest gains)
if image_size_mp > 80:
    enable_optimizations()

# Phase 2 (after 2 weeks): Enable for >50MP
if image_size_mp > 50:
    enable_optimizations()

# Phase 3 (after 4 weeks): Default for all sizes
enable_optimizations()  # Always on
```

**Pros**: Gradual risk reduction, data-driven  
**Cons**: Requires monitoring

---

## Timeline

| Phase | Duration | Activity |
|-------|----------|----------|
| **Week 1** | 2-3 days | Execute benchmark tests (T1-T4) |
| **Week 1** | 1 day | Analyze results, write report |
| **Week 2** | 1 day | Review with team, decide rollout strategy |
| **Week 2-4** | 2 weeks | Gradual rollout Phase 1 (>80MP) |
| **Week 4-6** | 2 weeks | Gradual rollout Phase 2 (>50MP) |
| **Week 6+** | Ongoing | Monitor production, default ON for all |

---

## Risk Mitigation

### If Performance Gains < Expected
- Investigate I/O bottlenecks (disk speed, network storage)
- Try different compression algorithms (zstd vs lzw)
- Adjust tile size (256, 512, 1024)
- Check for CPU vs I/O bound operations

### If Quality Issues Detected
- Fallback to compression=None
- Compare SHA256 of outputs
- Visual inspection of compressed vs uncompressed
- Adjust compression quality if available

### If Memory Usage Increases
- Reduce tile size (512 → 256)
- Check for memory leaks in tifffile
- Profile with memory-profiler
- Consider streaming writes (future PR-3)

---

## Next Actions

1. ✅ **Create benchmark scripts** (scripts/benchmark_export.py)
2. ✅ **Run T1 (Baseline)** on Pool, Aerial, GreatRoom, Kitchen
3. ✅ **Run T2 (Tiled)** on same images
4. ✅ **Compare results** and analyze
5. ✅ **Write PHASE2_SLICE3_PERFORMANCE_RESULTS.md**
6. ✅ **Decide rollout strategy**
7. ✅ **Execute gradual rollout**

---

## Contact

Questions or issues during validation:
- Performance Team: @performance-team
- Export Subsystem Owner: @transformation-portal-specialist
- Architecture Review: @transformation-portal-architect

---

**Status**: Document complete, ready for benchmark execution 🚀
