# Phase 2 + Materials v2 Feature Summary

**Quick Reference Guide for Production Use**

---

## 🎯 What's New

### Phase 2: Performance Enhancements (2-3× throughput)

1. **Parallel Processing** - Process 2-4 images concurrently
2. **Intelligent Caching** - Model and depth map caching (18-30s/batch savings)
3. **Async I/O** - Background TIFF writing (5-7× faster I/O)
4. **Storage Tiering** - Automatic migration to external storage (T9 SSD)
5. **Tile-Based Upscaling** - Memory-efficient progressive output

### Materials v2: Quality Enhancements (5.7% overhead cached)

1. **Material-Specific Enhancement** - Wood, metal, glass, stone, water optimization
2. **Confidence Gating** - Realism control (threshold 0.6)
3. **Mask Caching** - 25.2% speedup on repeat processing
4. **Multi-Backend Support** - Heuristic, ONNX, SegFormer segmentation
5. **Soft Masking** - Natural blending, no hard edges

---

## 🚀 Quick Start

### Enable Everything (Recommended)

```bash
python3 -m lux_depth_v2.cli \
  --input-dir images/ \
  --output-dir output/ \
  --phase2-optimizations \
  --parallel-workers 2 \
  --materials-v2 \
  --cache-masks
```

**Result**: 2× throughput + quality improvements + 25% cache speedup

### Phase 2 Only (Performance)

```bash
python3 -m lux_depth_v2.cli \
  --input-dir images/ \
  --output-dir output/ \
  --phase2-optimizations \
  --parallel-workers 2
```

**Result**: 2× throughput with <2% overhead

### Materials v2 Only (Quality)

```bash
python3 -m lux_depth_v2.cli \
  --input-dir images/ \
  --output-dir output/ \
  --materials-v2 \
  --confidence-threshold 0.6 \
  --cache-masks
```

**Result**: Enhanced material realism with 5.7% overhead (cached)

---

## 📊 Performance at a Glance

| Configuration | Time/Image | Throughput | Use Case |
|---------------|-----------|------------|----------|
| **Phase 1 Baseline** | 19.3s | 187 img/hr | Existing production |
| **Phase 2 (2 workers)** | 19.9s | 361 img/hr | High-volume batches |
| **Materials v2 (cached)** | 20.4s | 177 img/hr | Quality-first projects |
| **Phase 2 + Materials v2** | 20.4s | 353 img/hr | **Recommended production** |

---

## 🔧 Key Features

### Phase 2 Features

| Feature | CLI Flag | Benefit | When to Use |
|---------|----------|---------|-------------|
| **Parallel Processing** | `--parallel-workers 2` | 2× throughput | Batches of 10+ images |
| **Model Cache** | `--model-cache` | 18-30s/batch savings | Any batch processing |
| **Depth Cache** | `--depth-cache` | Skip depth regen | Iterative refinement |
| **Async I/O** | `--async-io` | 5-7× I/O speedup | Large TIFF outputs |
| **Streaming Upscale** | `--streaming-upscale` | No 6GB buffering | Memory-constrained systems |
| **External Storage** | `--storage-external /path` | Auto-tiering | Large datasets (>50GB) |

### Materials v2 Features

| Feature | CLI Flag | Benefit | When to Use |
|---------|----------|---------|-------------|
| **Material Enhancement** | `--materials-v2` | Realistic textures | Architectural renders |
| **Confidence Gating** | `--confidence-threshold 0.6` | Realism control | Conservative enhancement |
| **Mask Caching** | `--cache-masks` | 25% speedup | Repeat processing |
| **Segmentation Backend** | `--segmentation-backend heuristic` | Speed/quality tradeoff | Performance tuning |
| **Soft Masking** | (default) | Natural blending | All production use |

---

## 🎨 Material Types Enhanced

### Supported Materials

1. **Wood** - Grain preservation, warmth, depth
2. **Metal** - Reflections, shine, contrast
3. **Glass** - Transparency, clarity, reflections
4. **Stone** - Texture detail, specular highlights
5. **Water** - Color accuracy, reflections, caustics
6. **Foliage** - Natural greens, texture
7. **Sky** - Blue preservation, clouds

### Edge Cases Validated

- ✅ Pool water (complex reflections, caustics)
- ✅ Glass shower doors (transparency, multi-layer)
- ✅ Granite countertops (high-frequency detail)
- ✅ Wood flooring (grain, color accuracy)
- ✅ Stainless steel appliances (specular highlights)

---

## ⚙️ Configuration Guide

### Conservative (Safe, Minimal Overhead)

```bash
python3 -m lux_depth_v2.cli \
  --input-dir images/ \
  --output-dir output/ \
  --parallel-workers 2 \
  --materials-v2 \
  --confidence-threshold 0.7 \
  --cache-masks
```

**Characteristics:**
- 2 workers (safe parallelism)
- High confidence threshold (conservative enhancement)
- Cached masks (25% speedup)
- ~5% total overhead

### Balanced (Recommended Production)

```bash
python3 -m lux_depth_v2.cli \
  --input-dir images/ \
  --output-dir output/ \
  --phase2-optimizations \
  --parallel-workers 2 \
  --materials-v2 \
  --confidence-threshold 0.6 \
  --cache-masks \
  --model-cache \
  --depth-cache
```

**Characteristics:**
- Full Phase 2 stack
- Standard confidence threshold
- All caching enabled
- 2× throughput, <10% overhead

### Aggressive (Maximum Performance)

```bash
python3 -m lux_depth_v2.cli \
  --input-dir images/ \
  --output-dir output/ \
  --phase2-optimizations \
  --parallel-workers 3 \
  --materials-v2 \
  --confidence-threshold 0.5 \
  --cache-masks \
  --async-io \
  --streaming-upscale \
  --storage-external /Volumes/T9 \
  --auto-migrate \
  --model-cache \
  --depth-cache \
  --tile-based-upscale
```

**Characteristics:**
- 3 workers (high parallelism)
- Lower confidence (more enhancement)
- External storage auto-tiering
- Full optimization stack
- 3× throughput (capacity permitting)

---

## 🔍 Tuning Guide

### Parallel Workers

| Workers | Throughput | Memory | VRAM | Use Case |
|---------|-----------|--------|------|----------|
| **1** | 1.0× | 25GB | 24GB | Baseline |
| **2** | 1.9× | 50GB | 24GB | **Recommended** |
| **3** | 2.7× | 75GB | 24GB | High-end systems |
| **4** | 3.5× | 100GB | 24GB | Maximum performance |

**Recommendation**: Start with 2 workers, increase if system can handle it.

### Confidence Threshold

| Threshold | Enhancement | Realism | Use Case |
|-----------|-------------|---------|----------|
| **0.5** | Maximum | Good | Aggressive enhancement |
| **0.6** | High | Very Good | **Recommended production** |
| **0.7** | Moderate | Excellent | Conservative projects |
| **0.8** | Minimal | Perfect | Subtle enhancement only |

**Recommendation**: 0.6 for most production work, 0.7 for high-stakes projects.

### Segmentation Backend

| Backend | Speed | Quality | Memory | Use Case |
|---------|-------|---------|--------|----------|
| **heuristic** | Fastest | Good | Low | High-volume batches |
| **onnx** | Fast | Very Good | Medium | **Recommended production** |
| **segformer** | Slow | Excellent | High | Quality-critical projects |

**Recommendation**: ONNX for balanced speed/quality.

---

## 🛠️ Troubleshooting

### Out of Memory (Parallel Processing)

**Symptom**: Workers crashing, "Out of memory" errors

**Solution**:
```bash
# Reduce workers
--parallel-workers 1

# OR reduce worker memory budget
--worker-memory-budget 20
```

### Materials v2 Too Aggressive

**Symptom**: Over-enhanced materials, unnatural look

**Solution**:
```bash
# Increase confidence threshold
--confidence-threshold 0.7

# OR use heuristic backend
--segmentation-backend heuristic
```

### Slow Performance (First Run)

**Symptom**: Much slower than expected

**Solution**:
```bash
# Enable all caching
--model-cache --depth-cache --cache-masks

# Second run will be 25-50% faster
```

### External Storage Not Working

**Symptom**: Files not migrating to T9 SSD

**Solution**:
```bash
# Check path
ls -la /Volumes/T9

# Enable auto-migration explicitly
--storage-external /Volumes/T9 --auto-migrate --storage-symlinks
```

---

## 📈 Expected Results

### Performance Improvements

- **Baseline → Phase 2 (2 workers)**: 1.9× throughput
- **Baseline → Phase 2 (3 workers)**: 2.7× throughput
- **Materials v2 first run**: +41% overhead (segmentation)
- **Materials v2 cached**: +5.7% overhead (optimal)
- **Cache speedup**: 25.2% faster (cached vs first-run)

### Quality Improvements

- **Wood**: Enhanced grain detail, preserved warmth
- **Metal**: Improved reflections and shine
- **Glass**: Better transparency and clarity
- **Stone**: Enhanced texture and specular highlights
- **Water**: Realistic color, reflections, caustics

### System Requirements

- **RAM**: 25GB per worker (50GB for 2 workers)
- **VRAM**: 24GB (shared across all workers)
- **Disk**: 20GB free (internal), optional T9 for large batches
- **CPU**: Apple M-series recommended (MPS acceleration)

---

## 📚 Documentation

### User Guides

- **Phase 2 Performance**: `lux_depth_v2/PHASE2_PERFORMANCE_DEPLOYMENT.md`
- **Materials v2 User Guide**: `docs/MATERIALS_V2_USER_GUIDE.md`

### Technical Specs

- **Phase 2 Architecture**: `docs/guides/PHASE2_IMPLEMENTATION_REPORT.md`
- **Materials v2 Technical**: `docs/MATERIALS_V2_TECHNICAL_SPEC.md`

### Validation Reports

- **Phase 2 Performance**: `docs/guides/PHASE2_DEPLOYMENT_SUMMARY.md`
- **Materials v2 Quality**: `docs/guides/MATERIALS_V2_VISUAL_COMPARISON.md`

---

## 🎯 Production Recommendations

### Standard Production Pipeline

```bash
#!/bin/bash
# production_pipeline.sh

python3 -m lux_depth_v2.cli \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --preset interior_luxury \
  --phase2-optimizations \
  --parallel-workers 2 \
  --materials-v2 \
  --confidence-threshold 0.6 \
  --cache-masks \
  --model-cache \
  --depth-cache \
  --verbose
```

**Expected Performance**:
- **Throughput**: 350+ images/hour
- **Quality**: Enhanced materials + Phase 1 baseline
- **Overhead**: <10% total
- **Cache speedup**: 25% on repeat processing

### High-Quality Pipeline (Conservative)

```bash
#!/bin/bash
# high_quality_pipeline.sh

python3 -m lux_depth_v2.cli \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --preset interior_luxury \
  --materials-v2 \
  --confidence-threshold 0.7 \
  --cache-masks \
  --segmentation-backend onnx \
  --verbose
```

**Expected Performance**:
- **Throughput**: 175+ images/hour (sequential)
- **Quality**: Maximum realism, conservative enhancement
- **Overhead**: 5.7% (cached)

### High-Throughput Pipeline (Speed)

```bash
#!/bin/bash
# high_throughput_pipeline.sh

python3 -m lux_depth_v2.cli \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --preset interior_luxury \
  --phase2-optimizations \
  --parallel-workers 3 \
  --async-io \
  --streaming-upscale \
  --model-cache \
  --depth-cache \
  --verbose
```

**Expected Performance**:
- **Throughput**: 500+ images/hour (3 workers)
- **Quality**: Phase 1 baseline (no Materials v2)
- **Overhead**: <2%

---

## ✅ Migration Checklist

### Pre-Migration

- [ ] Backup existing pipeline configuration
- [ ] Test on 5-10 sample images
- [ ] Verify external storage paths (if using T9)
- [ ] Check available RAM (25GB per worker)
- [ ] Review confidence threshold for your use case

### Migration

- [ ] Update CLI commands with new flags
- [ ] Enable Phase 2 optimizations (`--phase2-optimizations`)
- [ ] Add parallel workers (`--parallel-workers 2`)
- [ ] Enable Materials v2 (`--materials-v2 --cache-masks`)
- [ ] Test on full batch (compare outputs)

### Post-Migration

- [ ] Verify output quality (compare baseline vs enhanced)
- [ ] Monitor performance (throughput, memory usage)
- [ ] Collect cache statistics (hit rates)
- [ ] Fine-tune confidence threshold (if needed)
- [ ] Document any edge cases or issues

---

## 🎉 Success Metrics

After migration, you should see:

✅ **2-3× throughput** with parallel processing
✅ **5-10% overhead** with Materials v2 (cached)
✅ **Enhanced material realism** (wood, metal, glass, stone, water)
✅ **25% cache speedup** on repeat processing
✅ **Zero quality degradation** (Phase 1 baseline maintained)
✅ **Backward compatible** (can disable any feature)

---

**Questions or Issues?** See comprehensive documentation in:
- `lux_depth_v2/PHASE2_PERFORMANCE_DEPLOYMENT.md`
- `docs/MATERIALS_V2_USER_GUIDE.md`
