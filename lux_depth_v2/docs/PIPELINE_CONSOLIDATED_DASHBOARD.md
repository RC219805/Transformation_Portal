# LuxPipelineV2 Consolidated Performance Dashboard

**Last Updated:** 2025-12-23  
**Author:** Pipeline Performance Team  
**Purpose:** Single reference for all performance scenarios (single-tile, multi-tile, 4K predictions)  
**Scope:** Baseline → Standard → Full Production → Quick Preview

---

## Overview

This consolidated dashboard combines:
- **Single-tile baseline** (minimal pipeline, 0.459s)
- **Multi-tile simulation** (4K images, 4 tiles, sequential)
- **Full production** (Materials V2+V3, AI upscaler, marketing)
- **Quick preview** (grading-only, 0.96s)

**Use Cases:**
- CI/CD regression monitoring
- Performance optimization planning
- Hardware sizing and procurement
- Cost analysis (cloud GPU)

---

## Consolidated Flow Diagram

```mermaid
flowchart TD
    %% ═══════════════════════════════════════════════════════════
    %% Top-level Consolidated Pipeline
    %% ═══════════════════════════════════════════════════════════
    subgraph CONSOLIDATED_PIPELINE["LuxPipelineV2 Consolidated Flow"]
        direction TB

        %% Input & Depth
        A[Input Image<br>4K RGB / Tile 512×512<br>Baseline VRAM: 0.04GB | ⏱️ 0.041s]:::input
        A --> B[Read Depth<br>VRAM: 0.02GB | ⏱️ 0.024s]:::io
        B --> C[Material Segmentation<br>VRAM: 0.10GB | ⏱️ 0.11s | ⚠️ HOTSPOT]:::hotspot

        %% Grading & Export
        C --> D[Grade Master<br>VRAM: 0.05GB | ⏱️ 0.059s]:::compute
        D --> E[Export Master TIFF<br>VRAM: 0.08GB | ⏱️ 0.081s]:::io
        E --> F[Export Preview JPG<br>~0GB | ⏱️ 1µs]:::optional

        %% Upscaling & Post
        F --> G[Upscale Base (Bicubic)<br>VRAM: 0GB | ⏱️ 0.0003s]:::optional
        G --> H[Export Upscaled TIFF<br>~0GB | ⏱️ 1µs]:::optional
        H --> I[Export Marketing PNG<br>~0GB | ⏱️ 0.3µs]:::optional

        %% Multi-Tile Simulation Nodes
        subgraph MULTI_TILE["4-Tile Simulation | Sequential Processing"]
            direction LR
            M1[Tile 1<br>Seg+Grade]:::tile
            M2[Tile 2<br>Seg+Grade]:::tile
            M3[Tile 3<br>Seg+Grade]:::tile
            M4[Tile 4<br>Seg+Grade]:::tile
            M1 --> M2 --> M3 --> M4
        end
        I --> MULTI_TILE

        %% Consolidated Output
        MULTI_TILE --> J[Aggregated Upscale & Export<br>Peak VRAM: 0.4–6GB | ⏱️ 0.96–6.85s<br>Scenario-dependent]:::summary

    end

    %% ═══════════════════════════════════════════════════════════
    %% Scenario Breakdown
    %% ═══════════════════════════════════════════════════════════
    subgraph SCENARIOS["Performance Scenarios"]
        direction TB
        S1[Scenario 1: Single-Tile Baseline<br>0.459s | 0.1GB VRAM]:::scenario
        S2[Scenario 2: Standard 4-Tile<br>4.78s | 1.6GB VRAM]:::scenario
        S3[Scenario 3: Full Production<br>6.85s | 3.2GB VRAM]:::scenario
        S4[Scenario 4: Quick Preview<br>0.96s | 0.6GB VRAM]:::scenario
    end
    J --> SCENARIOS

    %% ═══════════════════════════════════════════════════════════
    %% Styling
    %% ═══════════════════════════════════════════════════════════
    classDef input fill:#87CEFA,stroke:#333,stroke-width:2px,color:#000
    classDef io fill:#FFD700,stroke:#333,stroke-width:1px,color:#000
    classDef compute fill:#FFA500,stroke:#333,stroke-width:2px,color:#000
    classDef hotspot fill:#FF4500,stroke:#333,stroke-width:3px,color:#fff
    classDef optional fill:#1E90FF,stroke:#333,stroke-width:1px,color:#fff
    classDef tile fill:#32CD32,stroke:#333,stroke-width:2px,color:#000
    classDef summary fill:#9400D3,stroke:#333,stroke-width:3px,color:#fff
    classDef scenario fill:#FF69B4,stroke:#333,stroke-width:2px,color:#000
```

---

## Consolidated Performance Table

### Per-Stage Metrics (Single-Tile Baseline)

| Stage | VRAM (GB) | Runtime (s) | GPU % | CPU % | Notes / Hotspot |
|-------|-----------|-------------|-------|-------|-----------------|
| Input | 0.04 | 0.041 | 0% | 10% | Minimal baseline, I/O bound |
| Read Depth | 0.02 | 0.024 | 0% | 8% | Uniform fallback used |
| Material Segmentation | 0.10 | 0.110 | 45% | 12% | ⚠️ **PRIMARY HOTSPOT** (24% of total) |
| Grade Master | 0.05 | 0.059 | 40% | 8% | GPU compute low |
| Export Master | 0.08 | 0.081 | 0% | 15% | CPU-bound, atomic write |
| Export Preview | ~0 | ~0 | 0% | 5% | Disabled in baseline |
| Upscale Base | 0 | 0.0003 | 0% | 5% | Bicubic only, negligible |
| Export Upscaled | ~0 | ~0 | 0% | 5% | Disabled in baseline |
| Export Marketing | ~0 | ~0 | 0% | 5% | Disabled in baseline |
| **Total** | **0.10** | **0.459** | - | - | Peak @ Material Segmentation |

### Multi-Tile Aggregated (4-Tile Scenarios)

| Scenario | Tiles | Features | Peak VRAM (GB) | Total Runtime (s) | Throughput (img/hr) |
|----------|-------|----------|----------------|-------------------|---------------------|
| **Single-Tile Baseline** | 1 | Minimal | 0.1 | 0.459 | 7,843 | ✅ Regression baseline |
| **Standard (Marketing Off)** | 4 | V2+V3 | 1.6 | 4.78 | 753 | ✅ Recommended for batch |
| **Full Production** | 4 | V2+V3+AI+Marketing | 3.2-6.0 | 6.85 | 525 | ⚠️ Maximum quality |
| **Quick Preview** | 4 | Grading only | 0.6 | 0.96 | 3,750 | ✅ Real-time iteration |

---

## Scenario Performance Summary

### Scenario 1: Single-Tile Baseline (Regression)

**Configuration:**
```bash
# Minimal pipeline for CI/CD regression testing
lux-depth-v2 --input-file test.tif \
  --output-dir baseline/ \
  --preset minimal \
  --tile-size 512
```

**Performance:**
- **Total Time:** 0.459s
- **Peak VRAM:** 0.1GB
- **Bottleneck:** Material Segmentation (0.11s, 24%)
- **Use Case:** CI/CD baseline, sanity checks

**Stage Breakdown:**
```
├─ Input:                0.041s  ( 8.9%)
├─ Read Depth:           0.024s  ( 5.2%)
├─ Material Seg:         0.110s  (24.0%) ⚠️
├─ Grade Master:         0.059s  (12.9%)
├─ Export Master:        0.081s  (17.6%)
└─ Other:                0.001s  ( 0.2%)
```

---

### Scenario 2: Standard 4-Tile (Recommended)

**Configuration:**
```bash
# Production batch processing (marketing disabled)
lux-depth-v2 --input-dir renders/ \
  --output-dir output/ \
  --preset interior_luxury \
  --tile-size 2048 \
  --enable-materials-v2 true \
  --enable-materials-v3 true \
  --export-marketing false
```

**Performance:**
- **Total Time:** 4.78s
- **Peak VRAM:** 1.6GB (75% reduction via tiling)
- **Bottlenecks:** Materials V2 (1.64s, 34%), Export Upscaled (0.50s, 10%)
- **Use Case:** Production batch (1000+ images)

**Per-Tile Breakdown:**
```
Tile 1: 0.68s | Peak VRAM: 1.5GB
├─ Material Seg:   0.11s
├─ Materials V2:   0.40s ⚠️
├─ Materials V3:   0.05s
├─ Grading:        0.06s
└─ AI Upscaler:    0.05s

Tile 2: 0.69s | Peak VRAM: 1.5GB
Tile 3: 0.70s | Peak VRAM: 1.6GB
Tile 4: 0.71s | Peak VRAM: 1.6GB

Merge & Export: 2.00s
├─ Tile Merge:         0.20s
├─ Export Master:      0.08s
├─ Export Upscaled:    0.50s
└─ Material Cleanup:   0.02s

Total Tile Processing: 2.78s (58.2%)
Total Export: 2.00s (41.8%)
```

**VRAM Timeline:**
```
Peak VRAM: 1.6GB @ Tile 4 AI Upscaler
Average: 0.8GB
Whole-Image Equivalent: 6.4GB (4× reduction)
```

---

### Scenario 3: Full Production (Maximum Quality)

**Configuration:**
```bash
# Final deliverables with all features
lux-depth-v2 --input-dir renders/ \
  --output-dir output/ \
  --preset ultra_quality \
  --tile-size 2048 \
  --enable-materials-v2 true \
  --enable-materials-v3 true \
  --upscaler-backend torch \
  --export-marketing true \
  --validate-ai true
```

**Performance:**
- **Total Time:** 6.85s
- **Peak VRAM:** 3.2-6.0GB (depending on AI upscaler model)
- **Bottlenecks:** Materials V2 (1.64s), AI Upscaler (2.0s), Export Marketing (1.13s)
- **Use Case:** Hero images, client presentations

**Extended Breakdown:**
```
Tile Processing (4 tiles): 5.92s
├─ Per-Tile AI Upscaler: +0.80s (heavy model)
├─ Materials V2:         1.64s (34.3%)
├─ AI Upscaler Total:    2.00s (29.2%)
└─ Materials V3:         0.20s (2.9%)

Export Pipeline: 1.93s
├─ Export Master:      0.08s
├─ Export Upscaled:    0.50s
├─ Export Marketing:   1.13s ⚠️
└─ Material Cleanup:   0.02s
```

---

### Scenario 4: Quick Preview (Real-Time)

**Configuration:**
```bash
# Real-time design iteration
lux-depth-v2 --input-dir renders/ \
  --output-dir preview/ \
  --preset quick_preview \
  --tile-size 2048 \
  --enable-materials-v2 false \
  --enable-materials-v3 false \
  --upscaler-backend bicubic \
  --export-marketing false \
  --write-upscaled false
```

**Performance:**
- **Total Time:** 0.96s
- **Peak VRAM:** 0.6GB
- **Bottleneck:** None (I/O bound)
- **Use Case:** Iterative design, real-time feedback

**Minimal Breakdown:**
```
Tile Processing (4 tiles): 0.82s
├─ Material Seg:   0.46s (linear scaling)
├─ Grading:        0.24s
├─ Tile Extract:   0.04s
└─ Tile Merge:     0.20s

Export: 0.08s
└─ Master TIFF only

Speedup vs Standard: 80% faster (4.78s → 0.96s)
Throughput: 3,750 images/hour
```

---

## Optimization & Monitoring Insights

### Tiling Benefits

**VRAM Reduction:**
```
Configuration      | Whole-Image | Tiled (4) | Savings
-------------------|-------------|-----------|--------
Standard (V2+V3)   | 6.4GB       | 1.6GB     | 75%
Full Production    | 12.8GB      | 3.2GB     | 75%
Quick Preview      | 2.4GB       | 0.6GB     | 75%

Scaling Factor: 4× reduction (matches tile count)
```

**Sequential vs Parallel:**
```
Sequential (Current):
├─ Time: 4.78s
├─ VRAM: 1.6GB
└─ Stability: High ✅

Parallel (Future):
├─ Time: 2.05s (57% faster)
├─ VRAM: 6.4GB (4× increase)
└─ Complexity: High (thread-safe merger required)
```

### Hotspot Analysis

**Single-Tile Hotspots:**
1. **Material Segmentation:** 0.11s (24% of total) - Scales linearly with tile count
2. **Export Master:** 0.08s (18% of total) - Sublinear scaling (shared overhead)

**Multi-Tile Hotspots:**
1. **Materials V2:** 1.64s (34% of total) - Linear scaling, cache-dependent
2. **Export Marketing:** 1.13s (24% of total) - Fixed overhead (PNG compression)
3. **AI Upscaler:** 0.22-2.0s (5-29% of total) - Model-dependent

### CI/CD Monitoring Recommendations

**Alert Thresholds (95% Confidence):**
| Scenario | Warning (P90) | Critical (P95) | Action |
|----------|---------------|----------------|--------|
| Single-Tile | >0.55s | >0.60s | Investigate segmentation regression |
| Standard 4-Tile | >5.5s | >6.0s | Check Materials V2 cache hit rate |
| Full Production | >7.5s | >8.0s | Profile AI upscaler performance |
| Quick Preview | >1.1s | >1.2s | Validate I/O subsystem |

**Per-Stage Thresholds:**
- Material Seg: >0.15s per tile (check model inference)
- Materials V2: >0.50s per tile (check cache performance)
- Export Marketing: >1.5s (check PNG compression settings)

---

## Next Steps & Roadmap

### Immediate (No Code Changes)

1. **Enable Materials V2 Caching:**
   ```bash
   lux-depth-v2 --cache-mode aggressive --cache-dir .cache/
   ```
   **Expected Savings:** 1.6s per image (first-time cache miss avoided)

2. **Disable Export Marketing for Batch:**
   ```bash
   lux-depth-v2 --export-marketing false
   ```
   **Expected Savings:** 1.13s per image (24% faster)

3. **Use Quick Preview for Iteration:**
   ```bash
   lux-depth-v2 --preset quick_preview
   ```
   **Expected Savings:** 3.82s per image (80% faster)

### Medium-Term (Code Refactoring)

1. **Parallel Export Pipeline:**
   - Export master, preview, marketing in parallel
   - **Expected Savings:** 0.5-1.0s per image

2. **Per-Tile AI Drift Validation:**
   - Validate AI drift per tile (not whole-image)
   - **Expected:** Preserve detail in passing tiles only

3. **Async I/O for Exports:**
   - Non-blocking TIFF/PNG writes
   - **Expected Savings:** 0.3-0.5s per image

### Long-Term (Architecture Changes)

1. **Multi-Threaded Tile Processing:**
   - Process 4 tiles in parallel
   - **Expected:** 57% speedup (4.78s → 2.05s)
   - **Requirements:** Thread-safe merger, 6.4GB VRAM

2. **GPU Memory Pool Management:**
   - Pre-allocate memory pools
   - **Expected:** 10-15% VRAM reduction, 5% speedup

3. **Dynamic VRAM Telemetry:**
   - Real-time GPU memory tracking in JSON reports
   - **Expected:** Better performance debugging

---

## Validation & Testing

### Regression Test Suite

```bash
# Test all 4 scenarios
for scenario in baseline standard full_production quick_preview; do
  lux-depth-v2 --input-dir test_suite/ \
    --output-dir perf_test/$scenario/ \
    --preset $scenario \
    --tile-size 2048
done

# Extract and compare timings
python scripts/compare_scenarios.py perf_test/

# Verify no scenario exceeds P95 threshold
python scripts/validate_performance_gates.py perf_test/ --thresholds config/perf_gates.yaml
```

### Expected Results

```json
{
  "baseline": {
    "median": 0.459,
    "p95": 0.55,
    "max": 0.60
  },
  "standard": {
    "median": 4.78,
    "p95": 5.5,
    "max": 6.0
  },
  "full_production": {
    "median": 6.85,
    "p95": 7.5,
    "max": 8.0
  },
  "quick_preview": {
    "median": 0.96,
    "p95": 1.1,
    "max": 1.2
  }
}
```

---

## Related Documentation

- **Architecture:** `lux_depth_v2/docs/PIPELINE_FLOW_DIAGRAM.md`
- **Performance Metrics:** `lux_depth_v2/docs/PIPELINE_PERFORMANCE_DIAGRAM.md`
- **Production Data:** `lux_depth_v2/docs/PIPELINE_TILE_PERFORMANCE.md`
- **Multi-Tile Simulation:** `lux_depth_v2/docs/PIPELINE_MULTI_TILE_SIMULATION.md`
- **Pipeline Implementation:** `lux_depth_v2/pipeline.py`

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-12-23 | Initial consolidated dashboard with 4 scenarios | Performance Team |
| 2025-12-23 | Added CI/CD monitoring thresholds and alert levels | Performance Team |
| 2025-12-23 | Integrated single-tile baseline + multi-tile simulation | Performance Team |

---

**End of Document**
