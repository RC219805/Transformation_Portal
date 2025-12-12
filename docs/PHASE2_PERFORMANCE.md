# Phase 2 Performance Benchmark Results

**Generated**: 2025-12-12 22:51:32 UTC

## Executive Summary

This benchmark measures Phase 2 feature overhead (CLIP classification, preset selection)
and pipeline initialization costs across Standard/Max/APEX quality tiers.

**Note**: This is a *fast* benchmark focusing on initialization overhead.
Full end-to-end processing benchmarks require significant compute time and are
better suited for dedicated performance testing environments.

## System Configuration

- **Platform**: Darwin 25.0.0
- **Machine**: arm64
- **Processor**: arm
- **CPU Cores**: 16 physical, 16 logical
- **Memory**: 48.0 GB
- **Python**: 3.11.14
- **CUDA Available**: False
- **MPS Available**: True

## Performance Summary

### Initialization Overhead by Preset

| Image | Preset | Total Init (s) | CLIP (s) | Preset Selection (s) | Pipeline Init (s) | Model Loading (s) | Peak Memory (MB) | Backend |
|-------|--------|----------------|----------|----------------------|-------------------|-------------------|------------------|---------|
| 750Picacho_Kitchen | interior_luxury | 2.92 | 1.964 | 0.000 | 0.96 | 0.77 | 809520 | auto |
| 750Picacho_Kitchen | interior_luxury_max_quality | 1.16 | 0.146 | 0.000 | 1.01 | 0.81 | 837456 | auto |
| 750Picacho_Kitchen | interior_luxury_apex_quality | 1.21 | 0.142 | 0.000 | 1.07 | 0.86 | 837664 | auto |
| 750Picacho_Pool | interior_luxury | 1.17 | 0.152 | 0.000 | 1.01 | 0.81 | 838624 | auto |
| 750Picacho_Pool | interior_luxury_max_quality | 1.15 | 0.150 | 0.000 | 1.00 | 0.80 | 838704 | auto |
| 750Picacho_Pool | interior_luxury_apex_quality | 1.25 | 0.152 | 0.000 | 1.10 | 0.88 | 838704 | auto |
| 750Picacho_PrimaryBedroom | interior_luxury | 1.23 | 0.170 | 0.000 | 1.06 | 0.85 | 914784 | auto |
| 750Picacho_PrimaryBedroom | interior_luxury_max_quality | 1.18 | 0.173 | 0.000 | 1.00 | 0.80 | 914848 | auto |
| 750Picacho_PrimaryBedroom | interior_luxury_apex_quality | 1.12 | 0.177 | 0.000 | 0.95 | 0.76 | 919808 | auto |

### Phase 2 Overhead Analysis

Phase 2 introduces CLIP-based scene classification for intelligent preset selection.

| Preset | Avg CLIP Time (s) | Avg Preset Selection (s) | Total Phase 2 Overhead (s) |
|--------|-------------------|--------------------------|----------------------------|
| interior_luxury | 0.762 | 0.000 | 0.762 |
| interior_luxury_max_quality | 0.156 | 0.000 | 0.156 |
| interior_luxury_apex_quality | 0.157 | 0.000 | 0.157 |

### Quality Tier Comparison

Average initialization time and memory usage by quality tier.

| Tier | Avg Init Time (s) | Avg Model Loading (s) | Avg Peak Memory (MB) | Segmentation Backend |
|------|-------------------|-----------------------|----------------------|----------------------|
| STANDARD | 1.01 | 0.81 | 854309 | auto |
| MAX | 1.00 | 0.80 | 863669 | auto |
| APEX | 1.04 | 0.83 | 865392 | auto |

## Key Findings

### Phase 2 Overhead

- **CLIP Classification**: ~0.359s per image (one-time cost for auto-preset)
- **Preset Selection**: ~0.000s per image

### Initialization Costs

- **APEX vs STANDARD Initialization**: +2.6% (1.04s vs 1.01s)
- **Model Loading Dominates**: ~80% of initialization time is model loading

## Recommendations

### Performance Optimization

1. **CLIP Model Caching**: Reuse CLIP model across multiple images in batch mode
2. **Preset Pinning**: Skip auto-preset for known scene types (use explicit `--preset`)
3. **Batch Processing**: Amortize model loading across many images
4. **Tier Selection Strategy**:
   - STANDARD: Quick previews and iteration
   - MAX: Production quality for most use cases
   - APEX: Final deliverables requiring maximum quality

### Phase 2 Feature Usage

- **Auto-Preset (`--auto-preset`)**: ~0.1-0.2s overhead, provides intelligent quality tier selection
- **Benefits**: Eliminates manual preset selection, optimizes quality/performance tradeoff
- **Best For**: Batch processing diverse scene types (interiors, exteriors, mixed lighting)

## Test Configuration

- **Test Images**: 3
- **Presets Tested**: interior_luxury, interior_luxury_max_quality, interior_luxury_apex_quality
- **CLIP Enabled**: True
- **Benchmark Type**: Initialization overhead only (fast, CI-friendly)

## Future Work

- **End-to-End Processing**: Full pipeline benchmarks (depth, seg, upscale, post)
- **Lighting Detection**: Benchmark lighting adaptation when implemented
- **EfficientSAM**: Compare SegFormer vs EfficientSAM segmentation backends
- **GPU Comparison**: CUDA vs MPS vs CPU performance matrix
